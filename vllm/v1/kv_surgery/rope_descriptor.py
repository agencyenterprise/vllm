# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where position lives in a layer's KV cache, derived from the loaded model.

Position is baked into stored K as a rotation. To shift a slot we need, per
layer: the cache view, the element offset and width of the rotary slice inside
a cache row, the pairing style, the inverse frequencies, and the fp8 scale.
All of it is read off vLLM's instantiated modules at load time; there is no
hand-written table.
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.gemma4_rope import (
    Gemma4RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.ntk_scaling_rope import (
    NTKScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)
from vllm.utils.torch_utils import is_quantized_kv_cache

# Exact types whose cos/sin cache is ``cos(p * inv_freq)`` (times a scalar
# mscale for YaRN) with a fixed inv_freq. Subclasses not listed fail loudly:
# an unknown ``_compute_inv_freq`` signature or a position-dependent cache
# (dynamic NTK) would silently produce wrong rotations.
_INV_FREQ_FROM_BASE = frozenset(
    {
        RotaryEmbedding,
        Llama3RotaryEmbedding,
        Gemma4RotaryEmbedding,
        NTKScalingRotaryEmbedding,
    }
)
_INV_FREQ_FROM_SCALING_FACTOR = frozenset(
    {DeepseekScalingRotaryEmbedding, YaRNScalingRotaryEmbedding}
)

_SUPPORTED_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")


@dataclass
class RopeDescriptor:
    """Everything the rotate kernel needs for one layer's KV cache."""

    layer_name: str
    kv_cache: torch.Tensor
    """``[num_blocks, num_heads, block_size, C]`` view; ``stride(-1) == 1``."""
    rot_offset: int
    """Element offset of the rotary slice within a cache row's ``C`` axis."""
    rotary_dim: int
    is_neox_style: bool
    inv_freq: torch.Tensor
    """float32 ``[rotary_dim // 2]`` on ``kv_cache.device``."""
    k_scale: torch.Tensor | None = None
    """Per-tensor fp8 scale (stored = value / scale), or None if unquantized."""

    def __post_init__(self) -> None:
        if self.kv_cache.dim() != 4:
            raise ValueError(f"{self.layer_name}: kv_cache must be [B, H, N, C]")
        if self.kv_cache.stride(-1) != 1:
            raise ValueError(f"{self.layer_name}: cache rows must be contiguous")
        if self.rotary_dim % 2 or self.rotary_dim <= 0:
            raise ValueError(f"{self.layer_name}: rotary_dim must be even")
        if self.rot_offset + self.rotary_dim > self.kv_cache.shape[-1]:
            raise ValueError(f"{self.layer_name}: rotary slice exceeds cache row")
        if self.inv_freq.shape != (self.rotary_dim // 2,):
            raise ValueError(f"{self.layer_name}: inv_freq has the wrong shape")
        if self.inv_freq.dtype != torch.float32:
            raise ValueError(f"{self.layer_name}: inv_freq must be float32")
        if self.k_scale is not None:
            if self.k_scale.numel() != 1 or self.k_scale.dtype != torch.float32:
                raise ValueError(f"{self.layer_name}: k_scale must be a f32 scalar")
            if self.kv_cache.element_size() != 1:
                raise ValueError(f"{self.layer_name}: fp8 cache must be 1 byte/elem")

    @property
    def num_heads(self) -> int:
        return self.kv_cache.shape[1]

    @property
    def block_size(self) -> int:
        return self.kv_cache.shape[2]


def inv_freq_of(rope: RotaryEmbedding) -> torch.Tensor:
    """The per-pair angular frequencies a ``RotaryEmbedding`` rotates with.

    Built from the module's own ``_compute_inv_freq``, never from its
    ``cos_sin_cache``: YaRN folds ``mscale`` into the cache and a delta
    rotation must be purely orthonormal. The cache is only used to
    cross-check the result (``atan2`` cancels the scalar mscale).

    Raises:
        NotImplementedError: for RoPE variants not known to be shift-invariant.
    """
    rope_type = type(rope)
    if rope_type in _INV_FREQ_FROM_BASE:
        inv_freq = rope._compute_inv_freq(rope.base)
    elif rope_type in _INV_FREQ_FROM_SCALING_FACTOR:
        inv_freq = rope._compute_inv_freq(rope.scaling_factor)
    else:
        raise NotImplementedError(
            f"KV surgery does not support {rope_type.__name__}: its cache is not "
            "known to be a fixed-frequency rotation"
        )
    inv_freq = inv_freq.to(torch.float32)

    cos_sin = rope._compute_cos_sin_cache()
    half = rope.rotary_dim // 2
    cos, sin = cos_sin[1, :half].double(), cos_sin[1, half:].double()
    recovered = torch.atan2(sin, cos).to(torch.float32)
    if not torch.allclose(recovered, inv_freq.to(recovered.device), atol=1e-5):
        raise RuntimeError(
            f"{rope_type.__name__}: inv_freq disagrees with the module's cos/sin "
            "cache; the RoPE descriptor derivation does not understand this module"
        )
    return inv_freq


def _find_rotary_emb(model: nn.Module, layer_name: str) -> RotaryEmbedding:
    """The ``RotaryEmbedding`` that rotated this attention layer's keys.

    Walks up the module tree from the attention layer and takes the nearest
    ancestor's ``rotary_emb`` child (DeepSeek-style indexers keep theirs under
    a different name, so the name matters).
    """
    parts = layer_name.split(".")
    for depth in range(len(parts) - 1, -1, -1):
        name = ".".join(parts[:depth])
        parent = model.get_submodule(name) if name else model
        ropes = {
            child_name: child
            for child_name, child in parent.named_children()
            if isinstance(child, RotaryEmbedding)
        }
        if "rotary_emb" in ropes:
            return ropes["rotary_emb"]
        if len(ropes) == 1:
            return next(iter(ropes.values()))
        if ropes:
            raise ValueError(
                f"{layer_name}: ambiguous rotary embeddings on {name}: {sorted(ropes)}"
            )
    raise ValueError(f"{layer_name}: no RotaryEmbedding found above this layer")


def derive_rope_descriptors(
    model: nn.Module, kv_caches: dict[str, torch.Tensor]
) -> list[RopeDescriptor]:
    """One descriptor per KV cache layer, from the loaded model.

    Args:
        model: The loaded model (this rank's shard).
        kv_caches: Layer name to ``[B, H, N, C]`` cache view, as returned by
            the worker's KV cache allocation.

    Raises:
        NotImplementedError: for a cache layer that is not dense or MLA
            attention, for cross-layer KV sharing, or for an unsupported
            KV cache dtype.
    """
    attn_layers = {
        module.layer_name: module
        for module in model.modules()
        if isinstance(module, Attention | MLAAttention)
    }
    seen: dict[tuple[int, tuple[int, ...], tuple[int, ...]], str] = {}
    descriptors = []
    for layer_name, kv_cache in kv_caches.items():
        layer = attn_layers.get(layer_name)
        if layer is None:
            raise NotImplementedError(
                f"{layer_name}: not an Attention/MLAAttention layer; KV surgery "
                "cannot address its cache"
            )
        key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()))
        if key in seen:
            raise NotImplementedError(
                f"{layer_name} shares its KV cache with {seen[key]}; KV surgery "
                "does not support cross-layer KV sharing"
            )
        seen[key] = layer_name

        rope = _find_rotary_emb(model, layer_name)
        if isinstance(layer, MLAAttention):
            rot_offset = layer.kv_lora_rank
            rotary_dim = layer.qk_rope_head_dim
            if rope.rotary_dim != rotary_dim:
                raise ValueError(
                    f"{layer_name}: rope rotary_dim {rope.rotary_dim} != "
                    f"qk_rope_head_dim {rotary_dim}"
                )
        else:
            rot_offset = 0
            rotary_dim = rope.rotary_dim
            if rope.head_size != layer.head_size:
                raise ValueError(
                    f"{layer_name}: rope head_size {rope.head_size} != attention "
                    f"head_size {layer.head_size}"
                )

        k_scale = None
        if is_quantized_kv_cache(layer.kv_cache_dtype):
            if layer.kv_cache_dtype not in _SUPPORTED_FP8_KV_DTYPES:
                raise NotImplementedError(
                    f"{layer_name}: KV surgery does not support kv_cache_dtype "
                    f"{layer.kv_cache_dtype!r}"
                )
            k_scale = layer._k_scale

        descriptors.append(
            RopeDescriptor(
                layer_name=layer_name,
                kv_cache=kv_cache,
                rot_offset=rot_offset,
                rotary_dim=rotary_dim,
                is_neox_style=rope.is_neox_style,
                inv_freq=inv_freq_of(rope).to(kv_cache.device),
                k_scale=k_scale,
            )
        )
    return descriptors
