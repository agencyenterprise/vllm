# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoPE descriptor derivation from vLLM's own rotary embedding modules."""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_scaling_rope import (
    DynamicNTKScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.v1.kv_surgery.rope_descriptor import _find_rotary_emb, inv_freq_of

HEAD = 64
MAX_POS = 4096


@pytest.fixture(autouse=True)
def _vllm_config(default_vllm_config):
    """RotaryEmbedding is a CustomOp and needs a current vLLM config."""


def test_plain_rope_inv_freq_is_the_textbook_formula():
    rope = RotaryEmbedding(HEAD, HEAD, MAX_POS, 50000.0, False, torch.float32)
    expected = 1.0 / (50000.0 ** (torch.arange(0, HEAD, 2, dtype=torch.float) / HEAD))
    torch.testing.assert_close(inv_freq_of(rope), expected)


def test_llama3_inv_freq_uses_the_scaled_frequencies():
    plain = RotaryEmbedding(HEAD, HEAD, MAX_POS, 500000.0, True, torch.float32)
    llama3 = Llama3RotaryEmbedding(
        HEAD, HEAD, MAX_POS, 500000.0, True, torch.float32, 32.0, 1.0, 4.0, 8192
    )
    inv_freq = inv_freq_of(llama3)
    torch.testing.assert_close(inv_freq, llama3._compute_inv_freq(500000.0))
    assert not torch.allclose(inv_freq, inv_freq_of(plain))


def test_yarn_inv_freq_excludes_mscale():
    """DeepSeek/YaRN fold ``mscale`` into cos_sin_cache. A delta rotation must
    not carry it, so the descriptor must come from inv_freq, not the cache."""
    rope = DeepseekScalingRotaryEmbedding(
        HEAD, HEAD, MAX_POS, 10000.0, False, 40.0, torch.float32, mscale=1.0
    )
    assert rope.mscale != 1.0
    inv_freq = inv_freq_of(rope)
    torch.testing.assert_close(inv_freq, rope._compute_inv_freq(40.0))
    half = HEAD // 2
    cached_cos = rope.cos_sin_cache[1, :half]
    assert not torch.allclose(cached_cos, torch.cos(inv_freq), atol=1e-3)
    torch.testing.assert_close(cached_cos, rope.mscale * torch.cos(inv_freq))


def test_dynamic_ntk_is_rejected():
    rope = DynamicNTKScalingRotaryEmbedding(
        HEAD, HEAD, MAX_POS, MAX_POS, 10000.0, True, 2.0, torch.float32
    )
    with pytest.raises(NotImplementedError, match="DynamicNTK"):
        inv_freq_of(rope)


def test_unknown_subclass_is_rejected():
    class Mystery(RotaryEmbedding):
        pass

    rope = Mystery(HEAD, HEAD, MAX_POS, 10000.0, True, torch.float32)
    with pytest.raises(NotImplementedError, match="Mystery"):
        inv_freq_of(rope)


def test_find_rotary_emb_prefers_the_sibling_named_rotary_emb():
    rope = RotaryEmbedding(HEAD, HEAD, MAX_POS, 10000.0, False, torch.float32)
    indexer_rope = RotaryEmbedding(HEAD, HEAD, MAX_POS, 10000.0, True, torch.float32)
    self_attn = nn.Module()
    self_attn.rotary_emb = rope
    self_attn.indexer_rope_emb = indexer_rope
    self_attn.mla_attn = nn.Module()
    self_attn.mla_attn.mla_attn = nn.Module()
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = self_attn
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.attn = nn.Module()

    assert _find_rotary_emb(model, "layers.0.self_attn.mla_attn.mla_attn") is rope
    with pytest.raises(ValueError, match="no RotaryEmbedding"):
        _find_rotary_emb(model, "layers.0.mlp.attn")
