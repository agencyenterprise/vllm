# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``rotate`` and ``gather`` against a from-scratch recompute.

The oracle is vLLM's own ``RotaryEmbedding.forward_native``: K stored for
position ``p`` and then rotated by ``delta`` must equal K freshly computed
for position ``p + delta``. Geometries mirror the Tier 1 targets: dense GQA
(Llama 3, neox) with K|V rows, and MLA (DeepSeek-V3 decoder, interleaved)
with the rope-k slice at the end of a ``[latent | rope-k]`` row.
"""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_surgery.kernels import apply_edit_plan, gather_slots, rotate_slots
from vllm.v1.kv_surgery.rope_descriptor import RopeDescriptor, inv_freq_of
from vllm.v1.kv_surgery.view import View, drop

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KV surgery kernels need a GPU"
)

DEVICE = "cuda"
MAX_POS = 8192
NUM_BLOCKS = 16
BLOCK_SIZE = 16
FP8_SCALE = 0.05

# name -> (num_heads, head_size, rotary_dim, row_width, rot_offset)
GEOMETRIES = {
    "dense": (4, 64, 64, 128, 0),
    "dense_partial_rotary": (2, 128, 64, 256, 0),
    "mla": (1, 64, 64, 576, 512),
}
TOLERANCE = {
    torch.float32: dict(atol=4e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=3e-2, rtol=2e-2),
    # fp8 e4m3 storage: both sides carry a half-ulp of 0.25 at |x| ~ 4, and the
    # rotation mixes each pair's error into its partner.
    torch.uint8: dict(atol=3e-1, rtol=2.5e-1),
}


@pytest.fixture(autouse=True)
def _vllm_config(default_vllm_config):
    """RotaryEmbedding is a CustomOp and needs a current vLLM config."""


def make_rope(kind: str, head_size: int, rotary_dim: int) -> RotaryEmbedding:
    if kind == "plain_interleaved":
        rope = RotaryEmbedding(
            head_size, rotary_dim, MAX_POS, 5e4, False, torch.float32
        )
    elif kind == "llama3":
        rope = Llama3RotaryEmbedding(
            head_size,
            rotary_dim,
            MAX_POS,
            5e5,
            True,
            torch.float32,
            32.0,
            1.0,
            4.0,
            8192,
        )
    elif kind == "yarn_mscale":
        rope = DeepseekScalingRotaryEmbedding(
            head_size, rotary_dim, MAX_POS, 1e4, False, 40.0, torch.float32, mscale=1.0
        )
        assert rope.mscale != 1.0
    else:
        raise ValueError(kind)
    return rope.to(DEVICE)


def make_cache(
    layout: KVCacheLayout, num_heads: int, row_width: int, dtype: torch.dtype
) -> torch.Tensor:
    """A random ``[B, H, N, C]`` view whose memory order follows ``layout``."""
    logical = (NUM_BLOCKS, num_heads, BLOCK_SIZE, row_width)
    order = layout.layer_view_order
    physical = torch.empty([logical[ax] for ax in order], dtype=dtype, device=DEVICE)
    if dtype == torch.uint8:
        physical.random_(0, 255)
    else:
        physical.normal_()
    inverse = [0] * 4
    for k, ax in enumerate(order):
        inverse[ax] = k
    view = physical.permute(inverse)
    assert view.shape == logical
    return view


def reference_rotated(
    rope: RotaryEmbedding, positions: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """vLLM's own rotation of ``k`` ``[n, H, head_size]`` at integer positions."""
    _, out = rope.forward_native(positions, torch.zeros_like(k), k.clone())
    return out


def write_k(desc: RopeDescriptor, slots: torch.Tensor, k: torch.Tensor) -> None:
    """Store ``k`` ``[n, H, rotary_dim]`` (float32) into the rotary slices."""
    blk, off = slots // desc.block_size, slots % desc.block_size
    sl = slice(desc.rot_offset, desc.rot_offset + desc.rotary_dim)
    if desc.k_scale is not None:
        vals = (k / desc.k_scale).to(current_platform.fp8_dtype()).view(torch.uint8)
    else:
        vals = k.to(desc.kv_cache.dtype)
    desc.kv_cache[blk, :, off, sl] = vals


def read_k(desc: RopeDescriptor, slots: torch.Tensor) -> torch.Tensor:
    """Dequantized rotary slices of ``slots`` as float32 ``[n, H, rotary_dim]``."""
    blk, off = slots // desc.block_size, slots % desc.block_size
    sl = slice(desc.rot_offset, desc.rot_offset + desc.rotary_dim)
    vals = desc.kv_cache[blk, :, off, sl]
    if desc.k_scale is not None:
        return vals.view(current_platform.fp8_dtype()).float() * desc.k_scale
    return vals.float()


def make_descriptor(
    rope: RotaryEmbedding,
    geometry: str,
    dtype: torch.dtype,
    layout: KVCacheLayout = KVCacheLayout.LBNHC,
) -> RopeDescriptor:
    num_heads, _, rotary_dim, row_width, rot_offset = GEOMETRIES[geometry]
    k_scale = None
    if dtype == torch.uint8:
        k_scale = torch.tensor(FP8_SCALE, dtype=torch.float32, device=DEVICE)
    return RopeDescriptor(
        layer_name=geometry,
        kv_cache=make_cache(layout, num_heads, row_width, dtype),
        rot_offset=rot_offset,
        rotary_dim=rotary_dim,
        is_neox_style=rope.is_neox_style,
        inv_freq=inv_freq_of(rope).to(DEVICE),
        k_scale=k_scale,
    )


def random_slots(n: int) -> torch.Tensor:
    perm = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=DEVICE)[:n]
    return perm.sort().values.to(torch.int64)


def assert_untouched_outside(
    desc: RopeDescriptor, before: torch.Tensor, slots: torch.Tensor
) -> None:
    """Everything except the rotary slices of ``slots`` is byte-identical."""
    after = desc.kv_cache.clone()
    blk, off = slots // desc.block_size, slots % desc.block_size
    sl = slice(desc.rot_offset, desc.rot_offset + desc.rotary_dim)
    after[blk, :, off, sl] = before[blk, :, off, sl]
    assert torch.equal(after, before)


@pytest.mark.parametrize("kind", ["plain_interleaved", "llama3", "yarn_mscale"])
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.uint8])
def test_rotate_matches_recompute_at_shifted_position(kind, geometry, dtype):
    torch.manual_seed(0)
    num_heads, head_size, rotary_dim, _, _ = GEOMETRIES[geometry]
    rope = make_rope(kind, head_size, rotary_dim)
    desc = make_descriptor(rope, geometry, dtype)

    n = 96
    slots = random_slots(n)
    positions = torch.randint(0, MAX_POS // 2, (n,), device=DEVICE)
    deltas = torch.randint(-MAX_POS // 2, MAX_POS // 2, (n,), device=DEVICE)
    deltas = deltas.clamp(-positions, MAX_POS - 1 - positions)
    k = torch.randn(n, num_heads, head_size, device=DEVICE)

    write_k(desc, slots, reference_rotated(rope, positions, k)[..., :rotary_dim])
    before = desc.kv_cache.clone()

    rotate_slots(desc, slots, deltas.to(torch.float64))

    expected = reference_rotated(rope, positions + deltas, k)[..., :rotary_dim]
    torch.testing.assert_close(read_k(desc, slots), expected, **TOLERANCE[dtype])
    assert_untouched_outside(desc, before, slots)


def test_fp8_rotation_is_dequant_rotate_requant():
    """Isolates the kernel from input quantization noise: compare against an
    exact rotation of the values actually stored, within one fp8 rounding."""
    torch.manual_seed(1)
    rope = make_rope("plain_interleaved", 64, 64)
    desc = make_descriptor(rope, "mla", torch.uint8)
    n = 64
    slots = random_slots(n)
    k = torch.randn(n, 1, 64, device=DEVICE)
    write_k(desc, slots, k)
    stored = read_k(desc, slots)  # what the kernel really sees
    delta = 123.0

    rotate_slots(
        desc, slots, torch.full((n,), delta, dtype=torch.float64, device=DEVICE)
    )

    angle = delta * desc.inv_freq.double()
    cos, sin = angle.cos().float(), angle.sin().float()
    x1, x2 = stored[..., 0::2], stored[..., 1::2]
    expected = torch.empty_like(stored)
    expected[..., 0::2] = x1 * cos - x2 * sin
    expected[..., 1::2] = x2 * cos + x1 * sin
    torch.testing.assert_close(read_k(desc, slots), expected, atol=5e-3, rtol=7e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shift_then_unshift_is_identity(dtype):
    torch.manual_seed(2)
    rope = make_rope("llama3", 64, 64)
    desc = make_descriptor(rope, "dense", dtype)
    slots = random_slots(50)
    before = read_k(desc, slots)
    deltas = torch.full((50,), 37.25, dtype=torch.float64, device=DEVICE)

    rotate_slots(desc, slots, deltas)
    rotate_slots(desc, slots, -deltas)

    tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else TOLERANCE[dtype]
    torch.testing.assert_close(read_k(desc, slots), before, **tol)


def test_fractional_shifts_compose():
    torch.manual_seed(3)
    rope = make_rope("plain_interleaved", 64, 64)
    half_steps = make_descriptor(rope, "mla", torch.float32)
    one_step = make_descriptor(rope, "mla", torch.float32)
    one_step.kv_cache.copy_(half_steps.kv_cache)
    slots = random_slots(40)

    half = torch.full((40,), 0.5, dtype=torch.float64, device=DEVICE)
    rotate_slots(half_steps, slots, half)
    rotate_slots(half_steps, slots, half)
    rotate_slots(one_step, slots, half + half)

    torch.testing.assert_close(
        read_k(half_steps, slots), read_k(one_step, slots), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC, KVCacheLayout.BLHNC]
)
def test_rotate_follows_cache_strides(layout):
    torch.manual_seed(4)
    rope = make_rope("plain_interleaved", 64, 64)
    desc = make_descriptor(rope, "dense", torch.float32, layout)
    n = 48
    slots = random_slots(n)
    positions = torch.randint(0, 1024, (n,), device=DEVICE)
    k = torch.randn(n, 4, 64, device=DEVICE)
    write_k(desc, slots, reference_rotated(rope, positions, k))
    before = desc.kv_cache.clone()

    rotate_slots(
        desc, slots, torch.full((n,), 100.0, dtype=torch.float64, device=DEVICE)
    )

    expected = reference_rotated(rope, positions + 100, k)
    torch.testing.assert_close(
        read_k(desc, slots), expected, **TOLERANCE[torch.float32]
    )
    assert_untouched_outside(desc, before, slots)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("offset", [-7, 5])
def test_gather_overlapping_ranges_move_simultaneously(dtype, offset):
    torch.manual_seed(5)
    cache = make_cache(KVCacheLayout.LBNHC, 2, 128, dtype)
    src = torch.arange(20, 90, device=DEVICE, dtype=torch.int64)
    dst = src + offset
    before = cache.clone()
    expected = cache.clone()
    expected[dst // BLOCK_SIZE, :, dst % BLOCK_SIZE, :] = before[
        src // BLOCK_SIZE, :, src % BLOCK_SIZE, :
    ]

    gather_slots(cache, src, dst, chunk_rows=16)

    assert torch.equal(cache, expected)


def test_gather_scattered_permutation():
    torch.manual_seed(6)
    cache = make_cache(KVCacheLayout.LBHNC, 3, 64, torch.bfloat16)
    src = random_slots(30)
    dst = src[torch.randperm(30, device=DEVICE)]
    expected = cache.clone()
    expected[dst // BLOCK_SIZE, :, dst % BLOCK_SIZE, :] = cache[
        src // BLOCK_SIZE, :, src % BLOCK_SIZE, :
    ]

    gather_slots(cache, src, dst)

    assert torch.equal(cache, expected)


def test_drop_plan_reproduces_recompute_for_survivors():
    """End to end on one layer: drop a middle span with close_gap, then the
    survivors after the span must hold K recomputed at their new positions
    from the same pre-rotation keys."""
    torch.manual_seed(7)
    rope = make_rope("llama3", 64, 64)
    desc = make_descriptor(rope, "dense", torch.bfloat16)
    n, start, end = 100, 30, 50
    view = View.contiguous(BLOCK_SIZE, list(range(7)), n)
    slots = torch.from_numpy(view.slot_ids()).to(DEVICE)
    positions = torch.from_numpy(view.positions).to(DEVICE).long()
    k = torch.randn(n, 4, 64, device=DEVICE)
    write_k(desc, slots, reference_rotated(rope, positions, k))
    v_before = desc.kv_cache[..., 64:].clone()

    new_view, plan = drop(view, start, end, close_gap=True)
    apply_edit_plan(plan, [desc])

    assert new_view.num_slots == n - (end - start)
    assert plan.freed_block_ids == (5, 6)
    new_slots = torch.from_numpy(new_view.slot_ids()).to(DEVICE)
    new_positions = torch.from_numpy(new_view.positions).to(DEVICE).long()
    survivors = torch.cat([k[:start], k[end:]])
    expected = reference_rotated(rope, new_positions, survivors)
    torch.testing.assert_close(
        read_k(desc, new_slots), expected, **TOLERANCE[torch.bfloat16]
    )
    # V moved with K, byte for byte.
    blk, off = new_slots // BLOCK_SIZE, new_slots % BLOCK_SIZE
    old_slots = torch.cat([slots[:start], slots[end:]])
    oblk, ooff = old_slots // BLOCK_SIZE, old_slots % BLOCK_SIZE
    assert torch.equal(desc.kv_cache[blk, :, off, 64:], v_before[oblk, :, ooff, :])
