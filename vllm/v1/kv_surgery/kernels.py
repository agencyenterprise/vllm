# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two cache passes KV surgery needs: ``rotate`` and ``gather``.

Both are memory-bound, trivially parallel, and run once per edit (about once
per 10^5 generated tokens), so they are written for clarity, not speed.
"""

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.kv_surgery.rope_descriptor import RopeDescriptor
from vllm.v1.kv_surgery.view import EditPlan


@triton.jit
def _rotate_slots_kernel(
    kv_ptr,
    slots_ptr,
    deltas_ptr,
    inv_freq_ptr,
    k_scale_ptr,
    stride_b,
    stride_h,
    stride_n,
    block_size,
    rot_offset,
    HALF: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    IS_NEOX: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    # One program per (slot, head): rotate the rotary slice of that K row by
    # delta positions. Pure orthonormal rotation; no mscale anywhere.
    i = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(slots_ptr + i).to(tl.int64)
    row = (
        kv_ptr
        + (slot // block_size) * stride_b
        + h * stride_h
        + (slot % block_size) * stride_n
        + rot_offset
    )

    j = tl.arange(0, BLOCK_HALF)
    mask = j < HALF
    if IS_NEOX:
        idx1 = j
        idx2 = j + HALF
    else:
        idx1 = 2 * j
        idx2 = 2 * j + 1

    x1 = tl.load(row + idx1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(row + idx2, mask=mask, other=0.0).to(tl.float32)
    if IS_FP8:
        scale = tl.load(k_scale_ptr)
        x1 = x1 * scale
        x2 = x2 * scale

    # Angle in float64, reduced to [-pi, pi) before the float32 cos/sin, so
    # large deltas keep full precision and rotate(d) is exactly rotate(-d)^-1.
    delta = tl.load(deltas_ptr + i)
    inv_freq = tl.load(inv_freq_ptr + j, mask=mask, other=0.0).to(tl.float64)
    angle = delta * inv_freq
    angle = angle - tl.floor(angle / 6.283185307179586 + 0.5) * 6.283185307179586
    angle = angle.to(tl.float32)
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if IS_FP8:
        o1 = o1 / scale
        o2 = o2 / scale
    tl.store(row + idx1, o1.to(row.dtype.element_ty), mask=mask)
    tl.store(row + idx2, o2.to(row.dtype.element_ty), mask=mask)


def rotate_slots(
    desc: RopeDescriptor, slots: torch.Tensor, deltas: torch.Tensor
) -> None:
    """Rotate the K of ``slots`` by ``deltas`` positions, in place.

    Args:
        desc: The layer's RoPE descriptor.
        slots: int64 ``[n]`` physical slot ids (``block_id * block_size + offset``).
        deltas: float64 ``[n]`` position deltas, one per slot; fractional allowed.
    """
    n = slots.numel()
    if n == 0:
        return
    if slots.dtype != torch.int64 or deltas.dtype != torch.float64:
        raise TypeError("slots must be int64 and deltas float64")
    if slots.shape != deltas.shape:
        raise ValueError("slots and deltas must have the same shape")

    kv = desc.kv_cache
    is_fp8 = desc.k_scale is not None
    if is_fp8:
        kv = kv.view(current_platform.fp8_dtype())
        k_scale = desc.k_scale
    else:
        k_scale = desc.inv_freq  # unused placeholder; IS_FP8 is False
    half = desc.rotary_dim // 2
    _rotate_slots_kernel[(n, desc.num_heads)](
        kv,
        slots,
        deltas,
        desc.inv_freq,
        k_scale,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        desc.block_size,
        desc.rot_offset,
        HALF=half,
        BLOCK_HALF=triton.next_power_of_2(half),
        IS_NEOX=desc.is_neox_style,
        IS_FP8=is_fp8,
    )


def gather_slots(
    kv_cache: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    chunk_rows: int = 8192,
) -> None:
    """Copy whole cache rows (all heads, K and V) from ``src_slots`` to ``dst_slots``.

    Semantics are a simultaneous move: every source row is read before any
    destination row is written, so overlapping ranges (compaction, insertion)
    behave like ``memmove``. The staging buffer holds all moved rows.

    Args:
        kv_cache: ``[num_blocks, num_heads, block_size, C]`` layer view.
        src_slots: int64 ``[n]`` physical slot ids to read.
        dst_slots: int64 ``[n]`` physical slot ids to write.
        chunk_rows: Rows per indexing call; bounds temporary index tensors.
    """
    n = src_slots.numel()
    if n == 0:
        return
    if src_slots.shape != dst_slots.shape:
        raise ValueError("src_slots and dst_slots must have the same shape")
    if kv_cache.dim() != 4:
        raise ValueError("kv_cache must be [B, H, N, C]")
    if kv_cache.element_size() == 1:
        kv_cache = kv_cache.view(torch.uint8)

    block_size = kv_cache.shape[2]
    staged = torch.empty(
        (n, kv_cache.shape[1], kv_cache.shape[3]),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    for start in range(0, n, chunk_rows):
        src = src_slots[start : start + chunk_rows]
        staged[start : start + chunk_rows] = kv_cache[
            src // block_size, :, src % block_size, :
        ]
    for start in range(0, n, chunk_rows):
        dst = dst_slots[start : start + chunk_rows]
        kv_cache[dst // block_size, :, dst % block_size, :] = staged[
            start : start + chunk_rows
        ]


def apply_edit_plan(plan: EditPlan, descriptors: list[RopeDescriptor]) -> None:
    """Apply a view op's cache mutations to every layer of one KV cache group.

    Gathers first, then rotations, as ``EditPlan`` specifies. Freed blocks are
    the scheduler's business and are ignored here.
    """
    if not descriptors:
        return
    device = descriptors[0].kv_cache.device
    if plan.gather_src.size:
        src = torch.from_numpy(plan.gather_src).to(device)
        dst = torch.from_numpy(plan.gather_dst).to(device)
        for desc in descriptors:
            gather_slots(desc.kv_cache, src, dst)
    if plan.rotate_slots.size:
        slots = torch.from_numpy(plan.rotate_slots).to(device)
        deltas = torch.from_numpy(np.ascontiguousarray(plan.rotate_deltas)).to(device)
        for desc in descriptors:
            rotate_slots(desc, slots, deltas)
