# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""View algebra for KV surgery. Pure Python + numpy; no model, no torch.

A ``View`` is what a request *is*: an ordered list of KV cache blocks and one
RoPE position per occupied slot. Logical slot ``i`` lives at physical slot
``block_ids[i // block_size] * block_size + i % block_size``; physical order is
logical order, always. Positions are per slot and may be fractional.

Every op is pure: it returns the new ``View`` together with an ``EditPlan``
listing the cache mutations that make the physical cache agree with it.
Forking a view is copying the value; sharing the blocks is the scheduler's
job (refcounts), not the view's.
"""

from dataclasses import dataclass

import numpy as np

from vllm.utils.math_utils import cdiv


@dataclass(frozen=True)
class View:
    block_size: int
    block_ids: tuple[int, ...]
    positions: np.ndarray
    """float64 ``[num_slots]``: the RoPE position baked into each slot's K."""

    def __post_init__(self) -> None:
        positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        if positions.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "block_ids", tuple(int(b) for b in self.block_ids))
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if len(self.block_ids) < self.num_blocks_needed:
            raise ValueError(
                f"{len(self.block_ids)} blocks cannot hold {self.num_slots} slots "
                f"of block size {self.block_size}"
            )

    @classmethod
    def contiguous(
        cls,
        block_size: int,
        block_ids: list[int] | tuple[int, ...],
        num_slots: int,
        start_position: float = 0.0,
    ) -> "View":
        """The stock vLLM request: positions ``start, start + 1, ...``."""
        positions = start_position + np.arange(num_slots, dtype=np.float64)
        return cls(block_size, tuple(block_ids), positions)

    @property
    def num_slots(self) -> int:
        return len(self.positions)

    @property
    def num_blocks_needed(self) -> int:
        return cdiv(self.num_slots, self.block_size)

    @property
    def next_position(self) -> float:
        """Position the next generated token gets by default."""
        if self.num_slots == 0:
            return 0.0
        return float(self.positions[-1]) + 1.0

    def slot_ids(self, start: int = 0, end: int | None = None) -> np.ndarray:
        """Physical slot ids of logical slots ``[start, end)``."""
        if end is None:
            end = self.num_slots
        self.check_span(start, end)
        idx = np.arange(start, end, dtype=np.int64)
        blocks = np.asarray(self.block_ids, dtype=np.int64)
        return blocks[idx // self.block_size] * self.block_size + idx % self.block_size

    def check_span(self, start: int, end: int) -> None:
        """Raise ``ValueError`` unless ``[start, end)`` lies within the view."""
        if not 0 <= start <= end <= self.num_slots:
            raise ValueError(
                f"span [{start}, {end}) is not within [0, {self.num_slots}]"
            )


@dataclass(frozen=True)
class EditPlan:
    """Cache mutations for one view op.

    Apply in field order: every row in ``gather_src`` is read before any row in
    ``gather_dst`` is written (staged copy), then ``rotate_slots`` are rotated
    by ``rotate_deltas``. Rotation slot ids refer to the post-gather layout.
    Slot ids are physical; deltas are positions (float64, fractional allowed).
    """

    gather_src: np.ndarray
    gather_dst: np.ndarray
    rotate_slots: np.ndarray
    rotate_deltas: np.ndarray
    freed_block_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("gather_src", "gather_dst", "rotate_slots"):
            object.__setattr__(
                self, name, np.ascontiguousarray(getattr(self, name), dtype=np.int64)
            )
        object.__setattr__(
            self,
            "rotate_deltas",
            np.ascontiguousarray(self.rotate_deltas, dtype=np.float64),
        )
        if self.gather_src.shape != self.gather_dst.shape:
            raise ValueError("gather_src and gather_dst must have the same shape")
        if self.rotate_slots.shape != self.rotate_deltas.shape:
            raise ValueError("rotate_slots and rotate_deltas must have the same shape")

    @property
    def is_empty(self) -> bool:
        return (
            self.gather_src.size == 0
            and self.rotate_slots.size == 0
            and not self.freed_block_ids
        )


_NO_SLOTS = np.zeros(0, dtype=np.int64)
_NO_DELTAS = np.zeros(0, dtype=np.float64)


def shift(view: View, start: int, end: int, delta: float) -> tuple[View, EditPlan]:
    """Move logical slots ``[start, end)`` by ``delta`` positions.

    Rotation only; nothing is copied. Exact for K because
    ``R(delta) R(p) k = R(p + delta) k``; V is position-free.
    """
    view.check_span(start, end)
    positions = view.positions.copy()
    positions[start:end] += delta
    new_view = View(view.block_size, view.block_ids, positions)
    slots = view.slot_ids(start, end)
    plan = EditPlan(
        gather_src=_NO_SLOTS,
        gather_dst=_NO_SLOTS,
        rotate_slots=slots,
        rotate_deltas=np.full(slots.shape, delta, dtype=np.float64),
    )
    return new_view, plan


def drop(view: View, start: int, end: int, close_gap: bool) -> tuple[View, EditPlan]:
    """Remove logical slots ``[start, end)``; later slots move up to fill the hole.

    With ``close_gap`` the survivors after the span are also shifted by
    ``-(positions[end] - positions[start])`` so the context stays contiguous
    in position space; their K keeps the values computed with the span present.
    Blocks no longer needed for the shorter view are reported freed.
    """
    view.check_span(start, end)
    n = view.num_slots
    removed = end - start
    kept = n - removed
    positions = np.concatenate([view.positions[:start], view.positions[end:]])
    if close_gap and end < n:
        gap = view.positions[end] - view.positions[start]
        positions[start:] -= gap
    else:
        gap = 0.0

    num_blocks = cdiv(kept, view.block_size)
    new_view = View(view.block_size, view.block_ids[:num_blocks], positions)

    if end < n:
        gather_src = view.slot_ids(end, n)
        gather_dst = view.slot_ids(start, kept)
        # Only an empty span yields identity rows (a non-empty span moves
        # every survivor); drop them so a no-op plan writes nothing, since
        # the scheduler reads gather_dst as the set of written slots. Sound
        # because gather_slots stages every read before any write.
        moved = gather_src != gather_dst
        gather_src, gather_dst = gather_src[moved], gather_dst[moved]
    else:
        gather_src = gather_dst = _NO_SLOTS
    if gap != 0.0:
        rotate_slots = new_view.slot_ids(start, kept)
        rotate_deltas = np.full(rotate_slots.shape, -gap, dtype=np.float64)
    else:
        rotate_slots, rotate_deltas = _NO_SLOTS, _NO_DELTAS

    plan = EditPlan(
        gather_src=gather_src,
        gather_dst=gather_dst,
        rotate_slots=rotate_slots,
        rotate_deltas=rotate_deltas,
        freed_block_ids=view.block_ids[num_blocks:],
    )
    return new_view, plan


def continue_positions(positions: np.ndarray | None, num_slots: int) -> np.ndarray:
    """Positions of ``num_slots`` slots after stock decode grew a view.

    ``positions`` are the per-slot positions recorded at the last edit (None
    for a never-edited request); slots added since continue from
    ``next_position`` one per slot, which is exactly what the model runner
    assigned them (slot index plus the request's position offset).
    """
    if positions is None:
        return np.arange(num_slots, dtype=np.float64)
    if num_slots < len(positions):
        raise ValueError(
            f"view has {len(positions)} slots but the request only has {num_slots}"
        )
    start = float(positions[-1]) + 1.0 if len(positions) else 0.0
    grown = start + np.arange(num_slots - len(positions), dtype=np.float64)
    return np.concatenate([positions, grown])


def splice(
    view: View,
    start: int,
    end: int,
    source: View,
    src_start: int,
    src_end: int,
    close_gap: bool,
    extra_block_ids: tuple[int, ...] = (),
) -> tuple[View, EditPlan]:
    """Replace logical slots ``[start, end)`` with a copy of ``source``'s
    slots ``[src_start, src_end)``.

    The copied slots keep the positions they were computed at; ``source`` may
    be another request's view or ``view`` itself. With ``close_gap`` the
    survivors after the span are shifted so the first of them sits one
    position after the last inserted slot (or, when nothing is inserted, at
    the position the span started at, which is ``drop``). A view that grows
    takes blocks from ``extra_block_ids`` in order; one that shrinks reports
    its tail blocks freed.
    """
    view.check_span(start, end)
    source.check_span(src_start, src_end)
    n = view.num_slots
    inserted = src_end - src_start
    kept = n - (end - start) + inserted

    tail = view.positions[end:]
    gap = 0.0
    if close_gap and end < n:
        resume = (
            float(source.positions[src_end - 1]) + 1.0
            if inserted
            else float(view.positions[start])
        )
        gap = float(view.positions[end]) - resume
        tail = tail - gap
    positions = np.concatenate(
        [view.positions[:start], source.positions[src_start:src_end], tail]
    )

    num_blocks = cdiv(kept, view.block_size)
    needed = max(0, num_blocks - len(view.block_ids))
    if len(extra_block_ids) != needed:
        # Exact count: a surplus block would belong to nobody.
        raise ValueError(
            f"splice needs {needed} extra blocks, got {len(extra_block_ids)}"
        )
    block_ids = (view.block_ids + tuple(extra_block_ids))[:num_blocks]
    new_view = View(view.block_size, block_ids, positions)

    gather_src = np.concatenate(
        [source.slot_ids(src_start, src_end), view.slot_ids(end, n)]
    )
    gather_dst = new_view.slot_ids(start, kept)
    # Copying a slot onto itself (the source shares blocks with this view)
    # is not a write; the scheduler reads gather_dst as the written slots.
    # Sound because gather_slots stages every read before any write.
    moved = gather_src != gather_dst
    if gap != 0.0:
        rotate_slots = new_view.slot_ids(start + inserted, kept)
        rotate_deltas = np.full(rotate_slots.shape, -gap, dtype=np.float64)
    else:
        rotate_slots, rotate_deltas = _NO_SLOTS, _NO_DELTAS

    plan = EditPlan(
        gather_src=gather_src[moved],
        gather_dst=gather_dst[moved],
        rotate_slots=rotate_slots,
        rotate_deltas=rotate_deltas,
        freed_block_ids=view.block_ids[num_blocks:],
    )
    return new_view, plan


def fork(
    view: View, num_slots: int, fresh_block_ids: tuple[int, ...] = ()
) -> tuple[View, EditPlan, int]:
    """A new view of the first ``num_slots`` slots that shares ``view``'s
    fully occupied blocks and owns a copy of the partially occupied one.

    Sharing is by block id; the scheduler holds the reference counts. The
    partial block (if any) is copied into ``fresh_block_ids[0]`` so both
    views can append to it independently. Returns the new view, its plan
    and the number of shared blocks.
    """
    view.check_span(0, num_slots)
    num_shared = num_slots // view.block_size
    shared = view.block_ids[:num_shared]
    positions = view.positions[:num_slots]
    needed = int(num_slots != num_shared * view.block_size)
    if len(fresh_block_ids) != needed:
        # Exact count: a surplus block would belong to nobody.
        raise ValueError(
            f"fork needs {needed} fresh block(s) for the partial last block, "
            f"got {len(fresh_block_ids)}"
        )
    if not needed:
        plan = EditPlan(_NO_SLOTS, _NO_SLOTS, _NO_SLOTS, _NO_DELTAS)
        return View(view.block_size, shared, positions), plan, num_shared
    new_view = View(view.block_size, shared + (fresh_block_ids[0],), positions)
    first_partial = num_shared * view.block_size
    plan = EditPlan(
        gather_src=view.slot_ids(first_partial, num_slots),
        gather_dst=new_view.slot_ids(first_partial, num_slots),
        rotate_slots=_NO_SLOTS,
        rotate_deltas=_NO_DELTAS,
    )
    return new_view, plan, num_shared
