# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""View algebra: pure bookkeeping, no model, no GPU.

The invariant under test is that physical slot order is logical order and
that every op's ``EditPlan`` is exactly the set of cache mutations needed to
keep it so.
"""

import numpy as np
import pytest

from vllm.v1.kv_surgery.view import (
    View,
    continue_positions,
    drop,
    fork,
    shift,
    splice,
)

BLOCK_SIZE = 4


def _view(num_slots: int, start_position: float = 0.0) -> View:
    # Non-consecutive block ids so physical slot ids differ from logical ones.
    block_ids = [10, 3, 7, 12, 5, 9][: -(-num_slots // BLOCK_SIZE)]
    return View.contiguous(BLOCK_SIZE, block_ids, num_slots, start_position)


def test_slot_ids_follow_block_table():
    view = _view(10)
    np.testing.assert_array_equal(view.slot_ids(0, 5), [40, 41, 42, 43, 12])
    np.testing.assert_array_equal(view.slot_ids(8, 10), [28, 29])
    assert view.next_position == 10.0


def test_too_few_blocks_rejected():
    with pytest.raises(ValueError, match="cannot hold"):
        View(BLOCK_SIZE, (1,), np.arange(5.0))


def test_shift_rotates_only():
    view = _view(10)
    new, plan = shift(view, 4, 8, -2.5)
    np.testing.assert_array_equal(new.positions, [0, 1, 2, 3, 1.5, 2.5, 3.5, 4.5, 8, 9])
    assert new.block_ids == view.block_ids
    assert plan.gather_src.size == 0
    np.testing.assert_array_equal(plan.rotate_slots, view.slot_ids(4, 8))
    np.testing.assert_array_equal(plan.rotate_deltas, [-2.5] * 4)
    assert plan.freed_block_ids == ()


def test_drop_middle_close_gap_compacts_then_rotates():
    view = _view(10)  # blocks 10, 3, 7
    new, plan = drop(view, 2, 5, close_gap=True)
    # Survivors keep their order; positions close the gap of 3.
    np.testing.assert_array_equal(new.positions, np.arange(7.0))
    assert new.num_slots == 7
    assert new.block_ids == (10, 3)
    assert plan.freed_block_ids == (7,)
    # Old logical [5, 10) moves to logical [2, 7).
    np.testing.assert_array_equal(plan.gather_src, view.slot_ids(5, 10))
    np.testing.assert_array_equal(plan.gather_dst, view.slot_ids(2, 7))
    # Rotation addresses the post-gather slots.
    np.testing.assert_array_equal(plan.rotate_slots, new.slot_ids(2, 7))
    np.testing.assert_array_equal(plan.rotate_deltas, [-3.0] * 5)


def test_drop_without_close_gap_keeps_positions():
    view = _view(10)
    new, plan = drop(view, 2, 5, close_gap=False)
    np.testing.assert_array_equal(new.positions, [0, 1, 5, 6, 7, 8, 9])
    assert plan.rotate_slots.size == 0
    assert plan.gather_src.size == 5


def test_drop_tail_only_frees():
    view = _view(10)
    new, plan = drop(view, 6, 10, close_gap=True)
    np.testing.assert_array_equal(new.positions, np.arange(6.0))
    assert plan.gather_src.size == 0
    assert plan.rotate_slots.size == 0
    assert plan.freed_block_ids == (7,)
    assert new.block_ids == (10, 3)


def test_drop_gap_is_measured_in_positions():
    """With fractional/non-contiguous positions the gap is the position
    difference across the span, not the slot count."""
    positions = np.array([0.0, 1.0, 2.5, 4.0, 10.0, 11.0])
    view = View(BLOCK_SIZE, (1, 2), positions)
    new, plan = drop(view, 2, 4, close_gap=True)
    np.testing.assert_array_equal(new.positions, [0.0, 1.0, 2.5, 3.5])
    np.testing.assert_array_equal(plan.rotate_deltas, [-7.5, -7.5])


@pytest.mark.parametrize("span", [(-1, 2), (3, 2), (0, 11)])
def test_bad_spans_rejected(span):
    view = _view(10)
    with pytest.raises(ValueError):
        shift(view, *span, 1.0)
    with pytest.raises(ValueError):
        drop(view, *span, close_gap=True)


def test_continue_positions_extends_from_next_position():
    np.testing.assert_array_equal(continue_positions(None, 4), [0.0, 1.0, 2.0, 3.0])
    edited = np.array([0.0, 1.0, 5.5])
    np.testing.assert_array_equal(
        continue_positions(edited, 5), [0.0, 1.0, 5.5, 6.5, 7.5]
    )
    np.testing.assert_array_equal(continue_positions(np.zeros(0), 2), [0.0, 1.0])
    with pytest.raises(ValueError):
        continue_positions(edited, 2)


def test_splice_copies_source_slots_and_closes_the_gap():
    view = _view(10)
    source = View.contiguous(BLOCK_SIZE, [20, 21], 6, start_position=100.0)
    new, plan = splice(view, 2, 5, source, 1, 4, close_gap=True)
    # Three inserted slots keep their positions; the tail resumes after them.
    np.testing.assert_array_equal(
        new.positions, [0, 1, 101, 102, 103, 104, 105, 106, 107, 108]
    )
    assert new.block_ids == view.block_ids
    np.testing.assert_array_equal(plan.gather_src, [81, 82, 83])
    np.testing.assert_array_equal(plan.gather_dst, [42, 43, 12])
    # The tail stays put (identity copies are dropped) but is re-rotated.
    np.testing.assert_array_equal(plan.rotate_slots, view.slot_ids(5, 10))
    np.testing.assert_array_equal(plan.rotate_deltas, 99.0)
    assert plan.freed_block_ids == ()


def test_splice_grows_into_extra_blocks_and_shrinks_by_freeing():
    view = _view(10)
    source = View.contiguous(BLOCK_SIZE, [20, 21, 22], 12, start_position=50.0)
    with pytest.raises(ValueError, match="extra blocks"):
        splice(view, 10, 10, source, 0, 8, True, extra_block_ids=(30,))
    with pytest.raises(ValueError, match="extra blocks"):  # surplus, too
        splice(view, 10, 10, source, 0, 8, True, extra_block_ids=(30, 31, 32))
    new, plan = splice(view, 10, 10, source, 0, 8, True, extra_block_ids=(30, 31))
    assert new.block_ids == view.block_ids + (30, 31)
    np.testing.assert_array_equal(new.positions[10:], 50 + np.arange(8))
    np.testing.assert_array_equal(plan.gather_src, source.slot_ids(0, 8))
    np.testing.assert_array_equal(plan.gather_dst, new.slot_ids(10, 18))
    assert plan.rotate_slots.size == 0

    new, plan = splice(view, 2, 8, source, 0, 1, close_gap=False)
    assert new.block_ids == (10, 3)
    assert plan.freed_block_ids == (7,)
    np.testing.assert_array_equal(new.positions, [0, 1, 50, 8, 9])


def test_splice_of_an_empty_span_is_drop():
    view = _view(10)
    dropped, drop_plan = drop(view, 3, 6, close_gap=True)
    spliced, splice_plan = splice(view, 3, 6, view, 0, 0, close_gap=True)
    np.testing.assert_array_equal(dropped.positions, spliced.positions)
    assert dropped.block_ids == spliced.block_ids
    for name in ("gather_src", "gather_dst", "rotate_slots", "rotate_deltas"):
        np.testing.assert_array_equal(
            getattr(drop_plan, name), getattr(splice_plan, name)
        )


def test_copies_onto_the_same_slot_are_not_writes():
    """A source view that shares blocks with the target (a fork) can be
    spliced over the shared prefix without writing it: the scheduler decides
    what is shared from ``gather_dst``, so identity rows must not appear."""
    view = _view(10)  # blocks (10, 3, 7)
    fork = View.contiguous(BLOCK_SIZE, [10, 20], 6)  # shares block 10
    new, plan = splice(view, 0, 4, fork, 0, 6, close_gap=False)
    assert new.num_slots == 12
    assert not set((plan.gather_dst // BLOCK_SIZE).tolist()) & {10}
    np.testing.assert_array_equal(
        plan.gather_src, [80, 81] + view.slot_ids(4, 10).tolist()
    )
    np.testing.assert_array_equal(plan.gather_dst, new.slot_ids(4, 12))
    # Dropping nothing writes nothing.
    _, plan = drop(view, 5, 5, close_gap=True)
    assert plan.is_empty


def test_self_splice_moves_overlapping_slots_as_a_staged_copy():
    """Duplicating a span inside one view: the tail moves over slots that are
    also gather sources, which is only right because ``EditPlan`` reads every
    source before writing any destination."""
    view = _view(10)
    new, plan = splice(view, 2, 2, view, 5, 9, close_gap=False, extra_block_ids=(30,))
    assert new.num_slots == 14
    assert new.block_ids == view.block_ids + (30,)
    np.testing.assert_array_equal(
        new.positions, [0, 1, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    # Some destinations are other rows' sources.
    assert set(plan.gather_dst.tolist()) & set(plan.gather_src.tolist())

    # Emulate the staged copy on a fake cache that labels each physical slot
    # with the logical slot it held before the edit.
    cache = np.full(40 * BLOCK_SIZE, -1)
    cache[view.slot_ids()] = np.arange(10)
    staged = cache[plan.gather_src].copy()
    cache[plan.gather_dst] = staged
    np.testing.assert_array_equal(
        cache[new.slot_ids()], [0, 1, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9]
    )


def test_fork_shares_whole_blocks_and_copies_the_partial_one():
    view = _view(10)
    new, plan, num_shared = fork(view, 8)
    assert num_shared == 2
    assert new.block_ids == (10, 3)
    assert plan.is_empty
    np.testing.assert_array_equal(new.positions, np.arange(8))

    with pytest.raises(ValueError, match="fresh block"):
        fork(view, 6)
    with pytest.raises(ValueError, match="fresh block"):  # surplus, too
        fork(view, 8, fresh_block_ids=(30,))
    new, plan, num_shared = fork(view, 6, fresh_block_ids=(30,))
    assert num_shared == 1
    assert new.block_ids == (10, 30)
    np.testing.assert_array_equal(plan.gather_src, [12, 13])
    np.testing.assert_array_equal(plan.gather_dst, [120, 121])
    assert new.next_position == 6.0
