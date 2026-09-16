# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""View algebra: pure bookkeeping, no model, no GPU.

The invariant under test is that physical slot order is logical order and
that every op's ``EditPlan`` is exactly the set of cache mutations needed to
keep it so.
"""

import numpy as np
import pytest

from vllm.v1.kv_surgery.view import View, drop, restrict, shift

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


def test_restrict_is_a_prefix_and_frees_nothing():
    view = _view(10)
    prefix = restrict(view, 5)
    np.testing.assert_array_equal(prefix.positions, np.arange(5.0))
    assert prefix.block_ids == (10, 3)
    assert view.num_slots == 10 and view.block_ids == (10, 3, 7)


@pytest.mark.parametrize("span", [(-1, 2), (3, 2), (0, 11)])
def test_bad_spans_rejected(span):
    view = _view(10)
    with pytest.raises(ValueError):
        shift(view, *span, 1.0)
    with pytest.raises(ValueError):
        drop(view, *span, close_gap=True)
