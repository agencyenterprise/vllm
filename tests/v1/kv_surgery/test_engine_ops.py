# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side edit_kv: planning, committing and re-admission. No GPU."""

import time

import msgspec
import msgspec.msgpack
import numpy as np
import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_surgery.engine_ops import commit_edit, plan_edit
from vllm.v1.kv_surgery.ops import DropOp, KVEditOp, ShiftOp, as_edit_op
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from ..core.utils import create_requests, create_scheduler

BLOCK_SIZE = 16
PROMPT_LEN = 40
NUM_DECODES = 6
MODEL = "meta-llama/Llama-3.2-1B"


def _step(scheduler: Scheduler, request: Request, token_id: int) -> None:
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[token_id]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


@pytest.fixture
def scheduler() -> Scheduler:
    return create_scheduler(
        model=MODEL,
        enable_prefix_caching=False,
        block_size=BLOCK_SIZE,
        num_blocks=64,
        use_v2_model_runner=True,
    )


@pytest.fixture
def running_request(scheduler: Scheduler) -> Request:
    """A decode request: PROMPT_LEN + NUM_DECODES computed slots, one pending
    sampled token."""
    (request,) = create_requests(
        1, num_tokens=PROMPT_LEN, max_tokens=64, block_size=BLOCK_SIZE
    )
    request.prompt_token_ids[:] = list(range(PROMPT_LEN))
    request._all_token_ids[:] = list(range(PROMPT_LEN))
    scheduler.add_request(request)
    for i in range(NUM_DECODES + 1):
        _step(scheduler, request, 1000 + i)
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == PROMPT_LEN + NUM_DECODES
    assert request.num_tokens == PROMPT_LEN + NUM_DECODES + 1
    return request


def _blocks(scheduler: Scheduler, request: Request) -> list[int]:
    return scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]


def test_drop_prompt_span_shrinks_everything_consistently(scheduler, running_request):
    request = running_request
    n = request.num_computed_tokens
    blocks_before = _blocks(scheduler, request)
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    tokens_before = list(request.all_token_ids)
    # 46 slots use 3 blocks; 26 survivors fit in 2, so one block comes back.
    start, end = 10, 30

    planned = plan_edit(scheduler, request.request_id, DropOp(start, end))
    plan = planned.plans[0]
    assert plan.gather_src.size == n - end
    np.testing.assert_array_equal(plan.rotate_deltas, -(end - start))
    assert plan.rotate_slots.size == n - end
    assert planned.position_offset == 0

    result = commit_edit(scheduler, planned)

    removed = end - start
    assert result.num_slots == request.num_computed_tokens == n - removed
    assert result.num_tokens == request.num_tokens == n - removed + 1
    assert result.next_position == n - removed
    assert list(request.all_token_ids) == tokens_before[:start] + tokens_before[end:]
    assert request.prompt_token_ids == list(range(start)) + list(range(end, PROMPT_LEN))
    assert request.num_prompt_tokens == PROMPT_LEN - removed
    assert list(request.output_token_ids) == tokens_before[PROMPT_LEN:]
    np.testing.assert_array_equal(request.kv_positions, np.arange(n - removed))
    assert request.position_offset == 0

    blocks_after = _blocks(scheduler, request)
    assert blocks_after == blocks_before[: len(blocks_after)]
    assert len(blocks_before) - len(blocks_after) == result.num_freed_blocks == 1
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        == free_before + result.num_freed_blocks
    )


def test_running_request_is_readmitted_as_new_request_data(scheduler, running_request):
    request = running_request
    result = commit_edit(
        scheduler, plan_edit(scheduler, request.request_id, DropOp(10, 20))
    )
    assert request.status == RequestStatus.WAITING
    assert request not in scheduler.running

    output = scheduler.schedule()
    (new_req,) = output.scheduled_new_reqs
    assert new_req.req_id == request.request_id
    assert new_req.num_computed_tokens == result.num_slots
    assert new_req.prefill_token_ids == list(request.all_token_ids)
    assert new_req.prompt_token_ids == request.prompt_token_ids
    assert new_req.block_ids == (_blocks(scheduler, request),)
    assert new_req.position_offset == 0
    assert output.num_scheduled_tokens[request.request_id] == 1
    assert request.status == RequestStatus.RUNNING


def test_drop_without_close_gap_leaves_a_position_gap(scheduler, running_request):
    request = running_request
    n = request.num_computed_tokens
    planned = plan_edit(scheduler, request.request_id, DropOp(10, 20, close_gap=False))
    assert planned.plans[0].rotate_slots.size == 0
    result = commit_edit(scheduler, planned)

    expected = np.concatenate([np.arange(10), np.arange(20, n)]).astype(np.float64)
    np.testing.assert_array_equal(request.kv_positions, expected)
    assert request.position_offset == 10
    assert result.next_position == n
    output = scheduler.schedule()
    assert output.scheduled_new_reqs[0].position_offset == 10


def test_edits_compose_and_positions_continue_across_decode(scheduler, running_request):
    request = running_request
    n = request.num_computed_tokens
    commit_edit(scheduler, plan_edit(scheduler, request.request_id, ShiftOp(0, n, 3)))
    assert request.position_offset == 3
    # Decode a few more tokens; the runner gives them positions slot + 3.
    for i in range(3):
        _step(scheduler, request, 2000 + i)
    assert request.num_computed_tokens == n + 3

    planned = plan_edit(scheduler, request.request_id, ShiftOp(0, 5, 0.5))
    np.testing.assert_array_equal(
        planned.new_positions,
        np.concatenate([np.arange(5) + 3.5, np.arange(5, n + 3) + 3.0]),
    )
    commit_edit(scheduler, planned)
    assert request.position_offset == 3


def test_fractional_next_position_is_rejected(scheduler, running_request):
    request = running_request
    n = request.num_computed_tokens
    with pytest.raises(ValueError, match="not an integer"):
        plan_edit(scheduler, request.request_id, ShiftOp(n - 4, n, 0.5))


def test_span_and_request_validation(scheduler, running_request):
    request = running_request
    n = request.num_computed_tokens
    with pytest.raises(KeyError):
        plan_edit(scheduler, "nope", DropOp(0, 1))
    with pytest.raises(ValueError, match="span"):
        plan_edit(scheduler, request.request_id, DropOp(0, n + 1))
    with pytest.raises(ValueError, match="span"):
        plan_edit(scheduler, request.request_id, ShiftOp(5, 4, 1.0))
    with pytest.raises(ValueError, match="max_model_len"):
        plan_edit(scheduler, request.request_id, ShiftOp(0, n, scheduler.max_model_len))
    with pytest.raises(ValueError, match="negative"):
        plan_edit(scheduler, request.request_id, ShiftOp(0, n, -1.0))


def test_drop_straddling_prompt_boundary_keeps_classification(
    scheduler, running_request
):
    request = running_request
    n = request.num_computed_tokens
    outputs_before = list(request.output_token_ids)
    commit_edit(
        scheduler,
        plan_edit(scheduler, request.request_id, DropOp(PROMPT_LEN - 3, n)),
    )
    assert request.prompt_token_ids == list(range(PROMPT_LEN - 3))
    assert request.num_prompt_tokens == PROMPT_LEN - 3
    # Only the pending (uncomputed) sampled token survives as output.
    assert list(request.output_token_ids) == outputs_before[-1:]
    assert request.num_computed_tokens == PROMPT_LEN - 3


def test_preemption_forgets_the_edited_layout(scheduler, running_request):
    request = running_request
    n = request.num_computed_tokens
    commit_edit(scheduler, plan_edit(scheduler, request.request_id, ShiftOp(0, n, 7)))
    scheduler.schedule()  # re-admitted, RUNNING again
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.time())
    assert request.kv_positions is None
    assert request.position_offset == 0


def test_ops_survive_the_utility_rpc_encoding():
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()
    for op in (DropOp(3, 9, close_gap=False), ShiftOp(0, 4, -2.5)):
        wire = decoder.decode(encoder.encode(op))
        assert isinstance(wire, dict)
        assert as_edit_op(wire) == op
        assert as_edit_op(op) is op
    with pytest.raises(msgspec.ValidationError):
        as_edit_op({"type": "explode"})
    assert msgspec.convert({"type": "drop", "start": 1, "end": 2}, KVEditOp) == (
        DropOp(1, 2)
    )
