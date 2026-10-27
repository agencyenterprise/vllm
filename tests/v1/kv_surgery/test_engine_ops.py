# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side edit_kv / fork_kv: planning, committing, re-admission,
block sharing. No model forward; a GPU is only needed to build VllmConfig."""

import time
from collections import deque
from dataclasses import fields, replace

import msgspec
import msgspec.msgpack
import numpy as np
import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import FinishReason, utility_failure_prefix
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import restore_utility_error
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_surgery.engine_ops import (
    commit_edit,
    commit_fork,
    inspect_kv,
    plan_edit,
    plan_fork,
)
from vllm.v1.kv_surgery.ops import (
    DropOp,
    KVEditLostError,
    KVEditOp,
    ShiftOp,
    SpliceOp,
    StaleSpanError,
    as_edit_op,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from ..core.utils import create_requests, create_scheduler

BLOCK_SIZE = 16
PROMPT_LEN = 40
NUM_DECODES = 6
MODEL = "meta-llama/Llama-3.2-1B"


def _step_many(scheduler: Scheduler, sampled: dict[str, int]) -> None:
    """One scheduler step in which every request in ``sampled`` is scheduled
    and samples the given token."""
    output = scheduler.schedule()
    assert set(sampled) <= set(output.num_scheduled_tokens)
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[sampled.get(req_id, 0)] for req_id in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _step(scheduler: Scheduler, request: Request, token_id: int) -> None:
    _step_many(scheduler, {request.request_id: token_id})


def _new_request(tokens: list[int], req_id: str, max_tokens: int = 8) -> Request:
    """A request as the engine builds it with prefix caching off: no block
    hasher and no hashes (create_requests attaches one regardless)."""
    (request,) = create_requests(
        1, num_tokens=len(tokens), max_tokens=max_tokens, block_size=BLOCK_SIZE
    )
    request.request_id = req_id
    request.prompt_token_ids[:] = tokens
    request._all_token_ids[:] = tokens
    request._block_hasher = None
    request.block_hashes = []
    return request


def _fork(
    scheduler: Scheduler,
    source: Request,
    new_tokens: list[int],
    num_slots: int | None,
    req_id: str = "fork",
) -> Request:
    """What EngineCore.fork_kv does, minus the worker pass."""
    request = _new_request(new_tokens, req_id)
    planned = plan_fork(scheduler, source.request_id, request, num_slots)
    commit_fork(scheduler, planned)
    scheduler.add_request(request)
    return request


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
    request = _new_request(list(range(PROMPT_LEN)), "0", max_tokens=64)
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


def test_inspect_token_ids_follow_the_slots(scheduler, running_request):
    """The token readout covers the computed slots only (not the sampled
    token still waiting for its forward pass) and tracks a drop."""
    request = running_request
    rid = request.request_id
    assert inspect_kv(scheduler, rid).token_ids is None
    n = request.num_computed_tokens
    tokens = list(request.all_token_ids)
    assert len(tokens) == n + 1
    assert inspect_kv(scheduler, rid, token_ids=True).token_ids == tokens[:n]
    commit_edit(scheduler, plan_edit(scheduler, rid, DropOp(10, 20)))
    expected = tokens[:10] + tokens[20:n]
    assert inspect_kv(scheduler, rid, token_ids=True).token_ids == expected


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


def test_stale_expected_num_slots_is_rejected(scheduler, running_request):
    """A span computed before the request decoded again is refused before
    any block moves; the matching count goes through."""
    request = running_request
    n = request.num_computed_tokens
    blocks_before = _blocks(scheduler, request)
    with pytest.raises(StaleSpanError, match=f"expected {n - 1} computed slots"):
        plan_edit(
            scheduler,
            request.request_id,
            DropOp(n - 3, n - 1, expected_num_slots=n - 1),
        )
    with pytest.raises(StaleSpanError, match=f"expected {n - 1} computed slots"):
        plan_edit(
            scheduler,
            request.request_id,
            SpliceOp(n - 3, n, request.request_id, 0, 1, expected_src_num_slots=n - 1),
        )
    assert _blocks(scheduler, request) == blocks_before
    planned = plan_edit(
        scheduler, request.request_id, DropOp(n - 3, n, expected_num_slots=n)
    )
    assert commit_edit(scheduler, planned).num_slots == n - 3


@pytest.mark.parametrize(
    ("failure_type", "expected"),
    [
        ("StaleSpanError", StaleSpanError),
        ("KVEditLostError", KVEditLostError),
        ("KeyError", KeyError),
        ("NotImplementedError", NotImplementedError),
        ("SomeInternalError", Exception),
        (None, Exception),
    ],
)
def test_utility_failures_keep_a_known_type_across_the_rpc(failure_type, expected):
    """The core reports a failed utility call as message plus exception class
    name; the client re-raises the known surgery-facing types (so a caller
    can catch a stale span or a lost request with either engine core) and
    anything else as the plain Exception it always was."""
    message = utility_failure_prefix("edit_kv") + "stale span: r: expected 1"
    error = restore_utility_error(message, failure_type)
    assert type(error) is expected
    assert message in str(error)


def test_sliding_window_allocation_is_rejected(scheduler, running_request, monkeypatch):
    """A window group evicts rows (null blocks), so a view over it is
    incomplete; the hybrid manager must be disabled for such models."""
    groups = scheduler.kv_cache_config.kv_cache_groups
    spec = groups[0].kv_cache_spec
    assert isinstance(spec, FullAttentionSpec)
    window_spec = SlidingWindowSpec(
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=spec.dtype,
        sliding_window=BLOCK_SIZE,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_config,
        "kv_cache_groups",
        [KVCacheGroupSpec(groups[0].layer_names, window_spec), *groups[1:]],
    )
    with pytest.raises(NotImplementedError, match="disable_hybrid_kv_cache_manager"):
        plan_edit(scheduler, running_request.request_id, DropOp(0, 1))


def test_unreviewed_full_attention_subclass_is_rejected(
    scheduler, running_request, monkeypatch
):
    """Only the exact full-attention and MLA spec types are allowed; a
    subclass nobody has looked at for surgery (attention sinks) is not."""
    groups = scheduler.kv_cache_config.kv_cache_groups
    spec = groups[0].kv_cache_spec
    sink = SinkFullAttentionSpec(
        **{f.name: getattr(spec, f.name) for f in fields(spec)}, sink_len=4
    )
    monkeypatch.setattr(
        scheduler.kv_cache_config,
        "kv_cache_groups",
        [KVCacheGroupSpec(groups[0].layer_names, sink), *groups[1:]],
    )
    with pytest.raises(NotImplementedError, match="SinkFullAttentionSpec"):
        plan_edit(scheduler, running_request.request_id, DropOp(0, 1))


def test_chunked_local_attention_is_rejected(scheduler, running_request, monkeypatch):
    """Chunk boundaries under a drop or shift are untested (Llama 4)."""
    groups = scheduler.kv_cache_config.kv_cache_groups
    chunked = replace(groups[0].kv_cache_spec, attention_chunk_size=BLOCK_SIZE)
    monkeypatch.setattr(
        scheduler.kv_cache_config,
        "kv_cache_groups",
        [KVCacheGroupSpec(groups[0].layer_names, chunked), *groups[1:]],
    )
    with pytest.raises(NotImplementedError, match="chunked local"):
        plan_edit(scheduler, running_request.request_id, DropOp(0, 1))


def test_compressed_kv_states_are_rejected(scheduler, running_request, monkeypatch):
    """DeepSeek V4 compressed layers keep one state per token group."""
    groups = scheduler.kv_cache_config.kv_cache_groups
    compressed = replace(groups[0].kv_cache_spec, tokens_per_state=4)
    monkeypatch.setattr(
        scheduler.kv_cache_config,
        "kv_cache_groups",
        [KVCacheGroupSpec(groups[0].layer_names, compressed), *groups[1:]],
    )
    with pytest.raises(NotImplementedError, match="tokens per state"):
        plan_edit(scheduler, running_request.request_id, DropOp(0, 1))


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
    with pytest.raises(ValueError, match="no computed KV"):
        plan_edit(scheduler, request.request_id, DropOp(0, n))


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


def _finished_outputs(scheduler: Scheduler) -> dict[str, FinishReason]:
    """One empty step; the finish reasons it reports, by request id."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    outs = scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[0] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return {
        out.request_id: out.finish_reason
        for eco in outs.values()
        for out in eco.outputs
        if out.finish_reason is not None
    }


def test_preempting_an_edited_request_finishes_it_as_kv_edit_lost(
    scheduler, running_request
):
    """Recomputing an edited request from its tokens would be a clean prefill
    that never saw the edit: valid-looking, wrong data. The scheduler finishes
    it instead, frees everything, and reports the reason at the next step."""
    request = running_request
    n = request.num_computed_tokens
    commit_edit(scheduler, plan_edit(scheduler, request.request_id, ShiftOp(0, n, 7)))
    scheduler.schedule()  # re-admitted, RUNNING again
    held = len(_blocks(scheduler, request))
    free_before = _free_blocks(scheduler)

    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.time())
    assert request.status == RequestStatus.FINISHED_KV_EDIT_LOST
    assert request.request_id not in scheduler.requests
    assert request.request_id in scheduler.finished_req_ids
    assert not scheduler.waiting and not scheduler.running
    assert _free_blocks(scheduler) == free_before + held
    with pytest.raises(KeyError, match="unknown request"):
        inspect_kv(scheduler, request.request_id)

    assert _finished_outputs(scheduler) == {
        request.request_id: FinishReason.KV_EDIT_LOST
    }
    assert _finished_outputs(scheduler) == {}  # reported once


def test_preemption_of_an_unedited_request_is_stock(scheduler, running_request):
    request = running_request
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.time())
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert scheduler.waiting.peek_request() is request
    assert _finished_outputs(scheduler) == {}


def test_preempting_a_fork_of_an_edited_source_loses_it_too(scheduler, running_request):
    """A fork inherits the source's edited layout (its shared rows carry
    edited keys), so preempting it is a lost edit; the source keeps its blocks
    and merely loses the share."""
    source = running_request
    n = source.num_computed_tokens
    commit_edit(scheduler, plan_edit(scheduler, source.request_id, ShiftOp(0, n, 5)))
    _step(scheduler, source, 3000)
    fork = _fork(scheduler, source, [1, 2], num_slots=None)
    assert fork.has_edited_kv_layout
    _step_many(scheduler, {source.request_id: 3001, "fork": 3002})
    assert inspect_kv(scheduler, source.request_id).num_shared_blocks == [2]
    src_blocks = _blocks(scheduler, source)

    scheduler.running.remove(fork)
    scheduler._preempt_request(fork, time.time())
    assert fork.status == RequestStatus.FINISHED_KV_EDIT_LOST
    assert inspect_kv(scheduler, source.request_id).num_shared_blocks == [0]
    assert _blocks(scheduler, source) == src_blocks
    assert source.status == RequestStatus.RUNNING


def test_preempting_a_fork_of_a_stock_source_recomputes_it(scheduler, running_request):
    """A fork of an unedited request is stock itself: preemption re-prefills
    its prefix-extended prompt, which reproduces the shared cache."""
    source = running_request
    n = source.num_computed_tokens
    fork = _fork(scheduler, source, [1, 2], num_slots=None)
    assert not fork.has_edited_kv_layout
    _step_many(scheduler, {source.request_id: 3001, "fork": 3002})

    scheduler.running.remove(fork)
    scheduler._preempt_request(fork, time.time())
    assert fork.status == RequestStatus.PREEMPTED
    assert fork.num_computed_tokens == 0
    assert inspect_kv(scheduler, source.request_id).num_shared_blocks == [0]
    assert _blocks(scheduler, fork) == []
    assert list(fork.prompt_token_ids)[: n + 1] == list(source.all_token_ids)[: n + 1]


def test_scheduler_preempts_the_edited_request_but_never_a_paused_one(
    scheduler, running_request
):
    """With the pool exhausted the scheduler preempts running requests, the
    most recently admitted first; a paused resumable fork is not running and
    keeps every block. The edited victim is reported as kv_edit_lost."""
    source = running_request
    fork = _fork(scheduler, source, [1], num_slots=None)
    fork.resumable = True
    fork.max_tokens = 1
    _step_many(scheduler, {source.request_id: 3001, "fork": 3002})
    assert fork.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    fork_blocks = _blocks(scheduler, fork)

    # A long request takes most of the pool; the edit re-admits the source
    # after it, so the source is the first preemption victim.
    free = _free_blocks(scheduler)
    long = _new_request(list(range(10_000, 10_000 + (free - 2) * BLOCK_SIZE)), "long")
    long.max_tokens = 10_000
    scheduler.add_request(long)
    n = source.num_computed_tokens
    commit_edit(scheduler, plan_edit(scheduler, source.request_id, ShiftOp(0, n, 3)))

    finished: dict[str, FinishReason] = {}
    for _ in range(4 * BLOCK_SIZE):
        finished.update(_finished_outputs(scheduler))
        if finished:
            break
    assert finished == {source.request_id: FinishReason.KV_EDIT_LOST}
    assert long.status == RequestStatus.RUNNING
    assert fork.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert _blocks(scheduler, fork) == fork_blocks
    assert inspect_kv(scheduler, "fork").num_slots == n + 1


def test_ops_survive_the_utility_rpc_encoding():
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()
    for op in (
        DropOp(3, 9, close_gap=False),
        ShiftOp(0, 4, -2.5),
        SpliceOp(1, 2, "other", 3, 5, close_gap=False),
    ):
        wire = decoder.decode(encoder.encode(op))
        assert isinstance(wire, dict)
        assert as_edit_op(wire) == op
        assert as_edit_op(op) is op
    with pytest.raises(msgspec.ValidationError):
        as_edit_op({"type": "explode"})
    assert msgspec.convert({"type": "drop", "start": 1, "end": 2}, KVEditOp) == (
        DropOp(1, 2)
    )


def _free_blocks(scheduler: Scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


def test_fork_shares_full_blocks_and_copies_the_partial_one(scheduler, running_request):
    source = running_request
    n = source.num_computed_tokens  # 46 slots: two full blocks + 14 in the third
    src_blocks = _blocks(scheduler, source)
    free_before = _free_blocks(scheduler)

    request = _new_request([7, 8, 9], "fork")
    planned = plan_fork(scheduler, source.request_id, request, None)
    assert planned.num_slots == n
    assert len(planned.shared_blocks[0]) == 2
    assert len(planned.fresh_blocks[0]) == 1
    assert planned.plans[0].gather_src.size == n - 2 * BLOCK_SIZE
    assert planned.plans[0].rotate_slots.size == 0
    assert planned.position_offset == 0
    assert _free_blocks(scheduler) == free_before - 1

    commit_fork(scheduler, planned)
    scheduler.add_request(request)
    assert request.num_computed_tokens == n
    assert list(request.all_token_ids) == list(source.all_token_ids)[:n] + [7, 8, 9]
    assert request.num_prompt_tokens == n + 3
    assert request.kv_positions is None  # stock layout inherited
    fork_blocks = _blocks(scheduler, request)
    assert fork_blocks[:2] == src_blocks[:2]
    assert fork_blocks[2] not in src_blocks
    coordinator = scheduler.kv_cache_manager.coordinator
    for block in coordinator.single_type_managers[0].req_to_blocks[source.request_id]:
        assert block.ref_cnt == (2 if block.block_id in fork_blocks else 1)
    assert inspect_kv(scheduler, source.request_id).num_shared_blocks == [2]

    # The fork prefills its three tokens on top of the shared prefix while the
    # source keeps decoding; 49 slots need a fourth block.
    output = scheduler.schedule()
    (new_req,) = output.scheduled_new_reqs
    assert new_req.req_id == "fork"
    assert new_req.num_computed_tokens == n
    assert output.num_scheduled_tokens["fork"] == 3
    assert output.num_scheduled_tokens[source.request_id] == 1
    assert len(new_req.block_ids[0]) == 4
    assert new_req.block_ids[0][:2] == src_blocks[:2]

    # Freeing the fork drops the shared references, not the source's blocks.
    scheduler.finish_requests("fork", RequestStatus.FINISHED_ABORTED)
    assert inspect_kv(scheduler, source.request_id).num_shared_blocks == [0]
    assert _blocks(scheduler, source) == src_blocks
    assert _free_blocks(scheduler) == free_before


def test_fork_at_a_prefix_inherits_the_edited_layout(scheduler, running_request):
    source = running_request
    n = source.num_computed_tokens
    commit_edit(scheduler, plan_edit(scheduler, source.request_id, ShiftOp(0, n, 5)))
    _step(scheduler, source, 3000)  # re-admitted and decoding again
    n += 1

    fork = _fork(scheduler, source, [1, 2], num_slots=20)
    assert inspect_kv(scheduler, "fork").positions is None
    info = inspect_kv(scheduler, "fork", positions=True)
    assert info.status == "WAITING"
    assert info.num_slots == 20
    assert info.num_tokens == 22
    assert info.num_prompt_tokens == 22
    assert info.positions == [float(p) for p in range(5, 25)]
    assert info.next_position == 25
    assert info.num_shared_blocks == [1]  # slots [0, 16); [16, 20) was copied
    np.testing.assert_array_equal(fork.kv_positions, np.arange(20) + 5)
    assert fork.position_offset == 5

    with pytest.raises(ValueError, match="cannot fork"):
        _fork(scheduler, source, [1], num_slots=n + 1, req_id="too-long")
    with pytest.raises(ValueError, match="cannot fork"):
        _fork(scheduler, source, [1], num_slots=0, req_id="empty")
    with pytest.raises(ValueError, match="already in use"):
        _fork(scheduler, source, [1], num_slots=4, req_id="fork")
    with pytest.raises(KeyError):
        plan_fork(scheduler, "nope", fork, None)
    # The first sampled token must still fit under max_model_len (positions
    # here run 5 ahead of the slot count because of the shift above).
    room = scheduler.max_model_len - (
        source.num_computed_tokens + source.position_offset
    )
    with pytest.raises(ValueError, match="max_model_len"):
        _fork(scheduler, source, list(range(room)), num_slots=None, req_id="edge")
    _fork(scheduler, source, list(range(room - 1)), num_slots=None, req_id="fits")


def test_shared_blocks_are_read_only(scheduler, running_request):
    source = running_request
    n = source.num_computed_tokens
    _fork(scheduler, source, [1], num_slots=None)  # shares blocks 0 and 1

    with pytest.raises(NotImplementedError, match="shared"):
        plan_edit(scheduler, source.request_id, DropOp(5, 10))
    with pytest.raises(NotImplementedError, match="shared"):
        plan_edit(scheduler, "fork", ShiftOp(0, 4, 1.0))
    # A tail drop writes nothing, but must not leave the source decoding into
    # a block the fork still reads (slot 20 would land in shared block 1).
    with pytest.raises(NotImplementedError, match="decoding into"):
        plan_edit(scheduler, source.request_id, DropOp(20, n))
    # Slots in the source's own partial block are fair game.
    planned = plan_edit(scheduler, source.request_id, ShiftOp(2 * BLOCK_SIZE, n, 2.0))
    assert planned.plans[0].rotate_slots.size == n - 2 * BLOCK_SIZE
    # Dropping a tail that spans a shared block only releases the reference.
    free_before = _free_blocks(scheduler)
    result = commit_edit(
        scheduler, plan_edit(scheduler, source.request_id, DropOp(BLOCK_SIZE, n))
    )
    # Block 1 leaves the table but is still the fork's; only block 2 is freed.
    assert result.num_freed_blocks == 1
    assert _free_blocks(scheduler) == free_before + 1
    assert inspect_kv(scheduler, "fork").num_shared_blocks == [1]


def test_splice_from_a_fork_replaces_a_span_and_grows_the_view(
    scheduler, running_request
):
    source = running_request
    n = source.num_computed_tokens
    fork = _fork(scheduler, source, [7, 8, 9, 10, 11], num_slots=BLOCK_SIZE)
    _step_many(scheduler, {source.request_id: 3000, "fork": 3001})
    assert fork.num_computed_tokens == BLOCK_SIZE + 5
    n += 1
    tokens_before = list(source.all_token_ids)
    free_before = _free_blocks(scheduler)

    # Replace source slots [16, 18) with the fork's five new slots: 49 slots
    # need a fourth block, and the tail is re-rotated to follow the insert.
    op = SpliceOp(BLOCK_SIZE, BLOCK_SIZE + 2, "fork", BLOCK_SIZE, BLOCK_SIZE + 5)
    planned = plan_edit(scheduler, source.request_id, op)
    plan = planned.plans[0]
    assert len(planned.allocated_blocks[0]) == 1
    assert planned.new_tokens == [7, 8, 9, 10, 11]
    assert plan.gather_src.size == 5 + (n - BLOCK_SIZE - 2)
    np.testing.assert_array_equal(plan.gather_src[:5], _fork_slots(scheduler, fork))
    assert plan.rotate_slots.size == n - BLOCK_SIZE - 2
    np.testing.assert_array_equal(plan.rotate_deltas, 3.0)
    assert _free_blocks(scheduler) == free_before - 1

    result = commit_edit(scheduler, planned)
    assert result.num_slots == n + 3
    assert result.next_position == n + 3
    np.testing.assert_array_equal(source.kv_positions, np.arange(n + 3))
    assert list(source.all_token_ids) == (
        tokens_before[:BLOCK_SIZE] + [7, 8, 9, 10, 11] + tokens_before[BLOCK_SIZE + 2 :]
    )
    # Inserted inside the prompt, so the inserted tokens are prompt tokens.
    assert source.num_prompt_tokens == PROMPT_LEN - 2 + 5
    assert len(_blocks(scheduler, source)) == 4
    assert source.status == RequestStatus.WAITING
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[source.request_id] == 1


def _fork_slots(scheduler: Scheduler, fork: Request) -> np.ndarray:
    blocks = _blocks(scheduler, fork)
    idx = np.arange(BLOCK_SIZE, fork.num_computed_tokens)
    return np.asarray(blocks)[idx // BLOCK_SIZE] * BLOCK_SIZE + idx % BLOCK_SIZE


def test_splice_can_open_a_gap_and_classifies_output_inserts(
    scheduler, running_request
):
    source = running_request
    n = source.num_computed_tokens
    _fork(scheduler, source, [7, 8], num_slots=BLOCK_SIZE)
    _step_many(scheduler, {source.request_id: 3000, "fork": 3001})
    n += 1
    # Insert after the prompt with the gap left open: the two copied slots keep
    # positions 16 and 17 and the tail keeps its own.
    op = SpliceOp(
        PROMPT_LEN, PROMPT_LEN, "fork", BLOCK_SIZE, BLOCK_SIZE + 2, close_gap=False
    )
    result = commit_edit(scheduler, plan_edit(scheduler, source.request_id, op))
    assert result.num_slots == n + 2
    expected = np.concatenate(
        [np.arange(PROMPT_LEN), [16, 17], np.arange(PROMPT_LEN, n)]
    )
    np.testing.assert_array_equal(source.kv_positions, expected)
    assert result.next_position == n
    assert source.position_offset == -2
    assert source.num_prompt_tokens == PROMPT_LEN
    assert list(source.output_token_ids)[:2] == [7, 8]


def test_splice_validation_releases_blocks_it_took(scheduler, running_request):
    source = running_request
    n = source.num_computed_tokens
    _fork(scheduler, source, [7], num_slots=BLOCK_SIZE)  # shares block 0
    _step_many(scheduler, {source.request_id: 3000, "fork": 3001})
    free_before = _free_blocks(scheduler)

    with pytest.raises(KeyError):
        plan_edit(scheduler, source.request_id, SpliceOp(0, 0, "nope", 0, 1))
    with pytest.raises(ValueError, match="span"):
        plan_edit(scheduler, source.request_id, SpliceOp(0, 0, "fork", 0, 100))
    # Copying the shared prefix onto itself writes nothing, so it is allowed.
    plan = plan_edit(scheduler, source.request_id, SpliceOp(0, 1, "fork", 0, 17))
    assert (
        plan.plans[0].gather_dst // BLOCK_SIZE != _blocks(scheduler, source)[0]
    ).all()
    assert _free_blocks(scheduler) == free_before - 1
    commit_edit(scheduler, plan)
    n += 16
    # Inserting *into* the shared block is not; the extra block is taken and
    # then given back.
    with pytest.raises(NotImplementedError, match="shared"):
        plan_edit(scheduler, source.request_id, SpliceOp(1, 1, "fork", 0, 17))
    assert _free_blocks(scheduler) == free_before - 1
    free_before -= 1
    # Slots, not just positions, are bounded by max_model_len.
    max_model_len = scheduler.max_model_len
    scheduler.max_model_len = 60
    with pytest.raises(ValueError, match="more than max_model_len"):
        plan_edit(scheduler, source.request_id, SpliceOp(n, n, "fork", 0, 17))
    scheduler.max_model_len = max_model_len
    assert _free_blocks(scheduler) == free_before
    pool = scheduler.kv_cache_manager.block_pool
    hoard = pool.get_new_blocks(free_before)
    with pytest.raises(ValueError, match="free KV cache blocks"):
        plan_edit(scheduler, source.request_id, SpliceOp(n, n, "fork", 0, 17))
    pool.free_blocks(reversed(hoard))
    assert _free_blocks(scheduler) == free_before


def test_failed_drain_is_fatal_at_the_next_step():
    """A model failure that a drain surfaces inside a utility call (which
    only reports errors to its caller) still reaches the engine's failure
    path: the next step re-raises it, and so does any later drain."""
    core = object.__new__(EngineCore)
    core.batch_queue = deque([object()])
    core._drained_outputs = deque()
    core._drain_error = None

    def fail():
        raise RuntimeError("model failed")

    core._finish_oldest_batch = fail  # type: ignore[method-assign]
    drain, step = core._drain_batch_queue, core.step_with_batch_queue
    for call in (drain, step, drain):
        with pytest.raises(RuntimeError, match="model failed"):
            call()
