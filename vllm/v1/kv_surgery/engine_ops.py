# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side half of ``edit_kv``: plan an op, then commit it.

``plan_edit`` is pure: it validates the request, builds one ``View`` per KV
cache group from the scheduler's block tables, applies the op and returns the
resulting cache mutations. The engine core runs those on every worker, then
``commit_edit`` makes the scheduler agree: tail blocks are freed, the token
history shrinks, the request remembers its per-slot positions, and a running
request is re-queued so the worker rebuilds its state from the scheduler's
truth on the next step (the same path a streaming update takes).
"""

from dataclasses import dataclass

import numpy as np

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_surgery.ops import DropOp, KVEditOp, KVEditResult, ShiftOp
from vllm.v1.kv_surgery.view import EditPlan, View, continue_positions, drop, shift
from vllm.v1.request import Request, RequestStatus

_EDITABLE_STATUSES = frozenset(
    {
        RequestStatus.RUNNING,
        RequestStatus.WAITING,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
    }
)


@dataclass
class PlannedEdit:
    request: Request
    op: KVEditOp
    num_slots_before: int
    new_positions: np.ndarray
    position_offset: int
    plans: list[EditPlan]
    """One per KV cache group, indexed like ``kv_cache_config.kv_cache_groups``."""
    new_num_blocks: list[int]

    @property
    def has_cache_work(self) -> bool:
        return any(
            plan.gather_src.size or plan.rotate_slots.size for plan in self.plans
        )


def _check_supported(scheduler: Scheduler, request: Request) -> None:
    config = scheduler.vllm_config
    if not scheduler.use_v2_model_runner:
        raise NotImplementedError("KV surgery requires the V2 model runner")
    if config.cache_config.enable_prefix_caching:
        raise NotImplementedError("KV surgery requires prefix caching to be off")
    if config.scheduler_config.async_scheduling:
        raise NotImplementedError("KV surgery does not support async scheduling")
    parallel = config.parallel_config
    if parallel.pipeline_parallel_size > 1:
        raise NotImplementedError("KV surgery does not support pipeline parallelism")
    if parallel.data_parallel_size > 1:
        raise NotImplementedError("KV surgery does not support data parallelism")
    if (
        parallel.prefill_context_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
    ):
        # Slot ids in an EditPlan are global; CP shards slots across ranks and
        # the runner's PCP path ignores the position offset.
        raise NotImplementedError("KV surgery does not support context parallelism")
    if config.speculative_config is not None:
        raise NotImplementedError("KV surgery does not support speculative decoding")
    if scheduler.connector is not None:
        raise NotImplementedError("KV surgery does not support KV connectors")
    if request.mm_features or request.prompt_embeds is not None:
        raise NotImplementedError(
            "KV surgery does not support multimodal or embedding prompts"
        )
    if request.spec_token_ids or request.num_in_flight_tokens:
        raise RuntimeError(f"{request.request_id}: request has in-flight tokens")


def _request_views(
    scheduler: Scheduler, request: Request, positions: np.ndarray
) -> list[View]:
    groups = scheduler.kv_cache_config.kv_cache_groups
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    if not groups:
        raise NotImplementedError("model has no KV cache to edit")
    views = []
    for group, manager in zip(groups, managers):
        blocks = manager.req_to_blocks.get(request.request_id, [])
        for block in blocks:
            if block.is_null:
                raise NotImplementedError(
                    f"{request.request_id}: KV cache group with skipped (null) "
                    "blocks is not supported yet"
                )
            if block.ref_cnt != 1:
                raise NotImplementedError(
                    f"{request.request_id}: block {block.block_id} is shared with "
                    "another request; editing shared blocks is not supported yet"
                )
        views.append(
            View(
                group.kv_cache_spec.block_size,
                tuple(block.block_id for block in blocks),
                positions,
            )
        )
    return views


def plan_edit(scheduler: Scheduler, request_id: str, op: KVEditOp) -> PlannedEdit:
    """Validate ``op`` against the request and compute the cache mutations.

    Nothing is mutated. The op addresses the request's computed KV slots
    ``[0, num_computed_tokens)``; tokens past that (the pending sampled token,
    or an unfinished prefill) are untouched.

    Raises:
        KeyError: unknown request id.
        ValueError: the request has no editable cache, the span is out of
            range, or the result has an invalid position layout.
        NotImplementedError: a feature the surgery path does not support.
    """
    request = scheduler.requests.get(request_id)
    if request is None:
        raise KeyError(f"unknown request {request_id!r}")
    if request.is_finished() or request.status not in _EDITABLE_STATUSES:
        raise ValueError(
            f"{request_id}: request in state {request.status} has no editable KV"
        )
    _check_supported(scheduler, request)

    num_slots = request.num_computed_tokens
    positions = continue_positions(request.kv_positions, num_slots)
    views = _request_views(scheduler, request, positions)

    plans = []
    new_views = []
    for view in views:
        if isinstance(op, DropOp):
            new_view, plan = drop(view, op.start, op.end, op.close_gap)
        elif isinstance(op, ShiftOp):
            new_view, plan = shift(view, op.start, op.end, op.delta)
        else:
            raise TypeError(f"unknown edit op {op!r}")
        new_views.append(new_view)
        plans.append(plan)

    new_positions = new_views[0].positions
    next_position = new_views[0].next_position
    max_position = scheduler.max_model_len
    if new_positions.size and new_positions.min() < 0:
        raise ValueError(f"{request_id}: edit would make a position negative")
    if not float(next_position).is_integer():
        raise ValueError(
            f"{request_id}: next position {next_position} is not an integer; "
            "newly generated tokens need integer positions"
        )
    if next_position >= max_position or (
        new_positions.size and new_positions.max() >= max_position
    ):
        raise ValueError(
            f"{request_id}: edit would place a position at or beyond "
            f"max_model_len {max_position}"
        )

    return PlannedEdit(
        request=request,
        op=op,
        num_slots_before=num_slots,
        new_positions=new_positions,
        position_offset=int(next_position) - new_views[0].num_slots,
        plans=plans,
        new_num_blocks=[len(view.block_ids) for view in new_views],
    )


def _drop_tokens(request: Request, start: int, end: int) -> None:
    """Remove tokens ``[start, end)`` from the request's history.

    Tokens keep their prompt/output classification; only surviving tokens
    remain in either list. ``_all_token_ids`` and ``_output_token_ids`` are
    edited in place because the request's read-only views wrap them;
    ``prompt_token_ids`` is rebound because in-process engines share that
    list with the caller and the output processor.
    """
    assert request.prompt_token_ids is not None
    old_prompt_len = request.num_prompt_tokens
    new_all = request._all_token_ids[:start] + request._all_token_ids[end:]
    new_prompt_len = old_prompt_len - (
        min(end, old_prompt_len) - min(start, old_prompt_len)
    )
    request._all_token_ids[:] = new_all
    request.prompt_token_ids = new_all[:new_prompt_len]
    request._output_token_ids[:] = new_all[new_prompt_len:]
    request.num_prompt_tokens = new_prompt_len


def commit_edit(scheduler: Scheduler, planned: PlannedEdit) -> KVEditResult:
    """Make the scheduler agree with an edit whose cache work is done."""
    request = planned.request
    request_id = request.request_id
    if request.num_computed_tokens != planned.num_slots_before:
        raise RuntimeError(
            f"{request_id}: request changed between planning and committing"
        )

    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    num_freed = 0
    for manager, keep in zip(managers, planned.new_num_blocks):
        blocks = manager.req_to_blocks.get(request_id)
        if not blocks:
            continue
        freed = blocks[keep:]
        del blocks[keep:]
        if freed:
            manager.block_pool.free_blocks(reversed(freed))
            num_freed += len(freed)

    if isinstance(planned.op, DropOp):
        _drop_tokens(request, planned.op.start, planned.op.end)
    request.num_computed_tokens = len(planned.new_positions)
    request.kv_positions = planned.new_positions
    request.position_offset = planned.position_offset

    if request.status == RequestStatus.RUNNING:
        # Re-admit through the waiting queue: the next schedule() emits the
        # request as NewRequestData with its full token list, block ids and
        # position offset, and the worker rebuilds its state from that.
        scheduler.running.remove(request)
        scheduler._inflight_prefills.discard(request)
        request.status = RequestStatus.WAITING
        scheduler.waiting.prepend_request(request)

    return KVEditResult(
        request_id=request_id,
        num_slots=request.num_computed_tokens,
        num_tokens=request.num_tokens,
        next_position=request.num_computed_tokens + request.position_offset,
        num_freed_blocks=num_freed,
    )
