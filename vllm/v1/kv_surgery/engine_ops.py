# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side half of ``edit_kv`` and ``fork_kv``: plan, then commit.

``plan_edit`` / ``plan_fork`` validate the request(s), build one ``View`` per
KV cache group from the scheduler's block tables, apply the op and return the
resulting cache mutations. Planning takes new blocks from the block pool when
a view grows but touches nothing else; ``release`` gives them back if the
worker pass fails. The engine core runs the plans on every worker, then
``commit_edit`` / ``commit_fork`` make the scheduler agree: block tables and
token histories change, the request remembers its per-slot positions, and a
running edited request is re-queued so the worker rebuilds its state from
the scheduler's truth on the next step (the same path a streaming update
takes).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sampling_params import SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.kv_surgery.ops import (
    DropOp,
    KVEditOp,
    KVEditResult,
    KVViewInfo,
    ShiftOp,
    SpliceOp,
)
from vllm.v1.kv_surgery.view import (
    EditPlan,
    View,
    continue_positions,
    drop,
    fork,
    shift,
    splice,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output.request import StructuredOutputRequest

Managers = list[tuple[KVCacheGroupSpec, SingleTypeKVCacheManager]]
"""One ``(group spec, manager)`` pair per KV cache group, in group order."""

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
    allocated_blocks: list[list[KVCacheBlock]]
    """Fresh blocks (per group) the grown view will use; owned until commit."""
    new_tokens: list[int] = field(default_factory=list)
    """Tokens that take the place of the edited span (splice only)."""

    @property
    def has_cache_work(self) -> bool:
        return _needs_worker_pass(self.plans)


@dataclass
class PlannedFork:
    source: Request
    request: Request
    num_slots: int
    positions: np.ndarray
    position_offset: int
    plans: list[EditPlan]
    shared_blocks: list[list[KVCacheBlock]]
    fresh_blocks: list[list[KVCacheBlock]]

    @property
    def has_cache_work(self) -> bool:
        return _needs_worker_pass(self.plans)


def _needs_worker_pass(plans: list[EditPlan]) -> bool:
    """Whether the workers must run anything (freed blocks are scheduler-only)."""
    return any(plan.gather_src.size or plan.rotate_slots.size for plan in plans)


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
    reject_unsupported_inputs(
        request.mm_features, request.prompt_embeds, request.sampling_params
    )
    if request.pooling_params is not None:
        raise NotImplementedError("KV surgery does not support pooling requests")
    if request.spec_token_ids or request.num_in_flight_tokens:
        raise RuntimeError(f"{request.request_id}: request has in-flight tokens")


def reject_unsupported_inputs(
    mm_features: Sequence[MultiModalFeatureSpec] | None,
    prompt_embeds: torch.Tensor | None,
    sampling_params: SamplingParams | None,
) -> None:
    """The request kinds surgery never accepts. Shared with the engine core,
    which must reject them before ``preprocess_add_request`` runs (grammar
    compilation and the multimodal cache are its only side effects)."""
    if mm_features or prompt_embeds is not None:
        raise NotImplementedError(
            "KV surgery does not support multimodal or embedding prompts"
        )
    # Same decision Request.use_structured_output is built from.
    if StructuredOutputRequest.from_sampling_params(sampling_params) is not None:
        # The grammar tracks the token history, which edits rewrite.
        raise NotImplementedError("KV surgery does not support structured outputs")


def _editable_request(scheduler: Scheduler, request_id: str) -> Request:
    request = scheduler.requests.get(request_id)
    if request is None:
        raise KeyError(f"unknown request {request_id!r}")
    if request.is_finished() or request.status not in _EDITABLE_STATUSES:
        raise ValueError(
            f"{request_id}: request in state {request.status} has no editable KV"
        )
    _check_supported(scheduler, request)
    return request


def _managers(scheduler: Scheduler) -> Managers:
    groups = scheduler.kv_cache_config.kv_cache_groups
    if not groups:
        raise NotImplementedError("model has no KV cache to edit")
    return list(
        zip(groups, scheduler.kv_cache_manager.coordinator.single_type_managers)
    )


def _request_blocks(
    manager: SingleTypeKVCacheManager, request: Request
) -> list[KVCacheBlock]:
    blocks = manager.req_to_blocks.get(request.request_id, [])
    for block in blocks:
        if block.is_null:
            raise NotImplementedError(
                f"{request.request_id}: KV cache group with skipped (null) "
                "blocks is not supported yet"
            )
    return blocks


def _request_views(
    managers: Managers,
    blocks_per_group: list[list[KVCacheBlock]],
    positions: np.ndarray,
) -> list[View]:
    """One view per KV cache group. Blocks may be shared (read-only) here;
    ``_check_writable`` guards the ones a plan writes."""
    return [
        View(
            group.kv_cache_spec.block_size,
            tuple(block.block_id for block in blocks),
            positions,
        )
        for (group, _), blocks in zip(managers, blocks_per_group, strict=True)
    ]


def _check_writable(
    request: Request,
    manager: SingleTypeKVCacheManager,
    blocks: list[KVCacheBlock],
    plan: EditPlan,
    block_size: int,
    num_slots_after: int,
) -> None:
    """Reject a plan that writes into, or leaves decode appending into, a
    block the request does not solely own.

    Forks share fully occupied blocks by reference and there is no
    copy-on-write, so shared blocks are read-only for both sides: the plan
    must not write them, and the partially filled last block after the edit
    (where the next decoded tokens land) must not be one of them. The pool's
    own predicate also rules out hashed (prefix-cached) blocks.
    """
    written_block_ids = (
        np.concatenate([plan.gather_dst, plan.rotate_slots]) // block_size
    )
    touched = set(written_block_ids.tolist())
    if num_slots_after % block_size:
        touched.add(blocks[num_slots_after // block_size].block_id)
    for block in blocks:
        if block.block_id in touched and not manager.block_pool.is_block_writable(
            block
        ):
            raise NotImplementedError(
                f"{request.request_id}: block {block.block_id} is shared with "
                "another request or cached; editing it or decoding into it is "
                "not supported yet"
            )


def _positions_of(request: Request) -> np.ndarray:
    # A stock layout is the pair (kv_positions None, offset 0); commit_edit
    # and reset_kv_surgery_layout always set both, and continue_positions
    # ignores the offset for the None case.
    assert request.kv_positions is not None or request.position_offset == 0, (
        f"{request.request_id}: position offset without per-slot positions"
    )
    return continue_positions(request.kv_positions, request.num_computed_tokens)


def _check_position_layout(
    scheduler: Scheduler, request_id: str, positions: np.ndarray, next_position: float
) -> None:
    max_position = scheduler.max_model_len
    if positions.size and positions.min() < 0:
        raise ValueError(f"{request_id}: edit would make a position negative")
    if not float(next_position).is_integer():
        raise ValueError(
            f"{request_id}: next position {next_position} is not an integer; "
            "newly generated tokens need integer positions"
        )
    if next_position >= max_position or (
        positions.size and positions.max() >= max_position
    ):
        raise ValueError(
            f"{request_id}: edit would place a position at or beyond "
            f"max_model_len {max_position}"
        )


def _allocate(scheduler: Scheduler, counts: list[int]) -> list[list[KVCacheBlock]]:
    """Take blocks straight from the free pool, with no watermark: a large
    splice can leave running requests short and get them preempted. Edits
    are rare and the caller decides when, so that is the policy for now."""
    pool = scheduler.kv_cache_manager.block_pool
    total = sum(counts)
    if total > pool.get_num_free_blocks():
        raise ValueError(
            f"KV surgery needs {total} free KV cache blocks, "
            f"only {pool.get_num_free_blocks()} available"
        )
    return [pool.get_new_blocks(count) if count else [] for count in counts]


def _free(scheduler: Scheduler, blocks: list[list[KVCacheBlock]]) -> None:
    """Return blocks to the pool and empty the lists, so a second call is a
    no-op rather than a double decrement."""
    pool = scheduler.kv_cache_manager.block_pool
    for group_blocks in blocks:
        if group_blocks:
            pool.free_blocks(reversed(group_blocks))
            group_blocks.clear()


def plan_edit(scheduler: Scheduler, request_id: str, op: KVEditOp) -> PlannedEdit:
    """Validate ``op`` against the request and compute the cache mutations.

    Only the block pool is touched (a splice that grows the view takes its
    new blocks here; ``release`` returns them). The op addresses the
    request's computed KV slots ``[0, num_computed_tokens)``; tokens past
    that (the pending sampled token, or an unfinished prefill) are untouched.

    Raises:
        KeyError: unknown request id.
        ValueError: the request has no editable cache, a span is out of
            range, the result has an invalid position layout, or the pool
            has too few free blocks.
        NotImplementedError: a feature the surgery path does not support.
    """
    request = _editable_request(scheduler, request_id)
    num_slots = request.num_computed_tokens
    managers = _managers(scheduler)
    request_blocks = [_request_blocks(manager, request) for _, manager in managers]
    views = _request_views(managers, request_blocks, _positions_of(request))

    allocated: list[list[KVCacheBlock]] = [[] for _ in views]
    new_tokens: list[int] = []
    if isinstance(op, SpliceOp):
        source = _editable_request(scheduler, op.src_request_id)
        src_blocks = [_request_blocks(manager, source) for _, manager in managers]
        src_views = _request_views(managers, src_blocks, _positions_of(source))
        # Validate the spans before taking blocks from the pool; splice()
        # checks them again, but after allocation. Group 0 stands for all:
        # every group's view has the same slot count and positions.
        views[0].check_span(op.start, op.end)
        src_views[0].check_span(op.src_start, op.src_end)
        kept = num_slots - (op.end - op.start) + (op.src_end - op.src_start)
        if kept > scheduler.max_model_len:
            raise ValueError(
                f"{request_id}: splice would leave {kept} slots, more than "
                f"max_model_len={scheduler.max_model_len}"
            )
        allocated = _allocate(
            scheduler,
            [
                max(0, cdiv(kept, view.block_size) - len(view.block_ids))
                for view in views
            ],
        )
        new_tokens = source._all_token_ids[op.src_start : op.src_end]

    try:
        plans = []
        new_views = []
        for g, view in enumerate(views):
            if isinstance(op, DropOp):
                new_view, plan = drop(view, op.start, op.end, op.close_gap)
            elif isinstance(op, ShiftOp):
                new_view, plan = shift(view, op.start, op.end, op.delta)
            elif isinstance(op, SpliceOp):
                new_view, plan = splice(
                    view,
                    op.start,
                    op.end,
                    src_views[g],
                    op.src_start,
                    op.src_end,
                    op.close_gap,
                    tuple(block.block_id for block in allocated[g]),
                )
            else:
                raise TypeError(f"unknown edit op {op!r}")
            _check_writable(
                request,
                managers[g][1],
                request_blocks[g] + allocated[g],
                plan,
                view.block_size,
                new_view.num_slots,
            )
            new_views.append(new_view)
            plans.append(plan)

        new_positions = new_views[0].positions
        if new_positions.size == 0:
            # Symmetric with plan_fork's lower bound; drop everything means
            # "start over", which is a new request, not an edit.
            raise ValueError(f"{request_id}: edit would leave no computed KV")
        next_position = new_views[0].next_position
        _check_position_layout(scheduler, request_id, new_positions, next_position)
    except Exception:
        _free(scheduler, allocated)
        raise

    return PlannedEdit(
        request=request,
        op=op,
        num_slots_before=num_slots,
        new_positions=new_positions,
        position_offset=int(next_position) - new_views[0].num_slots,
        plans=plans,
        new_num_blocks=[len(view.block_ids) for view in new_views],
        allocated_blocks=allocated,
        new_tokens=list(new_tokens),
    )


def release(scheduler: Scheduler, planned: "PlannedEdit | PlannedFork") -> None:
    """Return the blocks a plan took from the pool (the plan is abandoned).

    A no-op after a successful commit: both commits hand the blocks to the
    request and empty the plan's lists.
    """
    if isinstance(planned, PlannedEdit):
        _free(scheduler, planned.allocated_blocks)
    elif isinstance(planned, PlannedFork):
        _free(scheduler, planned.fresh_blocks)
    else:
        raise TypeError(f"unknown plan {planned!r}")


def _assert_unhashed(request: Request) -> None:
    """Token histories are rewritten without touching ``block_hashes``; that
    is only sound because prefix caching is off (``_check_supported``), so
    the request has no hasher and no hashes."""
    assert request._block_hasher is None and not request.block_hashes, (
        f"{request.request_id}: KV surgery on a request with block hashes"
    )


def _splice_tokens(
    request: Request, start: int, end: int, new_tokens: list[int]
) -> None:
    """Replace tokens ``[start, end)`` of the request's history.

    Surviving tokens keep their prompt/output classification; inserted ones
    are prompt tokens iff ``start`` lies inside the prompt. ``_all_token_ids``
    and ``_output_token_ids`` are edited in place because the request's
    read-only views wrap them; ``prompt_token_ids`` is rebound because
    in-process engines share that list with the caller and the output
    processor.
    """
    assert request.prompt_token_ids is not None
    _assert_unhashed(request)
    old_prompt_len = request.num_prompt_tokens
    new_all = request._all_token_ids[:start] + new_tokens + request._all_token_ids[end:]
    new_prompt_len = old_prompt_len - (
        min(end, old_prompt_len) - min(start, old_prompt_len)
    )
    if start < old_prompt_len:
        new_prompt_len += len(new_tokens)
    request._all_token_ids[:] = new_all
    request.prompt_token_ids = new_all[:new_prompt_len]
    request._output_token_ids[:] = new_all[new_prompt_len:]
    request.num_prompt_tokens = new_prompt_len


def commit_edit(scheduler: Scheduler, planned: PlannedEdit) -> KVEditResult:
    """Make the scheduler agree with an edit whose cache work is done."""
    request = planned.request
    request_id = request.request_id
    if request.num_computed_tokens != planned.num_slots_before:
        # Cannot happen with synchronous scheduling (nothing runs between
        # plan and commit inside one utility call); if it ever does, the
        # worker pass has already run and the request's cache is suspect.
        release(scheduler, planned)
        raise RuntimeError(
            f"{request_id}: request changed between planning and committing; "
            "its cache may be partially edited and it should be aborted"
        )

    num_freed = 0
    for (_, manager), keep, allocated in zip(
        _managers(scheduler),
        planned.new_num_blocks,
        planned.allocated_blocks,
        strict=True,
    ):
        blocks = manager.req_to_blocks.get(request_id)
        if blocks is None:
            if not allocated:
                continue
            blocks = manager.req_to_blocks[request_id]
        blocks.extend(allocated)
        allocated.clear()  # owned by the request now; release() has nothing
        freed = blocks[keep:]
        del blocks[keep:]
        if freed:
            # A block another request still holds only loses a reference.
            num_freed += sum(block.ref_cnt == 1 for block in freed)
            manager.block_pool.free_blocks(reversed(freed))

    op = planned.op
    if isinstance(op, DropOp):
        _splice_tokens(request, op.start, op.end, [])
    elif isinstance(op, SpliceOp):
        _splice_tokens(request, op.start, op.end, planned.new_tokens)
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


def plan_fork(
    scheduler: Scheduler,
    src_request_id: str,
    request: Request,
    num_slots: int | None,
) -> PlannedFork:
    """Plan a new request whose cache starts as the source's first
    ``num_slots`` slots (all computed slots by default).

    ``request`` is the not-yet-added new request; its prompt is the tokens to
    append after the shared prefix and must not be empty. Fully occupied
    blocks are shared by reference, the partial last block is copied into a
    fresh block taken from the pool.

    Raises:
        KeyError: unknown source request.
        ValueError: the source has no editable cache, ``num_slots`` is out of
            range, the fork would exceed ``max_model_len``, the new request id
            is taken, or the pool has too few free blocks.
        NotImplementedError: a feature the surgery path does not support.
    """
    source = _editable_request(scheduler, src_request_id)
    if request.request_id in scheduler.requests:
        raise ValueError(f"request id {request.request_id!r} is already in use")
    _check_supported(scheduler, request)
    if num_slots is None:
        num_slots = source.num_computed_tokens
    if not 1 <= num_slots <= source.num_computed_tokens:
        raise ValueError(
            f"{src_request_id}: cannot fork {num_slots} slots of "
            f"{source.num_computed_tokens} computed"
        )
    num_new = request.num_tokens
    if num_new < 1:
        raise ValueError("a fork needs at least one new token")
    params = request.sampling_params
    assert params is not None  # pooling was rejected above
    if params.n != 1:
        raise ValueError("a fork samples one sequence; n must be 1")
    if params.prompt_logprobs is not None:
        # The frontend knows the fork's prompt as the appended tokens only.
        raise NotImplementedError("prompt_logprobs are not supported on a fork")
    # Slots and positions are bounded separately: a source shifted down can
    # have more slots than positions, and vice versa.
    if num_slots + num_new > scheduler.max_model_len:
        raise ValueError(
            f"fork of {num_slots} slots plus {num_new} new tokens needs more "
            f"than max_model_len={scheduler.max_model_len} slots"
        )
    source_positions = _positions_of(source)
    positions = source_positions[:num_slots]
    next_position = float(positions[-1]) + 1.0
    # The first sampled token lands after the appended prompt.
    _check_position_layout(
        scheduler, request.request_id, positions, next_position + num_new
    )

    managers = _managers(scheduler)
    source_blocks = [_request_blocks(manager, source) for _, manager in managers]
    views = _request_views(managers, source_blocks, source_positions)
    fresh = _allocate(
        scheduler, [int(num_slots % view.block_size != 0) for view in views]
    )
    plans = []
    shared_blocks = []
    try:
        for view, blocks, fresh_blocks in zip(views, source_blocks, fresh, strict=True):
            _, plan, num_shared = fork(
                view, num_slots, tuple(block.block_id for block in fresh_blocks)
            )
            plans.append(plan)
            shared_blocks.append(blocks[:num_shared])
    except Exception:
        _free(scheduler, fresh)
        raise

    return PlannedFork(
        source=source,
        request=request,
        num_slots=num_slots,
        positions=positions,
        position_offset=int(next_position) - num_slots,
        plans=plans,
        shared_blocks=shared_blocks,
        fresh_blocks=fresh,
    )


def commit_fork(scheduler: Scheduler, planned: PlannedFork) -> None:
    """Give the new request its shared prefix; the caller then adds it to the
    scheduler, which prefills the appended tokens on top of it.

    Relies on the scheduler treating a WAITING request with
    ``num_computed_tokens > 0`` as already holding that much KV (the branch
    the KV-connector path uses): it skips the prefix-cache lookup and only
    allocates for the tokens past ``num_computed_tokens``. That branch also
    records no prefill stats, so forks have no time-to-first-token metrics.
    """
    source, request, num_slots = planned.source, planned.request, planned.num_slots
    if source.num_computed_tokens < num_slots:
        release(scheduler, planned)
        raise RuntimeError(
            f"{source.request_id}: request changed between planning and committing"
        )
    assert request.prompt_token_ids is not None
    _assert_unhashed(request)

    prefix = source._all_token_ids[:num_slots]
    request.prompt_token_ids = prefix + list(request.prompt_token_ids)
    request._all_token_ids[:] = request.prompt_token_ids
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.num_computed_tokens = num_slots
    # Stock source (see _positions_of) means a stock fork.
    request.kv_positions = (
        None if source.kv_positions is None else planned.positions.copy()
    )
    request.position_offset = planned.position_offset

    # Refcounts first for every group, then the block tables. Touching is
    # increments only and cannot fail, so by the time anything can raise
    # every table entry holds exactly the references rollback_fork frees.
    pool = scheduler.kv_cache_manager.block_pool
    managers = _managers(scheduler)
    for shared in planned.shared_blocks:
        pool.touch(shared)
    for (_, manager), shared, fresh in zip(
        managers, planned.shared_blocks, planned.fresh_blocks, strict=True
    ):
        manager.req_to_blocks[request.request_id] = list(shared) + list(fresh)
        fresh.clear()  # owned by the request now; release() has nothing


def rollback_fork(scheduler: Scheduler, planned: PlannedFork) -> None:
    """Undo ``commit_fork`` for a request the scheduler never admitted.

    Frees what the fork's block tables hold, which is one reference per
    shared block plus the fresh blocks; exact because ``commit_fork``'s
    touch loop cannot fail and completes before any table is filled. The
    request itself is not
    removed from the scheduler: ``EngineCore.add_request`` runs its checks
    before ``Scheduler.add_request`` and the one step after it (the
    ``abort_immediately`` hook) is refused by ``fork_kv`` up front, so a
    failure means the scheduler never saw the request.
    """
    pool = scheduler.kv_cache_manager.block_pool
    for _, manager in _managers(scheduler):
        blocks = manager.req_to_blocks.pop(planned.request.request_id, [])
        if blocks:
            pool.free_blocks(reversed(blocks))


def inspect_kv(
    scheduler: Scheduler, request_id: str, positions: bool = False
) -> KVViewInfo:
    """Describe a live request's KV layout. Read-only.

    ``positions`` asks for the per-slot position list, which is as long as
    the context and crosses the utility RPC; leave it off when polling.

    ``num_shared_blocks`` counts, per KV cache group, the blocks referenced by
    more than one request: fork-shared blocks when prefix caching is off (the
    only configuration surgery ops accept), possibly prefix-cache hits
    otherwise; this call itself does not check the configuration.

    Raises:
        KeyError: unknown request id.
        NotImplementedError: data parallelism (the request lives on one
            engine but the call fans out to all of them).
    """
    if scheduler.vllm_config.parallel_config.data_parallel_size > 1:
        raise NotImplementedError("KV surgery does not support data parallelism")
    request = scheduler.requests.get(request_id)
    if request is None:
        raise KeyError(f"unknown request {request_id!r}")
    block_ids = []
    num_shared = []
    for _, manager in _managers(scheduler):
        blocks = manager.req_to_blocks.get(request_id, [])
        block_ids.append([block.block_id for block in blocks])
        num_shared.append(sum(block.ref_cnt > 1 for block in blocks))
    return KVViewInfo(
        request_id=request_id,
        status=request.status.name,
        num_slots=request.num_computed_tokens,
        num_tokens=request.num_tokens,
        num_prompt_tokens=request.num_prompt_tokens,
        next_position=request.num_computed_tokens + request.position_offset,
        positions=_positions_of(request).tolist() if positions else None,
        block_ids=block_ids,
        num_shared_blocks=num_shared,
    )
