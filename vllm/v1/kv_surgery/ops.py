# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The public edit ops and result of ``edit_kv``.

Ops are tagged msgspec structs so they survive the engine-core utility RPC:
the client sends a struct, the core receives a plain dict and converts it
back with ``as_edit_op``. Spans are logical slot indices ``[start, end)``
into the request's computed KV slots.
"""

import msgspec


class DropOp(msgspec.Struct, tag="drop", frozen=True):  # type: ignore[call-arg]
    """Remove slots ``[start, end)``; later slots move up (order preserved).

    With ``close_gap`` the survivors after the span are re-rotated so their
    positions stay contiguous with the slots before the span.

    The dropped tokens leave the request's token history too. Output tokens
    that were already streamed to the caller are not retracted, but they no
    longer count against ``max_tokens``, so the caller can receive more than
    ``max_tokens`` tokens in total over the life of the request.
    """

    start: int
    end: int
    close_gap: bool = True


class ShiftOp(msgspec.Struct, tag="shift", frozen=True):  # type: ignore[call-arg]
    """Add ``delta`` to the positions of slots ``[start, end)``.

    ``delta`` may be fractional, but the request's next position (the one
    the next generated token gets) must stay integral.
    """

    start: int
    end: int
    delta: float


class SpliceOp(msgspec.Struct, tag="splice", frozen=True):  # type: ignore[call-arg]
    """Replace slots ``[start, end)`` with a copy of another request's slots
    ``[src_start, src_end)``, tokens and positions included.

    This is how a restricted prefill lands: fork the request at a prefix,
    prefill new tokens on the fork (they attend to that prefix only), then
    splice the fork's new slots back in place of the span they replace. The
    source may also be the request itself. Copied slots keep the positions
    they were computed at; with ``close_gap`` the slots after ``end`` are
    re-rotated to continue right after the last inserted slot (with an empty
    source span this is ``DropOp``). Inserted tokens count as prompt tokens
    when ``start`` lies inside the prompt and as output tokens otherwise.

    As with ``ShiftOp``, only negative and out-of-range positions are
    rejected: whether the resulting layout is monotonic or overlaps is the
    caller's business (deliberately inconsistent caches are the point).
    """

    start: int
    end: int
    src_request_id: str
    src_start: int
    src_end: int
    close_gap: bool = True


KVEditOp = DropOp | ShiftOp | SpliceOp


def as_edit_op(obj: object) -> KVEditOp:
    """Accept an op instance or its msgpack/dict form."""
    if isinstance(obj, DropOp | ShiftOp | SpliceOp):
        return obj
    return msgspec.convert(obj, KVEditOp)


class KVEditResult(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """What ``edit_kv`` reports back (plain types only; crosses the RPC)."""

    request_id: str
    num_slots: int
    """Computed KV slots the request now has."""
    num_tokens: int
    """Tokens in the request's history, including not-yet-computed ones."""
    next_position: int
    """RoPE position the next generated token will get."""
    num_freed_blocks: int
    """Blocks returned to the pool. A block another request still holds
    (a fork's shared prefix) leaves this request's table but is not counted."""


class KVViewInfo(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """A request's KV layout as the scheduler sees it (``inspect_kv``)."""

    request_id: str
    status: str
    """``RequestStatus`` name, e.g. ``RUNNING`` or ``WAITING_FOR_STREAMING_REQ``."""
    num_slots: int
    num_tokens: int
    num_prompt_tokens: int
    next_position: int
    positions: list[float] | None
    """RoPE position of each computed slot, in logical order; only when
    asked for (``inspect_kv(..., positions=True)``)."""
    block_ids: list[list[int]]
    """Block table per KV cache group."""
    num_shared_blocks: list[int]
    """Per KV cache group, blocks referenced by more than one request: forks,
    or prefix-cache hits if caching is on (which surgery ops reject but this
    readout does not check)."""
