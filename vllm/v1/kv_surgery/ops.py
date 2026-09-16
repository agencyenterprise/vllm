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


KVEditOp = DropOp | ShiftOp


def as_edit_op(obj: object) -> KVEditOp:
    """Accept an op instance or its msgpack/dict form."""
    if isinstance(obj, DropOp | ShiftOp):
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
