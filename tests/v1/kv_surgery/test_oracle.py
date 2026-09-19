# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness oracle for KV surgery on Llama 3.2 1B (single GPU).

Every *consistent* edit must reproduce a from-scratch forward pass's
next-token logits: a tail drop is a rollback, shifting every slot is
invisible to RoPE, and tokens prefilled against an untouched prefix (on a
fork) then spliced back are the KV a fresh request would have computed.
Inconsistent edits (the summary flow that keeps T2's cache while replacing
T1) are only checked for well-formedness: for them "correct" means "matches
the edited-cache forward", which is what the engine computes by construction.

Logits come from probes: a fork with ``max_tokens=1`` and full-vocab
``logprobs`` in ``raw_logits`` mode commits nothing and frees itself. The
long-lived request is a resumable streaming request, paused with its cache
intact, which is how an agent context lives in this engine. The tolerance is
calibrated against the noise floor between a token decoded step by step and
the same context prefilled in one shot (bf16 GEMMs and attention kernels
differ between those paths); the first test reports it.
"""

import itertools

import numpy as np
import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.logprobs import FlatLogprobs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.kv_surgery.ops import DropOp, ShiftOp, SpliceOp

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skip_global_cleanup,
]

MODEL = "meta-llama/Llama-3.2-1B"
# Max abs logit difference allowed between an edited cache and a fresh
# prefill. Measured on an H200: copies (drop, splice, fork) reproduce the
# fresh logits exactly, a rotation costs one bf16 rounding of K, which moved
# logits by at most 0.125; the noise-floor test reports the stock baseline.
ATOL = 0.25
# A different context must move the logits by far more than the tolerance,
# otherwise the oracle could not tell an edit from a no-op.
MIN_CONTRAST = 4 * ATOL

PREFIX = (
    "You are a meticulous archivist. Rules: answer briefly, cite sources, "
    "never speculate. The reading room closes at six."
)
TRAJECTORY_1 = (
    " Visitor: I am looking for the 1887 harbor ledger. Archivist: It is in "
    "box 14, shelf C; the binding is fragile, use the cradle."
)
SUMMARY = " [Earlier: the visitor was pointed to the 1887 harbor ledger in box 14.]"


@pytest.fixture(scope="module")
def engine():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        engine = LLMEngine.from_engine_args(
            EngineArgs(
                model=MODEL,
                enable_prefix_caching=False,
                max_model_len=1024,
                max_num_seqs=8,
                gpu_memory_utilization=0.3,
                async_scheduling=False,
                seed=0,
                logprobs_mode="raw_logits",
                max_logprobs=-1,
            ),
            enable_multiprocessing=False,
        )
    yield engine
    engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def ids(engine) -> dict[str, list[int]]:
    tok = engine.renderer.tokenizer
    return {
        "prefix": tok.encode(PREFIX),
        "t1": tok.encode(TRAJECTORY_1, add_special_tokens=False),
        "summary": tok.encode(SUMMARY, add_special_tokens=False),
    }


_REQUEST_COUNTER = itertools.count(1)


class Harness:
    """Drives one in-process engine step by step."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.vocab_size = engine.model_config.get_vocab_size()
        self.block_size = engine.vllm_config.cache_config.block_size
        self.outputs: dict[str, RequestOutput] = {}
        self.internal_ids: dict[str, str] = {}

    @staticmethod
    def _next_id(kind: str) -> str:
        return f"{kind}-{next(_REQUEST_COUNTER)}"

    def close(self) -> None:
        """Abort whatever is still alive (paused contexts and forks)."""
        self.engine.abort_request(list(self.internal_ids))

    @staticmethod
    def _params(max_tokens: int, logits: bool) -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            logprobs=-1 if logits else None,
            flat_logprobs=logits,
        )

    def add(self, tokens: list[int], max_tokens: int, resumable: bool = False) -> str:
        """A request; a resumable one pauses at its first stop with its cache
        intact instead of finishing (the agent-context model)."""
        req_id = self._next_id("req")
        params = self._params(max_tokens, logits=False)
        request = self.engine.input_processor.process_inputs(
            req_id,
            {"prompt_token_ids": tokens},
            params,
            supported_tasks=self.engine.get_supported_tasks(),
            resumable=resumable,
        )
        self.internal_ids[req_id] = self.engine.add_request(req_id, request, params)
        return req_id

    def resume(self, req_id: str, tokens: list[int], max_tokens: int) -> None:
        """Feed the next input chunk to a paused request (what AsyncLLM's
        streaming input does)."""
        params = self._params(max_tokens, logits=False)
        request: EngineCoreRequest = self.engine.input_processor.process_inputs(
            self.internal_ids[req_id],
            {"prompt_token_ids": tokens},
            params,
            supported_tasks=self.engine.get_supported_tasks(),
            resumable=True,
        )
        request.external_req_id = req_id
        self.engine.output_processor.add_request(request, None, None, 0)
        self.engine.engine_core.add_request(request)
        self.outputs.pop(req_id, None)

    def fork(
        self,
        src: str,
        tokens: list[int],
        num_slots: int | None = None,
        max_tokens: int = 1,
        logits: bool = True,
        resumable: bool = False,
    ) -> str:
        req_id = self._next_id("fork")
        self.internal_ids[req_id] = self.engine.fork_request(
            src,
            req_id,
            {"prompt_token_ids": tokens},
            self._params(max_tokens, logits),
            num_slots,
            resumable=resumable,
        )
        return req_id

    def step(self) -> None:
        for out in self.engine.step():
            self.outputs[out.request_id] = out

    def until_stop(self, req_id: str) -> RequestOutput:
        """Run until the request finishes or pauses."""
        while True:
            out = self.outputs.get(req_id)
            if out is not None and (out.finished or out.outputs[0].finish_reason):
                return out
            self.step()

    def tokens(self, req_id: str) -> list[int]:
        """The engine's token history: computed slots plus the pending token."""
        scheduler = self.engine.engine_core.engine_core.scheduler  # type: ignore
        return list(scheduler.requests[self.internal_ids[req_id]].all_token_ids)

    def logits(self, out: RequestOutput) -> np.ndarray:
        flat = out.outputs[0].logprobs
        assert isinstance(flat, FlatLogprobs)
        start, end = flat.start_indices[0], flat.end_indices[0]
        logits = np.full(self.vocab_size, np.nan)
        logits[flat.token_ids[start:end]] = flat.logprobs[start:end]
        assert not np.isnan(logits).any()
        return logits

    def fresh_logits(self, tokens: list[int]) -> np.ndarray:
        req_id = self._next_id("fresh")
        self.internal_ids[req_id] = self.engine.add_request(
            req_id, {"prompt_token_ids": tokens}, self._params(1, logits=True)
        )
        return self.logits(self.until_stop(req_id))

    def probe(
        self, src: str, tokens: list[int], num_slots: int | None = None
    ) -> np.ndarray:
        """Next-token logits of ``src``'s view (first ``num_slots`` slots)
        continued with ``tokens``; nothing is committed."""
        return self.logits(self.until_stop(self.fork(src, tokens, num_slots)))

    def paused_context(self, tokens: list[int], num_generated: int) -> str:
        req_id = self.add(tokens, num_generated, resumable=True)
        out = self.until_stop(req_id)
        assert not out.finished
        assert self.engine.inspect_kv(req_id).status == "WAITING_FOR_STREAMING_REQ"
        return req_id


@pytest.fixture
def h(engine):
    harness = Harness(engine)
    yield harness
    harness.close()


def _max_diff(label: str, a: np.ndarray, b: np.ndarray) -> float:
    diff = float(np.abs(a - b).max())
    print(f"{label}: max |dlogit| = {diff:.4f}, argmax {a.argmax()} vs {b.argmax()}")
    return diff


def test_probe_matches_fresh_prefill_within_noise(h, ids):
    """Calibration: the unedited decoded cache vs a one-shot prefill."""
    req = h.paused_context(ids["prefix"] + ids["t1"], num_generated=8)
    ctx = h.tokens(req)
    n = len(ctx) - 1
    assert h.engine.inspect_kv(req).num_slots == n

    probed = h.probe(req, ctx[-1:])
    fresh = h.fresh_logits(ctx)
    assert _max_diff("noise floor", probed, fresh) < ATOL
    # Probing committed nothing and released its shared blocks.
    assert h.tokens(req) == ctx
    info = h.engine.inspect_kv(req)
    assert info.num_slots == n
    assert info.num_shared_blocks == [0]

    other = h.fresh_logits(ids["prefix"] + ids["summary"] + ctx[-1:])
    assert _max_diff("contrast", fresh, other) > MIN_CONTRAST


def test_tail_drop_is_a_rollback(h, ids):
    req = h.paused_context(ids["prefix"] + ids["t1"], num_generated=8)
    ctx = h.tokens(req)
    n = len(ctx) - 1
    result = h.engine.edit_kv(req, DropOp(n - 5, n))
    assert result.num_slots == n - 5
    kept = ctx[: n - 5] + ctx[n:]
    assert h.tokens(req) == kept

    probed = h.probe(req, kept[-1:])
    fresh = h.fresh_logits(kept)
    assert _max_diff("tail drop", probed, fresh) < ATOL


def test_shift_of_every_slot_is_invisible(h, ids):
    req = h.paused_context(ids["prefix"] + ids["t1"], num_generated=6)
    ctx = h.tokens(req)
    n = len(ctx) - 1
    result = h.engine.edit_kv(req, ShiftOp(0, n, 37))
    assert result.next_position == n + 37
    info = h.engine.inspect_kv(req, positions=True)
    assert info.positions == [float(p) for p in range(37, n + 37)]

    probed = h.probe(req, ctx[-1:])
    fresh = h.fresh_logits(ctx)
    assert _max_diff("shift all by 37", probed, fresh) < ATOL


def test_shift_roundtrip_returns_to_the_same_cache(h, ids):
    req = h.paused_context(ids["prefix"] + ids["t1"], num_generated=6)
    ctx = h.tokens(req)
    n = len(ctx) - 1
    before = h.probe(req, ctx[-1:])
    h.engine.edit_kv(req, ShiftOp(0, n, 100))
    h.engine.edit_kv(req, ShiftOp(0, n, -100))
    assert h.engine.inspect_kv(req, positions=True).positions == [
        float(p) for p in range(n)
    ]
    after = h.probe(req, ctx[-1:])
    # Two bf16 re-rotations of K, nothing else changed.
    assert _max_diff("shift roundtrip", after, before) < ATOL


def test_restricted_prefill_and_splice_match_fresh(h, ids):
    """The consistent half of the summary flow: [S][T1] -> [S][summary].

    The summary is prefilled on a fork that sees S only, so its KV equals a
    fresh prefill of [S][summary]; splicing it in place of T1 leaves the
    request holding exactly the cache a fresh [S][summary] request would.
    """
    prefix, summary = ids["prefix"], ids["summary"]
    req = h.paused_context(prefix + ids["t1"], num_generated=6)
    ctx = h.tokens(req)
    n, s = len(ctx) - 1, len(prefix)
    # Deliberately loud: this test exists to cover the partial-block copy in
    # fork(). If PREFIX is edited to a block-aligned token count, change it
    # back rather than weakening the check.
    assert s % h.block_size != 0

    fork = h.fork(req, summary, num_slots=s, resumable=True)
    fork_out = h.until_stop(fork)
    assert not fork_out.finished
    fresh_summary = h.fresh_logits(prefix + summary)
    assert _max_diff("restricted prefill", h.logits(fork_out), fresh_summary) < ATOL
    info = h.engine.inspect_kv(fork)
    assert info.num_slots == s + len(summary)
    assert info.status == "WAITING_FOR_STREAMING_REQ"
    assert h.engine.inspect_kv(req).num_shared_blocks == [s // h.block_size]

    result = h.engine.edit_kv(req, SpliceOp(s, n, fork, s, s + len(summary)))
    assert result.num_slots == s + len(summary)
    assert result.next_position == s + len(summary)
    spliced = prefix + summary + ctx[n:]
    assert h.tokens(req) == spliced
    h.engine.abort_request([fork])
    info = h.engine.inspect_kv(req, positions=True)
    assert info.num_shared_blocks == [0]
    assert info.positions == [float(p) for p in range(s + len(summary))]

    probed = h.probe(req, spliced[-1:])
    fresh = h.fresh_logits(spliced)
    assert _max_diff("splice", probed, fresh) < ATOL
    # The oracle can tell: the old context gives different logits.
    assert _max_diff("contrast", fresh, h.fresh_logits(ctx)) > MIN_CONTRAST


def test_summary_flow_keeps_t2_and_resumes(h, ids):
    """[S][T1][T2] -> [S][summary][T2]: T2's cache was computed with T1
    present, so no fresh forward reproduces it. Check the layout, that the
    request keeps generating after a streaming resume, and that the edit did
    not disturb the fork it copied from."""
    prefix, t1, summary = ids["prefix"], ids["t1"], ids["summary"]
    req = h.paused_context(prefix + t1, num_generated=10)
    ctx = h.tokens(req)
    n, s = len(ctx) - 1, len(prefix)
    t2 = ctx[s + len(t1) : n]
    assert len(t2) == 9

    fork = h.fork(req, summary, num_slots=s, resumable=True)
    fork_logits = h.logits(h.until_stop(fork))
    result = h.engine.edit_kv(req, SpliceOp(s, s + len(t1), fork, s, s + len(summary)))
    assert result.num_slots == s + len(summary) + len(t2)
    assert result.next_position == result.num_slots
    assert h.tokens(req) == prefix + summary + t2 + ctx[n:]
    info = h.engine.inspect_kv(req, positions=True)
    assert info.positions == [float(p) for p in range(result.num_slots)]
    assert info.num_prompt_tokens == s + len(summary)

    # The fork still holds its own cache: probing it reproduces its logits.
    # This probe is a fork of a fork, the suite's only coverage of a block
    # going from two references to three; keep it.
    fork_again = h.probe(fork, summary[-1:], s + len(summary) - 1)
    assert _max_diff("fork intact", fork_again, fork_logits) < ATOL
    h.engine.abort_request([fork])

    edited = h.probe(req, ctx[n:])
    assert np.isfinite(edited).all()
    # Resume the paused context with a new input chunk and keep generating:
    # five prompt tokens land on the edited cache, then the session decodes
    # until its (original) max_tokens again.
    h.resume(req, ids["t1"][:5], max_tokens=4)
    out = h.until_stop(req)
    assert all(0 <= t < h.vocab_size for t in out.outputs[0].token_ids[-4:])
    resumed = h.tokens(req)
    assert resumed[: result.num_slots] == prefix + summary + t2
    assert resumed[result.num_slots : result.num_slots + 5] == ids["t1"][:5]
    info = h.engine.inspect_kv(req, positions=True)
    assert info.num_slots >= result.num_slots + 5 + 3
    assert info.positions == [float(p) for p in range(info.num_slots)]


def test_forks_diverge_independently(h, ids):
    req = h.paused_context(ids["prefix"] + ids["t1"], num_generated=6)
    ctx = h.tokens(req)
    n = len(ctx) - 1
    base = h.probe(req, ctx[-1:])

    fork = h.fork(req, ctx[-1:], max_tokens=6, logits=False, resumable=True)
    fork_out = h.until_stop(fork)
    fork_ctx = h.tokens(fork)
    assert fork_ctx[: n + 1] == ctx
    assert len(fork_ctx) == n + 1 + 6
    assert h.engine.inspect_kv(req).num_shared_blocks == [n // h.block_size]

    # Shortening the fork back into a shared block is refused: its next
    # decode would overwrite slots the source still reads.
    shared_end = (n // h.block_size) * h.block_size
    if shared_end > 1:
        with pytest.raises(NotImplementedError, match="decoding into"):
            h.engine.edit_kv(fork, DropOp(shared_end - 1, len(fork_ctx) - 1))
    # Editing the fork's own tail touches no shared block and not the source.
    m = len(fork_ctx) - 1
    h.engine.edit_kv(fork, DropOp(m - 3, m))
    kept = fork_ctx[: m - 3] + fork_ctx[m:]
    assert (
        _max_diff("fork after drop", h.probe(fork, kept[-1:]), h.fresh_logits(kept))
        < ATOL
    )
    # Same cache, but the two probes ran in different batches and bf16
    # reductions are not batch-invariant, so only the calibrated tolerance
    # is owed here (measured: identical).
    assert _max_diff("source untouched", h.probe(req, ctx[-1:]), base) < ATOL
    assert h.tokens(req) == ctx
    h.engine.abort_request([fork])
    assert h.engine.inspect_kv(req).num_shared_blocks == [0]
    assert list(fork_out.outputs[0].token_ids) == fork_ctx[n + 1 :]
