# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""edit_kv end to end on Llama 3.2 1B (single GPU, V2 model runner).

Oracles that stock vLLM can compute exactly:
  * shifting the whole context by an integer changes nothing observable
    (RoPE is relative), so the continuation must match the unedited run;
  * dropping a tail span is a rollback: the survivors' KV never depended on
    the dropped tokens, so the continuation must match a fresh request with
    the shortened token list.
The deliberately inconsistent edits (dropping a middle span) are only checked
for bookkeeping here; their semantics are step 3's oracle harness.
"""

import os

# The in-process engine initializes CUDA in this process, so the engine core
# subprocess of the multiprocess test cannot be forked from it.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.engine.llm_engine import LLMEngine  # noqa: E402
from vllm.v1.kv_surgery.ops import DropOp, KVEditOp, ShiftOp  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    # The module-scoped engine must outlive the per-test distributed cleanup.
    pytest.mark.skip_global_cleanup,
]

MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards "
    "jump quickly. Sphinx of black quartz, judge my vow. Then the story began:"
)


def _make_engine(multiprocess: bool) -> LLMEngine:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if multiprocess else "0"
    args = EngineArgs(
        model=MODEL,
        enable_prefix_caching=False,
        max_model_len=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.3,
        # edit_kv runs between steps and needs no batch in flight.
        async_scheduling=False,
        seed=0,
    )
    return LLMEngine.from_engine_args(args, enable_multiprocessing=multiprocess)


@pytest.fixture(scope="module")
def engine():
    engine = _make_engine(multiprocess=False)
    yield engine
    engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def prompt_ids(engine) -> list[int]:
    return engine.renderer.tokenizer.encode(PROMPT)


class Driver:
    """Runs one greedy request step by step so edits can land between steps."""

    def __init__(self, engine: LLMEngine, req_id: str, tokens: list[int], n: int):
        self.engine = engine
        self.req_id = req_id
        self.generated: list[int] = []
        self.finished = False
        # The engine core knows the request under an internal id; edit_kv
        # accepts either that or the id the caller used.
        self.core_req_id = engine.add_request(
            req_id,
            {"prompt_token_ids": tokens},
            SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True),
        )

    def request(self):
        return _scheduler(self.engine).requests[self.core_req_id]

    def step(self) -> None:
        for out in self.engine.step():
            if out.request_id == self.req_id:
                self.generated = list(out.outputs[0].token_ids)
                self.finished |= out.finished

    def run_until(self, num_generated: int) -> None:
        while len(self.generated) < num_generated and not self.finished:
            self.step()
        assert len(self.generated) >= num_generated

    def finish(self) -> list[int]:
        while not self.finished:
            self.step()
        return self.generated

    def edit(self, op: KVEditOp):
        return self.engine.edit_kv(self.req_id, op)


def _scheduler(engine: LLMEngine):
    return engine.engine_core.engine_core.scheduler  # type: ignore[attr-defined]


def test_baseline_is_deterministic(engine, prompt_ids):
    a = Driver(engine, "det-a", prompt_ids, 12).finish()
    b = Driver(engine, "det-b", prompt_ids, 12).finish()
    assert a == b


def test_shift_whole_context_preserves_continuation(engine, prompt_ids):
    baseline = Driver(engine, "shift-base", prompt_ids, 20).finish()

    driver = Driver(engine, "shift-edit", prompt_ids, 20)
    driver.run_until(6)
    request = driver.request()
    n = request.num_computed_tokens
    result = driver.edit(ShiftOp(0, n, 37))
    assert result.num_slots == n
    assert result.next_position == n + 37
    assert request.position_offset == 37

    edited = driver.finish()
    assert edited == baseline
    # Decoded tokens kept following the shifted layout.
    np.testing.assert_array_equal(
        request.kv_positions, np.arange(n, dtype=np.float64) + 37
    )
    assert request.position_offset == 37


def test_shift_roundtrip_preserves_continuation(engine, prompt_ids):
    baseline = Driver(engine, "rt-base", prompt_ids, 20).finish()

    driver = Driver(engine, "rt-edit", prompt_ids, 20)
    driver.run_until(4)
    n = driver.request().num_computed_tokens
    driver.edit(ShiftOp(0, n, 100))
    driver.run_until(8)
    m = driver.request().num_computed_tokens
    assert m > n
    result = driver.edit(ShiftOp(0, m, -100))
    assert result.next_position == m
    assert driver.finish() == baseline


def test_drop_tail_is_a_rollback(engine, prompt_ids):
    driver = Driver(engine, "tail-edit", prompt_ids, 16)
    driver.run_until(8)
    request = driver.request()
    n = request.num_computed_tokens
    tokens_before = list(request.all_token_ids)
    assert len(tokens_before) == n + 1  # one pending sampled token

    result = driver.edit(DropOp(n - 4, n))
    assert result.num_slots == n - 4
    assert result.num_freed_blocks >= 0
    kept = tokens_before[: n - 4] + tokens_before[n:]
    assert list(request.all_token_ids) == kept
    edited = driver.finish()

    fresh = Driver(engine, "tail-fresh", kept, 16 - len(edited[:8]) + 4).finish()
    # After the edit both runs continue from the same token list.
    continuation = edited[8:]
    assert fresh[: len(continuation)] == continuation


def test_drop_middle_span_bookkeeping(engine, prompt_ids):
    driver = Driver(engine, "mid-edit", prompt_ids, 24)
    driver.run_until(3)
    scheduler = _scheduler(engine)
    request = driver.request()
    n = request.num_computed_tokens
    blocks_before = len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0])
    prompt_before = list(request.prompt_token_ids)

    result = driver.edit(DropOp(5, 25))
    assert result.num_slots == n - 20
    assert result.next_position == n - 20
    assert request.prompt_token_ids == prompt_before[:5] + prompt_before[25:]
    blocks_after = len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0])
    assert blocks_after <= blocks_before
    np.testing.assert_array_equal(request.kv_positions, np.arange(n - 20))

    # Now open a gap instead: survivors keep their positions.
    result = driver.edit(DropOp(2, 4, close_gap=False))
    assert result.num_slots == n - 22
    assert result.next_position == n - 20
    assert request.position_offset == 2

    out = driver.finish()
    assert len(out) == 24
    assert all(0 <= t < 128256 for t in out)


def test_edit_over_the_multiprocess_client(prompt_ids):
    """The op, the result and errors cross the engine-core utility RPC.

    The core runs ahead of the frontend here, so only the prompt slots are
    known to be computed; semantics are covered by the in-process tests.
    """
    engine = _make_engine(multiprocess=True)
    try:
        driver = Driver(engine, "mp-edit", prompt_ids, 12)
        driver.run_until(3)
        prompt_len = len(prompt_ids)
        result = driver.edit(ShiftOp(0, prompt_len, 11))
        assert result.request_id == driver.core_req_id
        assert result.num_slots >= prompt_len + 2
        assert result.num_tokens == result.num_slots + 1
        # The unshifted tail keeps the next position at the slot count.
        assert result.next_position == result.num_slots
        with pytest.raises(Exception, match="span"):
            driver.edit(DropOp(0, 10_000))
        with pytest.raises(Exception, match="unknown request"):
            engine.edit_kv("never-added", ShiftOp(0, 1, 1.0))
        assert len(driver.finish()) == 12
    finally:
        engine.engine_core.shutdown()
