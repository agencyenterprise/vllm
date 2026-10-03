# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode throughput on edited KV caches versus stock.

The design claim is that an edit changes nothing about decode speed: the
attention kernels are untouched and an edited cache is just another block
table. This script measures decode tokens/s for a batch of long-context
requests (a) stock, (b) after every request's cache was edited (a middle drop
that gathers and rotates the tail, then a full-view shift), optionally
repeating the edits every ``--edit-every`` timed steps so their stalls are
included, and reports the edit latency itself. Both edit timings
(``first_edit_ms_per_req``, ``timed_edit_s``) cover the ``edit_kv`` calls
only, not the ``inspect_kv`` calls that size them or the drain those trigger;
the decode throughput of the periodic variant includes everything.

``--async-scheduling`` applies to both modes; the core drains its batch
queue before each edit, which the periodic-edit variant measures.

Example::

    python benchmarks/kv_surgery/benchmark_edited_decode.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --num-seqs 64 \\
        --prompt-len 2048 --decode-steps 64 --mode edited
"""

import argparse
import json
import os
import time

# In-process engine core (edits are direct calls); this env var overrides the
# from_engine_args argument, so it is the one that has to be set.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np  # noqa: E402

from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.engine.llm_engine import LLMEngine  # noqa: E402
from vllm.v1.kv_surgery.ops import DropOp, ShiftOp  # noqa: E402

WARM_STEPS = 4
EDIT_DROP = 64
EDIT_SHIFT = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--mode", choices=["stock", "edited"], default="edited")
    parser.add_argument(
        "--edit-every",
        type=int,
        default=0,
        help="Re-edit every request every N timed steps (0: edit once, untimed).",
    )
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def edit_all(
    engine: LLMEngine, request_ids: list[str], prompt_len: int, shift: float
) -> float:
    """Drop ``EDIT_DROP`` tokens from the middle (gathers and re-rotates the
    tail), then shift the whole view by ``shift``. Returns the mean wall time
    of the two edits per request; the inspect calls (and, under async
    scheduling, the one batch-queue drain they trigger) are outside the
    timing."""
    num_slots = {rid: engine.inspect_kv(rid).num_slots for rid in request_ids}
    start = time.perf_counter()
    for request_id in request_ids:
        n = num_slots[request_id]
        drop_start = min(prompt_len // 4, n - EDIT_DROP - 1)
        engine.edit_kv(request_id, DropOp(drop_start, drop_start + EDIT_DROP))
        engine.edit_kv(request_id, ShiftOp(0, n - EDIT_DROP, shift))
    return (time.perf_counter() - start) / len(request_ids)


def main() -> None:
    args = parse_args()
    # Requests that finish prefill early start decoding while the rest are
    # still prefilling, so the cap must be well past the timed window.
    max_tokens = 4 * (WARM_STEPS + args.decode_steps) + 64
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            enable_prefix_caching=False,
            max_model_len=args.prompt_len + max_tokens + 128,
            max_num_seqs=args.num_seqs,
            max_num_batched_tokens=max(8192, args.prompt_len),
            gpu_memory_utilization=args.gpu_memory_utilization,
            async_scheduling=args.async_scheduling,
            seed=args.seed,
        ),
    )
    vocab = engine.model_config.get_vocab_size()
    rng = np.random.default_rng(args.seed)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    request_ids = [f"req-{i}" for i in range(args.num_seqs)]
    # The first edit drops 64 tokens; start edited requests that much longer
    # so both modes decode over the same context length.
    prompt_len = args.prompt_len + (EDIT_DROP if args.mode == "edited" else 0)
    for request_id in request_ids:
        tokens = rng.integers(1000, vocab - 1000, size=prompt_len).tolist()
        engine.add_request(request_id, {"prompt_token_ids": tokens}, params)

    # Prefill everything (untimed) until every request has produced a token.
    produced: dict[str, int] = dict.fromkeys(request_ids, 0)
    while min(produced.values()) == 0:
        for out in engine.step():
            produced[out.request_id] = len(out.outputs[0].token_ids)

    edit_ms = None
    if args.mode == "edited":
        edit_ms = 1e3 * edit_all(engine, request_ids, prompt_len, EDIT_SHIFT)
    for _ in range(WARM_STEPS):
        engine.step()

    for out in engine.step():
        produced[out.request_id] = len(out.outputs[0].token_ids)
    before = sum(produced.values())
    edits = 0
    edit_time = 0.0
    start = time.perf_counter()
    for step in range(args.decode_steps):
        if (
            args.mode == "edited"
            and args.edit_every
            and step
            and (step % args.edit_every == 0)
        ):
            # Alternate the shift so positions return to where they were.
            shift = -EDIT_SHIFT if edits % 2 == 0 else EDIT_SHIFT
            edit_time += len(request_ids) * edit_all(
                engine, request_ids, prompt_len, shift
            )
            edits += 1
        for out in engine.step():
            produced[out.request_id] = len(out.outputs[0].token_ids)
    elapsed = time.perf_counter() - start
    tokens = sum(produced.values()) - before
    assert not any(out.finished for out in engine.step()), "cap hit in window"
    engine.engine_core.shutdown()

    print(
        json.dumps(
            {
                "model": args.model,
                "mode": args.mode,
                "async_scheduling": args.async_scheduling,
                "num_seqs": args.num_seqs,
                "prompt_len": args.prompt_len,
                "decode_steps": args.decode_steps,
                "tokens": tokens,
                "decode_tok_per_s": round(tokens / elapsed, 1),
                "step_ms": round(1e3 * elapsed / args.decode_steps, 2),
                "first_edit_ms_per_req": None if edit_ms is None else round(edit_ms, 2),
                "timed_edit_rounds": edits,
                "timed_edit_s": round(edit_time, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
