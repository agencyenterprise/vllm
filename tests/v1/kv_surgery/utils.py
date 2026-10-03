# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def tp_sizes() -> list[int]:
    """Tensor-parallel sizes to run the GPU suites at, largest first.

    TP>1 workers are subprocesses and give their memory back on shutdown. An
    in-process (TP=1) engine may not: with the oracle fixture's reference
    dropped, after shutdown, gc and empty_cache, the sync Llama engine left
    2.5 GiB but Moonlight, Gemma 4 and async Llama left 29, 47 and 42 GiB of
    live tensors in the test process (measured on an H200). That is a leak
    whose holder has not been tracked down (async Llama's is about its whole
    KV cache), not expected behaviour. Running TP>1 first keeps it from
    starving the subprocess workers (TP=1 first fails TP=2 Moonlight and
    Gemma 4 worker startup for lack of free memory).

    TODO(kv-surgery): find what keeps those engines alive, then drop the
    ordering.
    """
    return [2, 1] if torch.accelerator.device_count() >= 2 else [1]
