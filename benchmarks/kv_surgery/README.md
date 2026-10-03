# KV surgery benchmarks

`benchmark_edited_decode.py` checks the design's central claim: decoding on an
edited KV cache runs at stock speed, because the attention kernels are
untouched and an edited cache is just another block table.

## Results (2026-10-02, one H200, fork of v0.29.0 at 98dff2a81d, 64 requests x 2048-token prompts, 64 timed decode steps)

Each edited request had 64 tokens dropped from the middle (gathers and
re-rotates the tail) and its whole view shifted by 32. Edited prompts start
64 tokens longer, so both modes decode over the same context length.
`edit ms/req` is the wall time of both edits for one request, including the
worker RPC and the request's re-admission (the inspect call that sizes them
is not timed).

| model | mode | async sched. | decode tok/s | step ms | edit ms/req |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | stock | off | 5626 | 11.38 | |
| Llama-3.1-8B-Instruct | edited | off | 5632 | 11.36 | 3.8 |
| Llama-3.1-8B-Instruct | stock | on | 6699 | 9.55 | |
| Llama-3.1-8B-Instruct | edited | on | 6707 | 9.54 | 3.8 |
| Moonlight-16B-A3B-Instruct | stock | off | 6270 | 10.21 | |
| Moonlight-16B-A3B-Instruct | edited | off | 6180 | 10.36 | 3.3 |
| Moonlight-16B-A3B-Instruct | stock | on | 7922 | 8.08 | |
| Moonlight-16B-A3B-Instruct | edited | on | 8012 | 7.99 | 3.3 |

Edited decode matches stock within run-to-run noise: Llama within 0.2%,
Moonlight within 1.5% either way (its stock sync throughput alone moved 1.8%
between this run and an earlier one). An earlier version of the script
measured edited runs over contexts 64 tokens shorter than stock, which made
them look 1-2% faster. Each table row is a separate run in a fresh process.

Under async scheduling the core drains its batch queue before the first
`edit_kv` or `fork_kv` call between two steps, or `inspect_kv` of a request
with a batch in flight (later calls in the same gap find it empty), so a
round of edits empties the pipeline. The next `step()` submits a new batch
before it returns the first held output, so the GPU idles from the drain to
that step; with the default queue of two, the step that returns the second
held output runs at depth one.

The edit cost is per call, not per row: at this context length the kernels
move a few MB and the ~2 ms per `edit_kv` is RPC and bookkeeping. Editing all
64 requests every 16 steps (a pathological rate; the design assumes edits
about once per 10^5 generated tokens) halves throughput. If edit rates ever
matter, batch several requests' plans into one worker pass.
