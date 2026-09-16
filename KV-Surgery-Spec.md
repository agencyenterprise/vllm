# KV-Surgery Inference Engine — Initial Spec

Status: v0 design, pre-implementation. Companion to Serger (the research tool);
this is the throughput engine.

## Purpose

High-throughput LLM generation over *deliberately inconsistent* KV caches. The
motivating workload is a Connectome-style rolling-window agent context:

```text
[fixed prefix S][trajectory T1][trajectory T2]  ==>  [S][T2]            (drop + close gap)
[fixed prefix S][trajectory T1][trajectory T2]  ==>  [S][summary(T1)][T2]  (replace)
```

where T2 keeps its original KV (computed with T1 present) but is re-positioned
so the context is contiguous. The model's hard-to-verbalize state about T1
survives in T2's cache; the tokens do not.

Workload shape: generation-heavy. Roughly one edit per ~10^5 generated tokens.
Optimize the 99.99% path (stock decode on an edited cache), not the edit.

## The core decision

**Fork vLLM (current release; the engine core lives under the `vllm/v1/` package — "V1" is the architecture name, not a version pin). Keep its write-time RoPE and every attention backend's cache
layout exactly as-is. KV surgery is a rare, out-of-band pass over the cache
between scheduler steps.** No attention kernel is modified.

Rejected alternative: store K unrotated and apply RoPE at read time (Serger's
design). Cheap in FLOPs, but it means forking FlashAttention, FlashMLA /
FlashInfer-MLA, and the DSA indexer kernels, per architecture, per vLLM
release. Not worth it at one edit per 10^5 tokens.

## Why the ops are cheap: memory order is irrelevant

Paged attention kernels gather K/V through a block table and know nothing
about positions; position lives entirely in the rotation baked into stored K.
For a decode step there is no causal mask among cached tokens. Therefore:

- **Drop** = remove slots from the block table (+ optional rotation of later
  keys to close the gap).
- **Move** = rotate K of the affected slots. Nothing is physically moved.
- **Insert / selective prefill** = ordinary chunked prefill against a
  temporarily restricted block table (any sub-view works, since order is free).

Rotation is exact for K: `R(Δ)·R(p)·k = R(p+Δ)·k`. V is position-free. The
"inconsistency" (T2's hidden states having attended to T1) is preserved
automatically.

## Abstractions

- **Physical slots**: immutable once written (append-only).
- **View**: ordered list of slots + a position per slot. A request *is* a view.
  Positions are per-slot from day one (not a per-request offset) so overlap
  and fractional positions are representable later.
- **Forward pass**: (view, new token ids, their positions) → new slots.

View algebra (own module, unit-tested without a model): `restrict`, `splice`,
`shift`, `fork`, `materialize_block_table`.

## Ops (v0)

| op | implementation |
| --- | --- |
| `drop(view, span, close_gap: bool)` | block-table edit; `gather` to compact partial blocks; `rotate` later slots by −len(span) if close_gap |
| `shift(view, span, Δ)` | `rotate` only |
| `prefill(view_subset, tokens, positions)` | truncate block table to subset → stock chunked prefill → splice result back |
| `fork(view)` | new request sharing blocks via refcount (prefix-cache machinery) |
| `probe(view)` | forward pass, return logits (+ optional residual hidden states at chosen layers), commit nothing |

Batched: every op takes a flat list of (slot, Δ) / (src, dst) pairs across
requests. Ops are scatter-style kernels, not per-request loops.

Deferred (noted so the design leaves room): duplicate-span, KV import/export,
blend, per-layer-differing views (requires one `kv_cache_group` per layer),
per-slot attention bias (the one op that *would* need an attention-kernel
change).

## Kernels (Triton, memory-bound, trivially parallel)

1. `rotate(slots, deltas, layer_ids)` — pairwise 2D rotation over the rotary
   dims only. Per-slot Δ (fractional allowed).
2. `gather(src_slots → dst_slots)` — copy. Covers compaction, duplication,
   import/export, and (with scale) blend.

Applied per **position-bearing cache tensor**. On dense GQA that is the K
cache; on MLA it is the rope-k slice of the `[latent | rope-k]` row; on
GLM-5 DSA it is additionally the indexer key cache. Latents and V untouched.

### RoPE descriptor (per model, per layer), derived at load time from vLLM's

instantiated `RotaryEmbedding` and MLA layout constants — never a hand table:

- `inv_freq` (per layer: Gemma uses different bases for local vs global layers)
- pairing style (`is_neox_style` half-split vs interleaved)
- `rotary_dim` and byte offset of the rotary slice within the cache row
- backend cache layout (pin one attention backend per model; assert at load)
- cache dtype / quant scales

**Trap:** build `cos(Δ·inv_freq)`, `sin(Δ·inv_freq)` from `inv_freq` alone.
Never from vLLM's `cos_sin_cache` — YaRN folds `mscale` into it and a delta
rotation would apply it twice. The rotation must be pure orthonormal.

FP8 KV: rotation is dequant→rotate→requant and lossy. Bounded in a rolling
window (each span shifted ~window/span times). Mitigations if it ever matters:
keep rope-k in bf16 (~11% of an MLA row), or re-rotate from an unrotated master.
Not a v0 concern.

## Engine changes (vLLM fork, `vllm/v1/` engine core)

1. **Decouple logical position from slot count.** `num_computed_tokens` drives
   block allocation; the positions tensor for new tokens must come from the
   view, not from slot count. Touches request state, scheduler, model runner.
2. **Edit-op queue on the engine core.** `edit_kv(request_id, op)` enqueues;
   the core drains between steps. Scheduler-side: block-table surgery.
   Worker-side: kernels dispatched via the executor's collective RPC (KV is
   sharded across TP ranks; every rank applies the same op to its shard).
3. **Prefix caching off (or blocks marked unhashed) for edited requests.**
   Block hashes are content-derived; an edited block is a lie relative to its
   hash.
4. **Fixed prefix S is block-aligned by design.** Avoids copy-on-write on a
   shared partial last block when compacting T2 up against S.
5. **Hybrid models (Gemma 4 sliding-window layers):** ops apply per
   `kv_cache_group`. A drop of a span already evicted from a window layer is a
   no-op for that group; the shift Δ is applied uniformly.
6. Do not build on the KV-connector API — it is for injection, not in-place
   mutation of live requests.

Disable in v0, revisit later: speculative decoding (correctness is fine —
target verifies — but mirror ops onto the draft cache or accept an acceptance
dip after edits; remove the variable while the oracle stabilizes), chunked
prefill of new tokens while an edit is pending.

## Tier 1 models

| family | attention | position-bearing tensors | notes |
| --- | --- | --- | --- |
| Llama 3 | GQA, global | K | simplest; do first |
| Gemma 4 | GQA, local/global hybrid | K | per-layer `inv_freq`; per-group ops |
| GLM-5.x | MLA + DSA | rope-k slice, indexer keys | two tensors; pin the DSA kernel path |
| DeepSeek V4 | verify at implementation time | — | if it retains MLA (+DSA), it is the GLM-5 path with different modules |
| Kimi K2.x | MLA (`deepseek_v3` decoder) | rope-k slice | same as DeepSeek V3 path |

Explicitly not supported (fail loudly at load): Kimi K3 and Qwen 3 (recurrent /
Mamba-style layers have no addressable per-token KV), dynamic-NTK scaling
(cache not shift-invariant), ALiBi.

## Correctness oracle

Every edit that is *consistent* (drop the final k tokens; shift a span whose
hidden states cannot depend on anything disturbed; re-prefill against an
untouched prefix) must reproduce a from-scratch forward pass's next-token
logits to numerical precision. Inconsistent edits are compared against a
Serger reference on small models. "Correct" means "matches the edited-cache
forward," not "matches a clean recompute" — say so in tests.

Kernel-level invariants: `shift(Δ)∘shift(−Δ) = id`; `rotate` of an unchanged
prefix by Δ equals recomputed K at position p+Δ.

## Sequence

1. **Kernels standalone.** `rotate` and `gather` against a reference that
   recomputes from scratch, on Llama 3 (neox) and one MLA model (interleaved,
   rope-k slice). Include the FP8 path. No engine.
2. **Fork vLLM at the latest release, single GPU, Llama 3.** Position/slot decoupling, per-slot
   view positions, edit-op queue, prefix caching off. `drop` and `shift` only.
3. **Oracle harness.** Consistent-edit equality tests as CI. Add `prefill`
   against a restricted view (the summary flow), `fork`, `probe`.
4. **TP, MoE, MLA targets (GLM-5.x, Kimi K2.x, then Gemma 4, DeepSeek V4).**
   RoPE descriptor derivation from vLLM modules; per-group ops; then
   throughput work — confirm decode tok/s on an edited cache matches stock
   vLLM within noise, which is the whole point of the design.
