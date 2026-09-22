# Branch `explore/decode-fast` — the retrieval-selection step

Worktree `/home/yx/IceCache/.claude/worktrees/decode-fast`, off `algorithm@ef228fa`.
Everything below is my own measurement unless the line says otherwise. Times come
from `quiet-run.sh`; CPU-sample shares come from a `quiet-run.sh` + `perf record`
run and are never quoted as times.

## 0. Verdict, up front

* The audit's arithmetic is confirmed. The DCI tree query really does cost ~2 ms
  live and the page_scan scan ~0.8 ms, and **78% of the DCI query is the descent
  itself plus the BLAS distance evaluations it performs** — there is no single
  wasteful block to delete.
* The three leads the brief nominated are all dead, each for a measured reason:
  the `calloc` is real but costs 0.5 ms/token (**0.47% of TPOT**); the OpenMP
  thread count is already at its optimum and provably does not change selection;
  `prop_to_visit`/`num_to_visit` are indeed dead parameters, and the two live
  levers cost F1.
* On the page_scan side the query is **not compute-bound at all** — it is
  per-op launch overhead. I made one selection-preserving change there (hoisting
  the per-call constants) and it survives: **predictions 200/200 identical, query
  p50 −8.75% (171/200 rows, sign test p=1.1e-25)**. That is worth ~0.84 ms/token,
  ~0.8% of decode TPOT — real, provable, and small.
* **A clean negative on the rest.** Nothing I found moves the §12.2 target; the
  branch's honest output is the breakdown plus a free 0.8% trim.

## 1. Where the DCI query's 2.03 ms goes (measured)

**Live path, cold vs warm** — `quiet-run.sh`, `dci_replay.py`, REPEAT=200,
hotpotqa index 0, `--threads 64 --n-reuse-layers 3 --backends dci
--max-new-tokens 3`, 24 groups = 4800 native calls. Calling the live query
`query()` again 200× per call site, recording each repetition separately:

| | n | p50 | mean | max |
|---|---|---|---|---|
| call 0 (live spacing) | 24 | **1.7991 ms** | 2.3078 | 12.898 |
| calls 1..199 (back-to-back) | 4776 | **0.8320 ms** | 0.8440 | 4.280 |

The live call is the cold one; it costs **2.16×** its own back-to-back cost. Tree
underneath: `num_points` 11712, `num_levels` 5, `num_leaves` 803–818/instance,
`token2node` `[8, 11712]`. Existing harness runs in the shared tmp dir put
`query.dci.p50_ms` at 1.77–2.00 across independent runs, so the audit's 2.03 is
representative.

**OpenMP thread-count rotation** — `quiet-run.sh`, `dci_replay2.py`, indices
0–7, `--max-new-tokens 32`, cold p50 per setting (the count is set around call 0
and restored immediately, so the model sees an unchanged runtime):

| OMP | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| cold p50 (ms) | 16.049 | 8.463 | 4.393 | 3.472 | 2.305 | 1.779 | **1.696** |

`output_mismatches []` — the thread count provably does **not** change the
returned ids. No free win here: the query is genuinely compute-bound and
near-ideally parallel, and 64 is already the production setting.

**Where inside the query** — `gpu-run.sh` (perf is not a time measurement),
`perf record -g --call-graph dwarf`, 17892 query samples = **26.10% of all
sampled CPU time**. Share of query samples by innermost frame:

| symbol | share |
|---|---|
| `vecmul` (BLAS distance work) | 48.79% |
| `dci_query_single_point_single_level` (the descent) | 28.85% |
| `msort_with_tmp` 5.46 + `dci_compare_data_idx_arr_dist` 4.04 (the E2 sort) | **9.50%** |
| `memmove`/`mempcpy`/`memset` | 4.96% |
| `dci_next_closest_proj` | 1.74% |
| `omp_set_lock` | 1.50% |
| `malloc`/`free`/chunk ops | ~1.2% |

So the descent and the distances it computes are ~78% of the query. The
unconditional `calloc` is in the 1.2% malloc band, which is what §2 settles.
The E2 full-sort degeneration measures **9.5%** here, not the 18% of doc 06 —
both are real, they are different runs, and 9.5% is the number I measured.

## 2. The three nominated leads, settled

**`calloc(max_leaves)` — confirmed present, refuted as a cost.** `dci.c:4082`
sits inside `dci_query_single_point` *above* the level loop (`:4125`), with the
matching `free` at `:4466`, so it is one calloc+free per call. `dci_query`'s
`num_queries == 1` branch (`dci.c:4788`) is what `py_dci_query` hits —
`py_dci.c:786` passes `num_query = shape[0] / num_query_head`, which is 1 for the
decode path — so it fires **32× per native query** (32 head-tasks), i.e. 384×
per token. Measured price, freshly, with `ctypes` on libc:

```
calloc( 818,1)+free = 1.273 us   -> 32x = 40.7 us/query -> 0.489 ms/token
calloc(2048,1)+free = 1.277 us   -> 32x = 40.9 us/query -> 0.490 ms/token
```

**0.49 ms/token against a ~104 ms decode TPOT is 0.47%.** It is not worth a
patch, a rebuild, or an A/B. The early-stop it guards is unreachable exactly as
the brief says (`prop_to_visit = 1.0`, `:1191`), but the allocation it would have
saved is worth half a percent of TPOT.

**`prop_to_visit` / `num_to_visit`** — confirmed dead, and dead is the safe
state: both are neutralised by clamps in `dci_query` against `num_points` and by
`prop_to_visit = 1.0`, so they cannot be used to trade accuracy for time without
the F1 cost doc 07 measured. Not retried.

**`num_neighbours` / `promotion_prob`** — live levers (10.6× / halving), both
already measured to cost F1 (docs 06/07). Not retried; the brief forbids it.

## 3. Where page_scan's 0.79 ms goes (measured)

**Live path** — `quiet-run.sh`, `ps_replay.py`, `--backends page_scan`,
REPEAT=50, 168 query sites: cold p50 **0.8344 ms**, warm p50 **0.7945 ms**,
`output_mismatches []`. The live call is *not* paying a stream-drain artifact:
it costs about the same warm.

Per-op CPU time inside the query (corrected: totals divided by the 50×
repetition count), against the 0.8344 ms cold p50:

| op | ms/query | | op | ms/query |
|---|---|---|---|---|
| `to` (H2D) | 0.0552 | | `__setitem__` | 0.0211 |
| `masked_fill` | 0.0473 | | `torch.full` | 0.0209 |
| `topk` | 0.0422 | | `torch.zeros` | 0.0203 |
| `torch.arange` | 0.0387 | | `gather` | 0.0193 |
| `bmm` (all the scan math) | 0.0373 | | `transpose` | 0.0103 |
| `reshape` | 0.0349 | | `expand_as` | 0.0089 |
| `cumsum` | 0.0319 | | | |
| `scatter_reduce_` | 0.0236 | | **sum** | **0.434** |
| `cpu` (D2H + sync) | 0.0225 | | | |

**Offline, no model** — `quiet-run.sh`, `ps_offline_profile.py`, a `PageScan`
built from `/tmp/pagprobe/keys.npz` (H=8, N=11770, n_pages=800) exactly as the
decode path builds it: plain wall **0.8289 ms/query**, `torch.profiler` says
**367 µs/query of GPU kernel time against 1278 µs/query of CPU-side op time**
(profiler-inflated, but the ordering is unambiguous). The whole scan is
`bmm` 34.8 µs CPU / 8.2 µs CUDA.

**It does not scale with the page count.** Across existing runs the live query
p50 is flat-to-inverted against `n_built` (141 pages → 0.8718 ms; 1078 pages →
0.7924 ms). And offline, at a fixed `n_pages`, deleting the single op that
scales with page count (`bmm`) would still leave ~0.62 ms.

The model that fits: **~20 CUDA ops × 20–45 µs of CPU dispatch each ≈ 0.83 ms**,
with the D2H sync on top. The query is launch-bound, not math-bound. That also
explains why the current profile spends 98.6 µs/query in `aten::nonzero` and
59.8 in `aten::index` (the `out[rows[m], pos[m]] = flat[m]` boolean-mask write
lowers to three `nonzero`s, each of which syncs the stream to size its output) —
and why `cudaStreamSynchronize` appears **6×** per query.

## 4. What I changed

`PageScan._query_device` rebuilt four tensors on every call that do not depend on
`q`. They depend only on `(H, budget, ratio, n_pages)`, all fixed once the
address space is reserved, and the query runs 12 anchor layers per decode token:

* `torch.arange(flat.shape[1]).expand_as(flat)` — candidate ranks
* `torch.arange(H).unsqueeze(1).expand_as(flat)` — row ids
* `torch.full((H, n_pages), ...)` — the dedup sentinel buffer
* `self._bias_t.unsqueeze(1) != 0` — the unbuilt-page mask

Commit `aa29df0` hoists all four into a cache (`_query_constants`) rebuilt on
staleness. Two subtleties, both real and both handled:

* `first` is **not** constant across calls. `scatter_reduce_(reduce="amin")`
  *reduces into* the buffer rather than writing it, so an earlier call's smaller
  rank survives at a page id this call did not rank and `keep` marks a
  non-first occurrence. I hit this in the probe (`shipped vs cached identical:
  False`) before I saw why. The caller re-arms it with `first.fill_(W)` every
  call; caching saves the allocation, not the fill.
* the mask is derived from `_bias_t`, which `insert()` mutates **in place** at
  two sites (`:850` rollback, `:863` publish). A mask cached on tensor identity
  alone would go stale and silently stop newly published pages from ever being
  selectable. It is keyed on a `_bias_ver` counter bumped at both sites.

The dedup structure is untouched — same kernels, same values, same order.

### Evidence that the selection does not move

`gpu-run.sh`, `verify_identical.py`: the patched path against the **verbatim
pre-patch body**, and against the numpy `query()` path, on the same object,
interleaved with real `insert()` calls so the in-place `_bias_t` write and its
invalidation are actually exercised:

```
checks=72  A!=B (patched vs pre-patch) = 0   B!=C (device vs numpy) = 0
pages emitted per insert: [1,1,1,1,1,1,1,1,1,1,1,1]   bias_ver=12
VERDICT: IDENTICAL
```

And end-to-end: **200/200 identical predictions** across all three A/B arms.

### Isolated size of the change

`quiet-run.sh`, `ps_cache_probe.py`, same offline `PageScan`, 40 queries:

| arm | ms/query |
|---|---|
| shipped `_query_device` | 0.6602 |
| hoisted constants (with the mandatory `fill_`) | **0.5350** |
| same ops with the D2H removed (floor) | 0.5604 |

**−0.125 ms/query, −19%** in isolation.

## 5. Live paired A/B

`quiet-run.sh`, `df_ab.sh`: `page_scan_compare.py`, Qwen3-4B / hotpotqa /
`--threads 64 --n-reuse-layers 3 --backends page_scan`, **one backend, one arm
per process**, indices 0–199, three arms alternating **base / opt / base** so a
monotone drift cannot land on one arm. The only thing swapped between arms is
`page_scan.py`. Each arm took ~21 min; worktree restored on exit.

| metric | base (2 arms) | opt | paired delta | opt faster | sign test |
|---|---|---|---|---|---|
| **query p50** | 0.8020 ms | 0.7318 ms | **−0.0702 (−8.75%)** | **171/200** | **p=1.1e-25** |
| TPOT | 104.362 ms | 102.537 ms | −1.825 (−1.75%) | 156/200 | p=6.8e-16 |
| TTFT | 5017.6 ms | 5009.6 ms | −8.1 (−0.16%) | 106/200 | p=0.44 |
| total | 5502.7 ms | 5487.4 ms | −15.4 (−0.28%) | 111/200 | p=0.14 |
| **predictions** | — | — | **200/200 identical** | | |

**Control arms** (base2 − base1, same code both sides, spanning the whole A/B
window): query p50 **+0.0119 ms**, TPOT **+1.92 ms**, TTFT **−170 ms**.

Reading it honestly:

* **The query-latency effect is solid.** −0.0702 ms against a same-code control
  drift of +0.0119 ms — a 6× margin — with 171/200 rows and p=1e-25. The live
  effect (−8.75%) is smaller than the isolated one (−19%), which is exactly the
  "kernels as latency padding" effect the project has measured before.
* **The TPOT number is not separable from drift at this n.** The point estimate
  is −1.83 ms, but the two *identical* base arms differ by +1.92 ms. Worse, the
  mechanism only accounts for 12 queries/token × 0.0702 ms = **0.84 ms/token**,
  so at most ~half of the measured TPOT delta can be the change; the rest is the
  drift. **Quote 0.84 ms/token, ~0.8% of TPOT, not the 1.75%.**
* **TTFT is the placebo.** The change cannot touch prefill, and it shows:
  p=0.44, 106/200 — no effect. That the harness reports a strong, specific
  effect on the one metric the change targets and nothing on the one it cannot
  touch is what makes the query result believable.

## 6. Rejected

* **Removing the boolean-mask write / rewriting the dedup.** On the do-not-retry
  list, and I did not retry it. The new evidence for *why* it is on that list:
  the write lowers to `aten::nonzero`, which sizes its output with a device→host
  copy, which is illegal inside a stream capture — so it is also what makes the
  whole query un-graphable. I tried to capture the query in a CUDA graph (the
  obvious way to attack a launch-bound 0.83 ms) and it fails on exactly this:
  `RuntimeError: CUDA error: operation failed due to a previous error during
  capture`. Combining the two — rewrite the dedup to make the query graphable —
  would be a *different* intervention from the one branch `explore/batch-knn`
  measured at 1.30–1.34× slower live, but it rests on the same premise that
  removing kernels helps a query whose cost the live queue absorbs. I stopped.
* **Cutting the per-op dispatch further** by caching `out`/`first` allocations:
  the `fill_`/`zero_` reset is mandatory (see §4), so what is left is small, and
  the live effect would be below the noise floor at any n I can afford.
* **The OpenMP thread count** as a free speedup: measured optimum is already
  the production setting (§1).
* **`num_neighbours` / `promotion_prob`**: real levers, both cost F1.

## 7. Bottom line

The page_scan query **is** reducible — but only by ~9%, because it is launch
overhead, and the only lever that could remove the launch overhead (a CUDA
graph) is blocked by the very code path the project already tried rewriting and
rejected. The DCI query is not reducible at all by anything selection-preserving:
it is a compute-bound, near-ideally-parallel descent whose only big blocks are
the descent itself and its BLAS distances.

**Mergeable:** yes, on correctness grounds — `aa29df0` is selection-identical by
construction and 200/200 identical predictions end to end, with a 6×-over-drift
improvement in the metric it targets. **But it is a 0.8% TPOT trim, not a
result.** It does not move §12.2 and should be merged, if at all, as part of a
batch of plumbing wins, never quoted as a headline.

What I would do next: the 26% of sampled CPU the DCI query occupies is the whole
baseline gap, and the only way to keep cutting decode is the *shared staging* the
other branch owns — which is where the brief itself says the remaining time is.

## 8. Provenance

All times from `quiet-run.sh`; the perf shares from `gpu-run.sh` (CPU samples,
not times); each arm of every A/B ran in its own process, one backend each.

| what | script | invocation |
|---|---|---|
| DCI cold/warm | `dci_replay.py` | `run_dci_replay.sh 200 0 3 a` |
| DCI OMP sweep | `dci_replay2.py` | `run_dci_omp.sh "0 1 2 3 4 5 6 7" 32 omp` |
| DCI perf shares | `agg2.py` | `perf record -g --call-graph dwarf` over the above |
| page_scan live ops | `ps_replay.py` | `run_ps_replay.sh` |
| page_scan offline | `ps_offline_profile.py`, `ps_cache_probe.py`, `ps_graph_probe.py` | |
| identity proof | `verify_identical.py` | `gpu-run.sh` |
| A/B | `df_ab.sh` / `df_analyze.py` | `df_ab.sh 3` |

Scratch lives in `/home/yx/.claude/jobs/497cc41a/tmp/df/`; no files outside the
worktree were modified. The large perf artifacts (`perf-dci.data` 1.1 GB,
`perf-replay-a.data` 573 MB, `script-replay-a.txt` 93 MB) were deleted after the
shares above were re-derived.
