# Branch D — exact-greedy-fast

Make `greedy_packed_pages(k, page_size, use_sim=False)` produce a **bit-identical**
partition much faster. Round 2 established that the greedy's partition is
load-bearing for accuracy and that no recall metric on this box can price a
partition change, so `torch.equal(new_packed, old_packed)` is the gate and a
score delta is not evidence.

## What changed

`use_sim=False` (the production batched path, `infer_state._page_scan_flush`)
now scans only the tokens that can still be chosen. The row for page `p` is
masked with `assigned` before the `topk`, so the `page_size * p` columns already
taken are read, scored and thrown away; the live columns are the only
candidates. The live keys are gathered (`[M, L, D]`) and the row is computed
against them, then scattered back into a full `[M, N]` buffer at the tokens' own
columns so the tensor `torch.topk` sees is identical to the one the old loop
handed it.

Rebuilding the live set costs a gather (read + write == two scans), so it is
amortised over a *span* of pages rather than done per page. Inside a span the
live array is a superset of the unassigned set; tokens assigned during the span
are still in it and are masked out of the compact row, exactly as the old loop
masks them out of the full row.

- `_greedy_packed_pages_live` — the new loop.
- `_greedy_packed_pages_rescan` — the old loop, kept verbatim as the reference
  the self-test holds the new one to (and as the escape hatch).
- `live_compact_span(n_built)` — default span, `sqrt(2 * n_built)` (44 pages at
  N=16072, P=16). The surviving scan fraction is `(n + s)(s + 2) / (2 n s)`:
  45% saved at n = 1005, flat within a point or two either side of the optimum.

Commits: `0be237e` (the live path), `1a1b332` (the CUDA-only dispatch),
`b5be38e` (docstring). One file, 253 insertions, 0 deletions.

A reviewer will notice that the `use_sim=False` branch of the shipped inline
loop is now unreachable but still present (the dispatch returns first). It was
left in place to keep the diff to one added line plus new functions; removing it
is a pure cleanup with no behaviour change.

## The gate: bit identity

The reference arm is the pristine module extracted from the base commit
(`git show 2ea0cab:IceCache/source/icecache/page_scan.py`, 1203 lines) loaded
through importlib, never re-typed, so the comparison cannot be vacuous.

    gpu-run.sh <wt> python /home/yx/.claude/jobs/497cc41a/tmp/d_identity.py

Every line below is `torch.equal(got, want) == True` with `diff=0` entries
(90,000+ tokens compared per real shape):

| case | spans |
|---|---|
| real L2 prefill keys, M=8, N=11770 | default, 16, 45, 128 |
| real L20 prefill keys, M=8, N=11770 | default, 16, 45, 128 |
| real L2 x 12 layers, M=96, N=11770 | default, 16, 45, 128 |
| synthetic M=96, N=16072 | default, 16, 45, 128 |
| synthetic M=96, N=16072, skewed norms | default, 16, 45, 128 |
| forced ties M=16, N=4096 (duplicate keys, zero-norm pair, constant rows) | None, 1, 3, 16 |

plus the **public entry point** on every one of those cases, and a randomised
fuzz (`d_fuzz.py cuda:0 24`: random M/N/P/D, spans 1/2/3/random/1e6, duplicate
keys, zero keys, constant keys, coarse-rounded keys) — **24 cases x 5 spans OK**.

`page_scan._self_test()` passes fully, including the new per-shape checks.

### Why it is exact

A gemv accumulates `sum_d A[d, n] * x[d]` independently per column `n`, so
removing columns cannot move a bit of the columns that stay. That is a property
of the *kernel*, not of the math, and it is checked directly:

    python d_gemv_eq.py cuda:0     ->  GEMV_IDENTITY OK

`bmm(s, k_live^T)` reproduces the corresponding columns of `bmm(s, k^T)` for
L in {16072, 14472, 8036, 4096, 3272, 997, 16} at M=96 N=16072 D=128 and at a
tie-heavy M=16 N=4096 shape: 0 of 1542912 entries differ, max relative error 0.

### Where it is not exact: CPU

CPU BLAS chooses its blocking from the operand shape, so the same property does
**not** hold there: on the fuzz case that first exposed it (M=5, N=584, P=9,
D=14, live L=403) the same column, gathered out of the full `bmm` row and
recomputed against the compacted operand, is off by one ULP on 6 of 2015
entries (max relative error 1.0e-07) — and 1275 of 2015 when the transposed
operand is materialised rather than strided. That is enough to flip a tied
`topk`, and it is not a bookkeeping bug: same column set, same order, same seed,
values gathered from the full row. The fuzz reports it as 5 failures out of 120
runs, all on tie-heavy inputs.

So `use_sim=False` dispatches to the live path **only on CUDA**; on CPU it keeps
the old full-width loop and its exact bytes (`1a1b332`). The dispatch is
bit-identical on CPU over all 24 fuzz cases, and the self-test now checks the
public entry point on CPU and the live path itself on CUDA.

## Timings (all from quiet-run.sh, one implementation per process)

### Offline, greedy only, production shape M=96 N=16072 D=128

`d_pair.sh`: ref / new(default span) / new(span 128) alternating, 4 rounds,
median of 5 reps per process:

| round | ref | new (default) | new (span 128) | speedup |
|---|---|---|---|---|
| 1 | 799.1 | 639.9 | 603.8 | 1.249 / 1.323 |
| 2 | 821.6 | 638.6 | 636.6 | 1.287 / 1.291 |
| 3 | 796.8 | 637.0 | 595.0 | 1.251 / 1.339 |
| 4 | 819.7 | 596.2 | 600.0 | 1.375 / 1.366 |
| median | **809.4** | **637.8** | 601.9 | **1.29 / 1.33** |

Between-process drift is the dominant error here (identical `ref` code measured
796.8 and 821.6 ms in different processes; `new` lands in a ~600 ms or a ~638 ms
mode per process), so the honest statement is **~1.3x on the greedy, ~170 ms
saved**, and the default span and span 128 are indistinguishable.

Span sweep (same protocol, median of 3):

| shape | span 16 | 32 | 64 | 45/128 |
|---|---|---|---|---|
| M=96 N=16072 | 680.3 | 642.8 | 629.4 | 598.9 / 596.9 |
| M=96 N=11770 | 433.5 | 411.8 | 405.1 | 376.2 / 374.0 |
| real M=8 N=11770 | 285.7 | 278.3 | 273.1 | 218.2 / 209.7 |

The curve is flat between 32 and 256 and rises steeply below it (span 8: 706 ms;
span 1, i.e. a compaction every page: 1741 ms, worse than not compacting at
all). `sqrt(2 n)` sits at the flat optimum and is what ships.

### Live paired A/B

4 arms (after/before/after/before) x 10 hotpotqa rows, `--model Qwen/Qwen3-4B
--dataset hotpotqa --gpu 0 --threads 64 --n-reuse-layers 3 --backends
page_scan`, each arm its own process through quiet-run.sh, only `page_scan.py`
swapped between arms (restored on exit).

Per-row TTFT (s), the two adjacent before/after pairs:

| row | before | after | ratio | row | before | after | ratio |
|---|---|---|---|---|---|---|---|
| 0 | 5.114 | 5.032 | 0.984 | 0 | 4.967 | 4.896 | 0.986 |
| 1 | 7.051 | 6.862 | 0.973 | 1 | 7.032 | 6.781 | 0.964 |
| 2 | 7.122 | 6.910 | 0.970 | 2 | 7.051 | 6.987 | 0.991 |
| 3 | 6.791 | 6.636 | 0.977 | 3 | 6.686 | 6.743 | 1.008 |
| 4 | 6.074 | 5.952 | 0.980 | 4 | 6.045 | 6.026 | 0.997 |
| 5 | 6.231 | 6.264 | 1.005 | 5 | 6.332 | 6.189 | 0.977 |
| 6 | 6.512 | 6.420 | 0.986 | 6 | 6.558 | 6.376 | 0.972 |
| 7 | 0.759 | 0.776 | 1.023 | 7 | 0.778 | 0.777 | 0.998 |
| 8 | 5.051 | 4.997 | 0.989 | 8 | 5.180 | 4.930 | 0.952 |
| 9 | 3.659 | 3.591 | 0.981 | 9 | 3.578 | 3.563 | 0.996 |
| mean | 5.437 | 5.344 | 0.9869 | mean | 5.421 | 5.327 | 0.9842 |

Combined over 20 paired rows: **delta mean -0.0932 s, median -0.0766 s, sd
0.0925, faster on 17/20, paired t = -4.51, ratio mean 0.9855**.

**Accuracy: scores exactly equal on 20/20 rows (max |diff| 0.0) and generated
tokens identical on 20/20** — the required signature of a bit-identical
partition. (Row 7 has a ~0.78 s TTFT and shows no delta: too short to build.)
The live 93 ms is smaller than the offline 170 ms because the live build runs at
the deployment's own M (3 reuse layers) and chunk N, and the greedy is only one
component of a ~6.15 s TTFT.

`retrieval_stats.build_ms`, `n_pages_built` and `queries` are `null` on every
row, so the harness offers no second, build-level view of the change (and per
the project's note `build_ms` excludes the greedy anyway).

## What was tried and rejected

- Live-set compaction on CPU — gated out, see above.
- `span=1` (a compaction every page): 1741 ms at M=96 N=16072, 2x slower than
  not compacting at all.
- Pointer-based seed selection and other op-count reductions were not pursued as
  the primary lever: the profile puts the row bmm at 518 ms of 823 ms (793 GB at
  1.53 TB/s, bandwidth-bound), so only reading fewer bytes moves the total;
  `topk` is 120 ms and everything else ~60 ms.

## Mergeability

The change is bit-identical where it runs (CUDA, the production device) and is
switched off where it is not (CPU), so the accuracy risk is zero by construction
rather than by measurement. Cost: ~250 lines of new code in `page_scan.py`, one
extra `[M, N]` buffer, and a gather allocation per span. It is a 1.29x
improvement on one component of TTFT, worth 1.5% end to end; the honest
alternative to merging it is to leave the greedy alone.

Not merged into `algorithm`; committed on `explore/exact-greedy-fast` only.
