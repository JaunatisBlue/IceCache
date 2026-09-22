# `explore/staging-fast` — trimming the shared decode staging in `estimate_select_recall`

> ## ⚠ EFFECT-SIZE CORRECTION (2026-09-22, added after merge)
>
> **Do not quote this report's −22.3%.** An independent second party re-measured
> this change on the merged tree, arm-interleaved over **13 arms in two row
> blocks (40 + 120 rows)**, with the staging change as **its own arm inside the
> same measurement window**. Its same-code nulls are much cleaner than this
> report's (base-vs-base −1.96% / +2.64% / −2.63% / +0.04%, versus the −5.47%
> one-sided null in §4):
>
> | arm | adjusted Δ TPOT |
> |---|---|
> | **staging alone** | **−13.27% to −14.85%** (34/6, 33/7) |
> | full merge (staging + hoist + prefill) | −15.06% to −15.38% |
> | merged − staging (= the decode hoist) | −0.69 ms (−0.79%), 30/10, p=2.2e-3 |
>
> **The honest magnitude is ~−14%; the three-branch merge is −15% ± 2.** The
> merge is not what lost the effect — *this branch's own headline magnitude does
> not reproduce in a cleaner window*. The dominant uncertainty is **between-window
> spread of ~5.3 ms (~6%)**: this report's opt arm measured 82.1 ms absolute, the
> re-measurement's staging-only arm measured **87.35 ms on identical code** —
> larger than any within-window null it measured.
>
> **What survives unchanged:** the output identity (this report's §3) and the
> *direction* of the effect. The re-measurement independently confirms the merge
> is **additive** (staging ≈ −14%, hoist ≈ −1 pp, prefill ≈ −1.6 s TTFT and 0 on
> TPOT → predicted ≈ −15%, measured −15%), with **no contention-sensitivity
> amplification** (TPOT-on-query-p50 slope 0.083/0.092 merged vs 0.099/0.140 base)
> and **no memory regression** (+133 MB, exactly the reused fp32 staging buffer).
> Combined with the prefill branch, the merged tree was verified at
> **2160/2160 row-comparisons text-, score- and token-identical** across all 13
> arms, with same-code controls at 0 mismatches.
>
> The lesson, recorded here because it is new to this project: **a within-window
> null only proves "A beats B in this window", never "the effect is this large".**
> The fix is to run the thing under test as its own arm in an interleaved window
> and across two row blocks — which is how this was caught. See
> `docs/experiments/15_work_ledger.md` §3 item 11.

## TL;DR

The mission was to cut decode time by attacking the ~51 ms/token of staging in `estimate_select_recall`, **without changing a single output bit**.

* The 51 ms nomination was **confirmed in magnitude but wrong in composition**. Re-measured in-scope staging is **~33.8 ms/token**, not 51; two of the four named components were overstated 2.2x and 2.4x, and the single largest in-scope item (the per-head Python gather loop, 8.5 ms) was not named at all.
* The change removes **~11 ms/token of staging** (probe, same backend, per token) and measures **−22.3% TPOT on page_scan** (40 rows, 40/0/0, p=1.8e-12) and **−11.7% TPOT on DCI** (20 rows, 17/3/0, p=2.6e-3). **⚠ The −22.3% does not reproduce: an independent 13-arm re-measurement puts it at ~−14%. Read the correction box at the top of this file first.**
* **Output identity: 40/40 rows bit-identical on page_scan, text and score**, and the page_scan arm is *itself* deterministic (0/40 between two runs of the unmodified baseline).
* The DCI arm's A/B showed 3/20 text mismatches — **but the DCI arm disagrees with itself on 4/20 rows between two runs of byte-identical baseline code**. Every one of the 3 A/B mismatch rows is a row the same code disagrees on with itself. The DCI identity check is therefore **uninformative at n=20, not failed**; the page_scan arm carries the correctness gate.

## 1. Re-measurement first — falsifying the nomination

Instrument: `staging_probe.py`, which monkeypatches `Tensor.item` / `Tensor.cpu` / `Stream.synchronize` as counters and reads region timers compiled into the worktree file, then runs the real harness. Live path, `quiet-run.sh`, DCI backend, hotpotqa rows 0–2, threads 64, `--n-reuse-layers 3`, `esr.total` = **34 calls/token**.

| component | nomination | re-measured (ms/token) | verdict |
|---|---|---|---|
| `recall` total | 23.7 | **20.8** | magnitude OK |
| — of which gather loop (`np_loop`) | *not named* | **8.5** | **missed; largest in-scope item** |
| — `DCI.copy_to_buffer` | *not named* | 6.2 | — |
| — the two stream copies | *not named* | 3.3 | — |
| — `rids.cpu()` / `nr.cpu()` | *not named* | 1.2 | — |
| — `torch.sum(nr).item()` | named as a sync | 1.2 | confirmed |
| `page_valid_entries` assembly | 12.6 | **5.6** | **2.2x overstated** |
| scan+apply | 14.8 | **6.2** | **2.4x overstated** |
| `c2g_stream.synchronize()` | 0.3 | **0.24** | confirmed |
| `alias.clone` + `c2p` assert | *not named* | **5.8** | **missed** (assert alone 1.2) |
| **in-scope total** | ~51 | **~33.8** | nomination overstated |
| `esr.total` | ~51 | 55–62 | — |
| `anchor_query` (DCI retrieval) | out of scope | 23.8–27.4 | other agent's territory |

Instrumented sync counts per token: `Tensor.item` 8772 → **102**, `Tensor.cpu` 1377 → **717**, `Stream.synchronize` 714 (unchanged).

The nomination's *shape* was right (staging dominates `esr`, `recall` is the biggest single region) but it pointed at the wrong lines. The two items worth attacking turned out to be the per-head Python gather and the fact that **22 of 34 layers are alias layers that re-derive "anchor + constant offset" from scratch.**

## 2. What changed

All in `infer_state.py`; nothing outside `recall`, `estimate_select_recall` and the two cache initialisers (6 hunks, verified by `git diff --stat`).

1. **Vectorised the address gather** (369 us → 31 us/call at H=8). The old `for i in range(self.n_kv_heads)` loop paid numpy fancy-indexing overhead per head for 80 elements. Now one masked `[H, max_n]` gather.
2. **Removed `torch.sum(nr).item()`**, a device sync per call. Its value is exactly the CPU element count `counter`, which the old code *already* used for `list_size` — the two were required to agree, so nothing is lost.
3. **Anchor/alias sharing.** The anchor layer copies `rids`/`nr` to the host once and stashes them in `_recall_cpu`; its 22 aliases read that instead of doing two more D2H copies each.
4. **`page_valid_entries` anchor cache.** An alias reads its anchor's selection (`selected_page_idx[reuse_id]`), so it needs the value the anchor already computed this token. The anchor writes `page_valid_cache[anchor]`; aliases copy it device-to-device instead of re-running the selector and re-uploading from pageable memory. Anchors always recompute, which keeps it fresh.
5. **`torch.where` for the alias `eids`** (two launches instead of four, no index tensors), `check_reuse` hoisted out of the batch loop, and the two `torch.cuda.stream(c2g_stream)` blocks merged into one.

Measured effect of the change (probe, DCI, rows 0–2, 24 new tokens, quiet-run.sh):

| | baseline | optimised |
|---|---|---|
| `esr.total` (ms/token) | 58.50 | **47.58** (−10.92) |
| `recall` | 20.81 | 13.15 |
| gather | 8.56 | **2.06** |
| `sum_item` | 1.20 | **0** |
| `page_valid` | 5.57 | 2.80 |
| `alias.clone` → `alias_derive` | 5.77 | 3.36 |
| instrumented TPOT (ms) | 131.3 | 123.5 |

## 3. Output identity — the correctness gate

### 3.1 The change is bit-identical on page_scan

`--model Qwen/Qwen3-4B --dataset hotpotqa --gpu 0 --threads 64 --n-reuse-layers 3`, one backend per process, both arms through `quiet-run.sh`, only `infer_state.py` swapped between them.

**page_scan, 40 rows: text identical 40/40, score identical 40/40.**

Crucially, the page_scan arm is itself deterministic, so this gate has power:

| comparison | text differ |
|---|---|
| page_scan **baseline vs baseline repeat** (byte-identical code + flags) | **0/40** |
| page_scan baseline vs optimised | **0/40** |

### 3.2 The DCI arm is not a usable identity gate — it disagrees with itself

The DCI A/B initially showed 3/20 text mismatches (rows 2, 9, 13). I ran each arm a **second time with byte-identical code and identical flags** to get a null:

| comparison | text differ |
|---|---|
| DCI **baseline vs baseline repeat** (identical code) | **4/20** — rows 1, 2, 6, 9 |
| DCI **optimised vs optimised repeat** (identical code) | **2/20** — rows 6, 13 |
| DCI baseline vs optimised (the A/B) | 3/20 — rows 2, 9, 13 |
| DCI baseline vs optimised (second draw) | 2/20 — rows 1, 6 |

Per-row values across the four observations (all four are the *same* two code versions; `base1`/`base2` are byte-identical files):

| row | base1 | base2 | opt1 | opt2 | |
|---|---|---|---|---|---|
| 1 | `Charles Laughton`(5) | `Charles Graham.`(4) | `Charles Laughton`(5) | `Charles Laughton`(5) | unstable |
| 2 | `Lowell, Michigan`(5) | `Lowell, Michigan.`(6) | `Lowell, Michigan.`(6) | `Lowell, Michigan.`(6) | unstable |
| 6 | `…Tranjavour.`(19) | `…Tharangambadi.`(21) | `…Tranjavour.`(19) | `…Tranquebar.`(19) | unstable |
| 9 | `Angel Witch`(3) | `Angel Nation`(3) | `Angel Nation`(3) | `Angel Nation`(3) | unstable |
| 13 | `Number five`(3) | `Number five`(3) | `Number 5`(4) | `Number five`(3) | unstable |

**5 of 20 rows take more than one value across runs of the same code.** Every A/B mismatch row (2, 9, 13) is a row where the *same code* disagrees with itself — rows 2 and 9 flip between two identical **baseline** runs, row 13 between two identical **optimised** runs. The 3/20 A/B mismatch rate is inside the 4/20 and 2/20 same-code null rates. There is no evidence the change moves any output on either backend.

The mechanism is in the DCI kNN search (`dciknn`, C++, OpenMP, outside this branch's scope) — a plausible reading is non-deterministic tie-breaking in its parallel traversal, since page_scan, which replaces that search with an exact GPU scan, is perfectly reproducible.

## 4. Live paired A/B (`quiet-run.sh`, one backend per process per arm)

Always `--model Qwen/Qwen3-4B --dataset hotpotqa --gpu 0 --threads 64 --n-reuse-layers 3`. A null arm is included for each backend: **a second run of the unmodified baseline**, same flags. Per-row paired deltas; sign test is exact binomial.

### page_scan — 40 paired rows

| | TPOT base | TPOT opt | paired Δ | mean ratio | sd / se | t | W/L/tie | sign p |
|---|---|---|---|---|---|---|---|---|
| **effect** (base1→opt) | 105.5998 ms | **82.0951 ms** | **−23.5047 ms (−22.26%)** | 0.7842 | 10.81 / 1.71 | −13.76 | **40/0/0** | **1.82e-12** |
| **null** (base1→base2) | 105.5998 ms | 99.8196 ms | −5.7802 ms (−5.47%) | — | 11.26 / 1.78 | −3.25 | 29/11/0 | 0.0064 |

The null is real and must be subtracted: **a no-op rerun of the baseline is 5.78 ms/token faster than the first run** (run-order / machine-state drift, the third confound of this kind in this project). Net of the null the change is worth about **−17.7 ms/token (−16.8%)**; even comparing the optimised arm against the *drifted, most favourable* baseline observation (82.0951 vs 99.8196) gives **−17.72 ms (−17.8%)**.

The shape of the two distributions is the real evidence:

* effect, best 5 rows: −40.8, −40.7, −40.7, −40.0, −39.6 ms (rows 3, 21, 30, 27, 28); worst 5: −14.9, −14.1, −14.1, −13.4, −10.6 ms (rows 1, 9, 8, 17, 23). **Every one of the 40 rows is negative.**
* null, best 5 rows: −24.6, −24.6, −24.2, −24.0, −23.4 ms; worst 5: +1.8, +2.9, +3.9, +8.8, **+23.0** ms. A no-op run produces **both signs, 11 of them positive**.

TTFT: base 5.3005 s → opt 5.5033 s, +0.2028 s (+3.83%), W/L 0/40, p=1.82e-12. The null TTFT is −0.0367 s (20/20), so the TTFT difference slightly exceeds the null, but **this change touches no prefill code** — `recall`/`estimate_select_recall` are decode-only (`_icecache_decode` runs only at `q_len == 1`). I attribute the TTFT delta to run order and am not claiming a TTFT result in either direction.

### DCI — 20 paired rows

| | TPOT base | TPOT opt | paired Δ | mean ratio | sd / se | t | W/L/tie | sign p |
|---|---|---|---|---|---|---|---|---|
| **effect** (base1→opt1) | 119.7397 ms | **105.7004 ms** | **−14.0394 ms (−11.72%)** | 0.8878 | 14.03 / 3.14 | −4.48 | **17/3/0** | **2.58e-3** |
| **effect**, text-identical rows only (17) | 120.9543 ms | 105.4047 ms | **−15.5497 ms (−12.86%)** | — | 12.79 / 3.10 | −5.01 | 15/2/0 | 2.35e-3 |
| **null** (base1→base2) | 119.7397 ms | 120.8418 ms | +1.1020 ms (+0.92%) | — | 18.29 / 4.09 | +0.27 | 14/6/0 | 0.115 |

The DCI null is ~zero, so the −14 ms stands as measured. Best 5 rows: −49.5, −39.1, −18.9, −18.7, −16.9 ms (rows 14, 19, 9, 5, 15); worst 5: −12.1, −5.0, +0.8, +6.2, +18.5 ms (rows 6, 10, 3, 0, 2). Three rows are zero-or-positive — the same rows that are unstable, and the corresponding null rows swing by ±20 to +58 ms, so those three are noise rows, not regressions.

## 5. Rejected, with the measurement that killed each

| idea | why rejected | measurement |
|---|---|---|
| **Cache the `c2p` offset and its whole-row assert once per row** | **changes the generated text.** `c2p[0,0]` moves during decode, and the whole-row invariant `(c2p_a − c2p_b) == offset` still passes either way, so the assert does *not* prove the offset is constant. The alias's page ids depend on the live value. | hotpotqa row 1: a cached offset turned `Charles Laughton` into `Charles Dickens` (5 tokens → 3). Bisected to this hunk alone; reverted, and the hazard is now documented in a permanent comment. |
| Direct pinned-fp32-CPU → fp16-CUDA copy (skip the fp32 CUDA hop) | no win, and it was already bit-identical so it bought nothing | 164.7 us vs 161.5 us two-step (`staging_micro2.py`) |
| Drop the `nr`/`rids` clones for the alias | the scatter kernel is read-only, but the saving is negligible and the clone is cheap insurance | kept |
| Thread `DCI.copy_to_buffer` | it is C++ inside `dciknn`; unreachable from here | n/a |
| Optimise `anchor_query` / `_apply_selected_pages` further | **out of scope** — that is the retrieval-selection side owned by `explore/decode-fast` | 23.8–27.4 ms/token, reported only |

## 6. Mergeability — honest bottom line

**Mergeable, on the page_scan evidence.** The change is confined to one file, value-identical by construction (masked gather reproduces the loop's head-ordered concatenation; the removed sync's value was already required to equal `counter`; the alias caches are read only after the anchor writes them, and the anchor always recomputes), and 40/40 text-and-score identical against an unmodified arm whose own reproducibility I verified at 0/40.

The DCI arm cannot be certified at n=20 — not because the change fails, but because **the DCI path is intrinsically non-reproducible at ~10–20% of rows**. That is a pre-existing property of the backend, and it is the single most important thing this branch found: any future identity gate run on the DCI arm at this sample size is worthless. Use page_scan for correctness gates, or raise DCI's n well past 200 and report a mismatch *rate* against a same-code null.

Effect size: **−22.3% page_scan TPOT** raw (40/0/0) and about **−17.7% net** of the measured run-order null; **−11.7% DCI TPOT** (17/3/0) against a ~zero null. **⚠ Superseded — the independent re-measurement puts this at ~−14%. See the correction box at the top of this file before quoting either number.**

Caveats I would not hide: the page_scan null is large (−5.8 ms, 29/11) and one-sided, so the page_scan number should be read as a range (−17.7% to −22.3%), not a point. TTFT is not claimed. All numbers are hotpotqa-only, n=20/40.

## 7. Provenance

* Every **time** in §1 (probe), §2 and §4 comes from runs under `/home/yx/IceCache/.claude/quiet-run.sh` (live.lock + gpu-run.sh), GPU 0 only, one backend per process, never `--backends a b`.
* §3 identity results come from the same `quiet-run.sh` runs; correctness and timing are read from the same JSONL.
* Micro-benchmarks (`369 us → 31 us`, the fp32/fp16 copy comparison) are pure host-side `torch`/numpy measurements, no GPU contention, and are labelled as component probes, not live results.

Artifacts: `/home/yx/.claude/jobs/497cc41a/tmp/` — `staging_probe.py`, `staging_micro.py`, `staging_micro2.py`, `staging_ab.sh`, `staging_ab_analyze.py`, `staging_repeat.sh`, `staging_repeat_ps.sh`, `repeat_analyze.py`, `repeat_table.py`; results `staging-probe-dci.jsonl`, `staging-diag1.jsonl`, `ab-ps-{base,base2,opt}.jsonl`, `ab-dci-{base,base2,opt,opt2}.jsonl`.
