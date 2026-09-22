# Adversarial review, pass 2: what would have to be true, and my pre-registered rebuttal

Scope: the replacement direction ("用PAG替换掉DCI" + "给每个head维护一棵树...借鉴PAG的方法去构建，
然后搜索"), reviewed against v1's critique (`paged_pag_critique.md`) and the new facts supplied.

Rules I held myself to: every claim is traceable to a file:line, to a number in the new-facts list,
or to a measurement I made and report inline. Measurements I made are marked **(measured here)** and
are **not timing measurements** — I ran no clock. Script: `/tmp/pagprobe/centroid_probe.py`,
output `/tmp/pagprobe/centroid.json`. Existing artifacts re-read: `pag_stage1_*.jsonl`,
`/tmp/pagprobe/{pages.py,paired.py,ceiling.py,*.json}`.

---

## 0. Verdict

I concede one of my two cost arguments and hold the other. The TPOT case in v1 §2.3 was **mostly an
artifact of the 8-worker pool**: of the measured 423 ms/token PAG−DCI gap, the query path explains
only 42.0 ms and insert another ~9 ms steady-state; the rest was serialization that the one-thread-
per-tree model removes, and I was also wrong about *why* builds are serial (`pag_bindings.cpp:94`
releases the GIL; it is the Python `for head` loop at `pag_retrieval.py:50-63`, which is deletable).
I stand by the prefill case, which is now *sharper*: 97.3% of the measured 202.02 s TTFT is build.

The line has moved from 393x to **10.4x at the most favourable operating point, 53.9x at the
token-level floor, and 202–359x at the operating point the current code is pinned to.** But the
10.4 s figure is the price of a **retrieval** index over pages that must already exist; the user's
goal is the **construction**. Those are different indexes with opposite requirements on the one
shared dial PAG exposes, and that — not raw speed — is the constraint I now believe is unsatisfiable.

---

## 1. What the new facts change, and what they do not

### 1.1 Conceded: the TPOT gap was mostly the pool

Decomposition of the measured TPOT gap (both numbers from `pag_stage1_pag_fast.jsonl` and
`pag_stage1_dci.jsonl`, `retrieval_stats.query`, n=204 samples each, Qwen3-4B, 34 retrieval layers):

| component | DCI | PAG | Δ per token |
|---|---|---|---|
| query p50 per layer (all 8 KV heads) | 1.940 ms | 3.176 ms | 34 × 1.236 = **+42.0 ms** |
| measured TPOT | 190.4 ms | 613.1 ms | **+422.7 ms** |

So **381 ms of the 423 ms gap was not the ANN at all** — it is the shared-pool serialization v1 §2.1
identified. Under one-thread-per-tree that term should largely vanish, and the modeled TPOT becomes
≈ 190 + 42 + insert ≈ 240 ms, i.e. **1.26x, not 3.2x**. I withdraw the 3.2x as a property of the
design; it was a property of the harness.

I also withdraw the implication in v1 §2.2 that PAG's build is *architecturally* serial.
`pag_bindings.cpp:94` wraps `build_impl` in `py::gil_scoped_release` (as do `search_impl`:64 and
`insert_batch_impl`:161). The comment at `pag_retrieval.py:108-110` — "pag.Index.build holds it,
which is why builds stay serial" — is **wrong about the mechanism**: builds are serial because
`pag_retrieval.py:50-63` loops heads in Python on one thread. That loop is the thing to delete, and
deleting it is legitimate. This makes the user's 64-thread measurement physically reachable.

A fact that now dominates everything: in the measured run, **196.6 s of the 202.02 s TTFT (97.3%) is
PAG build**, mean 5.78 s/layer over 34 layers (`retrieval_stats.pag_layers[*].build_ms`). TTFT ≈
build. Every prefill argument is a build argument.

### 1.2 Not conceded: one dial, two opposite jobs

PAG exposes exactly one breadth knob, `max_search_k` (msk), and it is hard-wired into both the
build and the search:

* **search is capped by it.** `PAG/pag.cpp:1486-1487` (`"max_search_k must be at least topk"`) and
  `PAG/pag.cpp:792-804` (`"Requested top_k exceeds this index's build-time max_search_k"`).
  So `top_k <= msk` always.
* **build is linear in it.** `pag.cpp:1144` sets `pif_entries_per_bucket = ComputeWorkingSetSize(msk)
  = max(10, msk)` (`pag.cpp:140-145`), and `pag_index_core.h:723-727` allocates
  `65536 × max(10,msk) × sizeof(PIFEntry=8B)` per head (`table_cols = (2·8)²·(2·8)² = 65536`,
  `assert(pif_projection_width_ == 8)` at `pag_index_core.h:841`). Measured, user's table:
  0.1983 s/tree at msk=16 → 0.2730 at 32 → 1.3194 at 128, i.e. **6.65x build for 8x msk**;
  8x msk for 16x N is also roughly linear, so build ≈ a + b·N·msk.

Now the two jobs:

**(a) Search wants msk >= rho·budget.** To return `budget = 12` distinct pages you must hand the
aggregator more than 12 tokens. The shipped code encodes its own estimate at
`pag_retrieval.py:106`: `top_m = max(initial_factor·budget, budget·page_size//2)` = **96** for
(12, 16). So the shipped configuration cannot run below `msk = 96`, and ships `msk = 128`
(`longbench_pred.py:--pag-max-search-k default 128`). Measured price: **358.9 s/prompt** (user's
table) / **202.02 s TTFT** (`pag_stage1_pag_fast.jsonl`). These two numbers describe the same
configuration and differ by 1.78x — §4, shape D, requires that to be reconciled.

**(b) Formation wants msk ≈ N.** This is new and it is the strongest thing in this document.
Forming pages by "greedy expansion from a seed" over PAG's neighborhoods requires each seed's
candidate list to still contain unassigned points after earlier pages have consumed them. I have
both ends of the curve measured:

| candidate list per seed | pages produced (N=11712) | recall@12 L2 / L20 | source |
|---|---|---|---|
| exact top-64 | **9844** (1.19 tokens/page) | **0.620 / 0.789** | `paired.py`, `greedy_pages_L2.npy` (max id 9843) |
| exact full row | 732 (16.0 slots/page) | **0.832 / 0.883** | `pages.py`; re-measured here as 0.8317 / 0.8827 |

At 64 candidates the partition **fragments to 9844 pages** and recall collapses to
**0.620/0.789 — that is DCI (paired: 0.635/0.705) or slightly below it at L2.** And PAG's
`top_k = 64` is an *approximate* top-64, so it is strictly worse than the exact top-64 row.

So: search wants msk ≈ 96–128, formation wants msk ≈ N ≫ 128, build is linear in msk, and **msk is
one variable.** The formation requirement is the larger one, which makes formation build O(N²) per
head — the 393 s floor of v1 is not a conservative worst case, it is the *floor* of the only
configuration in which PAG can form a good partition.

### 1.3 New measurement: the affordable operating point pays an 8–11% recall tax

The `N=732, msk=16 → 10.4 s` figure indexes *pages*. A page-level index must score a page with **one
vector**. But the page score IceCache actually uses, and the one all the recall numbers above are
computed with, is `S(P,q) = max_{i∈P} q·k_i` (`pag_retrieval.py:176-195`). I measured the gap on the
best partition I have (exact-kNN greedy, 732 pages of exactly 16 slots, 8 KV heads, 128 queries/head
= last 32 prompt positions × 4 q-heads, layers 2 and 20, mean over heads):

| page scoring | L2 | L20 |
|---|---|---|
| `max_{i∈P} q·k_i` (what the code does) | 0.8317 | 0.8827 |
| `q·mean(P)` (what a page-level index gives you) | **0.7400** | **0.8123** |
| relative loss | **−11.0%** | **−8.0%** |

**(measured here, `/tmp/pagprobe/centroid_probe.py`)** The max-score column reproduces v1's
0.832/0.883 exactly, which cross-validates the harness. Within-page key-norm CV is 0.029 (L2) /
0.018 (L20) against global 0.121 / 0.029, so pages are ~4x more norm-homogeneous than the corpus but
not homogeneous — the mean is still pulled by whichever member has the largest norm, independent of
the query direction. A *learned* page vector would recover some of this, but that is a training
procedure, not PAG's MIPS index.

This does not kill the page-level shape (0.740/0.812 still beats DCI's paired 0.635/0.705), but it
means **every recall claim made for a page-level PAG index is overstated by 8–11% relative unless
the design reports this number.**

### 1.4 Unchanged: there is still no accuracy evidence

The entire Phase-1 quality record is two samples with 7 and 5 generated tokens, scores identical to
DCI (0.3333, 1.0) — `pag_stage1_*.jsonl`, `generated_tokens` field. The nearest real
partition→accuracy datum in the repo is `experiment/07_mdci_promotion_sweep.md:38-43`: Qasper F1
26.93 → 31.23 (+4.3) when `ratio_1` goes 0.01 → 0.05, on **8 examples with no confidence interval**,
while leaves go 283 → 388. So partition changes *can* move accuracy by several points, and a
0.635→0.867 recall change is much larger than that one — but nothing in this repo establishes the
slope. This is the axis the user's acceptance criterion actually names, and it is untested.

---

## 2. Which constraint is unsatisfiable, and the minimal relaxation

**Unsatisfiable:** *"PAG both forms the pages per KV head and searches them."* Not on speed —
structurally. Formation needs `msk ≈ N` (§1.2b); the search-side cap `top_k <= msk` and the linear-
in-msk build mean that same `msk` then costs 264–550 s per prompt (interpolating the user's table at
the measured 0.0109 s/msk-unit slope above msk=32, for 272 trees). Page-level indexing escapes the
cost only by indexing pages that something else already formed.

**Minimal relaxation, in two flavours — the design must pick one and say so:**

* **R1 (weak):** PAG indexes page representatives; a non-PAG component forms the pages. This is
  coherent and cheap (10.4 s), and it is the only version whose cost is within one order of
  magnitude. Its price: PAG becomes decorative. 732 page vectors × 128 dim × 8 heads = 3.0 MB per
  layer; a brute-force scan is ~0.2 ms, while PAG's own PIF table costs
  `65536 × max(10,16) × 8B = 8 MiB` **per head** at msk=16 (`pag_index_core.h:723-727`) = **3.0 GiB
  across 272 heads**, plus 65536 `std::mutex` per head (`pag_index_core.h:727`). The index's
  bookkeeping is ~1000x the indexed data. §4-A2 attacks this directly; v2 needs an answer that is
  not "it matches the user's words".
* **R2 (strong):** keep PAG forming the pages, accept O(N²) build. Then TTFT must be struck from the
  paper's claims with the 1.0 s DCI number printed next to it. That is a **user decision, not a
  design decision** — v2 must present it as such rather than burying it.

I do not think R2 is acceptable, and I think R1 is honest but makes the user's stated goal only
nominally satisfied. I would rather the design agent argue me out of that than have v2 split the
difference silently.

---

## 3. The pass/fail line: numerically checkable predicates

Each row: what to measure, what passes, what fails. P2, P3 and P6 need **no timing** and cost
minutes — they are the ones I most want answered first, because P3 can close the direction outright.

| # | Predicate | Measurement | Pass | Fail |
|---|---|---|---|---|
| **P1** | Build wall clock | One number: wall seconds to build all 272 trees for one prompt, idle box, no other load | ≤ 3.0 s (3x DCI's 1.0 s) | > 10.0 s |
| **P1b** | Ambiguity resolution | Is the user's "full prompt" column (10.4 s at page-level msk=16, 358.9 s at token-level msk=128) a **wall clock** or `272 × per-tree`? | A single stated wall-clock number | Reporting per-tree times as if they were prompt latency |
| **P2** | Coverage ratio `rho` | Smallest `top_k` such that PAG's top-K covers ≥ 12 distinct real DCI pages; report p50/p90 over (layer, head, query) | p90(rho) ≤ 32 → 74.3 s/prompt | p90(rho) > 64 → ≥ 264 s/prompt |
| **P3** | Formation under PAG's own neighborhoods | Build PAG at msk ∈ {p90(rho), 64, 128}; greedy-expand pages using `index.search(seed, top_k=msk)`, seeds by descending ‖k‖; report pages produced, slots/page, and paired recall@12 | pages ≤ 900 **and** recall ≥ 0.700 (L2) / 0.789 (L20) | pages > 1500 **or** recall ≤ 0.635 / 0.705 |
| **P4** | Page-level proxy tax | If v2 indexes pages: recall@12 under centroid vs max on the same partition | centroid ≥ 0.635 / 0.705 | centroid < 0.635 / 0.705 |
| **P5** | Decode | TPOT ratio, same harness, **both arms with `--n_reuse_layers 3`** and prefetch on (`run_longbench.sh:27`), ≥3 samples, equal generated lengths | TPOT_pag ≤ 0.95 × TPOT_dci | ≥ 1.0 × |
| **P5b** | Insert path parallelism | Where `insert_batch` runs and on how many threads; amortized ms/token | ≤ 9 ms/token | > 20 ms/token |
| **P6** | Accuracy oracle gate | Swap DCI's page assignment for the co-occurrence partition (0.867/0.879, `ceiling.py`) with everything else byte-identical; ≥2 LongBench tasks × ≥20 samples | mean score +2.0 absolute, paired bootstrap 95% CI excl. 0 | ≤ +1.0 → accuracy axis closed; nothing PAG does can reopen it |
| **P7** | Reconciliation | Explain 358.9 s (user table, msk=128) vs 202.02 s (measured TTFT, ef_construction=50) for the same configuration | A stated cause | Unexplained 1.78x |

**Why P3 is decisive and why the prior is a fail.** `paired.py` already ran the msk=64 case with
*exact* candidates — an upper bound on what PAG@64 can do — and got 9844 pages and 0.620/0.789.
PAG's approximate top-64 cannot beat exact top-64. If P3 confirms the fragmentation at PAG's own
msk, then PAG cannot form a partition as good as DCI's **independently of cost**, and no cost
argument is needed. If P3 surprises me and lands at 732 pages / 0.83+, that is the only evidence
that would reopen the cost question, and I will say so.

**Why P6 is the cheapest decisive test.** The co-occurrence partition is trained on the query
distribution itself, so 0.867/0.879 is an upper bound on *any* query-independent partition,
PAG-derived or otherwise. If the upper bound does not move the benchmark by +2 points, no partition
does, and the accuracy axis — the axis the user's acceptance criterion names — is closed for the
whole programme. That result would also be a paper-worthy negative finding rather than a wasted run.

---

## 4. Pre-registered rebuttal

### Shape A — "One thread per tree + page-level PAG index at msk=16: 10.4 s build, query wins, TPOT wins."

**A1 — you owe the proxy tax.** Report §1.3. Any sentence of the form "PAG pages are as good, at
8–11% less recall" is the honest one; "PAG pages preserve the max-aggregation recall" is false by
0.8317→0.7400 (L2) and 0.8827→0.8123 (L20). If v2 quotes recall numbers obtained with a max-based
page score and claims them for a page-level index, that is the same conflation v1 §4 attacked.

**A2 — name the thing that forms the pages.** If it is DCI, you have kept DCI and added 10.4 s to
build it twice. If it is an exact N×N matmul, then PAG is decorative: the indexed object is 3.0 MB
of page vectors per layer, a brute-force scan is ~0.2 ms, and PAG's PIF table alone is 8 MiB/head =
**3.0 GiB** at msk=16 (`pag_index_core.h:723-727`) — the bookkeeping is orders of magnitude larger
than the data. "It matches the user's wording" is a scope justification, not a performance one, and
v2 should say which of the two it is offering.

**A3 — the 10.4 s number is the retrieval index, and the goal is the construction.** "借鉴PAG的
方法去构建" is about construction. A v2 that silently narrows to retrieval-only has changed the
assignment. If that is the intent, state the scope change in the first paragraph so the user can
reject it, rather than letting it be discovered later.

**A4 — 3.0 GiB is not free either.** At msk=16 the PIF tables are 8 MiB/head × 272 = 3.0 GiB
(`pag_index_core.h:723-727`), against a box whose other budgets are 40 GB GPU / 80 GB CPU
(`pag_stage1_compare.py` benchmodel defaults). v1 §2.2 costed msk=128 at 17 GiB; the "cheap" point
is not zero. Report it.

### Shape B — "Keep token-level, msk=128; TTFT is not in the acceptance criterion, so 202 s is fine."

**B1 — say it with the number next to it.** 97.3% of TTFT is build (196.6 s of 202.02 s). "TTFT is
not a criterion" therefore means "the paper's prefill column regresses 202–359x against DCI's 1.0 s
and we accept that". Put DCI's 1.0 s in the same table as PAG's number. If the user still says yes,
that is their call — but it must be an explicit call, not an inference from a criterion quoted out
of context.

**B2 — msk=128 is not chosen, it is forced.** `pag_retrieval.py:106` sets `top_m` to
`budget*page_size//2 = 96` because 12 pages × 16 slots cannot be filled from fewer candidates. If P2
shows p90(rho) ≤ 32, the design is paying 358.9 s for headroom nobody needs; if P2 shows
p90(rho) > 96, the shipped default is silently under-provisioned and the retry path
(`pag_retrieval.py:169-174`) is doing the work. Either way P2 is a prerequisite measurement, not a
tuning exercise.

**B3 — the query side still loses, and not because of PAG.** Measured per-layer p50: search 0.780 ms,
aggregate **1.470 ms**, per head (`pag_stage1_pag_fast.jsonl`, `pag_layers[*].search/aggregate`).
The numpy page aggregation is **46% of the query cost and nearly 2x the ANN search**. Even a
zero-cost PAG search leaves ≈ 2.4 ms/layer against DCI's 1.940 — a **loss**. A decode-speed claim
therefore requires deleting `np.unique`/`np.maximum.at`/`np.lexsort`
(`pag_retrieval.py:183-195`), which has nothing to do with PAG. If v2 claims TPOT on the strength of
the index, B3 refutes it with the run's own numbers.

### Shape C — "Hybrid: PAG forms pages from a coarsened/cheap index, or DCI forms and PAG searches."

**C1 — if PAG forms, P3 applies and the prior is a fail.** At 64 candidates the partition is 9844
pages / 0.620/0.789, from *exact* candidates (`paired.py`); PAG's top-64 is approximate, hence
strictly worse. "Coarsened" (merge tokens into super-nodes, index the super-nodes) is page-level
indexing with extra steps and lands on A2.

**C2 — if DCI forms and PAG searches, count what is bought.** You still build the token-level PAG
index (202–359 s) on top of DCI's 1.0 s, you still inherit `msk ≥ top_k ≥ 96`, and the entire
purchase is the index's share of the query — `3.176 − 0.780 = 2.40 ms` of which 0.78 is the ANN and
1.47 is Python. You would be paying 200–360 s of prefill to remove a numpy loop.

**C3 — do not let `insert` off the hook.** Measured insert p50 **66.5 ms/layer** for 16 tokens
(`pag_layers[*].insert.p50_ms`, mean over 34 layers), i.e. 2.26 s per flush event across layers; at
one flush per 16 offloaded tokens that is **141 ms/token if serial**, against a 190 ms TPOT. It is
only affordable if the 8 heads insert concurrently, and today they cannot: `_select_locked` holds
`self.layer_lock` across all heads (`pag_retrieval.py:94-119`) and `insert` takes the same lock
(`pag_retrieval.py:198`), with the heads looped serially at 221-223. P5b must be answered with a
structure, not an estimate.

### Shape D — meta: a v2 that reports per-tree times, ratios, or two-sample scores

**D1 — the single number that flips the whole verdict is unresolved and must be resolved first.**
The user's "full prompt" column equals exactly `272 × s/tree` in all six rows (0.0382×272 = 10.4;
1.3194×272 = 358.9). That is consistent with a serial sum, and **inconsistent with "64 concurrent
on 64 cores"** — under 64-way concurrency, page-level msk=16 would be 0.0382 × 272/64 = **0.16 s**,
i.e. PAG would be ~6x *faster* than DCI's 1.0 s and my entire §1.2 cost argument inverts. The two
readings differ by 64x. If v2 quotes 10.4 s without stating that it is a wall clock measured on an
otherwise-idle box, it inherits the wrong reading of its own evidence, and I will attack it on that
alone. (The measured `pag_fast` run gives the wall-clock ground truth for one configuration:
196.6 s of build for 272 trees at msk=128, ef_construction=50 — 1.78x below the user's 358.9 s for
the same nominal configuration. P7.)

**D2 — no accuracy claim from ≤2 samples.** Both Phase-1 samples generated 7 and 5 tokens and scored
identically to DCI. Any v2 sentence containing "accuracy" needs P6 or it is unfalsifiable.

**D3 — no claim that PAG "replaces" DCI while `retrieval_backend` still falls back.** `_DCI_query`
takes the PAG path only when `self.n_prefetch_layers <= 1` (`infer_state.py:899-902`) and
`PagPageSelector` is only constructed when `check_reuse(cur_id) == 0` (`infer_state.py:693-718`);
`pag_stage1_compare.py:172` hardcodes `n_prefetch_layers=0, n_reuse_layers=0` while the shipping
configuration is `--n_reuse_layers 3` (`run_longbench.sh:27`). A PAG-vs-DCI TPOT comparison at
reuse=0 is not a comparison against the configuration the paper ships, and P5 says so.

---

## 5. What I would sign off on

I will drop this line of attack if, and only if, v2 contains:

1. **One wall-clock number** for the whole 272-tree build on an idle box, alongside DCI's 1.0 s
   (P1/P1b, D1).
2. **P3's result**, or the statement that P3 is the first thing it will run — because a fail there
   ends the direction on quality grounds and makes every cost number irrelevant.
3. **P6's oracle gate**, or the explicit statement that the accuracy axis is not being claimed.
4. **P5 measured against reuse=3 on both arms** (or the explicit statement that the decode-speed
   claim is made against a DCI baseline stripped of its own optimizations).
5. For the page-level shape: **the §1.3 proxy tax in the same table as the recall claim**, and a
   named answer to "what forms the pages" that is not DCI.

Everything else in v2 — the tree-per-head structure, the page unit, the kernel-side masking
(`decode.cuh:113-119`, v1 §3.4), the id-space and `cc2pg`/`ccc` contracts (v1 §3.1–3.3) — I still
consider sound and unchanged by the new facts. The argument was never that PAG's data structures are
wrong. It is that PAG offers one knob for two jobs that need opposite settings, and that the
affordable setting buys an index over an object — the 16-slot page — that already exists and is
small enough to scan exactly.

---

## Appendix: artifacts

* Non-timing measurement in §1.3: `/tmp/pagprobe/centroid_probe.py` → `/tmp/pagprobe/centroid.json`.
  L2/L20, Qwen3-4B, hotpotqa row 0, 11712 indexed tokens, 8 KV heads, 128 queries/head, greedy
  partition seed order = descending ‖k‖, 732 pages of exactly 16.0 slots. The `max` column
  reproducing v1's 0.832/0.883 is the control.
* Formation-degeneration evidence: `/tmp/pagprobe/paired.py`, `/tmp/pagprobe/greedy_pages_L2.npy`
  (shape (8, 11712), max page id 9843), `/tmp/pagprobe/paired.json`.
* Recall ceiling: `/tmp/pagprobe/ceiling.py` → `ceiling.json` (coco 0.8667/0.8787, DCI 0.6346/0.7054,
  DCI@13 0.6603/0.7343).
* Measured per-layer build/query/insert: `IceCache/benchmark/pred/pag_stage1_pag_fast.jsonl`,
  field `retrieval_stats`. Build share: 196.6 s of 202.02 s.
* PAG constraint sites: `PAG/pag.cpp:140-145` (ComputeWorkingSetSize), `:792-804` and `:1486-1487`
  (`max_search_k >= topk`), `:1144` (pif_entries_per_bucket), `PAG/paglib/pag_index_core.h:716-727`
  (65536 × msk PIF table), `:841` (`pif_projection_width_ == 8`),
  `PAG/python/pag_bindings.cpp:64, 94, 161` (GIL released in search / build / insert_batch).
* Adapter under review: `IceCache/source/icecache/pag_retrieval.py` (lines 50-63 serial head build,
  106 top_m, 94-119 layer_lock across heads, 169-174 retry, 176-195 max aggregation, 198-228 insert).
* Harness asymmetry: `IceCache/benchmark/run_longbench.sh:27` (`--n_reuse_layers 3`) vs
  `IceCache/benchmark/pag_stage1_compare.py:172` (`n_prefetch_layers=0, n_reuse_layers=0`).
* Partition→accuracy anchor: `experiment/07_mdci_promotion_sweep.md:38-43` (n=8, no CI).
* Caveats inherited from v1: all page-recall numbers are from **one** prompt (hotpotqa row 0) and
  should be replicated on 3–5 before being treated as established. §1.3 inherits that caveat.
