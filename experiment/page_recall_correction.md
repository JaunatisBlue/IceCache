# Correction: the page-recall table in `paged_pag_critique.md` is not recall@12

Everything below is measured on this box (GPU 0), Qwen3-4B, hotpotqa row 0 (11770 prompt
tokens -> 11712 indexed), post-RoPE post-`k_norm`/`q_norm` keys, pages of 16, budget 12 pages,
queries = last 32 prompt positions x 4 q-heads per KV head. Script:
`/tmp/pagprobe/pag_pages2.py`; raw JSON `/tmp/pagprobe/pag_pages2.json`.

## 1. The bug

`/tmp/pagprobe/pages.py:117`, in `eval_partition`:

```python
mask = torch.zeros(npg, dtype=torch.bool, device=dev)
mask[top] = True                      # top is (128, 12) -- one row per query
acc.append(mask[pt][exact].float().mean().item())
```

`mask` is built **once per KV head, from the top-12 pages of all 128 queries at once**, and is
never reset per query. So it measures

> "is this exact-token's page in the union of any of the 128 queries' top-12?"

not recall@12. The union mask covers only 4.8-36.5% of the 732 pages, but the exact top-192
tokens are themselves concentrated in a few dozen pages, so the union scores ~0.58 where the
honest per-query recall is ~0.18.

**Reproduced exactly** (`/tmp/pagprobe/dbg2.py`), position-contiguous partition, per KV head:

| head | union (reviewer's formula) | per-query | 4-q-head group |
|---|---|---|---|
| 0 | 0.4382 | 0.1403 | 0.2224 |
| 1 | 0.3929 | 0.1938 | 0.2625 |
| 2 | 0.7752 | 0.2840 | 0.4218 |
| 3 | 0.7910 | 0.2783 | 0.4309 |
| 4 | 0.6358 | 0.1680 | 0.2798 |
| 5 | 0.4314 | 0.1119 | 0.1779 |
| 6 | 0.5277 | 0.1340 | 0.1880 |
| 7 | 0.6738 | 0.1267 | 0.2399 |
| **mean** | **0.5833** | 0.1796 | 0.2778 |

0.5833 is the published 0.583. Every published row reproduces the same way (see §2), which
confirms the harness is otherwise faithful — only the metric is wrong.

**Why it matters beyond magnitude:** the union metric is not comparable across partitions.
A partition that packs the exact neighbours of many queries into few pages has a small union
mask that still covers a lot; a partition that spreads them has a large mask. The metric
compresses exactly the differences it is being used to measure.

## 2. All rows, recomputed (`pag_pages2.py`, 8 heads, L2/L20)

`pq` = per-query recall@12 (strictest, honest). `grp` = 4 q-heads sharing a KV head at one
position — the grouping IceCache actually uses (`pag_retrieval.py` aggregates per KV head).
`union` = the reviewer's metric, kept to show the reproduction.

| partition | layer | method | pq | grp | union |
|---|---|---|---|---|---|
| logical | 2 | oracle (max over members) | 0.1796 | 0.2779 | **0.5832** |
| greedy_nn | 2 | oracle | 0.4137 | 0.5225 | **0.8317** |
| dci | 2 | oracle | 0.2777 | 0.4066 | **0.7008** |
| logical | 20 | oracle | 0.2798 | 0.4170 | **0.8128** |
| greedy_nn | 20 | oracle | 0.4059 | 0.5539 | **0.8827** |
| dci | 20 | oracle | 0.2415 | 0.3850 | **0.7897** |

Bold column matches the published table (0.583 / 0.832 / 0.700 / 0.813 / 0.883 / 0.789) to
3 decimals in all six rows.

> ⚠ **Read the `pq` column with its definition in mind.** Every row's `pq`/`grp` is the strict
> per-query recall@12 computed with `method = oracle`, i.e. a page is scored by its **best
> individual member**. The **shipped** page_scan scan scores a page by its **stored
> representative** (the mean of its members), which is a different quantity and gives
> different numbers — on the shipped definition the same comparison reads
> **0.3144 / 0.3196 → 0.4721 / 0.4710**. The `pq` column below is a **partition diagnostic**,
> not a prediction of shipped accuracy; it has been measured to be a poor proxy for accuracy
> (an ~8pp `pq` range buys ≤1.6pp live accuracy, with chaotic sign). Do not quote `pq` as an
> accuracy result, and do not assume these two columns are interchangeable.

## 3. What survives, and what inverts

**Survives and strengthens — the partition-quality claim.** Under the honest metric the gap
between the greedy partition and DCI's is *larger*, not smaller:

| comparison | L2 | L20 |
|---|---|---|
| dci oracle pq | 0.2777 | 0.2415 |
| greedy_nn oracle pq | 0.4137 | 0.4059 |
| **relative gain** | **+49%** | **+68%** |

Not the +19%/+12% the published table implied.

**Inverts — the representative "tax".** `paged_pag_critique_v2.md` §1.3 reports `q·mean(P)` at
0.7400/0.8123 vs max-over-members 0.8317/0.8827, i.e. an 8-11% penalty for scoring a page by
one vector. That is a union-metric artifact. Under the honest metric the mean-representative
ranker **beats** max-over-members in every row:

| partition | layer | oracle pq | mean-rep exact pq | medoid-rep exact pq |
|---|---|---|---|---|
| greedy_nn | 2 | 0.4137 | **0.4721** | 0.4230 |
| dci | 2 | 0.2777 | **0.3144** | 0.2768 |
| logical | 2 | 0.1796 | **0.3241** | 0.2370 |
| greedy_nn | 20 | 0.4059 | **0.4710** | 0.4089 |
| dci | 20 | 0.2415 | **0.3196** | 0.2594 |
| logical | 20 | 0.2798 | **0.3577** | 0.3129 |

Max-over-members is biased toward pages holding a single high-norm outlier; the mean is
robust to it. Design v2's medoid choice is *worse* than the mean in every row (A1 is
answered: medoid ranking does not preserve recall — it loses 5-11% relative to mean).

**New — PAG's search loses to a plain matmul over the same representatives.** Same 732 reps,
same partition, ranking pages by PAG's search instead of an exact scan:

| partition | layer | exact mean-rep scan | PAG16 over mean reps | PAG relative cost |
|---|---|---|---|---|
| greedy_nn | 2 | 0.4721 | 0.4024 | **-15%** |
| greedy_nn | 20 | 0.4710 | 0.4563 | -3% |
| dci | 2 | 0.3144 | 0.2261 | **-28%** |
| dci | 20 | 0.3196 | 0.2963 | -7% |

PAG over DCI's own partition (0.2261) is **worse than DCI's own page selection** (0.2777).
So the ANN swap is not merely a cost/benefit trade — at this scale it is a strict quality loss.

## 4. Decode-speed axis: PAG loses there too

Per head, N=732 page representatives, 128 simultaneous query vectors, top_k=12
(`/tmp/pagprobe/pag_query_lat_probe.py`). Baseline: DCI's real query is **1.94 ms per LAYER**
(all 8 KV heads in one call, `pag_stage1_*.jsonl`), i.e. 66 ms of the measured 190 ms TPOT.

| method | ms/head (min) | x8 heads, serial |
|---|---|---|
| **exact matmul scan over 732 reps** | **0.1544** | **1.24 ms** |
| PAG search, 1 thread, msk=16, ef=32 | 2.3022 | 18.4 ms |
| PAG search, 8 threads, msk=16, ef=32 | 0.3161 | 2.53 ms |
| PAG search, 8 threads, msk=64, ef=128 | 0.8868 | 7.09 ms |
| PAG search(64) + `pag_retrieval.aggregate` | 2.0613 | 16.5 ms |
| DCI (reference, whole layer) | — | 1.94 ms |

Two conclusions. First, `aggregate` (the numpy `np.maximum.at` + `lexsort`) dominates at
~1.7 ms/head and single-handedly erases any search-side gain — the v2 reviewer's §4-B point
"even a free ANN leaves ~2.4 ms/layer vs DCI's 1.940" is confirmed and is a *floor*, because it
prices the aggregate, not the search. Second, a bare matmul over 732 representatives is
0.154 ms/head — **15x faster than PAG's single-threaded search and 2x faster than PAG with 8
OpenMP threads** — while being exact. At this scale the index is not just unnecessary, it is
the slow option.

## 5. Consequence

At the operating point where PAG is affordable (page level, N=732, msk=16, build 10.4 s/prompt
vs DCI's 1.0 s), PAG

* ranks pages **3-28% worse** than an exact scan of the same representatives,
* is **worse than DCI's own query** on DCI's own partition, and
* costs **2.5 ms/layer** against DCI's 1.94 ms, before the aggregate.

The real, measured headroom is in page **formation** (+49%/+68% relative over DCI's B+ leaves)
— and formation is the one thing PAG cannot supply: it exposes no clustering, and driving
formation through its graph needs `max_search_k ~ N`, which makes the build O(N^2).

So: replacing DCI's tree build *and* retrieval with PAG does not win on accuracy or on decode
speed. The page *idea* wins; PAG is not the instrument that captures it.
