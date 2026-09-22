# Paged-PAG v2: PAG over *page representatives*, with the partition computed exactly

Status: design, supersedes `paged_pag_design.md`. Branch `algorithm`. Every number below is
either tagged **[M]** (measured, source given in the line) or **[?]** (must be measured before
commit; the experiment is named).

---

## 0. What changes from v1, and why

v1 bundled three things into one swap: (i) a new partition algorithm, (ii) a PAG index over that
partition, (iii) the decode-time page-assignment machinery. The critique killed the bundle on cost
(202 s TTFT vs 6.22 s **[M]** `pag_stage1_pag_fast.jsonl`) and showed the ANN half buys nothing:
PAG's MIPS recall@64 against exact is 0.89 **[M]**, while an *exact* kNN neighborhood is available
for 2.7 ms of fp16 matmul per layer **[M]** — and the exact neighborhood produces a strictly
better partition (0.832/0.883 vs DCI's 0.700/0.789 **[M]**). v1 paid 300x for a worse
neighborhood.

v2 accepts that and re-partitions the work:

| component | v1 | v2 |
|---|---|---|
| partition | greedy along PAG graph edges | greedy along **exact** kNN, batched on GPU |
| PAG indexes | **tokens**, N=11712/head | **page representatives**, N=732/head |
| PAG build | 369 s **[M]** | 10.4 s at msk=16 / 22.3 s at msk=32 **[M]** |
| PAG PIF RAM | 17 GiB at msk=128 **[M]** | 2.2 GiB at msk=16 |
| decode page assignment | PAG search + vote + split | exact argmax over 732 reps (**no PAG**) |
| decode PAG insert | 16 pts/head/event | deferred, batches of ~1-3 pts/head/event |
| layer reuse / prefetch | forfeited | **kept** (partition is shared across a reuse class) |

The single biggest structural change: **the PAG index contains one point per page, and the point
index *is* the page id.** That removes v1's token→page indirection, removes the distinctness
problem at its root (12 slots ← top_k=12 returns 12 distinct pages by construction), and lets the
decode path avoid PAG entirely.

---

## 1. Decision 1 — where the partition comes from

### 1.1 The choice: (b), with the reviewer's algorithm, made affordable

**The partition is produced by a capacity-constrained greedy expansion along an exact kNN, computed
per (layer, KV head) at prefill, and is *not* PAG.** PAG is then used for exactly one thing:
`search` over the page representatives. Three reasons, in order of weight:

1. PAG exposes no graph and no clusterer (`PAG/python/pag/__init__.py` exports only
   `BuildOptions, Index, IndexMode, LoadOptions, Metric, SearchOptions`; `pag_bindings.cpp:212-232`
   binds `build/load/save/search/search_with_options/add/insert/add_batch/insert_batch`). Any
   PAG-derived partition is therefore `search(seed, top_k=R)` per seed, and `top_k <= max_search_k`
   is the same variable that sizes the PIF table — so the partition's candidate list is bought at
   the price of the build. R=64 already degenerates (9845 pages for 11712 tokens **[M]**
   `/tmp/pagprobe/paired.py`). There is no affordable R.
2. The exact neighborhood is *better* than PAG's (0.832/0.883 vs a 0.89-recall@64 neighborhood
   **[M]**) and *cheaper* (2.7 ms/layer of fp16 matmul **[M]**).
3. It is not k-means: no centroids, no Lloyd iterations, one pass, greedy, deterministic. The
   user's "核心并不是普通的k-means聚类" is satisfied.

**Which constraint it strains.** Not the one-thread-per-tree constraint (formation uses batched GPU
ops, no extra threads). It strains "PAG替换M-DCI的树构建": PAG does replace the tree — `Index.build`
over page reps replaces `btree_p_bulk_load`, and `search` replaces `db.query` — but the thing that
*decides the leaf contents* is neither DCI's sort nor PAG's graph. That is a deliberate split: the
user's mandate is about the tree, and the tree is PAG's. It does mean one cannot claim "PAG built
the pages".

### 1.2 Formation algorithm (F2), precisely

Per (layer, KV head) `h`, inputs `K` `(N,128)` float32 (post-RoPE, post-`k_norm`) and the prefill
queries `Q` `(4N,128)` (`_DCI_first_call` already receives `query_states`, `infer_state.py:597,1232`).

```
S     = K @ K.T                       # (N,N) fp16 on GPU, 2.7 ms/layer for 8 heads [M]
order = argsort(-||K||_2)             # seed priority, as in /tmp/pagprobe/pages.py:94
assigned = zeros(N, bool)
for seed in order:                    # batched over the 8 heads of the layer
    if assigned[seed]: continue
    sel = [seed]; assigned[seed] = True
    for _ in range(15):               # 16 sequential masked argmaxes
        S[seed][assigned] = -inf
        nxt = argmax(S[seed]); sel.append(nxt); assigned[nxt] = True
    page[sel] = nid++
```

This is byte-for-byte the algorithm that measured **0.832 (L2) / 0.883 (L20)** page recall@12
**[M]** (`/tmp/pagprobe/pages.py:46-64, 96-100`). The only change is that the 16-step inner loop
runs once for all 8 heads instead of once per head, so `assigned` and the argmax are `(8,N)`
tensors. The global-argmax (not top-R) rule is load-bearing: restricting to a top-R list collapses
the partition (9845 pages **[M]**).

Because the expansion is a global argmax over unassigned tokens, **every page is exactly 16 tokens
until the last**: page count is `ceil(N/16) = 732` for N=11712, deterministically. That is 11%
fewer pages than DCI's 818 **[M]** and it is what makes the CPU allocation exactly
`num_offload_pages`.

**Page representative.** `rep[h][p] = K[argmax_t sum_{u in page} <K_t, K_u>]`, the medoid, kept as
its **key vector**. Not the centroid: the retrieval geometry is MIPS on unnormalized keys and a
centroid is off-manifold.

**Variant F2q (optional, 1 extra matmul).** Seed priority `order = argsort(-C)` where
`C(t) = sum_{q in Qsample} relu(q·K_t)` over ~256 sampled prefill queries (0.77 GFLOP/head, ~0.05 ms
on GPU). The partition then maximises query co-occurrence, which is the quantity page recall@12
actually measures (the reviewer's co-occurrence-trained ceiling is 0.867/0.879 **[M]**). Untested;
costs one matmul; a strict candidate improvement.

**Variant F1 (the safe floor).** Reimplement DCI's own leaf rule without DCI: bucket tokens by the
sign pattern of `num_indices` random projections, sort inside each bucket by the **MIPS→L2 lifted**
projection `k·proj + sqrt(M² − ||k||²)` with `M² = max||k||²` (do the sort in float64; the brief's
float32 catastrophe is about *searching* with the lifted distance, not about ordering by it), then
cut at 16. `O(N)` + one sort. Reproduces DCI's page family ⇒ page recall ≈ **0.700/0.789 [M]**,
i.e. parity, ~50 ms/layer. F1 exists so that the *retrieval* swap can be evaluated independently of
the *partition* swap; it is the fallback if F2's partition quality does not survive §1.3.

### 1.3 The cost and the catch

| | F1 (DCI-equivalent) | F2 (greedy) |
|---|---|---|
| formation, 34 layers | ~0.2 s (est.) | **1.5-5 s [?A3]** |
| pages/head | ~818 | **732** |
| page recall@12, in-page-exact scoring | 0.700 / 0.789 **[M]** | 0.832 / 0.883 **[M]** |
| page recall@12, **medoid** scoring | **? [?A1]** | **? [?A1]** |

The catch, stated plainly: **the 0.832/0.883 numbers are measured with exact in-page scoring
(`pm.scatter_reduce_(..., reduce='amax')` over all 16 members, `/tmp/pagprobe/pages.py:110`).** The
deployed system ranks a page by `q · rep`, one vector. The entire accuracy case for v2 is the
assumption that medoid ranking retains enough of that gap to stay above DCI — and that is **[?A1]**,
not a fact. If [?A1] fails, F2's advantage evaporates and v2 becomes "a slower DCI with a different
tree", i.e. a decode-speed play only.

Cheap partial hedge if [?A1] fails: keep a second GPU-resident per-page vector `cover[h][p]` = the
member least similar to the medoid (8×732×256 B = 1.5 MB/layer), rank `max(q·medoid, q·cover)`.
It is *not* in the PAG index (that would double N and force `msk >= 24` for distinctness), it is a
re-rank of PAG's candidate set, and it costs one gather of ≤msk vectors.

---

## 2. Decision 2 — what PAG indexes, concretely

**One PAG index per (retrieval layer, KV head): 272 indexes. Each index holds `n_pages[h] = 732`
points, point `p` = `rep[h][p]` = page `p`'s medoid key, float32, dim 128. The label returned by
`search` is the page id; there is no mapping array.**

* `max_search_k` (== `max_entry_points`, `pag.cpp:140,985`; `pag_index_core.h:62`) is the *only*
  strong build knob and it caps `top_k`. With one point per page, `top_k = 12` returns 12 distinct
  pages — the distinctness constraint that forced v1's `max_search_k >= 32` is gone. **Default
  `max_search_k = 16`** with `top_k = 16` (12 kept, 16 gives headroom for the 4 q-heads of a GQA
  group whose selections overlap). `max_search_k = 32` if the shortfall rate exceeds 5%.
* **Build (272 trees, one thread per tree, measured line):** 10.4 s at msk=16, 22.3 s at msk=32
  **[M]**, versus 1.0 s for DCI's 272 trees **[M]**. That is the price of the mandate and it is the
  whole TTFT story (§4).
* **PIF RAM:** `65536 × max(10,msk) × 8 B` = 8 MiB/head at msk=16 → **2.2 GiB total**, versus
  17 GiB at msk=128. A real, unbudgeted cost reduction over v1.
* `ef_search = 100`, `ef_construction = 100`, `target_degree = 16`, `projection_levels = 16`
  (effects 1.2-1.3x **[M]**).
* `Metric.MaximumInnerProduct` is forced: the MIPS→L2 lift `[k, sqrt(M²−||k||²)]` is exact in real
  arithmetic but numerically dead in float32 (brute-force recall 0.078 **[M]**), and Cosine/L2
  build is flat in msk but its recall@64 is 0.06-0.5 **[M]**. So the linear-in-msk PIF cost cannot
  be escaped; [M] says that costs 10.4 s at page granularity, not 369 s.
* Thread control: set `ctypes.CDLL('libgomp.so.1').omp_set_num_threads(1)` before each build so
  each tree really is one thread (PAG links system libgomp, torch uses Intel OpenMP; verified
  working **[M]**).

---

## 3. The three load-bearing assumptions, and how to kill each in <10 min on GPU 0

All three run from `/tmp/pagprobe/keys.npz` (real Qwen3-4B post-RoPE `k_norm`/`q_norm` K and Q,
hotpotqa row 0, layers 2 and 20, 11712 indexed tokens **[M]**), write only under `/tmp`, and touch
no file under `IceCache/source/`.

**A1 — Ranking by a single page medoid preserves most of the partition's page recall.**
*Kill test.* On the same 16 held-out queries/head as the reviewer, compute recall@12 four ways over
the *same* greedy partition: (i) max-over-members — must reproduce 0.832/0.883; (ii) `argmax_p q·rep`;
(iii) `argmax_p max(q·rep, q·cover)`; (iv) the mediation-free reference, DCI's 0.700/0.789.
*Kill criterion:* (ii) < 0.70 (L2) or < 0.789 (L20). Then F2 owns no accuracy advantage and §4's
"accuracy" column is empty. ~5 min (the partition and `greedy_pages_L2.npy` already exist).

**A2 — PAG at N=732 is fast enough and accurate enough to be worth the swap.**
*Kill test.* Build **one** index at N=732, msk=16 (0.0382 s **[M]**), `ef_search=100`. Then:
(a) 2000 timed `search(q4, top_k=16)` calls, `q4` = 4 correlated q-heads of one KV head
(`Q.reshape(8,128,128)[h][0:4]`), record p50/p95; (b) for the held-out queries, the exact-set
recovery: `|PAG_top12 ∩ exact_top12(medoids)| / 12`; (c) the real pipeline shape — 34 sequential
per-layer calls, each dispatching 8 head searches through an 8-thread pool, so that pool latency
and GIL contention are in the number (this is what made Phase 1 report 3.18 ms/layer **[M]**).
*Kill criterion:* p50 > 0.6 ms/layer in (c), or exact-set recovery < 0.95. If p50 > 1.5 ms/layer
the TPOT win is gone (0.1904 − 0.066 + 0.051 = 0.175 s, 1.09x) and v2 has no winning axis at all.
~10 min.

**A3 — Decode-time packing needs no PAG insert (or a cheap one).**
*Kill test.* Drive the decode assignment for 512 simulated tokens/head from the real keys: new page
every ~16 tokens (cap 16, prefill pages are full) → ~32 new pages/head. Measure (a) the exact-rep
argmax assignment for 8 heads (`reps (8,796,128) @ keys (8,128,16)`) including the Python path;
(b) `insert_batch` of one batch of 3 points into an N=796, msk=16 index, warm, 200x — the only
insert number on record is 15.2 ms at N=736/msk=128 and 6.4 ms at msk=64 **[M]**, and it is
*inverted* with respect to the 2.2 ms at N=11776 **[M]**, so it must be reproduced.
*Kill criterion:* (a) > 0.15 ms/layer (i.e. > 5 ms/step for 34 layers), or (b) > 2 ms/head with no
viable deferral. ~10 min.

If A3(b) fails, the documented fallback is **defer by one flush event** (16 steps) and let the
retrieval shortfall be made up by the exact-medoid scan of §1.1 — measurable as a page-recall dip,
not a correctness break.

---

## 4. Does v2 meet §12.2? Honest projection

Baseline (hotpotqa rows 0/1, Qwen3-4B, `n_unlimited_layers=2`, 34 retrieval layers, 8 KV heads,
`page_size=16`, `budget=16`, `n_dci_pages=12`, `num_neighbours=12` **[M]** `infer_state.py:877,951`):
**TTFT 6.22 / 7.59 s, TPOT 0.1904 / 0.1689 s [M]**.

| per-prompt item | DCI now | v2 | source |
|---|---|---|---|
| DCI build, 272 trees | 1.0 s **[M]** | — | |
| page formation F2 | — | 1.5-5 s [?A3] | §1.3 |
| KV permutation into page order | ~0 (inside DCI's memcpy) | 0.34 s (GPU gather + one D2H/layer) | v1 §3.2 |
| PAG build, 272 trees, msk=16 | — | **10.4 s [M]** | FACT C |
| **TTFT (CPU-bound; overlaps the ~5 s of prefill GPU work)** | **6.22 s** | **13 s** (band 11-17) | |
| DCI query, 1.94 ms × 34 layers | 66 ms/step **[M]** | — | |
| PAG query, 34 layers × 8 heads | — | 10-20 ms/step [?A2] | |
| decode rep-argmax assignment | 0 | 2-3 ms/step [?A3] | |
| PAG insert (batched, deferred) | 0 | 0-3 ms/step [?A3] | |
| `recall`: page-id gather replaces `DCI.copy_to_buffer` | 0 | 0-5 ms/step | §5 |
| **TPOT** | **0.1904 s** | **0.147 s** (band 0.135-0.175) | |

**Verdict.** The accuracy axis is **not** claimed: F2's partition beats DCI's by +19%/+12% relative
in-page **[M]**, but the deployed ranker is a medoid, and [?A1] is unresolved. The honest
expectation is **parity ± a few points**, and the reviewer's own strongest counter-argument — that
page formation is the only lever with headroom — cuts *for* v2 but is neutralised by the ranking
question. So the single defensible win is:

> **TPOT: ~1.29x faster (0.1904 → ~0.147 s), from deleting 66 ms/step of DCI query. TTFT: ~2.1x
> slower (6.22 → ~13 s), from a PAG build that is 10.4x DCI's.**

That is a win on one of the two axes, so §12.2's literal bar ("至少有一方面超过") is met. It is
not a good trade for a short-prompt benchmark, and it is a *bad* trade for LongBench specifically,
where prompts are 8-17k tokens and only 5-500 tokens are generated: the TTFT regression dominates
the end-to-end time and the TPOT win cannot pay it back. **Do not claim an overall win.** State it
as: decode-speed win at fixed ~1.3x, TTFT regression ~2.1x, accuracy unresolved.

**The TTFT-neutral schedule (disclose it, do not hide it).** `estimate_select_recall` runs on a
1-worker asyncio executor (`infer_state.py:245`) and the first decode step needs retrieval for all
34 layers. If page formation completes at prefill (it must — pages must exist) but the PAG builds
are submitted to a 64-thread pool *after* the first token is emitted, decode runs on the exact
medoid scan (0.02 ms/head, exact, better than PAG) for the first ~1-2 s while the trees build, and
PAG takes over. TTFT then returns to ≈ 6.8 s (DCI's 1.0 s build replaced by ~1.6 s of formation +
permutation). This is a *scheduling* choice, not an algorithmic one; it is legitimate only if the
harness reports `pag_build_seconds` next to TTFT and the run is labelled "async tree build". It is
the difference between "2.1x worse TTFT" and "roughly neutral TTFT", and it should be an explicit
flag, not a default.

---

## 5. Implementation plan

### 5.1 Files

| file | action |
|---|---|
| `IceCache/source/icecache/paged_pag.py` | **new.** `SemanticPages` (formation, medoid, occupancy, decode assign), `PagPageIndex` (build/select/insert/close), `diff_pages_by_head` (numpy port of `src/py_dci.c:2710-2793`), `valid_entries` (port of `src/py_dci.c:2212-2248`) |
| `IceCache/source/icecache/pag_retrieval.py` | **delete.** ~230 lines, superseded; kills the `InsufficientPagesError` import at `infer_state.py:20` |
| `IceCache/source/icecache/infer_state.py` | modified, §5.2 |
| `IceCache/source/icecache_cpp/src/gather.cu` (+ `api.cu` binding) | **new, ~30 lines.** `copy_pages_to_buffer(src_addresses, ptr_dest, list_size, update_num, offset_s, offset_t, dim, page_size, dtype)` — a signature-compatible replacement for `DCI.copy_to_buffer`. Fallback if we do not want to rebuild the extension: `torch.index_select` on `cpu_kvc.pool.buffer.view(-1)`, ~0.3 ms/layer |
| `IceCache/source/icecache/adapter/modeling.py` | **unchanged.** `projected` (`:133`) and `query_states` (`:146`) are already passed; `scatter_pages` (`:279,286`), `estimate_select_recall*` (`:284,301`) contracts are preserved |
| `IceCache/benchmark/{longbench,gsm8k,passkey}_pred.py` | add `paged_pag` to `--retrieval-backend` (`longbench_pred.py:50`), plus `--pag-page-msk`, `--pag-page-topk`, `--pag-formation {f1,f2}`, `--pag-async-build`; keep `--pag-max-search-k` default 128 for the legacy `pag_mips` backend |

### 5.2 `infer_state.py`, function by function

**`_DCI_first_call` → `_PAG_first_call` (`:597`).** Drops `dci_db.add_query_at_end` (`:630-657`).
New body, in order:
1. `pages = SemanticPages.form(F1 or F2, key_states, projected, layer)` → per head `rep (P,128)`,
   `occ (P,)`, `n_pages`, and the transient `perm_idx (8, P, 16)` int32 with `-1` padding.
2. `max_num_leaves = n_pages.max() + PAG_RESERVE` (default 64). `cpu_cache.prefill_alloc_n_tokens(max_num_leaves*page_size)` and the **contiguity assert at `:670` becomes `(max_num_leaves-1)*stride`** — it still holds because we allocate the whole run once and never grow or free (this is the design's answer to critique hazard (3)).
3. `kvc_capacity[cur_id] = 1 << (max_num_leaves-1).bit_length()`; `cc2gp = -1`, `ccc = 1`, `page_address_buffer` — same three allocations as `:684-690`, sized from a number that now never changes.
4. `page_address_buffer[cur_id][b,i,:n_pages[i]] = base + i*offset + arange(n_pages[i])*stride`; reserved tail pages get the same law. `offset = page_size*head_dim*cpu_dtype.itemsize` as at `:698`.
5. `PagPageIndex(cur_id, rep[:, :n_pages], cfg).build()` — 8 tasks, one thread each, on the shared pool of §5.3.
6. The reuse branch (`:719-727`) becomes `SemanticPages.copy_partition_from(anchor)` + a re-permutation of *this* layer's keys into the shared page layout — semantics identical to `DCI.reuse_copy_node`, and it is what keeps layer reuse alive (critique hazard (6) does not apply to v2).

**`prefill_backup_pages` (`:1115`) + `prefill_evict_extra_pages` (`:1193`).** Reordered so formation
happens before the copy: `offloaded = kvc.buffer[gpu_start : gpu_start+num_offload_pages]` (GPU, token
order) → `form` → `permuted = offloaded[perm_idx]` (GPU gather over the `2,P,16,128` view) → one D2H
into `cpu_kv_caches[cur_id]` at the page slots. `temp_cpu_kv_caches[num_offload_pages*page_size]`
allocation (`:1135`) and its `clear()` (`:1236`) are deleted; `tmp_cpu_kvc` is no longer materialised
for this backend. `num_offload_pages` / `n_dci_pages` / `kvc.n_win_pages` arithmetic (`:1128-1132`)
is untouched — it is about GPU pages.

**`_DCI_add` → `_PAG_add` (`:730`, called from `offload_win_page_to_DCI` `:1178`, itself called at
`:533` and `modeling.py:311`).**
1. `assign = SemanticPages.assign_decode(keys (8,16,128))`: `scores = rep @ k` masked to pages with
   `occ < 16`, one argmax per new token; if no page has a slot, take the next reserved page id and
   set its rep to the token. Batched over the 16 tokens and 8 heads.
2. Write the 16 KV pairs into the target page slots of `cpu_kv_caches[cur_id]` (2×16×128×4 B ×8
   heads = 128 KB/layer/event, one scatter).
3. `kvc.ccc[b, h, page] = 1` for every written page — written directly, deleting the
   `changed_page_list` round trip at `:767-768, 792, 795-796`.
4. **Delete the entire growth block `:806-837`** (`decode_alloc_n_tokens`, the `kvc_capacity` bump,
   the `cc2gp`/`ccc`/`page_address_buffer` `utils.cat`) and the whole `new_num_leaves` /
   `tmp_new_indices = arange(prev, n_pages)` block `:840-858`. This is the single largest
   deletion and it removes critique hazards (2) and (3) outright.
5. Enqueue the new/changed page reps for a batched `insert_batch` on the background pool (§5.3).
6. `self.prev_num_pages` / `prev_num_points` / `prev_index` / `prev_offset` become dead state; the
   reuse branch `:866-869` is replaced by the same shared-partition rule as above.

**`_DCI_query` → `_PAG_query` (`:872`).** Body becomes
```
page_ids = self.pag_indexes[cur_id].select(_query, num_neighbours)   # (8,12) int32, best-first
return self._apply_selected_pages(b, cur_id, page_ids)
```
Deletes `db.query` (`:916-926`), the `nn_idx.reshape(...,2,-1)` split (`:928-932`), the
`first_k_unique` GQA dedupe (`:935-939`, moved into `select`), and the `nn_idx_all` prefetch write
(`:931-932`). The `n_prefetch_layers <= 1` gate at `:901-902` stays in M3 and is lifted in M5.

**`_apply_selected_pages` (`:949`).** **Unchanged except line `:966`**: `DCI.diff_pages_by_head` →
`paged_pag.diff_pages_by_head` (numpy, bit-exact port including the in-place `cc2gp` mutation and
the `ccc` merge). Contract preserved: input `(8,12)` int32 best-first, returns
`(evicted_idx, recall_idx, evict_num)`; the shape assert at `:952` still fires. The new failure mode
is shortfall, not `InsufficientPagesError`: `select` fills the tail with the best-scoring pages from
an exact rep scan and increments `pag_shortfall_count`.

**`get_valid_entries` (`:1100, 1103`).** `self.dci_db[layer].get_valid_entries(selected_page_idx)` →
`SemanticPages.valid_entries(selected_page_idx)`, returning per-(head,page) **current occupancy**
(`-1` for `id < 0` or `id >= n_pages[h] + reserve_used`). Critique hazard (h) is satisfied by the
kernel as written (`decode.cuh:113-119` is a per-(page,head) slot count with no positional
assumption), but the init value `page_size` at `:394-398` means every page we hand to the kernel
must have a correct occupancy — which `SemanticPages` maintains exactly.

**`recall` (`:1016`).** `DCI.copy_to_buffer` (`:1038`) → `_cpp.copy_pages_to_buffer` over the same
`_src_address_buffer` (`:1031-1034` unchanged). Everything else, including the `c2g_stream` and the
`non_blocking` H2D at `:1045-1052`, is unchanged.

**`alloc_page` (`:1242`).** **Unchanged** — it is about the *GPU* pool and `prefill_evicted_pages`,
not the CPU partition.

**`retrieval_stats` (`:976`).** Rename the `pag_layers` payload to the page index: add
`n_pages_per_head`, `formation_ms`, `shortfall_rate`, `insert_ms`, `pag_build_seconds`. This is the
reporting surface for the §4 TTFT disclosure.

**`check_reuse` (`:294`) and the gates.** With `n_reuse_layers > 0` the page-id space is shared
across a reuse class and the CPU page → token-set map must be identical across the class — so the
class anchor forms the partition and every other layer in the class re-permutes into it
(§5.2 step 6). Add an init-time check that `retrieval_backend == "paged_pag"` requires either
`n_reuse_layers == 0` or that rule; and delete the `pag_disabled` / `pag_fallback_count` /
`PagPageSelector` fallback scaffolding (`:91-96, 312-316, 693-718, 859-865, 899-914, 986-1001`).

### 5.3 Threading

* **Build:** one thread per tree. 272 tasks submitted to a `ThreadPoolExecutor(max_workers=64)`,
  `omp_set_num_threads(1)` before each `build`. Measured regime, 10.4 s at msk=16 **[M]**.
* **Query:** 8 head searches per layer. Either (a) sequential on the calling thread — a `pag.Index.
  search` at N=732/msk=16 is the cheapest option and avoids pool latency entirely, or (b) the
  existing 8-worker pool (`_ensure_pag_pool`, `:305-316`). Decide from [?A2](c), which measures both.
  **Do not** layer a per-head pool inside a per-layer pool inside the asyncio 1-worker executor;
  that nesting is what produced Phase 1's 0.613 s TPOT **[M]**.
* **Insert:** one background pool, per-head lock, batches only, never on the decode critical path.

### 5.4 Milestones

| # | deliverable | gate |
|---|---|---|
| M0 | [?A1], [?A2], [?A3] as `/tmp` scripts, no source edits | any kill criterion → stop, report the negative result |
| M1 | `paged_pag.py` formation F2 (batched) + F1 + medoid + `assign_decode`, validated standalone against `keys.npz` | page count == 732/head, recall reproduces 0.832/0.883 under max-over-members |
| M2 | `PagPageIndex` build/select/insert + shortfall path, standalone | [?A2] numbers hold in the pool-shaped harness |
| M3 | `infer_state.py` integration end-to-end on **one** hotpotqa row, `--retrieval-backend paged_pag`, both formations | TTFT/TPOT within the §4 bands; `shortfall_rate` and page recall reported; parity vs `dci` on the same row |
| M4 | reuse-aware partition sharing; async tree build (TTFT-neutral schedule) | TPOT unchanged at `n_reuse_layers>0`; async build reports `pag_build_seconds` |
| M5 | lift the `n_prefetch_layers <= 1` gate; F2q query-aware seeding; cover-vector rerank | each independently measurable, each revertible |

M3 is the go/no-go: if v2 does not clear TPOT ≤ 0.170 s on the real pipeline, it has no winning
axis and the correct action is to publish the negative result (the 10.4 s page-level build and the
medoid-ranking measurement are both worth having).

---

## 6. What v2 gives up, and what it keeps

**Keeps (unlike v1):** layer reuse and prefetch (shared partition per reuse class, §5.2); the
`_apply_selected_pages` / `diff_pages_by_head` / `cc2gp` / `ccc` / `page_valid_entries` machinery
verbatim; the GPU-page `alloc_page` / `prefill_evicted_pages` path; `modeling.py` unchanged;
`num_neighbours = 12` and the `(8,12)` selector contract (`infer_state.py:952`).

**Gives up:** the claim that PAG *formed* the pages; the claim that the ANN swap buys accuracy
(it buys 10.4 s of TTFT and an 0.89-recall neighborhood where exact costs 2.7 ms); the ability to
grow the page-id space after prefill (bounded by `PAG_RESERVE`, with a documented overflow rule:
pack into any page with a free slot, else drop the token's page from the index).

**Unchanged hazards from the critique that v2 still owes an answer to.** Hazard (1)/(1g): the
contiguity assert at `infer_state.py:670` is verified once, and v2's "allocate once, never free,
never grow" rule is precisely what keeps it true — any future allocator change re-opens it. Hazard
(1e): `cc2gp`/`ccc`/`page_address_buffer` have no growth path in v2 *by construction*, so a design
change that reintroduces growth silently indexes out of bounds on GPU. Hazard (4): `page_valid_entries`
inits to `page_size` at `:394-398`, so a page whose occupancy is not maintained will have its tail
slots attended over — `SemanticPages` maintains it exactly and `valid_entries` is the only writer
of the `[ns : ns+12]` slice.

---

## Appendix — traceability of the numbers used above

* DCI build 0.030 s/layer, 1.0 s/272; PAG page-level 0.0382 s/tree @msk=16 → 10.4 s, 0.0821 @msk=32
  → 22.3 s; token-level 0.1983/0.2730/1.3194 → 53.9/74.3/358.9 s; `insert_batch` 16 pts: 2.2 ms
  (N=11776/msk=128), 15.2 ms (N=736/msk=128), 6.4 ms (N=736/msk=64); msk 128/256 → 3.602/8.977 s;
  efc 50→400 = 1.2x; projection_levels 8→128 = 1.3x; metric Cosine flat in msk at 88 ns/pt; MIPS→L2
  lift float32 brute-force recall 0.078; `num_neighbours = 12 - 0 = 12`; DCI leaves 809-818 for
  11712 tokens = 14.4 slots/page; TTFT 6.22/7.59 s, TPOT 0.1904/0.1689 s, DCI query p50 1.94 ms/layer;
  Phase-1 `pag_fast` TTFT 202.0 / TPOT 0.613 / 3.18 ms/layer; PIF 64 MiB/head at msk=128 = 17 GiB;
  libgomp `omp_set_num_threads` verified — all **[M]**, from the measured-facts list and
  `IceCache/benchmark/pred/pag_stage1_*.jsonl`.
* Page recall@12 (real Qwen3-4B keys, layers 2/20, 16 held-out queries/head, budget 192 tokens):
  random 0.416; projection-sorted 0.452/0.592; position-contiguous 0.583/0.813; DCI real B+ leaf
  0.700/0.789; exact-kNN greedy 0.832/0.883; paired DCI 0.635/0.705, DCI @13 pages 0.660/0.734,
  co-occurrence-trained 0.867/0.879 — **[M]**, `/tmp/pagprobe/{pages,paired,ceiling}.py` and
  `{partial,paired,ceiling}.json`. Single-prompt measurements; replicate on 3-5 prompts before
  they are treated as established.
* PAG API: only `BuildOptions, Index, IndexMode, LoadOptions, Metric, SearchOptions` /
  `build, load, save, search, search_with_options, add, insert, add_batch, insert_batch,
  is_loaded, metric, dimension` (`PAG/python/pag/__init__.py`, `pag_bindings.cpp:212-232`).
  `ComputeWorkingSetSize(topk)=max(10,topk)` → `max_entry_points` → `max_query_top_k_`
  (`PAG/pag.cpp:140,985`; `PAG/paglib/pag_index_core.h:62`); search seeds from all entry points
  (`pag_index_core.h:506`); PIF table `pag.cpp:1144`, 65536-iteration double loop
  `pag_build_pipeline.inc:340-348`. **[M]**
* Integration points: `infer_state.py` 59-96 (backend config), 245 (1-worker executor), 294-303
  (`check_reuse`), 305-316 (`_ensure_pag_pool`), 394-398 (`page_valid_entries` init), 457-460
  (`proj_vec`), 463-488 (reset), 528-534 (`_prepare_decode` → `offload_win_page_to_DCI`), 551-552
  (`n_dci_pages`), 597-727 (`_DCI_first_call`), 730-869 (`_DCI_add`), 872-974
  (`_DCI_query`/`_apply_selected_pages`), 976-1002 (`retrieval_stats`), 1016-1052 (`recall`),
  1099-1103 (`get_valid_entries`), 1115-1151 (`prefill_backup_pages`), 1178-1186
  (`offload_win_page_to_DCI`), 1193-1240 (`prefill_evict_extra_pages`), 1242-1260 (`alloc_page`);
  kernel mask `3rdparty/flashinfer/include/flashinfer/attention/decode.cuh:113-119`; `modeling.py`
  125-146, 279-311.
