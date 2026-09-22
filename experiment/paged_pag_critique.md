# Adversarial review: replacing M-DCI with PAG inside IceCache, keeping "semantic pages"

Scope: the three-layer split (PAG graph = *who is near whom*; SemanticPageManager = *how near
tokens pack into fixed-size pages*; IceCache = *page movement + sparse attention, unchanged*),
and the specific claim that M-DCI's fixed-size page is the **B+ tree leaf**, so the thing to
reproduce is
`semantic parent -> group tokens by parent -> sort within parent by projection -> bulk-load -> leaf = page -> token2nodeIndex/token2nodeOffset -> btree_p_split_leaf() on overflow`.

Everything below is traceable to a file:line in this repo / `/tmp/icecache-mdci-source/`, to a
number in the measured-facts list, or to a measurement I made and report inline. Measurements I
made are marked **(measured here)** and the scripts are in `/tmp/pagprobe/`.

---

## 0. The one-paragraph verdict

The claimed structural insight is **mostly right and unusually well-researched** — the leaf really
is the page, `BTREE_LEAF_MAX_NUM_SLOTS` really is 16, and `token2nodeIndex/token2nodeOffset` really
is token -> (page, offset). But it is wrong in four specifics that change the design, and the cost
verdict is not close: the measured best-case Phase-1 PAG run is **TTFT 202.0 s vs 6.2 s for DCI
(32.5x) and TPOT 0.613 s vs 0.190 s (3.2x)**, and that run used **DCI's own pages**, i.e. it did not
include the proposal's page-formation half at all. The page-formation half is where the real
opportunity is, and I measured that opportunity: a better partition moves page recall@12 from
~0.64-0.70 to ~0.83-0.88 (+25-35% relative). That gain is real, it is worth chasing — and it does
**not** need PAG. It needs an exact or near-exact kNN pass that costs 2.7 ms of GPU matmul plus a
~1.2 s/layer packing loop. PAG's MIPS index is a strictly worse neighborhood at ~300x the cost.

---

## 1. Where the claimed insight is right, and where it is subtly wrong

### 1.1 Verified as claimed

| Claim | Verdict | Evidence |
|---|---|---|
| `BTREE_LEAF_MAX_NUM_SLOTS == 16` | **TRUE** | `include/btree_common.h:21` |
| the leaf is the fixed-size page | **TRUE** | leaf holds `data_loc`/`inc_data_loc` = `sizeof(float)*dim*16` for K and V (`src/btree_p.c:105-106, 389-390, 579-580, 721-722`); IceCache maps leaf id -> CPU page address, `page_address_buffer[cur_id][b,i,:len]` (`infer_state.py:700-708`) |
| token -> (page, offset) is `token2nodeIndex`/`token2nodeOffset` | **TRUE** | `include/dci.h:48-49`; written at `src/btree_p.c:529-532` (insert), `612-613` (split), `745-748` (bulk_load), `1251-1252, 1345-1346, 1371-1379`; surfaced as `token2node` (`src/py_dci.c:120-137`) |
| IceCache allocates CPU pages from `num_leaves` | **TRUE** | `max_num_leaves = dci_db.num_leaves.max()` then `cpu_cache.prefill_alloc_n_tokens(max_num_leaves * self.page_size)` (`infer_state.py:665-668`); `num_leaves = num_leaf_nodes` (`src/py_dci.c:2149`) |
| the page family is a genuine partition of tokens | **TRUE** (I checked, because the proposal depends on it) | DCI with N=11770, dim=128, 8 heads: `num_leaves = [810, 815, 815, ...]`, all 810 leaves of head 0 occupied, 11770 distinct `(leaf, offset)` pairs, max occupancy 16, offsets observed `0..15`. Every token has exactly one page slot. **(measured here)** |

So the "reproduce this exact structure" instruction is well-founded, and the reviewer's instinct
that "the leaf, not the DCI node, is the page" is correct.

### 1.2 Wrong in four ways that matter

**(a) `num_leaves` is not `ceil(N/16)`, and pages are not full.**
`btree_p_bulk_load` sets `leaf->num_slots_used = (int)(num_items / (num_leaves - i))`
(`src/btree_p.c:732`), and `btree_p_split_leaf` splits at `mid = leaf->num_slots_used >> 1`
(`src/btree_p.c:556-560`), not at 16. Measured: for N=11712 offloaded tokens, DCI produces
**809-818 leaves per head, average 14.4 slots per page** — 11% more pages than the 732 the proposal
assumes, each 10% emptier. Three consequences the proposal misses:

* 12 recalled pages carry ~173 usable tokens, not 192. The proposal's whole recall arithmetic
  ("take the top-B pages", B = 12) silently assumes 16/page.
* `kvc_capacity = 1 << (int(max_num_leaves) - 1).bit_length()` (`infer_state.py:684`) is sized from a
  number that is a *high-water mark of leaf allocations across all the instance's B+ trees*, not
  from N. It is not a function of N and not predictable from the proposal's model.
* the CPU page pool is allocated for `max_num_leaves * page_size` tokens (`infer_state.py:668`), i.e.
  11% more than N/16 per layer.

**(b) The leaf is not ordered by "the projection value".**
M-DCI's sort key is the **MIPS->L2 lifted projection**, `local_dist`, not `proj_vec . k`:
`update_local_dist` adds `(n_term - o_term) * add_proj_vec[k]` where
`o_term = sqrt(point->max_sq_norm - ||x||^2)` (`src/dci.c:571-578`), the base value comes from
`data_projection` adding `sqrt(max_sq_norm - sq_norm) * add_proj_vec[j]` (`src/dci.c:364`), and the
whole subtree is re-sorted by `qsort` on `local_dist` before bulk load
(`src/dci.c:614, 625, 691, 830, 1739`). Two implications:

1. The proposal explicitly discards this lift ("the `||k||^2` term is annihilated in float32"), but
   the lift is exactly what makes the 1-D ordering a MIPS ordering. Dropping it and sorting by
   `proj_vec . k` is *not* reproducing M-DCI's leaf geometry. I measured the difference (below):
   projection-sorted pages are the **worst** of all the layouts I tried except random.
2. The key is **time-varying**: whenever the global `max_sq_norm` grows, every point's `local_dist`
   changes and the point's trees are re-sorted and re-bulk-loaded (`src/dci.c:580-640`,
   `add_and_update_max_sq_norm` at `642-706`). A page assignment in M-DCI is not a static function
   of the keys. Any SemanticPageManager spec that says "page = f(key)" is not equivalent.

**(c) Pages are also created at decode time by median splits, not by bulk load.**
`btree_p_insert` -> `btree_p_split_leaf` (`src/btree_p.c:363-410, 418-500, 556-620`) is the same
code path the proposal wants to mimic, but the split rule is "median of `num_slots_used`", moving
`mid` slots left and the rest right — **not** a 2-seed exact repartition of the ~17 vectors. The
proposal's replacement rule is a different algorithm, so citing M-DCI as precedent for it is not
justified. (It may still be a better rule; it is not the same rule.)

**(d) "Group tokens by parent" is not what M-DCI does, and "a page" is not a set of tokens.**
With `DCI(head_dim, 1, 1, ...)` (`infer_state.py:478`), `num_indices = 1`, so `dci_insert_to_indices`
inserts the token into exactly one tree of its parent (`src/dci.c:708-760`), which is why the
mapping is a clean partition. But that tree is one of several *parallel trees over the same child
list*, one per projection index. There is no "semantic parent -> group its children" step in the
sense the proposal describes: the leaf is a **projection-band bucket of one DCI node's child list**.

Worse for the spec: the physical CPU page is `[head][K|V]` for all 8 KV heads, while a B+ leaf is
**one head's** 16 slots (leaf `dim` == `head_dim`). Page id `p` means a *different token set per
head*: `address_update` is called with `indices = tile(arange(num_leaves), (n_kv_heads,1))` and a
per-head address list (`infer_state.py:706-708`), and `_DCI_add` handles per-head `num_leaves`
divergence explicitly (`new_num_leaves = dci_db.num_leaves - self.prev_num_pages`,
`infer_state.py:840-854`). So the proposal's `token2page`, `token2offset`, `page_version`,
`page_dirty` are under-specified: they must be **per (KV head, token)**, not per token, and splits
happen per head independently.

---

## 2. Cost verdict: where the wall clock lands

### 2.1 The measured end-to-end numbers already exist

From `IceCache/benchmark/pred/pag_stage1_*.jsonl` (Qwen3-4B, 2x A100, hotpotqa rows 0 and 1,
11770 / 17305 tokens, page_size 16, page_budget 16 -> `n_dci_pages = 12`, `num_neighbours = 12`,
34 retrieval layers, 8 KV heads = 272 indexes; prefetch and reuse both off in this harness):

| run | prompt tokens | TTFT (s) | TPOT (s) | query p50 (ms/layer) |
|---|---|---|---|---|
| `dci` | 11770 | **6.22** | **0.190** | 1.94 |
| `dci` | 17305 | **7.59** | **0.169** | 1.95 |
| `pag_seq` (sequential PAG) | 11770 | **590.7** | 0.851 | 7.65 |
| `pag` (shared pool) | 11770 | **557.5** | 0.696 | 3.26 |
| `pag_fast` (best PAG) | 11770 | **202.0** | 0.613 | 3.18 |

Ratios, best PAG vs DCI: **TTFT 32.5x**, **TPOT 3.2x**. Worst: 95x / 4.5x. The query-cost
component alone (3.18 vs 1.94 ms x 34 layers) explains at most 42 ms/token of the 423 ms/token TPOT
gap, so the rest is pool serialization across 34 layers sharing 8 workers.

Critically: **these runs generated 5-7 tokens, so no window page was ever flushed** —
`offload_win_page_to_DCI` only fires when `kv_last_page_len + 1 >= page_size`
(`infer_state.py:529-534, 1178-1186`). The proposal's decode half (`search` + vote + split +
`insert_batch`) is therefore **not in any measured TPOT number**. Phase 1 is a pure query-side
measurement.

### 2.2 What the proposal's own additions cost

* **Prefill build** — PAG `Index.build` per (layer, KV head): 272 x 3.60 s at `max_search_k=128`
  = 979 s serial; the measured one-thread-per-tree line is 272 trees ~= 393 s; Phase 1 measured
  585 s end-to-end with a shared 8-thread pool. The mechanism is pinned down: in MIPS mode
  `pag.cpp:1144` sets `pif_entries_per_bucket = ComputeWorkingSetSize(max_search_k)`, and
  `updatePIFTableForInsertedPoint` (`paglib/pag_index_core.h:829-935`) runs a **65536-iteration
  double loop** (`pair_cols = (2*8)^2 = 256`, `table_cols = 256*256 = 65536`) for **every inserted
  point** (`paglib/pag_build_pipeline.inc:340-348`), taking a mutex and doing an O(max_search_k)
  sorted insert whenever a bucket's tail score is beaten. That is why build is linear in
  `max_search_k` and why `Metric.Cosine` (which skips PIF entirely) is flat.
* **Unbudgeted RAM**: that table is `65536 * max(10, max_search_k) * sizeof(PIFEntry=8B)`
  = **64 MiB per head at msk=128**, x272 heads = **~17 GiB**, plus 65536 `std::mutex` per head
  (`pag_index_core.h:727`) ~= 0.7 GiB more. At msk=256 it is ~35 GiB. On a box with 40 GB GPU and
  80 GB CPU budgets already in play (`benchmodel` defaults in `pag_stage1_compare.py:44-45`), this is
  a real line item nobody has costed.
* **Page formation (the new half)** — the proposal's greedy expansion needs one `search` per seed per
  fill step, because PAG exposes no graph (see §4). That is O(N) searches per layer where M-DCI
  does one bulk load. Even a *fully vectorized, PAG-free* version of the same algorithm costs
  **(measured here)** 2.7 ms for the 8-head `11712 x 11712` fp16 similarity matmul plus a 1.19-1.23 s
  pure-numpy greedy packing loop per layer (8 heads) — call it **~40 s per prompt across 34 layers**
  in Python, or a few hundred ms in C++. That is 6x DCI's *entire* 272-tree build cost, before PAG
  is involved at all.
* **Decode flush** — measured `insert_batch` of 16 points, one head: 2.2 ms at N=11776/msk=128.
  Per flush event, all 272 heads flush (the window pages advance together, `_prepare_decode` ->
  `offload_win_page_to_DCI`): 272 x 2.2 ms = **598 ms serial**, or ~75 ms across the 8-worker pool,
  added to that step's TPOT. The proposal *also* wants 16 `search` calls per head per flush
  (one per new key) = 4352 searches; at the measured ~0.4 ms/search that is another ~1.7 s of work,
  ~217 ms/pool-round. So every 16th decode step pays +75 to +290 ms against a 190 ms DCI TPOT.

### 2.3 Verdict, stated plainly

**The full-replacement design cannot match DCI on either axis; it cannot come within an order of
magnitude on TTFT.** Best case 32.5x TTFT and 3.2x TPOT, and that best case excludes the entire
page-formation and decode-insert machinery the proposal adds — which by §2.2 adds at least another
+20% TTFT and +12-150% on flush-step TPOT. No configuration of PAG parameters closes a 32x gap:
`ef_construction` 50->400 changes build by 3.18->3.92 s (1.2x), `projection_levels` 8->128 by
3.11->4.11 s (1.3x), static mode and extra capacity do nothing, and `max_search_k` is the only
strong knob and it moves the wrong way (128->32 gives 1.65 s but halves the candidate pool).

Two more cost facts the proposal should be forced to answer:

* PAG is **architecturally gated off** when IceCache's own speed features are on: `_DCI_query` only
  takes the PAG path when `self.n_prefetch_layers <= 1` (`infer_state.py:899-902`), and
  `_DCI_first_call` only creates a `PagPageSelector` when `check_reuse(cur_id) == 0`
  (`infer_state.py:693-718`). So "PAG is 3.2x on TPOT" is measured against a DCI baseline that has
  *already had its prefetch and reuse optimizations removed*.
* The whole Phase-1 quality evidence is **two samples with 5 and 7 generated tokens**
  (`pag_stage1_*.jsonl`: `generated_tokens: 7` and `5`), and the scores are identical to DCI
  (0.3333, 1.0). There is no accuracy evidence for PAG at all, in either direction.

---

## 3. Correctness hazards the proposal glosses over

**(1) Splits remapping old tokens while the recall thread reads CPU addresses — real, and the
contract is stricter than the proposal assumes.**
`recall()` runs on the asyncio worker thread and reads `self.page_address_buffer[layer_idx]`
(`infer_state.py:1033`) with no lock, while the main thread writes it in `_DCI_add`
(`infer_state.py:830-837, 854`). M-DCI survives this because of three invariants the proposal does
not state:
* page addresses are indexed by leaf id and leaf ids are reused from a free stack
  (`btree_p_clear` / `pop(stack)`, `src/btree_p.c:377, 702-712`), so `num_leaves` is monotone and
  addresses only *append*;
* content changes are reported back through `changed_page_list=ccc` (`infer_state.py:767-796`),
  and `ccc` is the only thing `diff_pages_by_head` trusts to decide whether a recall is needed
  (`infer_state.py:966`);
* `_apply_selected_pages` clears `ccc` for exactly the pages it just made resident
  (`infer_state.py:971-972`).
A 2-seed repartition that rewrites two existing pages must therefore set `ccc = 1` for **both**,
refresh `page_valid_entries` for both, and invalidate `cc2gp` for both — and must not change either
page's *id*, because `selected_page_idx[cur_id]` (`infer_state.py:958-967`) holds ids from the
previous step that are compared against the new selection inside a C function
(`DCI.diff_pages_by_head`). If an id survives with different contents and `ccc` stays 0, the kernel
reads a stale GPU page and attention silently attends to the wrong keys. M-DCI's own
`btree_p_bulk_load` never has this problem for the pages the *query* just touched only because the
`ccc` round trip is wired through `track=True`.

**(2) The page-id space and `cc2gp` / `selected_page_idx` / `page_valid_entries` / `diff_pages_by_head`.**
`cc2gp`, `ccc` and `page_address_buffer` are all indexed by CPU page id with capacity
`kvc_capacity = 1 << (max_num_leaves-1).bit_length()` (`infer_state.py:684`), and growth is performed
*only* on the DCI path, in `_DCI_add`, and only from `dci_db.num_leaves` (`infer_state.py:806-837`).
A PAG-owned page-id space has no such growth path. If a PAG split produces a page id >= capacity,
`kvc.cc2gp[b, head_ids, padded_arrays]` (`infer_state.py:964, 966`) indexes out of bounds on GPU.
The SemanticPageManager must therefore either (a) allocate ids in M-DCI's append-only leaf id space
(which means it owns a fake `num_leaves` and duplicates the growth code), or (b) own a new id space
and add a parallel growth path for `cc2gp`/`ccc`/`page_address_buffer`/`kvc_capacity`. The proposal
lists the arrays but not this requirement.

**(3) `num_leaves` — and therefore the CPU allocation — *can* change after allocation: yes, and the
proposal must preserve IceCache's contiguity assumption.** `_DCI_add` grows the CPU allocation with
`cpu_cache.decode_alloc_n_tokens(...)` (`infer_state.py:806-810`) and the *only* reason the raw
pointer arithmetic works is
`assert cpu_cache[b, -1].data_ptr() - _base == (max_num_leaves - 1) * stride`
(`infer_state.py:670`), combined with `_base + i*offset + j*stride` (`infer_state.py:698-708`) and
`DCI.copy_to_buffer(self._src_address_buffer, ...)` copying from raw addresses
(`infer_state.py:1038-1042`). That assert is checked **once**, at first call. Any page allocator that
interleaves other allocations into the run, or that reuses freed CPU pages (the `alloc_page` path
frees pages, `infer_state.py:1242-1260`), silently corrupts every recall. This is a hard constraint
on the SemanticPageManager, not a detail.

**(4) GPU-side masking over a non-position-contiguous partition — actually fine; the proposal can
claim this one.** The flashinfer mask is
`is_valid = (page_offset < page_valid_entries[(page_idx)*num_heads + head_idx])`
(`3rdparty/flashinfer/include/flashinfer/attention/decode.cuh:113-119`), i.e. a per-(page, head)
slot count with no positional assumption, and `page_valid_entries` is refilled from
`get_valid_entries(selected_page_idx)` each step (`infer_state.py:1099-1103`;
`src/py_dci.c:2212-2251` returns `num_slots_used` per leaf, `-1` for unallocated). A semantic
partition whose pages are not position-contiguous is supported by the kernel as written. Note the
init value is `page_size` for all pages (`infer_state.py:394-398`), so a PAG page manager **must**
maintain a correct per-(page, head) occupancy, or short pages will have tail slots attended over:
that is exactly the 15/16-slot case in §1.2(a).

**(5) GQA and the 4 q-heads per KV head.** The partition is per KV head, and `cc2gp` is
`[batch, n_kv_heads, cap]` (`infer_state.py:685`), so the structure is compatible. Two specifics:
the selector must return exactly `(n_kv_heads, num_neighbours)` — `_apply_selected_pages` raises
`ValueError('Page selector returned the wrong shape')` otherwise (`infer_state.py:952`) — so any
unioning across the 4 q-heads of a group must happen inside the selector (DCI does it with
`first_k_unique`, `infer_state.py:935-939`); and the 4 q-heads of a group compete for the same 12
pages, so a per-head semantic packing cannot be evaluated per q-head.

**(6) `check_reuse` / layer reuse — this is the sharpest unaddressed hazard.** With
`n_reuse_layers > 0` the page-id space is *shared across layers*: layer `cur_id` clones
`kvc.ccc` from the source layer (`infer_state.py:802`), reuses `prev_eids`/`prev_rids`/`prev_nr`
(`infer_state.py:1087-1091`), and the only per-layer difference is a constant offset added to GPU
page ids (`infer_state.py:1085-1089`). That only works because the CPU page id -> token-set mapping
is **identical for every layer in a reuse class**. PAG's packing is derived from each layer's own
keys, so layer `i`'s semantic pages differ from layer `reuse_id`'s. The reuse path would then copy
layer `reuse_id`'s K/V into pages whose semantic membership belongs to layer `i`. The existing code
does not even try: the `PagPageSelector` construction is inside
`if self.check_reuse(cur_id) == 0:` (`infer_state.py:693-718`), and `_DCI_query` falls back to DCI
whenever `self.pag_selectors[cur_id] is None` (`infer_state.py:901-902`). So the proposal is only
coherent with reuse disabled — and reuse is one of IceCache's TPOT optimizations. The honest
statement is "PAG-based paging forfeits layer reuse and layer prefetch", not "IceCache is unchanged".

---

## 4. Is the page-formation algorithm even well-defined?

**No, not as written.** Four gaps:

**(a) PAG does not expose the graph.** Confirmed: `PAG/python/pag/__init__.py` exports exactly
`BuildOptions, Index, IndexMode, LoadOptions, Metric, SearchOptions`, and `Index` binds only
`build, load, save, search, search_with_options, add, insert, add_batch, insert_batch, is_loaded,
metric, dimension` (`PAG/python/pag_bindings.cpp:212-232`). There is no neighbor accessor, no
clusterer, no adjacency export. "Expand from a seed along high-similarity edges" is therefore
implemented as `search(seed, top_k=R)` — i.e. one full ANN query per seed per fill step — which is
what makes it cost O(N) searches per layer. The proposal does not say this, and it is the reason the
393 s figure is the *floor* for page formation, not the ceiling.

**(b) With what similarity does the expansion walk?** This is not a detail under MIPS. Inner product
is not a metric: there is no symmetry, no triangle inequality, and "the neighbor of my neighbor" is
ill-defined. The proposal indexes keys under `Metric.MaximumInnerProduct` and then talks about
"high-similarity edges" as if the graph were an L2 neighborhood graph. In MIPS-mode PAG the
`search` result is by inner product, so the expansion would be walking directed,
non-transitive edges.

**(c) The proposal's stated ordering key is the wrong one, and I measured it.** "Sort within the
parent by projection value" — I built exactly that partition on real keys (sort all indexed tokens
by a single random projection of the key, group 16 consecutive) and it is the *worst* non-random
layout, *worse than position-contiguous pages*:

Page recall@12 (= 192-token budget, matching `n_dci_pages - topk = 12`), Qwen3-4B, hotpotqa row 0
(11770 prompt tokens -> 11712 indexed), post-RoPE `k_norm`/`q_norm` keys and queries, layer 2 and
layer 20 (both retrieval layers), queries = last 32 prompt positions x 4 q-heads = 128 query
vectors per KV head, average over 8 KV heads. DCI numbers are from a real
`DCI(num_points=11712)` build + real `db.query(num_neighbours=12)`. **(measured here, `/tmp/pagprobe/pages.py`)**

| page partition | pages | recall@12 L2 | recall@12 L20 |
|---|---|---|---|
| random 16-groups | 732 | 0.416 | – |
| projection-sorted 16-groups ("sort by projection value") | 732 | **0.452** | **0.592** |
| position-contiguous 16-groups | 732 | 0.583 | 0.813 |
| **DCI B+ leaf (real build + real query)** | 818 | **0.700** | **0.789** |
| exact-kNN greedy expansion from a seed | 732 | **0.832** | **0.883** |

Two things fall out. First, the proposal's literal ordering rule is a regression
(0.452/0.592 vs 0.700/0.789): a single projection sort is a strictly coarser object than M-DCI's
*lifted* projection sort applied *inside* each node's child list. Second, **the proposal's actual
algorithm — greedy expansion from a seed along high-similarity edges — does work, and works well**:
its exact-neighborhood idealization reaches 0.832/0.883, i.e. **+19% / +12% relative** over DCI.
The algorithm is well-defined once you replace "graph edges" with "exact top-R neighbor list", and
the seed order matters (I used descending `||k||`); with only a 64-candidate list per seed the same
code degenerates to 9845 pages for the same 11712 tokens instead of 732 **(measured here,
`/tmp/pagprobe/paired.py`)** — an implementation shortcut that silently destroys the partition. PAG's
hard `top_k <= max_search_k` cap is exactly this failure mode at scale.

**(d) Is key->key inner product a good clustering geometry?** Partly. On raw cosine, DCI's own
leaves are only weakly clustered: mean within-leaf pairwise cosine 0.041 vs 0.000 +/- 0.090 for
random pairs (synthetic Gaussian keys, N=11770) — about 0.45 sigma **(measured here)**. On real
keys, the honest answer comes from the page-recall experiment: a *query-aware* partition does
dramatically better than DCI, which means the geometry question is really a *usage* question, not a
*similarity* question:

| partition (paired, same 16 held-out queries/head) | L2 | L20 |
|---|---|---|
| DCI leaves, 12 pages | 0.635 | 0.705 |
| DCI leaves, **13** pages (equal token budget, controls for DCI's 14.4/page occupancy) | 0.660 | 0.734 |
| co-occurrence-trained packing, 12 pages, trained on the other 16 queries | **0.867** | **0.879** |

**(measured here, `/tmp/pagprobe/ceiling.py`)** The occupancy confound is worth only +4% relative
(0.635->0.660), so the remaining **+31% / +25% relative is a genuine partition-quality gap**. That is
the real ceiling of this whole direction, and it is reachable.

---

## 5. The two strongest arguments

**Strongest argument that the full-replacement direction is a dead end (quantitative):**
The measured, best-case, fully-integrated PAG run is **202.0 s TTFT vs 6.22 s (32.5x)** and
**0.613 s TPOT vs 0.190 s (3.2x)** — and that run uses **DCI's pages**, so it prices only the ANN
swap, not the page-formation swap that is the actual idea. The page-formation swap buys at most
+31%/+25% relative page recall (0.635->0.867 at L2, 0.705->0.879 at L20, paired, occupancy-controlled),
and that ceiling is an *oracle* that is trained on the query distribution. No accuracy gain of that
size can be worth 32x TTFT when the baseline is already at 0.64-0.70 page recall and the paper's
headline claim rests on page *layout*, not page *recall*. Moreover the gain is not PAG-specific:
I obtained a better-than-DCI partition (0.832/0.883 vs 0.700/0.789) with exact kNN-derived
neighborhoods costing 2.7 ms of fp16 matmul plus a ~1.2 s/layer numpy loop, and PAG's MIPS
recall@64 against exact ground truth is 0.89 — a *worse* neighborhood than the exact one I used.
Paying 32x TTFT to get an approximate version of a neighborhood you can compute exactly for
~40 s/prompt is not a design, it is an inversion.

**Strongest argument that it is not a dead end:**
There is real, measured, unexploited headroom in page *formation*, and it is the only lever in this
system with that much headroom. DCI's pages are 0.700/0.789 recall; an ideal co-retrieval partition
is 0.867/0.879; the gap is worth roughly a third of the retrieval budget at fixed 12 pages x 16
slots, on real Qwen3-4B keys, and it survives an equal-token-budget control. Page formation is also
cheap *in principle*: it is a clustering problem over data you already have on the GPU at prefill
time, not an online ANN problem, and an exact solution costs 2.7 ms of matmul per layer. The
proposal's own §Prefill idea — greedy expansion from a seed along high-similarity edges — is the
right *shape* of algorithm and demonstrably works (0.832/0.883). The dead part is the *bundling*:
attaching page formation to a PAG index swap forces the 393 s build, the 17 GiB PIF tables, the
loss of prefetch and reuse, and the decode-time insert/query costs. Keep the page idea, drop PAG.

---

## 6. The cheap decisive experiment (well under an hour, no integration code)

I ran this; it takes ~10 minutes wall clock and touches nothing in `IceCache/source/`.

**Recipe.**
1. One forward pass of Qwen3-4B on one hotpotqa prompt (11770 tokens) with a
   `register_forward_pre_hook(with_kwargs=True)` on `model.model.layers[L].self_attn` for two
   retrieval layers (L=2, 20), re-deriving `q_norm(q_proj(h))` / `k_norm(k_proj(h))` and applying
   `icecache.adapter.modeling.apply_rotary_pos_emb` so the tensors match what
   `infer_state.py:602-604` feeds DCI. Keep only K and Q. (~90 s including model load.)
2. Slice exactly as IceCache does: `start = n_sink_pages*16`, `end = (n_kv_pages - n_win_pages)*16`
   (`adapter/modeling.py:130-132`) -> 11712 indexed tokens.
3. Build **one** DCI per layer (`DCI(128, 1, 1, num_points=11712, num_inst=8, transform=True)`,
   0.1-0.55 s) and run the real `db.query(num_neighbours=12, field_of_view=30,
   prop_to_retrieve=0.8)` to get DCI's 12 pages per query.
4. Compute exact top-192 per query (one `11712x11712` fp16 matmul, 2.7 ms) and evaluate
   `recall@12` = fraction of the exact top-192 that lives in the 12 selected pages, with the
   aggregation rule `S(P,q) = max_{i in P} q.k_i` (`pag_retrieval.py:176-195`).
5. Compare, on identical queries and identical 16-slot pages:
   DCI leaves / projection-sorted / position-contiguous / exact-kNN-greedy / random / a
   co-occurrence-trained partition (train on half the queries, test on the other half).
6. One extra control: DCI at 13 pages, to price out its 14.4-slot occupancy.

**Why it is decisive.** It separates the two questions that the proposal conflates:
*"does a better page partition exist?"* and *"does PAG find it?"*. Step 5 answers the first
(my answer: yes, +25-35% relative, and the proposal's own greedy rule captures most of it). Step 6
answers the second: build **one or two** PAG indexes (3.6 s each at msk=128) and run the same
recall@12 evaluation using `index.search(...)` token labels mapped through the greedy partition.
If PAG's neighborhoods land at or below the exact-kNN number, PAG buys nothing that an exact matmul
does not, and the entire ANN swap is refuted on quality grounds *independently* of its 32x cost.
If PAG lands above it, you have the only evidence that would justify re-opening the cost question.

**Secondary experiment worth folding in (5 minutes):** run the same recall@12 for the *position
contiguous* partition and report it next to DCI's. I measured 0.583 (L2) and **0.813 (L20)** —
position pages beat DCI's semantic pages at layer 20. If that reproduces across layers, it says
part of IceCache's headline "semantic pages" gain at deep layers is being provided by the window
pages anyway, and is a much cheaper thing to fix than an ANN swap. Queries at the very end of the
prompt are confounded by recency, so repeat with queries drawn from generated positions.

---

## Appendix: exact commands / artifacts

* M-DCI source under review: `/tmp/icecache-mdci-source/`
* Phase-1 paired results: `IceCache/benchmark/pred/pag_stage1_{dci,pag,pag_seq,pag_fast}.jsonl`
* Phase-1 adapter under review: `IceCache/source/icecache/pag_retrieval.py`
* Integration points: `IceCache/source/icecache/infer_state.py` lines 294-303 (`check_reuse`),
  394-398 (`page_valid_entries` init), 463-488 (reset + DCI alloc), 597-708 (`_DCI_first_call`),
  730-869 (`_DCI_add`), 872-947 (`_DCI_query`), 949-974 (`_apply_selected_pages`), 1016-1052
  (`recall`), 1099-1103 (`get_valid_entries`), 1115-1152 (`prefill_backup_pages`),
  1178-1186 (`offload_win_page_to_DCI`), 1242-1260 (`alloc_page`)
* Kernel mask: `IceCache/3rdparty/flashinfer/include/flashinfer/attention/decode.cuh:113-119`
* PAG API surface: `PAG/python/pag/__init__.py`, `PAG/python/pag_bindings.cpp:212-232`
* PAG MIPS build cost mechanism: `PAG/pag.cpp:140-145, 1144, 1163-1180`;
  `PAG/paglib/pag_index_core.h:690-728, 829-935`; `PAG/paglib/pag_build_pipeline.inc:340-348`
* My scripts and raw numbers: `/tmp/pagprobe/{extract.py,pages.py,ceiling.py,timing3.py}` and
  `/tmp/pagprobe/{keys.npz,partial.json,ceiling.json}`. The extracted K/Q are post-RoPE,
  post-`k_norm`/`q_norm`, fp32, from
  `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`, hotpotqa (LongBench) row 0.
  **All page-recall numbers above are from this single prompt.** They are single-prompt
  measurements and should be replicated on 3-5 prompts before being treated as established.
