# Design A — replace M-DCI with exact-kNN page packing + a matmul page scan

Decision: replace DCI's **tree build** and **tree retrieval** entirely, keep the semantic-page
idea, and do **not** use PAG as the retriever. Rationale and all measurements:
`experiment/page_recall_correction.md`. Summary of why:

* DCI's B+ leaves give honest per-query recall@12 of **0.2777 (L2) / 0.2415 (L20)**.
  Exact-kNN greedy packing gives **0.4137 / 0.4059** — **+49% / +68% relative**.

  > ⚠ **Definition caveat.** Those four numbers score a page by its **best individual
  > member**, not by the **stored page representative**. The shipped scan uses the
  > representative (the mean of the page's members), and on that definition the same
  > comparison reads **0.3144 / 0.3196 → 0.4721 / 0.4710**, i.e. **+50.2% / +47.4%**.
  > The direction and the conclusion are unchanged, but the two L20 figures differ
  > materially, and this document elsewhere uses "page recall" for three different
  > quantities — see `page_recall_correction.md` for the full disambiguation.
  > The corrected L20 pair leans on a DCI mean of **0.3196 that is single-sourced from
  > `pag_pages2.py`**; re-derive it before quoting it as a hard number.
* A matmul scan over 732 page representatives is **0.1544 ms/head**, i.e. **1.24 ms/layer**
  serial over 8 heads, against DCI's **1.94 ms/layer**. Decode-side win, and exact.
* PAG at page scale was 0.3161 ms/head (8 OpenMP threads) and ranked pages 3-28% worse than
  the same scan; its build is 10.4 s/prompt against DCI's 1.0 s. It loses on both axes.

So the new retrieval structure is **an exact matmul over page representatives**, and the new
page structure is **capacity-constrained greedy expansion along an exact kNN**.

---

## 1. What is replaced

| current | new |
|---|---|
| `DCI` B+ tree per (layer, KV head), `num_inst=8` | per-(layer, KV head) `token2page` array, built by greedy packing |
| `_DCI_first_call` -> `add_query_at_end` | `PageScan.build(...)` at prefill |
| `_DCI_add` -> `add_query` at decode | `PageScan.insert(...)` at decode |
| `_DCI_query` -> `db.query(...)`, 1.94 ms/layer | `PageScan.query(...)`, ~1.24 ms/layer |
| `dci_db[cid].token2node` (leaf, slot) | `PageScan.token2page`, `PageScan.offset_in_page` |
| `dci_db[cid].get_valid_entries(...)` | occupancy array, maintained exactly |
| `dci_db[cid].num_leaves` | `n_pages`, **fixed** at `ceil(N/16)` |

DCI's tree is gone: no B+ tree, no promotion, no leaf split, no `max_sq_norm` re-sort.

## 2. Page formation (prefill)

Per (layer, KV head), over the `N` indexed tokens (`N` = tokens between the sink and
window pages; 11712 for a 11770-token prompt):

```
seed order  = tokens sorted by descending ||k||
for seed in seed_order:
    if assigned[seed]: continue
    page = [seed]; assigned[seed] = True
    while len(page) < 16:
        cand = argmax_{t not assigned} <k_seed, k_t>
        page.append(cand); assigned[cand] = True
    emit page
```

* Exactly `ceil(N/16)` pages, **16 slots each except possibly the last**. Fixed count.
* `O(N^2)` similarity matmul per head (2.7 ms at N=11712 on GPU 0) + the expansion loop.
  **Cost measured: 17.2 s for all 8 heads of one layer** (Python/torch loop, the dominant
  prefill cost of the new design). Batched over the 8 heads of a layer.
* Reps: `rep[p] = mean of keys in page p` (float32). **Mean, not medoid** — measured, the mean
  beats both the medoid and max-over-members in every layer/partition
  (`experiment/page_recall_correction.md` §3).

Requirement: a GPU-resident implementation that keeps the 8 heads of a layer batched, so the
loop is over `ceil(N/16)` seeds, not `N`. Target: <= 3 s/layer for all 8 heads.

## 3. Retrieval (decode)

At each decode token, per (layer, KV head), for the `n_qo_heads` queries (32 for Qwen3-4B):

```
scores = q @ rep.T                      # [128, n_pages]
pages  = topk(scores, budget=12)        # per q-head
# then the existing GQA dedup: group 4 q-heads per KV head, take first 12 unique
```

Then the **existing** `pag_retrieval.aggregate` is *removed*; the deployed scoring rule stays
`S(P, q) = max_{i in P} q·k_i`, which is what `_apply_selected_pages` feeds to `scatter_pages`.

`n_pages = 732`, so `q @ rep.T` is `[128,128] @ [128,732]` = 0.1544 ms/head, 1.24 ms/layer.
Do **not** introduce an index.

## 4. Decode-time insert

Every 16 decode tokens, `infer_state.py:533` flushes a window page for all 34 layers
(`offload_win_page_to_DCI` -> `_DCI_add`), 16 new tokens per KV head.

New rule, per (layer, KV head): assign each of the 16 new keys to the page whose rep has the
highest inner product **and which has a free slot**; if none, emit a new page. This replaces
DCI's leaf split. Consequences to get right:

* The page-id space **must be pre-reserved** so it never grows at decode. Reserve
  `ceil((N + n_decode_budget)/16) + slack` pages at prefill and allocate the CPU buffer for
  that once. This keeps the one-shot contiguity assert at `infer_state.py:670` true.
* `page_valid_entries` (`infer_state.py:394-398`) inits to `page_size` everywhere. It must be
  maintained per (page, head) — the GPU mask
  (`3rdparty/flashinfer/include/flashinfer/attention/decode.cuh:113-119`) reads
  `is_valid = page_offset < page_valid_entries[...]`, so a stale `page_size` makes the kernel
  read uninitialised slots.

## 5. Layer reuse — preserve DCI's semantics exactly

`check_reuse(cur_id)` (`infer_state.py:294-303`) returns a source layer id or 0. DCI builds a
structure **only** when it returns 0; otherwise the layer aliases the source layer's
`dci_db`. Design A does the same: reuse layers alias the source layer's
`token2page`/`reps`/`n_pages`, and `_DCI_query`'s reuse branch
(`infer_state.py:1084-1092`), which shifts `eids` by a constant `c2p` offset and asserts
`(c2p[L] - c2p[reuse_id]) == offset`, must keep working. `run_longbench.sh:27` passes
`--n_reuse_layers 3`, so ~1/3 of layers build.

Because every head has exactly `ceil(N/16)` pages, the per-head page count is identical, so
the CPU page-id space is aligned across heads exactly as DCI's `num_inst=8` layout is.

## 6. Integration points (`IceCache/source/icecache/infer_state.py`)

| line | function | change |
|---|---|---|
| 478 | `DCI(...)` alloc | replace with `PageScan` alloc |
| 597-708 | `_DCI_first_call` | build partition + reps, CPU layout |
| 730-869 | `_DCI_add` | resolve the reuse branch, then `PageScan.insert` |
| 872-947 | `_DCI_query` | `PageScan.query`, keep the GQA dedup at 935-939 |
| 949-974 | `_apply_selected_pages` | keep; source pages from `PageScan` |
| 1016-1052 | `recall` | unchanged |
| 1099-1103 | `get_valid_entries` | read the maintained occupancy array |
| 1178-1186 | `offload_win_page_to_DCI` | -> `PageScan.insert` |
| 1242-1260 | `alloc_page` | pre-reserved page-id space |

Gate the new path behind `retrieval_backend == "page_scan"` **in the same shape as the existing
`pag_mips` branch** (`infer_state.py:901-914`, `pag_retrieval.py`), so the DCI path stays
runnable as the baseline arm and the two can be compared.

## 6b. Corrections from independent review (READ THESE — they override §1-§6 where they conflict)

**C1. Two different page counts, and the spec must not conflate them.** §1 says `n_pages` is
fixed at `ceil(N/16)`; §4 says the id space is pre-reserved for decode. Both are needed and
they are *different numbers*:
* `n_pages_built` = `ceil(N/16)` (732). These are the pages that exist and are selectable.
* `n_pages_reserved` = `ceil((N + decode_budget)/16) + slack`. The CPU address space.

`query` must slice `reps[:n_pages_built]` and only ever return ids `< n_pages_built`. Do **not**
rely on unbuilt reps scoring low: their reps are zero/NaN and can win `topk` outright. Ids
`>= n_pages_built` also have no CPU address, and `recall` (`infer_state.py:1033`) would copy
from `(uintp)-1`. `n_pages_built` grows only when `insert` emits a new page, and every id up to
`n_pages_reserved` must have a valid CPU address for that to be legal.

**C2. `get_valid_entries` must never return `-1`.** It is stored into an int32 tensor
(`infer_state.py:1099`, `**self._i32`) but the kernel parameter is `const uint32_t*`
(`3rdparty/flashinfer/include/flashinfer/attention/decode.cuh:81`). `-1` becomes `0xFFFFFFFF`,
so `page_offset < valid` is true for **all 16 slots** and the kernel attends uninitialised KV —
silently, with no assert. Return `0` for any page id that is negative or `>= n_pages_built`.

**C3. `num_neighbours` is per-layer, never a constant.** It is
`self.n_dci_pages - self.layer2topk[cur_id]` (`infer_state.py:951`, `:877`), and `layer2topk`
defaults to `budget // 2` when `page_topks is None` (`:124-125`). It is 12 only because both
harnesses pass `--page-topk 0`. Hardcoding 12 breaks the shape assert at `:952` under any other
`--page-topk`. Read it from the same expression DCI uses.

**C4. Reuse layers must be gated identically to DCI, including the insert.** `_DCI_add` skips
its whole body when `check_reuse(cur_id) != 0` (`infer_state.py:745`, `:856`); the reuse layer
reads the source layer's aliased structure. Design A's insert is a function of the layer's own
reps and keys, so if a reuse layer recomputes its own assignment it will silently diverge from
the source's layout with no assert. Mirror the gating exactly.

**C5. The last partial page is a PREFIX COUNT.** `page_valid_entries` is read as
`page_offset < valid`, so a page's `m` members must occupy slots `0..m-1` with **no holes**.
The initialisation to `page_size` everywhere (`infer_state.py:394-398`) is a lie that must be
overwritten for every page. And the build loop must iterate `n_pages_built`, not
`n_pages_reserved` — iterating the reserved count makes `argmax` on an all-assigned row return
0 and silently re-assign tokens.

**C6. The 16-step expansion loop collapses to one `topk`.** The seed's score row is fixed for
the whole page, so "argmax, mask, argmax, ..." over 16 steps is provably identical to
`topk(16)` on that row. Token-for-token identity was verified. Verbatim pseudocode costs
**2.222 s/layer** batched; the collapsed form costs **0.272 s/layer** — an 8x saving that is
the entire margin on the §2 budget. Prefer the collapse; if it is not used, say why.

**C7. Claim 2's target is backend-dependent.** The scan is **0.0368 ms/head on GPU** but
**0.1489 ms/head numpy-CPU** (the §3 budget of 0.13 ms is therefore *false* on numpy). Since
`_DCI_query` currently receives a CPU numpy array (`infer_state.py:889`), the CPU figure is the
one that applies today: budget **<= 0.16 ms/head numpy-CPU**, which is still 8 x faster than
DCI per head. Moving the scan to GPU is a worthwhile follow-up, not a requirement.

## 7. Hazards that must not be reintroduced

1. `infer_state.py:670` asserts CPU-page contiguity **once**. Never free or reuse a CPU page
   id, and never let the page count change after prefill.
2. `page_valid_entries` must be true per (page, head), not initialised to `page_size`.
3. `_apply_selected_pages` asserts `nn_idx_0.shape == (n_kv_heads, num_neighbours)` with
   `num_neighbours = n_dci_pages - layer2topk = 12`; the scan must return exactly that many
   distinct pages per KV head, and must never return `-1` for a live slot.
4. A page with fewer than 16 members must not have its members read from slots it does not
   own.
5. The partition depends on each layer's own keys, so a **reuse layer must not build**; a
   mismatch here silently corrupts recall.
