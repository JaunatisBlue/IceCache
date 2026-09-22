# Paged-PAG: replacing the DCI tree with PAG while preserving semantic pages

Status: design blueprint. Branch `algorithm`. Target files:
`IceCache/source/icecache/infer_state.py`, `IceCache/source/icecache/pag_retrieval.py`,
`IceCache/source/icecache/adapter/modeling.py`.

---

## 0. What a "page" actually is in the current code (verified against the C source)

IceCache's comments say "one DCI tree node == one physical page". That is wrong. Reading
`/tmp/icecache-mdci-source/`:

* `include/btree_common.h:21` — `#define BTREE_LEAF_MAX_NUM_SLOTS 16`, equal to the
  IceCache default `page_size` (16).
* `src/btree_p.c:672` `btree_p_bulk_load()` — takes a **sorted** `(key, data)` sequence and
  chops it into `ceil(num_items / leaf_max_num_slots)` leaves, filling leaf `i` with
  `num_items / (num_leaves - i)` consecutive entries. So **a leaf is a contiguous run of the
  sorted order, cut at 16**.
* `src/btree_p.c:555` `btree_p_split_leaf()` — when full, splits at `mid = num_slots_used >> 1`,
  allocates `newleaf->id = (*num_leaf_nodes)++` (i.e. **at the end, never reusing an id during
  online insert**), and rewrites `token2nodeIndex[id]/token2nodeOffset[id]` for every moved
  token.
* `src/dci.c:1714-1752` `update_max_sq_norm()` — the sort that feeds `bulk_load`: first
  `qsort` by `parent_id` (the DCI cell = sign pattern of the projections), then, per parent and
  per projection index `k`, `qsort` by the projection value `local_dist[k]`, then
  `btree_p_bulk_load()` per `(parent, k)`.

So the real chain is exactly the one in the brief: **semantic parent (sign-of-projection cell)
→ tokens grouped by parent → sorted within parent by projection value → bulk-loaded → leaf of
≤16 → `token2nodeIndex`/`token2nodeOffset` → `num_leaves` → CPU pages bound via
`address_update`**. The leaf is the page.

Consequences that the design below depends on and that are *not* obvious from the Python:

1. **Page ids are append-only during decode.** `_DCI_add` (`infer_state.py:840-854`) computes
   new-page CPU addresses with
   `tmp_new_indices = np.arange(prev_num_pages[inst], num_leaves[inst])`, i.e. it assumes every
   new leaf id forms a contiguous run at the *end* of the id space. `page id == CPU page index`.
2. **The KV bytes are physically permuted into leaf order at prefill.**
   `btree_p_bulk_load` ends with `memcpy(leaf->data_loc, temp_data_loc + cumulative*dim, ...)`
   (`src/btree_p.c:753-760`). This is the "data rearrangement" comment at
   `infer_state.py:696`. `prefill_backup_pages()` (`infer_state.py:1115`) only does a bulk
   GPU→CPU copy in *token* order; DCI then permutes.
3. **A split moves KV bytes.** `btree_p_split_leaf` copies `data_loc`/`inc_data_loc` for the
   moved slots. Any PAG design must do the same move in CPU memory.
4. **`num_leaves` is per-head.** `num_inst == n_kv_heads`, `dci_db.num_leaves` is an array of
   `n_kv_heads` counts; `_DCI_first_call:665` uses `num_leaves.max()` to size the CPU region.

### 0.1 The real numbers this design must hit

From `IceCache/benchmark/longbench_pred.py:36-51` (defaults) and
`IceCache/source/icecache/infer_state.py:102, 124-129, 551-552, 877, 1128-1131`:

| symbol | value | where |
|---|---|---|
| `page_size` | 16 | `--page-size 16` |
| `budget` | 16 pages | `--page-budgets 16` |
| `n_sink_pages` / `n_win_pages` | 2 / 2 | `--n-sink-pages 2 --n-win-pages 2` |
| `offload_ratio` | 2 | `infer_state.py:102` |
| `page_topk` (= `layer2topk`) | **0** | `--page-topks 0` (default), `infer_state.py:126-127` |
| `n_dci_pages` | **12** | `infer_state.py:1130-1131`, then `:551-552` |
| **`num_neighbours`** | **`12 - 0 = 12`** | `infer_state.py:877, 951` |
| `ratio` | 4 (32 qo heads / 8 kv heads) | Qwen3-4B |
| retrieval layers | 34 (`n_unlimited_layers=2`, 36 layers) | `--n-unlimited-layers 2` |
| **PAG indexes per prompt** | **34 x 8 = 272** | |

The `page_topk = 0` step is the one that is easy to get wrong. `infer_state.py:124-125` uses
`budget // 2 = 8` **only when `page_topks is None`**. The benchmark declares `--page-topks` with
`default=0` (`longbench_pred.py:41`; likewise `gsm8k_pred.py:65`, `passkey_pred.py:169`) and no
`run_*.sh` overrides it, so the `elif` branch at `infer_state.py:126-127` fires and
`layer2topk = 0` for every retrieval layer (0 passes the assert at `:156`). For a 11770-token
prompt (`n_real_pages = ceil(11770/16) = 736`) the prefill sizing at `infer_state.py:1128-1131`
gives

```
num_offload_pages = min((736 - 16) * 2, 736 - 2 - 2) = min(1440, 732) = 732
n_dci_pages       = min(732 - (736 - 16), 16 - 2 - 2) = min(12, 12)   = 12
kvc.n_win_pages   = 16 - 12 - 2 = 2
```

and the decode-time recomputation at `infer_state.py:551-552` agrees:
`n_dci_pages = budget - n_sink_pages - n_win_pages = 16 - 2 - 2 = 12`. So

**`num_neighbours = n_dci_pages - layer2topk = 12 - 0 = 12`**: the retrieval budget is **12 pages
per KV head = 192 tokens per head**, i.e. `12 * 16 / 11776` = **1.6% of the context** per head
per step.

This is the hardest constraint in the design: `_apply_selected_pages` (`infer_state.py:952`)
asserts

```python
if nn_idx_0.shape != (self.n_kv_heads, num_neighbours):
    raise ValueError('Page selector returned the wrong shape')
```

so every path must yield **exactly 8 x 12 page ids**. Two consequences the rest of the document
leans on:

* `page_valid_entries[layer][ns : ns + n_dci_pages - topk]` = `[2 : 14]` — the **entire** GPU
  offload region is rewritten every decode step (`infer_state.py:1099-1103`). There is no sticky
  `topk` prefix, so retrieval is a full re-selection each step and a misranked page costs exactly
  one step. Errors are transient, not sticky.
* A page-level selector must surface **12 distinct page ids from a single `search` call**. At
  page granularity each indexed point is its own page, so distinctness is guaranteed for
  `top_k >= 12`; at token granularity it is not (see 4.2).

### 0.2 The `_apply_selected_pages` contract (read it, do not change it)

`infer_state.py:949-974`. Input `nn_idx_0`: numpy **int32, shape `(n_kv_heads, 12)`**, values
are page ids, ordered best-first (position `i` is bound to GPU slot `arange(ns, ns+12)[i]` in
the first-call branch). It returns `(evicted_idx, recall_idx, evict_num)`, all CUDA int32,
consumed by `modeling.py:277-288` -> `scatter_pages()`. `recall()`'s transit buffers are sized
`n_kv_heads * (budget - ns - nw) = 8 * 12 = 96` pages (`infer_state.py:417-433`), which matches
`num_neighbours` exactly — no slack, so the selector must not over-produce.

Two of its callees are DCI C statics and **must be ported** because PAG replaces DCI entirely:

* `DCI.diff_pages_by_head(A, B, mask, pid)` — `src/py_dci.c:2710-2793`. Pure `O(heads * 12^2)`
  logic. Semantics: `out_A` = pages in `A` not present in `B` (or present but
  `ccc`-dirty) -> **must be re-recalled**; `out_B` = the GPU slots freed, aligned index-wise
  with `out_A`; `out_O` = the new `selected_page_idx`, a position-preserving merge of retained
  `B` entries and new `A` entries. **In-place side effects on `pid` (= `kvc.cc2gp`):**
  `pid[evicted_cpu_page] = -1`, then `pid[newly_recalled_cpu_page] = freed_gpu_slot`.
* `DCI.get_valid_entries(leaf_ids)` — `src/py_dci.c:2212-2248`. Returns
  `(n_heads, k)` int32 of `leaf.num_slots_used`, `-1` for `leaf_id < 0`.

`ccc` semantics (this is what makes splits correct): `kvc.ccc[b, head, cpu_page] == 1` means
**the GPU copy of that CPU page is stale**. It is set by the writer (`_DCI_add` passes it as
`changed_page_list` to DCI, `infer_state.py:792`) and cleared by the reader
(`_apply_selected_pages`, `infer_state.py:972`). `diff_pages_by_head` consults it to decide
"still selected but dirty -> treat as newly selected -> recall". A page split is *exactly* the
case this bit was built for.

---

## 1. `SemanticPageManager`

One instance per layer (mirroring one `DCI(num_inst=n_kv_heads)`), owning `n_kv_heads`
independent partitions. New file: `IceCache/source/icecache/semantic_pages.py`.

### 1.1 Data structures

```python
class SemanticPageManager:
    layer:   int
    n_heads: int          # 8
    page_size: int        # 16  (== IceCache page_size; NOT configurable)
    head_dim: int         # 128
    capacity: int         # max pages per head, power of two, grows like kvc_capacity

    # --- the partition (token -> page) ---
    token_page:   np.ndarray  # (n_heads, N_max) int32   -1 = not yet assigned
    token_offset: np.ndarray  # (n_heads, N_max) uint8   slot inside the page
    n_tokens:     np.ndarray  # (n_heads,) int32         high-water mark of assigned tokens

    # --- the pages ---
    occupancy:    np.ndarray  # (n_heads, capacity) int16, 0..16
    page_state:   np.ndarray  # (n_heads, capacity) uint8  0=EMPTY 1=OPEN 2=SEALED
    page_norm:    np.ndarray  # (n_heads, capacity) float32  ||medoid|| for MIPS scoring
    page_medoid:  np.ndarray  # (n_heads, capacity) int32   token id of the medoid (-1 if none)
    n_pages:      np.ndarray  # (n_heads,) int32            next free page id, monotone
    epoch:        int         # bumped on every publish; readers seqlock on it
    _lock:        threading.Lock
```

Invariants (each is a correctness requirement, not a nicety):

* **INV-1 — append-only ids.** `n_pages[h]` never decreases and an id is never reused. This is
  what makes `page_address_buffer[layer][b, h, p]` stable for the whole prompt and what keeps
  `infer_state.py:850`'s `arange(prev_num_pages[h], n_pages[h])` valid.
* **INV-2 — dense prefix.** Every `p < n_pages[h]` has a bound CPU address and
  `occupancy[h, p] >= 1`. `recall()` indexes `page_address_buffer[layer][b, h, rids]`
  (`infer_state.py:1033`) and will read garbage for a hole.
* **INV-3 — slot discipline.** `token_offset[h, t] < occupancy[h, token_page[h, t]] <= page_size`,
  and slots `[0, occupancy)` of a page are dense (no holes), because
  `page_valid_entries` tells the kernel to read exactly the first `occupancy` slots
  (`infer_state.py:1279`, `kernels.py:317-344`).
* **INV-4 — split renumbers, ids do not move.** A split creates a *new* id at the end; the old
  id survives with a smaller `occupancy` and its surviving slots compacted to `[0, occ)`.
* **INV-5 — version.** `page_version[h, p]` (uint32) increments on every content mutation. The
  writer sets `kvc.ccc[b, h, p] = 1` for every page with `page_version` increased since the
  reader last cleared it.

### 1.2 Operations

```python
def form_pages(self, head: int, keys: np.ndarray, proj0: np.ndarray) -> None
    """Prefill only. keys: (n, head_dim) float32. proj0: (n,) the already-computed
    scalar projection (see 3.0). Partitions tokens into pages of <= page_size,
    assigning ids 0..n_pages[head]-1. Fills token_page/token_offset/occupancy/
    page_medoid/page_state. O(n log n). Single-threaded, no locks (prefill is
    per-layer and per-head disjoint, and the readers do not exist yet)."""

def assign(self, head, keys, token_ids) -> np.ndarray
    """Decode. keys: (m, head_dim) new tokens. Returns (m,) int32 page ids.
    Greedy best-fit: candidate pages come from the PAG search on the new keys
    (section 4.2); among candidates with occupancy < page_size pick the
    highest-scoring; if none, open page n_pages[head] and bump n_pages.
    Sets ccc=1 on every page written."""

def split_page(self, head, page, keep_mask) -> int
    """Split `page` into two. keep_mask: (occupancy[page],) bool. Survivors stay in
    `page` and are compacted to slots [0, n_keep). Movers get new page id
    p_new = n_pages[head]; n_pages[head] += 1. Returns p_new.
    Caller must have already copied the moved KV bytes (section 4.4).
    Sets ccc=1 on both `page` and `p_new` and bumps both page_versions."""

def publish(self) -> int
    """epoch += 1 (release store). Returns the new epoch. Called under `_lock`
    after every mutation batch (assign/split/form_pages)."""

def snapshot(self, head) -> 'PageSnapshot'
    """Reader side. Returns (epoch, token_page_row, occupancy_row). Callers
    re-check `self.epoch == snap.epoch` after using it and retry on mismatch."""

def valid_entries(self, page_ids: np.ndarray) -> np.ndarray
    """Drop-in for DCI.get_valid_entries. page_ids: (n_heads, k) int32.
    Returns (n_heads, k) int32 of occupancy[head, page_id], or -1 where
    page_id < 0 or page_id >= n_pages[head]. O(n_heads * k)."""

def page_bytes_stale(self, head, page) -> bool  # == ccc query
```

### 1.3 Atomic publication — the part that is a correctness requirement

Readers run on three different threads:

* `estimate_select_recall` / `_PAG_query` on the main thread or the asyncio worker
  (`_loop_executor`, `max_workers=1`, `infer_state.py:245`), feeding `recall()`.
* `recall()` (`infer_state.py:1016`) reads `page_address_buffer[layer][b, h, rids]`.
* `scatter_pages` (`infer_state.py:1109`) copies from the transit buffer into the GPU pool.

The split is the hard case: **it remaps old tokens to new page ids while a background recall may
be reading the old page's address.** Three separate mechanisms, each for one hazard:

1. **Address stability (INV-1).** The old page id keeps its CPU address, so an in-flight
   `recall()` of the old page reads a *valid, allocated* address. It can never read freed
   memory. This is why ids must be append-only and never reused — reusing a freed id would
   alias a live reader onto unrelated KV bytes.
2. **Content staleness (INV-5 + `ccc`).** A split changes the *contents* of the old page
   (slots are compacted). The reader might copy the pre-split bytes into the GPU. That is
   benign **only because** the query that consumed it also selected the new page (which was
   fully written before publication), *and* the split sets `ccc = 1` on the old page so the
   next `diff_pages_by_head` re-recalls it. If `ccc` were not set, the GPU would keep serving
   the un-compacted page with the new (smaller) `occupancy` — reading slots that no longer
   belong to that page. **That is silent wrong output, not a crash.** `ccc=1` on a split is
   mandatory.
3. **Mapping torn-read (INV-4 + epoch).** `token_page` is read by the query thread to turn
   search labels into page ids. A split rewrites a row of it. Use a seqlock:
   ```python
   while True:
       e0 = self.epoch
       row_page, row_occ = self.token_page[h], self.occupancy[h]
       labels = index.search(q, top_k=top_m, ef_search=efs)
       pages  = row_page[labels]
       ... build page_ids ...
       if self.epoch == e0:
           return page_ids
   ```
   Retry cost is negligible: a split is ~100 us of work and happens O(1) times per 16 decode
   steps (section 4.4), while the reader's window is ~100 us.
4. **Publication order inside a mutation**, strictly:
   ```
   (a) write KV bytes into the new CPU page        (data first)
   (b) page_medoid / occupancy / page_state for the new page
   (c) page_address_buffer[layer][b, h, p_new]     (address)
   (d) n_pages[h] = p_new + 1                      (RELEASE: id becomes visible)
   (e) token_page / token_offset rewrites for moved tokens
   (f) ccc[b, h, old] = ccc[b, h, p_new] = 1
   (g) publish() : epoch += 1                      (RELEASE: mapping becomes visible)
   ```
   Nothing before (d) exposes the new id; nothing before (g) exposes the new mapping. A reader
   that saw the id at (d) but not the mapping at (g) will not reference it, because it only
   learns the id from the search (whose medoid was updated at (b)) *and* the mapping under the
   same epoch.

---

## 2. Exact interfaces replacing the DCI entry points

Three functions in `infer_state.py` are replaced; `_apply_selected_pages` is **untouched**.
`DCI.diff_pages_by_head` becomes a module-level numpy function
`pag_retrieval.diff_pages_by_head` (bit-exact port of `src/py_dci.c:2710-2793`, including the
in-place `pid` mutation). `DCI.copy_to_buffer` in `recall()` is replaced by a page-id gather
(section 5.6). `DCI.reuse_copy_node` / `reuse_update_node` are dead once `n_reuse_layers == 0`
(default); the reuse branch is specified in section 2.4 for completeness.

### 2.1 `_PAG_first_call`

```python
def _PAG_first_call(self, b: int, cur_id: int,
                    query_states: Tensor,   # (n_kv_heads, dci_len, head_dim), CPU, float32
                    key_states:   Tensor,   # (n_kv_heads, dci_len, head_dim), CPU, float32
                    value_states: Tensor,   # same
                    projected:    Tensor)   # (n_kv_heads, dci_len, 1) or (n_kv_heads*dci_len, 1)
        -> None
```

Replaces the call at `infer_state.py:1232`. Must reproduce **every side effect** of
`_DCI_first_call` (`infer_state.py:597-727`), because all of them are consumed downstream:

| side effect | line | how Paged-PAG produces it |
|---|---|---|
| `self.dci_db[cur_id]` non-None | 659 | `self.pag_index[cur_id]` set; `self.pag_pages[cur_id]` set |
| `dci_db.num_leaves` per head | 665 | `self.pag_pages[cur_id].n_pages` |
| `cpu_cache.prefill_alloc_n_tokens(max_num_leaves * page_size)` | 668 | identical, with `max_num_leaves = n_pages.max()` |
| contiguity assert `cpu_cache[b,-1] - _base == (max_leaves-1)*stride` | 670 | identical |
| `prefill_evicted_pages[cur_id] = ev_gpi` | 679 | identical |
| `kvc.c2p` sink/window surgery | 681 | identical |
| `kvc_capacity[cur_id] = 1 << (max_num_leaves-1).bit_length()` | 684 | identical |
| `kvc.cc2gp` = -1 | 685 | identical |
| `kvc.ccc` = 1 | 687 | identical |
| `page_address_buffer[cur_id]` | 689 | identical |
| DCI `address_update` | 707 | **`page_address_buffer[cur_id][b,i,:n_pages[i]] = base + i*offset + arange(n_pages[i])*stride`** — the address law is unchanged |
| **KV bytes permuted into page order** | 696-708 (inside DCI) | new: GPU gather + one bulk copy (section 3.2) |

Return `None` (matches today).

### 2.2 `_PAG_add`

```python
def _PAG_add(self, b: int, cur_id: int,
             key_states: Tensor,    # (n_kv_heads, m, head_dim), CPU float32, m = num_evict_win*16
             value_states: Tensor)  # same
        -> None
```

Replaces the call at `infer_state.py:1186` (`offload_win_page_to_DCI`). Must reproduce
`_DCI_add`'s side effects (`infer_state.py:730-869`):

* `self.prev_num_pages` snapshot (line 764) -> `prev = self.pag_pages[cur_id].n_pages.copy()`.
* `ccc` handshake: `_DCI_add` reads `kvc.ccc` and writes it back after the C routine mutated
  it (lines 767-768, 795-796). Paged-PAG does this directly: the writer sets
  `kvc.ccc[b, h, p] = 1` for every page it writes, no round-trip.
* CPU growth: `cpu_cache.decode_alloc_n_tokens((max_num_pages - prev_max)*page_size)` with the
  same assert (lines 806-810).
* `kvc_capacity` growth and `cc2gp`/`page_address_buffer` extension (lines 812-837).
* `page_address_buffer[cur_id][b, inst, tmp_new_indices] = tmp_addr` for the **new** page ids
  only (lines 844-854). New ids are contiguous at the end (INV-1), so the existing
  `np.arange(prev, n_pages)` logic transfers verbatim.
* **New:** assign pages, write the 16 tokens' KV into the CPU pages, `insert_batch` into the
  PAG index, set `ccc`, `publish()`.

Return `None`.

### 2.3 `_PAG_query`

```python
def _PAG_query(self, b: int, cur_id: int,
               query_states: Tensor)   # (n_qo_heads, 1, head_dim) CPU float32
        -> Tuple[Tensor, Tensor, Tensor]
```

Replaces the DCI call in `_DCI_query` and returns **exactly what `_apply_selected_pages`
returns**, i.e.

```python
page_ids = self._pag_select(cur_id, query_states.reshape(-1, self.head_dim))  # (n_kv_heads, 12) int32
return self._apply_selected_pages(b, cur_id, page_ids)
```
so the return is `(evicted_idx, recall_idx, evict_num)`, all CUDA int32, with
`evicted_idx.shape == recall_idx.shape == (8, 12)` and `evict_num.shape == (8,)`.

**Precise contract on `page_ids`** (this is the whole interface):

* dtype `np.int32`, shape **exactly `(self.n_kv_heads, num_neighbours)` = `(8, 12)`**, C-contiguous.
* values in `[0, n_pages[head])`, or `-1` for padding. `-1` is legal only in the trailing slots.
* sorted by descending relevance within each head (position `i` binds to GPU slot
  `arange(ns, ns+12)[i]` on the first call, `infer_state.py:963-964`).
* **no duplicates within a head.**
* **never raises.** `InsufficientPagesError` had a DCI fallback at `infer_state.py:906`; with
  DCI gone there is no fallback, so the failure mode inverts: on a shortfall `_pag_select` pads
  the trailing slots with the lowest-numbered pages of that head not already selected. That is
  safe — `_apply_selected_pages` will recall them and `valid_entries` reports their true
  occupancy — it only wastes 4 of the 12 recall slots. It must count these in a
  `pag_shortfall_count` so the distinctness headroom (5.2 row G) is observable in
  `retrieval_stats` (`infer_state.py:976`).
* side effect: appends to `self.query_seconds["pag_mips"]` (used by `retrieval_stats`,
  `infer_state.py:976`), and must **not** leave `layer_lock` held across
  `_apply_selected_pages` (it mutates `cc2gp`/`ccc` which are also touched by the writer).

`_PAG_query` does **not** set `page_valid_entries`; that stays in `estimate_select_recall`
(`infer_state.py:1099-1103`), with `DCI.get_valid_entries` swapped for
`SemanticPageManager.valid_entries` (section 5.6). Note that with `page_topk = 0` that call
fills the full `[ns : ns + 12]` region, so `valid_entries` is invoked with a `(8, 12)` id array
every step — the same shape `_apply_selected_pages` asserts on.

### 2.4 Reuse branch (`n_reuse_layers > 0`, dead by default)

`_DCI_first_call:719-727` and `_DCI_add:866-869` reuse a previous layer's DCI instance because
adjacent layers have nearly identical key geometry. Paged-PAG equivalent: reuse the previous
layer's **partition** (`token_page`, `n_pages`) but not its index (keys differ). Concretely, the
`else` branch calls `self.pag_pages[cur_id] = self.pag_pages[reuse_id].copy_partition()` and
skips `form_pages`; the PAG index is still built per layer. This saves only the ~2-3 ms/head of
page formation, so with `n_reuse_layers == 0` (the default) it is not worth implementing in v1.
State it as a follow-up, not a dependency.

---

## 3. Page formation at prefill

### 3.0 Free projection, already computed

`modeling.py:129-133` already computes, for every retrieval layer,

```python
projected = torch.matmul(key_states.reshape(n_kv_heads, -1, head_dim)[:, start:end, :],
                         state.proj_vec[:-1]).reshape(-1, 1)
```

over exactly the offloaded token range, with `state.proj_vec` a fixed unit vector
(`infer_state.py:457-460`). This is a per-head scalar projection of every offloaded token, and
it is already on the CPU and passed into `_DCI_first_call` at `infer_state.py:1232`. **Reuse it
as projection #0.** It costs nothing and it makes the page order follow the same random
direction DCI used — the single most useful piece of continuity between the two backends.

### 3.1 Algorithm: PCG-16 (Projection-ordered Cohesive Grouping, capacity 16)

Per `(layer, head)`. Input `keys` `(n, 128)` float32, `proj0` `(n,)`, output the partition.

**Pass 1 — bucket (this is the "semantic parent").**
* Compute `L = 4` extra projections: `P = keys @ R`, `R = randn(128, 4, seed=(layer, head))`
  (seeded, so the partition is reproducible). `P` is `(n, 4)`.
* Bucket by the sign pattern of `P` -> at most 16 buckets. Same idea as DCI's
  `parent_id = sign pattern of projections` (`src/dci.c:1714`), just narrower.

**Pass 2 — greedy expansion along the projection-ordered path.**
* Sort each bucket by `proj0` (the free projection). Within a bucket the sorted order is a
  **path graph**: consecutive tokens are the natural expansion neighbours.
* Walk the path. Maintain `cur` (the open page) and `seed`.
  * `s = <seed_hat, k_t_hat>`; append `t` to `cur` iff `s >= tau_bucket` and `|cur| < 16`.
  * Otherwise close `cur` and start a new page at `t`.
  * `tau_bucket` = the 60th percentile of the consecutive-token similarities inside that bucket
    (a per-bucket, scale-free threshold; measure it in pass 1 with one extra `n`-length dot).
* Close-out: pages with `|cur| < min_fill` (8) are dissolved and their tokens are greedily
  re-packed into the following pages, first-fit by similarity. This bounds the page count at
  `ceil(n / 8) ... ceil(n / 16)`, i.e. between 736 and 1472 pages for n = 11776.

This is a greedy expansion along graph edges from a seed, with a hard capacity of 16. It is
**not k-means**: no centroids, no Lloyd iterations, one pass, deterministic.

**Pass 3 — optional 2-seed refinement.** For each page compute the cohesion
`c = min_t <k_t_hat, centroid_hat>`. If `c < tau2`, take the **farthest pair** in the page as two
seeds, assign the other 14 by max similarity, and if either half has `< min_fill` members merge
it into the following page. This is the "2-seed split scheme"; it uses `split_page` (section
1.2). Off by default in v1 (it costs a second pass and the gain is unmeasured).

**Page representative.** `page_medoid[h, p]` = the token in the page maximizing the mean
similarity to the other 15 (`argmax_t sum_{u} <k_t_hat, k_u_hat>`). Store the medoid rather than
the centroid because MIPS on the raw (unnormalized) key manifold is what the retrieval index
uses, and a centroid is off-manifold.

### 3.2 Getting the KV bytes into page order

`prefill_backup_pages` (`infer_state.py:1115-1151`) copies the offload region GPU->CPU in
**token** order. Pages need it in **page** order (INV-2 + `recall`'s contiguous-block
assumption). DCI does this with a CPU `memcpy` inside `btree_p_bulk_load`; we do it on the GPU,
where the bytes already are:

```python
# src_idx: (n_heads, n_pages_head, page_size) int32, GPU. -1 for padding slots.
src_idx[h, p, s] = gpu_page_index(sub - gpu_start) of the token in slot s of page p
permuted = kvc.buffer[gpu_start + src_idx]          # GPU gather, ~50 us for 96 MB
tmp_cpu_kvc.buffer[...].copy_(permuted, non_blocking=True)   # one bulk H2D->CPU, ~8 ms/layer
```

`src_idx` is built in numpy from `token_page`/`token_offset` in a single scatter:
`src_idx[h, token_page[h,t], token_offset[h,t]] = t - offset_of_offload_region`.
This replaces DCI's in-C rearrangement and is the *only* new memory movement at prefill.

### 3.3 Complexity

| step | work | unit cost | per head |
|---|---|---|---|
| projections | `2*n*128*4` = 12 MFLOP | ~20 GFLOP/s single core | ~0.6 ms |
| bucket + threshold | `O(n)` | | ~0.1 ms |
| sort | `n log n` = 1.6e5 cmp | | ~1.5 ms |
| greedy gate | `n*128` = 1.5 MFLOP | | ~0.2 ms |
| medoids | `pages * 16^2 * 128` = 24 MFLOP | | ~1.0 ms |
| **total** | | | **~3.5 ms** |

`O(n log n)`. 272 heads -> **0.95 s serial, ~0.15 s on 8 threads**. Compared with the 1.0 s DCI
builds *all 272 trees*, this is free. (`proj0` is free, and it saves the 128x1 projection that
would otherwise be the second-largest term.)

---

## 4. Decode-time insert path

Trigger: `offload_win_page_to_DCI(l)` (`infer_state.py:1178`) is called from
`infer_state.py:533` (`_prepare_decode`, main thread) and `modeling.py:311` (main thread) once
the window page fills, i.e. **once per 16 decode steps, for all 34 layers**, with
`num_evict_win = 1` page = **16 new tokens per KV head per layer**
(`infer_state.py:1158-1162`). This is on the critical path; the budget below matters.

### 4.1 Per head: candidate pages from the PAG search

The new tokens are not in the index yet, but a PAG `search` query does not need to be an indexed
point. So:

```python
labels, scores = index[h].search(new_keys[h], top_k=top_m, ef_search=efs)   # (16, top_m)
```

`top_m = min(max_search_k, 64)`. This reuses the *same* index for both retrieval and
assignment, which is the reason the page-granularity index is cheap: one structure, two uses.

`max_search_k` is floor-limited by the **distinctness** requirement, not by the assignment.
`num_neighbours = 12` means the query path needs 12 distinct page ids out of one `search`. At
page granularity with a freshly built index (736 points, one per page) `top_k = 16` suffices; but
the index accumulates 16 inserted tokens per head per insert event, so duplicate page labels
appear immediately after the first insert. After 32 events (512 generated tokens) the index
holds ~1248 points over ~750 pages, and `top_k = 32` yields ~19 distinct pages on average while
`top_k = 12` yields ~7 — i.e. it shortfalls every step and burns the padding path. **`max_search_k
= 32` is the practical minimum; 64 is the safe choice.** For assignment, 32 candidates over 16
new tokens is ample headroom.

### 4.2 Assignment

```python
pages = index_labels_are_page_ids(labels)          # already page ids at page granularity
for t in 16:
    cand = [p for p in pages[t] if occupancy[h, p] < 16]
    if cand: p = max(cand, key=lambda p: score[t, p])   # greedy best-fit
    else:    p = new_page(head)                          # n_pages[h] += 1
    token_page[h, new_token] = p
    token_offset[h, new_token] = occupancy[h, p]
    occupancy[h, p] += 1
```

Greedy best-fit (highest-scoring candidate *that has a free slot*) rather than pure argmax is
what keeps the split rate near zero: pure argmax would pile all 16 tokens onto the single
nearest page and split it repeatedly. Best-fit spreads them over the top-64 candidates, which
are by construction the semantically right neighbourhood.

### 4.3 Splits

**The decode path never splits.** This is the main simplification of the design and it is worth
stating explicitly, because the brief assumes splits are a decode event:

* A page is an append-only container with capacity 16. A full page simply stops accepting; the
  token goes to the next-best candidate with room, or to a fresh page.
* Because each head receives exactly 16 new tokens per event, and there are ~736 prefill pages
  at 0.5-1.0 mean occupancy... in practice the prefill partition is near-full, so the steady
  state is ~1-3 new pages per head per event, i.e. **~17-50 new page ids per head over a
  512-token generation**, against ~736 prefill pages. Growth is bounded and append-only.
* `split_page` therefore exists for exactly two callers: (i) the optional prefill refinement
  (3.1 pass 3), and (ii) a future "rebalance when a page's cohesion degrades past a threshold"
  policy. Neither is on the decode critical path.

If a split *is* ever invoked at decode, it costs the KV move of section 4.4 and sets `ccc`.

### 4.4 Split cost when it happens

Moving `m <= 8` tokens for head `h` from CPU page `p` to `p_new`:

* `cpu_kv_caches[l][b, p][h, src_slots]` -> `cpu_kv_caches[l][b, p_new][h, 0:m]`, for K and V.
* bytes: `2 * m * head_dim * 4 = 8 KB` per head; `64 KB` across 8 heads.
* plus compaction of the survivors inside `p` (`occupancy_old - m` slots, same byte count).
* implementation: two numpy fancy-index assignments on the pinned CPU page buffers,
  **~100-300 us per split event per layer** including Python overhead.
* amortized over 16 decode steps -> `< 20 us/step`. Negligible.

The load-bearing part is not the bytes, it is section 1.3 step (6): `ccc = 1` on both pages and
a `publish()`. Skipping either is silent correctness loss.

### 4.5 Feeding `ccc` / `cc2gp`

No change to `_apply_selected_pages` or `diff_pages_by_head`. The writer's only obligations:

1. `kvc.ccc[b, h, p] = 1` for every page written this event (assignments and splits).
2. `note`: `_DCI_add` currently round-trips `ccc` through the C `changed_page_list` argument
   (`infer_state.py:767-768, 792, 795-796`). Delete that round-trip; set the bits directly in
   `SemanticPageManager.assign` / `split_page`.
3. `page_address_buffer[layer][b, h, p_new]` for new ids, *before* `n_pages[h]` is published.
4. `kvc.cc2gp` and `kvc.c2gp` logic at `infer_state.py:964` / `969` is unchanged — it consumes
   `diff_pages_by_head`'s in-place `pid` update, which the numpy port preserves.

---

## 5. Cost budget

### 5.1 Calibration

Single-tree build cost, **measured** (dim=128, MIPS, `target_degree=16`, `OMP_NUM_THREADS=8`,
idle machine, min of 3):

| N | msk=128, efc=200, pl=64 | msk=32, efc=100, pl=16 |
|---|---|---|
| 736 | 1.144 s | 0.314 s |
| 2944 | 2.068 s | — |
| 11776 | 3.660 s | — |
| 736, msk=12 | — | 0.171 s |

**Build cost is concave in N, not linear.** Per-point cost is ~1.56 ms/point at N~500 but
~0.22 ms/point at N~12k, so shrinking N from 11776 to 736 buys only **3.20x** (3.660 -> 1.144),
not the 16x a linear model predicts. `max_elements` has no measurable effect (736, 2048, and
11776+4096 are all within noise). The remaining gain comes from `max_search_k`, because PAG MIPS
sizes its projection metadata from it (`pag.cpp:965` `max_truth_k =
ComputeWorkingSetSize(topk)`; `pag.cpp:1144` `pif_entries_per_bucket`): at N=736, going msk
128 -> 32 plus efc 200 -> 100 plus pl 64 -> 16 buys a further **3.64x** (1.144 -> 0.314).
Combined, **~11.7x**.

Any model of the form `T = k * (N/11776) * (msk/128)` is therefore wrong at small N, and it errs
in the **optimistic** direction for the page-granularity design. Use the measured ratios:

```
rho_N        = 3.660 / 1.144 = 3.20     # N: 11776 -> 736
rho_msk_efc  = 1.144 / 0.314 = 3.64     # msk 128->32, efc 200->100, pl 64->16
rho_combined = 3.660 / 0.314 = 11.66
```

**272-tree concurrent baseline, measured** (all 272 trees, token level, N=11776, msk=128, 64
cores):

| mix | wall |
|---|---|
| 4 concurrent x 16 threads | 327 s |
| 8 x 8 | 334 s |
| 16 x 4 | 345 s |
| 32 x 2 | 368 s |
| **64 x 1 — the required model** | **369 s** |

This supersedes the earlier 393 s extrapolation, and it is a direct measurement of the required
parallelism model rather than an extrapolation to it. The box is bandwidth-saturated, so **one
thread per tree costs only ~13% versus the best mix (369 vs 327 s)**. The required model is
therefore fine as specified; do not argue for a different one.

One operational catch: `pag.Index.build` is internally OpenMP-parallel, and PAG links the system
`libgomp` while torch uses its own Intel OpenMP runtime, so the per-tree thread count must be set
explicitly — `ctypes.CDLL('libgomp.so.1').omp_set_num_threads(n)` before each build works per
call. Leaving it unset gives every tree the full `OMP_NUM_THREADS` (the 585 s oversubscribed
Phase 1).

### 5.2 The table

Scaling forward from the measured 369 s baseline with the measured ratios of 5.1:

| # | config | N/head | msk | efc / pl | T_build (272 trees) | vs DCI 1.0 s |
|---|---|---|---|---|---|---|
| A | token granularity (as shipped today) | 11776 | 128 | 200 / 64 | **369 s** (measured) | 369x |
| B | token granularity, msk cut | 11776 | 64 | 200 / 64 | ~185 s | 185x |
| C | token granularity, msk cut hard | 11776 | 32 | 100 / 16 | ~101 s | 101x |
| D | page granularity, full build knobs | 736 | 128 | 200 / 64 | ~115 s | 115x |
| E | page granularity, msk cut | 736 | 64 | 200 / 64 | ~58 s | 58x |
| **F** | **page granularity, recommended** | **736** | **32** | **100 / 16** | **~32 s** (band 20-50 s) | **32x** |
| G | page granularity, msk floor | 736 | 12 | 100 / 16 | ~17 s | 17x — disqualified, see below |

Rows A-C: **token granularity does not fit at any `max_search_k`.** Row A is the literal reading
of the brief and costs 369 s of build per prompt against ~2-5 s of prefill compute for a
12k-token prompt on Qwen3-4B, i.e. ~100x the thing it is supposed to hide behind. Cutting
`max_search_k` (B, C) helps but cannot rescue it: `max_search_k >= top_k`, and `top_k` must be
large enough to surface **12 distinct pages**, which floors it near 16-32. So token granularity
is stuck at ~100 s.

Row G is disqualified on a ground that has nothing to do with speed: with `num_neighbours = 12`
and an index that accumulates 16 inserted tokens per head per event, `top_k = 12` shortfalls
almost immediately (section 4.1). **`max_search_k = 32` is the practical floor; 64 if the budget
allows.**

Row F carries a real uncertainty band (20-50 s). `rho_N` and `rho_msk_efc` were measured on a
single tree with 8 threads on an *idle* box, while the 369 s baseline is 272 concurrent
single-thread builds on a *saturated* one. Small-N indexes have proportionally more fixed work
(projection matrix setup, `index_path` creation, `max_entry_points` warm-up, pfs bucket
allocation) and parallelise worse under contention, so the truth is likelier to sit at the top
of the band than the bottom. **This is the one number that must be measured in the real regime
before committing** (5.4, item 1).

### 5.3 Where the wall clock actually lands

Per prompt (34 retrieval layers, 1 prompt, `page_size=16`, `budget=16`, `num_neighbours=12`):

| step | frequency | per unit | per prompt | per decode step | dominant? |
|---|---|---|---|---|---|
| page formation (PCG-16, 272 heads) | 1/prompt | 3.5 ms/head | 0.95 s serial / 0.15 s @8 threads | — | no |
| KV permutation into page order | 1/prompt | ~10 ms/layer (96 MB) | 0.34 s | — | no |
| **PAG build (row F)** | **1/prompt** | **~118 ms/head** | **~32 s (band 20-50)** | — | **YES, >95%** |
| PAG build (row A, rejected) | 1/prompt | 1.357 s/head | 369 s | — | fails |
| PAG query (page level, msk=32, ef=100) | 8 heads x 34 layers / step | ~0.1-0.2 ms/head | — | **40-75 ms** | 2nd |
| `aggregate` (existing, unchanged) | 34 / step | ~0.2 ms | — | ~7 ms | no |
| PAG `insert_batch` | 8 heads x 34 layers / 16 steps | 1-15 ms/head (see 5.5) | — | **17-256 ms** | risk |
| split + CPU KV move | rare | 0.1-0.3 ms/layer | ~0 | <20 us | no |
| `recall` / `scatter_pages` | 34 / step | unchanged from DCI | — | unchanged | no |
| `diff_pages_by_head` numpy port | 34 / step | ~15 us (8 heads x 12^2) | — | ~0.5 ms | no |

**Conclusion: the design is build-dominated at ~32 s/prompt, which is 32x DCI's 1.0 s.** It is
not hidden by prefill compute (~2-5 s); it is an order of magnitude larger than the thing it
would have to hide behind. Everything else in the table is at or below the current DCI cost
profile. Row F is 4-10x over prefill but is at least in a range where a persistent 272-worker
build pool and a one-prompt-at-a-time schedule could absorb it; rows A (369 s), B (185 s) and
C (101 s) are not.

Second cost is the query at 40-75 ms/step, essentially unchanged from the 1.94 ms/layer x 34 =
66 ms/step DCI costs today (`DCI query p50 = 1.94 ms per layer`, and the current PAG adapter is
3.18 ms/layer = 108 ms/step). **The query path is where Paged-PAG is strictly better than the
current PAG Phase-1 adapter**, because a page-level index returns exactly the thing
`_apply_selected_pages` wants, with no token->page indirection and no `keys[labels] @ q`
re-scoring (`pag_retrieval.py:162`) — PAG's own returned scores are used directly.

### 5.4 The measurement plan (do these before writing the manager)

In priority order. Each is a single number that changes the design.

1. **272 concurrent one-thread-per-tree builds at row F parameters** (N=736, dim=128, MIPS,
   Online, `max_search_k=32`, `ef_construction=100`, `projection_levels=16`,
   `target_degree=16`, `OMP_NUM_THREADS=1`). *This is the one that decides* whether row F is
   ~20 s or ~50 s. The 369 s baseline is measured in the real regime; every ratio feeding
   row F was measured single-tree on an idle box, so this is the number that must replace the
   extrapolation.
2. **The same run at `max_search_k=64`**, to price the distinctness headroom that row G lacks.
3. **`insert_batch` of 16 points into a 736-point index, warm, repeated 100x.** Expected 1-3 ms.
   The brief's 15.2 ms at N=736 is *inverted* with respect to the 2.2 ms at N=11776 and must be
   reproduced or refuted. If it is real, see 5.5.
4. **`search` of 16 queries, `top_k=32`, `ef_search=100`, on that index, and the number of
   distinct page labels returned.** Distinctness is what `num_neighbours = 12` consumes and it
   degrades as inserts accumulate — measure it at 0, 16 and 32 insert events, and watch
   `pag_shortfall_count`.
5. **Sign convention of `search`'s second return value for `Metric.MaximumInnerProduct`.**
   `pag_retrieval.py:86-90` currently discards it. The page-level design depends on it being
   the inner product (not a negated distance, not normalized). Assert once: for the returned
   `labels`, `scores` must be non-increasing.
6. **PCG-16 page cohesion vs DCI leaf cohesion**, measured as mean intra-page
   `<k_i_hat, k_j_hat>` on a real prompt, and **page-level recall@12 against an exact
   inner-product ground truth** over the 12-page budget. This is the accuracy gate; see 6.

### 5.5 The `insert_batch` risk and its mitigation

If measurement 3 confirms ~15 ms/head at N=736, the decode path costs
`15 ms x 8 heads x 34 layers = 4.1 s` per insert event, i.e. 256 ms/step amortized — a 4-10x
decode slowdown. In decreasing order of preference:

1. **Cut `ef_construction` to 100** (halves link-search work) and confirm.
2. **Defer the insert by one event** onto the build pool: the 16 tokens become searchable one
   event (16 steps) later. The tokens are still *recallable* the whole time — they are in
   `token_page` and their KV is in the CPU page, so a query that reaches their page still finds
   them; only the *search* cannot see them for 16 steps. This is a recall dip, not a
   correctness break, and it is measurable.
3. Batching the 8 heads' inserts into one `insert_batch` call (if PAG allows non-contiguous
   labels across heads — it does not, labels are per index; so this is a pool-submission
   change, not a call change).
4. Last resort: **insert only medoid updates**, i.e. never insert new points, and accept that
   pages formed during decode are reachable only via their neighbours. Cheap but it degrades.

### 5.6 Ports required to actually drop the `dciknn` dependency

Not ANN, but they live in the `DCI` class and the import at `infer_state.py:19` must go:

| DCI API | used at | replacement | cost |
|---|---|---|---|
| `DCI.diff_pages_by_head` | `infer_state.py:966` | numpy port, bit-exact | ~15 us/step (8 heads x 12^2) |
| `DCI.get_valid_entries` | `infer_state.py:1100, 1103` | `SemanticPageManager.valid_entries` | <1 us/step |
| `DCI.copy_to_buffer` | `infer_state.py:1038` | page-id gather: `cpu_kv_caches[l][b, rids]` -> transit buffer | ~64 KB/layer/step, ~50-100 us |
| `DCI.reuse_copy_node/update_node` | `infer_state.py:727, 869` | dead when `n_reuse_layers == 0` | — |

The `copy_to_buffer` port is the only one worth a measurement note: it currently runs on the
`c2g_stream` inside `recall()` with `non_blocking=True`, so it is off the critical path; the
torch gather must preserve that (`recall` already synchronizes at `infer_state.py:1105`).

---

## 6. The single most likely failure mode

**Page-granularity retrieval does not preserve DCI's recall, and unlike the build cost there is
no in-design escape from it.**

The cost analysis in section 5 has an escape hatch: if the build is too slow, lower
`ef_construction`, keep `max_search_k` only as high as distinctness requires, or lean on the
measured threading result. The *accuracy* constraint has none, because every knob that would fix
it is frozen by the brief: `page_size` (16), `page_budget` (16), sink/window page counts — hence
`num_neighbours = n_dci_pages - topk = 12 - 0 = 12` (`infer_state.py:877, 951`).

Section 3.1 collapses N from 11776 tokens to ~736 page medoids per head, which changes the
retrieval problem from "best 12 pages out of 736, ranked by the best token in each" to "best 12
pages, ranked by their medoid". DCI's measured MIPS recall@64 of 0.89 is a *token-level* number
and does not transfer. The failure shape is **medoid mismatch**: the query's best token lives in
a page whose medoid is far away, so the page is never retrieved. This is systematic, not random
— pages are formed to be *cohesive*, which is exactly the regime where a single medoid is least
representative of the page's extremes. Cohesion, the property the partition algorithm optimizes
for, is what makes medoid-based retrieval lossy.

**Severity, re-derived for a budget of 12 rather than 4.** This is materially weaker than the
4-slot reading. The retrieval budget is 12 pages = 192 tokens per head = 1.6% of the context, 3x
more room than a 4-page budget, and displacing one of 12 recalls is far more forgiving than
displacing one of 4. Two structural factors further cap the damage:

* **Errors are transient, not sticky.** Because `page_topk = 0`, `page_valid_entries[layer][2:14]`
  is rewritten in full every decode step (`infer_state.py:1099-1103`); there is no sticky
  top-`k` prefix to accumulate. A misranked page costs exactly one step and is re-decided from
  scratch next step against a query that has moved.
* **The 12 slots are re-selected, not maintained.** There is no incremental state to drift.

So the honest claim is **not** "accuracy collapse". It is a **small, systematic, hard-to-attribute
regression** that will present as "PAG is a slightly worse ANN" while the real cause is that the
index granularity was changed from token to page to make the build budget close. That is worse
to diagnose than a collapse, not better: a collapse gets reverted, a 1-2 point LongBench drop gets
rationalised. And the two failure axes are coupled — row F is only affordable because
`max_search_k` is small, and a smaller `max_search_k` means the 12 slots are drawn from fewer
candidates, so buying build time costs recall directly.

**Cheap falsification, before any implementation:** take one prefill of a real LongBench prompt,
compute the exact inner-product top-12 pages per head under (a) DCI's page partition with
token-level scoring and (b) PCG-16 pages with medoid scoring, and report recall@12 against an
exact token-level oracle. If (b) is below ~0.7 of (a), page granularity is disqualified, and the
fallback is to accept row C (~101 s of build, token-granularity index, `max_search_k` floored at
32 by distinctness) and spend the engineering on hiding 101 s behind the prefill rather than on
hiding 32 s.

**Runner-up failure mode:** the 15.2 ms `insert_batch` measurement is real (section 5.5), which
turns a 1.0 s DCI decode into a 4.1 s/event PAG decode. It is second only because it is
externally measurable in an afternoon and has four documented mitigations.
