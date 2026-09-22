# Design A — pre-registered review checklist

Written **before** `IceCache/source/icecache/page_scan.py` exists (verified absent at the time of
writing). Nothing below is a review of code; it is the list of invariants I will check, the
falsification I will run, and the exact `file:line` evidence I will accept. Verdicts are
pre-registered so they cannot be rationalised after the fact.

Line numbers refer to `IceCache/source/icecache/infer_state.py` unless stated otherwise.

**Line numbers move when the edit lands.** Every item therefore names the *symbol*, and the line
number is only a hint. An item is only closed by evidence from the symbol, never by a line number.

## 0. Baselines — my own measurements, taken for this review

Scripts `/tmp/review_probe/{pack_bench,scan_bench,roundtrip_bench}.py`, GPU 0, load average
0.26 / 0.12 / 0.19 at the three runs. Real extracted Qwen3-4B keys, `k[:, 32:11744]`, N=11712,
`ceil(N/16)=732` pages.

| quantity | measured |
|---|---|
| greedy packing, spec pseudocode verbatim, batched over 8 heads | **2.222 s / layer** (min of 5) |
| greedy packing, inner loop collapsed to one `topk(16)` | **0.272 s / layer** (min of 5) |
| partition equality, naive vs topk | **identical** (page ids and in-page offsets) |
| `q @ reps.T`, 128 q-vectors, numpy CPU | **0.1489 ms / head** |
| `q @ reps.T`, 128 q-vectors, torch GPU per head | **0.0368 ms / head** |
| `q @ reps.T` + `topk(12)`, 128 q-vectors, 8 heads batched | **0.188 ms / layer** |
| `q @ reps.T` + `topk(12)`, decode-shaped (4 q-vectors/KV head), 8 heads batched | **0.0607 ms / layer** |
| numpy -> GPU -> matmul+topk -> numpy round trip, 128 q, 8 heads | **0.317 ms / layer** |

Two consequences I will hold the code to:

* **B1.** The 3 s/layer budget is met *only* if the inner 16-step loop is collapsed. The
  uncollapsed batched loop is 2.22 s — inside the budget by 1.35x, with no margin for the
  Python-side seed-advance bookkeeping the spec's `if assigned[seed]: continue` requires.
  Collapsing is legal and exact: `(seed) -> scores = seed @ K.T` is *fixed for the whole page*
  (the seed never changes inside the inner loop), so "argmax, delete, argmax" over a fixed
  vector is exactly `topk(16)`. I verified token-for-token equality. If the implementation keeps
  the 16-step loop and lands in 2.2-3.0 s, that is a **pass but a flagged margin**, not a win.
* **B2.** The 0.13 ms/head claim is met on GPU but **not** by the measurement method the spec's
  own 0.1544 ms reference used. Numpy-CPU serial is 0.1489 ms/head, i.e. above 0.13. The spec
  must say which backend it means; if `query()` is implemented as a numpy call on a
  `query_states.cpu().numpy()` array (the type `_DCI_query` receives today, `:889`), the claim
  as stated is false by 15%.

---

## 1. The five hazards of spec §7

### H1 — CPU page-id contiguity, asserted once (`_DCI_first_call`, `:670`)

**Invariant.** `assert cpu_cache[b,-1].data_ptr() - _base == (max_num_leaves-1)*stride` must hold
for the whole prompt, and `page_address_buffer[layer][b,h,:num_leaves[h]]` must hold a real CPU
address for **every id `PageScan.query` can return**. `recall()` indexes that buffer with the
returned ids (`:1033`) and `DCI.copy_to_buffer` dereferences them; an id outside `[0,num_leaves)`
holds the `-1` sentinel from `np.full(..., -1)` (`:689-690`).

**Trigger.** Run one long generation with `--n_reuse_layers 3`. At every `offload_win_page_to_DCI`
(`:1178`), `PageScan.insert` decides per KV head whether to open a new page. If it ever returns
an id `>= num_leaves[h]` (or `>= kvc_capacity`), the failure surfaces as a numpy `IndexError` in
`:1033` or, worse, as a copy from address `(uintp)-1` — a segfault or silent garbage attention.

**Failure is silent when:** the insert reuses an id that was freed, or an id in the *reserved*
tail that never got a CPU page.

**Falsify.** Assert in the new `insert` that every emitted id `< n_pages_reserved`, that
`page_address_buffer[layer][b,h,id] != -1` for every id the insert opens, and that
`page_address_buffer[layer][b,h,id]` equals `_base + h*page_bytes + id*stride` for all
`id < num_leaves[h]`. Dump `max(id)` over a 512-token generation.

**Accept.** A named assertion or counter at the insert site plus the dump, both inside
`page_scan.py` / at the `_DCI_add` call site. Absence of the assert = item open.

### H2 — `page_valid_entries` true per (page, head)

**Invariant.** `page_valid_entries[layer][table_pos, head]` is the number of *contiguous valid
slots from slot 0* in the CPU page occupying that table position. Init is `page_size` everywhere
(`:394-398`), which is a *lie* for every recalled page that is not full. The writer today is
`estimate_select_recall` (`:1099-1103`), rows `[ns : ns+num_neighbours)` only.

**Trigger.** Force a page to hold `m < 16` members (short prompt, or the reserved tail), get it
selected, and watch attention. Expected wrong behaviour: the kernel attends slots `m..15` of that
page, which contain **another page's or a stale token's** K/V.

**Falsify.** (a) Cross-check the tensor the mask is built from against an independently
maintained occupancy array, for all `(layer, table_pos, head)`, at ≥ 3 decode steps, with
`ns <= table_pos < ns+num_neighbours`. (b) Assert rows outside `[ns, ns+num_neighbours)` are
untouched (`ns+num_neighbours..budget` must stay `page_size` — those are sink+window, bounded by
`iter_bound`, and must not be "fixed" by the new code). (c) Confirm the *value* is a count, not a
bitmap: the kernel (`decode.cuh:119`) compares `page_offset < page_valid_entries[...]`, which can
only express a prefix. **Therefore members must be packed into slots `0..m-1` with no holes.**

**Accept.** The occupancy array is the single writer, it is packed-dense by construction, and the
cross-check in (a) passes. A code path that writes a member to a slot `>= occupancy` fails the
item even if occupancy is "correct".

### H3 — exactly `(n_kv_heads, num_neighbours)` distinct, never `-1`

**Invariant.** `_apply_selected_pages` raises `ValueError('Page selector returned the wrong
shape')` unless `nn_idx_0.shape == (n_kv_heads, num_neighbours)` (`:952`). Then
`kvc.cc2gp[b, head_ids, padded_arrays]` (`:964`) and `kvc.ccc[b, head_ids, padded_arrays]`
(`:966`, `:972`) index with those values. `-1` is a legal *index* in torch (wraps to the last
page) — so a `-1` does **not** raise, it silently maps to the last page id and corrupts `cc2gp`
for that head.

Worse, `-1` also flows to `page_valid_entries` via `get_valid_entries`, which returns `-1` for
`page_id < 0` (`experiment/paged_pag_design.md:125`). The tensor is **int32** (`**self._i32`,
`:1099`) but the kernel parameter is **`const uint32_t*`** (`decode.cuh:81`). `-1` reinterpreted
as uint32 is `0xFFFFFFFF`, and `page_offset < 0xFFFFFFFF` is true for *every* slot. The kernel
then attends all 16 slots of every selected page, including uninitialised ones, **with no assert
fired anywhere.** This is the single highest-severity silent-corruption path in the design and
the spec does not name it.

**Falsify.** (a) `assert (nn_idx_0 >= 0).all()` at the `_apply_selected_pages` boundary. (b) Run
`page_scan` and `dci` back-to-back on the same prompt with the same `--seed`; a shape assert that
does not fire plus a divergence in output tokens means a `-1` got through. (c) Grep the new
`query` for every return path and confirm all 12 slots are filled with a live id.

**Accept.** A `>= 0` check on the *returned* array, plus a written argument for why the top-12 is
always full (see §4).

### H4 — a partial page must not be read from slots it does not own

**Invariant.** Same as H2(c) from the reader side: the kernel reads a whole contiguous
`page_size` run at `table_pos*page_size`, then masks. It cannot skip a slot; it can only mask a
suffix.

**Trigger.** Put a page with `m < 16` members into a recall slot and check the K/V at slots
`m..15` of the GPU page. If they are a *different* page's tokens and `page_valid_entries` is
stale at 16, the model attends them.

**Falsify.** A targeted test that (i) builds a partition with a short page, (ii) recalls it,
(iii) compares `decode_sdpa` output against a dense reference. Also assert `occupancy <= 16` and
that `page_valid_entries` is written **before** the first decode that can select the page.

**Accept.** Either a test, or a proof in the code that pages are full except the tail and the
tail is excluded from selection. "The last page is always the only partial one" is *not* enough
once decode-time insert can open a page and leave it open.

### H5 — a reuse layer must not build

**Invariant.** `check_reuse(cur_id, start=2)` (`:294-303`) returns 0 for `cur_id <= 2` and for
`position_in_cycle == 0`; otherwise it returns the source layer id. With `n_reuse_layers=3` the
build layers are 0,1,2,5,8,... — note layers **0 and 1** also return 0 even though
`layer2budget` is `None` for them (`:120-123`). Build iff `check_reuse(cur_id) == 0`
**and** `layer2budget[cur_id]` is not `None`; both conditions are already at `:477`.

**Trigger.** Build a `PageScan` for a reuse layer, or alias without building. A build on a reuse
layer is *not* an exception — it silently produces a partition from the wrong layer's keys and
recall drops without any assert.

**Falsify.** Log `(layer, check_reuse(layer), built_or_aliased)` for all `n_layers` at prefill and
assert the set of build layers equals `{i : layer2budget[i] and check_reuse(i) == 0}`. Then run
`run_longbench.sh`'s config (`--n_reuse_layers 3`) and compare recall against `--n_reuse_layers 0`
on the same row: with 12 build layers out of 34 the aggregate recall must move, and the per-layer
build log must show 12, not 34.

**Accept.** The log, plus the reuse branch reading `self.dci_db[reuse_id]`-equivalent state
rather than a freshly built own structure. See §3 for the stronger version of this item.

---

## 2. The last partial page (`< 16` members)

Three separate questions, three separate checks.

**(a) Is `page_valid_entries` right?** At N=11712, `11712/16 = 732` exactly, so prefill produces
**no** partial page. A partial page can only come from (i) a prompt whose indexed length is not a
multiple of 16, or (ii) the decode-time insert opening a page and leaving it open, or (iii) a
reserved page id in the tail that was never emitted. (iii) is the dangerous one because it has
`occupancy = 0` *and* no CPU address. **Falsify:** assert `occupancy.sum(axis=-1) == N` for every
head at the end of build, and `occupancy > 0` for every id `< n_pages_emitted`.

**(b) Does `topk` ever select a page with 0 members?** Only if the topk domain is the *reserved*
id space rather than the *emitted* id space. A never-emitted page's rep is either the zero vector
(if rep = `sum / max(count,1)`) or **NaN** (if rep = `sum / count` with `count = 0`). `torch.topk`
returns NaN entries **first**, so a NaN rep is guaranteed to be selected. **Falsify:** construct
the reserved-tail case, put a NaN in one rep, and check `query`'s output. This is a
timing-insensitive CPU check. **Accept:** either topk is restricted to `n_pages_emitted`, or
empty reps are masked to `-inf`, *and* the code says which. Note this interacts with H1: if the
domain is the reserved space but the addresses only cover the emitted space, a selected empty
page has no address either.

**(c) Does the GPU mask read a slot it does not own?** No — provided H2(c)/H4 hold: the mask is
a prefix count, so slots `m..15` are masked as long as `page_valid_entries == m` at the moment
the page is in a recall slot and members are packed from slot 0. The failure is never "the mask
is wrong", it is "the mask is a stale `page_size`" (`:394-398` initialises exactly that lie).

---

## 3. `check_reuse` correctness

**Invariant 1 — alias, don't rebuild.** A reuse layer must not own a partition. Today `:660-662`
sets `dci_db = self.dci_db[reuse_id]` and every subsequent use goes through it. Design A must do
the same for `token2page`/`offset_in_page`/`reps`/`n_pages`/occupancy.

**Invariant 2 — the `c2p` offset identity.** `_DCI_query`'s reuse branch (`:1084-1092`) does

```
offset = self.kv_caches[layer_idx].c2p[0,0] - self.kv_caches[reuse_id].c2p[0,0]
assert ((self.kv_caches[layer_idx].c2p - self.kv_caches[reuse_id].c2p) == offset).all()
eids = self.prev_eids.clone(); eids[mask] += offset
```

`eids` are **GPU page-table entries** produced by the *source* layer, reused verbatim by the
reuse layer. Design A does not change `_apply_selected_pages` and does not change `c2p`, so the
identity is untouched **as long as design A keeps the same `kvc` allocation sequence**. It breaks
if design A changes how many GPU pages a layer requests, or when. **Falsify:** run
`--n_reuse_layers 3` and assert the identity on every decode step (the assert already exists —
confirm it is not deleted or `try`-wrapped).

**The load-bearing question the spec gets right but should say out loud.** `prev_eids` /
`prev_rids` / `prev_nr` are **single shared slots**, not per-layer (`:1080-1082`). A reuse layer
consumes the state produced by the *immediately preceding build layer*, and `offload_win_page_to_DCI`
iterates `for l in range(self.n_layers)` (`:522-525`). Ordering is therefore load-bearing: if
`_DCI_query` is ever reordered or run on a worker thread for a different layer ("layer
prefetching", `n_prefetch_layers > 1`, `:931-932`, `:897-900`), a reuse layer may read another
source layer's `eids`. Design A must preserve the `n_prefetch_layers <= 1` gate.

**Invariant 3 — the decode-time insert must be replicated, not recomputed.** `_DCI_add` is called
for **every** layer including reuse layers (`:1186` via `offload_win_page_to_DCI`). The reuse
branch (`:799-803`, `:866-869`) calls `DCI.reuse_update_node`, which copies the *same* keys into
the reuse layer's own CPU buffers at the *same* addresses implied by the source's
`token2node`. So the reuse layer needs the source's `(page, slot)` assignment, not its own.
Design A's insert rule is a function of `reps` and the incoming keys (`spec §4`: "the page whose
rep has the highest inner product"), and a reuse layer's keys are a *different layer's* keys.
**If the reuse layer recomputes the assignment it will get a different page and the CPU copies
will diverge from the source's page layout** — with no assert. This is the sharpest form of
hazard H5 and the spec's "reuse layers alias the source layer's `token2page`" covers it only if
`insert` is written to reuse the source's decision. **Falsify:** for a reuse layer, assert
`token2page[reuse] is token2page[reuse_id]` (identity, not equality) or that `insert` on a reuse
layer is a pure copy of the source's `(page_id, slot)` list.

**Can two heads in one layer produce different page counts?**
* At **prefill: no, provably.** Every token is assigned exactly once; each emitted page consumes
  `min(16, #unassigned)`; all heads see the same `N`. So every head emits exactly `ceil(N/16)`
  pages. The argument is airtight *only* because the loop terminates when tokens run out. If the
  implementation iterates over a **pre-reserved page count larger than `ceil(N/16)`**, the extra
  iterations run with zero unassigned tokens and `argmax` over an all-masked row returns index 0
  — silently re-assigning already-assigned tokens and producing pages with 0 or 17+ members.
  **Falsify:** assert `occupancy.sum(-1) == N` per head and `bincount(token2page).min() == 16`
  at the end of build. My own probe's `bincount` gives `min == max == 128` (8 heads x 16).
* At **decode: yes**, heads diverge, because each head's insert decision depends on its own reps
  and keys. This is fine — DCI's `new_num_leaves` is already per head (`:840-854`) and
  `max_num_leaves`/`new_num_leaves.max()` absorb it (`:806-851`). But the spec's §5 sentence
  ("the per-head page count is identical, so the CPU page-id space is aligned across heads
  exactly as DCI's `num_inst=8` layout is") is true **at prefill only** and must not be used to
  justify a shared/packed page-id space at decode. `page_address_buffer` is `[bsz, n_kv_heads,
  capacity]` (`:689-690`) — per head, like DCI. Keep it per head.

---

## 4. Decode-time insert: does the page-id space really never grow?

This is the item where the spec contradicts itself, and the contradiction is load-bearing.

* **Spec §1** says `dci_db[cid].num_leaves` becomes `n_pages`, **fixed** at `ceil(N/16)`.
* **Spec §4** says reserve `ceil((N + n_decode_budget)/16) + slack` pages at prefill.

These cannot both be the value of `n_pages`/`num_leaves`, and the two choices fail differently:

**(i) If `num_leaves == ceil(N/16) == 732`** then any page opened at decode has id `>= 732`.
`_DCI_first_call` assigns CPU addresses only for `j in range(num_leaves[i])` (`:702-704`) and
`sizes cpu_cache` to `max_num_leaves * page_size` (`:668`). So ids `>= 732` are **outside both the
address buffer and the CPU allocation**. `recall` (`:1033`) would index
`page_address_buffer[...][b,i,732+]`, which is the `-1` sentinel, and `copy_to_buffer` dereferences
`(uintp)-1`. **Violation surfaces at `recall` -> `DCI.copy_to_buffer`, `:1038-1042`**, as a
numpy `IndexError` if `kvc_capacity` is still 1024, or as a bad-pointer copy if it grew.

**(ii) If `num_leaves == ceil((N + budget)/16) + slack`** then addresses cover the whole space (if
`cpu_cache` and `kvc_capacity` are sized from the same number) but up to `slack + n_decode_budget/16`
reps are **empty**, and the topk domain now contains pages with 0 members — see §2(b).

**Where the growth would actually surface, traced:**

| site | line | what it does | how a violation appears |
|---|---|---|---|
| `alloc_page` | `:1242-1260` | GPU **window** page pool only; frees a previous layer's `prefill_evicted_pages` then `self._pool.alloc_page()` | unrelated to the CPU page-id space. Do **not** accept "the id space can't grow because `alloc_page` recycles" — different space. |
| `kvc_capacity` | `:684` | `1 << (max_num_leaves-1).bit_length()`; sizes `cc2gp`, `ccc`, `page_address_buffer` | if the reserved count exceeds this, growth at `:813-837`, which only runs `if b == 0` (`:812`) — bsz=1 today, a latent bug at bsz>1 |
| `estimate_select_recall` | `:1099-1103` | writes `page_valid_entries[ns : ns+num_neighbours]` | index-out-of-range only if the table got longer than `budget`; guarded by the `n_real_pages == budget` check at `:1072` |

**Falsify.** (a) `assert page_scan.n_pages_reserved == ceil((N + reserve)/16) + slack` and
`assert page_scan.n_pages_emitted <= n_pages_reserved` at every insert. (b) `assert
kvc_capacity >= n_pages_reserved` right after `:684`, so the growth path at `:813` is provably
dead. (c) A 512-token generation with an assert that no insert ever returns
`id >= n_pages_emitted_before`. (d) **The decisive one:** at the end of prefill, assert that
`page_address_buffer[layer][b,h,:n_pages_reserved]` is free of `-1` for every id that `query` can
return.

**Accept.** Both numbers present in the code, consistently: an *emitted* count that drives
`num_leaves`-equivalent bookkeeping (`prev_num_pages`, `new_num_leaves`, `arange(prev, cur)` at
`:850`) and a *reserved* count that drives allocation (`:668`, `:684`, `:689`). One number for
both is a fail.

---

## 5. The `num_neighbours = 12` budget

`num_neighbours = self.n_dci_pages - self.layer2topk[cur_id]` (`:877`, `:951`, `:617`).
With `--page-budget 16 --n-sink-pages 2 --n-win-pages 2 --page-topk 0`:
`n_dci_pages = 16 - 2 - 2 = 12` and `layer2topk = 0`, so `num_neighbours = 12`. **But
`layer2topk` is `[b and b // 2 for b in page_budgets]` when `page_topks is None` (`:124-125`)**,
i.e. 8 for budget 16, giving `num_neighbours = 4`. Both harnesses pass an explicit
`--page-topk 0` (`page_scan_compare.py`, `pag_stage1_compare.py`), so 12 holds *for these runs
only*. `num_neighbours` is **per layer** (`layer2topk[cur_id]`, and `layer2topk` is a list).

**Failure if the new code hardcodes 12:** `_apply_selected_pages` raises at `:952` for any layer
whose `layer2topk != 0`. **Falsify:** run the harness with `--page-topk 8` and confirm the new
path either honours 4 or fails loudly. **Accept:** `num_neighbours` continues to be read from
`n_dci_pages - layer2topk[cur_id]`, not a literal.

**Fewer than 12 candidate pages.** The topk domain is the page id space (732+), so 12 is
available whenever `>= 12` **non-empty** pages exist. It is not available when (ii) of §4 puts
empty pages in the domain — see §2(b).

**Dedup below 12.** `first_k_unique(row, 12)` (`utils.py:58-61`) does
`np.unique(row, return_index=True)` -> `idx_sorted = np.sort(idx)[:12]` -> `row[idx_sorted]`. With
`k=12` on a row of `ratio*12 = 48` entries it returns **12 iff the row has >= 12 distinct values**,
and **fewer otherwise**. A shorter row then makes `np.vstack` (`:938-939`) raise
`ValueError: all the input array dimensions ... must match`. Note there is **no fallback**
(contrast the PAG path, which catches `InsufficientPagesError` and falls back at `:906-907`).

* DCI is safe by construction: one q-head alone returns 12 distinct leaf ids, so the union of 4
  q-heads has `>= 12` distinct.
* Design A is safe for the same reason **iff** `topk(12)` per q-head returns 12 distinct ids from
  a domain with `>= 12` non-empty pages. Page ids are distinct by construction, so this reduces
  to "the domain has `>= 12` non-empty pages".
* `first_k_unique` is first-occurrence-in-row order, which is *interleaved* (`nn_idx_0.transpose(0,2,1)`,
  `:936-937`), i.e. round-robin across the 4 q-heads, not concatenated. Preserving that
  interleaving is not cosmetic: it decides which pages survive the cut. Design A must feed the
  dedup in the same layout.

**Falsify.** A unit test on `query` with a stub rep table where two q-heads see identical scores
(must still yield 12), and one where the domain has 11 non-empty pages (must fail loudly, not
emit a short row). **Accept.** `query` returns shape `(n_kv_heads, 12)` on the stub, and an
explicit guard/raise when the domain has `< 12` non-empty pages.

---

## 6. Additional hazards not in spec §7

| # | hazard | falsify | accept |
|---|---|---|---|
| **X1** | `-1` reaching `page_valid_entries` becomes `0xFFFFFFFF` and the kernel attends the whole page (`decode.cuh:81` uint32 vs `**self._i32` int32) — silent, no assert | `assert (nn_idx_0 >= 0).all()` before `:954` | the assert, stated as a fix for the unsigned-truncation path |
| **X2** | NaN/zero reps for never-emitted reserved pages win `topk` (§2b) | stub rep with one NaN, run `query` | topk restricted to emitted pages, or `-inf` mask |
| **X3** | build loop run over the *reserved* count re-assigns already-assigned tokens via `argmax(all-masked) == 0` | `bincount(token2page).min() == 16` and `occupancy.sum(-1) == N` | both asserts at end of build |
| **X4** | spec §1 (`n_pages = ceil(N/16)`) vs §4 (reserved count) — one number cannot serve both | §4 trace, test (d) | two distinct, consistently used numbers |
| **X5** | reuse-layer `insert` recomputes instead of copying the source's `(page, slot)` (§3 Invariant 3) | identity assert on the aliased arrays | `insert` on a reuse layer emits the source's assignment verbatim |
| **X6** | `num_idx_1` (in-page slot) is written to `nn_idx_all` (`:932`) and `nn_idx_all` is **never read** anywhere in the repo (verified by grep). Design A's `query` returns reps-based rankings; if it stops producing slot indices, nothing breaks *today* | grep for readers of `nn_idx_all` | either documented as dead, or produced; silence is not acceptable |
| **X7** | `page_valid_entries` rows outside `[ns, ns+num_neighbours)` must stay `page_size` — sink/window are bounded by `iter_bound`, not by the mask | assert those rows unchanged | code comment + assert |
| **X8** | `DCI.diff_pages_by_head` (`:966`) is a pure static, still needed; design A must not drop it with the rest of DCI | grep for the call | it survives |

---

## 7. Pre-registered verdicts

* **V1.** Claim 1 PASSES if the landed batched build is `<= 3.0 s` per layer at N=11712; it is
  FAILED if `> 3.0 s`. Measured floor for the two candidate implementations: 2.22 s (verbatim
  pseudocode) and 0.27 s (topk-collapsed). Anything above 2.22 s means extra bookkeeping, not
  extra work — flag it.
* **V2.** Claim 2 PASSES if `query` returns 12 pages per KV head in `<= 0.13 ms/head`; FAILED
  otherwise. Measured: GPU 0.037 ms/head (0.188 ms/layer batched) PASS; numpy CPU 0.149
  ms/head FAIL. The verdict is a function of the backend, and the spec must name it.
* **V3.** The design is **not mergeable** while X1 (uint32 `-1`) and X4 (§1 vs §4) are unresolved,
  regardless of recall or speed numbers, because both are silent-corruption paths with a
  plausible trigger in the default `run_longbench.sh` configuration.

## 8. Status of this review

Sections 0-7 were written before any implementation existed: `page_scan.py` was verified absent
at the start of this pass. **It landed at 19:46 on 2026-09-20, while §0's measurements were
running** (`IceCache/source/icecache/page_scan.py`, untracked, 23803 bytes). `infer_state.py` was
not modified during that window (mtime 16:10:20) and still contains the unmodified DCI path at
every integration point cited in spec §6.

**No code review was performed in this pass.** The checklist above is pre-registered and
unrevised against the landed implementation — that is the point of it. The review pass must
execute §1-§6 against `page_scan.py` and the `infer_state.py` diff, and must not amend an item
after seeing the code. The one thing that must be checked first, before any timing: whether the
implementation used the verbatim 16-step inner loop (2.22 s/layer, B1) or the exact
`topk(16)` collapse (0.27 s/layer), because that single choice decides V1's margin.

The measurements in §0 were taken on real extracted Qwen3-4B keys with my own scripts
(`/tmp/review_probe/`) under GPU 0 at load average 0.12-0.26.
