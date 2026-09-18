# Fixed-capacity decode batch (variable length, request exit/reuse)

Implemented on branch `batch`, starting from `57c24c5`. Each request performs
its own prefill and builds its own CPU DCI indexes. Every request allocates GPU
pages from one shared pool. Each decode step uses one model forward with
`input_ids.shape == [B, 1]`, batched projections/MLP, and one paged attention
call per layer. No CUDA kernel was changed.

## Interfaces

- `InferState(..., gpu_pool=shared_pool)` keeps request KV, DCI, maps and
  temporary buffers independent. Shared-pool states allow one initial prefill.
- Call `enable_icecache(model, ..., infer_state=state_a)` once, then bind each
  request with `icecache_state(model, state)` during its independent prefill.
  The binding restores the prior state even if forward raises and rejects a
  nested binding or a switch through `set_icecache_infer_state` while active.
- `BatchInferState.from_prefilled([state_0, ...], query_backend="native",
  query_threads=16)` validates the completed prefills, the shared GPU pool, the
  DCI indexes and physical page ownership. Request lengths need not match.
- `batch.step(model, input_ids)` runs exactly one `[B, 1]` decode forward over
  the *active* slots and constructs per-row absolute `position_ids` (each row
  carries its own request length) plus a length-1 `cache_position`. It binds and
  restores the model state automatically. The caller does not pass a
  Transformers KV cache; `use_cache` is fixed to false.
- `batch.retire(index)` frees that slot's pages in the shared GPU pool and
  **returns** the retired state; `batch.admit(index, state)` places a freshly
  prefilled request into a free slot after checking pool identity, page
  disjointness and configuration. `batch.active_indices` lists the live slots and
  `batch.batch_size` is their count. See "Resource ownership on retire" below for
  what retire can and cannot reclaim.
- `batch.build_attention_metadata(layer_idx)` returns the CSR `indices`,
  `indptr`, `last_page_len` and per-page/per-head valid entries used by the
  actual batched attention call, assembled in active-slot order.
- `query_backend="serial"` queries the active trees successively. `native`
  passes them to `_mdci_batch.batch_query` once, using one OpenMP team and
  dynamic scheduling over `(request, query_head)` tasks. It does not reserve a
  fixed share of workers per request. GQA heads share read-only indexes during
  the call.
- Querying and incremental insertion occur synchronously in this prototype;
  do not mutate the request indexes from another thread during a query.
- A failed decode step marks the batch unusable. Any partially changed request
  states must also be discarded and prefilled again; `close()` does not repair
  KV or free the caller-owned pool. This is not a continuous batching API.

Per-request DCI query accounting is exposed as `batch.query_counts` (one counter
per slot, length = capacity) and `batch.query_counts_by_layer`
(`[capacity][n_layers]`). `native_query_counts` is retained as a compatibility
alias that points at the same list as `batch.query_counts`.

Current scope: any number `B` of independent requests sharing one GPU KV pool
(slot capacity is fixed at construction; `batch_size` is the live active count),
variable-length requests (each row carries its own `position_ids`; `cache_position`
stays length 1), and explicit request exit/reuse via `retire(index)` /
`admit(index, state)`.  No cross-layer prefetch or reuse
(`n_prefetch_layers=n_reuse_layers=0`).  There is **no scheduler and no
continuous batching**: `admit` is called explicitly by the caller, never by a
loop.  The existing branch's ordinary decode uses recency `retrieve_blocks`.
This prototype explicitly queries DCI instead; its serial/native comparison
therefore uses the same new batch path in both arms.

## Native build

The separate extension uses M-DCI's private capsule layout. Tested upstream:
`https://github.com/yuzhenmao/M-DCI`, commit
`1137dbbdad85abfb8c70c1f5b1c6fe30200bfc7e`.

```bash
git clone https://github.com/yuzhenmao/M-DCI.git /tmp/icecache-mdci-batch-source
git -C /tmp/icecache-mdci-batch-source checkout 1137dbbdad85abfb8c70c1f5b1c6fe30200bfc7e
cd /home/yx/IceCache/IceCache/source
ICECACHE_MDCI_SOURCE=/tmp/icecache-mdci-batch-source \
  /home/yx/miniconda3/envs/icecache/bin/python setup_mdci_batch.py build_ext --inplace
```

The builder checks the clean source revision and patches a build-only copy of
`dci.c`: the returned buffer must allocate two requested-length channels,
even when fewer candidates were returned. The batch wrapper rejects short
results before page-map updates. Query tasks use a local descriptor copy to
disable nested parallelism without changing the owning tree's settings.

The installed conda package is not overwritten. Its binary SHA is recorded
in the new extension; replacing that package requires rebuilding and rerunning
the parity probe. This guard detects replacement, but is not a general ABI
compatibility guarantee. The tested installed `_dci.so` SHA256 is
`981ea6d5c8883afedee80bb00c601b26d510122338af9a1b7e87d1aef80de07d`.
Other M-DCI versions require compatibility review and parity validation.
Generated build files and the new `.so` are ignored by Git.

## Run

Use the `icecache` environment. `vllm_dev` is not used by this experiment.

For a single functional check, use the probe directly with
`--query-backend native --compare-native-raw --cpu-replay-repeats 0`. This
covers two completed prefills, native/serial candidate parity on frozen trees,
48 decode steps, per-request DCI queries, incremental insertion and pool page
isolation. For the state and attention contracts, run
`tests/test_batch_state_contract.py` (5 cases, CPU-only) and
`tests/test_batch_decode_csr.py` (GQA ratios 1 and 4).

The `run_batch_probe.sh` launcher is a separate historical A/B experiment:
it runs two arms and interleaved CPU replay measurements. It is unnecessary
for a functional check.

The historical launcher runs serial/native arms successively, each with two randomized
1024-token prompts, 16 resident pages of 16 tokens per layer, and 48 decode
steps. `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`.
It checks raw candidate equality on frozen indexes before decode and performs
20 interleaved CPU replay pairs on each of two layers (40 pairs per run).
Results/logs are under `experiment/batch_decode/` (Git-ignored). Existing
files of the same names are replaced; set `BATCH_RESULT_DIR` for a new run.

For a custom case, run `benchmark/batch_decode_probe.py --help` with
`PYTHONPATH=/home/yx/IceCache/IceCache/source`. `--compare-native-raw` requires
the optional native extension even when the decode backend is serial.

## Observed results, 2026-09-16 (pre-hardening)

> **Pre-hardening.** These numbers predate the 2026-09-17 change to
> `source/icecache/mdci_batch.c` (per-layer `field_of_view`; the `num_levels>=2`
> guard was removed) and the subsequent native extension rebuild (`.so` mtime
> 2026-09-17 10:52, after the `.c` at 10:47). The native batch query arm figures
> below were superseded by that rebuild. They remain here as the historical
> functional record and are **not** comparable to the 2026-09-17 probe. To
> reproduce an A/B timing conclusion, rerun both serial and native arms with
> `--cpu-replay-repeats 20`. No new A/B numbers have been fabricated.

A100 80GB PCIe, Llama-3.1-8B-Instruct, FP16, PyTorch 2.4.0+cu118.
These are functional probes and exploratory timings, not a quality benchmark
or a statistically established end-to-end speedup.

| Measurement | Serial query arm | Native batch query arm |
|---|---:|---:|
| Decode steps | 48 | 48 |
| DCI calls represented per request | 1536 | 1536 |
| DCI points, first layer of each request | 960 → 1024 | 960 → 1024 |
| Mean batch step after first 8 steps | 198.88 ms | 199.78 ms |
| Aggregate output throughput | 10.056 token/s | 10.011 token/s |
| Raw frozen-index candidate equality | 3072/3072 | 3072/3072 |

The two runs generated identical recorded token sequences for both requests.
Actual native scheduler diagnostics: **16 OpenMP threads, 64 query tasks**
per call. Ownership checks found no overlapping GPU pages between requests.
Both requests advanced to position 1072 and inserted new DCI points.

Frozen-index CPU replay means (one call handles the two requests):

- First run: serial 0.411 ms, native 0.384 ms.
- Second run: serial 0.580 ms, native 0.445 ms.

Both replays favor the native entry, but the size of the effect varies.
Full decode does not show a measurable improvement in this one-run-per-arm
comparison. `batch_query_seconds` also includes query transfer and Python page
bookkeeping; do not interpret it as pure native tree traversal time.
The 1024-token context and B=2 do not establish scaling to larger batches or
long contexts. Prefill is sequential and excluded from decode throughput.

Additional checks: CPU session tests passed (13; 7 GPU cases skipped in that
CPU invocation). Two dedicated GPU tests passed (GQA ratios 1 and 4), comparing
B=2 CSR sparse attention to two isolated B=1 calls with different per-head
valid counts and tail lengths. Syntax checks and `git diff --check` passed.

## Re-verification, 2026-09-17 (variable length + request exit/reuse)

Functional native probe only — **not** an A/B timing comparison and **not** an
end-to-end speedup measurement. The launch used `--cpu-replay-repeats 0`, so no
paired CPU replay was collected and no serial/native timing contrast exists in
this run. Do not read the per-step times below as a performance claim.

Environment: A100 80GB PCIe, Llama-3.1-8B-Instruct, FP16, PyTorch
2.4.0+cu118. `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=1`,
`OMP_DYNAMIC=FALSE`, `PYTHONPATH=/home/yx/IceCache/IceCache/source`. Native
extension: `source/icecache/_mdci_batch.cpython-310-x86_64-linux-gnu.so`,
rebuilt 2026-09-17 (mtime 10:52, after `source/icecache/mdci_batch.c` at 10:47)
to carry `field_of_view` per layer (the `num_levels>=2` guard was removed).

Command (page pool auto-sized, see the note below):

```
benchmark/batch_decode_probe.py --batch-size 2 --query-backend native \
  --compare-native-raw --cpu-replay-repeats 0 \
  --steps 48 --extra-steps 24 --post-admit-steps 8 --output <path>
```

| Measurement | Value |
|---|---:|
| Requests / prompt lengths | 2 → 1024 and 1040 tokens (slot 0 later replaced by 960) |
| Decode steps | 48 + 24 (slot 0 free) + 8 (after admit) = 80 |
| `raw_equal_elements` (frozen-index native raw vs per-request reference) | 3072 |
| `query_counts` (per-slot) | [256, 2560]; slot 0 resets on retire/admit |
| `query_counts_by_layer` | 8 and 80 per layer, respectively |
| `decode_steps` | 80 |
| `mean_step_ms_after_warmup` (main phase only) | 214.2 ms |
| `output_tokens_per_second_after_warmup` (main phase, B=2) | 9.34 token/s |
| `retire_step_ms` (24 steps, B=1) | ≈122–139 ms |
| `admit_step_ms` (8 steps, B=2) | ≈205–333 ms |
| `batch_query_seconds` | 5.02 s |
| `prefill_seconds` | [0.647, 0.449] |
| `gpu_peak_reserved_bytes` | 17005805568 |
| `native_scheduler` | {omp_team_size: 16, query_tasks: 64} |
| `auto gpu_pages` | 6080 (needed 6048) |

`dci_sha256 = 981ea6d5c8883afedee80bb00c601b26d510122338af9a1b7e87d1aef80de07d`,
matching the extension's built-in guard. Every request advanced and inserted new
DCI points; the three requests produce different token sequences from their
independent contexts.

What this run exercises beyond the fixed B=2 case: two different prompt lengths,
a **non-contiguous** active set (`active=[1]` while slot 0 is retired, i.e. B=1),
`retire` returning slot 0's GPU pages to the pool, `admit` of a third request of
a third length, and page isolation rechecked across the reused set. `decode_steps`
and every request's final `seq_len` matched the asserted values, so the row/slot
mapping (row order = active requests, *not* slot indices) holds on the
non-contiguous path.

**Contiguous-prefill constraint on `admit`.** `KvCache.prefill_alloc_n_tokens`
allocates one *contiguous* run of `ceil(tokens / page_size)` pages per layer, and a
prefill keeps all of a prompt's pages resident. The pool must therefore hold
`n_layers * sum(pages over all requests)` pages at once, and a retired request only
hands back runs as large as the request that owned them. Two consequences: the
probe now sizes the pool from the prompt lengths (`--gpu-pages 0` = auto; 6080
here, versus the old fixed 4096 that was already too small once the two prompts
differed in length), and the admitted request is kept shorter than the slot it
replaces. Admitting a request *longer* than the retired one can fail even with
enough total free pages; lifting that needs a block-level allocator.

This re-run confirms the batch path is functionally intact after the 2026-09-17
`mdci_batch.c` hardening, the extension rebuild and the variable-length /
request-exit rework. It deliberately omits the CPU replay arms, so it does **not**
re-establish the A/B timing conclusion from the pre-hardening 2026-09-16 table.
To recover an A/B comparison, rerun both arms with `--cpu-replay-repeats 20`
(matching the historical launcher).

## Batched prefill (one forward, per-request trees)

New on branch `batch` (2026-09-17). A batched prefill runs B prompts through a
single `model(...)` call, then builds each request's DCI tree independently. The
decode path, the KV-allocation strategy and the `b == 0` semantics of every
`InferState` are unchanged. This is the "批量 prefill、分别建树" feature.

### Interface

- `BatchInferState(...)` accepts a new `prefilled=False` flag
  (`batch.py:48`). The constructor skips the per-request `_check_prefilled` guard
  so it can be created *before* prefill; the probe's existing
  `from_prefilled(...)` keeps `prefilled=True` (default) for the serial path
  (`batch.py:186`).
- `batch.prefill_batch(model, prompts, token_budget=None)`
  (`batch.py:311`) performs the batched prefill:
  - `prompts`: list of 1-D `LongTensor` token ids, one per active slot in
    `active_indices` order.
  - With `token_budget=None` all prompts pad to a dense `[B, Lmax]`
    `input_ids` / `position_ids` (each row's real positions are `0..L_i-1`;
    padding rows are zero) and one `model(...)` is run. With a budget the rows
    are first split into length-similar groups by `_plan_prefill_groups`
    (`batch.py:286`, longest-first packing), each group then pads to its own
    group `Lmax` and gets its own `model(...)`; the split is only accepted when
    it removes at least `_PREFILL_GROUP_MIN_GAIN` (1.15x) of the padding, so a
    length-uniform batch still runs as a single forward. `prefill_groups`,
    `prefill_padded_rows` and `prefill_real_rows` report what happened.
  - Returns `(logits, next_tokens)` with `logits` shape `[B, 1, vocab]` and
    `next_tokens[i] = argmax(logits[i, 0])`. Only each request's last *real*
    token is materialised: the LM head is patched to `gather` that position
    first and then project it, so the head's GEMM and its output tensor scale
    with `B` rather than `B * Lmax` (at B=8, Lmax=1136: 8 rows instead of 9088,
    2 MB instead of 2.3 GB). `ICECACHE_PREFILL_FULL_LOGITS=1` restores the
    previous "project everything, then slice" behaviour for paired measurement.
- `ForwardMode.BATCH_PREFILL` (`infer_state.py:48`) is the new mode.
  `_icecache_attn_forward` (`adapter/modeling.py:478`) routes a `BatchInferState`
  whose `forward_mode` is `BATCH_PREFILL` to
  `BatchInferState.prefill_attention_forward` (`adapter/modeling.py:492-493`);
  every other `BatchInferState` still goes to the decode `attention_forward`, so
  the batch path can never accidentally take the decode branch.

### How padding is excluded

Inside `prefill_attention_forward` (`batch.py:278`, one call per layer, mirroring
`_icecache_prefill`):

- **KV write** (`batch.py`): for each active request `i` only its first `L_i`
  real tokens are sliced from the padded q/k/v and passed to that request's own
  `kv_caches[layer].prefill_alloc_n_tokens(L_i, ...)`
  (`infer_state.py:308`, still `bsz=1`) and `append_paged_kv_cache`
  (`infer_state.py:1355`). Padding positions never reach a `KvCache`.
- **Prefill attention** (`batch.py:385`): a single
  `BatchPrefillWithPagedKVCacheWrapper` call over the *flattened real tokens* of
  all requests, keyed by the per-request real-length CSR —
  `qo_indptr = cumsum([0, L_0, L_0+L_1, ...])` and
  `paged_kv_indptr = cumsum([0, n_pages_0, ...])`. Padding tokens are absent from
  `qo_indptr`, so they are neither queries nor attended-to keys. This is exactly
  the ragged batch-prefill pattern (vLLM V1 `query_start_loc` / `InputBatch`,
  SGLang `extend`).
- **Tree building** (`infer_state.py`): `prefill_evict_extra_pages`
  (`infer_state.py:1544`) -> `_DCI_first_call` (`infer_state.py:914`) runs
  **per request** on that request's real tokens. The query passed is that
  request's last real token, `query_states[pos:pos+1, -1:, :, :]` — dim 1 is the
  token axis in this layout. The `b == 0` argument used throughout `KvCache` /
  `_DCI_*` is untouched — each member is still an `InferState` with
  `batch_size == 1`.
- **Tensor layout must match the serial `_icecache_prefill` exactly.** After
  `apply_rotary_pos_emb` the tensors are `[B, heads, Lmax, hd]`; query/key are
  transposed to token-major `[B, Lmax, heads, hd]`, and **value must stay**
  `[B, Lmax, nkv, hd]` (its projection already produced that layout). Two failure
  modes were hit while bringing this up, both fixed: transposing value as well
  yields a strided `[B, nkv, Lmax, hd]` slice that `append_paged_kv_cache_prefill`
  rejects with `v must be contiguous`; and slicing the padded grid as
  `t[pos, :, :L_i]` truncates the *head* axis instead of the token axis, so
  `Lmax` tokens get written into an `L_i`-token cache (use `t[pos, :L_i]`).

### Page allocation

`prefill_alloc_n_tokens` still requires one contiguous page run per layer and a
prefill keeps every prompt page resident, so the peak GPU footprint during the
single batched forward is `n_layers * sum(ceil(L_i / page_size))` — identical to
the serial path, which simply reaches the same peak one prompt at a time. No new
discontiguous allocation is introduced. The probe already auto-sizes the pool
from the summed prompt lengths, so all B prompts fit simultaneously.

### Difference from the serial path

- Serial: one `model(...)` per request, each bound via `icecache_state`; trees
  built as a side effect of each forward. Decode then runs over the already
  prefilled batch.
- Batched: one `model(...)` for all B; the projection/MLP and attention-proj are
  vectorized over the padded `[B, Lmax]` grid, but the KV write, the (single
  ragged) attention CSR and the tree build remain per request. Padding rows cost
  wasted projection/MLP FLOPs but never enter attention or the caches.
- The probe records `prefill_mode` (`sequential`/`batched`) and `prefill_seconds`
  (a single batched-forward duration in batched mode) in the result JSON. Default
  is `sequential` — the serial path is unchanged.

### Usage

```
benchmark/batch_decode_probe.py --prefill-mode batched --batch-size 2 \
  --query-backend native --output <path>
```

### P0-3 status (per-slot decode handler begin_forward)

The batch decode path uses `self._handler` (`batch.py`), not the per-slot
`decode_handler_tab`. `BatchInferState._prepare_decode` still calls each slot's
`state._prepare_decode(1)`, which also begins that slot's own decode handler in
`infer_state.py`. The begin_forward there is redundant for the *batch compute*
path, but it is **not removed**: that same `InferState._prepare_decode` /
`_finish_decode` pair is the serial decode path (`decode_sdpa` reads
`decode_handler_tab[kvc.budget]`), and removing the begin_forward would break
serial decode and unbalance `begin`/`end_forward`. It is kept and annotated in
`batch.py`.


### Resource ownership on retire

`retire` can only hand back what is **shared**:

- **Shared, reclaimed here:** the request's pages in the batch's GPU `KvPool`
  (the `c2p` union across layers, deduplicated, with a double-free guard). A
  `--batch-size 2` run reports `retire_gpu_pages_returned = 512` — 16 resident
  pages x 32 layers, i.e. exactly that request's resident pages.
- **Per-request, but the batch still stops what it can:** the CPU `KvPool` is
  built inside `InferState.__init__` (`infer_state.py`), so there is no shared
  pool to return it to, and the DCI index exposes no destroy entry point.
  The background asyncio loop and its worker executor, however, *are* stopped:
  `retire` calls `InferState.shutdown()`, which shuts the executor down and stops
  the loop (the daemon thread then closes it and exits). Without that, every
  admit/retire cycle would strand one thread plus one executor. For the rest,
  `retire` clears the batch's and the state's own references
  (`_release_request_resources`, which also drops the raw `page_address_buffer`
  pointers into the CPU pool) and **returns the state** — dropping that last
  reference is what frees the CPU KV pool (hundreds of MB at the default page
  count) and the trees. The probe does exactly that (`del retired_state` +
  `gc.collect()`) and reports `retire_gpu_pages_returned`, `loop_stopped` and the
  thread-count delta, so the reclaim is observable.

So the exit path is closed for the shared GPU pages and for the per-request
threads/executor, and is *reference-driven* for the CPU KV pool and the DCI trees.
A library-grade version would add an explicit `release()` on `InferState` plus a
DCI destroy call; neither exists today, which is a documented limitation rather
than a hidden leak.

### Workspace sizing

The decode and prefill wrappers share one FlashInfer scratch buffer, mirroring the
serial path (which hands one 16 MiB buffer per budget group to both that group's
prefill *and* decode wrappers). The batch sizes it as `16 MiB x live rows`, grows
it when the batch grows (`_ensure_workspace`, called from `__init__`,
`prefill_batch` and `admit`) and re-points both wrappers through
`reset_workspace_buffer`. Observed values: 32 MiB at B=2, 64 MiB at B=4.
`ICECACHE_BATCH_WORKSPACE_MB` overrides the size outright; the true requirement
for large B is not characterised, and a warning fires above 1 GiB.

### Verification (2026-09-17)

**Numerical relation to the serial prefill (revised 2026-09-17).** An earlier
revision of this section claimed the two prefill modes produce *identical*
`generated_token_ids`. That happened to hold for the small B=2 / 4-step runs it
was measured on, but it is **not** a property of the implementation. The measured
relation is:

| Quantity | Result |
|---|---|
| batched prefill, run twice | bit-identical (`max abs delta = 0`) |
| sequential prefill, run twice | bit-identical (`max abs delta = 0`) |
| batched vs sequential, last real token's hidden state | `max abs delta = 0.016–0.0625` on `h` of magnitude 25–46, i.e. 0.5–2 fp16 ULP |
| batched vs sequential, last real token's logits | `max abs delta = 0.0156–0.0186` on logits of magnitude ~17–18 (ULP ~ 0.0166) |
| argmax at the prefill boundary, 8 requests (B=8) | **8/8 agree** |
| greedy token sequences, B=8, 116 tokens | only 36.2% of positions agree; 2/8 slots differ from the first decoded token |
| batched with `serial` vs `native` query backend | **8/8 identical** |
| unify the LM-head GEMM shape, then re-compare | difference unchanged (ratio 1.004) |

So the two prefill modes are **numerically equivalent to about one fp16 ULP, not
bit-identical**. The residual lives in the attention/KV path — the hidden states
themselves differ by 0.5–2 ULP — and *not* in the LM-head GEMM shape (unifying
that shape leaves the difference untouched). On the probe's synthetic word-salad
prompts the next-token margin is as small as 0.0156, the same size as the 1-ULP
difference, so greedy decoding amplifies it into token flips after one or two
steps.

Consequences: (a) token-exact equality is **not** a valid acceptance test for the
batched prefill; the valid checks are per-path determinism, ULP-level logit
agreement, argmax agreement at the prefill boundary, and identical per-request
bookkeeping (`native_points_*`, `query_counts_by_layer`, page ownership) — all of
which hold. (b) Any quality A/B that compares generated text must expect
run-to-run divergence at fp16, or compare at logit level.

**B = 3.** `--batch-size 3 --prefill-mode batched` also passed (prompt lengths
1024 / 1040 / 1056, one prefill forward, `auto gpu_pages = 8192`): a single
`model(...)` for all three prompts, then `retire(0)` leaving the **non-contiguous**
active set `[1, 2]`, four steps at B=2, `admit(0)` of a 960-token request, four
steps at B=3. `decode_steps = 12` and the per-slot token counts `[4, 12, 12]`
matched the assertions exactly, which also exercises the row-order = active-request
(not slot index) invariant at B=3.

**Unchanged behaviour.** The `--prefill-mode sequential` B=2 variable-length run
(48+24+8 decode steps, native backend, `raw_equal_elements = 3072`) is untouched
by this work; the batched path is opt-in and defaults off.

**Re-checked after the resource/workspace work.** `sequential` and `batched` runs
at `--batch-size 2` (4+4+4 steps) both exit 0, report the same `workspace_bytes`
(32 MiB) and the same `retire_gpu_pages_returned` (512); their token sequences
agreed in that particular run, which — per the numerical caveat above — is luck
rather than a guarantee. A `--batch-size 4 --prefill-mode batched` run also
passed: 64 MiB workspace, 512 pages returned, `decode_steps = 12`, per-slot token
counts `[4, 12, 12, 12]`.

**Thread release on retire.** With `InferState.shutdown()` wired into `retire`,
the same runs report `loop_stopped=True` and the process thread count drops by 2
at the retire (B=2: 6 -> 4; B=4: 10 -> 8) — the asyncio loop thread and its
executor worker. Repeated admit/retire cycles therefore no longer accumulate
threads, which was the last item of the exit path that could be closed without
upstream changes.


## Files

- `source/icecache/batch.py`: batch state, query dispatch and CSR assembly.
- `source/icecache/mdci_batch.c`: native cross-request dynamic task scheduling.
- `source/setup_mdci_batch.py`: pinned optional extension build.
- `benchmark/batch_decode_probe.py`: prefill/decode, query parity and metrics.
- `tests/test_batch_state_contract.py`: CPU-only state/contract checks (5 cases).
- `tests/test_batch_decode_csr.py`: fixed-Q/K/V batched attention equivalence (GQA ratios 1 and 4).
