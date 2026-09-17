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
- `batch.retire(index)` frees that slot's GPU pages back to the shared pool and
  marks the slot free; `batch.admit(index, state)` places a freshly prefilled
  request into a free slot after checking pool identity, page disjointness and
  configuration. `batch.active_indices` lists the live slots and
  `batch.batch_size` is their count.
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

## Files

- `source/icecache/batch.py`: batch state, query dispatch and CSR assembly.
- `source/icecache/mdci_batch.c`: native cross-request dynamic task scheduling.
- `source/setup_mdci_batch.py`: pinned optional extension build.
- `benchmark/batch_decode_probe.py`: prefill/decode, query parity and metrics.
- `tests/test_batch_state_contract.py`: CPU-only state/contract checks (5 cases).
- `tests/test_batch_decode_csr.py`: fixed-Q/K/V batched attention equivalence (GQA ratios 1 and 4).
