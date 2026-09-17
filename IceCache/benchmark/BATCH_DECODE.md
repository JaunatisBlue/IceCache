# B=2 fixed-batch decode prototype

Implemented on branch `batch`, starting from `57c24c5`. Each request performs
its own prefill and builds its own CPU DCI indexes. Both requests allocate
GPU pages from one pool. Every decode step uses one model forward with
`input_ids.shape == [2, 1]`, batched projections/MLP, and one paged attention
call per layer. No CUDA kernel was changed.

## Interfaces

- `InferState(..., gpu_pool=shared_pool)` keeps request KV, DCI, maps and
  temporary buffers independent. Shared-pool states allow one initial prefill.
- Call `enable_icecache(model, ..., infer_state=state_a)` once, then
  `set_icecache_infer_state(model, state)` before each independent prefill.
- `BatchInferState([state_a, state_b], query_backend="native", query_threads=16)`
  creates the decode controller. Bind it with `set_icecache_infer_state`.
- `query_backend="serial"` queries A and B successively. `native` passes both
  requests to `_mdci_batch.batch_query` once, using one OpenMP team and dynamic
  scheduling over `(request, query_head)` tasks. It does not reserve half the
  workers for each request. GQA heads share read-only indexes during the call.
- Querying and incremental insertion occur synchronously in this prototype;
  do not mutate the request indexes from another thread during a query.
- `close()` stops use of the batch controller. The request states and GPU pool
  remain caller-owned; this is not a request recycling or continuous batching API.

Current scope: exactly two equal-length requests, same model/layout/budgets,
no cross-layer prefetch or reuse (`n_prefetch_layers=n_reuse_layers=0`).
The existing branch's ordinary decode uses recency `retrieve_blocks`.
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

```bash
cd /home/yx/IceCache
tmux new-session -d -s icecache-b2-ab \
  'bash /home/yx/IceCache/IceCache/benchmark/run_batch_probe.sh'
```

The launcher runs serial/native arms successively, each with two randomized
1024-token prompts, 16 resident pages of 16 tokens per layer, and 48 decode
steps. `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`.
It checks raw candidate equality on frozen indexes before decode and performs
20 interleaved CPU replay pairs on each of two layers (40 pairs per run).
Results/logs are under `experiment/batch_decode/` (Git-ignored). Existing
files of the same names are replaced; set `BATCH_RESULT_DIR` for a new run.

For a custom case, run `benchmark/batch_decode_probe.py --help` with
`PYTHONPATH=/home/yx/IceCache/IceCache/source`. `--compare-native-raw` requires
the optional native extension even when the decode backend is serial.

## Observed results, 2026-09-16

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

## Files

- `source/icecache/batch.py`: batch state, query dispatch and CSR assembly.
- `source/icecache/mdci_batch.c`: native cross-request dynamic task scheduling.
- `source/setup_mdci_batch.py`: pinned optional extension build.
- `benchmark/batch_decode_probe.py`: prefill/decode, query parity and metrics.
- `tests/test_batch_decode_csr.py`: fixed-Q/K/V batched attention equivalence.
