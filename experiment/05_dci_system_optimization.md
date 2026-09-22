# Experiment 05 — Single-request DCI system optimization

## Fixed configuration

- Llama-3.1-8B-Instruct, 40,060-token Passkey context
- 64 GPU pages, page size 16, sink/window 2/2
- cross-layer reuse 3, DCI parallel level 2
- 64 generated tokens, 8 warmup tokens, 55 measured tokens
- GPU 1: A100 80 GB PCIe
- CPU: dual Xeon Gold 5218, 32 physical cores / 64 logical CPUs

## Results

| Variant | TPOT ms | DCI ms/tok | Native query ms/tok | Recall wait ms/tok | Passkey |
|---|---:|---:|---:|---:|---:|
| Baseline | 115.21 | 41.41 | 31.15 | 2.17 | pass |
| Async recall ring | 129.72 | 48.27 | 37.85 | shifted to next query | pass |
| DCI visit ratio 0.50 | 118.22 | 43.56 | 33.65 | 2.23 | pass |
| DCI visit ratio 0.25 | 113.26 | 40.14 | 29.59 | 2.20 | pass |
| Reuse 6 layers | 98.20 | 23.19 | 17.84 | 2.78 | pass |
| Next-layer prefetch | 125.88 | 44.68 | 30.68 | 1.34 | pass |
| Contiguous native scratch | 115.30 | 41.00 | 30.62 | 2.20 | pass |
| AVX2 dual accumulator | 118.72 | 41.95 | 31.44 | 2.19 | pass |
| 32 physical workers | 111.40 | 38.04 | 28.61 | 2.20 | pass |
| 16 physical workers | 126.68 | 53.55 | 44.45 | 2.41 | pass |
| FP16 recall only | 113.53 | 40.51 | 30.48 | 0.28 | pass |
| 32 workers + FP16 recall | **109.20** | **37.86** | **28.60** | **0.30** | pass |
| + native GQA page merge | **105.60** | **34.12** | **28.68** | **0.29** | pass |

The safe combined improvement is 8.3% in TPOT. The larger 14.8% result from
six-layer reuse changes the retrieval policy and is not quality-safe.

## Quality gate

The deterministic 20-sample Qasper subset used indices
`[33, 39, 48, 51, 53, 57, 59, 79, 82, 85, 92, 106, 119, 133, 134, 150, 153, 165, 191, 194]`.

| Variant | Qasper F1 |
|---|---:|
| Reuse 3 baseline | 45.07 |
| Reuse 4 | 40.42 |
| Reuse 6 | 41.30 |
| 32 workers + FP16 recall, reuse 3 | **45.43** |
| + native GQA page merge | **45.48** |

The safe combination preserves subset quality; fixed wider layer reuse does
not and remains an experiment only.

## CPU profile conclusion

`perf record` captured 168k cycle samples during the decode run. OpenMP
runtime accounted for 36.85% of sampled cycles, `vecmul` for 9.57%, and the
single-level tree query for 4.19%. Reducing allocator calls and rewriting the
dot product did not improve TPOT. Matching the 32 Q-head tasks to 32 physical
cores was the useful deployment-level control.

FP16 recall adds M-DCI `copy_to_buffer(dtype=2)`: AVX/F16C conversion from
FP32 index pages into an FP16 pinned staging buffer. A unit test matched NumPy
FP16 conversion exactly. IceCache performs a startup capability probe before
enabling this path.

The native GQA merge replaces 80 small NumPy/Python ordered-dedup calls per
token. Randomized equivalence tests matched the original encounter order for
ratios 1/2/4/8. It reduced dedup from about 4.0 to 0.28 ms/token and total
query postprocessing from about 8–9 to 4.22 ms/token.

## NUMA placement and cross-layer batched gather

The visible GPU 0 is attached to NUMA node 0 (`0000:3b:00.0`). Binding host
memory allocation to node 0 reduced an earlier isolated recall-gather result
by about 2.4 ms/token, although that run also had unrelated GPU waiting and
must not be compared with the GPU 1 table above.

For layer reuse 3, batched gather keeps each layer's KV contents distinct but
lays out one staging transfer as `[L0 K,V | L1 K,V | L2 K,V]`. This reduces
30 native gather/H2D submissions per token to 10 without changing DCI query
or page selection. A matched 40,594-token Passkey A/B on GPU 0, NUMA node 0:

| Variant | TPOT runs (ms) | Mean TPOT (ms) | Gather (ms/tok) | Submissions/tok | Passkey |
|---|---:|---:|---:|---:|---:|
| FP16, per-layer gather | 113.82 / 112.75 | 113.28 | 16.09 / 15.74 | 30 | pass/pass |
| FP16, batched gather | 110.75 / 110.85 | **110.80** | 14.09 / 14.24 | 10 | pass/pass |

This `page_topks=32` case improves mean TPOT by 2.2%. Recall-wait time moved
from about 16.3 to 21.0 ms/token because the larger asynchronous copy is now
paid at the anchor layer; therefore stage timers should not be added as
independent costs.

The established `page_topks=0` configuration recalls roughly twice as many
pages and reverses the result: per-layer gather took 154.13 ms TPOT versus
155.62 ms for batched gather (1.0% slower). Gather CPU time fell only 17.67 to
17.05 ms while recall wait rose 39.86 to 44.71 ms. Keeping small transfers
pipelined with each layer's GPU computation matters more than saving native
submissions at this load. Batched recall therefore remains default-off and is
not part of the recommended configuration.

Binding both computation and memory to NUMA node 0 with 16 local physical
cores was also rejected: TPOT rose to 168.52 ms and native DCI query time to
46.92 ms/token. Use both sockets' 32 physical cores for query throughput, but
place staging/index memory on the GPU-local NUMA node.
