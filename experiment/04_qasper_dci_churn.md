# Experiment 04 — Consecutive-token DCI page churn on Qasper

## Question

Does full reference DCI select nearly the same pages for adjacent decode
tokens, such that cross-token reuse or a small incremental repair is a sound
default assumption?

## Setup

- Dataset: Qasper, 20 deterministic length-stratified examples
- Model: Meta-Llama-3.1-8B-Instruct
- GPU: A100 80 GB, GPU 0
- Page budget / size: 64 / 16
- Sink / window pages: 2 / 2
- Cross-layer reuse: 3
- Cross-token reuse: disabled
- DCI query: unchanged reference implementation
- Selection comparison granularity: adjacent DCI calls from the same anchor
  layer and KV head; comparisons never cross sample boundaries

Subset indices:

`[33, 39, 48, 51, 53, 57, 59, 79, 82, 85, 92, 106, 119, 133, 134, 150, 153, 165, 191, 194]`

## Metric

For selection sets `P_t` and `P_(t+1)` of equal width:

`overlap = |P_t intersect P_(t+1)| / |P_(t+1)|`

The same metric is also computed for the first 25% and first 50% of the
returned order. These prefix metrics describe DCI return-order stability, not
true attention mass.

## Results

Reference Qasper score on this fixed subset: **45.07 F1**.

There are 29,040 layer/head adjacent-token comparisons.

| Selection portion | Mean overlap | P10 | Median | P90 |
|---|---:|---:|---:|---:|
| Full top-k | 64.54% | 45.00% | 66.67% | 81.67% |
| First 50% | 58.12% | 33.33% | 60.00% | 80.00% |
| First 25% | 52.97% | 26.67% | 53.33% | 80.00% |

Additional stability rates:

| Metric | Result |
|---|---:|
| Exactly identical ordered selection | 0.00% |
| Full-set overlap >= 90% | 0.79% |
| Full-set overlap >= 75% | 27.08% |

Mean full-set overlap by KV head ranges from **60.23%** (head 5) to
**68.34%** (head 1). Mean overlap by anchor layer ranges from **55.58%**
(layer 2) to **69.43%** (layer 26). Thus the churn is not confined to one
isolated head, although early layers are less stable than later layers.

The benchmark's existing wall-clock timer reported a mean decode latency of
546.93 ms/token across 19 finite per-example measurements (one example ended
too early for that timer and produced NaN). This is a diagnostic run with
trace overhead, so this number is not used as a clean performance baseline.

## Interpretation

The default premise for whole-set cross-token reuse is not supported on this
Qasper subset. On average, about 35% of selected pages change between adjacent
queries, and only 0.79% of comparisons retain at least 90% of their pages.

The data also does not support the specific hypothesis "the head pages are
stable and only low-ranked tail pages rotate": DCI's first-quarter return
overlap is lower than its full-set overlap. This makes unconditional partial
repair of only a few tail pages unsafe as the next implementation.

This experiment does **not** establish that every changed page matters to the
attention output. The DCI API used here does not expose a directly usable
page-level attention contribution, and return-order prefix is only a proxy.
It remains possible that many replacements are near-ties with little quality
impact.

## Decision

Do not prioritize blind whole-page reuse or a fixed "replace two pages"
incremental algorithm. The next low-risk direction should reduce or hide the
cost of producing a fresh result:

1. algorithmic early termination with a correctness fallback;
2. candidate-level prediction/prefetch followed by real-query reranking;
3. cross-layer pipelining where enough DCI work can be overlapped;
4. reuse only if a future confidence signal predicts selection stability,
   rather than assuming stability from token adjacency.

## Artifacts

- Runner: `experiment/run_04_qasper_churn.sh`
- Log: `experiment/logs/dci_opt/qasper_churn20.log`
- Predictions: `IceCache/benchmark/pred/llama-3.1/dci_churn_qasper20/qasper.jsonl`
- Score: `IceCache/benchmark/pred/llama-3.1/dci_churn_qasper20/result.json`
- Instrumentation: `IceCache/source/icecache/infer_state.py`

