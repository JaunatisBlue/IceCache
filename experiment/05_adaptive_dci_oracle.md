# Experiment 05 — Per-head adaptive DCI early-stop feasibility

## Goal

Test whether each KV head can stop DCI search according to the current query's
observed convergence, rather than using a context-length rule or sharing one
fixed decision across heads.

## Method

For every normal reference DCI query, run additional oracle probes with visit
caps of 12.5%, 25%, and 50%. The original 100% result remains the only result
used by model inference. For each KV head independently:

1. compare consecutive probe page sets;
2. stop at the first stage whose overlap reaches a threshold;
3. compare that chosen result against the unchanged 100% result.

Thresholds tested: 80%, 90%, and 95%. This is a feasibility diagnostic, not a
production implementation: repeated probe calls add overhead.

## Configuration

- Qasper: 8 deterministic length-stratified examples
- Model: Meta-Llama-3.1-8B-Instruct
- Page budget / size: 64 / 16
- Sink / window: 2 / 2
- Cross-layer reuse: 3
- Cross-token reuse: disabled
- Comparisons: 11,680 KV-head query instances

## Page recall versus full DCI

| Visit cap | Mean page recall | P10 | Median |
|---:|---:|---:|---:|
| 12.5% | 99.03% | 100% | 100% |
| 25% | 99.96% | 100% | 100% |
| 50% | 100% | 100% | 100% |

## Per-head adaptive oracle

| Convergence threshold | Mean chosen cap | Mean oracle recall | Recall >=95% share |
|---:|---:|---:|---:|
| 80% | 25.23% | 99.994% | 99.974% |
| 90% | 25.87% | 100% | 100% |
| 95% | 26.85% | 100% | 100% |

The 90% rule is the best conservative result in this sample: individual heads
choose their own stopping point, the mean nominal cap is 25.87%, and all
chosen page sets match the full result under the recorded set-recall metric.

## Timing check

A separate two-example run loaded the timing-enabled instrumentation. Mean
native time per layer-level DCI call was:

| Nominal visit cap | Native DCI time |
|---:|---:|
| 12.5% | 29.17 ms |
| 25% | 29.12 ms |
| 50% | 29.11 ms |
| 100% | 31.60 ms |

Although the nominal cap falls by 75%, a 25% query is only 7.8% faster than
the full query. Therefore `num_to_visit=num_points` is predominantly a maximum
bound in this workload; M-DCI usually terminates through another internal
condition before reaching it, or most time is spent in work unaffected by
this cap.

The external staged prototype is not a speedup: executing 12.5% and then 25%
costs roughly 58 ms, versus 31.6 ms for one full call. A real adaptive method
must operate inside one native tree traversal and preserve its state between
checkpoints.

## Conclusion

Later C-level instrumentation (Experiment 06) showed that the partial probes
did not actually perform proportionally less search. Therefore these recall
numbers do **not** establish that heads converge early; they only establish
that changing this nominal cap usually leaves the effective query unchanged.
The per-head convergence claim remains a hypothesis. Changing only
`num_to_visit` cannot produce a large speedup.

The next implementation target is the M-DCI native query loop:

- expose actual visited-node/candidate counts per KV head;
- identify the active internal stop condition and time spent in candidate
  expansion versus result maintenance;
- add a per-head convergence check within the single traversal;
- deactivate converged heads while difficult heads continue searching.

This differs from fixing neighbour count by context length: the active-head
mask evolves from the current query's own candidate stability.

## Artifacts

- Runner: `experiment/run_05_qasper_adaptive_oracle.sh`
- Accuracy log: `experiment/logs/dci_opt/qasper_adaptive_oracle.log`
- Aggregate stats: `experiment/logs/dci_opt/qasper_adaptive_oracle_stats.json`
- Timing log: `experiment/logs/dci_opt/qasper_adaptive_timing.log`
- Timing stats: `experiment/logs/dci_opt/qasper_adaptive_timing_stats.json`
