# Experiment 08 — Promotion validation and layerwise schedule

## Setup

- Qasper: 20 deterministic length-stratified examples
- Model: Meta-Llama-3.1-8B-Instruct
- Page budget / size: 64 / 16
- Cross-layer reuse: 3
- DCI parallel level: 2; CPU threads: 32
- FP16 recall enabled; cross-token reuse disabled
- Reference runs once with promotion 0.01

Three configurations:

1. baseline: all anchor layers use 0.01;
2. global fast tree: all anchor layers use 0.05;
3. layerwise hybrid: anchor layers 2–14 use 0.01 and 17–29 use 0.05.

## Results

| Configuration | Qasper F1 | TPOT ms | DCI ms/tok | Native query ms/tok | Recall wait ms/tok |
|---|---:|---:|---:|---:|---:|
| all 0.01 | 45.48 | 138.52 | 21.63 | 16.11 | 44.91 |
| all 0.05 | 41.23 | 129.32 | 13.75 | 9.10 | 44.75 |
| 0.01 early / 0.05 late | 44.69 | 136.44 | 17.90 | 12.68 | 45.51 |

Global 0.05 reduces native-query time by 43.5% and TPOT by 6.6%, but loses
4.25 benchmark F1 points. The layerwise schedule recovers most of the score:
only -0.79 F1 versus baseline, while reducing native-query time by 21.3%.
Its TPOT improvement is only 1.5%, because recall wait (~45 ms/token) is now
much larger than native DCI query time and is unaffected by promotion.

## Paired quality inspection

The global configuration's score loss is highly concentrated. One yes/no
example changes from `Yes.` (100 F1) to a semantically correct but verbose
`Yes, ...` answer (11.11 F1), accounting for most of the aggregate decline.
This is metric-sensitive generation drift rather than a gibberish/corrupted-KV
failure. Nevertheless, the official Qasper metric penalizes it, so the global
setting cannot yet be called quality-preserving.

The layerwise configuration avoids that catastrophic metric outlier. Its
remaining per-example changes are mixed and relatively small, except one
sample dropping 20 points while another gains 5.38 points.

## Decision

The broader 20-example run confirms that promotion is a strong native-query
lever, but rejects global 0.05 as a quality-neutral default. A layerwise tree
schedule is feasible and gives a controllable quality/speed tradeoff.

The next research step should not be another global promotion sweep. It should
measure native-query cost and quality sensitivity by layer group, then assign
the finer tree only to layers with high query cost and low output sensitivity.
Because recall now dominates TPOT, further tree optimization alone has a
limited system-level ceiling unless combined with the existing recall-system
work.

## Artifacts

- Global runner: `experiment/run_08_qasper_promotion20.sh`
- Hybrid runner: `experiment/run_08_qasper_promotion_hybrid20.sh`
- Logs: `experiment/logs/dci_opt/promotion_0p01_qasper20.log`,
  `promotion_0p05_qasper20.log`, and `promotion_hybrid_l17_qasper20.log`
- Predictions: `IceCache/benchmark/pred/llama-3.1/promotion_*_qasper20/`

