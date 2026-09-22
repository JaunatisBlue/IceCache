# Experiment 07 — M-DCI promotion/tree-shape sweep

## Motivation

Experiment 06 found that many leaf-level calls contain roughly the same order
of points as the requested 60 neighbours and therefore enter the full-sort
branch. This experiment tests the tree-construction knob `promotion_prob`
instead of another query-time cap.

## Synthetic structure/query probe

Configuration matches IceCache's DCI dimensions: 4096 points, head dimension
128, four Q heads per KV head, 60 neighbours, `promotion_prob_subseq=0.2`, and
parallel level 2. Each point is repeated across three additional runs; the
trend was stable.

| promotion | levels | leaves | N/leaves proxy | median query ms |
|---:|---:|---:|---:|---:|
| 0.0025 | 3 | 264 | 15.52 | 0.72–0.74 |
| 0.005 | 4 | 271 | 15.11 | 0.67–0.82 |
| 0.01 | 4 | 283 | 14.47 | 0.92–0.94 |
| 0.02 | 4 | 305 | 13.43 | 0.90–0.91 |
| 0.05 | 5 | 388 | 10.56 | 0.52–0.54 |
| 0.10 | 5 | 503 | 8.14 | 0.36–0.37 |

Important correction: increasing first-level promotion did not create fewer,
larger leaves in this implementation. It created more leaves and a finer
partition, while substantially reducing query time. `N/leaves` is only a
global proxy and is not identical to the local node-size statistic collected
by the C probe.

## Qasper quality and clean decode check

Eight deterministic length-stratified examples were run with identical model,
budget, cross-layer reuse, 32 CPU threads, parallel level 2, and FP16 recall.
Only `ratio_1` changed.

| promotion | Qasper F1 | mean per-example decode ms |
|---:|---:|---:|
| 0.01 | 26.93 | 149.83 |
| 0.05 | 31.23 | 138.29 |
| 0.10 | 31.33 | 148.80 |

The subset is too small for a final quality claim, but neither higher setting
caused a quality collapse. The per-example timer is noisy because generated
lengths and early EOS differ across variants.

## Two-example detailed profile

| promotion | TPOT ms | DCI total ms/tok | native query ms/tok | recall wait ms/tok |
|---:|---:|---:|---:|---:|
| 0.01 | 155.16 | 24.95 | 18.95 | 48.33 |
| 0.05 | 135.99 | 14.35 | 9.56 | 49.95 |
| 0.10 | 132.19 | 11.77 | 6.96 | 47.54 |

Relative to 0.01, promotion 0.05 halves native-query time and lowers profiled
TPOT by 12.4%. Promotion 0.10 lowers native-query time by 63.3% and profiled
TPOT by 14.8%. Recall wait is essentially unchanged, confirming that the gain
comes from tree querying rather than moving fewer KV pages.

## Interpretation and next direction

Promotion probability is a real algorithmic lever, unlike the ineffective
`num_to_visit` cap. The current evidence favors a finer M-DCI hierarchy, not
the proposed strategy of enlarging leaves to avoid the leaf full-sort branch.
The full-sort count alone was therefore not a sufficient optimization target:
smaller local sorts plus better partitioning can be faster even if the tree has
more leaves.

`promotion=0.05` is the conservative candidate because it improves the larger
eight-example speed measurement and quality. `0.10` has the best native-query
profile but noisier end-to-end timing. The next minimal validation should use
the same fixed Qasper subset with more examples for only 0.01 versus 0.05,
rather than another wide parameter sweep.

## Artifacts

- Microbenchmark: `experiment/probe/mdci_promotion_sweep.py`
- Runner: `experiment/run_07_qasper_promotion_sweep.sh`
- Profile runner: `experiment/run_07_qasper_promotion_profile.sh`
- Logs: `experiment/logs/dci_opt/promotion_*_qasper8.log`
- Profile logs: `experiment/logs/dci_opt/promotion_profile_*_qasper2.log`

