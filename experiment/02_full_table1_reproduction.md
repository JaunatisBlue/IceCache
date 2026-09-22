# Experiment 02 — Table 1 ICE reproduction

## Scope

Reproduce all 16 LongBench columns for Llama-3.1-8B-Instruct with vanilla
IceCache at cache budgets 64, 128, and 256.

## Execution

`run_longbench_ice_partitioned.sh` partitions non-overlapping datasets across
GPU 0 and GPU 1. Predictions are checkpointed after every example, logs are
kept in `experiment/logs/<run-name>/`, and the script invokes the supplied
`longbench_eval.py` only after both workers complete.

## Table 1 reference scores

| Budget | NrtvQA | Qasper | MF-en | Avg |
| --- | ---: | ---: | ---: | ---: |
| 64 | 27.4 | 43.2 | 55.7 | 47.8 |
| 128 | 30.0 | 44.7 | 56.5 | 48.6 |
| 256 | 30.6 | 44.7 | 56.3 | 49.0 |

## Status

- 64: in progress; existing predictions are resumed.
- 128: pending.
- 256: pending.
