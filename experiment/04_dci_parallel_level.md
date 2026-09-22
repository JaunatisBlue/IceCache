# Experiment 04 — DCI parallel-level sweep

## Question

Does the DCI native CPU parallel route reduce or increase decode latency at a
long context?  Only `ICECACHE_DCI_PARALLEL_LEVEL` changes between runs.

## Fixed configuration

- Model: Meta-Llama-3.1-8B-Instruct
- Method: IceCache; cross-token DCI gate disabled
- Context: 40,060 tokens (Passkey)
- GPU page budget / page size: 64 / 16
- Sink / window pages: 2 / 2
- Cross-layer reuse cycle: 3
- One fixed prompt, 32 generated tokens; first 8 tokens excluded from timing
- GPU: one otherwise idle A100 80 GB

## Results

| `ICECACHE_DCI_PARALLEL_LEVEL` | DCI select ms/token | DCI share of decode | Decode TPOT ms |
|---:|---:|---:|---:|
| 0 | 598.12 | 77.14% | 775.37 |
| 1 | 55.90 | 24.73% | 226.10 |
| 2 | **44.26** | **20.51%** | **215.75** |

All three runs returned the correct Passkey.  They performed the same 230 DCI
selection calls across 23 measured tokens (10 calls/token), so latency changes
are attributable to the native DCI parallel-level route rather than a change
in call count.

## Conclusion

On this isolated machine, level 2 is fastest: versus level 0 it lowers DCI
selection time by 92.6% and TPOT by 72.2%.  Level 1 already removes most of
the slowdown; level 2 gives a further 20.8% DCI-time reduction.

This sweep does **not** support the claim that OpenMP parallelism is inherently
the main overhead in the isolated run.  Rather, the level-0 native route is
very slow, and the parallel route is beneficial.  The code comment about
oversubscription remains relevant when multiple workloads share CPU cores or
when the native implementation nests parallel regions, but that needs a
separate thread-count / concurrent-load experiment.

Even with level 2, DCI remains 44.26 ms/token and 20.5% of TPOT.  Thus the
next algorithmic target remains reducing DCI work or calls, while level 2
should be used as the CPU baseline for those experiments.

## Raw logs

- `experiment/logs/dci_opt/parallel_level_0_37k.log`
- `experiment/logs/dci_opt/parallel_level_1_37k.log`
- `experiment/logs/dci_opt/parallel_level_2_37k.log`
