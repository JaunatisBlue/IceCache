## Early-stop experiments (run15/run16)

| config | F1 | TPOT(ms) | native_query(ms/t) | dci_select(ms/t) | pages/t | wait(ms/t) | gather(ms/t) | steps |
|---|---|---|---|---|---|---|---|---|
| earlystop_1p0_qasper20 | 45.43 | 143.07 | 17.13 | 22.99 | 5196 | 44.78 | 18.80 | 344 |
| earlystop_0p5_qasper20 | 45.63 | 138.82 | 15.50 | 21.17 | 5172 | 44.68 | 18.04 | 358 |
| earlystop_trunc_1p0_qasper20 | 45.46 | 143.07 | 17.40 | 23.25 | 5202 | 44.90 | 18.65 | 344 |
| earlystop_trunc_0p5_qasper20 | 45.72 | 140.08 | 17.24 | 22.94 | 5181 | 44.68 | 17.50 | 353 |
| earlystop_trunc_0p25_qasper20 | 45.57 | 143.56 | 17.33 | 23.24 | 5205 | 44.84 | 18.55 | 322 |

Baseline (prop=1.0): F1=45.43, TPOT=143.07ms