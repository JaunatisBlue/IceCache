# Experiment 01 — IceCache 64-token budget, Single-Document QA

## Objective

Reproduce the `Llama-3.1-8B-Instruct / ICE / Budget=64` row of Table 1 for
`narrativeqa`, `qasper`, and `multifieldqa_en`.

Paper reference scores: 27.4, 43.2, and 55.7 respectively.

## Configuration

- Model: `/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- Method: IceCache
- Table budget: 64 GPU KV pages
- Page size: 16 tokens/page (the compiled extension supports 16 and 32)
- Page top-k reserve: 0 pages
- Sink pages / window pages: 2 / 2
- Seed: 42 (set by `longbench_pred.py`)
- CPU threads: 64

The executable command is `experiment/run_01_ice64_single_document_qa.sh`.

## Status

The repository's `page-budgets` argument denotes pages, not tokens. A literal
four-page configuration would reserve all pages for the two sink and two window
regions and fails the cache manager's minimum-budget invariant, so it cannot
represent the published `Budget=64` row.

Pending installation of the external M-DCI `dciknn` package and full benchmark
execution.
