#!/usr/bin/env bash
set -euo pipefail

# Table 1, Llama-3.1-8B-Instruct, ICE, cache budget 64.
# In this codebase the table budget is passed directly as the number of GPU KV
# pages. The compiled attention extension supports a page size of 16 tokens.
cd "$(dirname "$0")/../IceCache/benchmark"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export PYTHONPATH="$(pwd)/../source:${PYTHONPATH:-}"

/home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache \
  --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --datasets narrativeqa qasper multifieldqa_en \
  --page-size 16 \
  --page-budgets 64 \
  --page-topks 0 \
  --n-sink-pages 2 \
  --n-win-pages 2 \
  --name ice64_single_document_qa

/home/yx/miniconda3/envs/icecache/bin/python longbench_eval.py \
  --model llama-3.1 \
  --name ice64_single_document_qa
