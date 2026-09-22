#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yx/IceCache"
cd "$ROOT/IceCache/benchmark"

# Stable host-side DCI configuration.  Existing JSONL records are resumed.
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export ICECACHE_DCI_PARALLEL_LEVEL=0
export PYTHONPATH="$ROOT/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"

exec /home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --name ice64_table1 \
  --datasets narrativeqa
