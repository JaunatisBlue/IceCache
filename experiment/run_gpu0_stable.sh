#!/usr/bin/env bash
set -euo pipefail

# Resume the GPU-0 LongBench partition with serial DCI queries.  Existing
# JSONL rows are retained by longbench_pred.py, so this starts at NarrativeQA
# item 134 and subsequently handles the remaining GPU-0 datasets.
ROOT="/home/yx/IceCache"
BENCHMARK_DIR="$ROOT/IceCache/benchmark"
PYTHON_BIN="/home/yx/miniconda3/envs/icecache/bin/python"
MODEL_PATH="/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
RUN_NAME="${1:-ice64_table1}"

cd "$BENCHMARK_DIR"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${ICECACHE_OMP_NUM_THREADS:-16}"
export ICECACHE_DCI_PARALLEL_LEVEL="${ICECACHE_DCI_PARALLEL_LEVEL:-0}"
export PYTHONPATH="$ROOT/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" longbench_pred.py \
  --icecache --model llama-3.1 --model-path "$MODEL_PATH" \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --name "$RUN_NAME" \
  --datasets narrativeqa 2wikimqa musique gov_report multi_news samsum passage_count lcc
