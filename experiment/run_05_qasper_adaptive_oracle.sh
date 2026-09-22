#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yx/IceCache"
cd "$ROOT/IceCache/benchmark"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=16
export ICECACHE_DCI_PARALLEL_LEVEL=0
export ICECACHE_CROSS_TOKEN_DCI=0
export ICECACHE_TRACE_DCI_CHURN=0
export ICECACHE_TRACE_DCI_ADAPTIVE=1
export ICECACHE_TRACE_DCI_LEVELS="0.125,0.25,0.5,1.0"
export ICECACHE_TRACE_DCI_THRESHOLDS="0.8,0.9,0.95"
RUN_NAME="${RUN_NAME:-dci_adaptive_oracle_qasper8}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
export ICECACHE_TRACE_DCI_ADAPTIVE_PATH="${ICECACHE_TRACE_DCI_ADAPTIVE_PATH:-$ROOT/experiment/logs/dci_opt/qasper_adaptive_oracle_stats.json}"
export PYTHONPATH="$ROOT/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"

exec /home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
  --max-samples "$MAX_SAMPLES" --name "$RUN_NAME" \
  --datasets qasper
