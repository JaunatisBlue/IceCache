#!/usr/bin/env bash
# Experiment 10 Step 2 — Whitelist verification on Qasper20.
# After Step 1 screening identifies layers whose DCI can be skipped without
# quality loss, verify the combined skip set on the full 20-sample subset.
# Edit CANDIDATE below based on Step 1 results.
set -euo pipefail

ROOT="/home/yx/IceCache"
cd "$ROOT/IceCache/benchmark"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=32
export ICECACHE_DCI_PARALLEL_LEVEL=2
export ICECACHE_FP16_RECALL=1
export ICECACHE_BATCH_LAYER_RECALL=0
export ICECACHE_CROSS_TOKEN_DCI=0
export ICECACHE_TRACE_DCI_CHURN=0
export ICECACHE_TRACE_DCI_ADAPTIVE=0
export ICECACHE_PROMOTION_FAST_START_LAYER=-1
export PYTHONPATH="$ROOT/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"

PY=/home/yx/miniconda3/envs/icecache/bin/python

# EDIT ME after Step 1: comma-separated anchor layers whose DCI is skipped.
CANDIDATE=""

name="layer_sens_whitelist_qasper20"
log="$ROOT/experiment/logs/dci_opt/${name}.log"
ICECACHE_SKIP_DCI_LAYERS="$CANDIDATE" \
$PY longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
  --ratio_1 0.01 --ratio_2 0.2 \
  --profile-dci --profile-warmup-tokens 2 \
  --max-samples 20 --name "$name" --datasets qasper > "$log" 2>&1
$PY longbench_eval.py --model llama-3.1 --name "$name" >> "$log" 2>&1

echo "DONE $CANDIDATE -> $name : $(grep -o '"qasper": [0-9.]*' "$ROOT/IceCache/benchmark/pred/llama-3.1/$name/result.json" 2>/dev/null || echo NA)"