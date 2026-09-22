#!/usr/bin/env bash
set -euo pipefail
# run_16: early-stop + return-truncation (v2 kernel)
#  - prop=1.0: baseline (truncation disabled, early-stop disabled)
#  - prop=0.5/0.25/0.125: early-stop + truncation active
# Primary metric: recall_pages_per_token must DROP (vs run_15's flat 5172),
# and F1 must stay >= 45.43.

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
export PYTHONPATH="$ROOT/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"

run_one() {
  local prop="$1"
  local suffix="${prop/./p}"
  local name="earlystop_trunc_${suffix}_qasper20"
  local log="$ROOT/experiment/logs/dci_opt/${name}.log"
  ICECACHE_DCI_PROP_TO_VISIT="$prop" \
  /home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
    --icecache --model llama-3.1 \
    --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --page-size 16 --page-budgets 64 --page-topks 0 \
    --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
    --ratio_1 0.01 --ratio_2 0.2 \
    --profile-dci --profile-warmup-tokens 2 \
    --max-samples 20 --name "$name" --datasets qasper > "$log" 2>&1
  /home/yx/miniconda3/envs/icecache/bin/python longbench_eval.py \
    --model llama-3.1 --name "$name" >> "$log" 2>&1
  echo "DONE $name prop=$prop" >> "$log"
}

run_one 1.0
run_one 0.5
run_one 0.25
run_one 0.125