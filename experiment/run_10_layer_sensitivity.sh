#!/usr/bin/env bash
# Experiment 10 Step 1 — Layer-sensitivity screening on Qasper8.
# For each anchor layer L in {2,5,8,...,29}: run Qasper8 with
# ICECACHE_SKIP_DCI_LAYERS=L (that layer reuses previous selection, no DCI/recall).
# Then eval each. Baseline (no skip) is layer_sens_baseline_qasper20 (45.48 @20 samples);
# here we also run a fresh no-skip Qasper8 as the 8-sample anchor.
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
export ICECACHE_SKIP_DCI_LAYERS=""   # reset

PY=/home/yx/miniconda3/envs/icecache/bin/python

run_one () {
  local skip="$1" name="$2"
  export ICECACHE_SKIP_DCI_LAYERS="$skip"
  local log="$ROOT/experiment/logs/dci_opt/${name}.log"
  $PY longbench_pred.py \
    --icecache --model llama-3.1 \
    --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --page-size 16 --page-budgets 64 --page-topks 0 \
    --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
    --ratio_1 0.01 --ratio_2 0.2 \
    --profile-dci --profile-warmup-tokens 2 \
    --max-samples 8 --name "$name" --datasets qasper > "$log" 2>&1
  $PY longbench_eval.py --model llama-3.1 --name "$name" >> "$log" 2>&1
  local f1
  f1=$(grep -o '"qasper": [0-9.]*' "$ROOT/IceCache/benchmark/pred/llama-3.1/$name/result.json" 2>/dev/null | head -1) || true
  echo "$skip -> $name : ${f1:-NA}"
}

# 8-sample no-skip anchor
run_one "" "layer_sens_base_qasper8"

# per-anchor-layer skip: layers 2..29 step 3
for L in 2 5 8 11 14 17 20 23 26 29; do
  run_one "$L" "layer_sens_skip${L}_qasper8"
done

echo "ALL DONE"