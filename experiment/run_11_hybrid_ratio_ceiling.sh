#!/usr/bin/env bash
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
export PYTHONPATH="$ROOT/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"

# hybrid l17 + ratio 0.10 tail: more aggressive tail acceleration
export ICECACHE_PROMOTION_FAST_START_LAYER=17
export ICECACHE_PROMOTION_FAST_RATIO=0.10
name="promotion_hybrid_l17_r10_qasper20"
log="$ROOT/experiment/logs/dci_opt/${name}.log"
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
echo "DONE $name" >> "$log"
