#!/usr/bin/env bash
# perf profile of the decode path, to find where CPU cycles actually go.
set -euo pipefail

ROOT="/home/yx/IceCache"
cd "$ROOT/IceCache/benchmark"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32
export ICECACHE_DCI_PARALLEL_LEVEL=2
export ICECACHE_FP16_RECALL=1
export ICECACHE_BATCH_LAYER_RECALL=0
export ICECACHE_CROSS_TOKEN_DCI=0
export ICECACHE_TRACE_DCI_CHURN=0
export ICECACHE_TRACE_DCI_ADAPTIVE=0
export ICECACHE_PROMOTION_FAST_START_LAYER=-1
export PYTHONPATH="$ROOT/IceCache/source"

OUT=/tmp/icecache_perf.data
rm -f "$OUT"

perf record -F 1999 -g -o "$OUT" -- \
  /home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 --n_prefetch_layers 0 \
  --ratio_1 0.01 --ratio_2 0.2 \
  --profile-dci --profile-warmup-tokens 2 \
  --max-samples 3 --name perf_probe --datasets qasper > /tmp/perf_run.log 2>&1 || true

echo "=== samples collected ==="
perf report --stdio -i "$OUT" 2>/dev/null | grep -E "Event count|Samples" | head -3

echo
echo "=== TOP SELF SYMBOLS ==="
perf report --stdio -i "$OUT" --sort symbol --percent-limit 0.7 2>/dev/null | sed -n '5,45p'

echo
echo "=== BY DSO ==="
perf report --stdio -i "$OUT" --sort dso --percent-limit 1.0 2>/dev/null | sed -n '5,20p'

echo
echo "=== BY COMM ==="
perf report --stdio -i "$OUT" --sort comm --percent-limit 1.0 2>/dev/null | sed -n '5,20p'
