#!/usr/bin/env bash
# Experiment 19 - bytes-vs-quality frontier on the PCIe x1 link.
#
# Transfer time is size-linear at 0.82 GB/s, so shrinking the per-layer page
# budget directly removes transfer time.  This sweep measures how much Qasper
# quality that costs, i.e. the quality-per-byte curve that any
# bandwidth-aware policy has to trade against.
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

SAMPLES="${SAMPLES:-20}"
BUDGET="${BUDGET:-64}"
TOPKS="${TOPKS:-0}"
PFLAYERS="${PFLAYERS:-0}"
NAME="${NAME:-budget${BUDGET}_qasper${SAMPLES}}"

mkdir -p "$ROOT/experiment/logs/dci_opt"
log="$ROOT/experiment/logs/dci_opt/${NAME}.log"
echo "=== RUN start $(date -Is) name=$NAME samples=$SAMPLES budget=$BUDGET topks=$TOPKS pf=$PFLAYERS ==="
/home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets "$BUDGET" --page-topks "$TOPKS" \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
  --n_prefetch_layers "$PFLAYERS" \
  --ratio_1 0.01 --ratio_2 0.2 \
  --profile-dci --profile-warmup-tokens 2 \
  --max-samples "$SAMPLES" --name "$NAME" --datasets qasper > "$log" 2>&1

/home/yx/miniconda3/envs/icecache/bin/python longbench_eval.py \
  --model llama-3.1 --name "$NAME" >> "$log" 2>&1

echo "=== RUN done $(date -Is) ==="
grep -oE '"decode_tpot_ms": [0-9.]+|"recall_pages_per_token": [0-9.]+|"recall_wait_ms_per_token": [0-9.]+|"recall_gather_ms_per_token": [0-9.]+|"native_query_ms_per_token": [0-9.]+|"dci_select_ms_per_token": [0-9.]+' "$log" | tail -8 || true
echo "--- F1 ---"
cat "$ROOT/IceCache/benchmark/pred/llama-3.1/$NAME/result.json" 2>/dev/null || echo "NO RESULT"
