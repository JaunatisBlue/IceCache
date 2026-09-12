#!/usr/bin/env bash
# Experiment 17 — ICECACHE_DIAG joint diagnostic.
# One run yields BOTH:
#   (A) the five-segment recall timing split (addr-prep / copy_to_buffer / H2D / cast / wait)
#   (B) per (layer, head) selected leaf ids + real CPU addresses, dumped to .npz
# Same config as the exp-10 Step0 qasper20 baseline, so F1 stays comparable to 45.48.
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

SAMPLES="${SAMPLES:-2}"
NAME="${NAME:-diag_qasper${SAMPLES}}"
export ICECACHE_DIAG="${DIAG:-1}"
export ICECACHE_DIAG_MAX_RECORDS="${MAXREC:-2000}"
export ICECACHE_DIAG_MAX_EVENTS="${MAXEV:-1000}"

mkdir -p "$ROOT/experiment/logs/dci_opt"
export ICECACHE_DIAG_DUMP="$ROOT/experiment/logs/dci_opt/${NAME}.npz"

log="$ROOT/experiment/logs/dci_opt/${NAME}.log"
echo "=== RUN start $(date -Is) name=$NAME samples=$SAMPLES diag=$ICECACHE_DIAG ==="
/home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
  --ratio_1 0.01 --ratio_2 0.2 \
  --profile-dci --profile-warmup-tokens 2 \
  --max-samples "$SAMPLES" --name "$NAME" --datasets qasper > "$log" 2>&1

/home/yx/miniconda3/envs/icecache/bin/python longbench_eval.py \
  --model llama-3.1 --name "$NAME" >> "$log" 2>&1

echo "=== RUN done $(date -Is) ==="
grep -E "ICECACHE-DIAG" "$log" || true
echo "--- profile lines ---"
grep -E "recall_gather_ms_per_token|recall_wait_ms_per_token|native_query_ms_per_token|decode_tpot_ms|F1|f1" "$log" | tail -20 || true
echo "--- npz ---"
ls -la "$ICECACHE_DIAG_DUMP" 2>/dev/null || echo "NO NPZ"
