#!/usr/bin/env bash
# Experiment 18 - quantify layer-prefetch overlap on the PCIe x1 link.
#
# Baseline config is identical to exp-10 Step0 / exp-17 diag_baseline:
#   qasper20, budget 64, page 16, reuse 3, FP16 recall, promotion 0.01
# Only --n_prefetch_layers changes (default 0 => recall is issued and awaited
# on the main thread, so the whole H2D lands on the critical path).
#
# PFLAYERS=1 launches the next layer's DCI selection + gather + H2D early and
# consumes it one layer later, so the transfer can overlap GPU compute.
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
PFLAYERS="${PFLAYERS:-0}"
NAME="${NAME:-pf${PFLAYERS}_qasper${SAMPLES}}"
DIAG="${DIAG:-0}"
export ICECACHE_DIAG="$DIAG"
export ICECACHE_DIAG_MAX_RECORDS="${MAXREC:-2000}"
export ICECACHE_DIAG_MAX_EVENTS="${MAXEV:-1000}"

mkdir -p "$ROOT/experiment/logs/dci_opt"
export ICECACHE_DIAG_DUMP="$ROOT/experiment/logs/dci_opt/${NAME}.npz"

log="$ROOT/experiment/logs/dci_opt/${NAME}.log"
echo "=== RUN start $(date -Is) name=$NAME samples=$SAMPLES n_prefetch_layers=$PFLAYERS diag=$DIAG ==="
/home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
  --n_prefetch_layers "$PFLAYERS" \
  --ratio_1 0.01 --ratio_2 0.2 \
  --profile-dci --profile-warmup-tokens 2 \
  --max-samples "$SAMPLES" --name "$NAME" --datasets qasper > "$log" 2>&1

/home/yx/miniconda3/envs/icecache/bin/python longbench_eval.py \
  --model llama-3.1 --name "$NAME" >> "$log" 2>&1

echo "=== RUN done $(date -Is) ==="
grep -oE '"decode_tpot_ms": [0-9.]+|"recall_wait_ms_per_token": [0-9.]+|"recall_gather_ms_per_token": [0-9.]+|"native_query_ms_per_token": [0-9.]+|"dci_select_ms_per_token": [0-9.]+|"dci_share": [0-9.]+' "$log" | tail -10 || true
echo "--- F1 ---"
cat "$ROOT/IceCache/benchmark/pred/llama-3.1/$NAME/result.json" 2>/dev/null || echo "NO RESULT"
