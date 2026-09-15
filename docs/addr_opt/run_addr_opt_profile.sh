#!/usr/bin/env bash
# Decode-side incremental DCI index-update profile (IceCache sys-optimize).
#
# Usage: run_addr_opt_profile.sh <run-name> <max-samples>
# Env passthrough (all optional, default = production/off):
#   ICECACHE_VEC_ADDR=0 (now the *opt-out*)  old per-leaf data_ptr() loop
#   ICECACHE_VEC_ADDR=1 (production default)  vectorised address preparation
#   ICECACHE_ADDR_EQUIV_CHECK=1        in-vivo address equivalence assertion
#   ICECACHE_PROFILE_CALL_DUMP=<path>  dump per-call index-update records (JSON)
#   ICECACHE_PROFILE_STEP_SAMPLES=1    collect per-decode-step latencies so
#                                      DCI_PROFILE carries mean/std/p50/p95
# Canonical single-run entry point.  See
# docs/DeepSeek_增量DCI地址路径优化_结果.md sections 2 and 5 for the
# measured numbers this reproduces.  Logs land in
# experiment/logs/addr_opt/ (that tree is gitignored -- run artefacts).
set -euo pipefail

ROOT="/home/yx/IceCache"
cd "$ROOT/IceCache/benchmark"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export ICECACHE_DCI_PARALLEL_LEVEL=2
export ICECACHE_FP16_RECALL=1
export ICECACHE_BATCH_LAYER_RECALL=0
export ICECACHE_CROSS_TOKEN_DCI=0
export ICECACHE_TRACE_DCI_CHURN=0
export ICECACHE_TRACE_DCI_ADAPTIVE=0
export ICECACHE_PROMOTION_FAST_START_LAYER=-1
export PYTHONPATH="$ROOT/IceCache/source"

: "${ICECACHE_VEC_ADDR:=1}"
: "${ICECACHE_ADDR_EQUIV_CHECK:=0}"
: "${ICECACHE_PROFILE_STEP_SAMPLES:=0}"
export ICECACHE_VEC_ADDR ICECACHE_ADDR_EQUIV_CHECK ICECACHE_PROFILE_STEP_SAMPLES

NAME="${1:-index_profile_probe}"
MAXS="${2:-2}"

echo "RUN $NAME max_samples=$MAXS VEC_ADDR=$ICECACHE_VEC_ADDR" \
     "EQUIV=$ICECACHE_ADDR_EQUIV_CHECK STEP_SAMPLES=$ICECACHE_PROFILE_STEP_SAMPLES" \
     "CALL_DUMP=${ICECACHE_PROFILE_CALL_DUMP:-}"

/home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 --n_prefetch_layers 0 \
  --ratio_1 0.01 --ratio_2 0.2 \
  --profile-dci --profile-warmup-tokens 2 \
  --max-samples "$MAXS" --name "$NAME" --datasets qasper
