#!/usr/bin/env bash
# 36k passkey A/B for ICECACHE_VEC_ADDR, 2 repetitions per arm, interleaved.
#
# Why this workload: 36k context is where the decode-side incremental DCI
# update fires many times (page boundary every 16 tokens) and where the recall
# chain dominates, so it is the harshest setting for the address path.
#
# Design (deliberately removes the confounds found on 2026-09-14):
#   * `N_GARBAGES=134775` + `--num-tests 1` -> one fixed, deterministic 36k-token
#     prompt.
#     ⚠️ `N_GARBAGES` slices *characters* (`garbage_inf[:n_garbage]`), not
#     tokens (~3.75 chars/token).  Calibrated with the real tokenizer at
#     loc=0.0:
#         N_GARBAGES= 36000 ->  9,660 tokens   (the obvious-but-wrong guess)
#         N_GARBAGES=134775 -> 36,000 tokens   <- this run
#         N_GARBAGES=140000 -> 37,394 tokens   (the docs' historical "passkey 37k")
#         N_GARBAGES=150000 -> 40,060 tokens   (the docs' historical "passkey 40k")
#     The historical 36k/37k/40k runs were never scripted -- `N_GARBAGES` appears
#     nowhere else in the repo -- so this file is the first reproducible setting.
#   * `--profile-new-tokens 96`             -> both arms decode the SAME number
#                                              of tokens (no composition skew)
#   * `--profile-warmup-tokens 8`           -> drop the prefill-adjacent steps
#   * interleaved A,B,A,B                   -> drifts cancel in the arm means
#
# Usage: run_passkey36k_ab.sh [tag]
#   tag defaults to p36k  -> logs p36k_A1, p36k_B1, p36k_A2, p36k_B2
#
# Result summary:
#   python docs/probe/summarize_reps.py \
#     A=experiment/logs/addr_opt/p36k_A1.log,experiment/logs/addr_opt/p36k_A2.log \
#     B=experiment/logs/addr_opt/p36k_B1.log,experiment/logs/addr_opt/p36k_B2.log
#
# See docs/DeepSeek_增量DCI地址路径优化_结果.md section 5.6.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/yx/IceCache
OUT="$ROOT/experiment/logs/addr_opt"
mkdir -p "$OUT"

TAG="${1:-p36k}"
REPS="${REPS:-2}"
NUM_TESTS="${NUM_TESTS:-1}"
NEW_TOKENS="${NEW_TOKENS:-96}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export N_GARBAGES=134775
export ICECACHE_DCI_PARALLEL_LEVEL=2
export ICECACHE_FP16_RECALL=1
export ICECACHE_BATCH_LAYER_RECALL=0
export ICECACHE_CROSS_TOKEN_DCI=0
export ICECACHE_TRACE_DCI_CHURN=0
export ICECACHE_TRACE_DCI_ADAPTIVE=0
export ICECACHE_PROMOTION_FAST_START_LAYER=-1
export ICECACHE_PROFILE_STEP_SAMPLES=1
export PYTHONPATH="$ROOT/IceCache/source"

step() { echo "[$(date +%H:%M:%S)] $*"; }

run_arm() {
  local vec="$1" name="$2"
  ( cd "$ROOT/IceCache/benchmark"
    ICECACHE_VEC_ADDR="$vec" \
    /home/yx/miniconda3/envs/icecache/bin/python passkey_pred.py \
      --icecache --model llama-3.1 \
      --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
      --num-tests "$NUM_TESTS" --page-size 16 --page-budgets 64 --page-topks 0 \
      --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
      --n_prefetch_layers 0 --ratio_1 0.01 --ratio_2 0.2 \
      --profile-dci --profile-warmup-tokens 8 --profile-new-tokens "$NEW_TOKENS" \
      --name "$name" )
}

for rep in $(seq 1 "$REPS"); do
  step "${TAG}_A${rep} (baseline, vec=0)"
  run_arm 0 "${TAG}_A${rep}" > "$OUT/${TAG}_A${rep}.log" 2>&1
  step "${TAG}_B${rep} (vectorised, vec=1)"
  run_arm 1 "${TAG}_B${rep}" > "$OUT/${TAG}_B${rep}.log" 2>&1
done

step "$TAG ALLDONE"
