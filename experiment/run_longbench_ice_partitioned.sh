#!/usr/bin/env bash
set -euo pipefail

# Usage: run_longbench_ice_partitioned.sh <page_budget> <run_name>
# Each worker owns a disjoint dataset set. longbench_pred.py resumes from any
# existing JSONL rows, so interrupted jobs are safe to restart.
BUDGET="${1:?page budget required}"
RUN_NAME="${2:?run name required}"
ROOT="/home/yx/IceCache"
BENCHMARK_DIR="$ROOT/IceCache/benchmark"
PYTHON_BIN="/home/yx/miniconda3/envs/icecache/bin/python"
MODEL_PATH="/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
LOG_DIR="$ROOT/experiment/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --icecache
  --model llama-3.1
  --model-path "$MODEL_PATH"
  --page-size 16
  --page-budgets "$BUDGET"
  --page-topks 0
  --n-sink-pages 2
  --n-win-pages 2
  --name "$RUN_NAME"
)

run_worker() {
  local gpu="$1"
  local log="$2"
  shift 2
  (
    cd "$BENCHMARK_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu"
    # Avoid nested OpenMP/DCI parallelism, which has deadlocked on long
    # NarrativeQA samples.  This affects host-side search speed only, not
    # cache selection semantics or model outputs.
    export OMP_NUM_THREADS="${ICECACHE_OMP_NUM_THREADS:-16}"
    export ICECACHE_DCI_PARALLEL_LEVEL="${ICECACHE_DCI_PARALLEL_LEVEL:-0}"
    export PYTHONPATH="$ROOT/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" longbench_pred.py "${COMMON_ARGS[@]}" --datasets "$@"
  ) >"$log" 2>&1
}

# Work is balanced by the published average context lengths. Worker 0 also
# resumes the partially completed narrativeqa file; worker 1 resumes qasper
# and multifieldqa_en if their JSONL files already exist.
run_worker 0 "$LOG_DIR/gpu0.log" \
  narrativeqa 2wikimqa musique gov_report multi_news samsum passage_count lcc &
PID0=$!
run_worker 1 "$LOG_DIR/gpu1.log" \
  qasper multifieldqa_en hotpotqa qmsum trec triviaqa passage_retrieval_en repobench-p &
PID1=$!

wait "$PID0"
wait "$PID1"

(
  cd "$BENCHMARK_DIR"
  "$PYTHON_BIN" longbench_eval.py --model llama-3.1 --name "$RUN_NAME"
) >"$LOG_DIR/eval.log" 2>&1
