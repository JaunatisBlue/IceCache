#!/usr/bin/env bash
# Control: is the DCI selection itself run-to-run reproducible?
#
# Runs the PRE-PATCH infer_state.py twice with the same full-run diag dump, using
# a private copy of the package, so the working tree is never touched.
#
# Purpose: the baseline-vs-vectorised divergence (~71.7% of records) is only
# meaningful next to the divergence between two runs of identical code.  This
# script produces that control.  Result (2026-09-14): 72.2% -- i.e. the
# within-config divergence is as large as the cross-config one.
#
# Usage:
#   run_head_control.sh                        # use committed HEAD
#   run_head_control.sh /path/to/infer_state.py  # use an explicit older copy
#
# See docs/DeepSeek_增量DCI地址路径优化_结果.md section 5.3.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/yx/IceCache
OUT="$ROOT/experiment/logs/addr_opt"
HEADSRC=/tmp/icecache_head_src
mkdir -p "$OUT"

rm -rf "$HEADSRC"
mkdir -p "$HEADSRC"
cp -r "$ROOT/IceCache/source/icecache" "$HEADSRC/"

if [ $# -ge 1 ] && [ -f "$1" ]; then
  cp "$1" "$HEADSRC/icecache/infer_state.py"
  echo "using supplied copy: $1"
else
  git -C "$ROOT" show HEAD:IceCache/source/icecache/infer_state.py \
    > "$HEADSRC/icecache/infer_state.py"
  echo "using committed HEAD:IceCache/source/icecache/infer_state.py"
  echo "  (pass an explicit path as \$1 to use the pre-patch working tree copy)"
fi

python3 - "$HEADSRC/icecache/infer_state.py" \
          "$ROOT/IceCache/source/icecache/infer_state.py" <<'PY'
import hashlib
import sys
for label, p in (("control ", sys.argv[1]), ("worktree", sys.argv[2])):
    print("%s infer_state.py sha256 = %s"
          % (label, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]))
PY

run_head() {
  local name="$1" dump="$2"
  ( cd "$ROOT/IceCache/benchmark"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 \
    ICECACHE_DCI_PARALLEL_LEVEL=2 ICECACHE_FP16_RECALL=1 \
    ICECACHE_BATCH_LAYER_RECALL=0 ICECACHE_CROSS_TOKEN_DCI=0 \
    ICECACHE_TRACE_DCI_CHURN=0 ICECACHE_TRACE_DCI_ADAPTIVE=0 \
    ICECACHE_PROMOTION_FAST_START_LAYER=-1 \
    ICECACHE_DIAG=1 ICECACHE_DIAG_MAX_RECORDS=60000 \
    ICECACHE_DIAG_DUMP="$dump" \
    PYTHONPATH="$HEADSRC" \
    /home/yx/miniconda3/envs/icecache/bin/python longbench_pred.py \
      --icecache --model llama-3.1 \
      --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
      --page-size 16 --page-budgets 64 --page-topks 0 \
      --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 \
      --n_prefetch_layers 0 --ratio_1 0.01 --ratio_2 0.2 \
      --profile-dci --profile-warmup-tokens 2 \
      --max-samples 3 --name "$name" --datasets qasper )
}

echo "[$(date +%H:%M:%S)] head_diag1"
run_head head_diag1 /tmp/diag_head1.npz > "$OUT/head_diag1.log" 2>&1
echo "[$(date +%H:%M:%S)] head_diag2"
run_head head_diag2 /tmp/diag_head2.npz > "$OUT/head_diag2.log" 2>&1

echo "[$(date +%H:%M:%S)] comparing the two pre-patch runs"
/home/yx/miniconda3/envs/icecache/bin/python \
  "$SCRIPT_DIR/../probe/find_divergence.py" \
  /tmp/diag_head1.npz /tmp/diag_head2.npz
echo "[$(date +%H:%M:%S)] HEADDONE"
