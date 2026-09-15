#!/usr/bin/env bash
# A/B matrix for the decode-side vectorised CPU address preparation.
#
#   phase 1  correctness: diag dump of selected leaf ids + their CPU addresses
#   phase 2  timing: 3-sample Qasper subset, interleaved, two repetitions
#
# NOTE: the "hand the NumPy array straight to the M-DCI binding" variant
# (`ICECACHE_VEC_ADDR_NP=1`) was falsified -- it segfaults or hangs.  It has
# been removed from this matrix and from the code; see
# docs/DeepSeek_增量DCI地址路径优化_结果.md section 4.2.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$SCRIPT_DIR/run_addr_opt_profile.sh"
OUT=/home/yx/IceCache/experiment/logs/addr_opt
mkdir -p "$OUT"

step() { echo "[$(date +%H:%M:%S)] $*"; }

# --- correctness: diag dump of selected leaf ids + their CPU addresses -----
step "corr_base (vec=0)"
ICECACHE_DIAG=1 ICECACHE_DIAG_DUMP=/tmp/diag_base.npz ICECACHE_VEC_ADDR=0 \
  bash "$PROF" corr_base 2 > "$OUT/corr_base.log" 2>&1
step "corr_vec (vec=1)"
ICECACHE_DIAG=1 ICECACHE_DIAG_DUMP=/tmp/diag_vec.npz ICECACHE_VEC_ADDR=1 \
  bash "$PROF" corr_vec 2 > "$OUT/corr_vec.log" 2>&1

# --- timing: 3-sample Qasper subset, interleaved, two repetitions ---------
for rep in 1 2; do
  step "ab_A${rep} (baseline)"
  ICECACHE_VEC_ADDR=0 bash "$PROF" ab_A${rep} 3 \
    > "$OUT/ab_A${rep}.log" 2>&1
  step "ab_B${rep} (vec)"
  ICECACHE_VEC_ADDR=1 bash "$PROF" ab_B${rep} 3 \
    > "$OUT/ab_B${rep}.log" 2>&1
done
step ALLDONE
