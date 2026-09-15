#!/usr/bin/env bash
# Follow-up passes after the A/B matrix:
#   1. full-run DCI-state equivalence (leaf ids + addresses for every recall)
#   2. raw per-call records for the T_insert characterisation
# See docs/DeepSeek_增量DCI地址路径优化_结果.md sections 5.3 and 6.1.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$SCRIPT_DIR/run_addr_opt_profile.sh"
OUT=/home/yx/IceCache/experiment/logs/addr_opt
mkdir -p "$OUT"

step() { echo "[$(date +%H:%M:%S)] $*"; }

step "diag_full_base"
ICECACHE_DIAG=1 ICECACHE_DIAG_MAX_RECORDS=60000 \
  ICECACHE_DIAG_DUMP=/tmp/diag_full_base.npz ICECACHE_VEC_ADDR=0 \
  bash "$PROF" diag_full_base 3 > "$OUT/diag_full_base.log" 2>&1

step "diag_full_vec"
ICECACHE_DIAG=1 ICECACHE_DIAG_MAX_RECORDS=60000 \
  ICECACHE_DIAG_DUMP=/tmp/diag_full_vec.npz ICECACHE_VEC_ADDR=1 \
  bash "$PROF" diag_full_vec 3 > "$OUT/diag_full_vec.log" 2>&1

step "tinsert_base"
ICECACHE_VEC_ADDR=0 ICECACHE_PROFILE_CALL_DUMP=/tmp/call_records_base.json \
  bash "$PROF" tinsert_base 3 > "$OUT/tinsert_base.log" 2>&1

step "tinsert_vec"
ICECACHE_VEC_ADDR=1 ICECACHE_PROFILE_CALL_DUMP=/tmp/call_records_vec.json \
  bash "$PROF" tinsert_vec 3 > "$OUT/tinsert_vec.log" 2>&1

step FOLLOWUPDONE
