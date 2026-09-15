#!/usr/bin/env bash
# Control experiment: is the DCI selection itself run-to-run reproducible?
# Runs a second full-run diag dump for each config so we can separate
# within-config variance from the baseline-vs-vectorised difference.
# See docs/DeepSeek_增量DCI地址路径优化_结果.md section 5.3.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$SCRIPT_DIR/run_addr_opt_profile.sh"
OUT=/home/yx/IceCache/experiment/logs/addr_opt
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] diag_full_base2 (vec=0, repeat)"
ICECACHE_DIAG=1 ICECACHE_DIAG_MAX_RECORDS=60000 \
  ICECACHE_DIAG_DUMP=/tmp/diag_full_base2.npz ICECACHE_VEC_ADDR=0 \
  bash "$PROF" diag_full_base2 3 > "$OUT/diag_full_base2.log" 2>&1

echo "[$(date +%H:%M:%S)] diag_full_vec2 (vec=1, repeat)"
ICECACHE_DIAG=1 ICECACHE_DIAG_MAX_RECORDS=60000 \
  ICECACHE_DIAG_DUMP=/tmp/diag_full_vec2.npz ICECACHE_VEC_ADDR=1 \
  bash "$PROF" diag_full_vec2 3 > "$OUT/diag_full_vec2.log" 2>&1

echo "[$(date +%H:%M:%S)] CONTROLDONE"
