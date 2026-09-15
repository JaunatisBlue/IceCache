#!/usr/bin/env bash
# Third control: two more baseline (vec=0) runs with per-call records, so the
# `new_leaves_total` / `native_insert_ms` sequences can be compared across
# three *identical* runs as well as against the vectorised run.
# See docs/DeepSeek_增量DCI地址路径优化_结果.md section 5.3.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$SCRIPT_DIR/run_addr_opt_profile.sh"
OUT=/home/yx/IceCache/experiment/logs/addr_opt
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] tinsert_base2"
ICECACHE_VEC_ADDR=0 ICECACHE_PROFILE_CALL_DUMP=/tmp/call_records_base2.json \
  bash "$PROF" tinsert_base2 3 > "$OUT/tinsert_base2.log" 2>&1

echo "[$(date +%H:%M:%S)] tinsert_base3"
ICECACHE_VEC_ADDR=0 ICECACHE_PROFILE_CALL_DUMP=/tmp/call_records_base3.json \
  bash "$PROF" tinsert_base3 3 > "$OUT/tinsert_base3.log" 2>&1

echo "[$(date +%H:%M:%S)] LEAFCONTROLDONE"
