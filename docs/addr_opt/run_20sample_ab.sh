#!/usr/bin/env bash
# 20-sample Qasper end-to-end A/B for ICECACHE_VEC_ADDR.
#
# One repetition per arm (as requested).  Arms are interleaved A -> B so a slow
# drift cannot masquerade as an arm effect; one repetition cannot resolve a
# sub-1% TPOT delta (see the report), so the acceptance criteria are
# F1-within-noise and "no regression / no failure", not a TPOT win.
#
# Analysis afterwards:
#   python docs/probe/report_ab_tpot.py  A=.../q20_A.log B=.../q20_B.log
#   (cd IceCache/benchmark && python longbench_eval.py --model llama-3.1 --name q20_A)
#
# See docs/DeepSeek_增量DCI地址路径优化_结果.md sections 5 and 10.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$SCRIPT_DIR/run_addr_opt_profile.sh"
OUT=/home/yx/IceCache/experiment/logs/addr_opt
mkdir -p "$OUT"

SAMPLES=20
step() { echo "[$(date +%H:%M:%S)] $*"; }

# Per-step latency samples so DCI_PROFILE carries TPOT mean/std/p50/p95.
export ICECACHE_PROFILE_STEP_SAMPLES=1

step "q20_A (baseline, vec=0)"
ICECACHE_VEC_ADDR=0 bash "$PROF" q20_A "$SAMPLES" > "$OUT/q20_A.log" 2>&1

step "q20_B (vectorised, vec=1)"
ICECACHE_VEC_ADDR=1 bash "$PROF" q20_B "$SAMPLES" > "$OUT/q20_B.log" 2>&1

step "comparing"
/home/yx/miniconda3/envs/icecache/bin/python \
  "$SCRIPT_DIR/../probe/report_ab_tpot.py" \
  "A(baseline)=$OUT/q20_A.log" "B(vectorised)=$OUT/q20_B.log"

step "scoring F1"
( cd /home/yx/IceCache/IceCache/benchmark
  for n in q20_A q20_B; do
    PYTHONPATH=/home/yx/IceCache/IceCache/source \
      /home/yx/miniconda3/envs/icecache/bin/python longbench_eval.py \
      --model llama-3.1 --name "$n" > /dev/null 2>&1
    printf '%-8s ' "$n"; tr -d '\n ' < "pred/llama-3.1/$n/result.json"; echo
  done )

step Q20ABDONE
