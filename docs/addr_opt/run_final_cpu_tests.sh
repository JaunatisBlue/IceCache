#!/usr/bin/env bash
# Final CPU-side validation for ICECACHE_VEC_ADDR, four phases.
#
#   phase 1  address equivalence under vec=1 (must be mismatch=0, and the
#            logical-page-id shortcut must NOT match -- otherwise the test is a
#            false positive)
#   phase 2  stage-level A/B over SEVERAL 36k passkey prompts (4 tests/run)
#   phase 3  quality gate on the FULL Qasper set (200 samples, both arms)
#   phase 4  small interleaved end-to-end TPOT test: 1 discarded warm-up run,
#            then 3 reps per arm interleaved
#
# Usage: run_final_cpu_tests.sh [tag]   (default tag = final)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/yx/IceCache
OUT="$ROOT/experiment/logs/addr_opt"
PROF="$SCRIPT_DIR/run_addr_opt_profile.sh"
PY=/home/yx/miniconda3/envs/icecache/bin/python
TAG="${1:-final}"
mkdir -p "$OUT"

step() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================ phase 1 =====================================
step "PHASE 1/4 address equivalence (vec=1 + EQUIV_CHECK)"
ICECACHE_ADDR_EQUIV_CHECK=1 ICECACHE_VEC_ADDR=1 \
  bash "$PROF" "${TAG}_addr_equiv" 3 > "$OUT/${TAG}_addr_equiv.log" 2>&1
$PY "$SCRIPT_DIR/../probe/show_addr_equiv.py" "$OUT/${TAG}_addr_equiv.log"

# ============================ phase 2 =====================================
step "PHASE 2/4 stage-level A/B on 4x 36k passkey per run, 2 reps interleaved"
NUM_TESTS=4 REPS=2 bash "$SCRIPT_DIR/run_passkey36k_ab.sh" "${TAG}36km" \
  > "$OUT/${TAG}_phase2_driver.log" 2>&1
step "PHASE 2 summary"
$PY "$SCRIPT_DIR/../probe/summarize_reps.py" \
  "A(baseline)=$OUT/${TAG}36km_A1.log,$OUT/${TAG}36km_A2.log" \
  "B(vec)=$OUT/${TAG}36km_B1.log,$OUT/${TAG}36km_B2.log"

# ============================ phase 3 =====================================
step "PHASE 3/4 full Qasper quality gate (200 samples per arm)"
for arm in A B; do
  if [ "$arm" = "A" ]; then vec=0; else vec=1; fi
  step "  ${TAG}_q200_${arm} (vec=$vec)"
  ICECACHE_VEC_ADDR=$vec bash "$PROF" "${TAG}_q200_${arm}" 200 \
    > "$OUT/${TAG}_q200_${arm}.log" 2>&1
done
step "PHASE 3 F1 (full 200-sample Qasper)"
( cd "$ROOT/IceCache/benchmark"
  for n in "${TAG}_q200_A" "${TAG}_q200_B"; do
    PYTHONPATH="$ROOT/IceCache/source" $PY longbench_eval.py \
      --model llama-3.1 --name "$n" > /dev/null 2>&1
    printf '  %-16s ' "$n"; tr -d '\n ' < "pred/llama-3.1/$n/result.json"; echo
  done )

# ============================ phase 4 =====================================
step "PHASE 4/4 interleaved end-to-end TPOT (discard 1 warm-up, 3 reps/arm)"
step "  warm-up run (vec=0, discarded)"
ICECACHE_VEC_ADDR=0 bash "$PROF" "${TAG}_warm" 20 \
  > "$OUT/${TAG}_warm.log" 2>&1
for rep in 1 2 3; do
  step "  rep${rep} A (vec=0)"
  ICECACHE_VEC_ADDR=0 bash "$PROF" "${TAG}_i_A${rep}" 20 \
    > "$OUT/${TAG}_i_A${rep}.log" 2>&1
  step "  rep${rep} B (vec=1)"
  ICECACHE_VEC_ADDR=1 bash "$PROF" "${TAG}_i_B${rep}" 20 \
    > "$OUT/${TAG}_i_B${rep}.log" 2>&1
done
step "PHASE 4 interleaved means (warm-up excluded)"
$PY "$SCRIPT_DIR/../probe/summarize_reps.py" \
  "A(baseline)=$OUT/${TAG}_i_A1.log,$OUT/${TAG}_i_A2.log,$OUT/${TAG}_i_A3.log" \
  "B(vec)=$OUT/${TAG}_i_B1.log,$OUT/${TAG}_i_B2.log,$OUT/${TAG}_i_B3.log"
step "PHASE 4 per-sample paired (same 20 documents)"
for rep in 1 2 3; do
  $PY "$SCRIPT_DIR/../probe/analyze_ab_paired.py" \
    "A${rep}=$OUT/${TAG}_i_A${rep}.log" "B${rep}=$OUT/${TAG}_i_B${rep}.log" \
    | sed -n '5,14p'
done

step "${TAG} ALLDONE"
