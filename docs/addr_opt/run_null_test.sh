#!/usr/bin/env bash
# Null test: is the sample-2 token flip caused by the address change, or by any
# change in the wall-clock timing of the decode path?
#
# `ICECACHE_ADDR_EQUIV_CHECK=1` keeps the ORIGINAL per-leaf data_ptr() path in
# production and only *recomputes* the vectorised formula for comparison (it
# raises on mismatch).  It therefore changes timing while changing no value at
# all -- exactly the null perturbation we need.
#
# STATUS (2026-09-14): NOT RUN.  `run_control.sh` turned out to be a stronger
# null (identical config twice), and it showed the baseline itself diverges as
# much as baseline-vs-vectorised.  Kept as a cheaper alternative if a null with
# a purely local perturbation is ever wanted.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$SCRIPT_DIR/run_addr_opt_profile.sh"
OUT=/home/yx/IceCache/experiment/logs/addr_opt
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] null_equiv1 (vec=0, equiv check on)"
ICECACHE_VEC_ADDR=0 ICECACHE_ADDR_EQUIV_CHECK=1 \
  bash "$PROF" null_equiv1 3 > "$OUT/null_equiv1.log" 2>&1

echo "[$(date +%H:%M:%S)] null_equiv2 (vec=0, equiv check on, repeat)"
ICECACHE_VEC_ADDR=0 ICECACHE_ADDR_EQUIV_CHECK=1 \
  bash "$PROF" null_equiv2 3 > "$OUT/null_equiv2.log" 2>&1

echo "[$(date +%H:%M:%S)] NULLDONE"
