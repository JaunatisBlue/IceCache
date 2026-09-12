#!/usr/bin/env bash
# Attribute the CPU time of the decode path with cProfile.
set -euo pipefail

ROOT="/home/yx/IceCache"
cd "$ROOT/IceCache/benchmark"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32
export ICECACHE_DCI_PARALLEL_LEVEL=2
export ICECACHE_FP16_RECALL=1
export ICECACHE_BATCH_LAYER_RECALL=0
export ICECACHE_CROSS_TOKEN_DCI=0
export ICECACHE_TRACE_DCI_CHURN=0
export ICECACHE_TRACE_DCI_ADAPTIVE=0
export ICECACHE_PROMOTION_FAST_START_LAYER=-1
export PYTHONPATH="$ROOT/IceCache/source"

rm -f /tmp/cprof.out
/home/yx/miniconda3/envs/icecache/bin/python -m cProfile -o /tmp/cprof.out \
  longbench_pred.py \
  --icecache --model llama-3.1 \
  --model-path /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --page-size 16 --page-budgets 64 --page-topks 0 \
  --n-sink-pages 2 --n-win-pages 2 --n_reuse_layers 3 --n_prefetch_layers 0 \
  --ratio_1 0.01 --ratio_2 0.2 \
  --profile-dci --profile-warmup-tokens 2 \
  --max-samples 3 --name cprof_probe --datasets qasper > /tmp/cprof_run.log 2>&1 || true

echo "=== TOP 35 by tottime (self time) ==="
/home/yx/miniconda3/envs/icecache/bin/python - <<'PY'
import pstats
st = pstats.Stats('/tmp/cprof.out')
st.sort_stats('tottime').print_stats(35)
PY

echo
echo "=== TOP 25 by cumtime, filtered to icecache ==="
/home/yx/miniconda3/envs/icecache/bin/python - <<'PY'
import pstats, io
st = pstats.Stats('/tmp/cprof.out')
buf = io.StringIO()
st.sort_stats('cumtime').stream = buf
st.print_stats(120)
lines = buf.getvalue().splitlines()
keep = [l for l in lines if 'icecache' in l or 'ncalls' in l or 'function calls' in l]
print("\n".join(keep[:40]))
PY
