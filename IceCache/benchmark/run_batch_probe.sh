#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
probe_python="${ICECACHE_PYTHON:-/home/yx/miniconda3/envs/icecache/bin/python}"
result_dir="${BATCH_RESULT_DIR:-$repo_root/experiment/batch_decode}"
export PYTHONPATH="$repo_root/IceCache/source${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${BATCH_QUERY_THREADS:-16}"
export OMP_DYNAMIC=FALSE
export TOKENIZERS_PARALLELISM=false
mkdir -p "$result_dir"

# Each arm gets fresh independent trees.  These two runs are exploratory;
# the frozen-index replay inside each run provides the paired CPU comparison.
for query_backend in serial native; do
    "$probe_python" -u "$script_dir/batch_decode_probe.py" \
        --prompt-tokens 1024 --steps 48 \
        --query-backend "$query_backend" --query-threads "$OMP_NUM_THREADS" \
        --compare-native-raw --cpu-replay-repeats 20 \
        --output "$result_dir/${query_backend}_1024_48.json" \
        > "$result_dir/${query_backend}_1024_48.log" 2>&1
done
