#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/M-DCI" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mdci_root="$(cd "$1" && pwd)"

git -C "$mdci_root" apply --check "$repo_root/patches/mdci-fp16-recall.patch"
git -C "$mdci_root" apply "$repo_root/patches/mdci-fp16-recall.patch"
python -m pip install --no-build-isolation --no-deps --force-reinstall "$mdci_root"

echo "Installed patched M-DCI. Enable with ICECACHE_FP16_RECALL=1."
