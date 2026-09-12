#!/usr/bin/env python3
"""ICECACHE_NO_RECALL_SYNC: diagnostic-only removal of the host-side recall sync.

Purpose: size the CEILING of any stream-reordering work.  recall() issues the
H2D on c2g_stream and _estimate_select_recall_impl then calls
c2g_stream.synchronize(), which the profiler shows as 46% of the decode-side
work.  That number is host-side idle time, not necessarily critical-path time:
if the GPU is still busy with the previous layer, the wait is already
overlapped and worth nothing.

Setting ICECACHE_NO_RECALL_SYNC=1 skips the sync.  Results are NOT valid
(transit-buffer reuse is racy) and F1 must be ignored; only TPOT is
informative.  This exists to decide whether the double-buffer / event-ordering
rework is worth building.

Usage:
    python3 apply_no_sync_patch.py /path/to/infer_state.py
"""
import sys
import os
import shutil

OLD = (
    "                c2g_stream.synchronize()\n"
    "                if recall_wait_start is not None:\n"
)
NEW = (
    "                if not self.no_recall_sync:\n"
    "                    c2g_stream.synchronize()\n"
    "                if recall_wait_start is not None:\n"
)

INIT_OLD = "        self.diag_saved = False\n"
INIT_NEW = (
    "        # [ICECACHE-NOSYNC] diagnostic-only; see apply_no_sync_patch.py\n"
    "        self.no_recall_sync = bool(int(os.environ.get(\"ICECACHE_NO_RECALL_SYNC\", \"0\")))\n"
    "        self.diag_saved = False\n"
)

PATCHES = [(OLD, NEW, "P1-skip-sync"), (INIT_OLD, INIT_NEW, "P2-init")]


def main():
    if len(sys.argv) != 2:
        print("usage: apply_no_sync_patch.py /path/to/infer_state.py")
        return 2
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "ICECACHE-NOSYNC" in src:
        print("ALREADY_PATCHED")
        return 3
    for old, new, tag in PATCHES:
        n = src.count(old)
        if n != 1:
            print("ANCHOR_FAIL %s: %d occurrences" % (tag, n))
            return 4
        src = src.replace(old, new, 1)
        print("OK %s" % tag)
    backup = path + ".bak_prenosync"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print("backup -> %s" % backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("PATCH_APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
