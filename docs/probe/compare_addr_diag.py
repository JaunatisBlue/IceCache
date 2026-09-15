"""Compare two `ICECACHE_DIAG` dumps: DCI-selected leaf ids and CPU addresses.

Usage:
    python compare_addr_diag.py base.npz new.npz
"""

import sys

import numpy as np


def main(a_path, b_path):
    a = np.load(a_path)
    b = np.load(b_path)
    print("A=%s  B=%s" % (a_path, b_path))
    ok = True
    for key in ("layer", "head", "offsets"):
        same = np.array_equal(a[key], b[key])
        ok &= same
        print("  %-10s identical=%s  n=%d" % (key, same, a[key].size))
    for key in ("flat_leaf",):
        if a[key].size != b[key].size:
            print("  %-10s SIZE DIFFERS a=%d b=%d"
                  % (key, a[key].size, b[key].size))
            ok = False
            continue
        same = np.array_equal(a[key], b[key])
        ok &= same
        n_bad = int(np.count_nonzero(a[key] != b[key]))
        print("  %-10s identical=%s  n=%d  differing=%d"
              % (key, same, a[key].size, n_bad))

    # CPU addresses are absolute virtual addresses: the pinned pool is mapped
    # at a different base in every process, so the *relative* structure is the
    # only meaningful invariant.  `delta[k] = addr_new[k] - addr_base[k]` must
    # be a single constant across all leaves, and every per-record relative
    # offset must match exactly.
    a_addr = a["flat_addr"].astype(np.int64)
    b_addr = b["flat_addr"].astype(np.int64)
    if a_addr.size != b_addr.size:
        print("  flat_addr SIZE DIFFERS a=%d b=%d" % (a_addr.size, b_addr.size))
        return 1
    delta = b_addr - a_addr
    uniq = np.unique(delta)
    print("  flat_addr n=%d  span A=%d B=%d  shift_unique=%d"
          % (a_addr.size, int(a_addr.max() - a_addr.min()),
             int(b_addr.max() - b_addr.min()), uniq.size))
    if uniq.size == 1:
        print("  flat_addr shifted by a single constant %d "
              "(= pool base pointer difference)" % int(uniq[0]))
    rel_ok = np.array_equal(a_addr - a_addr.min(), b_addr - b_addr.min())
    print("  flat_addr relative-to-min identical=%s" % rel_ok)
    ok &= rel_ok and uniq.size == 1

    # Per-record (layer, head) relative offsets, immune to the base pointer.
    offs = a["offsets"].astype(np.int64)
    per_rec_ok = True
    n_rec = len(offs) - 1
    for r in range(n_rec):
        s, e = offs[r], offs[r + 1]
        if not np.array_equal(a_addr[s:e] - a_addr[s],
                              b_addr[s:e] - b_addr[s]):
            per_rec_ok = False
            print("    record %d (layer=%d head=%d) relative offsets differ"
                  % (r, a["layer"][r], a["head"][r]))
            break
    print("  per-record relative offsets identical=%s (%d records)"
          % (per_rec_ok, n_rec))
    ok &= per_rec_ok

    print("RESULT: %s" % ("IDENTICAL" if ok else "DIFFERS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
