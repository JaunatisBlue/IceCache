#!/usr/bin/env python3
"""Offline analysis of the ICECACHE_DIAG dump (timing split + address mergeability).

Usage:
    python analyze_diag.py /path/to/icecache_diag.npz [--tpot-ms X]

Reports
  (A) five-segment recall timing split
  (B) per (layer, head) leaf-run structure: n_sel / n_runs / merge_ratio
  (C) cross-head leaf-id collisions inside one recall step
  (D) a four-way verdict on whether the layout direction is worth pursuing
"""
import sys
import argparse
import numpy as np


def runs_of_consecutive(sorted_ids):
    """Count maximal runs of consecutive integers in a sorted array."""
    if len(sorted_ids) == 0:
        return 0
    gaps = np.diff(sorted_ids)
    return int(1 + np.count_nonzero(gaps != 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--tpot-ms", type=float, default=None,
                    help="TPOT in ms, to convert absolute timings into %% of TPOT")
    ap.add_argument("--trace-seconds", type=float, default=None,
                    help="measured decode seconds, to convert to ms/token "
                         "(optional; otherwise report raw totals)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    layer = d["layer"]
    head = d["head"]
    flat_leaf = d["flat_leaf"]
    flat_addr = d["flat_addr"]
    offs = d["offsets"]

    addr_prep_s = float(d["addr_prep_seconds"])
    copy_buf_s = float(d["copy_buffer_seconds"])
    h2d_ms = float(d["h2d_ms"])
    cast_ms = float(d["cast_ms"])
    h2d_n = int(d["h2d_count"])
    cast_n = int(d["cast_count"])

    nrec = len(offs) - 1
    print("=" * 72)
    print("ICECACHE_DIAG ANALYSIS   records=%d  selected_pages=%d"
          % (nrec, len(flat_leaf)))
    print("=" * 72)

    # ---------------------------------------------------------- (A) timing
    print("\n[A] Recall timing split")
    print("  addr_prep (CPU address list)   : %10.4f s  (n=%d recalls)"
          % (addr_prep_s, nrec))
    print("  copy_to_buffer (CPU gather)    : %10.4f s" % copy_buf_s)
    print("  H2D  (CUDA event, stream)      : %10.4f ms (n=%d)" % (h2d_ms, h2d_n))
    print("  cast (CUDA event, stream)      : %10.4f ms (n=%d)" % (cast_ms, cast_n))
    if h2d_n:
        print("  -> per-H2D  : %.4f ms" % (h2d_ms / h2d_n))
    if cast_n:
        print("  -> per-cast : %.4f ms" % (cast_ms / cast_n))
    cpu_s = addr_prep_s + copy_buf_s
    gpu_ms = h2d_ms + cast_ms
    total_ms = cpu_s * 1000.0 + gpu_ms
    if total_ms > 0:
        print("  -> CPU(addr+gather) share      : %.1f%%" % (100.0 * cpu_s * 1000 / total_ms))
        print("  -> GPU(H2D+cast) share         : %.1f%%" % (100.0 * gpu_ms / total_ms))
    if args.tpot_ms:
        print("  -> as %% of TPOT(%.1f ms/tok):" % args.tpot_ms)
        if nrec:
            print("     addr_prep %.2f%%  copy_buf %.2f%%  H2D %.2f%%  cast %.2f%%"
                  % (100 * (addr_prep_s * 1000 / max(nrec, 1)) / args.tpot_ms,
                     100 * (copy_buf_s * 1000 / max(nrec, 1)) / args.tpot_ms,
                     100 * (h2d_ms / max(h2d_n, 1)) / args.tpot_ms,
                     100 * (cast_ms / max(cast_n, 1)) / args.tpot_ms))

    # ------------------------------------------- (B) per (layer, head) runs
    per_key = {}
    for k in range(nrec):
        s, e = int(offs[k]), int(offs[k + 1])
        leaves = np.sort(flat_leaf[s:e])
        addrs = flat_addr[s:e]
        key = (int(layer[k]), int(head[k]))
        if len(leaves) == 0:
            continue
        nr = runs_of_consecutive(leaves)
        # observed address stride between consecutive leaf ids
        order = np.argsort(flat_leaf[s:e])
        ls = flat_leaf[s:e][order]
        ad = addrs[order]
        dif = np.diff(ad.astype(np.int64))
        dl = np.diff(ls)
        unit = dif[dl == 1]
        stride = int(np.median(unit)) if len(unit) else None
        per_key.setdefault(key, []).append((len(leaves), nr, stride))

    sel_tot = sum(v[0] for lst in per_key.values() for v in lst)
    run_tot = sum(v[1] for lst in per_key.values() for v in lst)
    print("\n[B] Per-(layer,head) leaf-run structure  (n_groups=%d)" % len(per_key))
    print("  total selected (head,leaf) pages : %d" % sel_tot)
    print("  total merged runs if DFS-strided : %d" % run_tot)
    if sel_tot:
        print("  overall merge_ratio (1-run/sel)  : %.3f" % (1 - run_tot / sel_tot))
    strides = [v[2] for lst in per_key.values() for v in lst if v[2] is not None]
    if strides:
        uniq, cnt = np.unique(np.asarray(strides, dtype=np.int64), return_counts=True)
        print("  observed addr stride (leaf+1)    : %s"
              % ", ".join("%d x%d" % (u, c) for u, c in zip(uniq[:4], cnt[:4])))

    # ------------------------------------- (C) cross-head collisions
    # Group records into recall steps using the head-reset heuristic.
    groups = []
    cur = []
    prev_head = -1
    for k in range(nrec):
        h = int(head[k])
        if h <= prev_head and cur:
            groups.append(cur)
            cur = []
        cur.append(k)
        prev_head = h
    if cur:
        groups.append(cur)

    collisions = 0
    multi = 0
    for g in groups:
        if len(g) < 2:
            continue
        multi += 1
        counts = {}
        for k in g:
            s, e = int(offs[k]), int(offs[k + 1])
            for lf in flat_leaf[s:e]:
                counts[int(lf)] = counts.get(int(lf), 0) + 1
        collisions += sum(c - 1 for c in counts.values() if c > 1)
    print("\n[C] Cross-head leaf collisions inside one recall step")
    print("  recall steps with >=2 heads      : %d" % multi)
    print("  duplicate (leaf shared by heads) : %d" % collisions)
    if sel_tot:
        print("  cross-head merge_ratio           : %.3f" % (collisions / sel_tot))

    # --------------------------------------------------------- (D) verdict
    print("\n[D] Verdict inputs")
    if total_ms > 0:
        print("  CPU-gather share of measured recall cost : %.1f%%" % (100.0 * cpu_s * 1000 / total_ms))
        print("  GPU H2D share of measured recall cost    : %.1f%%" % (100.0 * h2d_ms / total_ms))
    mr = (1 - run_tot / sel_tot) if sel_tot else 0.0
    print("  within-head merge_ratio                  : %.3f" % mr)
    print("  cross-head merge_ratio                   : %.3f"
          % (collisions / sel_tot if sel_tot else 0.0))
    print("""
  Decision rules (gather-dominant = CPU share > GPU H2D share):
    * gather-dominant AND (merge_ratio high)        -> keep going: subtree-aware gather
    * gather-dominant AND clustered-but-not-merge   -> study head-major layout
    * gather-dominant AND merge_ratio ~ 0           -> drop the layout mainline
    * H2D-dominant                                  -> pivot to dynamic k reduction
""")


if __name__ == "__main__":
    main()
