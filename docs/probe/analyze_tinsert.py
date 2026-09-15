"""Characterise T_insert from the raw per-call index-update records.

Reads the JSON dumped by `ICECACHE_PROFILE_CALL_DUMP` and answers:

  * does the native incremental insert scale with the number of points already
    in the tree (`prev_num_points`)?
  * does it scale with the number of inserted tokens / new leaves?
  * is the address-preparation path tree-size dependent or leaf-count dependent?
  * how do anchor and reuse layers differ?
  * is the latency distribution heavy tailed (leaf split / relocation)?

Usage: python analyze_tinsert.py /tmp/call_records_base.json [more.json ...]
"""

import json
import statistics
import sys
from collections import defaultdict


def bucket_name(p):
    if p < 0:
        return "unknown"
    step = 4096
    lo = (p // step) * step
    return "%5d-%5d" % (lo, lo + step - 1)


def fmt(xs):
    if not xs:
        return "     -"
    return "%6.3f" % (sum(xs) / len(xs))


def main(paths):
    for path in paths:
        with open(path) as fh:
            recs = json.load(fh)
        print("=== %s  (%d records) ===" % (path, len(recs)))
        if not recs:
            continue
        groups = defaultdict(list)
        for r in recs:
            groups[(r.get("anchor"), bucket_name(r.get("prev_num_points", -1)))
                   ].append(r)

        print("%-6s %-13s %5s %8s %9s %10s %11s %11s %10s %10s"
              % ("role", "prev_pts", "n", "newleaf", "ins_ms", "ins_p95",
                 "native_addr", "reuse_upd", "addr_prep", "leaf_loop"))
        for (anchor, bname) in sorted(groups, key=lambda k: (not k[0], k[1])):
            rs = groups[(anchor, bname)]
            ins = sorted(r.get("native_insert_ms", 0.0) for r in rs)
            p95 = ins[int(0.95 * (len(ins) - 1))] if ins else 0.0
            leaves = [r.get("new_leaves_total", 0) for r in rs]
            print("%-6s %-13s %5d %8.1f %9.3f %10.3f %11.4f %11.4f %10.3f %10.3f"
                  % ("anchor" if anchor else "reuse", bname, len(rs),
                     sum(leaves) / len(rs),
                     sum(ins) / len(ins), p95,
                     sum(r.get("native_addr_update_ms", 0.0)
                         for r in rs) / len(rs),
                     sum(r.get("reuse_update_ms", 0.0) for r in rs) / len(rs),
                     sum(r.get("addr_prep_ms", 0.0) for r in rs) / len(rs),
                     sum(r.get("addr_leaf_ms", 0.0) for r in rs) / len(rs)))

        # Correlation between tree size and insert cost (anchor layers only).
        pts = [(r["prev_num_points"], r.get("native_insert_ms", 0.0),
                r.get("new_leaves_total", 0), r.get("insert_tokens", 0))
               for r in recs
               if r.get("anchor") and r.get("prev_num_points", -1) > 0]
        if len(pts) > 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = (sum((x - mx) ** 2 for x in xs)
                   * sum((y - my) ** 2 for y in ys)) ** 0.5
            r_ = num / den if den else float("nan")
            print("anchor-only corr(prev_num_points, native_insert_ms) = %.3f "
                  "over %d calls" % (r_, len(pts)))
            # Exponent of the scaling law between the lowest and highest tree
            # sizes actually visited (anchor calls only, same 16-token batch).
            import math

            lo = min(pts, key=lambda p: p[0])
            hi = max(pts, key=lambda p: p[0])
            print("  lowest  tree: prev_pts=%d  insert=%.3f ms  leaves=%.1f"
                  % (lo[0], lo[1], lo[2]))
            print("  highest tree: prev_pts=%d  insert=%.3f ms  leaves=%.1f"
                  % (hi[0], hi[1], hi[2]))
            if lo[1] > 0 and lo[0] > 0:
                print("  ratio prev_pts x%.2f  ->  ratio insert x%.2f  "
                      "(empirical exponent %.2f)"
                      % (hi[0] / lo[0], hi[1] / lo[1],
                         math.log(hi[1] / lo[1]) / math.log(hi[0] / lo[0])))

        # Address path vs tree size (must be flat -> leaf-count driven).
        pts2 = [(r["prev_num_points"], r.get("addr_leaf_ms", 0.0),
                 r.get("new_leaves_total", 0))
                for r in recs if r.get("prev_num_points", -1) > 0]
        if pts2:
            xs = [p[0] for p in pts2]
            ys = [p[1] for p in pts2]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = (sum((x - mx) ** 2 for x in xs)
                   * sum((y - my) ** 2 for y in ys)) ** 0.5
            print("corr(prev_num_points, addr_leaf_ms) = %.3f "
                  "(expect ~0: address cost tracks new leaves, not tree size)"
                  % (num / den if den else float("nan")))
            per_leaf = [p[1] / p[2] for p in pts2 if p[2]]
            if per_leaf:
                print("addr_leaf ms per new leaf: median=%.4f mean=%.4f "
                      "(n=%d)" % (statistics.median(per_leaf),
                                  statistics.mean(per_leaf), len(per_leaf)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
