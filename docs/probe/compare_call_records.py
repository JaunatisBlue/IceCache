"""Compare the per-call index-update record sequences across several runs.

Used to separate "the vectorised address path changed something" from plain
run-to-run variance in the native DCI insertion (new leaf counts, insert cost).

Usage: python compare_call_records.py label=path.json ...
"""

import json
import statistics
import sys


def main(specs):
    runs = []
    for spec in specs:
        label, path = spec.split("=", 1)
        with open(path) as fh:
            runs.append((label, json.load(fh)))

    print("records per run: " + ", ".join(
        "%s=%d" % (l, len(d)) for l, d in runs))

    # The call order is deterministic (layer-major per page boundary), so the
    # record sequences are index-aligned.
    n = min(len(d) for _, d in runs)
    for label, d in runs:
        anchors = [r for r in d[:n] if r.get("anchor")]
        print("%-14s anchor_calls=%-4d insert_ms mean=%.3f  "
              "new_leaves_total sum=%d mean=%.2f"
              % (label, len(anchors),
                 statistics.mean(r.get("native_insert_ms", 0.0)
                                 for r in anchors),
                 sum(r.get("new_leaves_total", 0) for r in anchors),
                 statistics.mean(r.get("new_leaves_total", 0)
                                 for r in anchors)))

    print("\npairwise |new_leaves_total| difference on anchor calls")
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            la, da = runs[i]
            lb, db = runs[j]
            diffs = []
            for r in range(n):
                if not da[r].get("anchor"):
                    continue
                diffs.append(abs(da[r].get("new_leaves_total", 0)
                                 - db[r].get("new_leaves_total", 0)))
            if diffs:
                print("  %-26s mean=%.2f max=%d n=%d  identical=%d/%d"
                      % ("%s vs %s" % (la, lb),
                         statistics.mean(diffs), max(diffs), len(diffs),
                         sum(1 for x in diffs if x == 0), len(diffs)))

    print("\npairwise |native_insert_ms| difference on anchor calls")
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            la, da = runs[i]
            lb, db = runs[j]
            rel = []
            for r in range(n):
                if not da[r].get("anchor"):
                    continue
                a = da[r].get("native_insert_ms", 0.0)
                b = db[r].get("native_insert_ms", 0.0)
                if a:
                    rel.append(100.0 * abs(b - a) / a)
            if rel:
                print("  %-26s mean|Δ|=%.1f%%  median=%.1f%%  max=%.1f%%"
                      % ("%s vs %s" % (la, lb),
                         statistics.mean(rel), statistics.median(rel),
                         max(rel)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
