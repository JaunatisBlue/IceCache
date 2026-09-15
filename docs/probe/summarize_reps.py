"""Arm means over several repetitions of the same configuration.

Companion to report_ab_tpot.py: that one compares two single runs; this one
takes N runs per arm and reports the per-run values plus the arm mean, which is
what "run it twice and average" actually requires.

Usage:
    python summarize_reps.py A=log1,log2 B=log3,log4
"""

import json
import statistics
import sys

PREFIX = "DCI_PROFILE "

ROWS = [
    ("TPOT mean (ms/token)", "_step_mean"),
    ("TPOT std (ms/token)", "_step_std"),
    ("TPOT p50", "_step_p50"),
    ("TPOT p95", "_step_p95"),
    ("decode steps measured", "decode_steps_measured"),
    ("index update total", "index_update_ms_per_token"),
    ("address prepare", "index_address_prepare_ms_per_token"),
    ("per-leaf data_ptr loop", "index_addrprep_leaf_ms_per_token"),
    ("native DCI insert", "index_native_insert_ms_per_token"),
    ("CPU page alloc", "index_page_alloc_ms_per_token"),
    ("reuse_update_node", "index_reuse_update_ms_per_token"),
    ("recall_wait", "recall_wait_ms_per_token"),
    ("native_query", "native_query_ms_per_token"),
    ("recall_gather", "recall_gather_ms_per_token"),
    ("page_metadata", "page_metadata_ms_per_token"),
]


def load(path):
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith(PREFIX):
                p = json.loads(line[len(PREFIX):])
                break
        else:
            raise SystemExit("no DCI_PROFILE line in %s" % path)
    lat = p.get("decode_step_latency")
    for src, dst in (("mean_ms", "_step_mean"), ("std_ms", "_step_std"),
                     ("p50_ms", "_step_p50"), ("p95_ms", "_step_p95")):
        p[dst] = (lat or {}).get(src)
    p["steps_n"] = (lat or {}).get("n")
    return p


def main(specs):
    arms = {}
    for spec in specs:
        label, paths = spec.split("=", 1)
        arms[label] = [load(p) for p in paths.split(",") if p]

    labels = list(arms)
    print("%-28s %s" % ("metric", "".join("%14s" % l for l in labels)))
    print("-" * (28 + 15 * len(labels)))
    for name, key in ROWS:
        row = "%-28s" % name
        for l in labels:
            vals = [r.get(key) for r in arms[l] if r.get(key) is not None]
            if not vals:
                row += "%14s" % "-"
            elif len(vals) == 1:
                row += "%14.4f" % vals[0]
            else:
                row += "%14.4f" % statistics.mean(vals)
        print(row)

    print()
    print("per-run detail")
    for l in labels:
        print("  %s:" % l)
        for i, r in enumerate(arms[l], 1):
            print("    run%d  steps_n=%-4s tpot_mean=%-9.3f std=%-8.3f "
                  "p95=%-9.3f addr_prep=%-8.4f idx_update=%.4f"
                  % (i, r.get("steps_n"), r.get("_step_mean") or -1,
                     r.get("_step_std") or -1, r.get("_step_p95") or -1,
                     r.get("index_address_prepare_ms_per_token") or -1,
                     r.get("index_update_ms_per_token") or -1))

    if len(labels) == 2:
        la, lb = labels
        print()
        print("arm-mean delta (%s - %s)" % (lb, la))
        for name, key in ROWS:
            va = [r[key] for r in arms[la] if r.get(key) is not None]
            vb = [r[key] for r in arms[lb] if r.get(key) is not None]
            if not va or not vb:
                continue
            ma, mb = statistics.mean(va), statistics.mean(vb)
            pct = ("%+.2f%%" % (100 * (mb - ma) / ma)) if ma else "-"
            extra = ""
            # spread of the per-run values, so the reader can see whether the
            # delta is bigger than the run-to-run scatter
            if len(va) > 1 and len(vb) > 1:
                sa = statistics.stdev(va)
                sb = statistics.stdev(vb)
                extra = "  (within-arm sd: %s=%.4f %s=%.4f)"
                extra = extra % (la, sa, lb, sb)
            print("  %-28s %+10.4f  %10s%s" % (name, mb - ma, pct, extra))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
