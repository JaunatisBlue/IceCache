"""A/B report: decode-step latency (mean / std / percentiles) + key stages.

Reads run logs that contain a `DCI_PROFILE {...}` line and prints a side-by-side
table for the arms given on the command line.

Usage:
    python report_ab_tpot.py A=path/a.log B=path/b.log
"""

import json
import sys

PREFIX = "DCI_PROFILE "

# (display name, key, unit, higher_is_better)
ROWS = [
    ("decode steps measured", "decode_steps_measured", "", True),
    ("**TPOT mean (ms/token)**", "_step_mean", "ms", False),
    ("**TPOT std (ms/token)**", "_step_std", "ms", False),
    ("TPOT CV (std/mean)", "_step_cv", "", False),
    ("TPOT p50 (ms/token)", "_step_p50", "ms", False),
    ("TPOT p95 (ms/token)", "_step_p95", "ms", False),
    ("aggregate decode_tpot_ms", "decode_tpot_ms", "ms", False),
    ("index update total", "index_update_ms_per_token", "ms", False),
    ("  address prepare", "index_address_prepare_ms_per_token", "ms", False),
    ("    per-leaf data_ptr loop", "index_addrprep_leaf_ms_per_token", "ms", False),
    ("  native DCI insert", "index_native_insert_ms_per_token", "ms", False),
    ("  CPU page alloc", "index_page_alloc_ms_per_token", "ms", False),
    ("  reuse_update_node", "index_reuse_update_ms_per_token", "ms", False),
    ("recall_wait", "recall_wait_ms_per_token", "ms", False),
    ("native_query", "native_query_ms_per_token", "ms", False),
    ("recall_gather", "recall_gather_ms_per_token", "ms", False),
    ("page_metadata", "page_metadata_ms_per_token", "ms", False),
    ("index update calls", "index_update_calls", "", True),
]


def load(spec):
    label, path = spec.split("=", 1)
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith(PREFIX):
                prof = json.loads(line[len(PREFIX):])
                break
        else:
            raise SystemExit("no DCI_PROFILE line in %s" % path)
    lat = prof.get("decode_step_latency")
    prof["_step_mean"] = lat["mean_ms"] if lat else None
    prof["_step_std"] = lat["std_ms"] if lat else None
    prof["_step_cv"] = lat["cv"] if lat else None
    prof["_step_p50"] = lat["p50_ms"] if lat else None
    prof["_step_p95"] = lat["p95_ms"] if lat else None
    prof["_step_n"] = lat["n"] if lat else None
    return label, prof


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "%.4f" % v if abs(v) < 10 else "%.2f" % v
    return str(v)


def main(specs):
    runs = [load(s) for s in specs]
    if len(runs) != 2:
        raise SystemExit("expected exactly two arms")
    (la, a), (lb, b) = runs

    print("%-30s %14s %14s %12s %10s"
          % ("metric", la, lb, "delta", "delta%"))
    print("-" * 84)
    for name, key, unit, _ in ROWS:
        va, vb = a.get(key), b.get(key)
        if va is None and vb is None:
            continue
        sa, sb = fmt(va), fmt(vb)
        if (isinstance(va, (int, float)) and isinstance(vb, (int, float))
                and not isinstance(va, bool)):
            d = vb - va
            pct = ("%+.2f%%" % (100.0 * d / va)) if va else "-"
            print("%-30s %14s %14s %12s %10s"
                  % (name, sa, sb, ("%+.4f" % d if abs(d) < 10
                                    else "%+.2f" % d), pct))
        else:
            print("%-30s %14s %14s %12s %10s" % (name, sa, sb, "-", "-"))

    lat_a = a.get("decode_step_latency") or {}
    lat_b = b.get("decode_step_latency") or {}
    print()
    if lat_a and lat_b:
        print("per-step samples: A n=%d  B n=%d"
              % (lat_a.get("n", -1), lat_b.get("n", -1)))
        print("A: mean %.3f  std %.3f  cv %.3f  min %.3f  max %.3f ms"
              % (lat_a["mean_ms"], lat_a["std_ms"], lat_a["cv"],
                 lat_a["min_ms"], lat_a["max_ms"]))
        print("B: mean %.3f  std %.3f  cv %.3f  min %.3f  max %.3f ms"
              % (lat_b["mean_ms"], lat_b["std_ms"], lat_b["cv"],
                 lat_b["min_ms"], lat_b["max_ms"]))
        # Welch-style sanity: is the mean shift bigger than the standard error?
        se = ((lat_a["std_ms"] ** 2 / lat_a["n"])
              + (lat_b["std_ms"] ** 2 / lat_b["n"])) ** 0.5
        d = lat_b["mean_ms"] - lat_a["mean_ms"]
        print("mean delta = %+.3f ms  (pooled SE %.3f ms)  =>  %.2f SE"
              % (d, se, (d / se) if se else float("nan")))
    else:
        print("per-step latency samples not collected "
              "(set ICECACHE_PROFILE_STEP_SAMPLES=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
