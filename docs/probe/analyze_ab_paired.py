"""Paired per-sample TPOT analysis for the 20-sample Qasper A/B.

The aggregate `DCI_PROFILE` mean is confounded: the two arms decoded a
different number of tokens (363 vs 348), so their steady-state means are not
comparable.  `longbench_pred.py` prints one decode latency per sample
(`Decode latencies: <mean over that sample's decode tokens>`), which gives a
paired comparison over the same 20 documents.

Usage: python analyze_ab_paired.py A=path/a.log B=path/b.log
"""

import re
import statistics
import sys

LAT = re.compile(r"Decode latencies:\s*([0-9.]+)")
CTX = re.compile(r"Context length:\s*([0-9]+)")


def load(path):
    ctx, lat = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = CTX.search(line)
            if m:
                ctx.append(int(m.group(1)))
            m = LAT.search(line)
            if m:
                lat.append(float(m.group(1)) * 1e3)  # ms
    return ctx, lat


def main(specs):
    (la, pa), (lb, pb) = [s.split("=", 1) for s in specs]
    ca, ta = load(pa)
    cb, tb = load(pb)
    print("samples: %s=%d  %s=%d" % (la, len(ta), lb, len(tb)))
    if len(ta) != len(tb):
        print("WARNING: different sample counts, pairing aborted")
        return 1
    if ca != cb:
        print("WARNING: context lengths differ between arms")
        for i, (x, y) in enumerate(zip(ca, cb)):
            if x != y:
                print("  sample %d: %d vs %d" % (i, x, y))
    else:
        print("context lengths identical across arms (%d samples, "
              "min %d / max %d)" % (len(ca), min(ca), max(ca)))

    n = len(ta)
    ma, sa = statistics.mean(ta), statistics.stdev(ta)
    mb, sb = statistics.mean(tb), statistics.stdev(tb)
    print()
    print("per-sample decode latency (ms/token), n=%d" % n)
    print("  %-14s mean=%8.3f  std=%7.3f  min=%8.3f  max=%8.3f"
          % (la, ma, sa, min(ta), max(ta)))
    print("  %-14s mean=%8.3f  std=%7.3f  min=%8.3f  max=%8.3f"
          % (lb, mb, sb, min(tb), max(tb)))
    print("  unpaired delta = %+.3f ms" % (mb - ma))

    d = [y - x for x, y in zip(ta, tb)]
    md = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / (n ** 0.5)
    wins = sum(1 for x in d if x < 0)
    print()
    print("paired per-sample delta (%s - %s)" % (lb, la))
    print("  mean=%+.3f ms  std=%7.3f  SE=%6.3f  =>  %+.2f SE"
          % (md, sd, se, md / se if se else float("nan")))
    print("  median=%+.3f  min=%+.3f  max=%+.3f"
          % (statistics.median(d), min(d), max(d)))
    print("  samples where %s is faster: %d/%d" % (lb, wins, n))
    # 95% CI on the paired mean (normal approximation, n=20)
    lo, hi = md - 1.96 * se, md + 1.96 * se
    print("  95%% CI on paired delta: [%+.3f, %+.3f] ms" % (lo, hi))
    print("  => CI %s zero" % ("EXCLUDES" if lo * hi > 0 else "includes"))

    print()
    print("per-sample table (context_len, %s ms, %s ms, delta ms)" % (la, lb))
    for i in range(n):
        print("  %2d  len=%-6d  A=%8.3f  B=%8.3f  d=%+8.3f"
              % (i, ca[i] if i < len(ca) else -1, ta[i], tb[i], d[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
