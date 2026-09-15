"""Pairwise comparison of qasper prediction files from the A/B matrix.

Answers: is the vectorised address path distinguishable from plain run-to-run
variance?  Compares every pair of runs, per sample, on the generated text.

Usage: python compare_preds.py NAME=path/qasper.jsonl [NAME=path ...]
"""

import difflib
import itertools
import json
import sys


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def main(specs):
    runs = []
    for spec in specs:
        label, path = spec.split("=", 1)
        runs.append((label, load(path)))
    n_samples = min(len(d) for _, d in runs)
    print("samples per run: %d   runs: %s"
          % (n_samples, ", ".join(l for l, _ in runs)))

    print("\nper-sample identity matrix (1 = byte identical `pred`)")
    header = "        " + "".join("%-10s" % l for l, _ in runs)
    print(header)
    for i in range(n_samples):
        row = "s%-7d" % i
        for _, d in runs:
            row += "%-10s" % ("same" if d[i]["pred"] == runs[0][1][i]["pred"]
                              else "diff")
        print(row)

    print("\npairwise agreement on `pred`")
    print("%-22s %-8s %8s %8s %10s" % ("pair", "sample", "identical", "ratio",
                                       "len_a/len_b"))
    for (la, da), (lb, db) in itertools.combinations(runs, 2):
        for i in range(n_samples):
            x, y = da[i]["pred"], db[i]["pred"]
            ratio = difflib.SequenceMatcher(None, x, y).ratio()
            print("%-22s %-8d %8s %8.3f %10s"
                  % ("%s vs %s" % (la, lb), i, x == y, ratio,
                     "%d/%d" % (len(x), len(y))))

    # Aggregate: how many of the n samples differ, per pair.
    print("\nsummary: samples whose `pred` differ")
    for (la, da), (lb, db) in itertools.combinations(runs, 2):
        n = sum(1 for i in range(n_samples) if da[i]["pred"] != db[i]["pred"])
        print("  %-22s %d/%d" % ("%s vs %s" % (la, lb), n, n_samples))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
