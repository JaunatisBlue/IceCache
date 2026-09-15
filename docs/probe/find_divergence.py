"""Find where two full-run diag dumps first diverge."""

import sys

import numpy as np


def main(pa, pb):
    a = np.load(pa)
    b = np.load(pb)
    oa, ob = a["offsets"].astype(np.int64), b["offsets"].astype(np.int64)
    la_, lb_ = a["flat_leaf"], b["flat_leaf"]
    na, nb = len(oa) - 1, len(ob) - 1
    print("records A=%d B=%d" % (na, nb))
    n_diff_recs = 0
    first = None
    for r in range(min(na, nb)):
        na_s, nb_s = oa[r + 1] - oa[r], ob[r + 1] - ob[r]
        differs = (
            na_s != nb_s
            or a["layer"][r] != b["layer"][r]
            or a["head"][r] != b["head"][r]
            or not np.array_equal(la_[oa[r]:oa[r + 1]], lb_[ob[r]:ob[r + 1]]))
        if differs:
            n_diff_recs += 1
            if first is None:
                first = r
    print("records differing = %d / %d" % (n_diff_recs, min(na, nb)))
    if first is None:
        print("RESULT: every record identical (leaf ids)")
        return 0
    print("first differing record = %d  (layer=%d head=%d)  n_sel A=%d B=%d"
          % (first, a["layer"][first], a["head"][first],
             oa[first + 1] - oa[first], ob[first + 1] - ob[first]))
    print("identical records before it = %d  (~decode step %.1f of the run)"
          % (first, first / (30.0 * 8)))
    for r in range(first, min(first + 3, min(na, nb))):
        print("  rec %d layer=%d head=%d nA=%d nB=%d"
              % (r, a["layer"][r], a["head"][r],
                 oa[r + 1] - oa[r], ob[r + 1] - ob[r]))
        print("      leafA[:8]=%s" % la_[oa[r]:oa[r] + 8])
        print("      leafB[:8]=%s" % lb_[ob[r]:ob[r] + 8])
        sa = set(int(x) for x in la_[oa[r]:oa[r + 1]])
        sb = set(int(x) for x in lb_[ob[r]:ob[r + 1]])
        print("      |A|=%d |B|=%d |A&B|=%d" % (len(sa), len(sb),
                                                len(sa & sb)))
    tot_a, tot_b = int(oa[-1]), int(ob[-1])
    print("total selected leaves A=%d B=%d (%.3f%%)"
          % (tot_a, tot_b, 100.0 * (tot_b - tot_a) / tot_a))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
