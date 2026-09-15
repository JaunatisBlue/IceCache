"""Compact report for the decode-side incremental DCI index-update profile.

Reads run logs that contain `DCI_PROFILE {...}` lines and prints ms/token,
ms/page-boundary and ms/layer-update, plus the tree-size-bucketed breakdown.

Usage:
    python parse_index_profile.py LABEL=path/to/log [LABEL=path ...]
"""

import json
import sys

# The DCI window is installed on layers [2, n_layers); at n_reuse_layers=3 that
# is 30 layers, 10 anchor updates and 20 reuse updates per page boundary.
DCI_LAYERS = 30

ROWS = [
    ("decode TPOT (all)", "decode_tpot_ms"),
    ("index update total", "index_update_ms_per_token"),
    ("  window KV pack", "index_pack_ms_per_token"),
    ("  Tensor->NumPy", "index_numpy_ms_per_token"),
    ("  insert preparation", "index_prepare_ms_per_token"),
    ("  native DCI insert", "index_native_insert_ms_per_token"),
    ("  ccc writeback", "index_ccc_writeback_ms_per_token"),
    ("  CPU page alloc/capacity", "index_page_alloc_ms_per_token"),
    ("  address update overall", "index_address_update_ms_per_token"),
    ("    address prepare", "index_address_prepare_ms_per_token"),
    ("      meta/new_indices", "index_addrprep_meta_ms_per_token"),
    ("      per-leaf data_ptr loop", "index_addrprep_leaf_ms_per_token"),
    ("      list->ndarray", "index_addrprep_np_ms_per_token"),
    ("      page_address_buffer write", "index_addrprep_write_ms_per_token"),
    ("    native address_update", "index_native_address_update_ms_per_token"),
    ("    reuse_update_node", "index_reuse_update_ms_per_token"),
]


def load(paths):
    runs = []
    for spec in paths:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = spec, spec
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("DCI_PROFILE "):
                    runs.append((label, json.loads(line[len("DCI_PROFILE "):])))
    return runs


def report(label, p):
    measured = p["decode_steps_measured"]
    calls = p["index_update_calls"]
    boundaries = calls / DCI_LAYERS if calls else 0.0
    tok_per_b = measured / boundaries if boundaries else float("nan")
    print("=== %s ===" % label)
    print("decode_steps_measured=%d  index_update_calls=%d  "
          "page_boundaries=%.1f  tokens/boundary=%.2f  warmup=%d"
          % (measured, calls, boundaries, tok_per_b, p["warmup_tokens"]))
    print("%-32s %10s %12s %14s" % ("stage", "ms/token", "ms/boundary",
                                    "ms/layer-update"))
    for name, key in ROWS:
        ms = p.get(key)
        if ms is None:
            continue
        per_b = ms * tok_per_b
        per_l = ms * measured / calls if calls else float("nan")
        print("%-32s %10.3f %12.2f %14.4f" % (name, ms, per_b, per_l))

    buckets = (p.get("index_addrprep_buckets") or {}).get("by_tree_size") or {}
    if buckets:
        print("-- by pre-insertion tree size (per layer-level update) --")
        print("%-16s %6s %7s %7s %10s %12s %12s %12s %10s"
              % ("bucket", "calls", "anchor", "reuse", "prev_pts",
                 "addr_prep", "leaf_loop", "native_ins", "new_leaves"))
        for key in sorted(buckets):
            b = buckets[key]
            print("%-16s %6d %7d %7d %10s %12.4f %12.4f %12.4f %10.1f"
                  % (key, b["calls"], b["anchor_calls"], b["reuse_calls"],
                     "%s-%s" % (b["prev_num_points_min"],
                                b["prev_num_points_max"]),
                     b["addr_prep_ms_per_call"], b["addr_leaf_ms_per_call"],
                     b["native_insert_ms_per_call"],
                     b["new_leaves_total"] / max(b["calls"], 1)))
    eq = p.get("addr_equiv")
    if eq:
        print("-- address equivalence (in vivo) --")
        print("checks=%d fast_mismatch=%d logical_shortcut_mismatch=%d "
              "distinct_physical_pages=%d"
              % (eq["checks"], eq["fast_mismatch"],
                 eq["logical_shortcut_mismatch"], eq["distinct_physical_pages"]))
    print()


def main():
    runs = load(sys.argv[1:])
    if not runs:
        print("no DCI_PROFILE lines found")
        return 1
    for label, p in runs:
        report(label, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
