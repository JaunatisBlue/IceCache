"""Extract addr_equiv summary from DCI_PROFILE lines of a run log."""
import json
import sys

PREFIX = "DCI_PROFILE "


def main(paths):
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                if not line.startswith(PREFIX):
                    continue
                d = json.loads(line[len(PREFIX):])
                eq = d.get("addr_equiv")
                print("== %s" % path)
                if not eq:
                    print("addr_equiv not armed")
                    continue
                print("checks=%d fast_mismatch=%d logical_shortcut_mismatch=%d "
                      "distinct_physical_pages=%d"
                      % (eq["checks"], eq["fast_mismatch"],
                         eq["logical_shortcut_mismatch"],
                         eq["distinct_physical_pages"]))
                layers = eq["by_layer"]
                tot = sum(v["n"] for v in layers.values())
                logi = sum(v["logical_shortcut_mismatch"]
                           for v in layers.values())
                print("layers_checked=%d elements=%d "
                      "logical_shortcut_wrong_elements=%d (%.1f%%)"
                      % (len(layers), tot, logi, 100.0 * logi / max(tot, 1)))
                bad = {k: v for k, v in layers.items()
                       if v["fast_mismatch"] or v["logical_shortcut_mismatch"]}
                print("layers_with_any_mismatch=%d/%d"
                      % (len(bad), len(layers)))
                print("per-layer (layer: n/fast/logical):")
                items = sorted(layers.items(), key=lambda kv: int(kv[0]))
                print("  " + "  ".join(
                    "%s:%d/%d/%d" % (k, v["n"], v["fast_mismatch"],
                                     v["logical_shortcut_mismatch"])
                    for k, v in items))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
