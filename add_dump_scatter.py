#!/usr/bin/env python3
"""Add an env-gated dump (ICECACHE_DUMP_TRANSIT=1) to scatter_pages.

Works on BOTH patched (ICECACHE_DOUBLE_BUFFER) and unpatched infer_state.py:

  - unpatched: anchors on the plain 'transit = self.cuda_cast_buffer' body
  - patched  : anchors on the 'if self.double_buffer:' body (dump the actual
               slot being consumed, i.e. _last_slot)

Usage: python3 add_dump_scatter.py /path/to/infer_state.py
"""
import sys

ANCHOR_NDB = (
    "        transit = self.cuda_cast_buffer\n"
    "        if self.batch_layer_recall and layer_idx in self._batched_recall_slices:\n"
)
NEW_NDB = (
    "        transit = self.cuda_cast_buffer\n"
    "        _dump_transit(\"ndb\", layer_idx, eids, nr, transit, None)\n"
    "        if self.batch_layer_recall and layer_idx in self._batched_recall_slices:\n"
)

ANCHOR_DB = (
    "        if self.double_buffer:\n"
    "            # [ICECACHE-DOUBLEBUF] consume slot _last_slot, ordered by event\n"
)
NEW_DB = (
    "        if self.double_buffer:\n"
    "            # [ICECACHE-DOUBLEBUF] consume slot _last_slot, ordered by event\n"
    "            _dump_transit(\"db\", layer_idx, eids, nr, self.cuda_cast_buffer, self._last_slot)\n"
)

HELPER = (
    "def _dump_transit(tag, layer_idx, eids, nr, transit, last_slot):\n"
    "    import os\n"
    "    if os.environ.get('ICECACHE_DUMP_TRANSIT') != '1':\n"
    "        return\n"
    "    import torch\n"
    "    try:\n"
    "        nr_sum = int(nr.sum().item()) if nr is not None else -1\n"
    "        e8 = eids.flatten()[:8].tolist() if eids is not None else None\n"
    "        n8 = nr.flatten()[:8].tolist() if nr is not None else None\n"
    "        if isinstance(transit, list):\n"
    "            transit = transit[last_slot if last_slot is not None else 0]\n"
    "        t = transit.float()\n"
    "        tsum = float(t.sum().item())\n"
    "        tabs = float(t.abs().sum().item())\n"
    "        t8 = t.flatten()[:8].tolist()\n"
    "        line = ('DUMP tag=%s layer=%d last=%s nr_sum=%d eids8=%s nr8=%s '\n"
    "                'tr_sum=%.6f tr_abs=%.6f tr8=%s' % (\n"
    "                    tag, layer_idx, last_slot, nr_sum, e8, n8, tsum, tabs, t8))\n"
    "    except Exception as ex:\n"
    "        line = 'DUMP tag=%s layer=%d EXC %r' % (tag, layer_idx, ex)\n"
    "    with open(os.environ.get('ICECACHE_DUMP_PATH', '/tmp/transit_dump.txt'), 'a') as fh:\n"
    "        fh.write(line + '\\n')\n"
    "    print(line, flush=True)\n"
    "\n"
)


def main():
    if len(sys.argv) != 2:
        print("usage: add_dump_scatter.py /path/to/infer_state.py")
        return 2
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "_dump_transit" in src:
        print("ALREADY_DUMPED")
        return 3
    patched = "ICECACHE-DOUBLEBUF" in src
    if patched:
        anchors = ((ANCHOR_DB, NEW_DB, "dump-db"),)
    else:
        anchors = ((ANCHOR_NDB, NEW_NDB, "dump-ndb"),)
    for old, new, tag in anchors:
        n = src.count(old)
        if n != 1:
            print("ANCHOR_FAIL %s: %d" % (tag, n))
            return 4
        src = src.replace(old, new, 1)
        print("OK %s" % tag)
    lines = src.split("\n")
    idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("class ") or ln.startswith("@register"):
            idx = i
            break
    if idx is None:
        print("NO_CLASS_ANCHOR")
        return 5
    lines.insert(idx, HELPER.rstrip("\n"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("HELPER_INSERTED at line", idx + 1)
    print("PATCH_APPLIED patched=%s" % patched)
    return 0


if __name__ == "__main__":
    sys.exit(main())