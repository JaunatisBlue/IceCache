#!/usr/bin/env python3
"""ICECACHE_VALID_CACHE: compute page_valid_entries once per anchor layer.

n_reuse_layers=3 means three consecutive layers share one DCI selection.  The
reuse layers call
    self.dci_db[reuse_id].get_valid_entries(self.selected_page_idx[reuse_id])
which is byte-for-byte the same expression the anchor layer just evaluated, so
the work is done three times per group (30 times per token instead of 10).

This patch caches the transposed GPU tensor per (anchor layer, seq_len) and
reuses it inside the group.  seq_len increments once per generated token, so it
is a reliable staleness marker.  No numerics change.

Usage:
    python3 apply_valid_cache_patch.py /path/to/infer_state.py
"""
import sys
import os
import shutil

OLD = (
    "                    if self.check_reuse(layer_idx) == 0:\n"
    "                        self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(\n"
    "                            self.dci_db[layer_idx].get_valid_entries(self.selected_page_idx[layer_idx]), **self._i32).T\n"
    "                    else:\n"
    "                        self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(\n"
    "                            self.dci_db[reuse_id].get_valid_entries(self.selected_page_idx[reuse_id]), **self._i32).T\n"
)

NEW = (
    "                    # [ICECACHE-VALIDCACHE] The reuse layers in a group ask\n"
    "                    # for exactly the same valid-entry tensor as their\n"
    "                    # anchor, so compute it once per group.\n"
    "                    _vw = self.n_dci_pages - self.layer2topk[layer_idx]\n"
    "                    if self.valid_cache:\n"
    "                        _vkey = (\n"
    "                            layer_idx if self.check_reuse(layer_idx) == 0\n"
    "                            else reuse_id, self.kv_caches[0].seq_len)\n"
    "                        if self._valid_cache_key != _vkey:\n"
    "                            if self.check_reuse(layer_idx) == 0:\n"
    "                                _vt = torch.tensor(\n"
    "                                    self.dci_db[layer_idx].get_valid_entries(\n"
    "                                        self.selected_page_idx[layer_idx]),\n"
    "                                    **self._i32).T\n"
    "                            else:\n"
    "                                _vt = torch.tensor(\n"
    "                                    self.dci_db[reuse_id].get_valid_entries(\n"
    "                                        self.selected_page_idx[reuse_id]),\n"
    "                                    **self._i32).T\n"
    "                            self._valid_cache_tensor = _vt\n"
    "                            self._valid_cache_key = _vkey\n"
    "                        self.page_valid_entries[layer_idx][ns: ns + _vw] = (\n"
    "                            self._valid_cache_tensor[:_vw])\n"
    "                    else:\n"
    "                        if self.check_reuse(layer_idx) == 0:\n"
    "                            self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(\n"
    "                                self.dci_db[layer_idx].get_valid_entries(self.selected_page_idx[layer_idx]), **self._i32).T\n"
    "                        else:\n"
    "                            self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(\n"
    "                                self.dci_db[reuse_id].get_valid_entries(self.selected_page_idx[reuse_id]), **self._i32).T\n"
)

INIT_OLD = (
    "        self.diag_saved = False\n"
)
INIT_NEW = (
    "        # [ICECACHE-VALIDCACHE] reuse-group memoisation of valid-entry tensors\n"
    "        self.valid_cache = bool(int(os.environ.get(\"ICECACHE_VALID_CACHE\", \"1\")))\n"
    "        self._valid_cache_key = None\n"
    "        self._valid_cache_tensor = None\n"
    "        self.diag_saved = False\n"
)

PATCHES = [(OLD, NEW, "P1-valid-cache"), (INIT_OLD, INIT_NEW, "P2-init")]


def main():
    if len(sys.argv) != 2:
        print("usage: apply_valid_cache_patch.py /path/to/infer_state.py")
        return 2
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "ICECACHE-VALIDCACHE" in src:
        print("ALREADY_PATCHED")
        return 3
    for old, new, tag in PATCHES:
        n = src.count(old)
        if n != 1:
            print("ANCHOR_FAIL %s: %d occurrences" % (tag, n))
            return 4
        src = src.replace(old, new, 1)
        print("OK %s" % tag)
    backup = path + ".bak_prevalidcache"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print("backup -> %s" % backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("PATCH_APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
