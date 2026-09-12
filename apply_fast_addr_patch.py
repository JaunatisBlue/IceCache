#!/usr/bin/env python3
"""ICECACHE_FASTADDR: vectorised source-address construction in recall().

Why: measured gather cost is 16-18 ms/token and is essentially independent of
the number of pages transferred (18.28 ms at 5181 pages/token vs 16.17 ms at
1480 pages/token).  So it is per-call overhead, not data movement.  The
original loop, executed 8 times per recall and 30 recalls per token, calls
`nr_cpu[i].item()` twice and indexes a NumPy array with a *torch* tensor
(forcing a tensor->array conversion on every iteration).

This patch is A/B gated by ICECACHE_FAST_ADDR so the original path stays
available; it changes no numerics.

Usage:
    python3 apply_fast_addr_patch.py /path/to/infer_state.py
"""
import sys
import os
import shutil

PATCHES = []


def patch(old, new, tag):
    PATCHES.append((old, new, tag))


patch(
    "        self.diag_saved = False\n",
    "        self.diag_saved = False\n"
    "        # [ICECACHE-FASTADDR] vectorised source-address construction (A/B gate)\n"
    "        self.fast_addr = bool(int(os.environ.get(\"ICECACHE_FAST_ADDR\", \"0\")))\n",
    "P1-init",
)

patch(
    "        rids_cpu = rids.cpu()\n"
    "        nr_cpu = nr.cpu()\n"
    "\n"
    "        counter = 0\n"
    "        for i in range(self.n_kv_heads):\n"
    "            self._src_address_buffer[counter:counter+nr_cpu[i].item()] = self.page_address_buffer[layer_idx][b, i, rids_cpu[i, :nr_cpu[i]]]\n"
    "            counter += nr_cpu[i].item()\n"
    "        # [ICECACHE-DIAG] end of CPU address preparation\n",
    "        if self.fast_addr:\n"
    "            # [ICECACHE-FASTADDR] Build the source-address list from NumPy\n"
    "            # views.  The original loop re-converts a torch tensor into a\n"
    "            # NumPy index array and calls .item() on every iteration, which\n"
    "            # costs hundreds of microseconds per recall and is independent\n"
    "            # of how many pages are actually transferred.\n"
    "            rids_cpu = rids.cpu().numpy()\n"
    "            nr_cpu = nr.cpu().numpy()\n"
    "            counter = 0\n"
    "            for i in range(self.n_kv_heads):\n"
    "                c = int(nr_cpu[i])\n"
    "                if c:\n"
    "                    self._src_address_buffer[counter:counter + c] = (\n"
    "                        self.page_address_buffer[layer_idx][\n"
    "                            b, i, rids_cpu[i, :c]])\n"
    "                    counter += c\n"
    "        else:\n"
    "            rids_cpu = rids.cpu()\n"
    "            nr_cpu = nr.cpu()\n"
    "\n"
    "            counter = 0\n"
    "            for i in range(self.n_kv_heads):\n"
    "                self._src_address_buffer[counter:counter+nr_cpu[i].item()] = self.page_address_buffer[layer_idx][b, i, rids_cpu[i, :nr_cpu[i]]]\n"
    "                counter += nr_cpu[i].item()\n"
    "        # [ICECACHE-DIAG] end of CPU address preparation\n",
    "P2-addr-loop",
)

patch(
    "        for i in range(self.n_kv_heads):\n"
    "            cnt = int(nr_cpu[i].item())\n"
    "            if cnt <= 0:\n"
    "                continue\n"
    "            leaves = rids_cpu[i, :cnt].numpy().astype(np.int64)\n",
    "        for i in range(self.n_kv_heads):\n"
    "            cnt = int(nr_cpu[i])\n"
    "            if cnt <= 0:\n"
    "                continue\n"
    "            leaves = np.asarray(rids_cpu[i, :cnt]).astype(np.int64)\n",
    "P3-diag-collect",
)


def main():
    if len(sys.argv) != 2:
        print("usage: apply_fast_addr_patch.py /path/to/infer_state.py")
        return 2
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "ICECACHE-FASTADDR" in src:
        print("ALREADY_PATCHED")
        return 3
    for old, new, tag in PATCHES:
        n = src.count(old)
        if n != 1:
            print("ANCHOR_FAIL %s: %d occurrences" % (tag, n))
            return 4
        src = src.replace(old, new, 1)
        print("OK %s" % tag)
    backup = path + ".bak_prefastaddr"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print("backup -> %s" % backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("PATCH_APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
