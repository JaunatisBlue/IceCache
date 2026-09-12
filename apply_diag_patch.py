#!/usr/bin/env python3
"""Apply the ICECACHE_DIAG joint-diagnostic patch to infer_state.py.

Pure-additive, env-gated (ICECACHE_DIAG=1). When the env var is absent the
instrumented code reduces to a couple of boolean checks and behaviour is
byte-identical to the unpatched file.

Usage:
    python3 apply_diag_patch.py /path/to/infer_state.py
"""
import sys
import shutil
import os

PATCHES = []


def patch(old, new, tag):
    PATCHES.append((old, new, tag))


# ---------------------------------------------------------------- P1 imports
patch(
    "import os\nfrom threading import Thread\n",
    "import os\nimport atexit\nfrom threading import Thread\n",
    "P1-import-atexit",
)

# ------------------------------------------------------------- P2 init state
patch(
    "        self.profile_cross_token_boundary_refreshes = 0\n"
    "        self._profile_decode_start = None\n",
    "        self.profile_cross_token_boundary_refreshes = 0\n"
    "        self._profile_decode_start = None\n"
    "\n"
    "        # === ICECACHE_DIAG: joint diagnostic (timing split + address mergeability) ===\n"
    "        # Pure-additive, opt-in.  When ICECACHE_DIAG is unset this whole block\n"
    "        # costs two bool checks per recall and changes no numerical behaviour.\n"
    "        self.diag_enabled = bool(int(os.environ.get(\"ICECACHE_DIAG\", \"0\")))\n"
    "        self.diag_dump_path = os.environ.get(\"ICECACHE_DIAG_DUMP\", \"\")\n"
    "        self.diag_max_records = int(\n"
    "            os.environ.get(\"ICECACHE_DIAG_MAX_RECORDS\", \"400\"))\n"
    "        # Bound the number of CUDA events created so a long run cannot\n"
    "        # accumulate unbounded event objects.\n"
    "        self.diag_event_budget = int(\n"
    "            os.environ.get(\"ICECACHE_DIAG_MAX_EVENTS\", \"800\"))\n"
    "        self.diag_addr_prep_seconds = 0.0\n"
    "        self.diag_copy_buffer_seconds = 0.0\n"
    "        self.diag_h2d_ms = 0.0\n"
    "        self.diag_cast_ms = 0.0\n"
    "        self.diag_h2d_count = 0\n"
    "        self.diag_cast_count = 0\n"
    "        self.diag_records = []\n"
    "        self.diag_pending = []\n"
    "        self.diag_saved = False\n"
    "        if self.diag_enabled:\n"
    "            atexit.register(self._save_diag)\n",
    "P2-init-state",
)

# ---------------------------------------------------- P3 helper methods
patch(
    "    def get_profile_stats(self):\n",
    "    def _diag_collect(self, layer_idx, b, rids_cpu, nr_cpu):\n"
    "        # [ICECACHE-DIAG] record the selected leaf ids + their real CPU\n"
    "        # addresses, per (layer, head).  Only the first\n"
    "        # diag_max_records recalls are kept, so the dump stays tiny.\n"
    "        if len(self.diag_records) >= self.diag_max_records:\n"
    "            return\n"
    "        for i in range(self.n_kv_heads):\n"
    "            cnt = int(nr_cpu[i].item())\n"
    "            if cnt <= 0:\n"
    "                continue\n"
    "            leaves = rids_cpu[i, :cnt].numpy().astype(np.int64)\n"
    "            addrs = self.page_address_buffer[layer_idx][\n"
    "                b, i, leaves].astype(np.uint64)\n"
    "            self.diag_records.append((layer_idx, i, leaves, addrs))\n"
    "        if (len(self.diag_records) >= self.diag_max_records\n"
    "                and not self.diag_saved):\n"
    "            self._save_diag()\n"
    "\n"
    "    def _save_diag(self):\n"
    "        if not self.diag_enabled:\n"
    "            return\n"
    "        path = self.diag_dump_path or os.path.join(\n"
    "            os.getcwd(), \"icecache_diag.npz\")\n"
    "        try:\n"
    "            if self.diag_records:\n"
    "                layer = np.asarray(\n"
    "                    [r[0] for r in self.diag_records], dtype=np.int32)\n"
    "                head = np.asarray(\n"
    "                    [r[1] for r in self.diag_records], dtype=np.int32)\n"
    "                flat_leaf = np.concatenate([r[2] for r in self.diag_records])\n"
    "                flat_addr = np.concatenate([r[3] for r in self.diag_records])\n"
    "                offs = np.zeros(len(self.diag_records) + 1, dtype=np.int64)\n"
    "                for k, r in enumerate(self.diag_records):\n"
    "                    offs[k + 1] = offs[k] + len(r[2])\n"
    "            else:\n"
    "                layer = np.zeros(0, np.int32)\n"
    "                head = np.zeros(0, np.int32)\n"
    "                flat_leaf = np.zeros(0, np.int64)\n"
    "                flat_addr = np.zeros(0, np.uint64)\n"
    "                offs = np.zeros(1, np.int64)\n"
    "            np.savez(\n"
    "                path, layer=layer, head=head, flat_leaf=flat_leaf,\n"
    "                flat_addr=flat_addr, offsets=offs,\n"
    "                addr_prep_seconds=self.diag_addr_prep_seconds,\n"
    "                copy_buffer_seconds=self.diag_copy_buffer_seconds,\n"
    "                h2d_ms=self.diag_h2d_ms, cast_ms=self.diag_cast_ms,\n"
    "                h2d_count=self.diag_h2d_count,\n"
    "                cast_count=self.diag_cast_count)\n"
    "            self.diag_saved = True\n"
    "            print(\"[ICECACHE-DIAG] records=%d -> %s\" % (\n"
    "                len(self.diag_records), path))\n"
    "            print(\"[ICECACHE-DIAG] addr_prep_s=%.4f copy_buffer_s=%.4f \"\n"
    "                  \"h2d_ms=%.3f cast_ms=%.3f h2d_n=%d cast_n=%d\" % (\n"
    "                      self.diag_addr_prep_seconds,\n"
    "                      self.diag_copy_buffer_seconds,\n"
    "                      self.diag_h2d_ms, self.diag_cast_ms,\n"
    "                      self.diag_h2d_count, self.diag_cast_count))\n"
    "        except Exception as exc:  # never let diagnostics break a run\n"
    "            print(\"[ICECACHE-DIAG] save failed: %r\" % (exc,))\n"
    "\n"
    "    def get_profile_stats(self):\n",
    "P3-helper-methods",
)

# ------------------------------------------- P4 recall(): addr-prep boundary
patch(
    "            counter += nr_cpu[i].item()\n"
    "\n"
    "        with torch.cuda.stream(c2g_stream):\n"
    "\n"
    "            DCI.copy_to_buffer(self._src_address_buffer, "
    "ptr_dest=cast(self.cpu_transit_buffer[b].data_ptr(), c_void_p).value,\n",
    "            counter += nr_cpu[i].item()\n"
    "        # [ICECACHE-DIAG] end of CPU address preparation\n"
    "        diag_addr_prep_end = perf_counter() if self.diag_enabled else None\n"
    "        if self.diag_enabled and gather_start is not None:\n"
    "            self.diag_addr_prep_seconds += (diag_addr_prep_end - gather_start)\n"
    "            self._diag_collect(layer_idx, b, rids_cpu, nr_cpu)\n"
    "\n"
    "        with torch.cuda.stream(c2g_stream):\n"
    "\n"
    "            DCI.copy_to_buffer(self._src_address_buffer, "
    "ptr_dest=cast(self.cpu_transit_buffer[b].data_ptr(), c_void_p).value,\n",
    "P4-addr-prep-boundary",
)

# ------------------------------- P5 recall(): copy_to_buffer + CUDA events
patch(
    "            self.profile_recall_pages += n_transit_pages\n"
    "        ############################################################\n"
    "\n"
    "        with torch.cuda.stream(c2g_stream):\n"
    "            dst = self.cuda_transit_buffer[:, : 2 * n_transit_pages, :]\n"
    "            src = self.cpu_transit_buffer[:, : 2 * n_transit_pages, :]\n"
    "            dst.copy_(src, non_blocking=True)\n"
    "\n"
    "            self.cuda_cast_buffer[:, : 2 * n_transit_pages, :].copy_(\n"
    "                dst, non_blocking=True\n"
    "            )\n",
    "            self.profile_recall_pages += n_transit_pages\n"
    "        # [ICECACHE-DIAG] copy_to_buffer segment (CPU-side scattered gather)\n"
    "        if self.diag_enabled and diag_addr_prep_end is not None:\n"
    "            self.diag_copy_buffer_seconds += (perf_counter() - diag_addr_prep_end)\n"
    "        ############################################################\n"
    "\n"
    "        with torch.cuda.stream(c2g_stream):\n"
    "            dst = self.cuda_transit_buffer[:, : 2 * n_transit_pages, :]\n"
    "            src = self.cpu_transit_buffer[:, : 2 * n_transit_pages, :]\n"
    "            # [ICECACHE-DIAG] CUDA-event brackets for H2D and cast segments\n"
    "            diag_on = self.diag_enabled and self.diag_event_budget > 0\n"
    "            diag_e0 = torch.cuda.Event(enable_timing=True) if diag_on else None\n"
    "            diag_e1 = torch.cuda.Event(enable_timing=True) if diag_on else None\n"
    "            diag_e2 = torch.cuda.Event(enable_timing=True) if diag_on else None\n"
    "            if diag_e0 is not None:\n"
    "                diag_e0.record(c2g_stream)\n"
    "            dst.copy_(src, non_blocking=True)\n"
    "            if diag_e1 is not None:\n"
    "                diag_e1.record(c2g_stream)\n"
    "\n"
    "            self.cuda_cast_buffer[:, : 2 * n_transit_pages, :].copy_(\n"
    "                dst, non_blocking=True\n"
    "            )\n"
    "            if diag_e2 is not None:\n"
    "                diag_e2.record(c2g_stream)\n"
    "            if diag_e0 is not None:\n"
    "                self.diag_pending.append((diag_e0, diag_e1, diag_e2))\n"
    "                self.diag_event_budget -= 1\n",
    "P5-copybuf-events",
)

# --------------------------- P6 read event timings after the existing sync
patch(
    "                c2g_stream.synchronize()\n"
    "                if recall_wait_start is not None:\n"
    "                    self.profile_recall_wait_seconds += (\n"
    "                        perf_counter() - recall_wait_start)\n",
    "                c2g_stream.synchronize()\n"
    "                if recall_wait_start is not None:\n"
    "                    self.profile_recall_wait_seconds += (\n"
    "                        perf_counter() - recall_wait_start)\n"
    "                # [ICECACHE-DIAG] read H2D/cast event timings now the stream drained\n"
    "                if self.diag_enabled and self.diag_pending:\n"
    "                    for _e0, _e1, _e2 in self.diag_pending:\n"
    "                        self.diag_h2d_ms += _e0.elapsed_time(_e1)\n"
    "                        self.diag_cast_ms += _e1.elapsed_time(_e2)\n"
    "                    self.diag_h2d_count += len(self.diag_pending)\n"
    "                    self.diag_cast_count += len(self.diag_pending)\n"
    "                    self.diag_pending = []\n",
    "P6-read-events",
)


def main():
    if len(sys.argv) != 2:
        print("usage: apply_diag_patch.py /path/to/infer_state.py")
        return 2
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if "ICECACHE_DIAG" in src:
        print("ALREADY_PATCHED: refusing to apply twice")
        return 3

    for old, new, tag in PATCHES:
        n = src.count(old)
        if n != 1:
            print("ANCHOR_FAIL %s: found %d occurrences (need exactly 1)" % (tag, n))
            return 4
        src = src.replace(old, new, 1)
        print("OK %s" % tag)

    backup = path + ".bak_prediag"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print("backup -> %s" % backup)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("PATCH_APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
