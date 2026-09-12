#!/usr/bin/env python3
"""ICECACHE_DOUBLE_BUFFER: ping-pong the transit buffers, order with events.

Background: recall() issues the H2D on c2g_stream and the call site then calls
c2g_stream.synchronize().  That host wait is 46% of the decode-side work.  It
cannot simply be deleted (measured: 2x slower, KV corruption) because
cpu_transit_buffer / cuda_transit_buffer / cuda_cast_buffer are single shared
buffers and the next layer would overwrite pinned memory still being read by
the in-flight DMA.

This patch gives each buffer two slots and expresses the three real hazards
with CUDA events:
  ev_read[p] : pinned source for slot p is free again  (recorded on c2g after H2D)
  ev_cast[p] : slot p's cast is done                   (recorded on c2g after cast)
  ev_used[p] : slot p's consumer finished reading it   (recorded on the default
                                                        stream after scatter)
The host only blocks on ev_read[p], which belongs to the slot used two layers
ago, so the gather for layer L overlaps the DMA of layer L-1.

Env-gated, default OFF.  Incompatible with batched layer recall and with layer
prefetch (both assert out).

Usage:
    python3 apply_double_buffer_patch.py /path/to/infer_state.py
"""
import sys
import os
import shutil

PATCHES = []


def patch(old, new, tag):
    PATCHES.append((old, new, tag))


# ------------------------------------------------------------------ P1 alloc
patch(
    "        else:\n"
    "            self.cuda_cast_buffer = self.cuda_transit_buffer\n"
    "\n"
    "        self.n_kv_pages = (q_len + self.page_size - 1) // self.page_size\n",
    "        else:\n"
    "            self.cuda_cast_buffer = self.cuda_transit_buffer\n"
    "\n"
    "        # [ICECACHE-DOUBLEBUF] ping-pong transit buffers; order with events\n"
    "        # instead of c2g_stream.synchronize().  See apply_double_buffer_patch.py.\n"
    "        self.double_buffer = bool(int(os.environ.get(\"ICECACHE_DOUBLE_BUFFER\", \"0\")))\n"
    "        self._db_slot = 0\n"
    "        self._last_slot = 0\n"
    "        self._db_ev_read = [None, None]\n"
    "        self._db_ev_cast = [None, None]\n"
    "        self._db_ev_used = [None, None]\n"
    "        if self.double_buffer:\n"
    "            if self.batch_layer_recall or self.n_prefetch_layers:\n"
    "                raise ValueError(\n"
    "                    \"ICECACHE_DOUBLE_BUFFER=1 is incompatible with batched \"\n"
    "                    \"layer recall and with layer prefetch\")\n"
    "            _cast_alias = (self.cuda_cast_buffer is self.cuda_transit_buffer)\n"
    "            if self.n_reuse_layers > 0:\n"
    "                _w = 2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages) * self.n_reuse_layers\n"
    "            else:\n"
    "                _w = 2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages)\n"
    "            self.cpu_transit_buffer = [\n"
    "                torch.empty([self.batch_size, _w, self.page_size * self.head_dim],\n"
    "                            **self._recall_fp, pin_memory=True) for _ in range(2)]\n"
    "            self.cuda_transit_buffer = [\n"
    "                torch.empty([self.batch_size, _w, self.page_size * self.head_dim],\n"
    "                            **self._fp, pin_memory=False) for _ in range(2)]\n"
    "            if _cast_alias:\n"
    "                self.cuda_cast_buffer = self.cuda_transit_buffer\n"
    "            else:\n"
    "                self.cuda_cast_buffer = [\n"
    "                    self.cuda_cast_buffer,\n"
    "                    torch.empty_like(self.cuda_cast_buffer)]\n"
    "\n"
    "        self.n_kv_pages = (q_len + self.page_size - 1) // self.page_size\n",
    "P1-alloc",
)

# ------------------------------------------------------- P2 recall: slot bind
patch(
    "        profile_stage = self._profile_is_measured_step()\n"
    "        gather_start = perf_counter() if profile_stage else None\n"
    "\n"
    "        n_transit_pages = torch.sum(nr).item()\n",
    "        profile_stage = self._profile_is_measured_step()\n"
    "        gather_start = perf_counter() if profile_stage else None\n"
    "\n"
    "        # [ICECACHE-DOUBLEBUF] bind this recall to a buffer slot\n"
    "        if self.double_buffer:\n"
    "            _p = self._db_slot\n"
    "            _ptb = self.cpu_transit_buffer[_p]\n"
    "            _ctb = self.cuda_transit_buffer[_p]\n"
    "            _cbuf = self.cuda_cast_buffer[_p]\n"
    "            _evr = self._db_ev_read[_p]\n"
    "            if _evr is not None:\n"
    "                _evr.synchronize()\n"
    "        else:\n"
    "            _p = 0\n"
    "            _ptb = self.cpu_transit_buffer\n"
    "            _ctb = self.cuda_transit_buffer\n"
    "            _cbuf = self.cuda_cast_buffer\n"
    "\n"
    "        n_transit_pages = torch.sum(nr).item()\n",
    "P2-recall-slot",
)

# ------------------------------------------- P3 recall: copy_to_buffer dest
patch(
    "            DCI.copy_to_buffer(self._src_address_buffer, ptr_dest=cast(self.cpu_transit_buffer[b].data_ptr(), c_void_p).value,\n",
    "            DCI.copy_to_buffer(self._src_address_buffer, ptr_dest=cast(_ptb[b].data_ptr(), c_void_p).value,\n",
    "P3-copybuf-dest",
)

# ------------------------------- P4 recall: event wait + H2D + cast + record
patch(
    "        with torch.cuda.stream(c2g_stream):\n"
    "            dst = self.cuda_transit_buffer[:, : 2 * n_transit_pages, :]\n"
    "            src = self.cpu_transit_buffer[:, : 2 * n_transit_pages, :]\n",
    "        with torch.cuda.stream(c2g_stream):\n"
    "            # [ICECACHE-DOUBLEBUF] slot reuse must wait for the consumer\n"
    "            if self.double_buffer and self._db_ev_used[_p] is not None:\n"
    "                c2g_stream.wait_event(self._db_ev_used[_p])\n"
    "            dst = _ctb[:, : 2 * n_transit_pages, :]\n"
    "            src = _ptb[:, : 2 * n_transit_pages, :]\n",
    "P4-h2d-wait",
)

patch(
    "            dst.copy_(src, non_blocking=True)\n"
    "            if diag_e1 is not None:\n"
    "                diag_e1.record(c2g_stream)\n"
    "\n"
    "            self.cuda_cast_buffer[:, : 2 * n_transit_pages, :].copy_(\n"
    "                dst, non_blocking=True\n"
    "            )\n",
    "            dst.copy_(src, non_blocking=True)\n"
    "            if diag_e1 is not None:\n"
    "                diag_e1.record(c2g_stream)\n"
    "            if self.double_buffer:\n"
    "                if self._db_ev_read[_p] is None:\n"
    "                    self._db_ev_read[_p] = torch.cuda.Event()\n"
    "                self._db_ev_read[_p].record(c2g_stream)\n"
    "\n"
    "            _cbuf[:, : 2 * n_transit_pages, :].copy_(\n"
    "                dst, non_blocking=True\n"
    "            )\n",
    "P5-h2d-record",
)

patch(
    "            if diag_e2 is not None:\n"
    "                diag_e2.record(c2g_stream)\n"
    "            if diag_e0 is not None:\n"
    "                self.diag_pending.append((diag_e0, diag_e1, diag_e2))\n"
    "                self.diag_event_budget -= 1\n",
    "            if diag_e2 is not None:\n"
    "                diag_e2.record(c2g_stream)\n"
    "            if diag_e0 is not None:\n"
    "                self.diag_pending.append((diag_e0, diag_e1, diag_e2))\n"
    "                self.diag_event_budget -= 1\n"
    "            if self.double_buffer:\n"
    "                if self._db_ev_cast[_p] is None:\n"
    "                    self._db_ev_cast[_p] = torch.cuda.Event()\n"
    "                self._db_ev_cast[_p].record(c2g_stream)\n"
    "                self._last_slot = _p\n"
    "                self._db_slot ^= 1\n",
    "P6-cast-record",
)

# --------------------------------------------------- P7 call site: skip sync
patch(
    "                c2g_stream.synchronize()\n"
    "                if recall_wait_start is not None:\n",
    "                # [ICECACHE-DOUBLEBUF] with two slots the consumer stream\n"
    "                # waits on an event, so the host does not block here\n"
    "                if not self.double_buffer:\n"
    "                    c2g_stream.synchronize()\n"
    "                if recall_wait_start is not None:\n",
    "P7-skip-sync",
)

# ------------------------------------------------------- P8 scatter ordering
patch(
    "    def scatter_pages(self, layer_idx, eids, nr):\n"
    "        transit = self.cuda_cast_buffer\n"
    "        if self.batch_layer_recall and layer_idx in self._batched_recall_slices:\n"
    "            start, width = self._batched_recall_slices[layer_idx]\n"
    "            transit = transit[:, start:start + width, :]\n"
    "        _cpp.scatter_pages(transit,\n"
    "                           self.kv_caches[layer_idx].pool.buffer,\n"
    "                           eids, nr)\n",
    "    def scatter_pages(self, layer_idx, eids, nr):\n"
    "        if self.double_buffer:\n"
    "            # [ICECACHE-DOUBLEBUF] consume slot _last_slot, ordered by event\n"
    "            _p = self._last_slot\n"
    "            _cur = torch.cuda.current_stream()\n"
    "            if self._db_ev_cast[_p] is not None:\n"
    "                _cur.wait_event(self._db_ev_cast[_p])\n"
    "            _cpp.scatter_pages(self.cuda_cast_buffer[_p],\n"
    "                               self.kv_caches[layer_idx].pool.buffer,\n"
    "                               eids, nr)\n"
    "            if self._db_ev_used[_p] is None:\n"
    "                self._db_ev_used[_p] = torch.cuda.Event()\n"
    "            self._db_ev_used[_p].record(_cur)\n"
    "            return\n"
    "        transit = self.cuda_cast_buffer\n"
    "        if self.batch_layer_recall and layer_idx in self._batched_recall_slices:\n"
    "            start, width = self._batched_recall_slices[layer_idx]\n"
    "            transit = transit[:, start:start + width, :]\n"
    "        _cpp.scatter_pages(transit,\n"
    "                           self.kv_caches[layer_idx].pool.buffer,\n"
    "                           eids, nr)\n",
    "P8-scatter-order",
)


def main():
    if len(sys.argv) != 2:
        print("usage: apply_double_buffer_patch.py /path/to/infer_state.py")
        return 2
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "ICECACHE-DOUBLEBUF" in src:
        print("ALREADY_PATCHED")
        return 3
    for old, new, tag in PATCHES:
        n = src.count(old)
        if n != 1:
            print("ANCHOR_FAIL %s: %d occurrences" % (tag, n))
            return 4
        src = src.replace(old, new, 1)
        print("OK %s" % tag)
    backup = path + ".bak_predoublebuf"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print("backup -> %s" % backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("PATCH_APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
