#!/usr/bin/env python3
"""ICECACHE_DOUBLE_BUFFER fix v2: per-layer slot bookkeeping, correct timing.

Previous fix v1 was wrong: recall() did
    _p = self._db_used_slot.get(layer_idx, 0)
    self._db_used_slot[layer_idx] = (_p + 1) % 2
i.e. stored the *next* slot before scatter read it, so scatter consumed the
slot that the *next* recall would use -> wrong data -> corrupted KV ->
model never emits EOS -> 2x slower (measured: GPU 0%, CPU 2800%, no output).

Correct semantics (DB=1, PF=0 -> same-thread sync order):
  - recall() binds slot _p = self._db_slot_for.get(layer_idx, 0)
  - after the slot is safely recorded for this layer's recall, increment:
        self._db_slot_for[layer_idx] = (_p + 1) % 2
  - scatter_pages() reads _p = self._db_slot_for.get(layer_idx, 0)
Because modeling.py calls estimate_select_recall() (which ends with recall())
synchronously before scatter_pages() for the same layer, the per-layer value
is always written by this layer's own recall and consumed by the same
layer's scatter, never clobbered by another layer.  Across-token reuse
(cross_token_reused path skips recall) reads the previous token's slot value,
which is correct because the KV content didn't change.

The increment must happen in recall() *after* the slot is bound but the key
point vs v1: scatter reads the *stored* value (post-increment of the
previous recall, i.e. this recall's slot), because recall stored (_p+1)
at the END of the *previous* recall of this layer.  Wait, no: with this
scheme:
  recall #1: _p = get() = 0; store 1; uses slot 0
  recall #2: _p = get() = 1; store 0; uses slot 1
  scatter after recall #1: get() = 1  <- WRONG (should consume 0)
So storing post-increment makes scatter read the next slot.  The correct
storage is to store the *used* slot, and advance a separate counter only to
decide the *next* recall's slot:

  recall() binds _p via a per-layer cursor, stores the USED slot:
    cursor = self._db_cursor.get(layer_idx, 0)
    _p = cursor % 2
    self._db_cursor[layer_idx] = cursor + 1
    self._db_used_slot[layer_idx] = _p
  scatter_pages() reads self._db_used_slot.get(layer_idx, 0).
This decouples "which slot does the next recall take" (cursor) from "which
slot does the last recall of this layer put data in" (used_slot).  scatter
always consumes the used slot of THIS layer's most recent recall.
"""
import sys

OLD_P1 = (
    "        self._db_slot = 0\n"
    "        self._last_slot = 0\n"
)
NEW_P1 = (
    "        self._db_cursor = {}\n"
    "        self._db_used_slot = {}\n"
)

OLD_P2 = (
    "        if self.double_buffer:\n"
    "            _p = self._db_slot\n"
    "            _ptb = self.cpu_transit_buffer[_p]\n"
    "            _ctb = self.cuda_transit_buffer[_p]\n"
    "            _cbuf = self.cuda_cast_buffer[_p]\n"
    "            _evr = self._db_ev_read[_p]\n"
    "            if _evr is not None:\n"
    "                _evr.synchronize()\n"
)
NEW_P2 = (
    "        if self.double_buffer:\n"
    "            _cursor = self._db_cursor.get(layer_idx, 0)\n"
    "            _p = _cursor % 2\n"
    "            self._db_cursor[layer_idx] = _cursor + 1\n"
    "            self._db_used_slot[layer_idx] = _p\n"
    "            _ptb = self.cpu_transit_buffer[_p]\n"
    "            _ctb = self.cuda_transit_buffer[_p]\n"
    "            _cbuf = self.cuda_cast_buffer[_p]\n"
    "            _evr = self._db_ev_read[_p]\n"
    "            if _evr is not None:\n"
    "                _evr.synchronize()\n"
)

OLD_P6 = (
    "            if self.double_buffer:\n"
    "                if self._db_ev_cast[_p] is None:\n"
    "                    self._db_ev_cast[_p] = torch.cuda.Event()\n"
    "                self._db_ev_cast[_p].record(c2g_stream)\n"
    "                self._last_slot = _p\n"
    "                self._db_slot ^= 1\n"
)
NEW_P6 = (
    "            if self.double_buffer:\n"
    "                if self._db_ev_cast[_p] is None:\n"
    "                    self._db_ev_cast[_p] = torch.cuda.Event()\n"
    "                self._db_ev_cast[_p].record(c2g_stream)\n"
)

OLD_P8 = (
    "        if self.double_buffer:\n"
    "            # [ICECACHE-DOUBLEBUF] consume slot _last_slot, ordered by event\n"
    "            _p = self._last_slot\n"
)
NEW_P8 = (
    "        if self.double_buffer:\n"
    "            # [ICECACHE-DOUBLEBUF] consume this layer's used slot\n"
    "            _p = self._db_used_slot.get(layer_idx, 0)\n"
)


def main():
    if len(sys.argv) != 2:
        print("usage: fix_double_buffer_slot.py /path/to/infer_state.py")
        return 2
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "ICECACHE-DOUBLEBUF" not in src:
        print("NOT_PATCHED")
        return 5
    if "ICECACHE-DOUBLEBUF-FIX2" in src:
        print("ALREADY_FIXED")
        return 3
    for old, new, tag in ((OLD_P1, NEW_P1, "P1-init"),
                          (OLD_P2, NEW_P2, "P2-slot"),
                          (OLD_P6, NEW_P6, "P6-record"),
                          (OLD_P8, NEW_P8, "P8-slot")):
        n = src.count(old)
        if n != 1:
            print("ANCHOR_FAIL %s: %d" % (tag, n))
            return 4
        src = src.replace(old, new, 1)
        print("OK %s" % tag)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("FIX2_APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())