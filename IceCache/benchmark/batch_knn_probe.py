"""Instrumented page_scan run: split the decode retrieval path into stages.

Wraps `page_scan_compare.main()` so the model, prompts, seeds, generation
settings and timing are identical to the paired harness; the only difference is
that live methods are monkey-patched with copies of themselves carrying timers.
The brief warns (twice-observed in this project) that isolated component probes
mislead, so everything here is driven from the real harness.

Two scopes, selected by `BATCH_KNN_PROBE_SCOPE`:

  scan    (default) `PageScan._query_device` split into its own ops. This is
          the function the exploration names.
  path    the enclosing decode retrieval path: `InferState.estimate_select_recall`
          split into `_DCI_query` / `recall` / `c2g_stream.synchronize` / the
          page_valid_entries H2D block / python residual. 34 of these run per
          decode token (one per offload layer), so it is the larger frame.

`BATCH_KNN_PROBE_SYNC` (scan scope): 1 (default) syncs after every stage, so
each number is that stage's own device+launch time -- a true split, at the price
of extra syncs; 0 uses enqueue-only timers, so the stage numbers are pure CPU
launch overhead and `total` is the real unsynchronised cost.

`pre_drain` is the queue already outstanding when the call starts, measured by
draining *before* the first timer. Without it the first stage (and, in the real
unsynced path, the trailing D2H) absorbs the tail of the preceding decode
attention kernels -- which is the standing suspicion about why
`query_seconds` p50/layer reads as high as it does. Measured at 0.036 ms, i.e.
that suspicion is wrong for the p50; the scan really does cost ~0.9 ms.
"""

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "source"))

SYNC = os.environ.get("BATCH_KNN_PROBE_SYNC", "1") == "1"
SCOPE = os.environ.get("BATCH_KNN_PROBE_SCOPE", "scan")

SCAN_RECORDS = []
PATH_RECORDS = []
STATE = {"frame": None}
CNT = Counter()


def _pct(records, key):
    vals = [r[key] * 1e3 for r in records if key in r]
    return statistics.median(vals), float(np.percentile(vals, 90)), statistics.fmean(vals), sum(vals)


def _table(records, keys, title, denom_key):
    if not records:
        print(f"PROBE[{title}]: no calls recorded", flush=True)
        return
    denom = sum(r[denom_key] for r in records) * 1e3
    print(f"\n===== {title} (n={len(records)}) =====", flush=True)
    r0 = records[0]
    print(f"first call: " + " ".join(f"{k}={r0[k]}" for k in r0
                                     if isinstance(r0[k], (int, str, bool))), flush=True)
    print(f"{'stage':16s} {'median ms':>10s} {'p90 ms':>9s} {'mean ms':>9s} "
          f"{'sum ms':>10s} {'% of total':>10s}", flush=True)
    for k in keys:
        med, p90, mean, sm = _pct(records, k)
        print(f"{k:16s} {med:10.3f} {p90:9.3f} {mean:9.3f} {sm:10.1f} "
              f"{100 * sm / denom:9.1f}%", flush=True)


# ------------------------------------------------------------------ scan scope

SCAN_STAGES = ("h2d", "bmm", "mask", "topk", "dedup", "d2h")
# The dedup is ~10 tiny ops; this splits it into its individual kernels so the
# 59% can be attributed to one of them rather than to "the dedup".
DEDUP_OPS = ("d_flat", "d_arange", "d_full", "d_screduce", "d_gather",
             "d_cumsum", "d_rows", "d_zeros", "d_write")


def install_scan_probe():
    from icecache.page_scan import PageScan

    perf = time.perf_counter

    def timed_query_device(self, q, budget, ratio):
        H = self.n_kv_heads
        t = Counter()
        sync = torch.cuda.synchronize

        t0 = perf()
        sync(self.device)
        t["pre_drain"] += perf() - t0

        def tick(stage, fn):
            t0 = perf()
            out = fn()
            if SYNC:
                sync(self.device)
            t[stage] += perf() - t0
            return out

        def _h2d():
            if isinstance(q, torch.Tensor):
                return q.detach().to(self.device, torch.float32).reshape(
                    H, ratio, self.head_dim)
            return torch.as_tensor(np.ascontiguousarray(q, dtype=np.float32),
                                   device=self.device).reshape(
                H, ratio, self.head_dim)

        qr = tick("h2d", _h2d)
        with torch.no_grad():
            scores = tick("bmm", lambda: torch.bmm(
                qr, self._reps_t.transpose(1, 2)))
            scores = tick("mask", lambda: scores.masked_fill(
                self._bias_t.unsqueeze(1) != 0, float("-inf")))
            top = tick("topk", lambda: scores.topk(budget, dim=-1).indices)

            flat = tick("d_flat", lambda: top.transpose(1, 2).reshape(H, -1))
            ar = tick("d_arange", lambda: torch.arange(
                flat.shape[1], device=self.device).expand_as(flat))
            full = tick("d_full", lambda: torch.full(
                (H, self.n_pages), flat.shape[1], dtype=torch.long,
                device=self.device))
            first = tick("d_screduce", lambda: full.scatter_reduce_(
                1, flat, ar, reduce="amin"))
            keep = tick("d_gather", lambda: first.gather(1, flat) == ar)
            pos = tick("d_cumsum", lambda: keep.cumsum(1) - 1)
            rows = tick("d_rows", lambda: torch.arange(
                H, device=self.device).unsqueeze(1).expand_as(flat))
            out = tick("d_zeros", lambda: torch.zeros(
                (H, budget), dtype=torch.long, device=self.device))
            m = keep & (pos < budget)
            tick("d_write", lambda: out.__setitem__(
                (rows[m], pos[m]), flat[m]))
            t["dedup"] += sum(t[o] for o in DEDUP_OPS)
            res = tick("d2h", lambda: out.to(torch.int32).cpu().numpy())

        rec = {"H": H, "ratio": ratio, "budget": budget,
               "n_pages": int(self._reps_t.shape[1]),
               "n_built": int(self._n_built.min()),
               "q_was_cuda": isinstance(q, torch.Tensor) and q.is_cuda,
               "pre_drain": t["pre_drain"]}
        rec.update({s: t[s] for s in SCAN_STAGES + DEDUP_OPS})
        rec["total"] = sum(t[s] for s in SCAN_STAGES)
        SCAN_RECORDS.append(rec)
        return res

    PageScan._query_device = timed_query_device


# ------------------------------------------------------------------ path scope

PATH_STAGES = ("query", "recall", "valid_entries", "sync_time", "residual")


def install_path_probe():
    """Attribute `estimate_select_recall` to the work it does inside.

    `_query_device` is left alone here (if the scan probe is also installed it
    is wrapped in enqueue-only mode by the caller) so the outer numbers are the
    unperturbed ones.
    """
    import icecache.infer_state as istate
    from time import perf_counter as perf

    InferState = istate.InferState

    def frame():
        return STATE["frame"]

    # `recall` stages the CPU->GPU page copy for one layer and returns without
    # draining -- its cost lands in the c2g_stream sync, not here.
    orig_recall = InferState.recall

    def timed_recall(self, layer_idx, b, rids, nr):
        f = frame()
        if f is None:
            return orig_recall(self, layer_idx, b, rids, nr)
        t0 = perf()
        try:
            return orig_recall(self, layer_idx, b, rids, nr)
        finally:
            f["recall"] += perf() - t0

    # Every `estimate_select_recall` ends in one of these; 34 per decode token.
    orig_sync = torch.cuda.Stream.synchronize

    def timed_stream_sync(self, *a, **k):
        f = frame()
        if f is None:
            return orig_sync(self, *a, **k)
        t0 = perf()
        try:
            return orig_sync(self, *a, **k)
        finally:
            f["sync_time"] += perf() - t0

    orig_query = InferState._DCI_query

    def timed_dci_query(self, b, cur_id, query_states):
        f = frame()
        if f is None:
            return orig_query(self, b, cur_id, query_states)
        t0 = perf()
        try:
            return orig_query(self, b, cur_id, query_states)
        finally:
            f["query"] += perf() - t0

    # The page_valid_entries block is `get_valid_entries` (numpy) + a
    # `torch.tensor(...)` H2D + a `.T` strided write into the layer's buffer; it
    # has no clean wrap point, so it is the residual.
    def timed_estimate_select_recall(self, layer_idx, query_states):
        kvc = self.kv_caches[layer_idx]
        f = Counter() if (kvc.n_real_pages == kvc.budget and self.use_dci) else None
        prev, STATE["frame"] = STATE["frame"], f
        t0 = perf()
        try:
            return orig_estimate(self, layer_idx, query_states)
        finally:
            elapsed = perf() - t0
            STATE["frame"] = prev
            if f is not None:
                f["total"] = elapsed
                f["valid_entries"] = elapsed - f["query"] - f["recall"] - f["sync_time"]
                f["layer"] = layer_idx
                PATH_RECORDS.append(dict(f))

    orig_estimate = InferState.estimate_select_recall

    InferState.recall = timed_recall
    torch.cuda.Stream.synchronize = timed_stream_sync
    InferState._DCI_query = timed_dci_query
    InferState.estimate_select_recall = timed_estimate_select_recall


# -------------------------------------------------------------------- reporting

def report(path):
    out = {"sync": SYNC, "scope": SCOPE}
    if SCOPE in ("scan", "both"):
        _table(SCAN_RECORDS, ("pre_drain",) + SCAN_STAGES, "scan scope", "total")
        _table([r for r in SCAN_RECORDS], DEDUP_OPS, "dedup sub-split", "dedup")
        out["scan"] = SCAN_RECORDS
    if SCOPE in ("path", "both"):
        _table(PATH_RECORDS, PATH_STAGES, "path scope (estimate_select_recall)", "total")
        per_layer = Counter()
        for r in PATH_RECORDS:
            per_layer[r["layer"]] += r["total"]
        print("\npath scope per layer (sum s over the run): " +
              ", ".join(f"L{k}={v:.2f}" for k, v in sorted(per_layer.items())), flush=True)
        out["path"] = PATH_RECORDS
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"PROBE: raw -> {path}", flush=True)


def main():
    import page_scan_compare

    argv = sys.argv[1:]
    probe_out = None
    if "--probe-out" in argv:
        i = argv.index("--probe-out")
        probe_out = argv[i + 1]
        del argv[i:i + 2]
    if SCOPE in ("scan", "both"):
        install_scan_probe()
    if SCOPE in ("path", "both"):
        install_path_probe()
    sys.argv = [sys.argv[0]] + argv
    try:
        page_scan_compare.main()
    finally:
        if probe_out:
            report(probe_out)


if __name__ == "__main__":
    main()
