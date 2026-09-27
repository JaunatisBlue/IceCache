#!/usr/bin/env python3
"""Split the per-output-token decode interval of a page_scan_compare run into four buckets,
each measured as a DEVICE interval with torch.cuda.Event.

Lives at `benchmark/page_scan_breakdown.py` and is run from that directory, with the
same argv `page_scan_compare.py` would be given -- see USAGE at the end of this
docstring.

WHY THIS SHAPE
--------------
`benchmark/batch_knn_probe.py` established the pattern this file follows: do not
re-implement the harness, monkey-patch copies of the live methods and call
`page_scan_compare.main()` **in process**, so the model, prompt, tokenizer, seed,
generation settings and the harness's own timing stay bit-identical to a normal
run. Everything below is a read-only wrapper; nothing under the repo is touched.

The one structural difference from batch_knn_probe.py is that this probe does NOT
own the generation loop -- neither file does. The harness builds its own
`TokenTimer` inside `run_backend` (`page_scan_compare.py:281`) and passes it as
the only stopping criterion, so the per-token counter has to be installed by
patching that class -- see `install_token_timer`, which keeps the harness's own
per-token `torch.cuda.synchronize()` as the SOLE sync.

WHY EVENTS, NOT perf_counter
----------------------------
An earlier revision of this file timed the wrappers on the host with
`time.perf_counter()`. That is wrong for this workload, for two separate reasons:

  * `PageScan.query` ends in `.cpu().numpy()` (page_scan.py:729), which DRAINS THE
    DEFAULT STREAM. So a host clock around the scan measures the scan PLUS every
    kernel already enqueued before it -- the previous layer's `decode_sdpa`, its
    MLP, the projections -- i.e. it is inflated by transformer compute that is not
    retrieval. The docstring of `_DCI_query`'s page_scan branch says the same
    thing in the source itself ("that result copy is an implicit stream
    synchronisation"), so this is not an inference: the old Query number was a
    host-critical-path number that borrowed foreign device time.
  * `decode_sdpa` (infer_state.py:2024) and the MLP are pure launches with no host
    wait, so host-timing them yields launch overhead (a few percent of the real
    cost), not their device time.

A CUDA event pair fixes both. `record()` enqueues a timestamp command at the BACK
of the stream's queue, so:
  * the start event completes only after everything already queued, i.e. foreign
    work *delays* the start but is NOT inside `[start, end]` -- it is excluded,
    not absorbed;
  * the end event is queued after the wrapped call's own kernels, so
    `start.elapsed_time(end)` is exactly the device interval of those kernels,
    however long the host takes to get around to enqueuing the end event.
Neither event SYNCHRONISES anything, so the harness's one-sync-per-token structure
is preserved. That is NOT the same as "the harness's host timings are untouched",
which an earlier revision of this docstring claimed and which the measurements
falsify: the wrapper's own per-call host work (two `torch.cuda.Event`
constructions, two `record` enqueues, the accumulator, and -- for the wrappers that
resolve a stream -- a `torch.cuda.current_stream()`) is real wall time inside the
token interval. This decode is HOST-BOUND, so that time is exposed rather than
hidden under device work: the same row measures `tpot_s` = 112.2-113.6 ms with the
probe against a clean 4-pass mean of 89.53 ms, i.e. +26%, while its device frame
`d:attn_frame` stays at 88.9-90.2 ms, within 1% of clean. So a probe row's own
`ttft_s` / `tt2t_s` / `tpot_s` must NOT be quoted as clean numbers. The wrappers'
host cost is MEASURED per run rather than assumed -- see the `wrapper_host` block in
the report and the `probe overhead (MEASURED ...)` line -- and the shares to plot are
the frame-denominated ones, see WHICH SHARES TO PLOT below.

THE FOUR BUCKETS, AND THE STREAM EACH ONE IS RECORDED ON
--------------------------------------------------------
Streams were read off the source (working tree at HEAD `57c0049` plus the
uncommitted edits present at 2026-09-26 10:00). All file:line references are
working-tree line numbers; they drift, so re-check the named SYMBOL.

  Query     `InferState._DCI_query` (infer_state.py:1449), called from
            `estimate_select_recall` (infer_state.py:1766); for the page_scan
            backend it forwards to `PageScan.query` (page_scan.py:591) ->
            `_query_device` (page_scan.py:696), whose result copy is the
            `.cpu().numpy()` at page_scan.py:729.
            What the pair covers: the scan (`bmm` / mask / `topk` / dedup), the
            H2D of the selected page ids in `_apply_selected_pages`, and the D2H
            of the scan result. It does NOT contain an H2D of `q`: infer_state.py:1763
            passes `q` resident on this path and `_query_device` casts in place on
            the device (only the DCI backend pays the `.cpu()` round trip).
            STREAM = the CURRENT stream (`torch.cuda.current_stream()`).
            Proof: neither `_DCI_query` nor `PageScan.query`/`_query_device`
            contains a `with torch.cuda.stream(...)`; the scan is plain
            `torch.bmm` / mask / `topk` / `scatter` on device tensors, and the
            production call site passes `q` already resident (infer_state.py:1763
            skips the DCI path's `.cpu()`), so every op lands on whatever stream
            the caller had current. With `--n-reuse-layers 0` there is no prefetch
            thread either: modeling.py:318 calls `estimate_select_recall`
            synchronously on the main thread, whose current stream is the default
            stream.

  Loading   `InferState.recall` (infer_state.py:1660): the host page gather and
            `DCI.copy_to_buffer` (infer_state.py:1718) followed by the pageable
            H2D + D2D cast (infer_state.py:1720-1724).
            STREAM = the c2g stream, NOT the default stream. Proof: the copies sit
            inside `with torch.cuda.stream(c2g_stream):` at infer_state.py:1716,
            and the stream is picked at infer_state.py:1687-1693 (thread-local
            override) with the caller's identical pick at infer_state.py:1738-1744.
            `_c2g_stream` below mirrors that selection exactly, and the report
            prints the resolved stream handle per bucket so a silent fallback to
            the default stream (which would make Loading read ~0) is visible.
            CAVEAT, and it is a real one: the event pair brackets the WHOLE
            `recall`, and the H2D is only enqueued after the host-side staging
            (numpy address gather, the deferred-write join, the pinned-buffer
            memcpy inside `copy_to_buffer`). The start event therefore completes
            at call entry and the interval contains that host gap as stream idle
            time. That is why the report carries a nested `Loading.copy` pair,
            recorded on the same stream from the moment `copy_to_buffer` returns
            (the H2D enqueue point) to the end of `recall`: it is the device copy
            alone, and `Loading - Loading.copy` is the host staging that the
            bucket also contains.

  Decoding  `InferState.decode_sdpa` (infer_state.py:2024), called only from
            modeling.py:355, and every decoder layer's `*.mlp.forward`.
            STREAM = the CURRENT stream, both. Proof: `decode_sdpa` forwards to
            `decode_handler_tab[...].forward` with no stream switch; the MLP is
            invoked by the stock `LlamaDecoderLayer.forward` right after
            `self.self_attn(...)`, i.e. after the patched attention module has
            returned, running stock `nn.Linear` ops on the current stream. Both are
            siblings inside the layer, so they cannot overlap each other.

  Others    The residual: the harness host interval minus the three device
            buckets. It absorbs the q/k/v and o projections, rope, norms,
            `append_paged_kv_cache`, `scatter_pages`, `_prepare_decode`'s rollover,
            lm_head, sampling, host Python in `generate`, and the probe's own
            event-record overhead.

DOUBLE COUNTING AND OVERLAP: WHAT WAS CHECKED
---------------------------------------------
  * The three device buckets are disjoint in STREAM-time for the default-stream
    members: Query, `decode_sdpa` and the MLP are sequential on one stream, so
    their event intervals are sequential (adjacent, non-overlapping) segments of
    that stream's timeline -- unless some other device work was enqueued between
    them by a third party, which would then be attributed to whoever's interval
    spans it. The nested `d:` diagnostics show where such work is.
  * `Loading` is on ANOTHER stream and therefore CAN overlap the default stream's
    segments, so `Query + Loading + Decoding` is not guaranteed to be bounded by
    the token interval. That is the one structural reason `Others` can legitimately
    go negative -- see NEGATIVE OTHERS below. It is reported, not hidden.
  * `recall` has exactly ONE caller, infer_state.py:1798, inside
    `estimate_select_recall` -- so it cannot be counted twice.
  * `_DCI_query` -> `_apply_selected_pages` (infer_state.py:1546) is a child of
    Query, recorded as `Query.select` but never added to Query.
  * `PageScan.query` / `DCI.query` are children of Query, recorded as
    `Query.scan` / `Query.tree`, never added.
  * `Loading.copy` is a child of Loading (same stream, nested pair), never added.
  * Every other name in the accumulator is prefixed `d:` and is a NESTED frame or
    a leaf that is deliberately outside the partition. The merge reads only the
    names in `BUCKET_LEAVES`.

NEGATIVE OTHERS
---------------
`Others = interval - Query - Loading - Decoding` is computed per token and CAN be
negative, and the output says so loudly rather than printing a stacked share line
that renders as "Query 300% / Others -200%". Mechanism: Loading runs on
`c2g_stream` while the default stream may still be draining the previous layer's
MLP/sdpa, so the same wall-clock window is billed to two buckets at once, and the
same token's wall time is not an upper bound on the sum. A negative `Others` means
a 100%-stacked bar is the wrong presentation for that row: read the per-bucket ms,
not the shares. Where `Others >= 0` for every token the shares are printed
normally, and the report says which case it is.

WHICH SHARES TO PLOT
--------------------
Two share sets are emitted, on two different denominators, and they are NOT
interchangeable:

  * `shares_pct`, printed as "share of the mean interval". The denominator is the
    harness's own token interval (`marks[k] - marks[k-1]`, the quantity `tpot_s` is
    built from), and that interval CONTAINS the probe's own host cost -- every
    `torch.cuda.Event` construction, every `record()` enqueue, every accumulator
    call. It is therefore INFLATED: the same row's host `tpot_s` is 112.2-113.6 ms
    against the clean 4-pass mean 89.53 ms (+26%). Do not plot this set.
  * `frame_shares_pct`, printed as "share of the device frame d:attn_frame". The
    denominator is a CUDA-event interval (the sum of the per-call `d:attn_frame`
    device intervals inside the token). The three bucket numerators and this
    denominator are all device intervals, so the probe's own host cost cannot
    inflate either one. It is the INFLATION-FREE denominator, and the in-run
    evidence that it is clean is that this frame reads 88.9-90.2 ms in the same
    probe runs whose host interval is +26%: the device work is unperturbed even
    though the host timing is not. PLOT THIS ONE. That is a control, not an
    assumption: how much of the probe's own host cost is actually charged inside
    this interval -- the amount that could inflate this denominator -- is measured
    per run and printed as the inside/outside split (see `wrapper_host`).

The report prints both, each labelled with the denominator it uses, and carries
both in the JSON (`means.shares_pct` and `means.frame_shares_pct`).
`frame_shares_pct` is absent when `d:attn_frame` did not run or one of its event
pairs was unreadable, in which case only the inflated set exists and the report says
so rather than silently substituting one for the other.

PERTURBATION
------------
No sync is added; the harness's own per-token `torch.cuda.synchronize()` remains
the only one, and `ttft_s` / `tt2t_s` / `tpot_s` on the row are still computed
from the harness's own timestamps. But event pairs are not free: each wrapped call
costs two `torch.cuda.Event` allocations, two `record` enqueues (host) and two
timestamp commands on the device. At ~11 wrapped names x ~30 retrieval layers plus
one pair per layer for `decode_sdpa` and the MLP, that is a few hundred records
per token. HOW much host time that costs is measured, not bounded by argument: the
wrappers bracket their OWN work with `perf_counter`, and the report prints the
result per bucket per token (`row["wrapper_host"]`, and the `probe overhead
(MEASURED ...)` line), split into the part charged INSIDE the `d:attn_frame`
interval and the part outside it -- because that split is exactly what decides
whether the frame denominator is itself inflated. The device-side cost of a record
lands in whichever bucket's interval spans it. The diagnostic `d:` names can be
deleted from `install_path_probe` to shrink the whole thing.

USAGE
-----
    cd IceCache/benchmark
    python page_scan_breakdown.py <page_scan_compare argv...> --probe-out out.json

i.e. run it where and how `page_scan_compare.py` is run, adding `--probe-out
<path>`; the rest of the argv (--model, --prompt-file, --output, --gpu, --backends,
--threads, --page-*, --max-new-tokens, --max-input-tokens, --n-reuse-layers) is
forwarded verbatim, so the row it writes is the row the harness would have written,
plus the probe's own overhead. The flags must match the arm being decomposed -- the
breakdown is only meaningful for the configuration that was actually run, and the
harness default for `--max-input-tokens` (32760) silently truncates a 36k prompt.

`--probe-out` is stripped from `sys.argv` before `page_scan_compare.main()` runs,
so the harness's own argv (which it records in the row, `page_scan_compare.py:313`)
is unchanged apart from that flag. Everything else is passed straight through.

The report also carries the harness's `retrieval_stats` (previously dropped) and
prints the probe's per-call Query device time next to the harness's own
`query.page_scan.p50_ms` for the same region -- the cheapest detector that the
Query interval is still absorbing foreign device time, since the harness timer is
a host timer INSIDE the same call and must therefore be at least as large as the
device-only interval.
"""

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path


# --------------------------------------------------------------------------- repo
# The root is resolved from THIS file and the tree's own `source/` is inserted at
# the FRONT of sys.path, so the PYTHON tree a run executes is normally the one this
# script lives in. That matters here: this project has already had a bare worktree
# run silently execute the MAIN checkout, because `icecache` / `icecache_cpp` are
# editable installs whose finder runs after PathFinder, so `import icecache` from
# elsewhere resolves to main with no error at all.
#
# That guarantee is narrower than "the tree the script lives in", and there are two
# verified ways it does not hold. `_check_tree()` below warns about both, before any
# measurement runs:
#   * `ICECACHE_REPO` overrides the root, so the tree executed is then NOT the one
#     this file lives in;
#   * `source/icecache_cpp.*` is gitignored (.gitignore:1), so a fresh worktree has
#     no compiled artifact on its sys.path and the editable finder resolves
#     `icecache_cpp` to the MAIN checkout's `.so`. A worktree run would then import
#     the worktree's Python package and main's compiled kernels -- and this project
#     has shown a rebuilt DCI `.so` is a DIFFERENT binary that does not reproduce
#     the installed one, so such a breakdown is byte-indistinguishable from one
#     taken before the swap. That is why the provenance block records the resolved
#     `icecache_cpp` / `dciknn` / `dciknn._dci` paths and sizes.
HERE = Path(__file__).resolve().parent          # <repo>/IceCache/benchmark
REPO = Path(os.environ.get("ICECACHE_REPO", HERE.parent)).resolve()
_BENCH = REPO / "benchmark"
_SRC = REPO / "source"
for _p in (_BENCH, _SRC):
    if not _p.is_dir():
        raise SystemExit(f"page_scan_breakdown: not a checkout: {_p} (set ICECACHE_REPO)")
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_BENCH))

import torch  # noqa: E402

import page_scan_compare as psc  # noqa: E402
import icecache.infer_state as istate  # noqa: E402
import icecache.adapter as icecache_adapter  # noqa: E402
from icecache.infer_state import InferState  # noqa: E402
from icecache.page_scan import PageScan  # noqa: E402
from icecache.adapter import modeling  # noqa: E402
from transformers import StoppingCriteria  # noqa: E402


# ------------------------------------------------------------------ tree check
def _check_tree():
    """Warn loudly, BEFORE any measurement, if the tree about to run is not ours.

    The repo comment above states the guarantee this checks: the PYTHON tree is
    resolved from this file, but `ICECACHE_REPO` can override it and the compiled
    `icecache_cpp` artifact can still come from another checkout, because
    `source/icecache_cpp.*` is gitignored (.gitignore:1) so a fresh worktree has
    none on its sys.path and the editable finder falls back to main's. Neither is
    fatal, so this warns rather than aborts -- but the warning goes out before
    `page_scan_compare.main()`, so it is in the captured output and impossible to
    miss. `_warn` is defined below, in the accumulator section.
    """
    env_repo = os.environ.get("ICECACHE_REPO")
    if env_repo:
        env_path = Path(env_repo).resolve()
        if env_path != HERE.parent:
            _warn(f"ICECACHE_REPO overrides this file's own tree: the run will "
                  f"execute {env_path} (ICECACHE_REPO={env_repo}), NOT "
                  f"{HERE.parent}, the tree page_scan_breakdown.py lives in.")
    cpp = getattr(sys.modules.get("icecache_cpp"), "__file__", None)
    if not cpp:
        _warn("icecache_cpp has no __file__ (not imported, or a namespace package): "
              "the compiled kernels the numbers depend on cannot be attributed to "
              "any tree.")
        return
    try:
        cpp_under = Path(cpp).resolve().is_relative_to(REPO)
    except AttributeError:                       # pragma: no cover - py<3.9
        cpp_under = str(Path(cpp).resolve()).startswith(str(REPO) + os.sep)
    if not cpp_under:
        _warn(f"icecache_cpp resolved OUTSIDE the tree being run: {cpp} is not under "
              f"{REPO}. source/icecache_cpp.* is gitignored, so a fresh worktree has "
              f"no compiled artifact on sys.path and the editable finder falls back "
              f"to the MAIN checkout's kernels -- this run would use {REPO}'s Python "
              f"package with another checkout's compiled kernels, and a rebuilt "
              f"icecache_cpp/DCI .so is a different binary. Numbers from this run "
              f"are not comparable across a .so swap.")


# ---------------------------------------------------------------------- buckets
# The partition. Only these names are summed into the four buckets. `Others` is
# the residual and is never wrapped.
BUCKET_LEAVES = {
    "Query": ("Query",),                              # device, current stream
    "Loading": ("Loading",),                          # device, c2g stream
    "Decoding": ("Decoding.sdpa", "Decoding.mlp"),    # device, current stream
}
BUCKETS = tuple(BUCKET_LEAVES)   # + "Others" = the residual, which is never wrapped

# The stream policy, in one place. `STREAM_CURRENT` is a sentinel meaning "record
# on whatever stream the calling thread has current", which is what a wrapped call
# that contains no `with torch.cuda.stream(...)` will use -- see the module
# docstring for the per-bucket proof. `Loading` is the one that is NOT this.
STREAM_CURRENT = None

# What each name is, so the JSON explains itself. "NESTED" means the duration is
# INSIDE one of the bucket frames (or overlaps another `d:` entry) and must not be
# added to anything; it is reported only so the residual can be read. "HOST" means
# the number is a host perf_counter, not a device interval.
NAME_NOTES = {
    "Query": "BUCKET (device, current stream): InferState._DCI_query -- both backends",
    "Loading": "BUCKET (device, c2g stream): InferState.recall -- host staging + H2D + D2D cast",
    "Decoding.sdpa": "BUCKET (device, current stream): InferState.decode_sdpa",
    "Decoding.mlp": "BUCKET (device, current stream): <layer>.mlp.forward",
    "Loading.copy": "NESTED in Loading, same c2g stream: copy_to_buffer->return to end of recall = the H2D+D2D alone",
    "Query.scan": "NESTED in Query: PageScan.query (the GPU scan; host-path when _reps_t is None)",
    "Query.tree": "NESTED in Query: dciknn DCI.query (host-side; device interval ~0 by construction)",
    "Query.select": "NESTED in Query: _apply_selected_pages (page diff / address staging)",
    "d:attn_frame": "FRAME (device, current stream): _icecache_decode, the whole attention module",
    "d:attn_forward": "FRAME (device, current stream): _icecache_attn_forward (same frame, one level up)",
    "d:estimate_select_recall": "FRAME (device, current stream): estimate_select_recall; contains Loading, which is on another stream",
    "d:append_kv": "append_paged_kv_cache (KV scatter kernel launch)",
    "d:scatter": "scatter_pages (page->pool scatter kernel launch)",
    "d:dci_add": "_DCI_add (page write path; runs on window rollover)",
    "d:offload_win": "offload_win_page_to_DCI (all layers, on window rollover)",
    "d:prepare_decode": "_prepare_decode (layer 0, each token)",
    "d:finish_decode": "_finish_decode (last layer, each token)",
    "d:prefill": "_icecache_prefill (token 0 only, outside the decode window)",
    "d:loading_wait": "HOST ms, not a device interval: c2g_stream.synchronize inside estimate_select_recall (infer_state.py:1821); it is already inside Loading's device interval as host time",
    "d:sync_outside_frame": "HOST ms, not a device interval: Stream.synchronize NOT inside estimate_select_recall (not bucketed)",
}
# Names in the accumulator that are HOST perf_counter measurements.
HOST_NAMES = ("d:loading_wait", "d:sync_outside_frame")
# Print order for the component block. Anything the accumulator saw that is not
# listed here is still printed (appended), so nothing is hidden.
COMPONENT_ORDER = (
    "Query", "Query.scan", "Query.tree", "Query.select",
    "Loading", "Loading.copy",
    "Decoding.sdpa", "Decoding.mlp",
    "d:estimate_select_recall", "d:attn_frame", "d:attn_forward",
    "d:append_kv", "d:scatter", "d:dci_add", "d:offload_win",
    "d:prepare_decode", "d:finish_decode", "d:prefill",
    "d:loading_wait", "d:sync_outside_frame",
)

# ------------------------------------------------------------------- accumulator
# Device cells: (row, token, name) -> list of [start_event, end_event, stream_handle,
# thread_name]. The event OBJECTS are what keep the timestamps alive until the
# report builder reads them; they are never recycled.
EV = {}
# Host cells (diagnostics only): (row, token, thread, name) -> [calls, seconds].
HOST = {}
ROW = [0]           # row index of the measurement in progress
ROW_N = [0]         # next row index (one per TokenTimer construction)
TOKEN = [0]         # monotonic per-generated-token counter for the current row
TOKEN_BY_ROW = {}   # row_index -> [that row's counter], see M.__call__
MARKS = {}          # row_index -> [perf_counter at each token boundary]
M_BY_ROW = {}       # row_index -> the M instance owning that row's marks
LOCK = threading.Lock()
_INSTALLED = []     # human-readable log of what was actually patched
_WARNINGS = []      # things the report must say out loud

_perf = time.perf_counter

# ------------------------------------------------------- wrapper self-cost (Fix 1)
# The probe is not free, and the honest way to price it is to measure the wrappers
# themselves instead of estimating. Every wrapper brackets ITS OWN host work -- the
# `stream_of` resolution / `torch.cuda.current_stream()`, the two `torch.cuda.Event`
# constructions, the two `record()` enqueues and the accumulator bookkeeping -- with
# `_perf()`, in two segments so that the WRAPPED CALL is never inside a bracket (the
# body of `fn` and the whole `finally` cleanup are excluded by construction).
# Deliberately OUTSIDE the brackets, and therefore not in the reported number: the
# thread-local depth bookkeeping (one getattr + one setattr, ~0.1 us/call) and the
# `_acc_wrap` call itself (~0.2 us/call), i.e. another ~0.1 ms/token of meter. They
# are named here so the number is a floor on the wrapper cost, not a guess at it.
#
# Cost of the meter itself: 4 `_perf()` calls per wrapped call, ~376 wrapped calls
# per decode token, i.e. ~1.5k clock reads/token. `perf_counter` is a vDSO call in
# the tens-of-nanoseconds range, so the meter costs on the order of 10-20 us/token
# against the milliseconds of host work it prices -- negligible, and stated here
# rather than assumed. The accumulation is plain float adds into a preallocated
# [calls, seconds, inside_calls, inside_seconds] cell picked by an INTEGER slot
# assigned once at wrapper-creation time: no string is hashed on the hot path.
WRAP_NAMES = []      # slot -> the wrapper's `name`
WRAP_BUCKET = []     # slot -> bucket label for the per-token rollup
WRAP_SLOT = {}       # name -> slot (string lookup happens ONCE, at wrap time)
# (row, token) -> [[calls, seconds, inside_frame_calls, inside_frame_seconds], ...]
WRAP_CELLS = {}
_WRAP_CUR = [None, None]      # [cells, (row, token)] the fast path is counting into
# Per-thread nesting depth of the `d:attn_frame` wrapper. A wrapper cost incurred
# while this is > 0 is charged INSIDE the frame interval and therefore inflates the
# frame denominator; anything else is outside it. This is the split that answers the
# question the report used to answer by guessing.
_FRAME = threading.local()
# The harness host interval a CLEAN (no-probe) run of the decomposed config produced
# on this box: mean of 4 warm passes, range 87.4-94.5 ms (2026-09-26). It is an
# external control, documented in README "page_scan Decode-Time Breakdown", NOT a
# measurement of the current run -- the report labels it as such wherever it uses it.
CLEAN_TPOT_MS = 89.53


def _wrap_slot(name):
    """Integer slot for `name`, created once. Never called on a hot path."""
    slot = WRAP_SLOT.get(name)
    if slot is None:
        with LOCK:
            slot = WRAP_SLOT.get(name)
            if slot is None:
                slot = WRAP_SLOT[name] = len(WRAP_NAMES)
                WRAP_NAMES.append(name)
                WRAP_BUCKET.append(_wrap_bucket_of(name))
                # A slot created after some (row, token) cells were handed out would
                # be indexed past the end of those lists on the hot path; pad them.
                for cells in WRAP_CELLS.values():
                    cells.append([0.0, 0.0, 0.0, 0.0])
    return slot


def _wrap_bucket_of(name):
    """Which rollup a wrapper's own cost belongs to, for the per-token report."""
    for bucket, leaves in BUCKET_LEAVES.items():
        if name in leaves:
            return bucket
    return "frame/diagnostic" if name.startswith("d:") else "nested (not bucketed)"


def _frame_depth():
    return getattr(_FRAME, "d", 0)


def _acc_wrap(slot, dt, inside_frame):
    """One wrapper's own host seconds -> the (row, token, slot) cell.

    Fast path is one tuple compare and four float adds; the (row, token) cells are
    handed out once per token. With `--n-reuse-layers 0` every wrapped call runs on
    the main thread; a prefetch worker thread would race these adds (a lost update
    of a few us, on the diagnostic only), which is accepted rather than paying a
    lock per wrapped call.
    """
    key = (ROW[0], TOKEN[0])
    cells = _WRAP_CUR[0]
    if cells is None or _WRAP_CUR[1] != key:
        with LOCK:
            cells = WRAP_CELLS.get(key)
            if cells is None:
                cells = WRAP_CELLS[key] = [[0.0, 0.0, 0.0, 0.0] for _ in WRAP_NAMES]
        _WRAP_CUR[0] = cells
        _WRAP_CUR[1] = key
    c = cells[slot]
    c[0] += 1.0
    c[1] += dt
    if inside_frame:
        c[2] += 1.0
        c[3] += dt


def _warn(msg):
    if msg not in _WARNINGS:
        _WARNINGS.append(msg)
    print("PROBE WARNING: " + msg, flush=True)


def _acc_dev(name, s, e, handle):
    """Accumulate one device event pair into the (row, token, name) cell."""
    key = (ROW[0], TOKEN[0], name)
    rec = [s, e, handle, threading.current_thread().name]
    with LOCK:
        EV.setdefault(key, []).append(rec)


def _acc_host(name, dt):
    """Accumulate one host duration into the (row, token, thread, name) cell."""
    key = (ROW[0], TOKEN[0], threading.current_thread().name, name)
    with LOCK:
        rec = HOST.get(key)
        if rec is None:
            rec = HOST[key] = [0, 0.0]
        rec[0] += 1
        rec[1] += dt


def _stream_handle(st):
    """A comparable integer for the stream an event was recorded on.

    `None` means the current stream at record time, which is resolved HERE, on the
    recording thread, so it cannot be confused with the report thread's stream.
    """
    if st is None:
        try:
            st = torch.cuda.current_stream()
        except Exception:
            return None
    return int(getattr(st, "cuda_stream", 0)) or None


def dev_timed(name, stream_of=None, frame=False):
    """Wrap a call in a CUDA event pair recorded on `stream_of(*args)`.

    `stream_of` returns the stream the wrapped call actually runs on, or
    `STREAM_CURRENT` (None) for "whatever the calling thread has current". The
    events are recorded BEFORE and AFTER the call, so the pair brackets exactly
    the work the call enqueues on that stream -- no synchronisation, ever.

    The wrapper also prices ITSELF: the segment before the start record and the
    segment after the call (end record + accumulator) are timed with `_perf()` and
    accumulated per token. The wrapped call is not inside either bracket, so the
    number is the probe's own host cost, not the measured method's. `frame=True`
    additionally makes the call's body the "inside" region for the
    inside/outside-`d:attn_frame` split (see `_acc_wrap`).
    """
    slot = _wrap_slot(name)

    def deco(fn):
        def wrapper(*a, **kw):
            t0 = _perf()
            st = stream_of(*a, **kw) if stream_of is not None else STREAM_CURRENT
            handle = _stream_handle(st)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(st)
            dt_pre = _perf() - t0
            depth = _frame_depth()
            if frame:
                _FRAME.d = depth + 1
            try:
                return fn(*a, **kw)
            finally:
                if frame:
                    _FRAME.d = depth
                t1 = _perf()
                e.record(st)
                _acc_dev(name, s, e, handle)
                _acc_wrap(slot, dt_pre + (_perf() - t1), depth > 0)
        wrapper.__name__ = getattr(fn, "__name__", name)
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper
    return deco


def host_timed(name):
    """Host wall-clock wrapper, for calls that BLOCK the host (stream waits).

    Not a device interval: a host timer around `Stream.synchronize` measures the
    wait, which is the only thing that wrapper is good for. The bucketed numbers
    never come from here.
    """
    def deco(fn):
        def wrapper(*a, **kw):
            t0 = _perf()
            try:
                return fn(*a, **kw)
            finally:
                _acc_host(name, _perf() - t0)
        wrapper.__name__ = getattr(fn, "__name__", name)
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper
    return deco


def patch(cls, meth, name, required=True, stream_of=None, host=False):
    """Patch `cls.meth` -> event pair (or host timer) accumulator, on `stream_of`."""
    orig = getattr(cls, meth, None)
    if orig is None:
        msg = f"{cls.__name__}.{meth} not found"
        if required:
            # Fail loud: this is a wrap point the partition depends on.
            raise AttributeError("page_scan_breakdown: " + msg)
        _INSTALLED.append(f"SKIPPED (optional): {msg}")
        return False
    wrap = host_timed(name) if host else dev_timed(name, stream_of)
    setattr(cls, meth, wrap(orig))
    kind = "host-timer" if host else "events"
    _INSTALLED.append(f"patched {cls.__module__}.{cls.__name__}.{meth} -> {name} ({kind})")
    return True


# ----------------------------------------------------- Loading: the c2g stream
# Mirror of infer_state.py:1687-1693 / 1738-1744: a worker thread's own stream if
# it has one, otherwise the instance's. Getting this wrong is the documented
# failure mode -- recorded on the default stream, Loading reads ~0 -- so the
# resolved handle is reported and cross-checked against Query's.
def _c2g_stream(self):
    try:
        thread_id = threading.get_ident()
        tl = getattr(self, "_thread_locals", None)
        if tl is not None and thread_id in tl:
            return tl[thread_id].c2g_stream
        return self.c2g_stream
    except Exception as exc:      # pragma: no cover - defensive
        _warn(f"could not resolve InferState.c2g_stream ({exc!r}); Loading was "
              f"recorded on the current stream and will read ~0")
        return STREAM_CURRENT


# Thread-local slot for the Loading call in progress, so the `copy_to_buffer`
# hook can attach a nested start event to the SAME c2g stream without knowing
# which stream that is. Only ever populated on the thread inside `recall`.
_PENDING = threading.local()


def _loading_wrapper(orig):
    """`InferState.recall`: the one bucket that does not run on the default stream.

    Also opens the nested `Loading.copy` pair, whose start is recorded by the
    `DCI.copy_to_buffer` hook below: `copy_to_buffer` is the last host-side
    statement before the H2D is enqueued (infer_state.py:1718-1724), so
    `copy_start -> recall end` is the device copy with the host staging removed.
    """
    slot_id = _wrap_slot("Loading")

    def wrapper(self, *a, **kw):
        t0 = _perf()
        st = _c2g_stream(self)
        handle = _stream_handle(st)
        slot = {"stream": st, "copy_starts": []}
        s = torch.cuda.Event(enable_timing=True)
        s.record(st)
        dt_pre = _perf() - t0
        _PENDING.loading = slot
        try:
            return orig(self, *a, **kw)
        finally:
            _PENDING.loading = None
            t1 = _perf()
            e = torch.cuda.Event(enable_timing=True)
            e.record(st)
            _acc_dev("Loading", s, e, handle)
            for cs in slot["copy_starts"]:
                _acc_dev("Loading.copy", cs, e, handle)
            _acc_wrap(slot_id, dt_pre + (_perf() - t1), _frame_depth() > 0)
    wrapper.__name__ = getattr(orig, "__name__", "recall")
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    return wrapper


def _copy_to_buffer_wrapper(orig):
    """Record the `Loading.copy` start event the moment the H2D is enqueued.

    `DCI.copy_to_buffer` is a `@staticmethod` (dciknn/core.py:142) called from
    exactly one place in the tree, infer_state.py:1718, inside `recall`'s
    `with torch.cuda.stream(c2g_stream)` block. Recorded AFTER the call returns
    (the host memcpy into the pinned transit buffer happens inside it and is
    host time, not device time). Outside `recall` the hook is a no-op.
    """
    slot_id = _wrap_slot("Loading.copy")

    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        slot = getattr(_PENDING, "loading", None)
        if slot is not None:
            t0 = _perf()
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(slot["stream"])
            slot["copy_starts"].append(ev)
            _acc_wrap(slot_id, _perf() - t0, _frame_depth() > 0)
        return out
    wrapper.__name__ = getattr(orig, "__name__", "copy_to_buffer")
    return wrapper


def install_loading_probe():
    orig_recall = InferState.recall
    InferState.recall = _loading_wrapper(orig_recall)
    _INSTALLED.append("patched icecache.infer_state.InferState.recall -> Loading "
                      "(events on c2g_stream, not the default stream)")

    DCI = getattr(istate, "DCI", None)
    if DCI is None:
        try:
            from dciknn import DCI as DCI  # noqa: N813
        except Exception as exc:
            DCI = None
            _INSTALLED.append(f"SKIPPED (optional): DCI for Loading.copy ({exc!r})")
    if DCI is not None:
        orig_copy = getattr(DCI, "copy_to_buffer", None)
        if orig_copy is None:
            _INSTALLED.append("SKIPPED (optional): DCI.copy_to_buffer not found")
        else:
            # Optional diagnostic only: the Loading bucket itself does not depend
            # on it, so a class that refuses the setattr (a pybind11 type would)
            # must not take the run down with it.
            try:
                setattr(DCI, "copy_to_buffer",
                        staticmethod(_copy_to_buffer_wrapper(orig_copy)))
                _INSTALLED.append("patched dciknn.DCI.copy_to_buffer -> Loading.copy "
                                  "(nested start on the same c2g stream)")
            except Exception as exc:
                _INSTALLED.append(f"SKIPPED (optional): DCI.copy_to_buffer not patchable "
                                  f"({exc!r}); Loading.copy will be absent")


# ------------------------------------------------------------- sync scope (host)
# The only live `Stream.synchronize` in the decode path is infer_state.py:1821,
# the last statement of `estimate_select_recall`. It is HOST time already inside
# the Loading device interval, so it is not a bucket: it is recorded as
# `d:loading_wait` to show how much of the H2D the host actually blocked on. A
# sync seen anywhere else is recorded as `d:sync_outside_frame` and is not
# bucketed either. The scope is attributed by thread-local depth rather than by
# identifying the stream object: worker threads get their own c2g stream.
_SCOPE = threading.local()


def _scope_depth():
    return getattr(_SCOPE, "depth", 0)


_orig_stream_sync = torch.cuda.Stream.synchronize


def _timed_stream_sync(self, *a, **kw):
    t0 = _perf()
    try:
        return _orig_stream_sync(self, *a, **kw)
    finally:
        dt = _perf() - t0
        _acc_host("d:loading_wait" if _scope_depth() > 0 else "d:sync_outside_frame", dt)


def install_path_probe():
    """Query / Loading / Decoding leaves plus their nested children."""
    DCI = getattr(istate, "DCI", None)
    if DCI is None:
        from dciknn import DCI as DCI  # noqa: N813

    # -- Query: the single entry point both backends go through, on the current
    #    stream (see the module docstring for the proof).
    patch(InferState, "_DCI_query", "Query")
    patch(PageScan, "query", "Query.scan", required=False)
    patch(DCI, "query", "Query.tree", required=False)
    patch(InferState, "_apply_selected_pages", "Query.select", required=False)

    # -- Loading: the c2g-stream H2D (+ the host staging before it) and its nest.
    install_loading_probe()

    # Scope marker for the host sync timer. Wrapping the method (not replacing the
    # sync) keeps the depth correct even if the caller is an executor thread.
    orig_esr = InferState.estimate_select_recall
    slot_id = _wrap_slot("d:estimate_select_recall")

    def scoped_estimate_select_recall(self, *a, **kw):
        t0 = _perf()
        _SCOPE.depth = _scope_depth() + 1
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        handle = _stream_handle(STREAM_CURRENT)
        s.record(STREAM_CURRENT)
        dt_pre = _perf() - t0
        try:
            return orig_esr(self, *a, **kw)
        finally:
            _SCOPE.depth -= 1
            t1 = _perf()
            e.record(STREAM_CURRENT)
            _acc_dev("d:estimate_select_recall", s, e, handle)
            _acc_wrap(slot_id, dt_pre + (_perf() - t1), _frame_depth() > 0)

    InferState.estimate_select_recall = scoped_estimate_select_recall
    _INSTALLED.append("InferState.estimate_select_recall -> d:estimate_select_recall "
                      "(current-stream events, + sync scope)")
    torch.cuda.Stream.synchronize = _timed_stream_sync
    _INSTALLED.append("torch.cuda.Stream.synchronize -> d:loading_wait (in scope) / "
                      "d:sync_outside_frame  [HOST timers]")

    # -- Decoding: attention over the stitched cache, on the current stream.
    patch(InferState, "decode_sdpa", "Decoding.sdpa")

    # -- Diagnostics only: frames that nest the buckets, so the residual can be read.
    patch(InferState, "append_paged_kv_cache", "d:append_kv", required=False)
    patch(InferState, "scatter_pages", "d:scatter", required=False)
    patch(InferState, "_DCI_add", "d:dci_add", required=False)
    patch(InferState, "offload_win_page_to_DCI", "d:offload_win", required=False)
    patch(InferState, "_prepare_decode", "d:prepare_decode", required=False)
    patch(InferState, "_finish_decode", "d:finish_decode", required=False)

    # The attention module frame. `_icecache_attn_forward` is called from a lambda
    # bound at enable_icecache time, which resolves the name in modeling's globals
    # at CALL time, so patching the module attribute is enough.
    orig_decode = modeling._icecache_decode
    orig_prefill = modeling._icecache_prefill
    orig_attn = modeling._icecache_attn_forward
    # frame=True: this call's body is the "inside the frame" region for the
    # wrapper-host-cost split, because `d:attn_frame` is the frame-denominated
    # denominator (WHICH SHARES TO PLOT).
    modeling._icecache_decode = dev_timed("d:attn_frame", frame=True)(orig_decode)
    modeling._icecache_prefill = dev_timed("d:prefill")(orig_prefill)
    modeling._icecache_attn_forward = dev_timed("d:attn_forward")(orig_attn)
    _INSTALLED.append("modeling._icecache_decode -> d:attn_frame")
    _INSTALLED.append("modeling._icecache_attn_forward -> d:attn_forward")
    _INSTALLED.append("modeling._icecache_prefill -> d:prefill")


# -------------------------------------------------------------- Decoding: the MLP
# The MLP is NOT patched by the adapter -- only modules whose class name contains
# "Attention" are (modeling.py:459-466) -- and the stock decoder layer calls
# `self.mlp(hidden_states)` after the attention returns. So the MLP is wrapped
# per-instance here. The handle on the model comes from wrapping the adapter's
# `enable_icecache`, which the harness calls as `adapter.enable_icecache(model,...)`
# (page_scan_compare.py:265) -- a module attribute lookup, hence patchable.
MLP_PATHS = []


def _wrap_mlp(mod):
    orig = mod.forward
    # Same slot for every MLP module: the rollup is per wrapper KIND, and this path
    # is different from dev_timed only because the module is patched per instance.
    slot_id = _wrap_slot("Decoding.mlp")

    def forward(*a, **kw):
        t0 = _perf()
        handle = _stream_handle(STREAM_CURRENT)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(STREAM_CURRENT)
        dt_pre = _perf() - t0
        try:
            return orig(*a, **kw)
        finally:
            t1 = _perf()
            e.record(STREAM_CURRENT)
            _acc_dev("Decoding.mlp", s, e, handle)
            _acc_wrap(slot_id, dt_pre + (_perf() - t1), _frame_depth() > 0)

    forward.__name__ = "probe_mlp_forward"
    mod.forward = forward


def install_mlp_wrappers(model):
    n = 0
    for name, mod in model.named_modules():
        if name != "mlp" and not name.endswith(".mlp"):
            continue
        if getattr(mod, "_probe_mlp_wrapped", False):
            continue
        _wrap_mlp(mod)
        mod._probe_mlp_wrapped = True
        n += 1
        if len(MLP_PATHS) < 3:
            MLP_PATHS.append(f"{name} ({type(mod).__name__})")
    if n:
        _INSTALLED.append(f"mlp.forward x{n} -> Decoding.mlp, current-stream events "
                          f"(e.g. {MLP_PATHS[0]})")
    return n


def install_mlp_hook():
    orig_enable = icecache_adapter.enable_icecache

    def probe_enable_icecache(model, *a, **kw):
        out = orig_enable(model, *a, **kw)
        target = out if out is not None else model
        if install_mlp_wrappers(target) == 0:
            _warn("no module named '*.mlp' found; Decoding.mlp is empty")
        return out

    icecache_adapter.enable_icecache = probe_enable_icecache
    _INSTALLED.append("icecache.adapter.enable_icecache -> + MLP walk")


# ------------------------------------------------------------- per-token counter
class M(StoppingCriteria):
    """Token-boundary marker: one sync, one monotonic counter increment.

    No event is recorded here. The per-token interval is the HARNESS's own host
    interval between two of its timestamps (`marks[k] - marks[k-1]`), which is what
    `ttft_s` / `tpot_s` are computed from; the buckets are the device intervals
    inside it.

    `__call__` performs exactly what `page_scan_compare.TokenTimer.__call__`
    performs -- `torch.cuda.synchronize(self.device)` then a `perf_counter()`
    timestamp (`page_scan_compare.py:152-155`) -- and additionally increments
    `TOKEN[0]`. It is NOT an extra synchronisation: `install_token_timer` patches
    the harness's `TokenTimer.__call__` to run this and then hand its mark to the
    harness as that timer's own timestamp, so a run with the probe has the same one
    sync per token that the harness already does. That is a claim about
    SYNCHRONISATION only, and an earlier revision of this docstring wrongly extended
    it to the timings: `ttft_s` / `tpot_s` on a probe row are NOT what the harness
    would have recorded on a clean run. The wrappers' own host cost is inside the
    interval and this decode is host-bound, so it is exposed -- measured +26%
    (112.2-113.6 ms vs the clean 4-pass mean 89.53 ms; see PERTURBATION and the
    `wrapper_host` measurement in the report). Quote the clean control, not the
    probe row's own `tpot_s`. As a free side effect, that sync is also what
    guarantees every event pair recorded for the token is complete before the next
    token's interval starts.
    """

    def __init__(self, device, row=0):
        self.device = device
        self.row = row
        self.marks = []

    def __call__(self, input_ids, scores, **kwargs):
        torch.cuda.synchronize(self.device)
        self.marks.append(_perf())
        # Re-bind the globals to THIS row before counting. In the harness's call
        # order this is a no-op; if a call ever landed after the next row's timer
        # was constructed, it is attributed to its own row's counter (where it
        # falls past that row's decode window and is dropped) instead of becoming
        # the next row's token 1.
        ctr = TOKEN_BY_ROW.setdefault(self.row, [0])
        ROW[0] = self.row
        ctr[0] += 1
        TOKEN[0] = ctr[0]
        return False


def install_token_timer():
    orig_init = psc.TokenTimer.__init__
    orig_call = psc.TokenTimer.__call__

    def probe_init(self, device):
        orig_init(self, device)
        row = ROW_N[0]
        ROW_N[0] = row + 1
        # A row is one (backend, sample) = one TokenTimer = one JSONL line, and the
        # harness writes them in construction order (page_scan_compare.py:281/315).
        ROW[0] = row
        TOKEN_BY_ROW[row] = [0]
        TOKEN[0] = 0
        self._probe_row = row
        M_BY_ROW[row] = M(device, row)
        MARKS[row] = M_BY_ROW[row].marks

    def probe_call(self, input_ids, scores, **kwargs):
        m = M_BY_ROW.get(getattr(self, "_probe_row", ROW[0]))
        if m is None:          # never happens: the timer is constructed before generate
            return orig_call(self, input_ids, scores, **kwargs)
        m(input_ids, scores, **kwargs)      # sync + mark + TOKEN[0] += 1
        # Identical in effect to TokenTimer.__call__'s `self.times.append(perf_counter())`.
        self.times.append(m.marks[-1])
        return False

    psc.TokenTimer.__init__ = probe_init
    psc.TokenTimer.__call__ = probe_call
    _INSTALLED.append("page_scan_compare.TokenTimer.__init__/__call__ -> M (TOKEN counter)")


# -------------------------------------------------------------------- read-back
def _index(cells):
    """{row: {token: {name: payload}}} -- O(1) lookup instead of a full scan."""
    idx = {}
    for (r, t, n), payload in cells.items():
        idx.setdefault(r, {}).setdefault(t, {})[n] = payload
    return idx


def _read_pair(s, e, notes):
    """Elapsed ms of one completed event pair, or None if it never completed.

    `query()` is not a synchronisation: it just asks whether the event has been
    reached. Anything not completed (an aborted row, a token whose work was still
    in flight when the process left the harness) is reported as missing rather
    than read as garbage -- `elapsed_time` on an incomplete event raises.
    """
    try:
        if not (s.query() and e.query()):
            notes.add("an event pair had not completed at report time; skipped")
            return None
        return float(s.elapsed_time(e))
    except Exception as exc:                       # pragma: no cover - defensive
        notes.add(f"elapsed_time failed: {exc!r}")
        return None


def _dev_cell(idx_row, token, name, notes):
    """(calls, sum_ms, per_call_ms, handles, threads, n_unreadable) for one cell."""
    cell = idx_row.get(token, {}).get(name)
    if not cell:
        return 0, 0.0, [], [], [], 0
    per_call, handles, threads = [], [], []
    total = 0.0
    bad = 0
    for s, e, handle, thread in cell:
        ms = _read_pair(s, e, notes)
        handles.append(handle)
        threads.append(thread)
        if ms is None:
            bad += 1
            continue
        per_call.append(ms)
        total += ms
    return len(cell), total, per_call, handles, threads, bad


def _host_index():
    """{row: {token: {name: [calls, seconds]}}}, summed over threads."""
    idx = {}
    for (r, t, _thr, n), (c, sec) in HOST.items():
        slot = idx.setdefault(r, {}).setdefault(t, {}).setdefault(n, [0, 0.0])
        slot[0] += c
        slot[1] += sec
    return idx


def _bucket_ms(idx_row, token, notes):
    sums = {}
    counts = {}
    per_call = {}
    handles = {}
    unreadable = {}
    for bucket in BUCKETS:
        total = 0.0
        calls = 0
        bad = 0
        for leaf in BUCKET_LEAVES[bucket]:
            c, ms, pc, hs, _thr, b = _dev_cell(idx_row, token, leaf, notes)
            calls += c
            total += ms
            bad += b
            if pc:
                per_call.setdefault(bucket, []).extend(pc)
            if hs:
                handles.setdefault(bucket, set()).update(h for h in hs if h is not None)
        sums[bucket] = total
        counts[bucket] = calls
        unreadable[bucket] = bad
    return sums, counts, per_call, handles, unreadable


def _mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _pct(x, tot):
    if x is None or not tot:
        return None
    return 100.0 * x / tot


def _load_rows(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows, f"harness output {p} does not exist"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:      # a torn final line
            return rows, f"unparsable JSONL line: {exc}"
    return rows, None


# -------------------------------------------------------------- retrieval_stats
def _harness_query_stats(hrow):
    """The harness's own measurements of the same region the Query bucket covers.

    `retrieval_stats["query"]["page_scan"]` is `_DCI_query`'s host timer around its
    page_scan branch (infer_state.py:1466/1472): it INCLUDES the `.cpu()` D2H drain
    (page_scan.py:729) and, unlike the probe's Query bucket, EXCLUDES
    `_apply_selected_pages`. `page_scan_layers[layer]["query"]` is `PageScan.query`'s
    own host timer around the device scan (page_scan.py:615-618), i.e. the closest
    harness analogue of the nested `Query.scan` pair. Both are host times, so both
    must be >= the corresponding device-only event interval; a probe number that is
    LARGER than the harness p50 means the event pair is spanning foreign device
    work, which is exactly the failure mode this file exists to remove.
    """
    stats = hrow.get("retrieval_stats") or {}
    out = {
        "query_page_scan": (stats.get("query") or {}).get("page_scan"),
        "query_dci": (stats.get("query") or {}).get("dci"),
        "page_scan_layers_query": {},
        "page_scan_layers_build_ms": {},
        "actual_backend_configured": stats.get("configured_backend"),
        "disabled_layers": stats.get("disabled_layers"),
        "fallback_queries": stats.get("fallback_queries"),
        "n_layers_with_scan": 0,
        "n_device_path_layers": 0,
    }
    for layer, entry in sorted((stats.get("page_scan_layers") or {}).items(),
                               key=lambda kv: int(kv[0])):
        q = entry.get("query") or {}
        out["page_scan_layers_query"][layer] = q
        out["page_scan_layers_build_ms"][layer] = entry.get("build_ms")
        if q.get("count"):
            out["n_layers_with_scan"] += 1
            # `PageScan.query` only appends to its `query_seconds` on the DEVICE
            # path (page_scan.py:616-619); the numpy fallback never does.
            out["n_device_path_layers"] += 1
    return out


def _cross_check(probe_per_call_ms, calls, stats):
    """Probe device ms/call vs the harness host p50 for the same region."""
    res = {"probe_ms_per_call": probe_per_call_ms, "probe_calls": calls}
    if probe_per_call_ms is None or calls == 0:
        res["verdict"] = "no probe calls; nothing to cross-check"
        return res
    for key, label in (("query_page_scan", "retrieval_stats.query.page_scan"),
                       ("page_scan_layers_query", "retrieval_stats.page_scan_layers[].query")):
        if key == "query_page_scan":
            entry = stats.get(key) or {}
            p50 = entry.get("p50_ms")
            count = entry.get("count")
        else:
            p50s = [q.get("p50_ms") for q in stats.get(key, {}).values() if q.get("p50_ms") is not None]
            counts = [q.get("count") or 0 for q in stats.get(key, {}).values()]
            p50 = (sum(p50s) / len(p50s)) if p50s else None
            count = sum(counts) if counts else None
        res[label + ".p50_ms"] = p50
        res[label + ".count"] = count
        if p50:
            res[label + ".probe_over_harness"] = probe_per_call_ms / p50
    ratio = res.get("retrieval_stats.query.page_scan.probe_over_harness")
    if ratio is None:
        res["verdict"] = "harness p50 unavailable; cannot cross-check"
    elif ratio > 1.5:
        res["verdict"] = (f"SUSPECT: probe device {probe_per_call_ms:.3f} ms/call is "
                          f"{ratio:.2f}x the harness host p50 for the same region. A "
                          f"device-only interval cannot exceed the host time of a call "
                          f"that contains it: the event pair is spanning foreign work.")
    elif ratio < 0.25:
        res["verdict"] = (f"probe is {ratio:.2f}x the harness p50 -- plausible for a "
                          f"device-only interval, but check that the scan really ran on "
                          f"the device (see n_device_path_layers).")
    else:
        res["verdict"] = f"OK ({ratio:.2f}x the harness host p50)"
    return res


def _mod_file(modname):
    """(resolved file, size in bytes) of an ALREADY-IMPORTED module, else (None, None).

    Provenance only -- nothing here is imported, `sys.modules` is read, so a missing
    module records None instead of taking the report down. The compiled kernels are
    recorded because the numbers depend on them and this project has shown a rebuilt
    DCI `.so` is a different binary that does not reproduce the installed one: a
    breakdown taken across a `.so` swap is otherwise byte-indistinguishable from one
    taken before it. `getsize` keeps this to `os`/`pathlib`, already imported.
    """
    path = getattr(sys.modules.get(modname), "__file__", None)
    if not path:
        return None, None
    try:
        return path, os.path.getsize(path)
    except OSError:                     # pragma: no cover - defensive
        return path, None


def _wrapper_host_stats(row, n_intervals):
    """What the probe's own wrappers cost, MEASURED per decode token (Fix 1).

    This replaces a guess: the earlier revision printed "~1 us each" for the event
    records. Every wrapper now brackets its own host work (stream resolution,
    `torch.cuda.Event` construction, `record()`, bookkeeping) and this rolls those
    brackets up per decode token, per bucket, plus the QUESTION the guess could not
    answer: how much of it is charged INSIDE the `d:attn_frame` interval -- i.e.
    whether the frame denominator is itself inflated -- and how much is outside.
    Token 0 (the prefill window) is excluded, like every other decode stat here.
    """
    by_name = {}
    calls = 0.0
    total = 0.0
    inside = 0.0
    for token in range(1, n_intervals + 1):
        cells = WRAP_CELLS.get((row, token))
        if not cells:
            continue
        for i, c in enumerate(cells):
            if not c[0]:
                continue
            name = WRAP_NAMES[i]
            e = by_name.get(name)
            if e is None:
                e = by_name[name] = {"bucket": WRAP_BUCKET[i], "calls": 0.0,
                                     "seconds": 0.0, "inside_frame_seconds": 0.0}
            e["calls"] += c[0]
            e["seconds"] += c[1]
            e["inside_frame_seconds"] += c[3]
            calls += c[0]
            total += c[1]
            inside += c[3]
    denom = n_intervals or 1
    buckets = {}
    for name, e in by_name.items():
        b = buckets.setdefault(e["bucket"], {"calls": 0.0, "seconds": 0.0,
                                             "inside_frame_seconds": 0.0})
        b["calls"] += e["calls"]
        b["seconds"] += e["seconds"]
        b["inside_frame_seconds"] += e["inside_frame_seconds"]
    return {
        "method": ("time.perf_counter brackets inside the wrappers around their OWN "
                   "work (stream resolution, Event construction, record(), "
                   "bookkeeping). The wrapped call is never inside a bracket. Four "
                   "clock reads per wrapped call, ~376 wrapped calls per token: "
                   "tens of us/token of meter against the ms it prices."),
        "n_decode_intervals": n_intervals,
        "calls_per_token": calls / denom,
        "total_ms_per_token": total * 1e3 / denom,
        "inside_attn_frame_ms_per_token": inside * 1e3 / denom,
        "outside_attn_frame_ms_per_token": (total - inside) * 1e3 / denom,
        "per_bucket_ms_per_token": {b: v["seconds"] * 1e3 / denom
                                    for b, v in sorted(buckets.items())},
        "per_bucket_calls_per_token": {b: v["calls"] / denom
                                       for b, v in sorted(buckets.items())},
        "per_bucket_inside_frame_ms_per_token": {b: v["inside_frame_seconds"] * 1e3 / denom
                                                 for b, v in sorted(buckets.items())},
        "per_name_ms_per_token": {n: e["seconds"] * 1e3 / denom
                                  for n, e in sorted(by_name.items())},
        "per_name_calls_per_token": {n: e["calls"] / denom
                                     for n, e in sorted(by_name.items())},
        "clean_control_tpot_ms": CLEAN_TPOT_MS,
        "clean_control_provenance": ("mean of 4 warm no-probe passes of this config on "
                                     "this box (range 87.4-94.5 ms), documented in "
                                     "README 'page_scan Decode-Time Breakdown'; it is "
                                     "NOT measured by this run"),
    }


def build_report(harness_path, argv, error):
    rows, load_err = _load_rows(harness_path)
    # Row i of the JSONL is the i-th TokenTimer the harness constructed, because
    # one timer is built per (backend, sample) and the row is appended right after
    # that sample's generate (page_scan_compare.py:281/315). If the counts differ,
    # the row->bucket mapping below is NOT trustworthy and must be said out loud
    # rather than silently applied.
    if len(rows) != ROW_N[0] and not load_err:
        load_err = (f"row/bucket mismatch: {len(rows)} JSONL row(s) but "
                    f"{ROW_N[0]} TokenTimer(s); the row_index mapping is suspect")
        _warn(load_err)
    # The compiled kernels the numbers depend on: `icecache.infer_state` does
    # `import icecache_cpp as _cpp` and `from dciknn import DCI`, and `dciknn._dci`
    # is the loaded `_dci*.so` -- note that site-packages/dciknn/ also carries a
    # `_dci.cpython-310-x86_64-linux-gnu.so.bak_unpatched`, so the DIRECTORY is not
    # enough provenance: this records which one was actually loaded. All three are
    # imported by the time this block runs; a missing one records None.
    _cpp_file, _cpp_bytes = _mod_file("icecache_cpp")
    _dciknn_file, _dciknn_bytes = _mod_file("dciknn")
    _dci_file, _dci_bytes = _mod_file("dciknn._dci")
    report = {
        "probe": {
            "argv": argv,
            "repo": str(REPO),
            "page_scan_compare_file": getattr(psc, "__file__", None),
            "icecache_file": getattr(istate, "__file__", None),
            "icecache_cpp_file": _cpp_file,
            "icecache_cpp_bytes": _cpp_bytes,
            "dciknn_file": _dciknn_file,
            "dciknn_bytes": _dciknn_bytes,
            "dciknn_dci_file": _dci_file,
            "dciknn_dci_bytes": _dci_bytes,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "n_token_timers": ROW_N[0],
            "n_harness_rows": len(rows),
            "installed": list(_INSTALLED),
            "warnings": list(_WARNINGS),
            "mlp_paths": list(MLP_PATHS),
            "buckets": list(BUCKETS),
            "bucket_leaves": {k: list(v) for k, v in BUCKET_LEAVES.items()},
            "bucket_streams": {
                "Query": "torch.cuda.current_stream() (page_scan.py:591/696, no stream switch)",
                "Loading": "InferState.c2g_stream (infer_state.py:1687-1693, 1716)",
                "Decoding.sdpa": "torch.cuda.current_stream() (infer_state.py:2024, no stream switch)",
                "Decoding.mlp": "torch.cuda.current_stream() (stock LlamaMLP.forward)",
                "Others": "n/a -- residual, interval - the three above",
            },
            "name_notes": dict(NAME_NOTES),
            "host_timed_names": list(HOST_NAMES),
            "wrapper_host_slots": {n: WRAP_BUCKET[i] for i, n in enumerate(WRAP_NAMES)},
            "clean_control_tpot_ms": CLEAN_TPOT_MS,
            "errors": ([error] if error else []) + ([load_err] if load_err else []),
        },
        "rows": [],
    }

    ev_idx = _index(EV)
    host_idx = _host_index()

    for i, hrow in enumerate(rows):
        marks = MARKS.get(i, [])
        n_intervals = max(0, len(marks) - 1)
        notes = set()
        idx_row = ev_idx.get(i, {})
        host_row = host_idx.get(i, {})
        per_token = []
        # The frame denominator reads into its OWN notes set: `events_incomplete`
        # gates the interval-denominated stack, and a frame read failure must not
        # silently suppress that unrelated share line (it gates the frame one).
        frame_notes = set()
        n_frame_bad = 0
        # Token index 0 is everything before the FIRST mark: the prefill / first
        # forward. It is collected separately and excluded from the decode window.
        pre0_calls = {}
        for name in comp_names(idx_row):
            c, _ms, _pc, _h, _th, _b = _dev_cell(idx_row, 0, name, notes)
            if c:
                pre0_calls[name] = c
        for k in range(1, n_intervals + 1):
            interval_ms = (marks[k] - marks[k - 1]) * 1e3
            sums, counts, per_call, handles, unreadable = _bucket_ms(idx_row, k, notes)
            others = interval_ms - sums["Query"] - sums["Loading"] - sums["Decoding"]
            lc_calls, lc_ms, lc_pc, _lc_h, _lc_th, _lc_bad = _dev_cell(
                idx_row, k, "Loading.copy", notes)
            # The device frame that the second share set is denominated on. Summed
            # over its calls in the token, exactly as the `components` block does.
            fr_calls, fr_ms, _fr_pc, _fr_h, _fr_th, fr_bad = _dev_cell(
                idx_row, k, "d:attn_frame", frame_notes)
            n_frame_bad += fr_bad
            n_bad = sum(unreadable.values()) + _lc_bad
            rec = {
                "token": k,
                "interval_ms": interval_ms,
                "Query_ms": sums["Query"],
                "Loading_ms": sums["Loading"],
                "Loading_copy_ms": lc_ms if lc_calls else None,
                "Loading_host_stage_ms": (sums["Loading"] - lc_ms) if lc_calls else None,
                "Decoding_ms": sums["Decoding"],
                "Others_ms": others,
                "bucket_sum_ms": sums["Query"] + sums["Loading"] + sums["Decoding"],
                "attn_frame_ms": fr_ms if fr_calls else None,
                "attn_frame_calls": fr_calls,
                "calls": counts,
                "calls_per_bucket": counts,
                "mean_ms_per_call": {b: _mean(per_call.get(b, [])) for b in BUCKETS},
                "loading_copy_calls": lc_calls,
                "stream_handles": {b: sorted(h) for b, h in handles.items()},
                "unreadable_pairs": n_bad,
                "negative_others": others < 0,
            }
            per_token.append(rec)

        def mean(key):
            return _mean([r[key] for r in per_token])

        tpot = hrow.get("tpot_s")
        mean_interval = _mean([r["interval_ms"] for r in per_token])
        mean_query = mean("Query_ms")
        mean_loading = mean("Loading_ms")
        mean_decoding = mean("Decoding_ms")
        mean_others = mean("Others_ms")
        mean_loading_copy = mean("Loading_copy_ms")
        mean_loading_stage = mean("Loading_host_stage_ms")
        neg_tokens = [r["token"] for r in per_token if r["negative_others"]]
        tot = mean_interval
        shares = None
        # A stack is only drawn when every token has (a) all three buckets present
        # and (b) a non-negative residual. An unreadable event pair reports 0.0 ms,
        # which would otherwise render as a clean-looking 0% bucket.
        events_incomplete = bool(notes)
        stack_valid = (bool(per_token) and not neg_tokens and not events_incomplete
                       and None not in (mean_query, mean_loading, mean_decoding,
                                        mean_others, tot))
        if stack_valid:
            shares = {
                "denominator": "the harness host token interval (marks[k]-marks[k-1]); "
                               "PROBE-INFLATED -- it contains the wrappers' own host "
                               "cost. Do not plot.",
                "denominator_ms": tot,
                "Query_pct": _pct(mean_query, tot),
                "Loading_pct": _pct(mean_loading, tot),
                "Decoding_pct": _pct(mean_decoding, tot),
                "Others_pct": _pct(mean_others, tot),
                "bucket_sum_pct": _pct((mean_query or 0) + (mean_loading or 0) + (mean_decoding or 0), tot),
            }

        # The second share set: same numerators, device-frame denominator, so the
        # probe's own host cost cannot inflate it. Gated on its own terms -- the
        # interval's negative-`Others` refusal is a cross-stream double-billing
        # argument, and the frame pair is recorded on ONE stream, so it does not
        # gate this one. The frame residual CAN still be negative (mlp outside the
        # frame; Loading overlapping it from c2g); that is reported, not hidden.
        mean_frame = mean("attn_frame_ms")
        mean_frame_calls = mean("attn_frame_calls")
        frame_shares = None
        if (per_token and not events_incomplete and not n_frame_bad
                and mean_frame and None not in (mean_query, mean_loading, mean_decoding)):
            frame_bucket_sum = mean_query + mean_loading + mean_decoding
            frame_others = mean_frame - frame_bucket_sum
            frame_shares = {
                "denominator": "d:attn_frame, a CUDA device interval summed over its "
                               "calls in the token -- INFLATION-FREE (the numerators "
                               "are device intervals too). PLOT THIS SET.",
                "denominator_ms": mean_frame,
                "denominator_calls_per_token": mean_frame_calls,
                "Query_ms": mean_query, "Query_pct": _pct(mean_query, mean_frame),
                "Loading_ms": mean_loading, "Loading_pct": _pct(mean_loading, mean_frame),
                "Decoding_ms": mean_decoding, "Decoding_pct": _pct(mean_decoding, mean_frame),
                "bucket_sum_ms": frame_bucket_sum,
                "bucket_sum_pct": _pct(frame_bucket_sum, mean_frame),
                "Others_from_frame_ms": frame_others,
                "Others_pct": _pct(frame_others, mean_frame),
                "note": "Others_pct is negative whenever the three buckets exceed the "
                        "frame: Decoding.mlp is NOT inside the attention module's frame "
                        "(stock LlamaDecoderLayer.forward calls self.mlp after "
                        "self.self_attn returns), and Loading runs on the c2g stream, "
                        "so it can overlap the frame rather than sit inside it. Read "
                        "the per-bucket ms and the per-bucket percentages, not the "
                        "frame residual.",
            }
        means = {
            "interval_ms": mean_interval,
            "Query_ms": mean_query,
            "Loading_ms": mean_loading,
            "Loading_copy_ms": mean_loading_copy,
            "Loading_host_stage_ms": mean_loading_stage,
            "Decoding_ms": mean_decoding,
            "Others_from_interval_ms": mean_others,
            # What the old host-timing revision asked for: the residual taken
            # against the HARNESS's tpot. It should agree with
            # Others_from_interval_ms because mean(interval) == tpot_s exactly
            # (both are (marks[-1]-marks[0])/(n-1)); a disagreement is a bookkeeping bug.
            "Others_from_tpot_ms": (tpot * 1e3 - (mean_query + mean_loading + mean_decoding))
            if (tpot is not None and per_token and None not in (mean_query, mean_loading, mean_decoding)) else None,
            "tpot_ms_from_row": None if tpot is None else tpot * 1e3,
            "mean_ms_per_call": {b: _mean([r["mean_ms_per_call"].get(b) for r in per_token]) for b in BUCKETS},
            "calls_per_token": {b: _mean([r["calls"].get(b) for r in per_token]) for b in BUCKETS},
            "attn_frame_ms": mean_frame,
            "attn_frame_calls_per_token": mean_frame_calls,
            # TWO share sets, two denominators. `shares_pct` is the harness host
            # interval, which the probe inflates (+26% measured); `frame_shares_pct`
            # is a CUDA device interval and is the one to plot. See WHICH SHARES TO
            # PLOT in the module docstring.
            "shares_pct": shares,
            "frame_shares_pct": frame_shares,
            "shares_denominator_rule": (
                "plot frame_shares_pct; shares_pct is inflated by the wrappers' own "
                "host cost (measured in row['wrapper_host']), not by the method"),
            "stack_valid": stack_valid,
            "frame_shares_valid": bool(frame_shares),
            "events_incomplete": events_incomplete,
            "negative_others_tokens": neg_tokens,
            "n_negative_others": len(neg_tokens),
        }

        # Device names come from the event cells; host names only ever appear in HOST.
        seen = comp_names(idx_row) | {n for (r, _t, _thr, n) in HOST if r == i}
        names = [n for n in COMPONENT_ORDER if n in seen]
        names += sorted(n for n in seen if n not in COMPONENT_ORDER)
        components = {}
        for name in names:
            calls = 0
            total_ms = 0.0
            per_call_all = []
            handles = set()
            host = name in HOST_NAMES
            for k in range(1, n_intervals + 1):
                if host:
                    cell = host_row.get(k, {}).get(name)
                    if cell:
                        calls += cell[0]
                        total_ms += cell[1] * 1e3
                    continue
                c, ms, pc, hs, _thr, _b = _dev_cell(idx_row, k, name, notes)
                calls += c
                total_ms += ms
                per_call_all.extend(pc)
                handles.update(h for h in hs if h is not None)
            if calls or total_ms:
                components[name] = {
                    "calls": calls,
                    "sum_ms": total_ms,
                    "mean_ms": (total_ms / n_intervals) if n_intervals else None,
                    "mean_ms_per_call": _mean(per_call_all),
                    "kind": "host" if host else "device",
                    "stream_handles": sorted(handles),
                    "note": NAME_NOTES.get(name, "UNCLASSIFIED"),
                }

        stats = _harness_query_stats(hrow)
        q_per_call = means["mean_ms_per_call"].get("Query")
        q_calls = sum(r["calls"].get("Query", 0) for r in per_token)
        scan_per_call = components.get("Query.scan", {}).get("mean_ms_per_call")
        cross = {
            "Query_bucket": _cross_check(q_per_call, q_calls, stats),
            "Query_scan_nested": _cross_check(
                scan_per_call,
                components.get("Query.scan", {}).get("calls", 0), stats),
        }
        # Stream sanity: Loading must NOT have been recorded on the same stream as
        # Query, or it is reading the ~0 failure mode.
        q_streams = set()
        l_streams = set()
        for r in per_token:
            q_streams.update(r["stream_handles"].get("Query", []))
            l_streams.update(r["stream_handles"].get("Loading", []))
        cross["stream_check"] = {
            "Query_stream_handles": sorted(q_streams),
            "Loading_stream_handles": sorted(l_streams),
            "same_stream": bool(q_streams & l_streams),
        }
        if q_streams & l_streams:
            _warn(f"row {i}: Loading was recorded on stream "
                  f"{sorted(q_streams & l_streams)}, the same stream as Query -- the "
                  f"c2g stream was NOT resolved; Loading is not a real measurement.")
        if stats["n_device_path_layers"] == 0 and stats["n_layers_with_scan"] == 0:
            _warn(f"row {i}: retrieval_stats has no page_scan layer with a recorded "
                  f"query latency -- if PageScan ran its numpy fallback (_reps_t is "
                  f"None, page_scan.py:616) the Query bucket is host-only and ~0.")

        report["rows"].append({
            "row_index": i,
            "sample_id": hrow.get("sample_id"),
            "backend": hrow.get("backend"),
            "actual_backend": hrow.get("actual_backend"),
            "inject_pages": hrow.get("inject_pages"),
            "prompt_tokens": hrow.get("prompt_tokens"),
            "generated_tokens": hrow.get("generated_tokens"),
            "n_decode_intervals": n_intervals,
            "marks_s": marks,
            "intervals_s": [marks[k] - marks[k - 1] for k in range(1, n_intervals + 1)],
            "per_token": per_token,
            "means": means,
            "wrapper_host": _wrapper_host_stats(i, n_intervals),
            "components": components,
            "prefill_window_calls": pre0_calls,
            "event_read_notes": sorted(notes),
            "frame_read_notes": sorted(frame_notes),
            "threads_seen": sorted({_thr for (r, _t, _thr, _n) in HOST if r == i} |
                                   {th for cell in idx_row.values()
                                    for lst in cell.values() for (_s, _e, _h, th) in lst}),
            "harness_retrieval_stats": hrow.get("retrieval_stats"),
            "harness_query_cross_check": cross,
            "harness_row": {
                k: hrow.get(k) for k in
                ("ttft_s", "tt2t_s", "tpot_s", "total_s", "score", "page_size",
                 "page_budget", "page_topk", "page_scan_defer_write", "threads",
                 "model", "dataset", "seed", "gpu_peak_allocated_bytes",
                 "host_peak_rss_bytes", "argv")
            },
        })

    # Raw cells, so anything above can be recomputed without the JSONL. Device
    # cells are aggregated (a per-call list for every name would be ~10^5 floats);
    # the four buckets keep their per-call lists.
    bucket_leaves = {leaf for leaves in BUCKET_LEAVES.values() for leaf in leaves} | {"Loading.copy"}
    raw = {}
    raw_bucket_ms = {}
    raw_notes = set()
    for (r, t, n), cell in sorted(EV.items()):
        ms_list = [_read_pair(s, e, raw_notes) for (s, e, _h, _th) in cell]
        ms_list = [m for m in ms_list if m is not None]
        raw[f"{r}|{t}|{n}"] = [len(cell), sum(ms_list)]
        if n in bucket_leaves:
            raw_bucket_ms[f"{r}|{t}|{n}"] = ms_list
    report["raw"] = {
        # Wrapper self-cost cells: [calls, seconds, inside_frame_calls,
        # inside_frame_seconds] per (row, token, wrapper name). The rollup is
        # `rows[].wrapper_host`; this is the un-rounded source.
        "wrapper_host_cells": {
            f"{r}|{t}|{WRAP_NAMES[i]}": c
            for (r, t), cells in sorted(WRAP_CELLS.items())
            for i, c in enumerate(cells) if c[0]},
        "device_cells": raw,
        "host_cells": {f"{r}|{t}|{thr}|{n}": [c, s]
                       for (r, t, thr, n), (c, s) in sorted(HOST.items())},
        "device_bucket_per_call_ms": raw_bucket_ms,
        "notes": sorted(raw_notes),
    }
    return report


def comp_names(idx_row):
    """Every wrapper name that appears anywhere in one row's device cells."""
    names = set()
    for cell in idx_row.values():
        names.update(cell.keys())
    return names


# --------------------------------------------------------------------- printing
def _num(x, width=9, prec=3):
    return f"{'n/a':>{width}}" if x is None else f"{x:>{width}.{prec}f}"


def _print_table(rep):
    for e in rep["probe"].get("errors", []):
        # Last line only: a traceback is already printed in full by main().
        print("PROBE ERROR: " + str(e).strip().splitlines()[-1], flush=True)
    for row in rep["rows"]:
        pt = row["per_token"]
        hdr = (f"row {row['row_index']} sample={row['sample_id']} "
               f"backend={row['backend']} actual={row['actual_backend']} "
               f"prompt_tok={row['prompt_tokens']} gen={row['generated_tokens']} "
               f"decode intervals={row['n_decode_intervals']}")
        print("\n" + "=" * 100, flush=True)
        print(hdr, flush=True)
        if not pt:
            print("  (no decode intervals: the row has no token marks)", flush=True)
            continue
        print("  device intervals from CUDA events; interval = the harness's own "
              "host interval between two token marks", flush=True)
        print(f"{'token':>6} {'interval':>10} {'Query':>9} {'Loading':>9} "
              f"{'Decoding':>9} {'Others':>9}   (ms)", flush=True)
        shown = pt if len(pt) <= 64 else pt[:8] + [None] + pt[-8:]
        for r in shown:
            if r is None:
                print(f"{'...':>6}", flush=True)
                continue
            flag = "  <-- NEGATIVE Others" if r["negative_others"] else ""
            if r["unreadable_pairs"]:
                flag += f"  <-- {r['unreadable_pairs']} pair(s) UNREADABLE (bucket(s) 0)"
            print(f"{r['token']:>6} {r['interval_ms']:>10.3f} "
                  f"{r['Query_ms']:>9.3f} {r['Loading_ms']:>9.3f} "
                  f"{r['Decoding_ms']:>9.3f} {r['Others_ms']:>9.3f}{flag}",
                  flush=True)
        m = row["means"]
        print(f"{'MEAN':>6} {_num(m['interval_ms'], 10)} {_num(m['Query_ms'])} "
              f"{_num(m['Loading_ms'])} {_num(m['Decoding_ms'])} "
              f"{_num(m['Others_from_interval_ms'])}", flush=True)

        # -- the stack, or a loud refusal to draw one -------------------------
        if m["shares_pct"]:
            s = m["shares_pct"]
            print(f"       share of the mean interval: Query {s['Query_pct']:5.1f}%  "
                  f"Loading {s['Loading_pct']:5.1f}%  Decoding {s['Decoding_pct']:5.1f}%  "
                  f"Others {s['Others_pct']:5.1f}%   "
                  f"(Query+Loading+Decoding = {s['bucket_sum_pct']:5.1f}% of the interval)"
                  f"   <-- DENOMINATOR IS THE HOST INTERVAL: probe-inflated, DO NOT PLOT",
                  flush=True)
        else:
            n_neg = m["n_negative_others"]
            print("", flush=True)
            print("  " + "!" * 88, flush=True)
            if n_neg:
                print(f"  !! NEGATIVE OTHERS on {n_neg}/{len(pt)} decode tokens "
                      f"(tokens {m['negative_others_tokens'][:12]}"
                      f"{'...' if n_neg > 12 else ''}).", flush=True)
                print("  !! The four buckets do NOT stack here, so no share line is printed.", flush=True)
                print("  !! Mechanism: Loading is measured on InferState's c2g_stream while the", flush=True)
                print("  !! default stream may still be draining the previous layer's MLP/sdpa,", flush=True)
                print("  !! so the same wall-clock window is billed to two buckets at once and", flush=True)
                print("  !! the token interval is not an upper bound on their sum. Read the", flush=True)
                print("  !! per-bucket ms below; do NOT present this row as a 100% stacked bar.", flush=True)
            else:
                print(f"  !! NO SHARE LINE: {len(row['event_read_notes'])} event pair(s) could not", flush=True)
                print("  !! be read, so an affected token reports a bucket as 0.0 ms when it was", flush=True)
                print("  !! simply not measured. Treat those tokens as MISSING, not as zero:", flush=True)
                print("  !! the shares would be a fabrication. The per-bucket ms are still", flush=True)
                print("  !! read out below.", flush=True)
            print("  " + "!" * 88, flush=True)

        # -- the SAME buckets on the inflation-free denominator ----------------
        # Two share sets, two denominators, NOT interchangeable -- see WHICH SHARES
        # TO PLOT in the module docstring. The interval-denominated one above is
        # measured against wall clock the probe itself inflated; this one is
        # measured entirely in CUDA-event intervals.
        fs = m.get("frame_shares_pct")
        if fs:
            print(f"       share of the device frame d:attn_frame "
                  f"({fs['denominator_ms']:.3f} ms = "
                  f"{fs['denominator_calls_per_token']:.1f} call(s)/token, CUDA events): "
                  f"Query {fs['Query_pct']:5.1f}%  Loading {fs['Loading_pct']:5.1f}%  "
                  f"Decoding {fs['Decoding_pct']:5.1f}%  Others {fs['Others_pct']:5.1f}%"
                  f"   <-- PLOT THIS SET", flush=True)
            print(f"       (denominator rule: the interval-denominated set is % of the "
                  f"harness HOST interval, {_num(m['interval_ms'], 0, 3).strip()} ms, which "
                  f"CONTAINS the probe's own wrapper cost and is therefore inflated; this "
                  f"set is % of a CUDA DEVICE interval and is inflation-free. "
                  f"{fs['note']})", flush=True)
        else:
            print("       (no device-frame share set for this row: d:attn_frame did not "
                  "run, or one of its event pairs was unreadable. Only the inflated, "
                  "interval-denominated set exists here -- do not plot it.)", flush=True)

        # -- per-call device cost, and the cross-check -----------------------
        print(f"  per-call device ms: "
              + "  ".join(f"{b}={_num(m['mean_ms_per_call'][b], 0, 3).strip()}"
                          f"(n={_num(m['calls_per_token'][b], 0, 1).strip()})"
                          for b in rep["probe"]["buckets"]), flush=True)
        if m["Loading_copy_ms"] is not None:
            print(f"  Loading split: copy(H2D+D2D, device)={m['Loading_copy_ms']:.3f} ms  "
                  f"host staging inside the same pair={m['Loading_host_stage_ms']:.3f} ms  "
                  f"(staging = address gather + defer-write join + pinned memcpy; it is "
                  f"stream-idle time, and it is inside the Loading bucket)", flush=True)
        tpot_ms = m["tpot_ms_from_row"]
        o_tpot = m["Others_from_tpot_ms"]
        print(f"  check: mean(interval)={_num(m['interval_ms'], 0, 3)} ms vs harness "
              f"tpot_s={_num(tpot_ms, 0, 3)} ms | Others(tpot)={_num(o_tpot, 0, 3)} ms "
              f"vs Others(interval)={_num(m['Others_from_interval_ms'], 0, 3)} ms",
              flush=True)

        cross = row["harness_query_cross_check"]
        print("  Query cross-check (probe device interval vs the harness's own host "
              "timer for the same region):", flush=True)
        for label in ("Query_bucket", "Query_scan_nested"):
            c = cross.get(label) or {}
            if c.get("probe_ms_per_call") is None:
                continue
            print(f"    {label:18s} probe={c['probe_ms_per_call']:.3f} ms/call "
                  f"n={c['probe_calls']}  |  harness "
                  f"query.page_scan p50={_num(c.get('retrieval_stats.query.page_scan.p50_ms'), 0, 3)} "
                  f"(n={c.get('retrieval_stats.query.page_scan.count')})  "
                  f"page_scan_layers p50(mean)="
                  f"{_num(c.get('retrieval_stats.page_scan_layers[].query.p50_ms'), 0, 3)}  "
                  f"-> {c.get('verdict')}", flush=True)
        sc = cross.get("stream_check") or {}
        print(f"    stream check: Query on {sc.get('Query_stream_handles')}  "
              f"Loading on {sc.get('Loading_stream_handles')}  "
              f"same_stream={sc.get('same_stream')} "
              f"({'OK, Loading is on its own stream' if not sc.get('same_stream') else 'BROKEN'})",
              flush=True)
        if row["prefill_window_calls"]:
            print(f"    prefill window (token index 0, excluded from the decode mean): "
                  f"{row['prefill_window_calls']}", flush=True)
        if row["components"]:
            print("  components (device intervals unless marked HOST; nested names "
                  "overlap their bucket -- do not add):", flush=True)
            for name, c in row["components"].items():
                tag = "HOST " if c["kind"] == "host" else "     "
                print(f"    {name:26s}{tag} calls={c['calls']:>6} "
                      f"mean={_num(c['mean_ms'])} ms  "
                      f"per_call={_num(c['mean_ms_per_call'])} ms "
                      f"sum={c['sum_ms']:>9.1f} ms   [{c.get('note')}]", flush=True)
        if row["event_read_notes"]:
            print("  EVENT READ NOTES: " + "; ".join(row["event_read_notes"]), flush=True)
        print(f"  threads seen: {row['threads_seen']}", flush=True)
        if row["actual_backend"] == "no_retrieval":
            print("  WARNING: actual_backend=no_retrieval -- no retrieval ran; the "
                  "Query/Loading buckets are structural zeros, not a measurement.",
                  flush=True)
        # The probe's own cost, MEASURED rather than estimated. The earlier revision
        # printed "~1 us each of host+device time" here, which was a guess and was
        # wrong by orders of magnitude; every wrapper now times its own host work.
        n_rec = sum(c["calls"] for c in row["components"].values() if c["kind"] == "device")
        if row["n_decode_intervals"]:
            print(f"  probe overhead: {n_rec} device event records over "
                  f"{row['n_decode_intervals']} decode tokens "
                  f"({n_rec / row['n_decode_intervals']:.0f}/token)", flush=True)
        wh = row.get("wrapper_host") or {}
        if wh and row["n_decode_intervals"]:
            per_b = "  ".join(f"{b}={v:.3f}" for b, v in
                              wh["per_bucket_ms_per_token"].items())
            print(f"  probe overhead (MEASURED wrapper host time, not estimated): "
                  f"{wh['total_ms_per_token']:.3f} ms/token over "
                  f"{wh['calls_per_token']:.0f} wrapped calls/token  [{per_b}]  (ms/token)",
                  flush=True)
            print(f"    split: {wh['inside_attn_frame_ms_per_token']:.3f} ms/token is charged "
                  f"INSIDE the d:attn_frame interval and "
                  f"{wh['outside_attn_frame_ms_per_token']:.3f} ms/token OUTSIDE it. "
                  f"Only the inside part can inflate the frame denominator "
                  f"({_num(m['attn_frame_ms'], 0, 3).strip()} ms/token, "
                  f"{_num(m['attn_frame_calls_per_token'], 0, 1).strip()} call(s)/token).",
                  flush=True)
            tpot_ms = m["tpot_ms_from_row"]
            frame_ms = m["attn_frame_ms"]
            if tpot_ms and frame_ms:
                clean = wh["clean_control_tpot_ms"]
                extra = tpot_ms - clean
                inside = wh["inside_attn_frame_ms_per_token"]
                print(f"    reading rule: this row's host interval is {tpot_ms:.3f} ms vs the "
                      f"clean no-probe control {clean:.2f} ms (documented 4-pass mean on this "
                      f"box, NOT measured by this run) = {100.0 * extra / clean:+.0f}%, i.e. a "
                      f"delta of {extra:+.3f} ms/token.", flush=True)
                print(f"    of that delta, {inside:.3f} ms/token is charged INSIDE the "
                      f"d:attn_frame interval and {extra - inside:+.3f} ms/token OUTSIDE it. "
                      f"The inside part is {100.0 * inside / frame_ms:+.1f}% of the frame's "
                      f"own {frame_ms:.3f} ms: that fraction is the frame denominator's "
                      f"inflation.",
                      flush=True)
    if not rep["rows"]:
        print("PROBE: no harness rows to report on.", flush=True)
    for w in rep["probe"].get("warnings", []):
        print("PROBE WARNING: " + w, flush=True)


# -------------------------------------------------------------------------- main
def main():
    argv = list(sys.argv)
    probe_out = None
    # Strip --probe-out (either spelling) so the harness's argparse never sees it
    # and the argv it records in the row stays a valid invocation.
    rest = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--probe-out":
            if i + 1 >= len(argv):
                raise SystemExit("--probe-out needs a path")
            probe_out = argv[i + 1]
            i += 2
            continue
        if a.startswith("--probe-out="):
            probe_out = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    sys.argv = [argv[0]] + rest

    if not rest:
        raise SystemExit(__doc__.split("USAGE")[-1].strip())

    # Before anything is patched or measured: say out loud if the tree/kernels this
    # run will use are not the ones this file lives in.
    _check_tree()

    install_path_probe()
    install_mlp_hook()
    install_token_timer()

    harness_out = None
    for j, a in enumerate(rest):
        if a == "--output" and j + 1 < len(rest):
            harness_out = rest[j + 1]
        elif a.startswith("--output="):
            harness_out = a.split("=", 1)[1]
    if harness_out is None:
        print("PROBE WARNING: no --output found in argv; cannot read the harness row back",
              flush=True)

    print("=" * 100, flush=True)
    print(f"PROBE: repo={REPO}", flush=True)
    print(f"PROBE: page_scan_compare={getattr(psc, '__file__', None)}", flush=True)
    print(f"PROBE: icecache={getattr(istate, '__file__', None)}", flush=True)
    # The compiled kernels, for the same reason the JSON records them: a swapped .so
    # is a different binary and the row would look identical without this line.
    print(f"PROBE: icecache_cpp={getattr(sys.modules.get('icecache_cpp'), '__file__', None)}",
          flush=True)
    print(f"PROBE: dciknn._dci={getattr(sys.modules.get('dciknn._dci'), '__file__', None)}",
          flush=True)
    # The banner a reader of the log actually sees. It must not repeat the claim
    # that the probe leaves the harness's timings alone: it adds no SYNCHRONISATION,
    # but its wrappers' own host cost is inside the token interval and is exposed by
    # this host-bound decode (measured +26% on this box; see PERTURBATION).
    print("PROBE: four buckets are CUDA device intervals; the probe adds no "
          "synchronisation -- but its own wrapper host cost IS inside the token "
          "interval, so THIS ROW's tpot_s/ttft_s are probe-inflated (measured +26% vs "
          "the clean 4-pass mean 89.53 ms) and must not be quoted as clean numbers. "
          "The inflation-free shares are the frame-denominated ones "
          "('share of the device frame d:attn_frame'); the interval-denominated ones "
          "are inflated.", flush=True)
    for line in _INSTALLED:
        print(f"PROBE: {line}", flush=True)
    print("=" * 100, flush=True)

    error = None
    try:
        psc.main()
    except BaseException:
        error = traceback.format_exc()
        print("PROBE: page_scan_compare.main() raised:\n" + error, flush=True)

    report = build_report(harness_out, argv, error)
    _print_table(report)

    if probe_out:
        p = Path(probe_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                     encoding="utf-8")
        print(f"\nPROBE: wrote {p}", flush=True)
    else:
        print("\nPROBE: no --probe-out given; nothing written to disk", flush=True)

    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
