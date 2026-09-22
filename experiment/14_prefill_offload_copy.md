# `explore/prefill-fast` — the prefill offload copy

Worktree `/home/yx/IceCache/.claude/worktrees/prefill-fast`, off `algorithm@ef228fa`.
One commit: **`4b1b939`** — `IceCache/source/icecache/infer_state.py`, +46/−3.

## 0. Bottom line

Real, shared, bit-identical and large: **−1.67 s mean paired TTFT on DCI and
−1.61 s on page_scan over 9 rows, 9/9 wins each** (0.65× / 0.70× TTFT at 16k).
46 lines in one file plus one 131 MB device buffer.

**Honest consequence:** the path is **shared by both backends**, so both arms gain
the same *absolute* amount and the page_scan/DCI TTFT **ratio gets slightly
worse**: 6.147/5.299 = 1.160 → 4.283/3.434 = 1.247. This does not close the
§12.2 gap; it moves both arms down. The remaining page_scan gap is the serial
flush, not the offload path.

## 1. Is the path shared or DCI-only? — **SHARED**

`adapter/modeling.py:129-148` submits `prefill_evict_extra_pages_wrapper` on
every prefill layer whose `state.layer2budget[cur_id] is not None`. Nothing in
that gate, in `prefill_evict_extra_pages` (infer_state.py:1623) or
`prefill_backup_pages` (:1505) references `retrieval_backend` (merged-`algorithm`
numbering; the reports' other line references to these functions predate the
merge and are one revision off). Measured: at row
12, DCI gained −1.865 s and page_scan −1.864 s — same absolute amount to within
1 ms.

## 2. What changed

`prefill_backup_pages` did `dst.copy_(src)` with an **fp16 device** source and an
**fp32 pinned host** destination. CUDA cannot DMA a cross-dtype copy, so it fell
back to an element-by-element host store. The fix: upcast on the device into a
reused fp32 staging buffer, then DMA the same-dtype result with
`non_blocking=True` into the pinned CPU pool, and sync the worker's own stream
instead of the device.

Three separable effects:

1. the cast moves to the device — **this is the win**;
2. the device-wide `torch.cuda.synchronize()` is replaced by `Stream.synchronize()`
   on the stream the copy was issued on — the old sync waited on the *main
   thread's* kernels and was 272 ms/row by itself;
3. the staging buffer is reused rather than reallocated — **not** a speed claim
   (`.to()` and the buffer form measure identically), it removes 34 × 131 MB of
   allocator traffic per prompt.

## 3. Re-measured breakdown and the bandwidth calculation

Offline, live row-12 shape (998 pages, 65.4 MB fp16 in, 130.8 MB fp32 out),
median of 15, stream sync inside the timed region.

| variant | ms | GB/s |
|---|---|---|
| fp16 dev → fp16 pinned, bare DMA | 4.99 | 26.2 |
| fp32 dev → fp32 pinned, bare DMA | 9.97 | 13.1 |
| fp16 dev → fp32 pinned, `dst.copy_(src)` **(shipped)** | **57.43** | **2.28** |
| upcast on device, then fp32 pinned DMA **(this)** | **10.28** | 12.72 |

Back of envelope, verified: 34 × 16022 × 8 × 128 × 2 × 2 B = 1.74 GB fp16 read +
3.5 GB fp32 written per prompt. At PCIe Gen4 x16 that is ~200 ms, not 2.7 s.
**The op is overhead-bound, not bandwidth-bound** — the bare op runs 5.8× below
the fp32 pinned-DMA rate the *identical byte traffic* achieves when dtypes match,
so the deficit is the cast-in-the-copy path. The fix recovers 5.3× of it.

## 4. Live decomposition (worker thread, row 12, DCI)

| stage | before | after | n |
|---|---|---|---|
| `prefill_evict_extra_pages` total | 5223.1 ms | 2577.7 ms | |
| — `prefill_backup_pages` | 2697.6 | 802.6 | 34 |
| — — the copy op | 2301.8 | 677.6 (in the stream sync) | 34 |
| — — device-wide `cuda.synchronize` | 272.0 | **gone** | 34 |
| — `_DCI_first_call` (incl. `dci.C.add_query_at_end` 431.2) | 1136.2 | 1181.5 | 34 |
| — `permute`/`reshape` of the offloaded block | 534.7 | 562.2 | 34 |
| — `clear()` | 23.9 | 25.7 | 34 |
| TTFT (same process) | 5.94 s | 3.88 s | |

**The 1389 ms residual is now fully attributed** — it was not a hidden stage. The
only material work left is `permute`/`reshape` (15.7 ms/layer, 534.7 ms total)
and `clear()` (0.7 ms/layer); 4506 ms of the 4496 measured. The extra ~800 ms the
first instrument showed was run-to-run variance in the worker (same code, same
composition, 5223 vs 4496 ms in two runs). There is no fifth thing.

`permute`/`reshape` is ~390 MB of CPU traffic per layer: `tmp_cpu_kvc[b, :n]` is
*advanced indexing*, so already a 131 MB gather, and the
`permute(1,0,2,3,4).permute(0,2,1,3,4).reshape(...)` cannot be a view
(stride(998) = 32768 ≠ 16·128), so it copies another 131 MB.

page_scan row 12 after: `_finish_prefill` 1564.1 ms serial on the main thread (of
which `_page_scan_flush` 1563.9 = batched greedy + 8-thread page writes at
~287 ms wall), worker total 1792.8 ms (fixed copy 809.5, permute 550.5,
`_page_scan_first_call` 406.0), TTFT 4743 ms. Row 7 contrast: DCI worker
734.0 → 628.4 ms, TTFT 1.098 s; the fix scales with prompt length.

## 5. Output-identity evidence

**(a) The op, real shape, 32.7M elements:** `torch.equal` **True**.

**(b) The shipped function, real shape.** `pf_verify.py` builds a real
`KvCache`/`KvPool` at live geometry (page 16 × 8 heads × 128 dim, 994 offload
pages, fp16 device → fp32 pinned host), calls the worktree's real
`prefill_backup_pages` and a verbatim copy of the old body **on identical source
data**:

```
num_offload_pages   : 994 994
fp16 in 62.1 MB -> fp32 out 124.2 MB
torch.equal(dst)    : True
max abs diff        : 0.0
orig median ms      : 56.10   staged median ms : 10.60   speedup 5.29x
```

**(c) Live per row, 9 hotpotqa rows, one process per arm:** page_scan **9/9**
text-identical and 9/9 scores identical in both arms; DCI 8/9 text, 9/9 scores.

**Caveat — measured, not assumed.** Repeat-run control (unmodified twice, fixed
twice, one process per arm):

| pair | text differs | score differs |
|---|---|---|
| unmodified vs unmodified | 1/9 (row 6) | 1/9 (1.0 → 0.2222) |
| fixed vs fixed | 1/9 (row 1) | 0/9 |
| before vs after (the A/B pair) | 1/9 (row 1) | 0/9 |

The DCI arm has ~1-in-9 row-level nondeterminism independent of this change, the
affected row differs each time, and the *unmodified* repeat moved a score.
page_scan (the backend that matters going forward) is 9/9 in both arms. Timing is
reproducible at the total level (two unmodified repeats 44.601 s vs 44.639 s,
0.09% apart), which is why the TTFT deltas are trustworthy where per-row text is
not.

## 6. Live paired A/B

`quiet-run.sh`, `--threads 64 --n-reuse-layers 3 --page-budget 16`,
`--max-new-tokens 8`, one backend per process, one arm per process; before arms
via `git stash`.

### DCI — mean paired TTFT delta −1.670 s, 9 wins / 0 losses

| row | ptok | before | after | delta | ratio | TPOT Δ |
|---|---|---|---|---|---|---|
| 0 | 11770 | 4.310 | 2.993 | −1.317 | 0.694 | +9.3 |
| 1 | 17305 | 5.933 | 3.953 | −1.980 | 0.666 | +3.2 |
| 2 | 17371 | 5.986 | 3.927 | −2.060 | 0.656 | +0.6 |
| 3 | 16747 | 5.776 | 3.743 | −2.033 | 0.648 | +19.8 |
| 4 | 15153 | 5.247 | 3.405 | −1.842 | 0.649 | +5.8 |
| 5 | 15733 | 5.399 | 3.517 | −1.882 | 0.651 | −2.2 |
| 6 | 16111 | 5.550 | 3.634 | −1.917 | 0.655 | +14.9 |
| 7 | 2300 | 1.100 | 0.965 | −0.135 | 0.878 | +2.4 |
| 12 | 16022 | 5.299 | 3.434 | −1.865 | 0.648 | +18.4 |

Totals 44.601 → 29.571 s. TPOT mean +8.02 ms, 1 win / 8 losses, p = 0.039.

### page_scan — mean paired TTFT delta −1.614 s, 9 wins / 0 losses

| row | ptok | before | after | delta | ratio | TPOT Δ |
|---|---|---|---|---|---|---|
| 0 | 11770 | 4.914 | 3.532 | −1.382 | 0.719 | +2.1 |
| 1 | 17305 | 6.754 | 4.783 | −1.971 | 0.708 | +0.3 |
| 2 | 17371 | 6.742 | 4.842 | −1.900 | 0.718 | +0.7 |
| 3 | 16747 | 6.471 | 4.576 | −1.896 | 0.707 | +1.9 |
| 4 | 15153 | 5.832 | 4.114 | −1.719 | 0.705 | +0.4 |
| 5 | 15733 | 6.093 | 4.323 | −1.770 | 0.710 | +24.8 |
| 6 | 16111 | 6.339 | 4.375 | −1.964 | 0.690 | −0.0 |
| 7 | 2300 | 0.773 | 0.712 | −0.061 | 0.921 | −0.4 |
| 12 | 16022 | 6.147 | 4.283 | −1.864 | 0.697 | −0.4 |

Totals 50.064 → 35.539 s. TPOT mean +3.27 ms, 3 wins / 6 losses, p = 0.51.

Sign tests are exact binomial: one-sided p = 0.0020 for 9/9, two-sided 0.0039.

**On the DCI TPOT +8 ms:** reported as measured, but not a regression from this
change. At `--max-new-tokens 8` with 2–7 tokens actually generated, "TPOT" is a
1–6 interval statistic dominated by the first interval. page_scan, whose TPOT is
stable across the run (94–98 ms in every row of both arms), shows +3.27 ms at
p = 0.51 — no effect.

## 7. What was rejected / deliberately not done

* **Collapsing the two `permute`s into `permute(1,2,0,3,4)` and slicing `c2p`
  instead of gathering** (worth ~534 ms + half the permute cost at 16k). Rejected
  **on measurement, not effort**: after the fix the worker is no longer the
  critical path — `prefill_evict_extra_pages` is 2578 ms against a 3880 ms DCI
  TTFT and 1793 ms against 4743 ms for page_scan — so shaving it further cannot
  move TTFT, and it would put view-aliasing semantics into `_DCI_first_call` for
  nothing.
* **page_scan's `_finish_prefill` tail** (1564 ms of a 4743 ms TTFT at 16k,
  serial on the main thread, almost all `_page_scan_flush` = batched greedy +
  threaded page write). Now the largest remaining TTFT item in the system and the
  obvious next lever — named, not attempted; the greedy floor is branch D's
  territory.
* Two earlier probe runs were killed and relaunched when their `timeout` would
  have expired queued behind another agent's 200-row sweep.

## 8. Provenance

Every time reported here came from `quiet-run.sh` (both locks). The offline
micro-benchmark, the function-level identity check, the live decomposition probes
and both A/B arms all ran under `quiet-run.sh`. Nothing was timed anywhere else.

Scratch: `/home/yx/.claude/jobs/497cc41a/tmp/pf*` — `pf_probe.py`,
`pf_probe2.py`, `pf_micro*.py`, `pf_verify.py`, `pf_ab.sh`, `pf_repeat.sh`,
`pf_paired.py`, `pf_all.sh`; results `pfAB-{dci,page_scan}-{before,after}*.jsonl`,
`pf_ship_*.json`, `pf_orig_dci.json`, `pf_probe_dci_detail.json`.
