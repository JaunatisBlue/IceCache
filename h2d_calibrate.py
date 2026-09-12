#!/usr/bin/env python3
"""Calibrate pinned H2D on this GPU: bandwidth vs per-call overhead.

Motivation: the ICECACHE_DIAG run showed ~1.67 ms per recall H2D for what
looks like ~0.5 MB.  If a 0.5 MB pinned copy can actually finish in tens of
microseconds, the recall H2D is overhead/serialisation-bound, not
bandwidth-bound -- which changes which direction is worth pursuing.
"""
import time
import torch

dev = "cuda"
sizes_kb = [64, 128, 256, 512, 1024, 2048, 8192]

print("=== A. streaming bandwidth (single event pair around N copies) ===")
print("%10s %12s %12s" % ("size", "per-copy", "GB/s"))
for kb in sizes_kb:
    n = kb * 1024 // 2  # fp16
    src = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dst = torch.empty(n, dtype=torch.float16, device=dev)
    s = torch.cuda.Stream()
    for _ in range(20):
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)
    s.synchronize()
    iters = 300
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s):
        e0.record(s)
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        e1.record(s)
    s.synchronize()
    ms = e0.elapsed_time(e1) / iters
    print("%8d KB %10.1f us %10.2f" % (kb, ms * 1000, (kb * 1024) / (ms / 1000) / 1e9))

print()
print("=== B. per-call event pair (includes per-call overhead + sync) ===")
print("%10s %14s %14s" % ("size", "event-us", "wall-us"))
for kb in [128, 256, 512, 1024, 2048]:
    n = kb * 1024 // 2
    src = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dst = torch.empty(n, dtype=torch.float16, device=dev)
    s = torch.cuda.Stream()
    for _ in range(20):
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)
        s.synchronize()
    iters = 200
    ev_tot = 0.0
    w0 = time.perf_counter()
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(s):
            a.record(s)
            dst.copy_(src, non_blocking=True)
            b.record(s)
        s.synchronize()
        ev_tot += a.elapsed_time(b)
    wall = (time.perf_counter() - w0) / iters
    print("%8d KB %11.1f us %11.1f us" % (kb, ev_tot / iters * 1000, wall * 1e6))

print()
print("=== C. many small copies in one batch (is per-op overhead additive?) ===")
for n_copies in [1, 2, 4, 8, 16, 32]:
    n = 512 * 1024 // 2
    src = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dst = torch.empty(n, dtype=torch.float16, device=dev)
    s = torch.cuda.Stream()
    for _ in range(10):
        with torch.cuda.stream(s):
            for _ in range(n_copies):
                dst.copy_(src, non_blocking=True)
    s.synchronize()
    iters = 50
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s):
        e0.record(s)
        for _ in range(iters):
            for _ in range(n_copies):
                dst.copy_(src, non_blocking=True)
        e1.record(s)
    s.synchronize()
    total = e0.elapsed_time(e1) / iters
    print("  %2d x 512KB -> %8.1f us total, %7.1f us/copy" % (n_copies, total * 1000, total * 1000 / n_copies))
