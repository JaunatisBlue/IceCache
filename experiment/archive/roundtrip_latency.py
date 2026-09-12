#!/usr/bin/env python3
"""Quantify the per-round-trip cost of small CPU<->GPU transfers on this link.

Motivation: page_metadata costs 0.217 ms per layer yet writes only ~2 KB, and
the recall path performs several small CPU<->GPU round trips per layer.  If a
small transfer costs ~0.2 ms of *latency* rather than bandwidth, then the CPU
side is dominated by round-trip count, not by data volume, and the fix is to
collapse round trips rather than to move fewer bytes.
"""
import time
import numpy as np
import torch

dev = "cuda"
print("device:", torch.cuda.get_device_name(0))


def timed(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


print()
print("=== small pinned H2D / D2H latency (blocking, includes sync) ===")
print("%10s %14s %14s" % ("size", "H2D us", "D2H us"))
for kb in [0.5, 2, 8, 64]:
    n = int(kb * 1024) // 4
    h = torch.empty(n, dtype=torch.float32, pin_memory=True)
    d = torch.empty(n, dtype=torch.float32, device=dev)
    h2d = timed(lambda: d.copy_(h))
    d2h = timed(lambda: h.copy_(d))
    print("%8.1f KB %12.1f %14.1f" % (kb, h2d, d2h))

print()
print("=== numpy -> GPU tensor creation (the page_metadata pattern) ===")
print("%14s %14s %14s" % ("shape", "cpu->gpu us", "gpu->cpu us"))
for r in [(8, 60), (8, 480)]:
    a = np.arange(r[0] * r[1], dtype=np.int32).reshape(r)
    t_gpu = timed(lambda: torch.tensor(a, dtype=torch.int32, device=dev))
    t = torch.tensor(a, dtype=torch.int32, device=dev)
    t_cpu = timed(lambda: t.cpu())
    print("%14s %12.1f %14.1f" % (str(r), t_gpu, t_cpu))

print()
print("=== CPU tensor -> GPU tensor (already-hosted source) ===")
a = np.arange(8 * 60, dtype=np.int32).reshape(8, 60)
t_cpu = torch.from_numpy(a)
print("  torch.from_numpy(...) .to(cuda): %8.1f us"
      % timed(lambda: t_cpu.to(dev, non_blocking=False)))
print("  torch.from_numpy(...) .to(cuda, non_blocking): %8.1f us"
      % timed(lambda: t_cpu.to(dev, non_blocking=True)))

print()
print("=== trivial CUDA kernel launch / event overhead ===")
s = torch.cuda.Stream()
x = torch.ones(1024, device=dev)
print("  y = x + 1 on default stream: %8.1f us" % timed(lambda: x.add_(1)))
print("  torch.cuda.synchronize():    %8.1f us" % timed(lambda: torch.cuda.synchronize()))

print()
print("=== derived: cost of N round trips per layer, 30 layers per token ===")
for kb in [2]:
    n = int(kb * 1024) // 4
    h = torch.empty(n, dtype=torch.float32, pin_memory=True)
    d = torch.empty(n, dtype=torch.float32, device=dev)
    per = timed(lambda: d.copy_(h))
    print("  %.1f KB H2D = %.1f us -> 30 layers/token x 1 trip = %.2f ms/token"
          % (kb, per, per * 30 / 1000))
