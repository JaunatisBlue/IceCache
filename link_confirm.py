#!/usr/bin/env python3
"""Confirm the H2D/D2H ceiling and check whether it is symmetric (link-limited)."""
import time
import torch

dev = "cuda"
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
torch.cuda.init()

def bench(direction, kb=2048, iters=200):
    n = kb * 1024 // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dev_t = torch.empty(n, dtype=torch.float16, device=dev)
    s = torch.cuda.Stream()
    for _ in range(20):
        with torch.cuda.stream(s):
            if direction == "H2D":
                dev_t.copy_(host, non_blocking=True)
            else:
                host.copy_(dev_t, non_blocking=True)
    s.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s):
        e0.record(s)
        for _ in range(iters):
            if direction == "H2D":
                dev_t.copy_(host, non_blocking=True)
            else:
                host.copy_(dev_t, non_blocking=True)
        e1.record(s)
    s.synchronize()
    ms = e0.elapsed_time(e1) / iters
    gbps = (kb * 1024) / (ms / 1000) / 1e9
    print("  %s %4d KB : %8.1f us  %.2f GB/s" % (direction, kb, ms * 1000, gbps))

print("=== directional bandwidth ===")
bench("H2D"); bench("D2H")

print("=== device-to-device (control: should be ~1-2 TB/s) ===")
n = 64 * 1024 * 1024 // 2
a = torch.empty(n, dtype=torch.float16, device=dev)
b = torch.empty(n, dtype=torch.float16, device=dev)
for _ in range(5):
    b.copy_(a)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    b.copy_(a)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 20
print("  D2D 64 MB : %.3f ms  %.1f GB/s" % (ms * 1000, (64 * 1024**2) / ms / 1e9))
