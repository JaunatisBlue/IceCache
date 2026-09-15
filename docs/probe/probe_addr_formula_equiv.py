"""Unit-level equivalence test for the IceCache CPU page address formula.

Claim under test: for the CPU KV page pool, the byte address of (cache page
`logical_j`, KV head `head_h`) is

    pool.buffer.data_ptr() + c2p[b, logical_j] * page_stride + head_h * head_stride

which must equal the slow reference used by `InferState._DCI_add`:

    cpu_cache[b, logical_j].data_ptr() + head_h * head_stride

The tempting shortcut `pool.buffer.data_ptr() + logical_j * page_stride + ...`
is *not* generally equivalent, because the CPU pool hands out physical pages
from a free-id set and recycles them in place, so `c2p` is fragmented and out
of order during decode.

Coverage: multiple heads, multiple leaves, a fragmented / out-of-order `c2p`,
and a `c2p` growth (the page-table expansion at a decode page boundary).

Run:
    PYTHONPATH=/home/yx/IceCache/IceCache/source \
        /home/yx/miniconda3/envs/icecache/bin/python \
        experiment/probe/probe_addr_formula_equiv.py
"""

import sys

import numpy as np
import torch

PAGE_SIZE = 16
N_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.float32
PAGE_STRIDE = 2 * PAGE_SIZE * N_KV_HEADS * HEAD_DIM * DTYPE.itemsize
HEAD_STRIDE = PAGE_SIZE * HEAD_DIM * DTYPE.itemsize


def build(n_max_pages=64, budget=16, batch_size=1):
    from icecache.kv_cache import KvPool, KvCache

    pool = KvPool(n_max_pages, PAGE_SIZE, N_KV_HEADS, HEAD_DIM, DTYPE,
                  torch.device("cpu"))
    kvc = KvCache(pool=pool, batch_size=batch_size, budget=budget,
                  n_sink_pages=2, n_win_pages=2, n_groups=1, offload_ratio=2)
    return pool, kvc


RESULTS = []


def check(kvc, label, expect_logical_ok=None):
    base = kvc.pool.buffer.data_ptr()
    c2p = kvc.c2p.numpy()
    n_pages = int(c2p.shape[-1])
    n_total = n_fast_bad = n_logical_bad = 0
    for b in range(kvc.batch_size):
        for j in range(n_pages):
            for head in range(N_KV_HEADS):
                slow = int(kvc[b, j].data_ptr()) + head * HEAD_STRIDE
                phys = int(c2p[b, j])
                fast = base + phys * PAGE_STRIDE + head * HEAD_STRIDE
                logical = base + j * PAGE_STRIDE + head * HEAD_STRIDE
                n_total += 1
                n_fast_bad += int(fast != slow)
                n_logical_bad += int(logical != slow)
    phys_ids = sorted(int(x) for x in np.unique(c2p))
    ok = (n_fast_bad == 0)
    if expect_logical_ok is not None:
        ok = ok and (n_logical_bad == 0) == expect_logical_ok
    print("[%-28s] pages=%-3d checks=%-5d fast_mismatch=%-3d "
          "logical_shortcut_mismatch=%-4d c2p[0:12]=%s"
          % (label, n_pages, n_total, n_fast_bad, n_logical_bad,
             list(int(x) for x in c2p[0][:12])))
    RESULTS.append((label, ok, n_total, n_fast_bad, n_logical_bad, phys_ids))
    return n_fast_bad, n_logical_bad


def main():
    print("page_stride=%d head_stride=%d (page_size=%d heads=%d head_dim=%d)"
          % (PAGE_STRIDE, HEAD_STRIDE, PAGE_SIZE, N_KV_HEADS, HEAD_DIM))

    # --- 1. contiguous prefill allocation -------------------------------
    # Prefill allocates a contiguous block, so here logical ids *are* physical.
    pool, kvc = build()
    kvc.prefill_alloc_n_tokens(8 * PAGE_SIZE)
    check(kvc, "prefill contiguous", expect_logical_ok=True)
    assert kvc.c2p.numpy()[0].tolist() == list(range(8)), "not contiguous"

    # --- 2. real free/realloc churn (fragmentation from the allocator) ---
    for p in range(0, 8, 2):
        pool.free_page(p)
    kvc.decode_alloc_n_tokens(4 * PAGE_SIZE)
    check(kvc, "free/realloc churn")

    # --- 3. explicit out-of-order c2p (synthetic permutation) -----------
    # A pure address-formula test: the pool does not care which physical page
    # a cache page points at, and after decode-time recycling the mapping is
    # genuinely arbitrary.
    pool2, kvc2 = build()
    kvc2.prefill_alloc_n_tokens(8 * PAGE_SIZE)
    perm = [7, 0, 5, 1, 6, 2, 4, 3]
    import torch as _t
    kvc2.c2p = _t.tensor([perm], dtype=_t.int32)
    check(kvc2, "synthetic out-of-order", expect_logical_ok=False)

    # --- 4. c2p growth (page-table expansion across a boundary) ---------
    pool3, kvc3 = build()
    kvc3.prefill_alloc_n_tokens(8 * PAGE_SIZE)
    kvc3.c2p = _t.tensor([[7, 0, 5, 1, 6, 2, 4, 3]], dtype=_t.int32)
    check(kvc3, "before c2p growth")
    # grow the page table exactly the way a decode page boundary does
    kvc3.c2p = torch.cat(
        [kvc3.c2p, torch.tensor([[9, 8, 11, 10]], dtype=torch.int32)], dim=-1)
    check(kvc3, "after c2p growth")

    # --- 5. every physical page exercised at least once -----------------
    seen = set()
    for _, _, _, _, _, phys in RESULTS:
        seen.update(phys)
    print("distinct physical pages touched: %d -> %s" % (len(seen), sorted(seen)))

    bad = [r for r in RESULTS if not r[1]]
    print()
    if bad:
        print("FAIL: %d/%d scenarios violated the expected relation"
              % (len(bad), len(RESULTS)))
        for label, _, n_total, n_fast, n_logical, _ in bad:
            print("   %-28s fast=%d logical=%d of %d"
                  % (label, n_fast, n_logical, n_total))
        return 1
    print("PASS: vectorised formula matched data_ptr() in all %d scenarios"
          % len(RESULTS))
    print("      (the logical-page-id shortcut is wrong once c2p is not "
          "the identity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
