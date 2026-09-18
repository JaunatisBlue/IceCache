"""Page-allocation invariants that chunked prefill / mixed batching relies on.

Two things are checked here, and they are different in kind:

* **Policy invariants of the two-ended allocator** -- single pages for decode come
  from the low end, contiguous runs for prefill from the high end, and a run is
  released high up so decode growth cannot take it back one page at a time.

* **The reservation contract** ``reserve_prefill_pages`` + ``prefill_alloc_n_tokens``
  -- a chunked prefill must find its pages *already adjacent* to the previous
  chunk's.  The previous revision replaced ``c2p`` with a fresh run sized only for
  the delta on every call, which leaked the run it discarded and put the second
  chunk in an unrelated block, so chunking could not work at all. That contract is
  the precondition for everything in the mixed prefill/decode path.

What is deliberately *not* asserted: that the old policy fragments where the new
one does not.  An allocated run cannot be taken by decode in either policy (decode
only draws from the free set), so the fragmentation hypothesis did not reproduce
as a binding constraint -- the observed failure was capacity, not fragmentation.
``churn()`` prints both policies' largest free run for information only.

Run: python benchmark/test_page_pool_two_ended.py
"""

import sys

import torch

sys.path.insert(0, "/home/yx/IceCache/IceCache/source")

from icecache.kv_cache import KvCache, KvPool, PagePool  # noqa: E402

PAGE = 16
DEV = torch.device("cpu")


class LegacyPagePool(PagePool):
    """The pre-change policy: smallest free single page, lowest contiguous run."""

    def alloc_page(self):
        return self._free_ids.pop()

    def alloc_contiguous_pages(self, num):
        if num <= 0:
            return []
        if len(self._free_ids) < num:
            return None
        sorted_ids = sorted(self._free_ids)
        for i in range(len(sorted_ids) - num + 1):
            if sorted_ids[i + num - 1] - sorted_ids[i] == num - 1:
                result = list(range(sorted_ids[i], sorted_ids[i] + num))
                self._free_ids -= set(result)
                return result
        return None


def make(pool_cls, n_pages):
    return pool_cls(n_pages, (1,), torch.float16, DEV)


def make_cache(n_pages):
    pool = KvPool(n_pages, PAGE, 1, 1, torch.float16, DEV, (0, 1, 2, 3))
    return KvCache(pool, batch_size=1), pool


def churn(pool):
    """A retire/admit cycle: allocate runs, free some, then let decode grow."""
    runs = [pool.alloc_contiguous_pages(400) for _ in range(4)]
    for index in (0, 2):
        if runs[index]:
            for page_id in runs[index]:
                pool.free_page(page_id)
    for _ in range(1200):
        pool.alloc_page()
    return pool.max_contiguous_free_pages()


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def main():
    failures = 0

    def check(name, condition, detail=""):
        nonlocal failures
        print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not condition:
            failures += 1

    print("1. 取向：单页走低位、run 走高位（且 run 升序返回）")
    pool = make(PagePool, 64)
    check("连续 run 取自最高位且升序返回",
          pool.alloc_contiguous_pages(8) == [56, 57, 58, 59, 60, 61, 62, 63])
    check("单页取自最低位", pool.alloc_page() == 0)

    print()
    print("2. 回收")
    pool.alloc_page()
    pool.alloc_page()                       # -> 1, 2 ; 前向游标到 3
    pool.free_page(2)
    pool.free_page(0)
    check("被游标越过的回收页仍会被 alloc_page 复用（LIFO）",
          (pool.alloc_page(), pool.alloc_page(), pool.alloc_page()) == (0, 2, 3))

    pool2 = make(PagePool, 64)
    first = pool2.alloc_contiguous_pages(8)
    second = pool2.alloc_contiguous_pages(8)
    for page_id in first:
        pool2.free_page(page_id)
    check("释放回池的高位 run 会被下一个 run 优先复用",
          pool2.alloc_contiguous_pages(8) == first,
          f"first={first[0]}..{first[-1]} second={second[0]}..{second[-1]}")

    print()
    print("3. decode 只从低位取页：高位 run 不会被它切碎")
    pool3 = make(PagePool, 4096)
    held = [pool3.alloc_contiguous_pages(200) for _ in range(3)]
    for page_id in held[2]:
        pool3.free_page(page_id)
    for _ in range(2000):
        pool3.alloc_page()
    check("2000 次单页分配后，释放的 200 页 run 仍整段可复用",
          pool3.alloc_contiguous_pages(200) == sorted(held[2]),
          f"max_run={pool3.max_contiguous_free_pages()}")
    check("成功路径上失败计数为 0", pool3.n_alloc_run_failures == 0)

    print()
    print("4. 预留契约：chunked prefill 的前提")
    cache, cpool = make_cache(64)
    cache.reserve_prefill_pages(10)
    check("预留一次即拿到整个 prompt 的 run（升序）",
          cache.c2p[0].tolist() == list(range(54, 64)), f"got {cache.c2p[0].tolist()}")
    cache.prefill_alloc_n_tokens(3 * PAGE)
    check("chunk 1 只推进 seq_len，不再分配新页",
          cache.n_real_pages == 10 and cache.n_pages == 3,
          f"owned={cache.n_real_pages} visible={cache.n_pages}")
    cache.prefill_alloc_n_tokens(7 * PAGE)
    check("chunk 2 复用同一个 run，c2p 不变",
          cache.n_real_pages == 10 and cache.n_pages == 10
          and cache.c2p[0].tolist() == list(range(54, 64)))

    check("池里被预留的页确实已从 free 集合中取出",
          cpool.n_free_pages == 54, f"free={cpool.n_free_pages}")

    cache2, _ = make_cache(64)
    cache2.prefill_alloc_n_tokens(3 * PAGE)
    check("没有预留就分第二段：明确报错，而不是静默丢弃上一段",
          _raises(RuntimeError, lambda: cache2.prefill_alloc_n_tokens(4 * PAGE)))

    cache3, _ = make_cache(32)
    check("池里没有足够长的 run 时，报错信息里带所需页数与最大可用 run",
          "largest free run" in _message(lambda: cache3.reserve_prefill_pages(40)))

    print()
    print("5. 真的没有页了")
    pool4 = make(PagePool, 16)
    for _ in range(16):
        pool4.alloc_page()
    check("池空时 alloc_page 抛明确异常", _raises(RuntimeError, pool4.alloc_page))
    check("池空时 alloc_contiguous_pages 返回 None 并计数",
          pool4.alloc_contiguous_pages(1) is None and pool4.n_alloc_run_failures == 1)
    check("max_contiguous_free_pages 在池空时为 0", pool4.max_contiguous_free_pages() == 0)

    print()
    print("6. clear() 复位")
    pool.clear()
    check("free / 游标 / 回收列表都复位",
          (pool.n_free_pages, pool._next_low, pool._reclaimed_low) == (64, 0, []))

    print()
    print("（参考，不判定）同一 retire/admit 扰动下两策略的最大空闲 run：")
    print(f"  legacy    max_run={churn(make(LegacyPagePool, 4096))}")
    print(f"  two-ended max_run={churn(make(PagePool, 4096))}")

    print()
    print(f"failures = {failures}")
    return 1 if failures else 0


def _message(fn):
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
