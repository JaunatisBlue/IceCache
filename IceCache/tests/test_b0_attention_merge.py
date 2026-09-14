"""B0 -- paged split attention: return_lse + LSE merge.

Validates the numerical foundation for chunked continuation prefill:

    out_dense ≈ merge( attention(old, non-causal), attention(chunk, causal) )

Everything here is synthetic tensors -- no checkpoint and no InferState -- so it
runs in seconds and isolates kernel behaviour.

Three facts this file establishes:

1. IceCache's prefill/decode wrappers can already return ``lse`` (the C++ op has
   had ``return_lse`` all along; only the Python binding discarded it).
2. The vendored FlashInfer has **no** ``merge_state``, so ``icecache.kernels.merge_state``
   provides the equivalent merge.
3. ``BatchPrefillWithPagedKVCacheWrapper`` has **no** ``page_valid_entries``, so a
   chunk whose first page also holds already-counted old tokens cannot be
   expressed directly.  Packing the operand into a dense, page-aligned staging
   buffer solves it -- sound because the old-context pass is non-causal and hence
   permutation-invariant over keys.  The naive (non-packed) variant is shown to
   be wrong.

Run::

    cd /home/yx/IceCache/IceCache
    PYTHONPATH=/home/yx/IceCache/IceCache/source \\
      /home/yx/miniconda3/envs/icecache/bin/python -m pytest tests/test_b0_attention_merge.py -v -s
"""

from __future__ import annotations

import math

import pytest
import torch

from icecache import kernels

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA is required")

PAGE_SIZE = 16
HEAD_DIM = 128
N_KV_HEADS = 2
N_QO_HEADS = 2  # ratio 1 -> GROUP_SIZE 1, the only safe dispatch value here
N_PAGES = 8
WORKSPACE_MB = 64

OLD_LEN = 37  # 2 full pages + 5  -> exercises a partial tail page
CHUNK_LEN = 20


def _pool(n_pages=N_PAGES):
    """Paged KV pool in HND layout, matching KvPool's layout_map (0,2,1,3)."""
    return torch.zeros(
        n_pages, 2, N_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.float16, device="cuda"
    )


def _write_range(pool, k, v, start):
    """Write k/v so that logical position start+i lands at page (start+i)//PAGE_SIZE."""
    n = k.shape[0]
    for i in range(n):
        p = (start + i) // PAGE_SIZE
        o = (start + i) % PAGE_SIZE
        pool[p, 0, :, o, :] = k[i]
        pool[p, 1, :, o, :] = v[i]


def _pack(k, v, start=0, n_pages=None):
    """Pack k/v into a fresh, page-aligned buffer starting at position `start`."""
    n = k.shape[0]
    n_pages = n_pages or max(1, math.ceil((start + n) / PAGE_SIZE))
    pool = torch.zeros(
        n_pages, 2, N_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.float16, device="cuda"
    )
    _write_range(pool, k, v, start)
    return pool


def _span(start, end):
    """(paged_kv_indices, paged_kv_last_page_len) covering logical [start, end)."""
    first = start // PAGE_SIZE
    last = (end - 1) // PAGE_SIZE
    idx = torch.arange(first, last + 1, dtype=torch.int32, device="cuda")
    last_len = torch.tensor([end - last * PAGE_SIZE], dtype=torch.int32, device="cuda")
    return idx, last_len


def _indptr(idx):
    return torch.tensor([0, idx.numel()], dtype=torch.int32, device="cuda")


def _prefill(q, pool, idx, last_len, causal):
    """One paged prefill call returning (out, lse)."""
    wrapper = kernels.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device="cuda"), "HND"
    )
    q_len = q.shape[0]
    wrapper.begin_forward(
        torch.tensor([0, q_len], dtype=torch.int32, device="cuda"),
        _indptr(idx),
        last_len,
        N_QO_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
    )
    out, lse = wrapper.forward(
        q, pool, idx, causal=causal, return_lse=True
    )
    wrapper.end_forward()
    return out, lse


def _torch_reference(q, k, v, chunk_start, sm_scale):
    """Dense fp32 causal reference: query i (abs pos chunk_start+i) sees keys <= that pos.

    Assumes ratio 1 (n_qo_heads == n_kv_heads), which is what these tests use.
    """
    qf, kf, vf = q.float(), k.float(), v.float()
    C, Hq, D = qf.shape
    T = kf.shape[0]
    out = torch.empty(C, Hq, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(C, Hq, dtype=torch.float32, device=q.device)
    for h in range(Hq):
        s = qf[:, h, :] @ kf[:, h, :].T * sm_scale  # [C, T]
        qpos = torch.arange(C, device=q.device) + chunk_start
        mask = torch.arange(T, device=q.device)[None, :] > qpos[:, None]
        s = s.masked_fill(mask, float("-inf"))
        m = s.max(dim=-1, keepdim=True).values
        e = torch.exp(s - m)
        denom = e.sum(dim=-1, keepdim=True)
        out[:, h, :] = (e @ vf[:, h, :]) / denom
        lse[:, h] = (m + torch.log(denom)).squeeze(-1)
    return out, lse


def _report(name, a, b):
    d = (a.float() - b.float()).abs()
    rel = d.max().item() / max(b.float().abs().max().item(), 1e-6)
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()
    print(
        f"[{name}] max|diff|={d.max().item():.5f}  max|ref|={b.float().abs().max().item():.3f}  "
        f"rel={rel:.2e}  cosine={cos:.6f}  mean|diff|={d.mean().item():.2e}"
    )
    return d.max().item(), cos


@requires_cuda
def test_b0_1_return_lse_is_plumbed():
    """The wrappers can hand back fp32 lse with the documented shape."""
    torch.manual_seed(0)
    pool = _pool()
    k = torch.randn(OLD_LEN + CHUNK_LEN, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    v = torch.randn(OLD_LEN + CHUNK_LEN, N_KV_HEADS, HEAD_DIM, device="cuda")
    _write_range(pool, k.half(), v.half(), 0)

    q = (torch.randn(CHUNK_LEN, N_QO_HEADS, HEAD_DIM, device="cuda") * 0.3).half()
    idx, last = _span(OLD_LEN, OLD_LEN + CHUNK_LEN)
    out, lse = _prefill(q, pool, idx, last, causal=True)

    assert out.shape == (CHUNK_LEN, N_QO_HEADS, HEAD_DIM)
    assert lse.shape == (CHUNK_LEN, N_QO_HEADS), f"unexpected lse shape {tuple(lse.shape)}"
    assert lse.dtype == torch.float32
    assert torch.isfinite(lse).all()
    print(f"\n[B0.1] out {tuple(out.shape)} lse {tuple(lse.shape)} dtype={lse.dtype}")


@requires_cuda
def test_b0_3_single_pass_paged_matches_dense_torch():
    """Sanity: our wrapper usage reproduces a plain dense causal attention."""
    torch.manual_seed(1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    k = torch.randn(OLD_LEN + CHUNK_LEN, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    v = torch.randn(OLD_LEN + CHUNK_LEN, N_KV_HEADS, HEAD_DIM, device="cuda")
    q = torch.randn(CHUNK_LEN, N_QO_HEADS, HEAD_DIM, device="cuda") * 0.3

    pool = _pool()
    _write_range(pool, k.half(), v.half(), 0)

    idx, last = _span(0, OLD_LEN + CHUNK_LEN)
    out, lse = _prefill(q.half(), pool, idx, last, causal=True)
    ref_out, ref_lse = _torch_reference(q, k, v, OLD_LEN, sm_scale)

    dmax, cos = _report("B0.3 dense vs torch", out, ref_out)
    assert cos > 0.9999, "paged causal prefill does not match the dense reference"
    assert dmax < 0.02

    lse_nat = kernels.lse_log2_to_natural(lse)
    base2_ratio = (lse / ref_lse).mean().item()
    lse_gap = (lse_nat - ref_lse).abs().max().item()
    print(
        f"[B0.3] lse: kernel/ref mean ratio = {base2_ratio:.6f} "
        f"(log2(e)={kernels.LOG2E:.6f}); max|diff| after base-2 -> natural = {lse_gap:.5f}"
    )
    # documents that the wrappers hand back lse in the base-2 (FlashInfer) domain
    assert abs(base2_ratio - kernels.LOG2E) < 1e-3
    assert lse_gap < 0.05


@requires_cuda
def test_b0_3_split_attention_lse_merge_matches_dense():
    """The B0.3 claim: merge(old non-causal, chunk causal) == dense causal(old+chunk)."""
    torch.manual_seed(2)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    total = OLD_LEN + CHUNK_LEN
    k = torch.randn(total, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    v = torch.randn(total, N_KV_HEADS, HEAD_DIM, device="cuda")
    q = torch.randn(CHUNK_LEN, N_QO_HEADS, HEAD_DIM, device="cuda") * 0.3

    pool = _pool()
    _write_range(pool, k.half(), v.half(), 0)

    # ground truth
    ref_out, ref_lse = _torch_reference(q, k, v, OLD_LEN, sm_scale)

    # single-pass dense causal
    idx_all, last_all = _span(0, total)
    dense_out, dense_lse = _prefill(q.half(), pool, idx_all, last_all, causal=True)

    # pass A: old context, non-causal, partial tail page handled by last_page_len
    idx_a, last_a = _span(0, OLD_LEN)
    out_a, lse_a = _prefill(q.half(), pool, idx_a, last_a, causal=False)
    assert lse_a.shape == (CHUNK_LEN, N_QO_HEADS)

    # pass B: the chunk's own K/V, packed from offset 0 so no already-counted
    # old tokens leak in through the first shared page
    pool_b = _pack(k[OLD_LEN:], v[OLD_LEN:])
    idx_b, last_b = _span(0, CHUNK_LEN)
    out_b, lse_b = _prefill(q.half(), pool_b, idx_b, last_b, causal=True)

    merged_out, merged_lse = kernels.merge_state(out_a, lse_a, out_b, lse_b)

    dmax, cos = _report("B0.3 merged vs torch", merged_out, ref_out)
    dmax_d, cos_d = _report("B0.3 dense  vs torch", dense_out, ref_out)
    lse_gap = (kernels.lse_log2_to_natural(merged_lse) - ref_lse).abs().max().item()
    print(f"[B0.3] merged lse max|diff| (base-2 -> natural) = {lse_gap:.5f}")

    assert cos > 0.9999 and cos_d > 0.9999
    assert dmax < 0.02
    assert lse_gap < 0.05


@requires_cuda
def test_b0_4_packing_required_when_chunk_is_not_page_aligned():
    """Naive reuse of the shared first page double-counts old tokens.

    This is precisely why the prefill wrapper needs ``page_valid_entries`` --
    which it does not have -- and why we pack instead.
    """
    torch.manual_seed(3)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    total = OLD_LEN + CHUNK_LEN  # OLD_LEN=37 -> chunk starts at page offset 5
    assert OLD_LEN % PAGE_SIZE != 0, "this test is about the non-aligned case"

    k = torch.randn(total, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    v = torch.randn(total, N_KV_HEADS, HEAD_DIM, device="cuda")
    q = torch.randn(CHUNK_LEN, N_QO_HEADS, HEAD_DIM, device="cuda") * 0.3

    pool = _pool()
    _write_range(pool, k.half(), v.half(), 0)
    ref_out, _ = _torch_reference(q, k, v, OLD_LEN, sm_scale)

    idx_a, last_a = _span(0, OLD_LEN)
    out_a, lse_a = _prefill(q.half(), pool, idx_a, last_a, causal=False)

    # (a) NAIVE: reuse the pool pages that hold the chunk, stale old slots and all
    idx_b_naive, last_b_naive = _span(OLD_LEN, total)
    out_bn, lse_bn = _prefill(q.half(), pool, idx_b_naive, last_b_naive, causal=True)
    naive_out, _ = kernels.merge_state(out_a, lse_a, out_bn, lse_bn)

    # (b) PACKED: copy the chunk into a fresh page-aligned buffer
    pool_b = _pack(k[OLD_LEN:], v[OLD_LEN:])
    idx_b, last_b = _span(0, CHUNK_LEN)
    out_b, lse_b = _prefill(q.half(), pool_b, idx_b, last_b, causal=True)
    packed_out, _ = kernels.merge_state(out_a, lse_a, out_b, lse_b)

    _, cos_naive = _report("B0.4 naive  vs torch", naive_out, ref_out)
    dmax_packed, cos_packed = _report("B0.4 packed vs torch", packed_out, ref_out)

    print(
        f"[B0.4] chunk starts at page offset {OLD_LEN % PAGE_SIZE}; "
        f"naive cosine={cos_naive:.5f}  packed cosine={cos_packed:.6f}"
    )
    assert cos_packed > 0.9999, "packed split attention must match dense"
    assert dmax_packed < 0.02
    assert cos_naive < 0.9999, (
        "expected the naive shared-page variant to break; if it now passes, the "
        "packing requirement needs revisiting"
    )


@requires_cuda
def test_b0_4_scattered_partial_pages_pack_cleanly():
    """DCI-style retained pages: full and partially-filled, scattered.

    Non-causal attention over the old context is permutation-invariant over keys,
    so packing a scattered/partial retained set into a dense buffer is sound and
    needs no new kernel argument.
    """
    torch.manual_seed(4)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    chunk_len = 16

    # retained keys live in two full pages and one partially filled page
    retained_offsets = list(range(0, PAGE_SIZE * 2)) + list(range(0, 5))
    n_ret = len(retained_offsets)
    k_ret = torch.randn(n_ret, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    v_ret = torch.randn(n_ret, N_KV_HEADS, HEAD_DIM, device="cuda")

    k_chunk = torch.randn(chunk_len, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    v_chunk = torch.randn(chunk_len, N_KV_HEADS, HEAD_DIM, device="cuda")
    q = torch.randn(chunk_len, N_QO_HEADS, HEAD_DIM, device="cuda") * 0.3

    # ground truth: chunk queries attend causally to the packed retained set+chunk
    k_all = torch.cat([k_ret, k_chunk], 0)
    v_all = torch.cat([v_ret, v_chunk], 0)
    ref_out, _ = _torch_reference(q, k_all, v_all, n_ret, sm_scale)

    pool_ret = _pack(k_ret, v_ret)
    idx_r, last_r = _span(0, n_ret)
    out_a, lse_a = _prefill(q.half(), pool_ret, idx_r, last_r, causal=False)

    pool_chunk = _pack(k_chunk, v_chunk)
    idx_c, last_c = _span(0, chunk_len)
    out_b, lse_b = _prefill(q.half(), pool_chunk, idx_c, last_c, causal=True)

    merged_out, _ = kernels.merge_state(out_a, lse_a, out_b, lse_b)
    dmax, cos = _report("B0.4 scattered+packed vs torch", merged_out, ref_out)
    print(
        f"[B0.4] retained={n_ret} tokens across 2 full pages + 1 page with 5 valid entries; "
        f"chunk={chunk_len}"
    )
    assert cos > 0.9999, "packed scattered-attention path must match dense"
    assert dmax < 0.02
