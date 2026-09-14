"""B0b -- per-KV-head valid lengths. Supersedes B0.4's packing argument.

B0.4 packed a retained set into a dense buffer and argued that packing is sound
because non-causal attention is permutation-invariant over keys.  That argument
silently assumed **every KV head has the same number of valid entries** — it
packed one shared set of offsets for all heads.

Real IceCache does not work that way.  ``page_valid_entries`` is indexed
``[page, kv_head]`` (see ``infer_state.py`` ``decode_sdpa``/``estimate_select_recall``)
and, measured on a sparse run (budget 16, prompt 900, 20 decode steps):

* layer 0 per-page head-min/max: ``[16,16,9,9,9,9,9,10,8,5,9,2,8,8,16,16]`` vs
  ``[16,...,16]`` -> 12/16 pages have non-uniform per-head valid counts;
* across all 32 layers 384/512 (page, head) rows are non-uniform;
* and ``selected_page_idx`` shows 12/12 slots where different KV heads point at
  **entirely different pages**.

So each KV head owns its own (page set, length).  A single ``last_page_len``
cannot express that, and neither can a shared slot space.

The fix validated here: put the **KV head in the batch dimension** of one paged
prefill call.  FlashInfer already gives every request its own
``paged_kv_indptr`` and ``paged_kv_last_page_len``, which is exactly the
per-head structure we need.  No kernel change, no per-head CUDA launches.

This file shows:
1. a single shared length is impossible when heads diverge;
2. per-head batching reproduces a dense per-head reference.
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
N_KV_HEADS = 4
RATIO = 1  # group_size 1 is in the dispatch table
N_QO_HEADS = N_KV_HEADS * RATIO
CHUNK_LEN = 20
RETAINED_LENS = [37, 32, 23, 11]  # deliberately divergent, as measured
WORKSPACE_MB = 64


def _stage(per_head_kv):
    """Pack each head's tokens into its own contiguous page run.

    Layout is HND with num_kv_heads == 1, one *request* per KV head:
      request h -> pages [off_h, off_h + ceil(L_h / page_size))
    Returns (buffer, indices, indptr, last_page_len).
    """
    bufs, indptr, last_len = [], [0], []
    for k, v in per_head_kv:
        L = k.shape[0]
        n_pages = math.ceil(L / PAGE_SIZE)
        buf = torch.zeros(
            n_pages, 2, 1, PAGE_SIZE, HEAD_DIM, dtype=torch.float16, device="cuda"
        )
        for i in range(L):
            buf[i // PAGE_SIZE, 0, 0, i % PAGE_SIZE, :] = k[i].half()
            buf[i // PAGE_SIZE, 1, 0, i % PAGE_SIZE, :] = v[i].half()
        bufs.append(buf)
        indptr.append(indptr[-1] + n_pages)
        last_len.append(L - (n_pages - 1) * PAGE_SIZE)
    buffer = torch.cat(bufs, 0)
    indices = torch.arange(buffer.shape[0], dtype=torch.int32, device="cuda")
    return (
        buffer,
        indices,
        torch.tensor(indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(last_len, dtype=torch.int32, device="cuda"),
    )


def _batched_prefill(q, staged, causal):
    """q: [C, n_kv_heads, D]. KV head goes in the batch dimension."""
    buffer, indices, indptr, last_len = staged
    C = int(q.shape[0])
    n_req = int(q.shape[1])
    assert n_req == N_KV_HEADS, f"expected {N_KV_HEADS} heads, got {n_req}"
    # [C, n_heads, D] -> [n_heads, C, D] -> [n_heads*C, RATIO, D]
    q = (
        q.permute(1, 0, 2)
        .reshape(n_req * C, RATIO, HEAD_DIM)
        .contiguous()
        .to(torch.float16)
    )

    wrapper = kernels.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device="cuda"), "HND"
    )
    wrapper.begin_forward(
        torch.arange(0, n_req * C + 1, C, dtype=torch.int32, device="cuda"),
        indptr,
        last_len,
        RATIO,
        1,
        HEAD_DIM,
    )
    out, lse = wrapper.forward(q, buffer, indices, causal=causal, return_lse=True)
    wrapper.end_forward()
    # [n_req*C, RATIO, D] -> [C, n_req*RATIO, D]
    out = out.reshape(n_req, C, RATIO, HEAD_DIM).permute(1, 0, 2, 3)
    lse = lse.reshape(n_req, C, RATIO).permute(1, 0, 2)
    return out.reshape(C, N_QO_HEADS, HEAD_DIM), lse.reshape(C, N_QO_HEADS)


def _dense_reference(q, per_head_kv, chunk_k, chunk_v):
    """Per-head dense causal reference: chunk query i sees retained_h then chunk[0..i]."""
    sm = 1.0 / math.sqrt(HEAD_DIM)
    C = q.shape[0]  # q is [C, n_kv_heads, D]
    out = torch.empty(C, N_QO_HEADS, HEAD_DIM, device="cuda")
    lse = torch.empty(C, N_QO_HEADS, device="cuda")
    for h in range(N_KV_HEADS):
        k_ret, v_ret = per_head_kv[h]
        k_all = torch.cat([k_ret, chunk_k[:, h, :]], 0).float()
        v_all = torch.cat([v_ret, chunk_v[:, h, :]], 0).float()
        n_keys = int(k_all.shape[0])
        Lr = n_keys - C
        s = q[:, h, :].float() @ k_all.T * sm  # [C, n_keys]
        qpos = Lr + torch.arange(C, device="cuda")
        keypos = torch.arange(n_keys, device="cuda")
        s = s.masked_fill(keypos[None, :] > qpos[:, None], float("-inf"))
        m = s.max(-1, keepdim=True).values
        e = torch.exp(s - m)
        den = e.sum(-1, keepdim=True)
        out[:, h, :] = (e @ v_all) / den
        lse[:, h] = (m + torch.log(den)).squeeze(-1)
    return out, lse


def _report(name, a, b):
    d = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()
    print(
        f"[{name}] max|diff|={d.max().item():.5f}  cosine={cos:.6f}  mean|diff|={d.mean().item():.2e}"
    )
    return d.max().item(), cos


def _build_case(seed=7):
    torch.manual_seed(seed)
    per_head_kv = []
    for h, L in enumerate(RETAINED_LENS):
        k = torch.randn(L, HEAD_DIM, device="cuda") * 0.3
        v = torch.randn(L, HEAD_DIM, device="cuda")
        per_head_kv.append((k, v))
    # queries: [C, n_heads, D]
    q = torch.randn(CHUNK_LEN, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    chunk_k = torch.randn(CHUNK_LEN, N_KV_HEADS, HEAD_DIM, device="cuda") * 0.3
    chunk_v = torch.randn(CHUNK_LEN, N_KV_HEADS, HEAD_DIM, device="cuda")
    return per_head_kv, q, chunk_k, chunk_v


@requires_cuda
def test_b0b_single_shared_length_cannot_express_divergent_heads():
    """A single last_page_len is impossible when heads have different lengths."""
    per_head_kv, q, chunk_k, chunk_v = _build_case()
    ref_out, _ = _dense_reference(q, per_head_kv, chunk_k, chunk_v)

    # Force one shared length for every head: the longest, zero-padded.  Heads with
    # fewer retained tokens now expose padding slots as real keys.
    sm = 1.0 / math.sqrt(HEAD_DIM)
    C = CHUNK_LEN
    shared = max(RETAINED_LENS)
    bad_out = torch.empty(C, N_QO_HEADS, HEAD_DIM, device="cuda")
    for h in range(N_KV_HEADS):
        k_ret, v_ret = per_head_kv[h]
        Lh = k_ret.shape[0]
        k_sp = torch.zeros(shared, HEAD_DIM, device="cuda")
        v_sp = torch.zeros(shared, HEAD_DIM, device="cuda")
        k_sp[:Lh] = k_ret
        v_sp[:Lh] = v_ret
        k_all = torch.cat([k_sp, chunk_k[:, h, :]], 0).float()
        v_all = torch.cat([v_sp, chunk_v[:, h, :]], 0).float()
        s = q[:, h, :].float() @ k_all.T * sm
        qpos = shared + torch.arange(C, device="cuda")
        s = s.masked_fill(
            torch.arange(shared + C, device="cuda")[None, :] > qpos[:, None],
            float("-inf"),
        )
        m = s.max(-1, keepdim=True).values
        e = torch.exp(s - m)
        bad_out[:, h, :] = (e @ v_all) / e.sum(-1, keepdim=True)

    _, cos = _report("B0b shared-length(impossible)", bad_out, ref_out)
    print(
        f"[B0b] retained lens per head = {RETAINED_LENS}; any single shared length "
        f"either pads (keys that do not exist) or truncates (keys that do)"
    )
    assert cos < 0.999, "expected a shared length to be wrong for divergent heads"


@requires_cuda
def test_b0b_per_head_batching_matches_dense():
    """KV head in the batch dimension reproduces the dense per-head reference."""
    per_head_kv, q, chunk_k, chunk_v = _build_case()
    ref_out, ref_lse = _dense_reference(q, per_head_kv, chunk_k, chunk_v)

    # ---- pass A: retained old tokens, non-causal -------------------------
    retain_kv = [(k.unsqueeze(0), v.unsqueeze(0)) for k, v in per_head_kv]
    a_out, a_lse = _batched_prefill(q, _stage([(k[0], v[0]) for k, v in retain_kv]), causal=False)

    # ---- pass B: the chunk, causal --------------------------------------
    chunk_pairs = [(chunk_k[:, h, :], chunk_v[:, h, :]) for h in range(N_KV_HEADS)]
    b_out, b_lse = _batched_prefill(q, _stage(chunk_pairs), causal=True)

    merged_out, merged_lse = kernels.merge_state(a_out, a_lse, b_out, b_lse)

    dmax, cos = _report("B0b per-head batched merged", merged_out, ref_out)
    lse_gap = (kernels.lse_log2_to_natural(merged_lse) - ref_lse).abs().max().item()
    print(
        f"[B0b] per-head lengths used as batch last_page_len = {RETAINED_LENS}; "
        f"lse max|diff| (base-2 -> natural) = {lse_gap:.5f}"
    )
    assert cos > 0.9999, "per-head batched packing must match the dense reference"
    assert dmax < 0.02
    assert lse_gap < 0.05


@requires_cuda
def test_b0b_per_head_batching_handles_a_head_that_differs_by_one_page():
    """The degenerate head (11 tokens -> one partial page) is still exact."""
    per_head_kv, q, chunk_k, chunk_v = _build_case(seed=11)
    ref_out, _ = _dense_reference(q, per_head_kv, chunk_k, chunk_v)

    a_out, a_lse = _batched_prefill(
        q, _stage([(k, v) for k, v in per_head_kv]), causal=False
    )
    b_out, b_lse = _batched_prefill(
        q, _stage([(chunk_k[:, h, :], chunk_v[:, h, :]) for h in range(N_KV_HEADS)]),
        causal=True,
    )
    merged_out, _ = kernels.merge_state(a_out, a_lse, b_out, b_lse)
    _, cos = _report("B0b divergent-page-count heads", merged_out, ref_out)
    counts = [math.ceil(L / PAGE_SIZE) for L in RETAINED_LENS]
    print(f"[B0b] pages per head = {counts} (deliberately differs across heads)")
    assert cos > 0.9999
