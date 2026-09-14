"""Unit test for ``InferState._pack_resident`` / ``_resident_valid_counts``.

This decouples the *gather* from the attention, so we can prove the packed
operand is the right token set and order before blaming the kernel.

A synthetic ``KvCache``-shaped pool is filled with per-(page, head) sentinel
values, ``page_valid_entries`` and ``c2p`` are fixed by hand, and the packed
output is compared against the exact expected token stream.

Run without a GPU (uses CPU tensors)::

    PYTHONPATH=/home/yx/IceCache/IceCache/source \\
      /home/yx/miniconda3/envs/icecache/bin/python -m pytest tests/test_c1b_pack_resident.py -v -s
"""

from __future__ import annotations

import types

import pytest
import torch

from icecache.infer_state import InferState

PS = 16
D = 8
H = 4  # n_kv_heads


def make_state(n_pages, c2p, pve, kv_last_page_len, page_valid_entries_table):
    """A bare InferState instance (no __init__) with just the pieces _pack_resident needs."""
    st = object.__new__(InferState)  # skip __init__; only methods are used
    st.n_kv_heads = H
    st.page_size = PS
    st.head_dim = D
    st.dtype = torch.float32
    st.device = torch.device("cpu")
    st.n_layers = 1
    st.kv_last_page_len = kv_last_page_len
    st._i32 = dict(dtype=torch.int32, device=st.device)

    n_phys = int(c2p.max().item()) + 1  # enough physical pages for the largest c2p id
    kvc = types.SimpleNamespace()
    kvc.n_real_pages = n_pages
    kvc.c2p = c2p[None, :]  # [1, n]
    kvc.buffer = torch.zeros(
        n_phys, 2, H, PS, D, dtype=torch.float32, device=st.device
    )
    # fill with sentinel: buffer[page, k_or_v, head, off, dim] = 1000*page + 100*head + off
    for p in range(n_phys):
        for h in range(H):
            for o in range(PS):
                kvc.buffer[p, 0, h, o, :] = 1000 * p + 100 * h + o
                kvc.buffer[p, 1, h, o, :] = 2000 * p + 100 * h + o

    st.kv_caches = [kvc]
    st.page_valid_entries = [page_valid_entries_table]
    return st


def call_pack(state, chunk_len):
    return state._pack_resident(0, chunk_len)


def expected_token_ids(counts, c2p):
    """The exact ordered token stream the pack must produce.

    For head h: for slot s in order 0..n-1, tokens off in 0..counts[s,h)-1,
    each token id = 1000*c2p[s] + 100*h + off (for K). V uses 2000 offset.
    """
    ks, vs = [], []
    for h in range(H):
        for s in range(len(c2p)):
            c = int(counts[s, h].item())
            for o in range(c):
                ks.append(1000 * c2p[s] + 100 * h + o)
                vs.append(2000 * c2p[s] + 100 * h + o)
    return ks, vs


def test_pack_resident_all_full_pages():
    """n=2 full pages, c2p = [5, 7], all heads valid 16."""
    c2p = torch.tensor([5, 7])
    pve = torch.full((2, H), PS, dtype=torch.int32)
    st = make_state(2, c2p, None, PS, pve)
    stage, indices, indptr, last_len = call_pack(st, chunk_len=4)

    # per head: 2*16 = 32 resident + 4 chunk = 36 -> 3 pages each, 12 total
    assert stage.shape == (12, 2, 1, PS, D), stage.shape
    assert indices.numel() == 12
    assert last_len.tolist() == [4, 4, 4, 4]  # 36 % 16 = 4
    assert indptr.tolist() == [0, 3, 6, 9, 12]

    ks, vs = expected_token_ids(pve, c2p)
    # stage [12, 2, 1, 16, D] -> slot-major [192, 2, D]; head h owns
    # slots [indptr[h]*16, indptr[h+1]*16), i.e. 48 slots, first 32 resident.
    flat = stage.permute(0, 3, 1, 2, 4).reshape(-1, 2, D)
    for h in range(H):
        start = int(indptr[h]) * PS
        run = flat[start : start + 32]
        got_k = run[:, 0, 0].tolist()
        got_v = run[:, 1, 0].tolist()
        exp_k = ks[h * 32 : h * 32 + 32]
        exp_v = vs[h * 32 : h * 32 + 32]
        assert got_k == exp_k, f"head {h} K mismatch"
        assert got_v == exp_v, f"head {h} V mismatch"


def test_pack_resident_divergent_counts_and_tail_clamp():
    """Divergent per-head counts + a clamped tail page (the real sparse case)."""
    c2p = torch.tensor([3, 9, 2])
    counts = torch.tensor(
        [[16, 16, 11, 6], [16, 16, 9, 5], [9, 7, 3, 1]], dtype=torch.int32
    )  # [slot, head]
    kv_last_page_len = 9  # clamps the tail slot's valid counts to <= 9
    st = make_state(3, c2p, None, kv_last_page_len, counts.clone())

    stage, indices, indptr, last_len = call_pack(st, chunk_len=0)

    # expected counts after tail clamp
    eff = counts.clone()
    eff[2] = torch.minimum(eff[2], torch.tensor(kv_last_page_len))
    ks, vs = expected_token_ids(eff, c2p)

    per_head = eff.sum(0).tolist()
    flat = stage.permute(0, 3, 1, 2, 4).reshape(-1, 2, D)
    for h in range(H):
        c = per_head[h]
        start = int(indptr[h]) * PS
        got_k = flat[start : start + c, 0, 0].tolist()
        got_v = flat[start : start + c, 1, 0].tolist()
        exp_k = ks[sum(per_head[:h]) : sum(per_head[:h]) + c]
        exp_v = vs[sum(per_head[:h]) : sum(per_head[:h]) + c]
        assert got_k == exp_k, f"head {h} K mismatch"
        assert got_v == exp_v, f"head {h} V mismatch"

    # per-head page counts and last_len must reflect the divergent lengths
    print(f"\n[C1b] per-head counts={per_head} last_len={last_len.tolist()} indptr={indptr.tolist()}")
    for h in range(H):
        assert last_len[h] == (per_head[h] - 1) % PS + 1
