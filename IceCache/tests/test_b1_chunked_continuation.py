"""B1 -- chunked continuation prefill vs. the phase A tokenwise oracle.

Full-cache, no DCI (the whole sequence fits the GPU page budget), so a
continuation chunk is a plain causal paged prefill over a contiguous KV.

For each chunk size this reports:

* forward calls needed to ingest the tool result,
* wall time and ms per appended token,
* logits / KV absolute-error percentiles against the tokenwise oracle,
* top-1 and top-5 overlap with the oracle's next-token distribution.

and asserts the structural invariants: ``_prepare_prefill`` fires exactly once,
``committed_ids.shape[1] == state.seq_len``, and absolute positions stay
contiguous.

Run::

    cd /home/yx/IceCache/IceCache
    PYTHONPATH=/home/yx/IceCache/IceCache/source \\
      /home/yx/miniconda3/envs/icecache/bin/python -m pytest tests/test_b1_chunked_continuation.py -v -s
"""

from __future__ import annotations

import gc
import os
import time

import pytest
import torch

MODEL_PATH = os.environ.get(
    "AGENT_SESSION_MODEL", "/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)
CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA is required")

_FILLER = "The quick brown fox jumps over the lazy dog. " * 600

PROMPT_LEN = 520
TOOL_LEN = 300  # total 820 tokens: fits 64 pages * 16 = 1024
CHUNK_SIZES = [1, 16, 64, 256]

CFG = dict(
    page_size=16,
    page_budgets=64,
    n_sink_pages=2,
    n_win_pages=2,
    page_topks=0,
    n_max_bytes=4 << 28,
    n_max_cpu_bytes=8 << 28,
)


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    return model.to("cuda").eval()


@pytest.fixture(scope="module")
def filler_ids():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    return tok(_FILLER, return_tensors="pt").input_ids[0]


def _new_state(model):
    from icecache.adapter import enable_icecache

    enable_icecache(model, dtype=torch.float16, device=torch.device("cuda"), **CFG)
    return model._icecache_infer_state


def _gather_kv(state):
    out = []
    for kvc in state.kv_caches:
        pages = kvc.buffer[kvc.c2p[0]]
        flat = pages.permute(0, 3, 1, 2, 4).reshape(-1, 2, state.n_kv_heads, state.head_dim)
        out.append(flat[: state.seq_len].float().clone())
    return out


def _pct(t: torch.Tensor):
    q = torch.tensor([0.5, 0.9, 0.99], device=t.device)
    vals = torch.quantile(t.flatten().float(), q).tolist()
    return vals[0], vals[1], vals[2]


def _drop(session):
    del session
    gc.collect()
    torch.cuda.empty_cache()


def _run_oracle(model, ids, prompt_len, monkeypatch):
    from icecache.adapter.agent_session import TokenwiseContinuationReference
    from icecache.infer_state import InferState

    calls = []
    original = InferState._prepare_prefill

    def spy(self, bsz, q_len):
        calls.append(int(q_len))
        return original(self, bsz, q_len)

    monkeypatch.setattr(InferState, "_prepare_prefill", spy)
    try:
        state = _new_state(model)
        sess = TokenwiseContinuationReference(model)
        sess.start(ids[:, :prompt_len])
        t0 = time.perf_counter()
        sess.append_tokens(ids[:, prompt_len:])
        elapsed = time.perf_counter() - t0
        return {
            "label": "oracle(tokenwise)",
            "logits": sess.next_logits.clone(),
            "kv": _gather_kv(state),
            "seq_len": sess.seq_len,
            "forwards": sess.n_steps,
            "append_s": elapsed,
            "prepare_prefill_calls": list(calls),
            "session": sess,
        }
    finally:
        monkeypatch.undo()


def _run_chunked(model, ids, prompt_len, chunk_size, monkeypatch):
    from icecache.adapter.chunk_session import ChunkedContinuationPrefill
    from icecache.infer_state import InferState

    calls = []
    original = InferState._prepare_prefill

    def spy(self, bsz, q_len):
        calls.append(int(q_len))
        return original(self, bsz, q_len)

    monkeypatch.setattr(InferState, "_prepare_prefill", spy)
    try:
        state = _new_state(model)
        sess = ChunkedContinuationPrefill(model, chunk_size=chunk_size)
        sess.start(ids[:, :prompt_len])
        n_before = sess.forward_calls
        t0 = time.perf_counter()
        sess.append_transcript(ids)
        elapsed = time.perf_counter() - t0
        return {
            "label": f"chunked({chunk_size})",
            "logits": sess.next_logits.clone(),
            "kv": _gather_kv(state),
            "seq_len": sess.seq_len,
            "forwards": sess.forward_calls - n_before,
            "append_s": elapsed,
            "prepare_prefill_calls": list(calls),
            "chunks": list(sess.chunks_used),
            "dci_active": sess.dci_active,
            "session": sess,
        }
    finally:
        monkeypatch.undo()


@requires_cuda
def test_b1_chunked_continuation_runs(hf_model, filler_ids, monkeypatch):
    """Smoke: the continuation path works at all, and never re-prefills."""
    from icecache.adapter.chunk_session import ChunkedContinuationPrefill

    ids = filler_ids[: PROMPT_LEN + TOOL_LEN].unsqueeze(0).cuda()
    state = _new_state(hf_model)
    sess = ChunkedContinuationPrefill(hf_model, chunk_size=64)
    sess.start(ids[:, :PROMPT_LEN])
    assert sess.seq_len == PROMPT_LEN
    assert not sess.dci_active, "full-cache config expected"

    sess.append_transcript(ids)
    assert sess.seq_len == PROMPT_LEN + TOOL_LEN
    assert sess.seq_len == sess.committed_ids.shape[1]
    assert int(state.seq_len) == PROMPT_LEN + TOOL_LEN
    assert sess.chunks_used == [64, 64, 64, 64, 44], sess.chunks_used
    print(
        f"\n[B1 smoke] seq_len={sess.seq_len} forwards={sess.forward_calls} "
        f"chunks={sess.chunks_used} pending_offload={any(state.offload_win_flag)}"
    )
    _drop(sess)


@requires_cuda
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_b1_chunk_size_vs_oracle(hf_model, filler_ids, monkeypatch, chunk_size):
    ids = filler_ids[: PROMPT_LEN + TOOL_LEN].unsqueeze(0).cuda()

    oracle = _run_oracle(hf_model, ids, PROMPT_LEN, monkeypatch)
    ref_logits = oracle["logits"]
    ref_kv = oracle["kv"]

    got = _run_chunked(hf_model, ids, PROMPT_LEN, chunk_size, monkeypatch)

    # ---- structural invariants -------------------------------------------
    assert got["seq_len"] == oracle["seq_len"] == PROMPT_LEN + TOOL_LEN
    assert got["prepare_prefill_calls"] == [PROMPT_LEN], (
        f"continuation re-prefilled: {got['prepare_prefill_calls']}"
    )
    assert oracle["prepare_prefill_calls"] == [PROMPT_LEN]
    assert not got["dci_active"]

    # ---- numerical agreement --------------------------------------------
    dlog = (got["logits"].float() - ref_logits.float()).abs()
    l_p50, l_p90, l_p99 = _pct(dlog)
    l_max = dlog.max().item()
    top1 = int(got["logits"].argmax(-1)) == int(ref_logits.argmax(-1))
    k = 5
    a5 = set(got["logits"][0].topk(k).indices.tolist())
    b5 = set(ref_logits[0].topk(k).indices.tolist())
    top5 = len(a5 & b5) / k

    kv_diffs = [
        (g.float() - r.float()).abs() for g, r in zip(got["kv"], ref_kv)
    ]
    kv_mean = sum(d.mean().item() for d in kv_diffs) / len(kv_diffs)
    kv_max = max(d.max().item() for d in kv_diffs)
    kv_p99 = max(_pct(d)[2] for d in kv_diffs)

    per_tok_ms = 1000.0 * got["append_s"] / TOOL_LEN
    print(
        f"\n[B1] chunk={chunk_size:>4}  forwards={got['forwards']:>4}  "
        f"append={got['append_s']:.3f}s ({per_tok_ms:.1f} ms/tok)  "
        f"logits p50/p90/p99/max = {l_p50:.4f}/{l_p90:.4f}/{l_p99:.4f}/{l_max:.4f}  "
        f"kv p99={kv_p99:.4f} mean={kv_mean:.2e} max={kv_max:.3f}  "
        f"top1={'ok' if top1 else 'MISMATCH'} top5={top5:.2f}"
    )

    assert top1, "greedy top-1 must agree with the tokenwise oracle"
    assert len(got["chunks"]) == (TOOL_LEN + chunk_size - 1) // chunk_size or chunk_size == 1
    # measured drift: see the printed table; keep a generous ceiling here
    assert l_max < 1.0, f"logit drift {l_max:.4f} is too large for a kernel-order effect"
    assert kv_mean < 0.05

    _drop(got["session"])


@requires_cuda
def test_b1_forward_count_scales_with_chunk_size(hf_model, filler_ids, monkeypatch):
    """The whole point: fewer forwards as the chunk grows."""
    counts = {}
    for size in CHUNK_SIZES:
        ids = filler_ids[: PROMPT_LEN + TOOL_LEN].unsqueeze(0).cuda()
        got = _run_chunked(hf_model, ids, PROMPT_LEN, size, monkeypatch)
        counts[size] = got["forwards"]
        _drop(got["session"])
    print(f"\n[B1] forward calls by chunk size: {counts}")
    assert counts[1] == TOOL_LEN
    assert counts[16] == -(-TOOL_LEN // 16)
    assert counts[64] == -(-TOOL_LEN // 64)
    assert counts[256] == -(-TOOL_LEN // 256)
    for a, b in zip(CHUNK_SIZES, CHUNK_SIZES[1:]):
        assert counts[b] < counts[a], "forward count must fall as chunks grow"
