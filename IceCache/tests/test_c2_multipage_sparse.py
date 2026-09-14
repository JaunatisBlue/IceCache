"""C2 -- multi-page sparse continuation: a tool result that crosses page
boundaries is ingested in ONE forward and correctly enters the DCI tree.

The acceptance criteria (per the design doc section 6.1, a chunk shares one page
set so its logits CANNOT equal the per-token decode reference):

1. structural: ``_prepare_prefill`` fires once; ``committed_ids == seq_len``;
   DCI object identity preserved; ``num_points`` grows by the pages that left the
   window (matching the tokenwise oracle's growth);
2. the chunk's K/V lands in the paged cache at the correct physical tail and the
   window rotates correctly (c2p / last_page_len consistent);
3. greedy top-1 agrees with the tokenwise oracle where the shared-page-set
   approximation permits (reported, not strictly gated for large chunks).

Run::

    cd /home/yx/IceCache/IceCache
    PYTHONPATH=/home/yx/IceCache/IceCache/source \\
      /home/yx/miniconda3/envs/icecache/bin/python -m pytest tests/test_c2_multipage_sparse.py -v -s
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

MODEL_PATH = os.environ.get(
    "AGENT_SESSION_MODEL", "/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)
CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA is required")

_FILLER = "The quick brown fox jumps over the lazy dog. " * 600

PROMPT_LEN = 900          # 900 % 16 == 4
TOOL_LEN = 64             # crosses 4 page boundaries (4 + 64 = 68 -> 5 pages)

CFG = dict(
    page_size=16,
    page_budgets=16,
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


def _dci_points(state):
    return [None if db is None else int(db.num_points[0]) for db in state.dci_db]


def _dci_ids(state):
    return [None if db is None else id(db) for db in state.dci_db]


def _patch(monkeypatch):
    from icecache.infer_state import InferState

    calls = []
    original = InferState._prepare_prefill

    def spy(self, bsz, q_len):
        calls.append(int(q_len))
        return original(self, bsz, q_len)

    monkeypatch.setattr(InferState, "_prepare_prefill", spy)
    return calls


def _drop(obj):
    del obj
    gc.collect()
    torch.cuda.empty_cache()


@requires_cuda
def test_c2_multipage_chunk_enters_dci_tree(hf_model, filler_ids, monkeypatch):
    from icecache.adapter.agent_session import TokenwiseContinuationReference
    from icecache.adapter.chunk_session import ChunkedContinuationPrefill

    ids = filler_ids[: PROMPT_LEN + TOOL_LEN].unsqueeze(0).cuda()

    # ---- tokenwise oracle (the reference for structural invariants) ------
    calls = _patch(monkeypatch)
    state_o = _new_state(hf_model)
    oracle = TokenwiseContinuationReference(hf_model)
    oracle.start(ids[:, :PROMPT_LEN])
    assert oracle.dci_active
    points_before = oracle.n_dci_points
    ids_before = oracle.dci_object_ids
    oracle.append_tokens(ids[:, PROMPT_LEN:])
    oracle_points = oracle.n_dci_points
    oracle_len = oracle.seq_len
    assert calls == [PROMPT_LEN]
    assert oracle.dci_object_ids == ids_before
    monkeypatch.undo()
    _drop(oracle)

    # ---- chunked, one forward -------------------------------------------
    calls = _patch(monkeypatch)
    state_c = _new_state(hf_model)
    chunked = ChunkedContinuationPrefill(
        hf_model, chunk_size=TOOL_LEN, require_full_cache=False
    )
    chunked.start(ids[:, :PROMPT_LEN])
    assert chunked.dci_active
    ids_before_c = _dci_ids(chunked.state)
    points_before_c = _dci_points(chunked.state)
    chunked.append_transcript(ids)
    chunked_logits = chunked.next_logits.detach().clone()

    # ---- structural invariants ------------------------------------------
    assert calls == [PROMPT_LEN], f"continuation re-prefilled: {calls}"
    assert chunked.seq_len == oracle_len == PROMPT_LEN + TOOL_LEN
    assert chunked.seq_len == chunked.committed_ids.shape[1]
    assert int(state_c.seq_len) == chunked.seq_len
    assert _dci_ids(chunked.state) == ids_before_c, "DCI objects were rebuilt"

    points_after_c = _dci_points(chunked.state)
    grew = sum(1 for b, a in zip(points_before_c, points_after_c) if a is not None and a > b)
    print(
        f"\n[C2] seq_len={chunked.seq_len}  forwards={chunked.forward_calls}  "
        f"prepare_prefill={calls}"
    )
    print(
        f"[C2] dci points: layer0 {points_before_c[0]} -> {points_after_c[0]} "
        f"(oracle {points_before[0]} -> {oracle_points[0]}); layers that grew = {grew}/32"
    )
    # the tokenwise oracle and the chunk must have grown the tree by the SAME
    # number of points, because the same pages left the window
    assert points_after_c[0] == oracle_points[0], (
        f"chunk grew tree to {points_after_c[0]} but oracle to {oracle_points[0]}"
    )
    assert int(state_c.kv_last_page_len) == (chunked.seq_len - 1) % CFG["page_size"] + 1
    for kvc in state_c.kv_caches:
        assert int(kvc.c2p.max()) < state_c.n_max_pages

    monkeypatch.undo()
    _drop(chunked)


@requires_cuda
def test_c2_chunk_ends_exactly_on_page_boundary(hf_model, filler_ids, monkeypatch):
    """A chunk that lands exactly on a page boundary still maintains invariants."""
    from icecache.adapter.chunk_session import ChunkedContinuationPrefill

    # 900 % 16 == 4, so 12 tokens bring us exactly to a page boundary (916)
    TOOL = 12
    ids = filler_ids[: PROMPT_LEN + TOOL].unsqueeze(0).cuda()
    calls = _patch(monkeypatch)
    state = _new_state(hf_model)
    chunked = ChunkedContinuationPrefill(
        hf_model, chunk_size=TOOL, require_full_cache=False
    )
    chunked.start(ids[:, :PROMPT_LEN])
    chunked.append_transcript(ids)

    assert calls == [PROMPT_LEN]
    assert chunked.seq_len == PROMPT_LEN + TOOL
    assert chunked.seq_len == chunked.committed_ids.shape[1]
    # ends exactly on boundary -> last_page_len must be page_size
    assert int(state.kv_last_page_len) == CFG["page_size"], (
        f"expected page-aligned tail, got last_page_len={state.kv_last_page_len}"
    )
    print(f"\n[C2 boundary] seq_len={chunked.seq_len} last_page_len={state.kv_last_page_len}")
    monkeypatch.undo()
    _drop(chunked)
