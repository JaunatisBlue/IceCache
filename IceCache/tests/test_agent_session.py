"""Tests for the Phase A tokenwise continuation reference.

Two groups:

* **unit** tests use a tiny fake model and need neither CUDA nor the 8B
  checkpoint.  They cover the session state machine, absolute positions, the
  commit-before-stop ordering, and prefix validation.
* **gpu** tests drive the real Llama-3.1-8B-Instruct through IceCache.  They are
  skipped automatically when CUDA is unavailable.

Run::

    cd /home/yx/IceCache/IceCache
    PYTHONPATH=/home/yx/IceCache/IceCache/source \\
      /home/yx/miniconda3/envs/icecache/bin/python -m pytest tests/test_agent_session.py -v -s
"""

from __future__ import annotations

import gc
import os
import types

import pytest
import torch

from icecache.adapter.agent_session import (
    IceCacheAgentSession,
    PrefixMismatchError,
    SessionPhase,
    SessionStateError,
    TokenwiseContinuationReference,
)

MODEL_PATH = os.environ.get(
    "AGENT_SESSION_MODEL", "/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)
HAS_CUDA = torch.cuda.is_available()
gpu_only = pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required")

# --------------------------------------------------------------------------
# fake model used by the unit tests
# --------------------------------------------------------------------------


class _StubState:
    def __init__(self, n_layers=2, device="cpu"):
        self.device = torch.device(device)
        self.use_dci = False
        self.offload_win_flag = [False] * n_layers
        self.dci_db = [None] * n_layers
        self.batch_size = 1
        self.seq_len = 0


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Stands in for an ``enable_icecache()``-patched model.

    Mirrors the contract points the session relies on: a multi-token forward
    behaves like a prefill (resetting ``InferState.seq_len`` to the prompt
    length), a single-token forward behaves like a decode step (appending one
    token), and ``outputs.logits`` has shape ``[1, 1, V]`` because ``lm_head``
    is sliced to the last position.
    """

    def __init__(self, vocab=8, n_layers=2):
        self._icecache_infer_state = _StubState(n_layers=n_layers)
        self.calls = []
        self.vocab = vocab

    def __call__(self, *, input_ids, position_ids, cache_position, use_cache, return_dict):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
                "cache_position": cache_position.clone(),
                "use_cache": use_cache,
            }
        )
        st = self._icecache_infer_state
        n = int(input_ids.shape[1])
        if n > 1:
            # multi-token forward == prefill: _prepare_prefill() replaces the
            # whole sequence state, so seq_len is reset to the prompt length.
            st.seq_len = n
        else:
            # single-token forward == decode: append one token.
            st.seq_len += 1
        logits = torch.zeros(1, 1, self.vocab)
        logits[0, 0, int(input_ids[0, -1]) % self.vocab] = 1.0
        return _FakeOutputs(logits)


def _fake_session(vocab=8):
    model = _FakeModel(vocab=vocab)
    return TokenwiseContinuationReference(model), model


# --------------------------------------------------------------------------
# unit tests
# --------------------------------------------------------------------------


def test_alias_is_the_same_object():
    assert IceCacheAgentSession is TokenwiseContinuationReference


def test_missing_infer_state_raises():
    with pytest.raises(SessionStateError, match="_icecache_infer_state"):
        TokenwiseContinuationReference(types.SimpleNamespace())


def test_start_rejects_single_token_prompt():
    sess, _ = _fake_session()
    with pytest.raises(ValueError, match="more than one token"):
        sess.start(torch.tensor([[7]]))
    assert sess.phase is SessionPhase.IDLE


def test_calls_before_start_raise():
    sess, _ = _fake_session()
    for call in (
        lambda: sess.step(torch.tensor([[1]])),
        lambda: sess.append_tokens(torch.tensor([[1, 2]])),
        lambda: sess.append_transcript(torch.tensor([[1, 2]])),
        lambda: sess.generate_greedy(max_new_tokens=1),
    ):
        with pytest.raises(SessionStateError):
            call()


def test_start_then_step_keeps_invariant():
    sess, model = _fake_session()
    sess.start(torch.tensor([[1, 2, 3]]))
    assert sess.phase is SessionPhase.READY
    assert sess.seq_len == 3
    assert sess.committed_ids.shape[1] == 3
    assert sess.next_logits.shape == (1, 8)
    assert not sess.dci_active  # stub reports use_dci=False
    assert sess.has_pending_offload is False

    sess.step(torch.tensor([[4]]))
    assert sess.seq_len == 4
    assert sess.committed_ids.shape[1] == 4
    assert sess.n_steps == 1
    assert model.calls[-1]["input_ids"].shape == (1, 1)


def test_positions_are_absolute_and_contiguous():
    sess, model = _fake_session()
    sess.start(torch.tensor([[1, 2, 3, 4, 5]]))
    assert model.calls[0]["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert model.calls[0]["cache_position"].tolist() == [0, 1, 2, 3, 4]

    sess.append_tokens(torch.tensor([[6, 7, 8]]))
    seen = [c["position_ids"].tolist() for c in model.calls[1:]]
    assert seen == [[[5]], [[6]], [[7]]]
    assert [c["cache_position"].tolist() for c in model.calls[1:]] == [[5], [6], [7]]
    assert all(c["use_cache"] is False for c in model.calls)


def test_stop_condition_sees_the_committed_token():
    """The sampled token must be inside the KV cache before we test stopping."""
    sess, _ = _fake_session()
    sess.start(torch.tensor([[1, 2, 3]]))

    observed = []

    def stop(committed):
        observed.append(int(committed.shape[1]))
        return committed.shape[1] >= 4

    generated = sess.generate_greedy(max_new_tokens=10, stop_condition=stop)
    assert generated.shape[1] == 1
    assert observed[0] == 4  # not 3: the token was committed before the check
    assert sess.seq_len == 4 == sess.committed_ids.shape[1]


def test_step_rejects_multi_token_input():
    sess, _ = _fake_session()
    sess.start(torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="exactly one token|single token"):
        sess.step(torch.tensor([[3, 4]]))


def test_append_transcript_appends_only_the_suffix():
    sess, model = _fake_session()
    sess.start(torch.tensor([[1, 2, 3]]))
    n = sess.append_transcript(torch.tensor([[1, 2, 3, 4, 5]]))
    assert n == 2
    assert sess.committed_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert [c["input_ids"].tolist() for c in model.calls[1:]] == [[[4]], [[5]]]


def test_append_transcript_with_no_new_tokens_is_a_noop():
    sess, model = _fake_session()
    sess.start(torch.tensor([[1, 2, 3]]))
    n_calls = len(model.calls)
    assert sess.append_transcript(torch.tensor([[1, 2, 3]])) == 0
    assert len(model.calls) == n_calls
    assert sess.seq_len == 3


def test_prefix_mismatch_raises_and_leaves_state_untouched():
    sess, model = _fake_session()
    sess.start(torch.tensor([[1, 2, 3]]))
    n_calls = len(model.calls)
    with pytest.raises(PrefixMismatchError, match="first divergence at index 2"):
        sess.append_transcript(torch.tensor([[1, 2, 99, 4, 5]]))
    # nothing was appended, no forward was issued
    assert len(model.calls) == n_calls
    assert sess.seq_len == 3
    assert sess.committed_ids.tolist() == [[1, 2, 3]]


def test_prefix_mismatch_on_shorter_transcript():
    sess, _ = _fake_session()
    sess.start(torch.tensor([[1, 2, 3, 4]]))
    with pytest.raises(PrefixMismatchError, match="fewer than"):
        sess.append_transcript(torch.tensor([[1, 2, 3]]))


def test_start_twice_raises_until_reset():
    sess, _ = _fake_session()
    sess.start(torch.tensor([[1, 2]]))
    with pytest.raises(SessionStateError, match="only be called once"):
        sess.start(torch.tensor([[1, 2]]))
    sess.reset()
    assert sess.phase is SessionPhase.IDLE
    sess.start(torch.tensor([[1, 2]]))
    assert sess.seq_len == 2


# --------------------------------------------------------------------------
# gpu tests
# --------------------------------------------------------------------------

_FILLER = "The quick brown fox jumps over the lazy dog. " * 600

CFG_FULL = dict(
    page_size=16,
    page_budgets=64,
    n_sink_pages=2,
    n_win_pages=2,
    page_topks=0,
    n_max_bytes=4 << 28,
    n_max_cpu_bytes=8 << 28,
)
CFG_SPARSE = dict(
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to("cuda")
    return model.eval()


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def filler_ids(tokenizer):
    return tokenizer(_FILLER, return_tensors="pt").input_ids[0]


def _ids(filler_ids, n):
    assert filler_ids.numel() >= n, f"filler has {filler_ids.numel()} tokens, need {n}"
    return filler_ids[:n].unsqueeze(0).cuda()


def _new_session(model, **cfg):
    from icecache.adapter import enable_icecache

    enable_icecache(
        model, dtype=torch.float16, device=torch.device("cuda"), **cfg
    )
    return IceCacheAgentSession(model)


def _gather_kv(state):
    """Per-layer K/V for tokens [0, seq_len) gathered from the GPU pool."""
    out = []
    for kvc in state.kv_caches:
        pages = kvc.buffer[kvc.c2p[0]]          # [n_pages, 2, H_kv, page, dim]
        flat = pages.permute(0, 3, 1, 2, 4).reshape(
            -1, 2, state.n_kv_heads, state.head_dim
        )
        out.append(flat[: state.seq_len].clone())
    return out


def _drop(session):
    del session
    gc.collect()
    torch.cuda.empty_cache()


@gpu_only
def test_full_cache_parity(hf_model, filler_ids):
    """one-shot prefill(A+B) vs start(A) + tokenwise append(B), no offloading."""
    ids = _ids(filler_ids, 260)
    A, B, AB = ids[:, :200], ids[:, 200:260], ids

    s1 = _new_session(hf_model, **CFG_FULL)
    s1.start(AB)
    logits1 = s1.next_logits.clone()
    kv1 = _gather_kv(s1.state)
    assert not s1.dci_active, "expected the full-cache path for this budget"
    top1_1 = int(logits1.argmax(-1))

    s2 = _new_session(hf_model, **CFG_FULL)
    s2.start(A)
    assert s2.seq_len == 200
    s2.append_tokens(B)
    logits2 = s2.next_logits.clone()
    kv2 = _gather_kv(s2.state)
    top1_2 = int(logits2.argmax(-1))

    assert s1.seq_len == s2.seq_len == 260
    assert s1.seq_len == s1.committed_ids.shape[1]
    assert s2.seq_len == s2.committed_ids.shape[1]

    logit_gap = (logits1 - logits2).abs().max().item()
    kv_gaps = [(a.float() - b.float()).abs() for a, b in zip(kv1, kv2)]
    kv_max = max(g.max().item() for g in kv_gaps)
    kv_mean = [g.mean().item() for g in kv_gaps]
    print(
        f"\n[parity] max |logit diff| = {logit_gap:.4f}   "
        f"top-1: one-shot={top1_1} tokenwise={top1_2}"
    )
    print(
        f"[parity] kv drift: layer0 mean={kv_mean[0]:.2e}  "
        f"all-layer mean max={max(kv_mean):.2e}  element-wise max={kv_max:.4f}"
    )

    # Why the K/V comparison is shaped like this:
    # layer 0's K/V is rope(k_proj(embed)) -- no attention involved -- so the only
    # difference between the two runs is batched GEMM (whole-prompt prefill) vs.
    # row-wise GEMV (one token at a time) fp16 rounding.  It is the sharpest
    # structural check available: a mis-placed token, a wrong page mapping or a
    # wrong RoPE position would blow it up immediately.
    assert kv_mean[0] < 1e-3, (
        f"layer-0 K/V mean drift {kv_mean[0]:.2e} points at a placement/position bug"
    )
    # Later layers amplify that rounding through 32 layers of attention+MLP.  A
    # structural error would shift the *mean*; a handful of outlier elements is
    # normal fp16 amplification.  Hence: gate on the mean, report the max.
    for i, m in enumerate(kv_mean):
        assert m < 0.05, f"layer {i}: mean K/V drift {m:.4f} is systematic, not rounding"
    assert top1_1 == top1_2, "greedy top-1 token must agree"
    torch.testing.assert_close(logits1, logits2, atol=0.2, rtol=0.0)

    _drop(s1)
    _drop(s2)


@gpu_only
def test_no_prepare_prefill_during_continuation(hf_model, filler_ids, monkeypatch):
    from icecache.infer_state import InferState

    calls = []
    original = InferState._prepare_prefill

    def spy(self, bsz, q_len):
        calls.append((int(bsz), int(q_len)))
        return original(self, bsz, q_len)

    monkeypatch.setattr(InferState, "_prepare_prefill", spy)

    # NOTE on prompt length: `prefill_backup_pages()` only leaves
    # n_win_pages == n_final_win_pages when `n_real_pages - budget` is at least
    # `budget - n_sink_pages - n_win_pages`.  A prompt merely a little longer
    # than the budget leaves a larger window, which then shrinks during decode
    # and changes `n_dci_pages` mid-sequence (see the xfail test below).  The
    # published benchmark uses contexts far longer than the budget, so we do the
    # same here: 900 tokens (57 pages) vs a 16-page budget.
    ids = _ids(filler_ids, 1000)
    A, B = ids[:, :900], ids[:, 900:1000]

    sess = _new_session(hf_model, **CFG_SPARSE)
    assert calls == []
    sess.start(A)
    after_start = len(calls)
    assert after_start == 1, f"initial prefill should call _prepare_prefill once, got {calls}"

    ids_before = sess.dci_object_ids
    sess.append_tokens(B)
    assert len(calls) == after_start, f"continuation re-prefilled: {calls}"
    assert sess.dci_object_ids == ids_before, "DCI objects were rebuilt"
    assert sess.seq_len == sess.committed_ids.shape[1] == 1000
    print(f"\n[_prepare_prefill] calls={calls} seq_len={sess.seq_len} dci_active={sess.dci_active}")

    _drop(sess)


@gpu_only
def test_dci_identity_and_points_grow(hf_model, filler_ids):
    """Sparse config: pages do get evicted, yet DCI objects must survive."""
    ids = _ids(filler_ids, 1000)
    A, B = ids[:, :900], ids[:, 900:1000]

    sess = _new_session(hf_model, **CFG_SPARSE)
    sess.start(A)
    assert sess.dci_active, "budget 16 with a 57-page prompt must engage DCI"

    ids_before = sess.dci_object_ids
    points_before = sess.n_dci_points

    sess.append_tokens(B)

    assert sess.dci_object_ids == ids_before, "DCI objects were rebuilt during continuation"
    points_after = sess.n_dci_points
    for before, after in zip(points_before, points_after):
        if before is None:
            assert after is None
        else:
            assert after >= before, "DCI point count must not shrink"
    assert sess.seq_len == sess.committed_ids.shape[1] == 1000
    print(f"\n[dci] points before={points_before}")
    print(f"[dci] points after ={points_after}")
    print(f"[dci] pending_offload={sess.has_pending_offload}")

    _drop(sess)


@gpu_only
def test_page_boundary(hf_model, filler_ids):
    """Append lengths that straddle page boundaries."""
    page_size = CFG_SPARSE["page_size"]
    A_LEN = 905  # 57 pages vs a 16-page budget -> window stays at n_final_win_pages
    ids = _ids(filler_ids, A_LEN + 8 * page_size)
    A = ids[:, :A_LEN]

    last_page_len = (A_LEN - 1) % page_size + 1
    remaining = page_size - last_page_len
    lengths = [1, remaining - 1, remaining, remaining + 1, page_size, 2 * page_size + 3]
    print(f"\n[boundary] A_LEN={A_LEN} last_page_len={last_page_len} remaining={remaining}")

    for L in lengths:
        B = ids[:, A_LEN : A_LEN + L]
        sess = _new_session(hf_model, **CFG_SPARSE)
        sess.start(A)
        ids_before = sess.dci_object_ids
        points_before = sess.n_dci_points

        sess.append_tokens(B)

        assert sess.seq_len == A_LEN + L, f"L={L}: seq_len={sess.seq_len}"
        assert sess.committed_ids.shape[1] == A_LEN + L
        assert int(sess.state.kv_last_page_len) == (A_LEN + L - 1) % page_size + 1, f"L={L}"
        assert sess.dci_object_ids == ids_before, f"L={L}: DCI objects rebuilt"
        for before, after in zip(points_before, sess.n_dci_points):
            if before is not None:
                assert after >= before, f"L={L}: DCI points shrank"
        # c2p must stay within the pool
        for kvc in sess.state.kv_caches:
            assert int(kvc.c2p.max()) < sess.state.n_max_pages, f"L={L}: page id out of range"

        print(
            f"[boundary] L={L:>3} seq_len={sess.seq_len} last_page_len="
            f"{int(sess.state.kv_last_page_len):>2} pending={sess.has_pending_offload} "
            f"dci_points={sess.n_dci_points[0]}"
        )
        _drop(sess)


def _vanilla_greedy(model, ids, n_new):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        out_tokens = [nxt]
        for _ in range(n_new - 1):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            out_tokens.append(nxt)
    return torch.cat(out_tokens, dim=-1)


@gpu_only
def test_vanilla_hf_oracle_full_cache(hf_model, filler_ids):
    """Un-patched HF model is the oracle -- valid only on the full-cache path."""
    from transformers import AutoModelForCausalLM

    K = 24
    ids = _ids(filler_ids, 300)
    A, B = ids[:, :240], ids[:, 240:300]

    vanilla = (
        AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        .to("cuda")
        .eval()
    )
    try:
        reference = _vanilla_greedy(vanilla, ids, K).tolist()
    finally:
        del vanilla
        gc.collect()
        torch.cuda.empty_cache()

    sess = _new_session(hf_model, **CFG_FULL)
    sess.start(A)
    sess.append_tokens(B)
    assert not sess.dci_active, "oracle comparison requires the full-cache path"
    got = sess.generate_greedy(max_new_tokens=K).tolist()

    print(f"\n[oracle] vanilla   ={reference}")
    print(f"[oracle] icecache  ={got}")
    assert got == reference, "greedy tokens diverged from the un-patched HF model"
    _drop(sess)


@gpu_only
def test_minimal_agent_loop(hf_model, filler_ids, monkeypatch):
    from icecache.infer_state import InferState

    calls = []
    original = InferState._prepare_prefill

    def spy(self, bsz, q_len):
        calls.append((int(bsz), int(q_len)))
        return original(self, bsz, q_len)

    monkeypatch.setattr(InferState, "_prepare_prefill", spy)

    ids = _ids(filler_ids, 1000)
    prompt = ids[:, :900]
    tool_result = ids[:, 900:1000]

    sess = _new_session(hf_model, **CFG_SPARSE)
    sess.start(prompt)

    # turn 1: assistant "emits a tool call", scripted pause after 8 tokens
    turn1 = sess.generate_greedy(
        max_new_tokens=8, stop_condition=lambda c: c.shape[1] >= 900 + 8
    )
    assert turn1.shape[1] == 8
    assert sess.seq_len == sess.committed_ids.shape[1] == 908
    print(f"\n[loop] after tool-call turn: seq_len={sess.seq_len} prepared_prefill={len(calls)}")

    # tool result arrives as text and is folded into the transcript
    full = torch.cat([sess.committed_ids, tool_result], dim=-1)
    appended = sess.append_transcript(full)
    assert appended == 100
    assert sess.seq_len == sess.committed_ids.shape[1] == 1008

    # turn 2: resume generation on top of the extended context
    turn2 = sess.generate_greedy(max_new_tokens=8)
    assert turn2.shape[1] == 8
    assert sess.seq_len == sess.committed_ids.shape[1] == 1016

    assert len(calls) == 1, f"the tree was rebuilt somewhere: {calls}"
    print(f"[loop] final seq_len={sess.seq_len} dci_points={sess.n_dci_points[0]} calls={calls}")

    _drop(sess)


@gpu_only
@pytest.mark.xfail(
    reason=(
        "Documented, pre-existing IceCache limitation -- unrelated to continuation. "
        "When the prompt is only modestly longer than the page budget, "
        "prefill_backup_pages() leaves n_win_pages > n_final_win_pages. During decode "
        "the window then shrinks, and _prepare_decode() recomputes "
        "n_dci_pages = budget - n_sink_pages - n_win_pages, which grows. "
        "_DCI_query() derives num_neighbours from the new n_dci_pages, so nn_idx_0 comes "
        "back wider than selected_page_idx left over from the previous step and "
        "DCI.diff_pages_by_head() trips its shape assert. "
        "The published benchmark uses contexts far longer than the budget, where "
        "n_dci_pages is capped and n_win_pages equals n_final_win_pages, so it never "
        "surfaces. Kept here so the regime is recorded rather than silently avoided."
    ),
    strict=False,
)
def test_documented_limit_dci_width_drift(hf_model, filler_ids):
    """400-token prompt vs a 16-page budget: the drift regime."""
    ids = _ids(filler_ids, 500)
    A, B = ids[:, :400], ids[:, 400:500]

    sess = _new_session(hf_model, **CFG_SPARSE)
    sess.start(A)
    sess.append_tokens(B)
    assert sess.seq_len == sess.committed_ids.shape[1] == 500
    _drop(sess)
