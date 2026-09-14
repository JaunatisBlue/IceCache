"""Phase A of IceCache agent-continuation support.

This module implements :class:`TokenwiseContinuationReference` -- a
*teacher-forced decode reference* that appends **known** tokens (e.g. a tool
result) to an existing IceCache KV cache and DCI tree.

Design intent
-------------
Every appended token costs one full model forward, because it is submitted
through IceCache's ordinary single-token decode path (``q_len == 1``).  This is
deliberately **not** a high-performance continuation prefill; it exists to be an
unambiguous correctness baseline:

* ``_prepare_prefill()`` must not run again (the DCI tree must survive),
* absolute RoPE positions must stay contiguous,
* the last tool-call token must be inside the KV cache before the tool runs,
* page-boundary / ``_DCI_add()`` timing must stay untouched,
* ``committed_ids.shape[1] == state.seq_len`` at all times.

The chunked continuation prefill (explicit INITIAL_PREFILL /
CONTINUATION_PREFILL / DECODE states, microchunks, split non-causal + causal
attention merged through ``merge_state``) is specified separately in
``docs/phase_b_continuation_prefill_design.md`` and is intentionally **not**
implemented here.

Agent-facing usage::

    from icecache.adapter import enable_icecache
    from icecache.adapter.agent_session import IceCacheAgentSession

    enable_icecache(model, dtype=torch.float16, device=dev, page_size=16,
                    page_budgets=64, n_sink_pages=2, n_win_pages=2,
                    n_max_bytes=..., n_max_cpu_bytes=...)
    session = IceCacheAgentSession(model)
    session.start(prompt_ids)                       # one initial prefill
    session.generate_greedy(max_new_tokens=64, stop_condition=is_tool_call)
    session.append_transcript(full_transcript_ids)  # tool result, token-wise
    session.generate_greedy(max_new_tokens=64, stop_condition=is_tool_call)
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, Sequence, Union

import torch
from torch import Tensor

__all__ = [
    "TokenwiseContinuationReference",
    "IceCacheAgentSession",
    "PrefixMismatchError",
    "SessionStateError",
    "SessionPhase",
]

TokenIds = Union[Tensor, Sequence[int], int]


class SessionPhase(Enum):
    """Session-level lifecycle.

    Note: this is *not* the attention-dispatch mode.  Phase A still lets
    ``modeling._icecache_attn_forward`` pick prefill vs decode purely from
    ``q_len``.  Phase B replaces that with explicit INITIAL_PREFILL /
    CONTINUATION_PREFILL / DECODE modes.
    """

    IDLE = "idle"
    READY = "ready"


class PrefixMismatchError(ValueError):
    """``committed_ids`` is not a strict token prefix of the new transcript.

    Raised instead of silently rebuilding the cache.  Typical causes: the chat
    template rewrote an older message, a BOS was duplicated, the tool-call JSON
    was reformatted, role markers changed, or the agent framework truncated or
    summarised history.
    """


class SessionStateError(RuntimeError):
    """The session was driven out of order, or the underlying state is unusable."""


class TokenwiseContinuationReference:
    """Teacher-forced decode reference for agent continuation ingestion.

    Single sequence (batch size 1), greedy generation only.
    """

    def __init__(self, model, *, require_dci: bool = False) -> None:
        state = getattr(model, "_icecache_infer_state", None)
        if state is None:
            raise SessionStateError(
                "model carries no `_icecache_infer_state`; call "
                "`icecache.adapter.enable_icecache(model, ...)` before creating a session"
            )
        self._model = model
        self._state = state
        self._require_dci = bool(require_dci)
        self._device = torch.device(state.device)
        self._phase = SessionPhase.IDLE
        self._committed_ids: Optional[Tensor] = None
        self._next_logits: Optional[Tensor] = None
        self._n_steps = 0

    # ------------------------------------------------------------- read views

    @property
    def state(self):
        """The underlying :class:`~icecache.infer_state.InferState`."""
        return self._state

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def committed_ids(self) -> Tensor:
        """``[1, N]`` token ids physically present in the KV cache."""
        if self._committed_ids is None:
            raise SessionStateError("session has not been started")
        return self._committed_ids

    @property
    def seq_len(self) -> int:
        """Number of committed tokens.

        ``InferState.seq_len`` reads ``kv_caches[0]``, which is ``None`` until the
        first prefill, so it must not be touched while the session is IDLE.
        """
        return 0 if self._committed_ids is None else int(self._state.seq_len)

    @property
    def next_logits(self) -> Tensor:
        """``[1, V]`` logits for the token following the last committed token."""
        if self._next_logits is None:
            raise SessionStateError("session has not been started")
        return self._next_logits

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def dci_active(self) -> bool:
        """Whether the initial prefill engaged the sparse DCI path."""
        return bool(getattr(self._state, "use_dci", False))

    @property
    def has_pending_offload(self) -> bool:
        """True when a window page was backed up but not yet inserted into DCI.

        This is a normal transient state at a page boundary: the actual
        ``_DCI_add()`` happens at the start of the next decode step.
        """
        flags = getattr(self._state, "offload_win_flag", None)
        return bool(flags) and any(flags)

    @property
    def dci_object_ids(self) -> List[Optional[int]]:
        """``id()`` of each layer's DCI object, for identity assertions."""
        return [None if db is None else id(db) for db in self._state.dci_db]

    @property
    def n_dci_points(self) -> List[Optional[int]]:
        out: List[Optional[int]] = []
        for db in self._state.dci_db:
            if db is None:
                out.append(None)
                continue
            try:
                out.append(int(db.num_points[0]))
            except Exception:
                out.append(None)
        return out

    # ------------------------------------------------------------------ drive

    def start(self, input_ids: TokenIds) -> Tensor:
        """Run the one and only initial prefill and return the next-token logits."""
        if self._phase is not SessionPhase.IDLE:
            raise SessionStateError("start() may only be called once; call reset() first")
        ids = self._as_batch(input_ids, what="input_ids")
        if ids.shape[1] <= 1:
            raise ValueError(
                f"initial prompt must contain more than one token, got {ids.shape[1]}; "
                "a single token would be dispatched to the decode path with no prepared state"
            )

        with torch.no_grad():
            position_ids = torch.arange(
                ids.shape[1], device=self._device, dtype=torch.long
            ).unsqueeze(0)
            outputs = self._model(
                input_ids=ids,
                position_ids=position_ids,
                cache_position=position_ids[0],
                use_cache=False,
                return_dict=True,
            )

        # Only now do the per-layer caches exist, so this is the earliest point
        # at which batch_size / seq_len may be read.
        if int(self._state.batch_size) != 1:
            raise SessionStateError(
                "TokenwiseContinuationReference supports batch_size == 1, got "
                f"{self._state.batch_size}"
            )
        if int(self._state.seq_len) != int(ids.shape[1]):
            raise SessionStateError(
                f"initial prefill did not commit the whole prompt: seq_len="
                f"{self._state.seq_len} vs prompt={ids.shape[1]}"
            )

        self._committed_ids = ids.clone()
        self._next_logits = outputs.logits[:, -1, :].detach()
        self._phase = SessionPhase.READY

        if self._require_dci and not self.dci_active:
            raise SessionStateError(
                "require_dci=True but the initial prefill did not engage the DCI path "
                f"(use_dci={self.dci_active}). Either the prompt is too short for a DCI tree "
                "to be built (q_len must exceed page_size*(n_sink_pages+n_win_pages)), or it "
                "fits entirely inside the GPU page budget so nothing needed offloading. Note "
                "that this codebase currently has no path that enables DCI or builds the tree "
                "later, mid-sequence, so a short initial prompt permanently pins the session "
                "to the full-cache path."
            )
        return self._next_logits

    def step(self, token_id: TokenIds) -> Tensor:
        """Commit one known token to the KV cache; return the following logits."""
        if self._phase is not SessionPhase.READY:
            raise SessionStateError("step() requires a completed start()")
        tok = self._as_single_token(token_id)

        pos = int(self._state.seq_len)  # absolute position of the token being committed
        with torch.no_grad():
            position_ids = torch.tensor([[pos]], dtype=torch.long, device=self._device)
            cache_position = torch.tensor([pos], dtype=torch.long, device=self._device)
            outputs = self._model(
                input_ids=tok,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
                return_dict=True,
            )

        new_len = int(self._state.seq_len)
        if new_len != pos + 1:
            raise SessionStateError(
                f"committed token did not advance seq_len by exactly 1: {pos} -> {new_len}"
            )

        self._committed_ids = torch.cat([self._committed_ids, tok], dim=-1)
        self._next_logits = outputs.logits[:, -1, :].detach()
        self._n_steps += 1

        if int(self._committed_ids.shape[1]) != new_len:
            raise SessionStateError(
                "invariant violated: committed_ids="
                f"{self._committed_ids.shape[1]} vs state.seq_len={new_len}"
            )
        return self._next_logits

    def append_tokens(self, token_ids: TokenIds) -> int:
        """Commit a run of known tokens one at a time. Returns the number appended.

        The DCI tree is never touched directly here: new pages stay in the GPU
        window and only leave it through the existing offload timing.
        """
        if self._phase is not SessionPhase.READY:
            raise SessionStateError("append_tokens() requires a completed start()")
        ids = self._as_batch(token_ids, what="token_ids")
        for i in range(ids.shape[1]):
            self.step(ids[:, i : i + 1])
        return int(ids.shape[1])

    def append_transcript(self, full_input_ids: TokenIds) -> int:
        """Append only the *new suffix* of a full transcript. Returns suffix length.

        The committed prefix is validated first, and a mismatch raises before any
        state is mutated.
        """
        if self._phase is not SessionPhase.READY:
            raise SessionStateError("append_transcript() requires a completed start()")
        full = self._as_batch(full_input_ids, what="full_input_ids")
        committed = self._committed_ids
        n = int(committed.shape[1])

        if full.shape[1] < n:
            raise PrefixMismatchError(
                f"new transcript has {full.shape[1]} tokens, fewer than the {n} committed tokens"
            )
        if not torch.equal(full[:, :n], committed):
            divergence = (full[0, :n] != committed[0]).nonzero()
            at = int(divergence[0].item()) if divergence.numel() else -1
            raise PrefixMismatchError(
                "committed_ids is not a strict prefix of the supplied transcript; "
                f"first divergence at index {at} "
                f"(committed={int(committed[0, at])}, new={int(full[0, at])})"
            )

        suffix = full[:, n:]
        if suffix.shape[1] == 0:
            return 0
        return self.append_tokens(suffix)

    def generate_greedy(
        self,
        *,
        max_new_tokens: int,
        stop_condition: Optional[Callable[[Tensor], bool]] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """Greedy decode. Commits each sampled token *before* testing stop conditions.

        That ordering matters: stopping on a freshly sampled token without
        committing it would leave a token present in the logical transcript but
        absent from the KV cache.
        """
        if self._phase is not SessionPhase.READY:
            raise SessionStateError("generate_greedy() requires a completed start()")

        produced: List[Tensor] = []
        for _ in range(int(max_new_tokens)):
            nxt = torch.argmax(self._next_logits, dim=-1, keepdim=True)  # [1, 1]
            self.step(nxt)  # commit first ...
            produced.append(nxt)
            # ... then decide whether to stop.
            if eos_token_id is not None and int(nxt.item()) == int(eos_token_id):
                break
            if stop_condition is not None and stop_condition(self._committed_ids):
                break

        if not produced:
            return torch.empty((1, 0), dtype=torch.long, device=self._device)
        return torch.cat(produced, dim=-1)

    def reset(self) -> None:
        """Drop session bookkeeping.

        The KV cache / DCI tree are *not* torn down here; the next ``start()``
        rebuilds them through ``_prepare_prefill()``, which remains the only
        place that resets IceCache state.
        """
        self._committed_ids = None
        self._next_logits = None
        self._n_steps = 0
        self._phase = SessionPhase.IDLE

    # ---------------------------------------------------------------- helpers

    def _as_batch(self, x: TokenIds, *, what: str) -> Tensor:
        if isinstance(x, int):
            x = torch.tensor([[x]], dtype=torch.long)
        elif isinstance(x, (list, tuple)):
            x = torch.tensor(list(x), dtype=torch.long)
        if not isinstance(x, Tensor):
            raise TypeError(
                f"{what} must be a torch.Tensor or a sequence of ints, got {type(x).__name__}"
            )
        x = x.to(device=self._device, dtype=torch.long)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"{what} must have shape [1, N] or [N], got {tuple(x.shape)}")
        return x

    def _as_single_token(self, token_id: TokenIds) -> Tensor:
        if isinstance(token_id, int):
            token_id = torch.tensor([[token_id]], dtype=torch.long)
        elif isinstance(token_id, (list, tuple)):
            token_id = torch.tensor(list(token_id), dtype=torch.long)
        if not isinstance(token_id, Tensor):
            raise TypeError(
                f"token_id must be a torch.Tensor or an int, got {type(token_id).__name__}"
            )
        tok = token_id.to(device=self._device, dtype=torch.long)
        if tok.ndim == 0:
            tok = tok.reshape(1, 1)
        elif tok.ndim == 1:
            if tok.numel() != 1:
                raise ValueError(f"step() takes exactly one token, got {tok.numel()}")
            tok = tok.reshape(1, 1)
        elif tok.ndim != 2 or tok.shape != (1, 1):
            raise ValueError(
                f"step() takes a single token with shape [1, 1] or a scalar, got {tuple(tok.shape)}"
            )
        return tok


#: Name used by the agent-side API. Same object: phase A is a reference.
IceCacheAgentSession = TokenwiseContinuationReference
