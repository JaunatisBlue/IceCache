"""Phase B1: multi-token continuation prefill over an existing full-cache KV.

This is the chunked counterpart of :class:`TokenwiseContinuationReference`
(phase A).  The oracle stays as it is and is NOT extended; this class exists to
be measured against it.

Scope is intentionally narrow (see ``AGENT.md``):

* full-cache only -- nothing may have been offloaded (``use_dci == False``), so
  the whole sequence is resident in the paged KV and a chunk is just a plain
  causal paged prefill over a contiguous KV;
* no DCI chunk retrieval, no page eviction, no window rotation;
* ``_prepare_prefill()`` / ``_finish_prefill()`` are never called during
  continuation;
* batch size 1, greedy only.

Because the KV is contiguous and fully resident here, no split
(non-causal-over-old + causal-over-chunk) attention and no LSE merge is needed.
That changes as soon as DCI-retained semantic pages enter the picture; the
building blocks for that case are validated separately in
``tests/test_b0_attention_merge.py`` and specified in
``docs/phase_b_continuation_prefill_design.md``.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Union

import torch
from torch import Tensor

from .agent_session import PrefixMismatchError, SessionPhase, SessionStateError
from ..infer_state import ForwardMode

__all__ = ["ChunkedContinuationPrefill"]

TokenIds = Union[Tensor, Sequence[int], int]

DEFAULT_CHUNK_SIZE = 64


class ChunkedContinuationPrefill:
    """Chunked continuation ingestion for IceCache agent sessions (phase B1)."""

    def __init__(
        self,
        model,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        require_full_cache: bool = True,
    ) -> None:
        state = getattr(model, "_icecache_infer_state", None)
        if state is None:
            raise SessionStateError(
                "model carries no `_icecache_infer_state`; call "
                "`icecache.adapter.enable_icecache(model, ...)` before creating a session"
            )
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._model = model
        self._state = state
        self._chunk_size = int(chunk_size)
        self._require_full_cache = bool(require_full_cache)
        self._device = torch.device(state.device)
        self._phase = SessionPhase.IDLE
        self._committed_ids: Optional[Tensor] = None
        self._next_logits: Optional[Tensor] = None

        # instrumentation
        self.forward_calls = 0
        self.forward_times: List[float] = []
        self.chunks_used: List[int] = []

    # ------------------------------------------------------------- read views

    @property
    def state(self):
        return self._state

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def committed_ids(self) -> Tensor:
        if self._committed_ids is None:
            raise SessionStateError("session has not been started")
        return self._committed_ids

    @property
    def seq_len(self) -> int:
        return 0 if self._committed_ids is None else int(self._state.seq_len)

    @property
    def next_logits(self) -> Tensor:
        if self._next_logits is None:
            raise SessionStateError("session has not been started")
        return self._next_logits

    @property
    def dci_active(self) -> bool:
        return bool(getattr(self._state, "use_dci", False))

    @property
    def total_forward_time(self) -> float:
        return float(sum(self.forward_times))

    # ------------------------------------------------------------------ drive

    def start(self, input_ids: TokenIds) -> Tensor:
        if self._phase is not SessionPhase.IDLE:
            raise SessionStateError("start() may only be called once; call reset() first")
        ids = self._as_batch(input_ids, what="input_ids")
        if ids.shape[1] <= 1:
            raise ValueError(f"initial prompt must exceed one token, got {ids.shape[1]}")

        outputs = self._forward(ids, ForwardMode.INITIAL_PREFILL, start_pos=0)

        if int(self._state.batch_size) != 1:
            raise SessionStateError(
                f"ChunkedContinuationPrefill supports batch_size == 1, got {self._state.batch_size}"
            )
        if int(self._state.seq_len) != int(ids.shape[1]):
            raise SessionStateError(
                f"initial prefill committed {self._state.seq_len} of {ids.shape[1]} tokens"
            )
        self._committed_ids = ids.clone()
        self._next_logits = outputs.logits[:, -1, :].detach()
        self._phase = SessionPhase.READY

        if self._require_full_cache and self.dci_active:
            raise SessionStateError(
                "phase B1 continuation requires a full-cache run (use_dci == False); this "
                "prefill offloaded pages, so the chunked path is not implemented for it. "
                "Increase the GPU page budget so the whole sequence fits."
            )
        return self._next_logits

    def step(self, token_id: TokenIds) -> Tensor:
        if self._phase is not SessionPhase.READY:
            raise SessionStateError("step() requires a completed start()")
        tok = self._as_single_token(token_id)
        pos = int(self._state.seq_len)
        outputs = self._forward(tok, ForwardMode.DECODE, start_pos=pos)
        new_len = int(self._state.seq_len)
        if new_len != pos + 1:
            raise SessionStateError(f"seq_len did not advance by 1: {pos} -> {new_len}")
        self._committed_ids = torch.cat([self._committed_ids, tok], dim=-1)
        self._next_logits = outputs.logits[:, -1, :].detach()
        self._check_invariant()
        return self._next_logits

    def append_tokens(self, token_ids: TokenIds, chunk_size: Optional[int] = None) -> int:
        """Commit a run of known tokens, `chunk_size` at a time.

        Returns the number of tokens appended.  The final chunk may be short.
        """
        if self._phase is not SessionPhase.READY:
            raise SessionStateError("append_tokens() requires a completed start()")
        ids = self._as_batch(token_ids, what="token_ids")
        n = int(ids.shape[1])
        size = int(chunk_size if chunk_size is not None else self._chunk_size)
        if size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {size}")

        i = 0
        while i < n:
            k = min(size, n - i)
            chunk = ids[:, i : i + k]
            start_pos = int(self._state.seq_len)
            if k == 1:
                # a single token is a decode step, not a continuation chunk
                self._step_inline(chunk)
            else:
                outputs = self._forward(chunk, ForwardMode.CONTINUATION_PREFILL, start_pos=start_pos)
                if int(self._state.seq_len) != start_pos + k:
                    raise SessionStateError(
                        f"continuation chunk committed {self._state.seq_len - start_pos} of {k} tokens"
                    )
                self._committed_ids = torch.cat([self._committed_ids, chunk], dim=-1)
                self._next_logits = outputs.logits[:, -1, :].detach()
                self.chunks_used.append(k)
            self._check_invariant()
            i += k
        return n

    def append_transcript(self, full_input_ids: TokenIds, chunk_size: Optional[int] = None) -> int:
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
                f"first divergence at index {at}"
            )
        suffix = full[:, n:]
        if suffix.shape[1] == 0:
            return 0
        return self.append_tokens(suffix, chunk_size=chunk_size)

    def generate_greedy(
        self,
        *,
        max_new_tokens: int,
        stop_condition: Optional[Callable[[Tensor], bool]] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """Greedy decode, committing each token before testing stop conditions."""
        if self._phase is not SessionPhase.READY:
            raise SessionStateError("generate_greedy() requires a completed start()")
        produced: List[Tensor] = []
        for _ in range(int(max_new_tokens)):
            nxt = torch.argmax(self._next_logits, dim=-1, keepdim=True)
            self.step(nxt)
            produced.append(nxt)
            if eos_token_id is not None and int(nxt.item()) == int(eos_token_id):
                break
            if stop_condition is not None and stop_condition(self._committed_ids):
                break
        if not produced:
            return torch.empty((1, 0), dtype=torch.long, device=self._device)
        return torch.cat(produced, dim=-1)

    def reset(self) -> None:
        self._committed_ids = None
        self._next_logits = None
        self._phase = SessionPhase.IDLE
        self.forward_calls = 0
        self.forward_times = []
        self.chunks_used = []

    # ---------------------------------------------------------------- helpers

    def _step_inline(self, tok: Tensor) -> None:
        pos = int(self._state.seq_len)
        outputs = self._forward(tok, ForwardMode.DECODE, start_pos=pos)
        if int(self._state.seq_len) != pos + 1:
            raise SessionStateError(f"decode did not advance seq_len by 1: {pos}")
        self._committed_ids = torch.cat([self._committed_ids, tok], dim=-1)
        self._next_logits = outputs.logits[:, -1, :].detach()

    def _forward(self, ids: Tensor, mode: ForwardMode, start_pos: int):
        n = int(ids.shape[1])
        position_ids = torch.arange(
            start_pos, start_pos + n, device=self._device, dtype=torch.long
        ).unsqueeze(0)
        state = self._state
        state.forward_mode = mode
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = self._model(
                    input_ids=ids,
                    position_ids=position_ids,
                    cache_position=position_ids[0],
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            state.forward_mode = None
            # the forward is async on GPU: without this sync, forward_times
            # records launch latency, not execution time (review finding)
            torch.cuda.synchronize()
            self.forward_times.append(time.perf_counter() - t0)
            self.forward_calls += 1
        return outputs

    def _check_invariant(self) -> None:
        if int(self._committed_ids.shape[1]) != int(self._state.seq_len):
            raise SessionStateError(
                "invariant violated: committed_ids="
                f"{self._committed_ids.shape[1]} vs state.seq_len={self._state.seq_len}"
            )

    def _as_batch(self, x: TokenIds, *, what: str) -> Tensor:
        if isinstance(x, int):
            x = torch.tensor([[x]], dtype=torch.long)
        elif isinstance(x, (list, tuple)):
            x = torch.tensor(list(x), dtype=torch.long)
        if not isinstance(x, Tensor):
            raise TypeError(f"{what} must be a torch.Tensor or a sequence of ints")
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
            raise TypeError("token_id must be a torch.Tensor or an int")
        tok = token_id.to(device=self._device, dtype=torch.long)
        if tok.ndim == 0:
            tok = tok.reshape(1, 1)
        elif tok.ndim == 1:
            tok = tok.reshape(1, -1)
        if tok.shape != (1, 1):
            raise ValueError(f"step() takes a single token, got {tuple(tok.shape)}")
        return tok
