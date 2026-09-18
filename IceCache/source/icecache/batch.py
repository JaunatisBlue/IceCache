"""Fixed-capacity decode batch over independently prefilled IceCache requests.

The batch controls a fixed number of slots (its ``capacity``).  A slot is either
*active* (it holds a prefilled request that participates in decode) or *free*.
``active_indices`` lists active slot indices in ascending order and drives the
forward, query and page-ownership checks.  The batch therefore supports any
number of concurrent requests ``B`` (``batch_size`` is the live active count),
variable-length requests (each row carries its own ``position_ids``), and
request exit/reuse via :meth:`retire` / :meth:`admit` -- but it does **not**
contain a scheduler: :meth:`admit` is called explicitly by the caller.
"""

import os
import hashlib
import warnings
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter
import asyncio

import numpy as np
import torch
import torch.nn.functional as F

from . import kernels
from .infer_state import ForwardMode


# Request-level attributes that every member must share so the batch can build
# one paged-attention call and one model forward over all of them.  ``seq_len``
# is deliberately excluded: requests may have different lengths.
_CONFIG_ATTRS = (
    "n_layers", "n_qo_heads", "n_kv_heads", "head_dim", "page_size",
    "dtype", "device", "layout", "n_sink_pages", "n_win_pages",
    "n_groups", "offload_ratio", "ratio", "search_ratio", "use_dci",
)

# FlashInfer scratch space reserved per concurrent request row.  The serial path
# allocates 16 MiB per handler (infer_state.py builds one buffer per budget
# group), which is the figure this mirrors.  See BatchInferState._workspace_bytes.
_WORKSPACE_PER_REQUEST = 16 * 1024 * 1024

# Length-grouped prefill (``prefill_batch(token_budget=...)``) splits the batch
# into several smaller forwards.  Splitting is only worth it when it removes
# enough padding to pay for the extra forwards, so it is accepted only if the
# grouped plan's padded-row count beats the single-forward plan by this factor.
# Near-uniform lengths leave the plan at one group and cost nothing.
_PREFILL_GROUP_MIN_GAIN = 1.15


class BatchInferState:
    """Synchronous decode over a fixed-capacity set of independently prefilled requests.

    Each member owns its DCI, CPU cache, temporary buffers and decode state.
    All members must have been prefilled separately into the same GPU KvPool.
    The batch size is the number of *active* slots and is not fixed at 2: callers
    may :meth:`retire` a finished slot and later :meth:`admit` a freshly
    prefilled request into it.
    """

    def __init__(self, states, query_backend="serial", query_threads=32, prefilled=True):
        if len(states) < 1:
            raise ValueError("the batch prototype needs at least one prefilled request")
        self.capacity = len(states)
        self.states = list(states)
        self._ref_state = states[0]
        self._pool = states[0]._pool
        for i, state in enumerate(self.states):
            if prefilled:
                self._check_prefilled(state, i)
            self._check_compatible(state, i)
        a = states[0]
        self.n_layers = a.n_layers
        self.n_qo_heads = a.n_qo_heads
        self.n_kv_heads = a.n_kv_heads
        self.head_dim = a.head_dim
        self.page_size = a.page_size
        self.dtype = a.dtype
        self.device = a.device
        self.layout = a.layout
        self.forward_mode = ForwardMode.DECODE
        if query_backend not in ("serial", "native"):
            raise ValueError("query_backend must be 'serial' or 'native'")
        if query_threads < 1:
            raise ValueError("query_threads must be positive")
        self.query_backend = query_backend
        self.query_threads = query_threads
        if query_backend == "native":
            if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
                raise ValueError("set OPENBLAS_NUM_THREADS=1 before native batch query")
            from . import _mdci_batch
            import dciknn._dci as installed_dci
            producer_sha = hashlib.sha256(Path(installed_dci.__file__).read_bytes()).hexdigest()
            if producer_sha != _mdci_batch.producer_sha256:
                raise RuntimeError("M-DCI producer binary changed; rebuild and validate the batch extension")
            self._native = _mdci_batch
        # Workspace shared by the decode and prefill wrappers.  The serial path
        # does exactly the same: infer_state.py builds one 16 MiB buffer per
        # budget group and hands that same buffer to the group's prefill *and*
        # decode wrappers, so sharing here mirrors the reference rather than
        # inventing anything.  Size it from the *live* row count rather than the
        # slot capacity (a capacity larger than the active set would otherwise
        # reserve memory for rows that never exist) and grow it on demand in
        # _ensure_workspace.
        self._workspace = torch.empty(
            self._workspace_bytes(len(self.active_indices)),
            dtype=torch.uint8, device=self.device)
        self.workspace_bytes = self._workspace.numel()
        self._handler = kernels.BatchDecodeWithPagedKVCacheWrapper(self._workspace, self.layout)
        self._prefill_handler = kernels.BatchPrefillWithPagedKVCacheWrapper(self._workspace, self.layout)
        self._prefill_real_lens = None
        self._prefill_lmax = 0
        # Slot indices of the rows currently inside ``prefill_attention_forward``.
        # With one forward per batch this is just ``active_indices``; length-
        # grouped prefill (see ``_plan_prefill_groups``) runs several forwards,
        # and each one covers only its own group.
        self._prefill_active = None
        # Padding accounting for the last ``prefill_batch`` call: how many model
        # forwards covered it, and how many rows of model compute they cost
        # (sum over groups of ``rows * Lmax``) versus the real token count.
        self.prefill_groups = 0
        self.prefill_padded_rows = 0
        self.prefill_real_rows = 0
        # Single stream for every request's recall copies in the batched decode
        # path.  Each InferState has its own c2g_stream, which would force one
        # synchronise per request per layer; sharing one stream here lets a single
        # synchronise per layer cover all of them.
        self._c2g_stream = torch.cuda.Stream(self.device)
        # Deep page-ownership validation is O(requests x layers) host reads, so it
        # is not run on every decode step (see _validate_step) unless asked for.
        self._deep_validate = os.environ.get("ICECACHE_BATCH_DEEP_VALIDATE") == "1"
        # Per-slot containers indexed by slot index; length stays at capacity so
        # callers can read a slot by index.  Inactive slots keep their last value
        # but are never incremented (and are reset on admit).
        self.query_counts = [0] * self.capacity
        self.query_counts_by_layer = [[0] * self.n_layers for _ in range(self.capacity)]
        # Kept for consumers of the original probe, regardless of backend.
        self.native_query_counts = self.query_counts
        self.batch_query_seconds = 0.0
        # Split of ``batch_query_seconds``, to tell the native tree search apart
        # from the per-request Python bookkeeping that follows it: the search is
        # already parallel (one OpenMP team), the bookkeeping is not, and the two
        # need different fixes.
        self.native_search_seconds = 0.0
        self.native_bookkeep_seconds = 0.0
        self.serial_query_seconds = 0.0
        self.batch_step_seconds = 0.0
        self.decode_steps = 0
        self._closed = False
        self._failed = False
        self._step_active = False
        self._forward_lock = Lock()
        self._step_thread = None
        self._query_active = False
        self._next_layer = 0
        self._decode_handlers_armed = False
        # Only a batch of already-prefilled requests can be validated here.  With
        # ``prefilled=False`` the members have no KvCache yet (kv_caches is still
        # [None] * n_layers), so validate_ready() would dereference None; the
        # check runs after prefill_batch instead (step/_step_impl re-validates).
        if prefilled:
            self.validate_ready()

    # ------------------------------------------------------------------ #
    # Slot bookkeeping
    # ------------------------------------------------------------------ #
    @property
    def active_indices(self):
        """Ascending list of slot indices that currently hold a request."""
        return [i for i, s in enumerate(self.states) if s is not None]

    @property
    def batch_size(self):
        """Number of currently active (non-free) request slots."""
        return len(self.active_indices)

    @property
    def seq_lens(self):
        """Per-row sequence lengths, in ascending active-slot order."""
        return tuple(s.seq_len for s in self.states if s is not None)

    def _active_states(self):
        return [self.states[i] for i in self.active_indices]

    def close(self):
        """Stop use of this batch; request caches and the shared pool stay caller-owned.

        ``close()`` does not repair any KV state and does not free the GPU pool
        (the caller owns both).  A failed step marks the batch unusable; it is
        not a request recycling or continuous-batching API.
        """
        if not self._forward_lock.acquire(blocking=False):
            raise RuntimeError("cannot close a batch during a model forward")
        try:
            self._closed = True
        finally:
            self._forward_lock.release()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("batch state is closed")
        if self._failed:
            raise RuntimeError("batch forward failed after modifying KV state; create fresh request states")
        if self._step_active and self._step_thread != get_ident():
            raise RuntimeError("batch state is in use by another thread")
        if self._forward_lock.locked() and self._step_thread != get_ident():
            raise RuntimeError("batch step is being prepared by another call")

    # ------------------------------------------------------------------ #
    # Workspace sizing
    # ------------------------------------------------------------------ #
    def _workspace_bytes(self, n_requests):
        """Bytes of FlashInfer scratch space for ``n_requests`` concurrent rows.

        The serial path gives every handler 16 MiB (``infer_state.py`` builds one
        such buffer per budget group), so a batch of B rows is sized at B x 16 MiB
        -- a conservative reading of the existing allocation, not a characterised
        requirement.  ``ICECACHE_BATCH_WORKSPACE_MB`` overrides it wholesale.
        """
        override = os.environ.get("ICECACHE_BATCH_WORKSPACE_MB")
        if override:
            try:
                mib = int(override)
            except ValueError:
                raise ValueError("ICECACHE_BATCH_WORKSPACE_MB must be an integer")
            if mib < 1:
                raise ValueError("ICECACHE_BATCH_WORKSPACE_MB must be positive")
            return mib * 1024 * 1024
        need = _WORKSPACE_PER_REQUEST * max(1, int(n_requests))
        if need > (1 << 30):
            warnings.warn(
                f"batch workspace would be {need >> 20} MiB for {n_requests} rows; "
                "set ICECACHE_BATCH_WORKSPACE_MB to override if unintended",
                stacklevel=2)
        return need

    def _ensure_workspace(self, n_requests):
        """Grow the shared workspace if the batch now needs more than it has."""
        need = self._workspace_bytes(n_requests)
        if need <= self.workspace_bytes:
            return self.workspace_bytes
        self._workspace = torch.empty(need, dtype=torch.uint8, device=self.device)
        self.workspace_bytes = self._workspace.numel()
        # The decode and prefill wrappers share this buffer, so both must be
        # re-pointed at the new one.
        self._handler.reset_workspace_buffer(self._workspace)
        self._prefill_handler.reset_workspace_buffer(self._workspace)
        return self.workspace_bytes

    def _check_prefilled(self, state, i):
        if (len(state.kv_caches) != state.n_layers or
                any(cache is None for cache in state.kv_caches) or
                state.seq_len < 1):
            raise ValueError(f"request {i} has not completed prefill")
        if not state.use_dci:
            raise ValueError(f"request {i} did not enter sparse DCI mode")
        if state.use_dci and any(
                state.layer2budget[layer] is not None and state.dci_db[layer] is None
                for layer in range(state.n_layers)):
            raise ValueError(f"request {i} has an unbuilt DCI index")
        if state._pool is not self._pool:
            raise ValueError(f"request {i} does not share the batch GPU KV pool")
        if state.batch_size != 1:
            raise ValueError(f"request {i} must have been prefilled with batch_size=1")

    def _check_compatible(self, state, i):
        ref = self._ref_state
        for attr in _CONFIG_ATTRS:
            if getattr(state, attr) != getattr(ref, attr):
                raise ValueError(f"request {i} configuration disagrees on {attr}")
        if state.layer2budget != ref.layer2budget:
            raise ValueError(f"request {i} uses a different layer budget set")
        if state.n_prefetch_layers or ref.n_prefetch_layers:
            raise ValueError("batch prototype does not support cross-layer prefetch")
        if state.n_reuse_layers or ref.n_reuse_layers:
            raise ValueError("batch prototype does not support cross-layer reuse")

    @classmethod
    def from_prefilled(cls, states, **kwargs):
        """Construct a batch from requests independently prefilled in one pool."""
        return cls(states, **kwargs)

    # ------------------------------------------------------------------ #
    # Batched prefill: one model forward over B padded prompts
    # ------------------------------------------------------------------ #
    def _plan_prefill_groups(self, real_lens, token_budget):
        """Partition active rows into groups of similar length.

        A forward over ``n`` rows padded to ``Lmax`` costs ``n * Lmax`` rows of
        model compute even though only ``sum(real_lens)`` of them carry real
        tokens, so one long request in an otherwise short batch wastes most of
        the forward.  Rows are packed longest-first, opening a new group only
        when the candidate group's padded cost would exceed ``token_budget``,
        which keeps each group's padding small while still amortising a forward
        over several requests.
        """
        groups = []
        for pos in sorted(range(len(real_lens)), key=lambda p: -real_lens[p]):
            for group in groups:
                nmax = max([real_lens[q] for q in group] + [real_lens[pos]])
                if (len(group) + 1) * nmax <= token_budget:
                    group.append(pos)
                    break
            else:
                groups.append([pos])
        for group in groups:
            group.sort()
        groups.sort(key=lambda g: g[0])
        return groups

    def prefill_batch(self, model, prompts, token_budget=None):
        # Prefill every active request with model forwards.  ``prompts`` is a list
        # of 1-D LongTensor token ids, one per active slot in ``active_indices``
        # order.  With ``token_budget=None`` (default) all prompts are padded to a
        # single dense [B, Lmax] grid and fed to ``model(...)`` once -- the
        # original behaviour, unchanged.  With a token budget the rows are split
        # into length-similar groups (see ``_plan_prefill_groups``) so each forward
        # pads to its own group's Lmax, which removes most of the padding when the
        # batch is length-skewed; near-uniform batches stay at one group.
        # Padding is excluded everywhere: each request writes only its real tokens
        # into its OWN KvCache (the b == 0 assumption is preserved, every member
        # still has batch_size == 1), the FlashInfer prefill attention is a ragged
        # batch keyed by each request's real length, and each DCI tree is still
        # built independently via the unchanged per-request ``_DCI_*`` methods.
        # After the forward each member is fully prefilled and ready for
        # :meth:`step`.
        #
        # Returns ``(logits, next_tokens)`` where ``logits`` has shape
        # [B, 1, vocab] (only each request's last *real* token is materialised --
        # the LM head is patched to gather that position instead of projecting the
        # whole padded grid) and ``next_tokens[i]`` is its argmax.
        active = self.active_indices
        if len(prompts) != len(active):
            raise ValueError("prompts length must match the number of active slots")
        # Guarantee the shared workspace covers this many concurrent rows before
        # any wrapper touches it.
        self._ensure_workspace(len(active))
        real_lens = [int(p.shape[0]) for p in prompts]
        if any(L < 2 for L in real_lens):
            raise ValueError("batched prefill requires q_len > 1 for every request")
        Bnum = len(active)
        device = self.device

        groups = [list(range(Bnum))]
        if token_budget is not None:
            if token_budget < 2:
                raise ValueError("prefill token budget must be at least 2")
            planned = self._plan_prefill_groups(real_lens, token_budget)
            single = Bnum * max(real_lens)
            grouped = sum(len(g) * max(real_lens[p] for p in g) for g in planned)
            if grouped * _PREFILL_GROUP_MIN_GAIN < single:
                groups = planned
        self.prefill_groups = len(groups)
        self.prefill_padded_rows = sum(
            len(g) * max(real_lens[p] for p in g) for g in groups)
        self.prefill_real_rows = sum(real_lens)

        logits = torch.empty(Bnum, 1, model.lm_head.out_features,
                             dtype=self.dtype, device=device)
        next_tokens = [None] * Bnum
        for group in groups:
            group_logits, group_tokens = self._prefill_group(
                model, group, prompts, real_lens)
            logits[group] = group_logits
            for k, pos in enumerate(group):
                next_tokens[pos] = group_tokens[k]
        # Restore the batch-wide view the single-forward path used to leave
        # behind, so nothing observing these afterwards sees a stale group.
        self._prefill_active = None
        self._prefill_real_lens = real_lens
        self._prefill_lmax = max(real_lens)
        return logits, next_tokens

    def _prefill_group(self, model, group, prompts, real_lens):
        """One padded model forward over ``group`` (row positions in active
        order), including per-request KV write, DCI tree build and finalisation.
        """
        from .adapter.modeling import icecache_state

        active = self.active_indices
        device = self.device
        rows = [active[pos] for pos in group]
        group_lens = [real_lens[pos] for pos in group]
        Lmax = max(group_lens)
        G = len(group)

        # Per-request prefill setup.  This resets each member's KvCache/handlers
        # and arms its DCI index exactly like the serial _icecache_prefill path;
        # b == 0 semantics are preserved because every InferState stays
        # batch_size == 1.
        for pos, i in enumerate(rows):
            state = self.states[i]
            state._prepare_prefill(1, group_lens[pos])
            state._dci_future = None

        # Pad to a dense grid; padding positions are excluded downstream.
        input_ids = torch.zeros(G, Lmax, dtype=torch.long, device=device)
        position_ids = torch.zeros(G, Lmax, dtype=torch.long, device=device)
        for pos, L in enumerate(group_lens):
            input_ids[pos, :L] = prompts[group[pos]]
            position_ids[pos, :L] = torch.arange(L, device=device)

        self._prefill_active = rows
        self._prefill_real_lens = group_lens
        self._prefill_lmax = Lmax
        last_pos = torch.tensor(
            [L - 1 for L in group_lens], dtype=torch.long, device=device)

        # Hold the same lock ``step`` uses: ``lm_head`` is patched for the whole
        # duration of this forward, so a concurrent batched decode step must not
        # interleave (it would observe the un-truncated head).
        if not self._forward_lock.acquire(blocking=False):
            raise RuntimeError("another batch forward is already active")
        self._step_thread = get_ident()
        self.forward_mode = ForwardMode.BATCH_PREFILL
        # The patched LM head truncates to the last position only; project just
        # the last *real* token of each row for the duration of the prefilling
        # forward.  Doing it here rather than restoring the full head keeps the
        # head's GEMM and its output tensor proportional to B instead of
        # B * Lmax (at B=8, Lmax=1136 that is 9088 rows of vocab projection
        # replaced by 8, and a 2.3 GB logits tensor replaced by 2 MB).
        _saved_lm = model.lm_head.forward
        _lm_weight = model.lm_head.weight
        _lm_bias = model.lm_head.bias

        def _last_token_logits(x):
            sel = x.gather(1, last_pos.view(-1, 1, 1).expand(-1, 1, x.shape[-1]))
            return torch.nn.functional.linear(sel, _lm_weight, _lm_bias)

        def _full_logits(x):
            return torch.nn.functional.linear(x, _lm_weight, _lm_bias)

        # ICECACHE_PREFILL_FULL_LOGITS=1 restores the previous behaviour (project
        # the whole padded grid, read the last real token out of the result) so
        # the gather's cost can be measured against it in a paired run.
        _lm_fn = (_full_logits if os.environ.get("ICECACHE_PREFILL_FULL_LOGITS") == "1"
                  else _last_token_logits)
        out_index = 0 if _lm_fn is _last_token_logits else None

        model.lm_head.forward = _lm_fn
        try:
            with icecache_state(model, self):
                out = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_position=torch.arange(Lmax, device=device),
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            model.lm_head.forward = _saved_lm
            self.forward_mode = ForwardMode.DECODE
            self._step_thread = None
            self._forward_lock.release()

        out_logits = out.logits                                    # [G, 1, vocab]
        if out_index is None:
            # Full-logits fallback: keep only each row's last real token so the
            # caller sees the same [G, 1, vocab] layout either way.
            out_logits = out_logits.gather(
                1, last_pos.view(-1, 1, 1).expand(-1, 1, out_logits.shape[-1]))
        # Finalize: await the last DCI eviction and finish each prefill.
        tokens = []
        for pos, i in enumerate(rows):
            state = self.states[i]
            if state._dci_future is not None:
                state._dci_future.result()
                state._dci_future = None
            state._finish_prefill(1, group_lens[pos])
            # A prompt that fits the page budget legitimately stays non-sparse:
            # the serial path sets use_dci = False and builds no tree for it, and
            # the decode path already handles a non-sparse member.  Only require
            # the trees once the request actually entered DCI mode.
            if state.use_dci and any(db is None for db in state.dci_db):
                raise AssertionError(
                    "request %d did not build every DCI tree during batched prefill" % i)
            # keepdim=True -> [1] shape, matching the serial path's tokens[i] used by run_steps
            tokens.append(out_logits[pos, 0].argmax(dim=-1, keepdim=True))
        return out_logits, tokens

    def prefill_attention_forward(self, attn, hidden_states, position_embeddings,
                                 output_attentions=False):
        # Layer attention for :meth:`prefill_batch` -- one ragged FlashInfer
        # prefill over all requests' real tokens.  Mirrors
        # adapter.modeling._icecache_prefill: q/k/v projections run once over the
        # padded [B, Lmax, hidden] grid, then per active request we (a) write only
        # the real tokens into that request's own KvCache (b == 0), (c) async-evict
        # and build that request's DCI tree via the unchanged per-request methods.
        # The attention itself is a single BatchPrefillWithPagedKVCacheWrapper call
        # whose qo_indptr / paged_kv_indptr are the per-request real-length CSR, so
        # padding tokens are never attended to or used as KV.
        from .adapter.modeling import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        # Rows covered by the *current* prefill forward.  With one forward per
        # batch this is ``active_indices``; length-grouped prefill runs several
        # forwards, so each one must attend only its own group's rows.
        active = self._prefill_active
        if active is None:
            raise RuntimeError("prefill_attention_forward called outside prefill_batch")
        if bsz != len(active):
            raise ValueError("batched prefill expects one padded row per prefilling request")
        if q_len != self._prefill_lmax:
            raise ValueError("batched prefill q_len must equal the padded Lmax")
        if output_attentions:
            raise ValueError("batched prefill does not return attention weights")

        layer_idx = attn.layer_idx
        cfg = attn.config
        real_lens = self._prefill_real_lens
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        hd = self.head_dim

        # q/k/v projection (vectorized over all padded rows).
        if getattr(cfg, "pretraining_tp", 1) > 1:
            kvs = (nkv * hd) // cfg.pretraining_tp
            qs = (nh * hd) // cfg.pretraining_tp
            query_states = torch.cat(
                [F.linear(hidden_states, w) for w in attn.q_proj.weight.split(qs, dim=0)], dim=-1)
            key_states = torch.cat(
                [F.linear(hidden_states, w) for w in attn.k_proj.weight.split(kvs, dim=0)], dim=-1)
            value_states = torch.cat(
                [F.linear(hidden_states, w) for w in attn.v_proj.weight.split(kvs, dim=0)], dim=-1)
        else:
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

        if "qwen3" in cfg._name_or_path.lower():
            query_states = attn.q_norm(query_states.view(bsz, q_len, nh, hd)).transpose(1, 2)
            key_states = attn.k_norm(key_states.view(bsz, q_len, nkv, hd)).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, nh, hd).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, nkv, hd).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, nkv, hd)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # Mirror the serial _icecache_prefill layout exactly.  After apply_rotary
        # the tensors are [B, heads, Lmax, hd]; query/key are transposed to the
        # token-major [B, Lmax, heads, hd] that append_paged_kv_cache and
        # prefill_sdpa expect, while value is *not* transposed -- its projection
        # already produced [B, Lmax, nkv, hd].  Transposing value here (as an
        # earlier revision did) yields a strided [B, nkv, Lmax, hd] whose real
        # slice is non-contiguous, and the append kernel rejects it with
        # "v must be contiguous".
        query_states = query_states.transpose(1, 2).contiguous()   # [B, Lmax, n_qo_heads, hd]
        key_states = key_states.transpose(1, 2).contiguous()       # [B, Lmax, n_kv_heads, hd]
        value_states = value_states.contiguous()                   # [B, Lmax, n_kv_heads, hd]

        flat_queries = []
        global_page_indices = []
        kv_indptr_parts = [0]
        qo_indptr_parts = [0]
        last_page_lens = []
        out_slices = []  # (row, start_q, L_i)
        for pos, i in enumerate(active):
            state = self.states[i]
            L_i = int(real_lens[pos])
            # Token-major slices: dim 1 is the token axis, so [pos, :L_i] keeps
            # exactly this request's real tokens and drops the padded tail.
            # (Indexing dim 2 instead would truncate the head axis and slice
            # nothing off the tokens, writing Lmax tokens into an L_i-token cache.)
            q_i = query_states[pos, :L_i]                # [L_i, n_qo_heads, hd]
            k_i = key_states[pos, :L_i].contiguous()     # [L_i, n_kv_heads, hd]
            v_i = value_states[pos, :L_i].contiguous()   # [L_i, n_kv_heads, hd]

            # (a) KV write: only the real tokens, into this request's own cache.
            state.kv_caches[layer_idx].prefill_alloc_n_tokens(L_i, state.alloc_page)
            state.append_paged_kv_cache(layer_idx, k_i[None], v_i[None])

            # (c) DCI eviction / tree build -- async, unchanged per-request path.
            if state._dci_future is not None:
                state._dci_future.result()
                state._dci_future = None
            projected = None
            if state.layer2budget[layer_idx] is not None:
                start = state.n_sink_pages * state.page_size
                end = (state.n_kv_pages - state.n_win_pages) * state.page_size
                kr = k_i.reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]
                projected = torch.matmul(kr, state.proj_vec[:-1]).reshape(-1, 1)
            # query_states is [B, Lmax, n_qo_heads, hd] (token-major), so the last
            # *real* token of this request is dim 1 -- matching the serial
            # path's query_states[:, -1:, ...] on its [B, L, heads, hd] tensor.
            state._dci_future = asyncio.run_coroutine_threadsafe(
                state.prefill_evict_extra_pages_wrapper(
                    layer_idx, query_states[pos:pos + 1, -1:, :, :].contiguous(), projected),
                state._loop)

            # Ragged-attention metadata for this request.
            kvc = state.kv_caches[layer_idx]
            n_pages = int(kvc.n_real_pages)
            global_page_indices.append(kvc.c2p[0, :n_pages].to(self.device).to(torch.int32))
            kv_indptr_parts.append(kv_indptr_parts[-1] + n_pages)
            last_page_lens.append(int(kvc.last_page_len))
            # q_i is already token-major [L_i, n_qo_heads, hd], which is exactly
            # the [total_tokens, n_qo_heads, hd] row order the ragged prefill
            # wrapper wants (it reshapes q to (-1, *q.shape[-2:]) internally).
            flat_queries.append(q_i.contiguous())
            start_q = qo_indptr_parts[-1]
            qo_indptr_parts.append(start_q + L_i)
            out_slices.append((pos, start_q, L_i))

        # One batched ragged FlashInfer prefill over all real tokens.
        flat_q = torch.cat(flat_queries, dim=0)                            # [total, n_qo_heads, hd]
        global_indices = torch.cat(global_page_indices).to(torch.int32).to(self.device)
        qo_indptr = torch.tensor(qo_indptr_parts, dtype=torch.int32, device=self.device)
        kv_indptr = torch.tensor(kv_indptr_parts, dtype=torch.int32, device=self.device)
        last_page_len = torch.tensor(last_page_lens, dtype=torch.int32, device=self.device)
        self._prefill_handler.begin_forward(
            qo_indptr, kv_indptr, last_page_len, self.n_qo_heads, self.n_kv_heads, self.head_dim)
        attn_output_flat = self._prefill_handler.forward(flat_q, self._pool.buffer, global_indices)
        self._prefill_handler.end_forward()

        # Scatter the flattened output back into the padded [B, Lmax, hidden].
        attn_output = attn_output_flat.new_zeros(bsz, q_len, self.n_qo_heads * hd)
        for pos, start_q, L_i in out_slices:
            attn_output[pos, :L_i] = attn_output_flat[start_q:start_q + L_i].reshape(L_i, -1)

        attn_output = attn_output.reshape(bsz, q_len, -1)
        if "llama" in cfg._name_or_path and getattr(cfg, "pretraining_tp", 1) > 1:
            outs = attn_output.split((hd * nh) // cfg.pretraining_tp, dim=2)
            weights = attn.o_proj.weight.split((hd * nh) // cfg.pretraining_tp, dim=1)
            attn_output = sum(F.linear(x, w) for x, w in zip(outs, weights))
        else:
            attn_output = attn.o_proj(attn_output)
        return attn_output, None

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, model, input_ids, position_ids=None, **kwargs):
        """Run one batched decode step through an already enabled model."""
        if not self._forward_lock.acquire(blocking=False):
            raise RuntimeError("another decode step is already active for this batch")
        self._step_thread = get_ident()
        try:
            return self._step_impl(model, input_ids, position_ids, **kwargs)
        finally:
            self._step_thread = None
            self._forward_lock.release()

    def _step_impl(self, model, input_ids, position_ids, **kwargs):
        """Validate the batch and bind it for exactly one model forward."""
        self._ensure_open()
        n_active = self.batch_size
        if n_active == 0:
            raise RuntimeError("batch has no active requests; do not step after retiring every slot")
        if not isinstance(input_ids, torch.Tensor) or tuple(input_ids.shape) != (n_active, 1):
            raise ValueError("input_ids must have shape [batch_size, 1]")
        if input_ids.device != self.device:
            raise ValueError("input_ids must be on the batch device")
        if self._step_active:
            raise RuntimeError("batch step is already active")
        self._validate_step()
        # Per-row absolute positions (variable length: each request sits at its
        # own sequence length, so position_ids is no longer shared across rows).
        expected_position = torch.tensor(
            [self.states[i].seq_len for i in self.active_indices],
            dtype=torch.long, device=self.device)[:, None]
        if position_ids is not None:
            if (not isinstance(position_ids, torch.Tensor) or
                    tuple(position_ids.shape) != (n_active, 1) or
                    not torch.equal(position_ids, expected_position)):
                raise ValueError("position_ids must match each active request sequence length")
        if "cache_position" in kwargs or "position_ids" in kwargs:
            raise ValueError("step manages cache_position and position_ids")
        if kwargs.get("use_cache", False):
            raise ValueError("step manages IceCache KV; use_cache must be False")
        if kwargs.get("past_key_values") is not None or kwargs.get("inputs_embeds") is not None:
            raise ValueError("step manages IceCache KV and takes input_ids, not past_key_values or inputs_embeds")
        if kwargs.get("output_attentions", False):
            raise ValueError("batch attention does not return attention weights")
        # Import lazily to avoid an adapter<->batch import cycle.
        from .adapter.modeling import icecache_state
        with icecache_state(model, self):
            self._step_active = True
            self._step_thread = get_ident()
            self._next_layer = 0
            try:
                # cache_position keeps length 1 even though requests differ in
                # length.  HF's decode path broadcasts it (and its own causal
                # mask is bypassed by IceCache's paged sparse attention, which
                # never reads cache_position); per-row positions travel through
                # position_ids.  We feed the maximum active length so any model
                # internal that does consult it stays within the live window.
                max_seq = int(expected_position.max().item())
                call = dict(kwargs)
                call.update(use_cache=False, position_ids=expected_position,
                            cache_position=torch.tensor([max_seq], dtype=torch.long, device=self.device))
                result = model(input_ids, **call)
                if self._next_layer != self.n_layers:
                    raise RuntimeError("model forward did not visit every IceCache attention layer")
                return result
            except BaseException:
                self._failed = True
                self._cleanup_decode_handlers()
                raise
            finally:
                self._step_active = False
                self._step_thread = None

    # ------------------------------------------------------------------ #
    # Page-ownership validation
    # ------------------------------------------------------------------ #
    def _validate_step(self):
        """Cheap per-step validation: no host reads of device tensors.

        :meth:`validate_ready` walks every page of every layer through ``tolist()``,
        which forces a device synchronisation per (request, layer) -- 256 of them
        per decode step at B=8.  The heavy ownership sweep now runs where the page
        mapping actually changes (construction, :meth:`admit`, or an explicit
        caller), while the per-step path only checks the invariants that are free
        on the host.  Set ``ICECACHE_BATCH_DEEP_VALIDATE=1`` to force the full
        sweep on every step.
        """
        if self._deep_validate:
            return self.validate_ready()
        for request_id in self.active_indices:
            state = self.states[request_id]
            for layer in range(self.n_layers):
                cache = state.kv_caches[layer]
                if cache is None or cache.pool is not self._pool:
                    raise RuntimeError(
                        f"request {request_id} layer {layer} uses another GPU pool")
                if (cache.c2p.ndim != 2 or cache.c2p.shape[0] != 1 or
                        cache.n_real_pages < 1):
                    raise RuntimeError(
                        f"request {request_id} layer {layer} has invalid page mapping")
                if not 1 <= cache.last_page_len <= self.page_size:
                    raise RuntimeError(
                        f"request {request_id} layer {layer} has invalid tail length")
        return True

    def validate_ready(self):
        """Recheck page ownership before a scheduler submits a decode step.

        Only active requests are checked.  The invariant "a physical GPU page
        belongs to exactly one (request, layer)" is preserved via the ``seen``
        map; this holds regardless of how many requests are active or of their
        individual lengths.
        """
        self._ensure_open()
        seen = {}
        for request_id in self.active_indices:
            state = self.states[request_id]
            for layer in range(self.n_layers):
                cache = state.kv_caches[layer]
                if cache.pool is not self._pool:
                    raise RuntimeError(f"request {request_id} layer {layer} uses another GPU pool")
                if cache.seq_len != state.seq_len:
                    raise RuntimeError(f"request {request_id} layer {layer} has inconsistent sequence length")
                if (cache.c2p.ndim != 2 or cache.c2p.shape[0] != 1 or
                        cache.c2p.dtype != torch.int32 or cache.c2p.device != self.device or
                        cache.n_real_pages < 1):
                    raise RuntimeError(f"request {request_id} layer {layer} has invalid page mapping")
                if not 1 <= cache.last_page_len <= self.page_size:
                    raise RuntimeError(f"request {request_id} layer {layer} has invalid tail length")
                entries = state.page_valid_entries[layer]
                if state.layer2budget[layer] is not None:
                    if (entries is None or entries.ndim != 2 or
                            entries.shape[0] < cache.n_real_pages or
                            entries.shape[1] != self.n_kv_heads or
                            entries.dtype != torch.int32 or entries.device != self.device):
                        raise RuntimeError(f"request {request_id} layer {layer} has invalid valid-entry metadata")
                for page in cache.c2p.reshape(-1).tolist():
                    if not 0 <= page < self._pool.n_max_pages:
                        raise RuntimeError(f"invalid physical GPU page {page}")
                    if page in self._pool._free_ids:
                        raise RuntimeError(f"physical GPU page {page} is marked free")
                    if page in seen:
                        raise RuntimeError(
                            f"physical GPU page {page} belongs to both {seen[page]} "
                            f"and request {request_id} layer {layer}")
                    seen[page] = (request_id, layer)
        return True

    # ------------------------------------------------------------------ #
    # CSR metadata assembly
    # ------------------------------------------------------------------ #
    def build_attention_metadata(self, layer_idx):
        """Return CSR page metadata across active requests, in active-slot order."""
        self._ensure_open()
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError("layer_idx out of range")
        active = self.active_indices
        caches = [self.states[i].kv_caches[layer_idx] for i in active]
        counts = [int(c.n_real_pages) for c in caches]
        if any(n < 1 for n in counts):
            raise RuntimeError("each active request needs at least one resident page")
        pages = [c.c2p[0, :n] for c, n in zip(caches, counts)]
        # CSR arrays stay on the device.  ``counts``/``last_page_len`` are plain
        # Python ints (shape metadata and arithmetic on ``seq_len``), so building
        # them costs no device read.
        indptr = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=self.device),
            torch.tensor(np.cumsum(counts), dtype=torch.int32, device=self.device),
        ])
        indices = torch.cat(pages).contiguous()
        # No physical page may belong to two active requests at this layer.  Doing
        # this on the device keeps the check but costs one small reduction per
        # layer instead of a host round trip per request: ``numel()`` is metadata,
        # so nothing here synchronises.
        if indices.numel() != torch.unique(indices).numel():
            raise RuntimeError(f"requests overlap in physical GPU pages at layer {layer_idx}")
        last = torch.tensor([c.last_page_len for c in caches], dtype=torch.int32, device=self.device)
        valid = []
        for i, cache, n in zip(active, caches, counts):
            state = self.states[i]
            sparse = state.use_dci and state.layer2budget[layer_idx] is not None
            entries = (state.page_valid_entries[layer_idx][:n].clone() if sparse else
                       torch.full((n, self.n_kv_heads), self.page_size, dtype=torch.int32, device=self.device))
            if entries.shape != (n, self.n_kv_heads):
                raise RuntimeError(f"request valid-entry metadata has wrong shape at layer {layer_idx}")
            entries[-1, :] = cache.last_page_len
            valid.append(entries)
        valid = torch.cat(valid).reshape(-1).contiguous()
        # One range check for the whole batch instead of one per request: a single
        # ``.item()`` per layer rather than a synchronisation for every request.
        if valid.numel() and bool(((valid < 0) | (valid > self.page_size)).any().item()):
            raise RuntimeError(f"request valid-entry metadata is out of range at layer {layer_idx}")
        return indices, indptr, last, valid

    # ------------------------------------------------------------------ #
    # Per-request DCI query
    # ------------------------------------------------------------------ #
    def _query_one(self, i, layer_idx, q, num_neighbours=None, field_of_view=None,
                   return_host=False):
        state = self.states[i]
        cache = state.kv_caches[layer_idx]
        if (not state.use_dci or cache.budget is None or
                cache.n_real_pages < cache.budget):
            return None
        if state.dci_db[layer_idx] is None:
            raise RuntimeError(f"request {i} layer {layer_idx} has no DCI index")
        # ``num_neighbours`` is validated by batch_query to equal each request's
        # fixed DCI neighbour count, which _DCI_query derives internally, so the
        # serial path honours it implicitly without re-deriving candidates.
        # ``q`` is normally already on the host here: serially callers still pass a
        # CUDA tensor, but batched callers transfer the whole batch's queries once
        # per layer, because every ``.cpu()`` forces a device synchronisation.
        if q.is_cuda:
            q = q.cpu()
        with torch.inference_mode():
            result = state._DCI_query(
                0, layer_idx, q.detach().transpose(0, 1),
                field_of_view_override=field_of_view, return_host=return_host)
        self.query_counts[i] += 1
        self.query_counts_by_layer[i][layer_idx] += 1
        return result

    def batch_query(self, layer_idx, query_states, num_neighbours=None,
                    field_of_view=None):
        """Query every active tree once; reject overlapping calls on this batch state."""
        self._ensure_open()
        if self._query_active:
            raise RuntimeError("another query is already active for this batch")
        self._query_active = True
        try:
            return self._batch_query_impl(layer_idx, query_states,
                                          num_neighbours, field_of_view)
        finally:
            self._query_active = False

    def _batch_query_impl(self, layer_idx, query_states, num_neighbours,
                          field_of_view):
        """Native backend uses one fixed OpenMP team across all active trees."""
        self._ensure_open()
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError("layer_idx out of range")
        if not isinstance(query_states, torch.Tensor):
            raise TypeError("query_states must be a tensor")
        bsz = self.batch_size
        if query_states.shape != (bsz, 1, self.n_qo_heads, self.head_dim):
            raise ValueError("query_states must be [batch,1,num_qo_heads,head_dim]")
        if query_states.device != self.device:
            raise ValueError("query_states must be on the batch device")
        start = perf_counter()
        if num_neighbours is not None and len(num_neighbours) != bsz:
            raise ValueError("num_neighbours must contain one value per active request")
        if field_of_view is not None and len(field_of_view) != bsz:
            raise ValueError("field_of_view must contain one value per active request")
        active = self.active_indices
        pos_of = {i: pos for pos, i in enumerate(active)}
        if num_neighbours is not None:
            for pos, i in enumerate(active):
                state = self.states[i]
                expected = state.n_dci_pages - state.layer2topk[layer_idx]
                if int(num_neighbours[pos]) != expected:
                    raise ValueError(
                        f"request {i} needs {expected} neighbours for its fixed page budget")
        if field_of_view is not None and any(int(value) < 1 for value in field_of_view):
            raise ValueError("field_of_view values must be positive")
        # One device->host transfer for the whole batch instead of one per request
        # inside the loop: every .cpu() synchronises, so the per-request form cost
        # B synchronisations per layer (256 per step at B=8) for a few kilobytes.
        q_cpu = query_states.detach().cpu()
        results = [None] * bsz
        if self.query_backend == "serial":
            t_query = perf_counter()
            for pos, i in enumerate(active):
                nn = None if num_neighbours is None else num_neighbours[pos]
                fv = None if field_of_view is None else field_of_view[pos]
                # query_states is in active-row order.  A slot index is only the
                # same as a row index while the active set is 0..B-1, which stops
                # being true as soon as a non-trailing slot is retired.
                results[pos] = self._query_one(i, layer_idx, q_cpu[pos], nn, fv,
                                               return_host=True)
            self.serial_query_seconds += perf_counter() - t_query
        else:
            q_np = np.ascontiguousarray(
                q_cpu.float().numpy().reshape(bsz, self.n_qo_heads, self.head_dim))
            dci_slots = [i for i in active
                         if self.states[i].use_dci
                         and self.states[i].kv_caches[layer_idx].budget is not None
                         and self.states[i].kv_caches[layer_idx].n_real_pages >= self.states[i].kv_caches[layer_idx].budget]
            capsules = []
            queries = []
            neighbours = []
            fields = []
            dci_positions = []
            for i in dci_slots:
                state = self.states[i]
                db = state.dci_db[layer_idx]
                if db is None:
                    raise RuntimeError(f"request {i} layer {layer_idx} has no DCI index")
                if db._orig_indices is not None:
                    raise NotImplementedError(
                        "native batch query requires DCI without original-index remapping")
                capsules.append(db._dci_inst)
                queries.append(np.ascontiguousarray(q_np[pos_of[i]]))
                neighbours.append(int(num_neighbours[pos_of[i]]) if num_neighbours is not None else
                                 state.n_dci_pages - state.layer2topk[layer_idx])
                fields.append(int(field_of_view[pos_of[i]]) if field_of_view is not None else
                              max(int(state.seq_len * state.search_ratio), 30))
                dci_positions.append(pos_of[i])
            if dci_slots:
                t_search = perf_counter()
                raw = self._native.batch_query(capsules, queries, neighbours,
                                               fields, self.states[active[0]].ratio,
                                               self.query_threads)
                self.native_search_seconds += perf_counter() - t_search
                if len(raw) != len(dci_slots):
                    raise RuntimeError("native batch DCI returned the wrong number of requests")
                t_book = perf_counter()
                for raw_i, i in enumerate(dci_slots):
                    candidates = np.asarray(raw[raw_i])
                    requested = neighbours[raw_i]
                    expected = self.n_qo_heads * 2 * requested
                    if candidates.size != expected:
                        raise RuntimeError(
                            f"request {i} returned {candidates.size} DCI candidates; expected {expected}")
                    if not np.issubdtype(candidates.dtype, np.integer) or np.any(candidates < 0):
                        raise RuntimeError(f"request {i} returned invalid DCI candidate IDs")
                    state = self.states[i]
                    with torch.inference_mode():
                        results[dci_positions[raw_i]] = state._DCI_query(
                            0, layer_idx,
                            q_cpu[pos_of[i]].detach().transpose(0, 1),
                            nn_idx_override=candidates, return_host=True)
                    self.query_counts[i] += 1
                    self.query_counts_by_layer[i][layer_idx] += 1
                self.native_bookkeep_seconds += perf_counter() - t_book
        # Upload every active request's scatter operands once for the whole layer
        # instead of once per request: each upload is a synchronising transfer, so
        # doing it here replaces 2B of them (plus B device reductions) with 2.
        pending = [(pos, res) for pos, res in enumerate(results) if res is not None]
        if pending and isinstance(pending[0][1][0], np.ndarray):
            eids_all = torch.from_numpy(np.stack([res[0] for _, res in pending])).to(self.device)
            nr_all = torch.from_numpy(np.stack([res[2] for _, res in pending])).to(self.device)
            for k, (pos, res) in enumerate(pending):
                results[pos] = (eids_all[k], res[1], nr_all[k])
        self.batch_query_seconds += perf_counter() - start
        return results

    # ------------------------------------------------------------------ #
    # Decode preparation / cleanup
    # ------------------------------------------------------------------ #
    def _prepare_decode(self):
        self._ensure_open()
        if self._step_active and self._next_layer != 0:
            raise RuntimeError("new decode step began before previous layers completed")
        self._step_start = perf_counter()
        # An individual InferState may fail after beginning only some handlers.
        # Mark before the first call so the failed step can release them all.
        self._decode_handlers_armed = True
        for i in self.active_indices:
            # Per-slot _prepare_decode also begins that slot's own decode
            # handler (infer_state._prepare_decode -> decode_handler_tab[b].
            # begin_forward).  That begin_forward is redundant for THIS batch's
            # attention, which uses the shared self._handler instead.  It is kept
            # because the same InferState._prepare_decode/_finish_decode pair is
            # the serial decode path (decode_sdpa reads decode_handler_tab[
            # kvc.budget]); removing it here would break serial decode and
            # unbalance begin/end_forward.  See P0-3 in BATCH_CODE_FLOW.md.
            self.states[i]._prepare_decode(1)

    def _cleanup_decode_handlers(self):
        if not self._decode_handlers_armed:
            return
        for i in self.active_indices:
            for handler in self.states[i].decode_handler_tab.values():
                if getattr(handler, "_paged_kv_indptr", None) is None:
                    continue
                try:
                    handler.end_forward()
                except Exception:
                    # Keep the original model/handler exception visible.
                    pass
        self._decode_handlers_armed = False

    def _recall_prepare(self, i, layer_idx, result):
        """Issue request ``i``'s recall copies (async) and collect its valid entries.

        Returns a prep tuple, or ``None`` when the request has nothing to recall.
        Splitting prepare from commit lets the caller issue every request's copies
        first and then synchronise **once per layer**, instead of once per request
        (which is 256 synchronisations per step at B=8).  All copies go on the
        batch's shared stream, so that one synchronise covers them.
        """
        if result is None:
            return None
        state = self.states[i]
        eids, rids, nr = result
        # Prefer the host page ids that the query itself produced -- using them
        # avoids a device->host round trip per request per layer.
        rids_np = getattr(state, "_last_recall_np", None)
        nr_np = getattr(state, "_last_nr_np", None)
        if rids_np is not None and nr_np is not None:
            state.recall_np(layer_idx, 0, rids_np, nr_np, stream=self._c2g_stream)
            n_recall = int(nr_np.sum())
        else:
            state.recall(layer_idx, 0, rids, nr, stream=self._c2g_stream)
            n_recall = int(nr.sum())
        ns = state.n_sink_pages
        n = state.n_dci_pages - state.layer2topk[layer_idx]
        entries = state.dci_db[layer_idx].get_valid_entries(
            state.selected_page_idx[layer_idx])          # host array, [H, n]
        entries_cpu = np.ascontiguousarray(np.asarray(entries).T)   # [n, H]
        return (i, eids, nr, ns, n, entries_cpu, n_recall)

    def _recall_commit(self, prep, layer_idx, entries_dev):
        """Scatter one prepared request and publish its valid entries."""
        if prep is None:
            return
        i, eids, nr, ns, n, entries_cpu, n_recall = prep
        state = self.states[i]
        if entries_dev is not None:
            state.page_valid_entries[layer_idx][ns:ns + n].copy_(entries_dev)
        if n_recall > 0:
            state.scatter_pages(layer_idx, eids, nr)

    def decode_attention(self, layer_idx, query, key, value):
        """Recall per active request, append per active request, execute one batched attention."""
        self._ensure_open()
        if not self._step_active:
            self.validate_ready()
        active = self.active_indices
        results = self.batch_query(layer_idx, query)
        preps = [self._recall_prepare(i, layer_idx, results[pos])
                 for pos, i in enumerate(active)]
        # One barrier for every request's recall copies (they were all issued on
        # the shared batch stream), instead of one synchronise per request.
        if any(prep is not None for prep in preps):
            self._c2g_stream.synchronize()
        # One host->device transfer for all the validity blocks, instead of one
        # per request.
        blocks = [prep[5] for prep in preps if prep is not None]
        entries_all = (torch.from_numpy(np.concatenate(blocks, axis=0)).to(self.device)
                       if blocks else None)
        offset = 0
        for pos, i in enumerate(active):
            prep = preps[pos]
            dev = None
            if prep is not None:
                nb = prep[5].shape[0]
                dev = entries_all[offset:offset + nb]
                offset += nb
            self._recall_commit(prep, layer_idx, dev)
            state = self.states[i]
            # key/value carry one row per *active* request (row order), whereas
            # ``i`` is a slot index; they coincide only while active == 0..B-1.
            state.append_paged_kv_cache(layer_idx, key[pos:pos+1], value[pos:pos+1])

        # The FlashInfer kernel reads valid entries in this exact CSR page order.
        indices, indptr, last, valid = self.build_attention_metadata(layer_idx)
        any_dci = any(self.states[i].use_dci and self.states[i].layer2budget[layer_idx] is not None
                      for i in active)
        began_forward = False
        try:
            self._handler.begin_forward(indptr, last, self.n_qo_heads, self.n_kv_heads,
                                        self.head_dim, self.page_size, data_type=self.dtype)
            began_forward = True
            output = self._handler.forward(query, self._pool.buffer, indices,
                                           page_valid_entries=valid, dci=any_dci)
        finally:
            if began_forward:
                self._handler.end_forward()
        if layer_idx == 0:
            for i in active:
                state = self.states[i]
                if state.offload_win_flag[-1]:
                    for l in range(state.n_layers):
                        state.default_stream.wait_stream(state.decode_backup_stream)
                        state.offload_win_page_to_DCI(l)
                        state.offload_win_flag[l] = False
        return output

    def _finish_decode(self):
        for i in self.active_indices:
            self.states[i]._finish_decode(1)
        self._decode_handlers_armed = False
        self.decode_steps += 1
        self.batch_step_seconds += perf_counter() - self._step_start

    def attention_forward(self, attn, hidden_states, position_embeddings,
                          output_attentions=False):
        from .adapter.modeling import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        if bsz != self.batch_size or q_len != 1:
            raise ValueError("BatchInferState supports only [batch,1,hidden] decode")
        if output_attentions:
            raise ValueError("batch attention does not return attention weights")
        layer_idx = attn.layer_idx
        if self._step_active and layer_idx != self._next_layer:
            raise RuntimeError(f"expected attention layer {self._next_layer}, got {layer_idx}")
        if layer_idx == 0:
            self._prepare_decode()
        cfg = attn.config
        if getattr(cfg, "pretraining_tp", 1) > 1:
            tp = cfg.pretraining_tp
            qs = attn.q_proj.weight.split(cfg.num_attention_heads * attn.head_dim // tp, dim=0)
            ks = attn.k_proj.weight.split(cfg.num_key_value_heads * attn.head_dim // tp, dim=0)
            vs = attn.v_proj.weight.split(cfg.num_key_value_heads * attn.head_dim // tp, dim=0)
            query = torch.cat([F.linear(hidden_states, w) for w in qs], dim=-1)
            key = torch.cat([F.linear(hidden_states, w) for w in ks], dim=-1)
            value = torch.cat([F.linear(hidden_states, w) for w in vs], dim=-1)
        else:
            query = attn.q_proj(hidden_states)
            key = attn.k_proj(hidden_states)
            value = attn.v_proj(hidden_states)
        if "qwen3" in cfg._name_or_path.lower():
            query = attn.q_norm(query.view(bsz, q_len, cfg.num_attention_heads, attn.head_dim)).transpose(1, 2)
            key = attn.k_norm(key.view(bsz, q_len, cfg.num_key_value_heads, attn.head_dim)).transpose(1, 2)
        else:
            query = query.view(bsz, q_len, cfg.num_attention_heads, attn.head_dim).transpose(1, 2)
            key = key.view(bsz, q_len, cfg.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, cfg.num_key_value_heads, attn.head_dim)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query, key = query.transpose(1, 2).contiguous(), key.transpose(1, 2).contiguous()
        output = self.decode_attention(layer_idx, query, key, value).reshape(bsz, q_len, -1)
        if "llama" in cfg._name_or_path and getattr(cfg, "pretraining_tp", 1) > 1:
            tp = cfg.pretraining_tp
            outs = output.split(cfg.num_attention_heads * attn.head_dim // tp, dim=2)
            weights = attn.o_proj.weight.split(cfg.num_attention_heads * attn.head_dim // tp, dim=1)
            output = sum(F.linear(x, w) for x, w in zip(outs, weights))
        else:
            output = attn.o_proj(output)
        if layer_idx == self.n_layers - 1:
            self._finish_decode()
        if self._step_active:
            self._next_layer += 1
        return output, None

    # ------------------------------------------------------------------ #
    # Request exit / reuse (explicit; not a scheduler)
    # ------------------------------------------------------------------ #
    def retire(self, index):
        """Release slot ``index``, free its shared GPU pages, and return the state.

        Rejects if the batch is closed/failed, a step or query is in flight, or
        the slot is already free.

        Only the *shared* resource can be handed back here: the request's pages in
        the batch's GPU ``KvPool`` are deduplicated and returned to it, and a page
        already marked free raises instead of allowing a double free.  Everything
        else on an ``InferState`` is private to that request -- its CPU ``KvPool``
        is built inside ``InferState.__init__`` (``infer_state.py``), the DCI index
        has no destroy entry point, and the transit buffers belong to it as well --
        so those are freed only when the last reference dies.  This method drops
        the batch's own references and **returns the state**; callers wanting
        deterministic reclamation should drop the returned object (the probe does).
        It also calls the request's ``InferState.shutdown()``, stopping the
        background asyncio loop and worker executor that the async offload path
        would otherwise leak across admit/retire cycles.
        """
        if self._closed:
            raise RuntimeError("cannot retire from a closed batch")
        if self._failed:
            raise RuntimeError("batch forward failed; create fresh request states")
        if self._step_active:
            raise RuntimeError("cannot retire a slot while a decode step is active")
        if self._forward_lock.locked():
            raise RuntimeError("cannot retire a slot while a batch forward is in flight")
        if self._query_active:
            raise RuntimeError("cannot retire a slot while a batch query is active")
        if not 0 <= index < self.capacity:
            raise IndexError("retire index out of range")
        state = self.states[index]
        if state is None:
            raise RuntimeError(f"slot {index} is already free")
        # The retired request's KV writes may still be queued (append runs on the
        # default stream, window backup on ``decode_backup_stream``).  A page
        # handed back to the pool must not still be the target of an in-flight
        # write, so drain the device before freeing.  retire is rare, which makes
        # a full synchronize the cheapest correct barrier here.
        torch.cuda.synchronize()
        # Stop the request's background asyncio loop and its worker executor
        # before anything is torn down: nothing else ever does, so leaving it
        # running would accumulate one thread plus one executor per retire/admit
        # cycle.  Doing it first also means a refusal (still-pending offload
        # future) leaves this slot completely intact.
        shutdown = getattr(state, "shutdown", None)
        if shutdown is not None:
            shutdown()
        freed = set()
        for layer in range(self.n_layers):
            cache = state.kv_caches[layer]
            if cache is None or cache.pool is not self._pool:
                continue
            for page in cache.c2p.reshape(-1).tolist():
                if page < 0:
                    continue
                if page in freed:
                    continue
                freed.add(page)
                if page in self._pool._free_ids:
                    raise RuntimeError(
                        f"physical GPU page {page} is already free; double free prevented")
                self._pool.free_page(page)
        # Drop the batch's references to the request's private resources, then
        # clear the slot.  The caller still holds the state object, which is why
        # it is returned -- dropping that reference is what actually frees the
        # CPU KV pool and the DCI trees.
        self._release_request_resources(state)
        self.states[index] = None
        self.query_counts[index] = 0
        self.query_counts_by_layer[index] = [0] * self.n_layers
        return state

    def _release_request_resources(self, state):
        """Drop the batch's references to a retired request's private resources.

        Every heavy allocation on an ``InferState`` is per-request, so there is no
        shared pool to return them to: ``_cpu_pool`` is constructed inside
        ``InferState.__init__`` (``infer_state.py``), the ``cpu_kv_caches`` and the
        offload/transit buffers hang off that pool, and the DCI index (``dci_db``)
        exposes no destroy call.  Clearing these fields means the memory -- notably
        the pinned CPU KV pool, hundreds of MB at the default page count -- is
        released as soon as the caller drops the returned state.  The retired state
        is not usable afterwards, which is the documented contract.
        """
        def _drop(attr):
            if getattr(state, attr, None) is not None:
                setattr(state, attr, None)

        # CPU caches first (they reference the pool), then the pool itself.
        for attr in ("cpu_kv_caches", "temp_cpu_kv_caches", "cpu_neighbour_caches",
                     "offload_win_caches"):
            if getattr(state, attr, None) is not None:
                setattr(state, attr, [None] * self.n_layers)
        for attr in ("_page_log", "cpu_transit_buffer", "cuda_transit_buffer",
                     "cuda_cast_buffer", "_src_address_buffer", "dci_db", "_cpu_pool",
                     # Holds raw pointers into _cpu_pool; drop it so the retired
                     # state carries no dangling addresses.
                     "page_address_buffer"):
            _drop(attr)

    def admit(self, index, state):
        """Place a freshly prefilled ``state`` into the free slot ``index``.

        Only a free slot may be written.  The incoming request must share the
        batch's GPU pool, use a disjoint set of GPU pages from every active
        request, and match the batch's fixed configuration.  This is an explicit
        call by the caller -- there is no scheduling loop, so this is *not*
        continuous batching.
        """
        if self._closed:
            raise RuntimeError("cannot admit into a closed batch")
        if self._failed:
            raise RuntimeError("batch forward failed; create fresh request states")
        if self._step_active:
            raise RuntimeError("cannot admit a slot while a decode step is active")
        if self._forward_lock.locked():
            raise RuntimeError("cannot admit a slot while a batch forward is in flight")
        if self._query_active:
            raise RuntimeError("cannot admit a slot while a batch query is active")
        if not 0 <= index < self.capacity:
            raise IndexError("admit index out of range")
        if self.states[index] is not None:
            raise RuntimeError(f"slot {index} is not free; retire it first")
        self._check_prefilled(state, index)
        self._check_compatible(state, index)
        incoming = set()
        for layer in range(self.n_layers):
            cache = state.kv_caches[layer]
            if cache is None:
                continue
            for page in cache.c2p.reshape(-1).tolist():
                if page >= 0:
                    incoming.add(page)
        for i in self.active_indices:
            for layer in range(self.n_layers):
                cache = self.states[i].kv_caches[layer]
                if cache is None:
                    continue
                for page in cache.c2p.reshape(-1).tolist():
                    if page >= 0 and page in incoming:
                        raise RuntimeError(
                            f"incoming request at slot {index} overlaps GPU page {page} "
                            f"with active slot {i}")
        self.states[index] = state
        self.query_counts[index] = 0
        self.query_counts_by_layer[index] = [0] * self.n_layers
        # The active set just grew; make sure the shared workspace covers it.
        self._ensure_workspace(self.batch_size)
