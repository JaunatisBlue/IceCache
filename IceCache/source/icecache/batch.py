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
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter

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


class BatchInferState:
    """Synchronous decode over a fixed-capacity set of independently prefilled requests.

    Each member owns its DCI, CPU cache, temporary buffers and decode state.
    All members must have been prefilled separately into the same GPU KvPool.
    The batch size is the number of *active* slots and is not fixed at 2: callers
    may :meth:`retire` a finished slot and later :meth:`admit` a freshly
    prefilled request into it.
    """

    def __init__(self, states, query_backend="serial", query_threads=32):
        if len(states) < 1:
            raise ValueError("the batch prototype needs at least one prefilled request")
        self.capacity = len(states)
        self.states = list(states)
        self._ref_state = states[0]
        self._pool = states[0]._pool
        for i, state in enumerate(self.states):
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
        self._workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self._handler = kernels.BatchDecodeWithPagedKVCacheWrapper(self._workspace, self.layout)
        # Per-slot containers indexed by slot index; length stays at capacity so
        # callers can read a slot by index.  Inactive slots keep their last value
        # but are never incremented (and are reset on admit).
        self.query_counts = [0] * self.capacity
        self.query_counts_by_layer = [[0] * self.n_layers for _ in range(self.capacity)]
        # Kept for consumers of the original probe, regardless of backend.
        self.native_query_counts = self.query_counts
        self.batch_query_seconds = 0.0
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
        self.validate_ready()
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
        # No physical page may belong to two active requests at this layer.
        seen = set()
        for p in pages:
            for page in p.tolist():
                if page in seen:
                    raise RuntimeError(f"requests overlap in physical GPU pages at layer {layer_idx}")
                seen.add(page)
        indptr = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=self.device),
            torch.tensor(np.cumsum(counts), dtype=torch.int32, device=self.device),
        ])
        indices = torch.cat(pages).contiguous()
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
            if torch.any((entries < 0) | (entries > self.page_size)):
                raise RuntimeError(f"request valid-entry metadata is out of range at layer {layer_idx}")
            valid.append(entries)
        return indices, indptr, last, torch.cat(valid).reshape(-1).contiguous()

    # ------------------------------------------------------------------ #
    # Per-request DCI query
    # ------------------------------------------------------------------ #
    def _query_one(self, i, layer_idx, q, num_neighbours=None, field_of_view=None):
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
        with torch.inference_mode():
            result = state._DCI_query(
                0, layer_idx, q.cpu().detach().transpose(0, 1),
                field_of_view_override=field_of_view)
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
        results = [None] * bsz
        if self.query_backend == "serial":
            for pos, i in enumerate(active):
                nn = None if num_neighbours is None else num_neighbours[pos]
                fv = None if field_of_view is None else field_of_view[pos]
                # query_states is in active-row order.  A slot index is only the
                # same as a row index while the active set is 0..B-1, which stops
                # being true as soon as a non-trailing slot is retired.
                results[pos] = self._query_one(i, layer_idx, query_states[pos], nn, fv)
        else:
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
                queries.append(np.ascontiguousarray(
                    query_states[pos_of[i]].reshape(-1, self.head_dim).float().cpu().numpy()))
                neighbours.append(int(num_neighbours[pos_of[i]]) if num_neighbours is not None else
                                 state.n_dci_pages - state.layer2topk[layer_idx])
                fields.append(int(field_of_view[pos_of[i]]) if field_of_view is not None else
                              max(int(state.seq_len * state.search_ratio), 30))
                dci_positions.append(pos_of[i])
            if dci_slots:
                raw = self._native.batch_query(capsules, queries, neighbours,
                                               fields, self.states[active[0]].ratio,
                                               self.query_threads)
                if len(raw) != len(dci_slots):
                    raise RuntimeError("native batch DCI returned the wrong number of requests")
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
                            query_states[pos_of[i]].cpu().detach().transpose(0, 1),
                            nn_idx_override=candidates)
                    self.query_counts[i] += 1
                    self.query_counts_by_layer[i][layer_idx] += 1
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

    def _recall_one(self, i, layer_idx, result):
        if result is None:
            return
        state = self.states[i]
        eids, rids, nr = result
        state.recall(layer_idx, 0, rids, nr)
        ns = state.n_sink_pages
        n = state.n_dci_pages - state.layer2topk[layer_idx]
        state.page_valid_entries[layer_idx][ns:ns+n] = torch.tensor(
            state.dci_db[layer_idx].get_valid_entries(
                state.selected_page_idx[layer_idx]), dtype=torch.int32,
            device=state.device).T
        state.c2g_stream.synchronize()
        if int(nr.sum()) > 0:
            state.scatter_pages(layer_idx, eids, nr)

    def decode_attention(self, layer_idx, query, key, value):
        """Recall per active request, append per active request, execute one batched attention."""
        self._ensure_open()
        if not self._step_active:
            self.validate_ready()
        active = self.active_indices
        results = self.batch_query(layer_idx, query)
        for pos, i in enumerate(active):
            self._recall_one(i, layer_idx, results[pos])
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
        """Release slot ``index`` and free the request's GPU pages.

        Rejects if the batch is closed/failed, a step or query is in flight, or
        the slot is already free.  GPU pages owned by the request (across all
        layers) are deduplicated and returned to the shared pool; a page already
        marked free raises instead of allowing a double free.  CPU KvCache and
        DCI index are owned by the caller and are reclaimed when the caller
        releases the request object -- we only drop this batch's reference here.
        """
        if self._closed:
            raise RuntimeError("cannot retire from a closed batch")
        if self._failed:
            raise RuntimeError("batch forward failed; create fresh request states")
        if self._step_active:
            raise RuntimeError("cannot retire a slot while a decode step is active")
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
        # Drop the reference; the request's CPU/DCI resources become unreferenced
        # and are reclaimed by the caller/GC.  We do not deep-copy or actively
        # free them here.
        self.states[index] = None
        self.query_counts[index] = 0
        self.query_counts_by_layer[index] = [0] * self.n_layers

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
