"""Fixed-size decode batch over independently prefilled IceCache requests."""

import os
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from . import kernels
from .infer_state import ForwardMode


class BatchInferState:
    """Synchronous B=2 decode using one model forward and one GPU attention call.

    Each member owns its DCI, CPU cache, temporary buffers and decode state.
    Both members must have been prefilled separately into the same GPU KvPool.
    """

    def __init__(self, states, query_backend="serial", query_threads=32):
        if len(states) != 2:
            raise ValueError("the fixed-batch prototype requires exactly two requests")
        a, b = states
        if a._pool is not b._pool:
            raise ValueError("requests must share one GPU KV pool from prefill")
        if a.batch_size != 1 or b.batch_size != 1:
            raise ValueError("each request must have been prefilled with batch_size=1")
        if a.seq_len != b.seq_len:
            raise ValueError("prototype requires equal sequence lengths; HF cache_position is shared")
        for attr in ("n_layers", "n_qo_heads", "n_kv_heads", "head_dim", "page_size", "dtype", "device", "layout"):
            if getattr(a, attr) != getattr(b, attr):
                raise ValueError(f"request configurations disagree on {attr}")
        if a.layer2budget != b.layer2budget:
            raise ValueError("both requests must use the same layer budgets")
        if a.n_prefetch_layers or b.n_prefetch_layers:
            raise ValueError("batch prototype does not support cross-layer prefetch")
        if a.n_reuse_layers or b.n_reuse_layers:
            raise ValueError("batch prototype does not support cross-layer reuse")
        for layer in range(a.n_layers):
            pages_a = set(a.kv_caches[layer].c2p.reshape(-1).tolist())
            pages_b = set(b.kv_caches[layer].c2p.reshape(-1).tolist())
            if pages_a & pages_b:
                raise ValueError(f"requests overlap in physical GPU pages at layer {layer}")
        self.states = tuple(states)
        self.n_layers = a.n_layers
        self.n_qo_heads = a.n_qo_heads
        self.n_kv_heads = a.n_kv_heads
        self.head_dim = a.head_dim
        self.page_size = a.page_size
        self.dtype = a.dtype
        self.device = a.device
        self.layout = a.layout
        self.forward_mode = ForwardMode.DECODE
        self._pool = a._pool
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
        self.native_query_counts = [0, 0]
        self.batch_query_seconds = 0.0
        self.batch_step_seconds = 0.0
        self.decode_steps = 0
        self._closed = False

    @property
    def batch_size(self):
        return 2

    @property
    def seq_lens(self):
        return tuple(s.seq_len for s in self.states)

    def close(self):
        """Stop use of this batch; request caches and the shared pool stay caller-owned."""
        if not self._closed:
            self._closed = True

    def _query_one(self, i, layer_idx, q):
        state = self.states[i]
        cache = state.kv_caches[layer_idx]
        if (not state.use_dci or cache.budget is None or
                cache.n_real_pages < cache.budget):
            return None
        if state.dci_db[layer_idx] is None:
            raise RuntimeError(f"request {i} layer {layer_idx} has no DCI index")
        # _DCI_query is the actual semantic tree search.  The original decode
        # path on this branch calls retrieve_blocks(), which bypasses DCI.
        # Inference mode is thread-local.  Prefill may have created inference
        # tensors, and _DCI_query updates per-request maps in place.
        with torch.inference_mode():
            result = state._DCI_query(0, layer_idx, q.cpu().detach().transpose(0, 1))
        self.native_query_counts[i] += 1
        return result

    def batch_query(self, layer_idx, query_states):
        """Query two request trees; native backend uses one fixed OpenMP team."""
        if query_states.shape != (2, 1, self.n_qo_heads, self.head_dim):
            raise ValueError("query_states must be [2,1,num_qo_heads,head_dim]")
        start = perf_counter()
        if self.query_backend == "serial":
            results = [self._query_one(i, layer_idx, query_states[i])
                       for i in range(2)]
        else:
            active = [i for i, state in enumerate(self.states)
                      if state.use_dci and state.kv_caches[layer_idx].budget is not None
                      and state.kv_caches[layer_idx].n_real_pages >= state.kv_caches[layer_idx].budget]
            results = [None, None]
            if active:
                capsules = []
                queries = []
                neighbours = []
                fields = []
                for i in active:
                    state = self.states[i]
                    db = state.dci_db[layer_idx]
                    if db is None:
                        raise RuntimeError(f"request {i} layer {layer_idx} has no DCI index")
                    if db._orig_indices is not None:
                        raise NotImplementedError(
                            "native batch query requires DCI without original-index remapping")
                    capsules.append(db._dci_inst)
                    queries.append(np.ascontiguousarray(
                        query_states[i].reshape(-1, self.head_dim).float().cpu().numpy()))
                    neighbours.append(state.n_dci_pages - state.layer2topk[layer_idx])
                    fields.append(max(int(state.seq_len * state.search_ratio), 30))
                raw = self._native.batch_query(capsules, queries, neighbours,
                                               fields, self.states[0].ratio,
                                               self.query_threads)
                for i, candidates in zip(active, raw):
                    state = self.states[i]
                    with torch.inference_mode():
                        results[i] = state._DCI_query(
                            0, layer_idx,
                            query_states[i].cpu().detach().transpose(0, 1),
                            nn_idx_override=candidates)
                    self.native_query_counts[i] += 1
        self.batch_query_seconds += perf_counter() - start
        return results

    def _prepare_decode(self):
        if self._closed:
            raise RuntimeError("batch state is closed")
        self._step_start = perf_counter()
        for state in self.states:
            state._prepare_decode(1)

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
        """Recall per request, append per request, execute one B=2 attention."""
        results = self.batch_query(layer_idx, query)
        for i, state in enumerate(self.states):
            self._recall_one(i, layer_idx, results[i])
            state.append_paged_kv_cache(layer_idx, key[i:i+1], value[i:i+1])

        # CSR page list: [A pages, B pages].  The custom FlashInfer kernel
        # indexes valid_entries as (global_page_idx * Hkv + head_idx), so
        # validity must be packed in exactly the same order.
        caches = [s.kv_caches[layer_idx] for s in self.states]
        counts = [c.n_real_pages for c in caches]
        indptr = torch.tensor([0, counts[0], sum(counts)], dtype=torch.int32,
                              device=self.device)
        indices = torch.cat([c.c2p[0, :n] for c, n in zip(caches, counts)]).contiguous()
        last = torch.tensor([c.last_page_len for c in caches], dtype=torch.int32,
                            device=self.device)
        validity = []
        any_dci = False
        for state, cache, n in zip(self.states, caches, counts):
            sparse = state.use_dci and state.layer2budget[layer_idx] is not None
            any_dci |= sparse
            if sparse:
                entries = state.page_valid_entries[layer_idx][:n].clone()
            else:
                entries = torch.full((n, self.n_kv_heads), self.page_size,
                                     dtype=torch.int32, device=self.device)
            entries[-1, :] = cache.last_page_len
            validity.append(entries)
        valid = torch.cat(validity, dim=0).reshape(-1).contiguous()
        self._handler.begin_forward(indptr, last, self.n_qo_heads, self.n_kv_heads,
                                    self.head_dim, self.page_size, data_type=self.dtype)
        try:
            output = self._handler.forward(query, self._pool.buffer, indices,
                                           page_valid_entries=valid, dci=any_dci)
        finally:
            self._handler.end_forward()
        if layer_idx == 0:
            for state in self.states:
                if state.offload_win_flag[-1]:
                    for l in range(state.n_layers):
                        state.default_stream.wait_stream(state.decode_backup_stream)
                        state.offload_win_page_to_DCI(l)
                        state.offload_win_flag[l] = False
        return output

    def _finish_decode(self):
        for state in self.states:
            state._finish_decode(1)
        self.decode_steps += 1
        self.batch_step_seconds += perf_counter() - self._step_start

    def attention_forward(self, attn, hidden_states, position_embeddings,
                          output_attentions=False):
        from .adapter.modeling import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        if bsz != 2 or q_len != 1:
            raise ValueError("BatchInferState supports only [2,1,hidden] decode")
        layer_idx = attn.layer_idx
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
        return output, None
