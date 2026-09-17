import warnings
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict
from functools import wraps
import asyncio
from contextlib import contextmanager


import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache

# use these classes just for hint
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    LlamaModel,
    repeat_kv,
)

from icecache.infer_state import InferState, ForwardMode
from icecache import kernels
from time import time
import numpy as np

global tt

class NotTestedError(NotImplementedError):
    """Exception raised when a specific feature is not tested."""
    pass

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    if k is None:
        return q_embed, None
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _icecache_prefill(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    if cur_id == 0:
        global tt
        tt = time()
        state._prepare_prefill(bsz, q_len)

    if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.config.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.config.num_attention_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    if "qwen3" in self.config._name_or_path.lower():
        query_states = self.q_norm(query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)).transpose(1, 2)
    else:
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    kvc = state.kv_caches[cur_id]

    cos, sin = position_embeddings

    # Q: should be updated in any cases
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()

    # TODO: key embedding projection (overlap with append_paged_kv_cache)
    projected = None
    # if q_len > 1:  # prefill
    #     assert do_send_pf == do_recv_pf == do_reuse_pf == False
    if state.layer2budget[cur_id] is not None:
        start = state.n_sink_pages * state.page_size
        end = (state.n_kv_pages - state.n_win_pages) * state.page_size
        key = key_states.reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]
        projected = torch.matmul(key, state.proj_vec[:-1]).reshape(-1, 1)

    state.attn_layers[cur_id] = self
    kvc.prefill_alloc_n_tokens(q_len, state.alloc_page)

    state.append_paged_kv_cache(cur_id, key_states, value_states)

    if state._dci_future is not None:
        _ = state._dci_future.result() # (timeout=1)
        state._dci_future = None

    # assert state._dci_future is None
    state._dci_future = asyncio.run_coroutine_threadsafe(
        state.prefill_evict_extra_pages_wrapper(
            cur_id, query_states[:, -1:, ...].contiguous(), projected), 
        state._loop)

    attn_output = state.prefill_sdpa(cur_id, query_states)

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if 'llama' in self.config._name_or_path and self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    if cur_id == n_layers - 1:
        # state.end_forward(bsz, q_len)

        if state._dci_future is not None:
            _ = state._dci_future.result() # (timeout=1)
            state._dci_future = None

        state._finish_prefill(bsz, q_len)

    return attn_output, attn_weights


def _icecache_decode(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    if cur_id == 0:
        global tt
        tt = time()
        # state.begin_forward(bsz, q_len)
        state._prepare_decode(bsz)

    if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.config.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.config.num_attention_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    if "qwen3" in self.config._name_or_path.lower():
        query_states = self.q_norm(query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)).transpose(1, 2)
    else:
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    kvc = state.kv_caches[cur_id]
    budget = state.layer2budget[cur_id]

    n_pf_layers = state.n_prefetch_layers
    do_send_pf = do_recv_pf = do_reuse = False
    if q_len == 1 and n_pf_layers > 0 and cur_id >= 2:
        pf_dst_id = cur_id + n_pf_layers
        if pf_dst_id < n_layers:
            dst_budget = state.layer2budget[pf_dst_id]
            if dst_budget is not None and dst_budget < state.n_pages:
                do_send_pf = True
        pf_src_id = cur_id - n_pf_layers
        if pf_src_id >= 2 and budget is not None and budget < state.n_pages:
            do_recv_pf = True
    
    cos, sin = position_embeddings

    # Q: should be updated in any cases
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()

    if do_recv_pf:
        if budget is not None and kvc.n_pages > budget:
            assert state._dci_future is not None
            eids, nr = state._dci_future.result() # (timeout=1)
            state._dci_future = None

            state.scatter_pages(cur_id, eids, nr)
        else:
            raise NotImplemented("kv cache is expected in the receive state")
    else:
        if budget is not None and kvc.n_pages > budget:
            eids, nr = state.retrieve_blocks(cur_id)
            # scatter_pages walks the full eids width regardless of nr, so a
            # no-change retrieval (nr == 0) must skip it -- otherwise it would
            # re-scatter stale cast-buffer content into the slots.
            if int(nr.sum()) > 0:
                state.scatter_pages(cur_id, eids, nr)
        # full-cache (n_pages <= budget): everything is resident, nothing to fetch

    if do_send_pf:
        query_states1 = (
            state.attn_layers[pf_dst_id]
            .q_proj(hidden_states)
            .view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
        )
        if "qwen3" in self.config._name_or_path.lower():
            query_states1 = state.attn_layers[pf_dst_id].q_norm(query_states1)
        query_states1, _ = apply_rotary_pos_emb(query_states1, None, cos, sin)
        query_states1 = query_states1.transpose(1, 2).contiguous()

        assert state._dci_future is None
        state._dci_future = asyncio.run_coroutine_threadsafe(
            state.estimate_select_recall_wrapper(pf_dst_id, query_states1), state._loop)

    state.append_paged_kv_cache(cur_id, key_states, value_states)

    attn_page_ids = kvc.c2p
    attn_output = state.decode_sdpa(cur_id, query_states, attn_page_ids)

    if cur_id == 0 and state.offload_win_flag[-1]:
        for l in range(state.n_layers):
            state.default_stream.wait_stream(state.decode_backup_stream)
            state.offload_win_page_to_DCI(l)
            state.offload_win_flag[l] = False

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if 'llama' in self.config._name_or_path and self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    if cur_id == n_layers - 1:
        state.end_forward(bsz, q_len)

    return attn_output, attn_weights


def _icecache_continuation(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Multi-token continuation chunk over an existing, fully resident KV.

    Phase B1: full-cache only.  The chunk is a plain causal paged prefill over
    the whole contiguous KV, so there is no DCI work, no offload and no
    split-attention merge here.  Absolute RoPE positions come from the caller's
    ``position_ids``; this function never resets sequence state.
    """
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    if cur_id == 0:
        if state.use_dci:
            state._prepare_continuation_sparse(bsz, q_len)
        else:
            state._prepare_continuation(bsz, q_len)

    if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.config.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.config.num_attention_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = torch.cat(
            [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)],
            dim=-1,
        )
        key_states = torch.cat(
            [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)],
            dim=-1,
        )
        value_states = torch.cat(
            [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)],
            dim=-1,
        )
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    if "qwen3" in self.config._name_or_path.lower():
        query_states = self.q_norm(query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)).transpose(1, 2)
    else:
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    kvc = state.kv_caches[cur_id]

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()

    # Same order as prefill: the chunk's K/V is written first so that the paged
    # prefill below sees it as the tail of the KV it attends over.
    if state.use_dci:
        # Sparse: refresh the resident semantic pages with BLOCK retrieval
        # (policy b -- recency over the CPU page log).  The selection is
        # query-independent, so the chunk and every decode step share the same
        # resident set.  Then pack the resident per KV head and run one causal
        # prefill with the KV head in the batch dimension.  The chunk's K/V is
        # written into the paged cache after all layers.
        eids, nr = state.retrieve_blocks(cur_id)
        if int(nr.sum()) > 0:
            state.scatter_pages(cur_id, eids, nr)

        stage, indices, indptr, last_len = state._pack_resident(cur_id, q_len)
        state._cont_pack[cur_id] = (stage, indices, indptr, last_len)
        counts = state._resident_valid_counts(cur_id)
        per_head = counts.sum(0)
        ps = state.page_size
        totals = per_head + q_len
        pages = (totals + ps - 1) // ps
        state._cont_per_head[cur_id] = per_head
        state._cont_run_start[cur_id] = torch.cumsum(pages * ps, 0) - pages * ps

        state._write_chunk_into_stage(cur_id, key_states, value_states)
        attn_output = state.continuation_sdpa_batched(
            cur_id, query_states, stage, indices, indptr, last_len
        )
        state._cont_kv[cur_id] = (key_states, value_states)
    else:
        state.append_paged_kv_cache(cur_id, key_states, value_states)
        attn_output = state.prefill_sdpa(cur_id, query_states)
    attn_output = attn_output.reshape(bsz, q_len, -1)

    if 'llama' in self.config._name_or_path and self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
        )
    else:
        attn_output = self.o_proj(attn_output)

    attn_weights = None

    if cur_id == n_layers - 1:
        if state.use_dci:
            state._finish_continuation_sparse(bsz, q_len)
        else:
            state._finish_continuation(bsz, q_len)

    return attn_output, attn_weights


def _icecache_attn_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    from icecache.batch import BatchInferState
    if isinstance(infer_state, BatchInferState):
        return infer_state.attention_forward(
            self, hidden_states, position_embeddings, output_attentions)
    _, q_len, _ = hidden_states.size()
    mode = getattr(infer_state, "forward_mode", None)

    if mode is ForwardMode.CONTINUATION_PREFILL:
        # Never inferred from q_len: a continuation chunk and a fresh prompt are
        # the same shape, so the caller has to declare which one this is.
        if q_len <= 1:
            raise ValueError(
                f"forward_mode=CONTINUATION_PREFILL requires q_len > 1, got {q_len}; "
                "use ForwardMode.DECODE for single tokens"
            )
        return _icecache_continuation(
            self, hidden_states, attention_mask, position_embeddings,
            past_key_value, output_attentions, use_cache, infer_state, debug, **kwargs,
        )
    if mode is ForwardMode.INITIAL_PREFILL:
        if q_len <= 1:
            raise ValueError(
                f"forward_mode=INITIAL_PREFILL requires q_len > 1, got {q_len}"
            )
        return _icecache_prefill(
            self, hidden_states, attention_mask, position_embeddings,
            past_key_value, output_attentions, use_cache, infer_state, debug, **kwargs,
        )
    if mode is ForwardMode.DECODE:
        if q_len != 1:
            raise ValueError(
                f"forward_mode=DECODE requires q_len == 1, got {q_len}; "
                "use ForwardMode.CONTINUATION_PREFILL for a multi-token chunk"
            )
        return _icecache_decode(
            self, hidden_states, attention_mask, position_embeddings,
            past_key_value, output_attentions, use_cache, infer_state, debug, **kwargs,
        )

    # forward_mode is None: legacy q_len dispatch.  Retained for HF generate(),
    # the benchmark scripts and the phase A oracle; the session classes always
    # set an explicit mode.
    if q_len > 1:
        return _icecache_prefill(
            self, hidden_states, attention_mask, position_embeddings,
            past_key_value, output_attentions, use_cache, infer_state, debug, **kwargs,
        )
    return _icecache_decode(
        self, hidden_states, attention_mask, position_embeddings,
        past_key_value, output_attentions, use_cache, infer_state, debug, **kwargs,
    )


def enable_icecache(
    self: LlamaForCausalLM,
    dtype: torch.dtype,
    device: torch.device,
    page_size=32,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
):
    if infer_state is None:
        config = self.model.config
        infer_state = InferState(
            n_layers=config.num_hidden_layers,
            n_qo_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim if config.head_dim is not None else config.hidden_size // config.num_attention_heads,
            page_size=page_size,
            dtype=dtype,
            device=device,
            debug=debug,
            **kwargs,
        )

    if hasattr(self, "lm_head"):
        _lm_head_forward = self.lm_head.forward
        self.lm_head.forward = lambda x: _lm_head_forward(x[:, -1:, :])

    self._icecache_infer_state = infer_state
    for mod in self.modules():
        mod_cls = str(mod.__class__)
        if "Attention" in mod_cls:
            mod.forward = (
                lambda mod, model: lambda *args, **kwargs: _icecache_attn_forward(
                    mod, *args, infer_state=model._icecache_infer_state,
                    debug=debug, **kwargs
                )
            )(mod, self)

    # Expose the state so a caller (e.g. an agent session driving continuation)
    # can read seq_len / dci_db without going through the attention closure.
    # Purely additive: attention patching behaviour is unchanged.
    self._icecache_infer_state = infer_state

    return self


def set_icecache_infer_state(model: LlamaForCausalLM, infer_state):
    """Select a prefill request state or a fixed decode batch without repatching."""
    if not hasattr(model, "_icecache_infer_state"):
        raise ValueError("call enable_icecache(model, ...) first")
    if infer_state is None or not hasattr(infer_state, "forward_mode"):
        raise TypeError("infer_state must be an initialized IceCache state")
    if getattr(model, "_icecache_state_scope_active", False):
        raise RuntimeError("cannot switch IceCache state during a bound forward")
    model._icecache_infer_state = infer_state


@contextmanager
def icecache_state(model: LlamaForCausalLM, infer_state):
    """Temporarily bind a request/batch state to an enabled model.

    This makes state switching exception-safe for schedulers that interleave
    requests, while retaining the lightweight patched attention functions.
    """
    if not hasattr(model, "_icecache_infer_state"):
        raise ValueError("call enable_icecache(model, ...) first")
    if infer_state is None or not hasattr(infer_state, "forward_mode"):
        raise TypeError("infer_state must be an initialized IceCache state")
    if getattr(model, "_icecache_state_scope_active", False):
        raise RuntimeError("nested IceCache state bindings are not supported")
    previous = model._icecache_infer_state
    model._icecache_state_scope_active = True
    model._icecache_infer_state = infer_state
    try:
        yield model
    finally:
        model._icecache_infer_state = previous
        model._icecache_state_scope_active = False
