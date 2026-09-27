import warnings
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict
from functools import wraps
import asyncio


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

from icecache.infer_state import InferState
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


def _repeat_kv_heads(key_states, value_states, rep):
    """Repeat each KV head ``rep`` times, so the kernels see group size 1.

    The paged kernels dispatch on the GQA group size, ``num_qo_heads /
    num_kv_heads``, and this build's generated table only instantiates
    {1, 4, 8} (``icecache_cpp/src/generated/dispatch.inc``). A model outside
    that set -- Qwen2.5-7B, 28 query heads over 4 KV heads -- has no kernel at
    all and dies at the first prefill with "failed to dispatch group_size 7".
    Expanding the K/V *tensors* rather than generating a kernel is what stock
    GQA attention does anyway: HF's ``repeat_kv`` repeats each KV head over the
    consecutive block of query heads that shares it, so expanded head
    ``h * rep + r`` is the KV head that query head ``h * rep + r`` attended to
    before the expansion, and attention is unchanged. The cost is ``rep`` x the
    KV cache -- and, for the retrieval backends, ``rep`` x the instances each
    layer builds -- paid only by models the kernels cannot serve as-is.

    ``rep == 1``, i.e. every model whose native group size is already 1, 4 or 8,
    returns the inputs untouched: this is a strict no-op for them.
    """
    if rep == 1:
        return key_states, value_states
    # `repeat_kv` takes and returns [bsz, n_kv_heads, seq_len, head_dim], which is
    # exactly the K layout here; it also documents itself as
    # `repeat_interleave(x, dim=1, repeats=rep)`. V is [bsz, seq_len,
    # n_kv_heads, head_dim] -- its head axis is dim 2, not dim 1 -- so it cannot
    # go through `repeat_kv`, and `repeat_interleave` on dim 2 gives the same
    # ordering: head h repeated rep times in place, i.e. repeat_kv's
    # (h, r) -> h * rep + r. qwen3's k_norm does not change the layout.
    return repeat_kv(key_states, rep), torch.repeat_interleave(value_states, rep, dim=2)


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

    # Present full multi-head K/V when the model's group size is one the kernels
    # lack; no-op at rep 1. Everything downstream reads the head count from
    # `state.n_kv_heads`, which the expansion below makes the tensor agree with.
    key_states, value_states = _repeat_kv_heads(
        key_states, value_states, state.kv_head_rep)

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

    # Same expansion as prefill -- decode inserts one token per step into the
    # same cache, so its head count must match what prefill laid out. No-op at
    # rep 1.
    key_states, value_states = _repeat_kv_heads(
        key_states, value_states, state.kv_head_rep)

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

            # `estimate_select_recall` returns (None, None) BY DESIGN when there is
            # nothing to select -- see its own `if eids is not None:` guard. That
            # guard is repeated at both call sites here because `scatter_pages`
            # forwards straight into the pybind11 binding, which requires four
            # tensors and raises `TypeError: scatter_pages(): incompatible function
            # arguments` on a None.
            #
            # Measured trigger (2026-09-26): a prompt of <= budget*page_size tokens
            # (= 256 at page_size 16 / budget 16) makes `prefill_evict_extra_pages`
            # take its `else: self.use_dci = False` branch -- nothing to offload --
            # and then the first decode step that pushes seq_len past 256 hits the
            # same `None` at the `else` site below, which is the one the traceback
            # names. This receive branch sees it only when n_prefetch_layers > 0
            # (it is 0 in every harness here). Deterministic, not a race: it killed
            # the llama-3.1 `multi_news` run at row 96, whose log line reads
            # `Context length: 222`.
            #
            # The threshold 256 comes from the code above, NOT from the log. The
            # 95 surviving rows all had token_length >= 505, but they are a
            # dataset-order prefix, so that is survivorship and cannot pin a
            # threshold -- anything in (222, 505] fits the same evidence.
            if eids is not None:
                state.scatter_pages(cur_id, eids, nr)
            else:
                # Keep the invariant loud. This guard is only correct because
                # `use_dci = False` means the machinery is off for this row. If
                # `eids` is None while `use_dci` is True, something else went
                # wrong (the caller tests `n_pages > budget`, the callee tests
                # `n_real_pages == budget` -- different quantities that today
                # coincide), and skipping the scatter would silently decode with
                # the previous token's recalled pages still in place.
                assert not state.use_dci, (
                    f"scatter_pages got eids=None while use_dci is True "
                    f"(layer {cur_id}); the caller/callee budget conditions have "
                    f"diverged. Refusing to skip silently.")
        else:
            raise NotImplemented("kv cache is expected in the receive state")
    else:
        if budget is not None and kvc.n_pages > budget:
            eids, nr = state.estimate_select_recall(cur_id, query_states)

            # Guarded for the same reason as the receive branch above -- and this
            # is the call site that actually crashed: the pre-fix traceback names
            # `modeling.py, line 286`, which is this statement (`git show HEAD:`
            # then line 286; the guard is uncommitted).
            #
            # `use_dci = False` means the whole retrieval machinery is off for
            # this row -- `decode_sdpa` branches on the same flag and
            # `estimate_select_recall` is gated on it -- so there is nothing
            # selected and nothing to scatter. A no-op is the intended meaning.
            if eids is not None:
                state.scatter_pages(cur_id, eids, nr)
            else:
                assert not state.use_dci, (
                    f"scatter_pages got eids=None while use_dci is True "
                    f"(layer {cur_id}); the caller/callee budget conditions have "
                    f"diverged. Refusing to skip silently.")

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
    _, q_len, _ = hidden_states.size()

    if q_len > 1:
        return _icecache_prefill(
            self,
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_value,
            output_attentions,
            use_cache,
            infer_state,
            debug,
            **kwargs,
        )
    else:
        return _icecache_decode(
            self,
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_value,
            output_attentions,
            use_cache,
            infer_state,
            debug,
            **kwargs,
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
        # `rep` is the model's GQA group size -- what the kernel dispatch calls
        # group_size -- and the kernels only cover {1, 4, 8}. Presenting full
        # multi-head K/V (group size 1) for anything else is what keeps a model
        # like Qwen2.5-7B (28 q / 4 kv) off the "failed to dispatch group_size
        # 7" path; `_repeat_kv_heads` does the repeating, here and in decode.
        n_qo_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        assert n_qo_heads % n_kv_heads == 0, (
            f"num_attention_heads ({n_qo_heads}) is not a multiple of "
            f"num_key_value_heads ({n_kv_heads})")
        rep = n_qo_heads // n_kv_heads
        # 1 means "do not expand": the supported ratios keep the config's own
        # head count and every head-indexed tensor, byte for byte, as before.
        kv_head_rep = 1 if rep in (1, 4, 8) else rep
        infer_state = InferState(
            n_layers=config.num_hidden_layers,
            n_qo_heads=n_qo_heads,
            n_kv_heads=n_kv_heads * kv_head_rep,
            kv_head_rep=kv_head_rep,
            # `head_dim` is absent on Qwen2Config and None on MistralConfig; `or`
            # covers both, while a config that sets it (Qwen3-4B: 128 vs an 80-wide
            # hidden_size/num_attention_heads) keeps it verbatim.
            head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
            page_size=page_size,
            dtype=dtype,
            device=device,
            debug=debug,
            **kwargs,
        )

    if hasattr(self, "lm_head"):
        _lm_head_forward = self.lm_head.forward
        self.lm_head.forward = lambda x: _lm_head_forward(x[:, -1:, :])

    for mod in self.modules():
        mod_cls = str(mod.__class__)
        if "Attention" in mod_cls:
            mod.forward = (
                lambda mod: lambda *args, **kwargs: _icecache_attn_forward(
                    mod, *args, infer_state=infer_state, debug=debug, **kwargs
                )
            )(mod)
    
    return self