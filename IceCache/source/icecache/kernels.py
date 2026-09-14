import torch
import math
from typing import Optional, Union

import icecache_cpp as _cpp

from .utils import (
    PosEncodingMode,
    TensorLayout,
    expand_5d,
    check_pos_encoding_mode,
    check_kv_layout,
    is_float8,
)


def rms_norm(
    inp: torch.Tensor,
    wgt: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return _cpp.rms_norm(
        inp,
        wgt,
        epsilon,
    )


def qk_apply_rotary_in_place(
    q: torch.Tensor,
    k: torch.Tensor,
    past_kv_len: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
):
    _cpp.qk_apply_rotary_in_place(
        q,
        k,
        past_kv_len,
        rope_scale,
        rope_theta,
    )


def qkq_apply_rotary_in_place(
    q: torch.Tensor,
    k: torch.Tensor,
    q1: torch.Tensor,
    past_kv_len: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
):
    _cpp.qkq_apply_rotary_in_place(
        q,
        k,
        q1,
        past_kv_len,
        rope_scale,
        rope_theta,
    )


def append_paged_kv_cache(
    k: torch.Tensor,  # [bsz, kv_len, n_kv_heads, head_dim]
    v: torch.Tensor,  # [bsz, kv_len, n_kv_heads, head_dim]
    kv_data: torch.Tensor,  # [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    kv_indices: torch.Tensor,  # [bsz, num_pages]
    kv_indptr: torch.Tensor,  # [bsz+1]
    kv_last_page_len: torch.Tensor,  # [bsz]
    layout: str = "NHD",
):
    (
        _cpp.append_paged_kv_cache_prefill
        if k.size(1) > 1
        else _cpp.append_paged_kv_cache_decode
    )(
        k,
        v,
        kv_data,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        TensorLayout[layout].value,
    )


def estimate_scores(
    q: torch.Tensor,  # [bsz, 1, num_heads, head_dim]
    dg_data: torch.Tensor,  # [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    dg_indices: torch.Tensor,  # [bsz, num_pages]
    dg_indptr: torch.Tensor,  # [bsz+1]
    dg_last_page_len: torch.Tensor,  # [bsz]
    dg_seq_len: int,
    layout: str = "NHD",
    n_groups: int = 1,
) -> torch.Tensor:  # [bsz, n_groups, n_kv_pages - 1]
    return _cpp.estimate_scores(
        q,
        dg_data,
        dg_indices,
        dg_indptr,
        dg_last_page_len,
        dg_seq_len,
        TensorLayout[layout].value,
        n_groups,
    )


def select_topk(
    scores: torch.Tensor,  # [bsz, n_kv_pages - 1]
    out_data: torch.Tensor,  # [bsz, topk]
    out_inds: torch.Tensor,  # [bsz, topk + ns + nw]
    new_in: torch.Tensor,  # [bsz, cap]
    incache: torch.Tensor,  # [bsz, n_kv_pages]
    pos_ids: torch.Tensor,  # [bsz, cap]
    recall_ids: torch.Tensor,  # [bsz, topk + 1]
    buf: torch.Tensor,
    topk: int,
    n_sink_pages: int,
    n_win_pages: int,
):
    _cpp.select_topk(
        scores,
        out_data,
        out_inds,
        new_in,
        incache,
        pos_ids,
        recall_ids,
        buf,
        topk,
        n_sink_pages,
        n_win_pages,
    )


def prefill_select_topk(
    scores: torch.Tensor,  # [bsz, n_kv_pages - 1]
    out_data: torch.Tensor,  # [bsz, topk]
    out_inds: torch.Tensor,  # [bsz, topk + ns + nw = cap]
    incache: torch.Tensor,  # [bsz, n_kv_pages]
    incache1: torch.Tensor,  # [bsz, n_kv_pages]
    pos_ids: torch.Tensor,  # [bsz, cap]
    buf: torch.Tensor,
    topk: int,
    n_sink_pages: int,
    n_win_pages: int,
):
    _cpp.prefill_select_topk(
        scores,
        out_data,
        out_inds,
        incache,
        incache1,
        pos_ids,
        buf,
        topk,
        n_sink_pages,
        n_win_pages,
    )


class BatchPrefillWithPagedKVCacheWrapper:
    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _cpp.BatchPrefillWithPagedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value
        )
        self._qo_indptr = None
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        # paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        batch_size = len(qo_indptr) - 1
        self._qo_indptr = qo_indptr
        self._paged_kv_indptr = paged_kv_indptr
        # self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len
        self._wrapper.begin_forward(
            self._workspace_buffer,
            qo_indptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )

    def end_forward(self):
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
        self._qo_indptr = None
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        causal: bool = True,
        pos_encoding_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        return_lse: bool = False,
    ):
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        assert not is_float8(q)
        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        ret = self._wrapper.forward(
            q.reshape(-1, *q.shape[-2:]),
            self._qo_indptr,
            paged_kv_data,
            self._paged_kv_indptr,
            # self._paged_kv_indices,
            paged_kv_indices,
            self._paged_kv_last_page_len,
            causal,
            PosEncodingMode[pos_encoding_mode].value,
            allow_fp16_qk_reduction,
            sm_scale,
            rope_scale,
            rope_theta,
            return_lse,
        )
        # The C++ op already returns {o} or {o, lse}; lse has shape
        # [nnz_qo, num_qo_heads] in fp32.
        if return_lse:
            return ret[0], ret[1]
        return ret[0]


class BatchDecodeWithPagedKVCacheWrapper:
    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _cpp.BatchDecodeWithPagedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value
        )
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        paged_kv_indptr: torch.Tensor,
        # paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        data_type: Union[str, torch.dtype] = "float16",
    ):
        self._paged_kv_indptr = paged_kv_indptr
        # self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len

        batch_size = len(paged_kv_indptr) - 1
        # NOTE(Zihao): the following tensor acts as placeholder to pass dtype info
        empty_data = torch.empty(
            0,
            dtype=(
                getattr(torch, data_type) if isinstance(data_type, str) else data_type
            ),
        )
        self._wrapper.begin_forward(
            self._workspace_buffer,
            paged_kv_indptr,
            paged_kv_last_page_len,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            PosEncodingMode[pos_encoding_mode].value,
            empty_data,
        )

    def end_forward(self):
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        page_valid_entries: Optional[torch.Tensor] = None,
        dci: Optional[bool] = False,
        return_lse: bool = False,
    ):
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        if page_valid_entries is None or dci is False:
            dci = False
            page_valid_entries = torch.empty(0)
        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        ret = self._wrapper.forward(
            q.reshape(-1, *q.shape[-2:]),
            paged_kv_data,
            self._paged_kv_indptr,
            # self._paged_kv_indices,
            paged_kv_indices,
            self._paged_kv_last_page_len,
            PosEncodingMode[pos_encoding_mode].value,
            sm_scale,
            rope_scale,
            rope_theta,
            return_lse,
            page_valid_entries,
            dci,
        )
        # lse is [batch_size, num_qo_heads] in fp32 when requested.
        if return_lse:
            return ret[0], ret[1]
        return ret[0]


# ---------------------------------------------------------------------------
# Split attention bookkeeping
#
# The two-pass continuation scheme (non-causal over the old context, causal
# over the new chunk) produces two partial results that must be combined by
# their log-sum-exp.  Upstream FlashInfer ships ``merge_state`` for this, but
# the vendored copy under ``3rdparty/flashinfer`` does NOT contain it -- only
# decode.cuh has been touched, to add ``page_valid_entries``.  So the merge is
# implemented here.
#
# IMPORTANT -- lse base.  The lse produced by these wrappers is in the **base-2**
# (log2) domain, matching FlashInfer's internal softmax.  Measured on this build:
# lse_kernel / lse_natural == log2(e) == 1.4427 (to fp32 precision), while the
# attention output itself matches a natural-log reference to ~2e-4.  The merge
# below is therefore written with exp2/log2 and returns lse in the same base-2
# domain as its inputs.  See docs/phase_b_continuation_prefill_design.md.
# ---------------------------------------------------------------------------

LOG2E = 1.4426950408889634


def lse_log2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    """Convert a base-2 lse (as returned by the wrappers) to natural log."""
    return lse / LOG2E


def lse_natural_to_log2(lse: torch.Tensor) -> torch.Tensor:
    """Convert a natural-log lse to the base-2 domain used by the wrappers."""
    return lse * LOG2E


def _lse_as_nh(lse: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Normalise an lse tensor to ``[N, num_heads]`` fp32."""
    if lse.ndim != 2:
        raise ValueError(f"lse must be 2-D, got shape {tuple(lse.shape)}")
    lse = lse.float()
    if lse.shape[-1] == num_heads:
        return lse
    if lse.shape[0] == num_heads:
        return lse.transpose(0, 1).contiguous()
    raise ValueError(
        f"lse shape {tuple(lse.shape)} matches neither [N, {num_heads}] nor [{num_heads}, N]"
    )


def merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
):
    """Equivalent of FlashInfer's ``merge_state`` for two disjoint partial results.

    ``out_*`` is ``[N, num_heads, head_dim]``; ``lse_*`` is ``[N, num_heads]``
    (or its transpose) in the **base-2** domain.  Returns ``(out, lse)`` with
    ``out`` in the dtype of ``out_a`` and ``lse`` still base-2.

    Formula (base-2)::

        m   = max(lse_a, lse_b)
        out = (out_a * 2**(lse_a - m) + out_b * 2**(lse_b - m))
              / (2**(lse_a - m) + 2**(lse_b - m))
        lse = m + log2(2**(lse_a - m) + 2**(lse_b - m))
    """
    if out_a.shape != out_b.shape:
        raise ValueError(f"out shapes differ: {tuple(out_a.shape)} vs {tuple(out_b.shape)}")
    num_heads = out_a.shape[-2]
    la = _lse_as_nh(lse_a, num_heads).unsqueeze(-1)
    lb = _lse_as_nh(lse_b, num_heads).unsqueeze(-1)

    m = torch.maximum(la, lb)
    wa = torch.exp2(la - m)
    wb = torch.exp2(lb - m)
    denom = wa + wb
    safe = denom.clamp_min(torch.finfo(denom.dtype).tiny)
    out = (out_a.float() * wa + out_b.float() * wb) / safe
    lse = (m + torch.log2(safe)).squeeze(-1)
    return out.to(out_a.dtype), lse


def merge_state_single(out_a, lse_a, out_b, lse_b):
    """Convenience wrapper returning only the merged output."""
    return merge_state(out_a, lse_a, out_b, lse_b)[0]
