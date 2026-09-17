from typing import List, Union, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import asyncio
from threading import Thread
import threading

import torch
from torch import Tensor

from .kv_cache import KvPool, KvCache
from . import kernels
from . import utils

import icecache_cpp as _cpp

import numpy as np
from time import time
from dciknn import DCI
from tqdm import tqdm
import copy
from ctypes import c_float, POINTER, cast, c_void_p

class DeprecatedError(NotImplementedError):
    pass


Digest = Tuple[Tensor, Tensor]


class ForwardMode(Enum):
    """Explicit attention mode for a forward pass.

    Set by the caller (a session) immediately before invoking the model and
    cleared afterwards.  ``None`` means "legacy": dispatch on ``q_len``, which is
    what ``generate()`` and the benchmark scripts rely on, and what phase A's
    oracle uses.  ``CONTINUATION_PREFILL`` is never inferred from ``q_len`` --
    it must be declared, because a multi-token continuation chunk is otherwise
    indistinguishable from a fresh prompt.
    """

    INITIAL_PREFILL = "initial_prefill"
    CONTINUATION_PREFILL = "continuation_prefill"
    DECODE = "decode"


class InferState:
    def __init__(
        self,
        n_layers,
        n_qo_heads,
        n_kv_heads,
        head_dim,
        page_size,
        dtype: torch.dtype,
        device: torch.device,
        page_budgets: Union[int, List[int]] = None,
        n_max_pages=None,
        n_unlimited_layers=None,
        n_max_bytes=None,
        page_topks: Union[int, List[int]] = None,
        n_max_cpu_pages=None,
        n_max_cpu_bytes=None,
        n_sink_pages=2,
        n_win_pages=2,
        use_sparse_attn=False,
        n_prefetch_layers=0,
        n_reuse_layers=0,
        group_size=None,
        n_groups=None,
        debug=False,
        ratio_1=0.01,
        ratio_2=0.2,
        gpu_pool=None,
        **kwargs,
    ) -> None:
        self.n_layers = n_layers
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.ratio = self.n_qo_heads // self.n_kv_heads
        assert self.ratio >= 1
        self.head_dim = head_dim
        self.ratio_1 = ratio_1
        self.ratio_2 = ratio_2

        self.dtype = dtype
        self.device = device
        self.cpu_dtype = torch.float32

        self.offload_ratio: int = 2  # >= 1
        self.search_ratio: float = 1e-3
        self.debug = debug
        self.use_dci = True
        self.parallel_level = 2

        if n_max_pages is None:
            assert n_max_bytes is not None
            n_max_pages = n_max_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        if n_max_cpu_pages is None:
            assert n_max_cpu_bytes is not None
            n_max_cpu_pages = n_max_cpu_bytes // (
                2 * page_size * n_kv_heads * head_dim * self.cpu_dtype.itemsize
            )
        if n_unlimited_layers is None:
            n_unlimited_layers = 0
        if not isinstance(page_budgets, (list, tuple)):
            page_budgets = [None] * n_unlimited_layers + [page_budgets] * (
                n_layers - n_unlimited_layers
            )
        if page_topks is None:
            page_topks = [b and b // 2 for b in page_budgets]
        elif not isinstance(page_topks, (list, tuple)):
            page_topks = [None] * n_unlimited_layers + [page_topks] * (
                n_layers - n_unlimited_layers
            )

        self.page_size = page_size
        self.n_max_pages = n_max_pages
        self.n_max_cpu_pages = n_max_cpu_pages
        self.layer2budget = page_budgets
        self.budget2layers = defaultdict(list)
        for i, b in enumerate(page_budgets):
            self.budget2layers[b].append(i)
        self.layer2topk = page_topks
        self.n_sink_pages = n_sink_pages
        self.n_win_pages = n_win_pages
        assert n_win_pages >= 2
        self.num_offload_pages = None  # number of pages offloaded to CPU
        # number of dci pages in GPU cache (originally, they are all sequential pages)
        self.n_dci_pages = None
        self.use_sparse_attn = use_sparse_attn
        self.page_valid_entries = [None] * n_layers
        self.selected_page_idx = [None] * n_layers

        self.cpu_n_bytes_per_page = 2 * page_size * \
            n_kv_heads * head_dim * self.cpu_dtype.itemsize

        for i in range(n_layers):
            b = self.layer2budget[i]
            k = self.layer2topk[i]
            if b is not None:
                assert k is not None and k < b - n_sink_pages - n_win_pages

        self.layout = "HND"
        self._i32 = dict(dtype=torch.int32, device=self.device)
        self._b = dict(dtype=torch.bool, device=self.device)
        self._u8 = dict(dtype=torch.uint8, device=self.device)
        self._fp = dict(dtype=self.dtype, device=self.device)
        self._cfp = dict(dtype=self.cpu_dtype, device=torch.device("cpu"))
        self._ci32 = dict(dtype=torch.int32, device=torch.device("cpu"))
        self._cb = dict(dtype=torch.bool, device=torch.device("cpu"))

        self._owns_gpu_pool = gpu_pool is None
        self._pool = gpu_pool if gpu_pool is not None else KvPool(
            n_max_pages, page_size, n_kv_heads, head_dim, dtype, device,
            (0, 2, 1, 3))
        if (self._pool.page_size != page_size or
                self._pool.n_kv_heads != n_kv_heads or
                self._pool.head_dim != head_dim or
                self._pool.dtype != dtype or self._pool.device != device or
                self._pool._layout_map != (0, 2, 1, 3)):
            raise ValueError("shared GPU KV pool configuration does not match InferState")
        self.kv_caches: List[KvCache] = [None] * self.n_layers
        self.dci_db = [None] * self.n_layers
        self._cpu_pool = KvPool(
            n_max_cpu_pages, page_size, n_kv_heads, head_dim, self.cpu_dtype, torch.device(
                "cpu"), (0, 2, 1, 3)
        )
        self.cpu_kv_caches: List[KvCache] = [None] * self.n_layers
        self.temp_cpu_kv_caches: List[KvCache] = [None] * self.n_layers
        self.cpu_neighbour_caches: List[KvCache] = [None] * self.n_layers
        self.offload_win_caches: List[KvCache] = [None] * self.n_layers
        self.offload_win_flag: List[bool] = [False] * self.n_layers

        self.kv_last_page_len = None
        self.kv_last_page_lens: Tensor = None
        self.kv_indptrs_tab: Dict[int, Tensor] = {
            b: None for b in self.budget2layers}
        self.kv_decode_indptrs_tab: Dict[int, Tensor] = {
            b: None for b in self.budget2layers
        }

        self.num_evict_win = None
        self.prev_selected = None
        self.n_offloaded_win_caches = None
        self.kvc_capacity = [None] * self.n_layers
        self.prev_num_points = None
        self.prev_num_pages = None
        self.prev_nr = None
        self.prev_eids = None
        self.prev_rids = None
        self.prev_index = None
        self.prev_offset = None

        # TODO (Qitong):
        wbufs = [torch.empty(16 * 1024 * 1024, **self._u8)
                 for _ in self.budget2layers]
        self.prefill_handler = kernels.BatchPrefillWithPagedKVCacheWrapper(
            wbufs[0], self.layout
        )
        self.prefill_handler_tab = {
            b: kernels.BatchPrefillWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }
        self.decode_handler_tab = {
            b: kernels.BatchDecodeWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }

        self.n_prefetch_layers = n_prefetch_layers
        self.n_reuse_layers = n_reuse_layers
        self.nn_idx_all = None
        self.attn_layers = [None] * n_layers

        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.decode_backup_stream = torch.cuda.Stream(self.device)
        self.prefetch_streams = None
        self.on_decode_prefetch = None

        self.c2g_stream = torch.cuda.Stream(self.device)

        self.c2g_copy_event = torch.cuda.Event(
            blocking=False, enable_timing=False)
        self.c2g_scatter_event = torch.cuda.Event(
            blocking=False, enable_timing=False)
        self.c2g_cast_event = torch.cuda.Event(
            blocking=False, enable_timing=False)

        # g2c_stream is for offloading to CPU
        self.g2c_stream = torch.cuda.Stream(self.device)

        self._thread_locals = {}
        self._thread_locals_lock = threading.Lock()

        self._loop = asyncio.new_event_loop()
        self._loop_executor = ThreadPoolExecutor(
            max_workers=1,
            initializer=self._worker_context_init
        )

        def _start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        Thread(target=_start_loop, args=(self._loop,), daemon=True).start()

        # Asyncio task future results for three states: [SEND, REUSE, RECEIVE]
        # self._future_dci_results = [None] * 3
        self._dci_future = None

        # Explicit forward mode; None => legacy q_len dispatch.  See ForwardMode.
        self.forward_mode = None

        if n_prefetch_layers > 0:
            self.n_reused_layers = self.n_prefetch_layers - 1
            if self.n_prefetch_layers > 1:
                self.nn_idx_all = np.full([self.batch_size, self.n_layers, self.n_kv_heads, self.ratio, (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages - self.layer2topk[-1])], -1, dtype=np.int32)
            
            assert self.n_prefetch_layers == 1
            self.finish_using_buffer = torch.cuda.Event(
                blocking=False, enable_timing=False)
            
        assert group_size is None or n_groups is None
        if group_size is None and n_groups is None:
            n_groups = 1
            group_size = n_kv_heads
        elif group_size is None:
            assert n_kv_heads % n_groups == 0
            group_size = n_kv_heads // n_groups
        else:
            assert n_kv_heads % group_size == 0
            n_groups = n_kv_heads // group_size
        self.group_size = group_size
        self.n_groups = n_groups

    @property
    def seq_len(self):
        return self.kv_caches[0].seq_len

    @property
    def n_pages(self):
        return self.kv_caches[0].n_pages

    @property
    def batch_size(self):
        return self.kv_caches[0].batch_size
    
    def check_reuse(self, cur_id, start=2):
        if self.n_reuse_layers == 0 or cur_id <= start:
            return 0
        position = cur_id - start
        cycle_length = self.n_reuse_layers
        position_in_cycle = position % cycle_length
        if position_in_cycle == 0:
            return 0
        else:
            return cur_id-position_in_cycle

    def _prepare_prefill(self, bsz, q_len):
        if not self._owns_gpu_pool and any(cache is not None for cache in self.kv_caches):
            raise RuntimeError("shared-pool request states support one initial prefill; create a fresh state")
        self.num_offload_pages = None
        self.n_dci_pages = None
        self.offload_win_flag = [False] * self.n_layers
        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.selected_page_idx = [None] * self.n_layers
        # ordered CPU page log (policy b): page j = the j-th offloaded page in
        # token order.  Fed at both ingestion points (prompt offload in
        # _DCI_first_call, chunk evictions in _prepare_continuation_sparse);
        # block retrieval reads THIS, not the tree's per-head leaf space.
        self._page_log = [[] for _ in range(self.n_layers)]
        self._block_sel = {}
        self.kv_caches = [None] * self.n_layers
        self.cpu_kv_caches = [None] * self.n_layers
        self.temp_cpu_kv_caches = [None] * self.n_layers
        self.cpu_neighbour_caches = [None] * self.n_layers
        self.offload_win_caches = [None] * self.n_layers
        self.offload_win_flag = [False] * self.n_layers
        self.kv_last_page_len = None
        self.kv_last_page_lens = None
        self.num_evict_win = None
        self.prev_num_points = None
        self.prev_num_pages = None
        self.prev_nr = None
        self.prev_eids = None
        self.prev_rids = None
        self.prev_index = None
        self.prev_offset = None
        self.n_offloaded_win_caches = None
        self.kvc_capacity = [None] * self.n_layers
        self.page_address_buffer = [None] * self.n_layers
        self.kv_indptrs_tab = {b: None for b in self.budget2layers}
        self.kv_decode_indptrs_tab = {b: None for b in self.budget2layers}
        self.prev_selected = None
        self.page_valid_entries = [None] * self.n_layers
        self.attn_layers = [None] * self.n_layers
        # A shared pool may already contain another request's prefilled KV.
        # Its owner, not an individual request, controls global clearing.
        if self._owns_gpu_pool:
            self._pool.clear()
        self.kv_caches = [
            KvCache(
                self._pool,
                bsz,
                self.layer2budget[i],
                self.n_sink_pages,
                self.n_win_pages,
                n_groups=self.n_groups,
                offload_ratio=self.offload_ratio,
            )
            for i in range(self.n_layers)
        ]
        self._cpu_pool.clear()
        self.cpu_kv_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]
        self.temp_cpu_kv_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]
        self.cpu_neighbour_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]
        self.offload_win_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]
        self.use_dci = True

        self.n_offloaded_win_caches = 1
        [kvc.prefill_alloc_n_tokens(self.page_size)
         for kvc in self.offload_win_caches if kvc]

        self.page_valid_entries = [
            self.layer2budget[i] and torch.full(
                [self.layer2budget[i], self.n_kv_heads], self.page_size, **self._i32)
            for i in range(self.n_layers)
        ]

        # Shape [num_max_pages, page_floats] = [b * n, s * d]
        # 2 * for both K and V
        if self.n_reuse_layers > 0:
            self.cpu_transit_buffer = torch.empty(
                [self.batch_size, 
                2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages) * self.n_reuse_layers,
                self.page_size * self.head_dim],
                **self._cfp,
                pin_memory=True
            )
            self.cuda_transit_buffer = torch.empty(
                [self.batch_size, 
                2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages) * self.n_reuse_layers,
                self.page_size * self.head_dim],
                **self._fp,
                pin_memory=False
            )
            self._src_address_buffer = np.zeros(self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages) * self.n_reuse_layers, dtype=np.int64)
        else:
            self.cpu_transit_buffer = torch.empty(
                [self.batch_size, 
                2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages),
                self.page_size * self.head_dim],
                **self._cfp,
                pin_memory=True
            )
            self.cuda_transit_buffer = torch.empty(
                [self.batch_size, 
                2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages),
                self.page_size * self.head_dim],
                **self._fp,
                pin_memory=False
            )
            self._src_address_buffer = np.zeros(self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages), dtype=np.int64)

        
        if self.dtype is not torch.float32:
            self.cuda_cast_buffer = torch.empty(
                [self.batch_size, 2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages),
                    self.page_size * self.head_dim],
                dtype=self.dtype, device=self.device,
                pin_memory=False
            )
        else:
            self.cuda_cast_buffer = self.cuda_transit_buffer

        self.n_kv_pages = (q_len + self.page_size - 1) // self.page_size
        self.kv_last_page_len = (q_len - 1) % self.page_size + 1
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        for b in self.kv_indptrs_tab:
            self.kv_indptrs_tab[b] = kv_indptr = torch.arange(
                0, bsz * self.n_kv_pages + 1, self.n_kv_pages, **self._i32
            )
        if q_len > self.page_size * (self.n_sink_pages + self.n_win_pages):
            # volume = 1 << (q_len - 1).bit_length()  # smallest_power_of_two -- is handled in DCI
            self.proj_vec = torch.nn.functional.normalize(
                torch.randn(1, self.head_dim + 1, **self._fp),
                p=2, dim=1
            ).reshape(-1)
            proj_vec = self.proj_vec.detach().cpu().numpy().astype(np.float32)

            self.dci_db = [None] * self.n_layers
            for i in range(self.n_layers):
                if self.layer2budget[i] and self.check_reuse(i) == 0:
                    self.dci_db[i] = DCI(self.head_dim, 1, 1, promotion_prob=self.ratio_1, promotion_prob_subseq=self.ratio_2, num_points=q_len, init=True, num_inst=self.n_kv_heads, debug=self.debug, transform=True, parallel_level=self.parallel_level, proj_vec=proj_vec)

        qo_indptr = torch.arange(0, bsz * q_len + 1, q_len, **self._i32)
        self.prefill_handler.begin_forward(
            qo_indptr,
            kv_indptr,
            self.kv_last_page_lens,
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
        )
    
    def _worker_context_init(self):
        """Initialize CUDA context in worker thread"""
        try:
            # Store CUDA objects in thread-local storage instead of self
            thread_local = threading.local()
            thread_local.c2g_stream = torch.cuda.Stream(self.device)
            thread_local.c2g_copy_event = torch.cuda.Event(
                blocking=False, enable_timing=False)
            thread_local.c2g_scatter_event = torch.cuda.Event(
                blocking=False, enable_timing=False)
            thread_local.c2g_cast_event = torch.cuda.Event(
                blocking=False, enable_timing=False)

            with self._thread_locals_lock:
                self._thread_locals[threading.get_ident()] = thread_local

        except Exception as e:
            # Log the error for debugging
            print(f"Error in worker context initialization: {e}")
            raise

    def _finish_prefill(self, bsz, q_len):
        for b, ls in self.budget2layers.items():
            n_kv_pages = utils.all_eq(
                self.kv_caches[l].n_real_pages for l in ls)
            self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )
        self.prefill_handler.end_forward()

        # Offload window pages if the last page is full
        if self.kv_caches[-1].n_real_pages >= self.kv_caches[-1].budget and self.kv_last_page_len == self.page_size:
            for l in range(self.n_layers):
                self.offload_win_flag[l] = True
            with torch.cuda.stream(self.decode_backup_stream):
                [self.decode_backup_win_page(l) for l in range(self.n_layers)]

    # ------------------------------------------------------------------
    # Phase B1: multi-token continuation over an existing, fully resident KV.
    #
    # Scope is deliberately narrow: no DCI retrieval, no page eviction, no
    # window rotation.  It must not call _prepare_prefill()/_finish_prefill(),
    # must not clear any pool, and must not touch the DCI objects.
    # ------------------------------------------------------------------

    def _prepare_continuation(self, bsz, q_len):
        """Allocate pages for a continuation chunk and arm the paged prefill.

        The chunk's attention is a plain causal paged prefill over the whole
        (contiguous, fully GPU-resident) KV, so no split-attention merge is
        needed -- see ``kernels.merge_state`` for why that is *not* true once
        DCI-retained semantic pages enter the picture.
        """
        if q_len <= 1:
            raise ValueError(f"continuation requires q_len > 1, got {q_len}")
        if self.use_dci:
            raise NotImplementedError(
                "continuation over the sparse DCI path is not implemented. This build "
                "supports continuation only when nothing was offloaded (use_dci == False), "
                "i.e. the whole sequence fits the GPU page budget. Chunked DCI retrieval is "
                "phase C; see docs/phase_b_continuation_prefill_design.md."
            )
        if self.kv_caches[0] is None:
            raise RuntimeError("continuation requires a completed initial prefill")

        # Full-cache contract: the whole sequence must stay resident.  A chunk
        # that would exceed the page budget must fail loudly -- the allocation
        # below would otherwise silently rotate the window (the decode path's
        # eviction branch) and destroy pages with no tree insertion
        # (review finding: "追加后超过 page budget 时仍可能静默轮转").
        kvc0 = self.kv_caches[0]
        if kvc0.budget is not None:
            projected_tokens = int(kvc0.seq_len) + q_len
            projected_pages = (
                projected_tokens + self.page_size - 1) // self.page_size
            if projected_pages > kvc0.budget:
                raise ValueError(
                    "full-cache continuation would exceed the page budget: "
                    f"{projected_tokens} tokens -> {projected_pages} pages > "
                    f"budget {kvc0.budget}. Enlarge page_budgets or ingest the "
                    "tool result in smaller chunks."
                )

        # Append the chunk's pages to every layer.  With no offload there is no
        # eviction, so this is a pure append; `alloc_page` still handles pool
        # bookkeeping.
        for kvc in self.kv_caches:
            kvc.decode_alloc_n_tokens(q_len, self.alloc_page)

        self.n_kv_pages = utils.all_eq(kvc.n_real_pages for kvc in self.kv_caches)
        self.kv_last_page_len = utils.all_eq(kvc.last_page_len for kvc in self.kv_caches)
        self.kv_last_page_lens = torch.tensor([self.kv_last_page_len] * bsz, **self._i32)

        kv_indptr = torch.arange(0, bsz * self.n_kv_pages + 1, self.n_kv_pages, **self._i32)
        for b in self.budget2layers:
            self.kv_indptrs_tab[b] = self.kv_decode_indptrs_tab[b] = kv_indptr

        qo_indptr = torch.arange(0, bsz * q_len + 1, q_len, **self._i32)
        self.prefill_handler.begin_forward(
            qo_indptr,
            kv_indptr,
            self.kv_last_page_lens,
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
        )

    def _finish_continuation(self, bsz, q_len):
        for b, ls in self.budget2layers.items():
            n_kv_pages = utils.all_eq(
                self.kv_caches[l].n_real_pages for l in ls)
            self.kv_indptrs_tab[b] = self.kv_decode_indptrs_tab[b] = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )
        self.prefill_handler.end_forward()

    # ------------------------------------------------------------------
    # Sparse continuation (DCI active): a whole chunk is ingested in one
    # forward instead of one token at a time.
    #
    # The resident KV is gathered into per-KV-head contiguous runs (see the
    # B0b note on page_valid_entries being [page, kv_head]) and the chunk is
    # appended after each head's resident run.  All resident tokens precede
    # every chunk token, so a *plain causal* mask over the concatenation is the
    # right attention -- no split, no LSE merge (design doc section 5.6).
    # ------------------------------------------------------------------

    def _resident_valid_counts(self, layer_idx):
        """[n_slots, n_kv_heads] valid entry count of the currently resident KV.

        Mirrors exactly what the decode kernel sees: ``page_valid_entries`` is
        indexed ``[page_idx, head]`` where ``page_idx`` runs over the resident
        slots in ``paged_kv_indices`` order (i.e. ``c2p`` order), and the tail
        slot's valid count is further clamped by ``kv_last_page_len``.
        """
        kvc = self.kv_caches[layer_idx]
        n = kvc.n_real_pages
        counts = self.page_valid_entries[layer_idx][:n].clone()   # [n, H]
        if self.kv_last_page_len < self.page_size:
            counts[n - 1] = torch.minimum(
                counts[n - 1],
                torch.full_like(counts[n - 1], self.kv_last_page_len),
            )
        return counts

    def _pack_resident(self, layer_idx, chunk_len):
        """Gather the resident KV into page-aligned per-KV-head runs.

        Returns ``(stage, indices, indptr, last_len)`` where ``stage`` is HND
        with ``num_kv_heads == 1`` and one *request* per KV head; each request's
        resident tokens are packed contiguously (slot-major within a head) and
        ``chunk_len`` empty slots are reserved after them.

        The packing order must match the decode kernel's view: for head ``h``,
        the resident tokens are ``slot 0..n-1`` each truncated to its per-head
        valid count, in that order.
        """
        kvc = self.kv_caches[layer_idx]
        H, ps, D = self.n_kv_heads, self.page_size, self.head_dim
        counts = self._resident_valid_counts(layer_idx)      # [n, H]
        n = kvc.n_real_pages
        slots = kvc.c2p[0]                                   # [n] absolute page ids

        per_head = counts.sum(0)                             # [H] resident token count
        totals = per_head + chunk_len
        pages = (totals + ps - 1) // ps
        run_start = torch.cumsum(pages * ps, 0) - pages * ps
        total_slots = int(run_start[-1].item() + pages[-1].item() * ps)
        stage = torch.zeros(total_slots, 2, D, dtype=self.dtype, device=self.device)

        # Pack head by head, slot by slot -- unambiguous, mirrors decode order.
        for h in range(H):
            run = run_start[h]
            for s in range(n):
                c = int(counts[s, h].item())
                if c == 0:
                    continue
                # buffer is [n_phys, 2, H, ps, D]; slice -> [2, c, D]; transpose -> [c, 2, D]
                src = kvc.buffer[slots[s], :, h, :c, :].transpose(0, 1).contiguous()
                stage[run : run + c] = src
                run += c

        # Rewrap into the [n_pages, 2, 1, ps, D] layout the kernel expects.
        n_pages = total_slots // ps
        stage = stage.reshape(n_pages, ps, 2, D).permute(0, 2, 1, 3).contiguous()
        stage = stage.unsqueeze(2)  # -> [n_pages, 2, 1, ps, D]

        last_len = ((totals - (pages - 1) * ps).to(torch.int32))
        indptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=self.device),
            torch.cumsum(pages, 0).to(torch.int32).to(self.device),
        ])
        indices = torch.arange(n_pages, dtype=torch.int32, device=self.device)
        return stage, indices, indptr, last_len

    def _write_chunk_into_stage(self, layer_idx, keys, vals):
        """Place this layer's chunk K/V right after each head's resident run.

        ``stage`` is ``[n_pages, 2, 1, ps, D]`` (page-major).  A linear slot
        index ``i`` maps to page ``i // ps``, offset ``i % ps``.  We write the
        chunk into the slots right after each head's resident run, using the
        (page, offset) decomposition directly so the write lands back in the
        real ``stage`` buffer (a flatten+permute view would be a detached copy).
        """
        stage, indices, indptr, last_len = self._cont_pack[layer_idx]
        H, D = self.n_kv_heads, self.head_dim
        ps = self.page_size
        C = keys.shape[1]
        per_head = self._cont_per_head[layer_idx]      # [H] resident token count
        run_start = self._cont_run_start[layer_idx]    # [H] page-aligned slot offset

        for h in range(H):
            start = int(run_start[h]) + int(per_head[h])
            for i in range(C):
                slot = start + i
                page = slot // ps
                off = slot % ps
                stage[page, 0, 0, off, :] = keys[0, i, h, :]
                stage[page, 1, 0, off, :] = vals[0, i, h, :]

    def continuation_sdpa_batched(self, layer_idx, q, stage, indices, indptr, last_len):
        """One paged prefill with the KV head in the batch dimension.

        ``q`` is ``[bsz, q_len, n_qo_heads, head_dim]`` (bsz == 1), the same
        shape ``prefill_sdpa`` consumes.  Each KV head keeps its own length,
        which a single ``last_page_len`` cannot express.
        """
        H, ratio, D = self.n_kv_heads, self.ratio, self.head_dim
        C = q.shape[1]
        # q: [1, C, H*ratio, D]. q head index = kv*ratio + g.  Move the head dim
        # to the front, split it into (kv, group), then reorder to kv-major batch.
        qb = (
            q[0].permute(1, 0, 2)                 # [H*ratio, C, D]
            .reshape(H, ratio, C, D)              # [H, ratio, C, D]
            .permute(0, 2, 1, 3)                  # [H, C, ratio, D]
            .reshape(H * C, ratio, D)
            .contiguous()
        )
        handler = self.prefill_handler
        handler.begin_forward(
            torch.arange(0, H * C + 1, C, dtype=torch.int32, device=self.device),
            indptr,
            last_len,
            ratio,
            1,
            D,
        )
        out = handler.forward(qb, stage, indices, causal=True)
        handler.end_forward()
        # out: [H*C, ratio, D] -> [C, H*ratio, D] -> [1, C, H*ratio, D]
        out = out.reshape(H, C, ratio, D).permute(1, 0, 2, 3).reshape(C, H * ratio, D)
        return out.unsqueeze(0)

    def _drain_pending_offload(self):
        """Insert any backed-up window page into the existing DCI tree."""
        self.default_stream.wait_stream(self.decode_backup_stream)
        if self.offload_win_flag[-1]:
            for l in range(self.n_layers):
                self.offload_win_page_to_DCI(l)
                self.offload_win_flag[l] = False

    def _prepare_continuation_sparse(self, bsz, q_len):
        """Allocate pages for a sparse continuation chunk, mirroring decode's
        window lifecycle.

        Deterministic CPU append ("page 追加到 cpu"): as the chunk's pages are
        allocated, the window pages they displace are captured and appended to
        the CPU store / DCI tree in **one bulk ``_DCI_add`` per layer**, in
        eviction order -- instead of decode's per-token backup/drain dance.

        Invariant kept: the tree and the window stay disjoint.  A page enters
        the tree only when it is about to leave (or has left) the window, so a
        later retrieval can never double-count tokens that are also resident.
        This is why the chunk's own pages do NOT enter the tree here: they are
        the new window content, and they flow to the CPU store when a later
        chunk/decode displaces them -- by the same deterministic path.

        The chunk's K/V is written by ``_finish_continuation_sparse`` AFTER all
        layers, landing at the physical tail (``append_start = seq_len - q_len``
        of the resident operand), which has room because the loop below
        allocated the chunk's pages.
        """
        if q_len <= 1:
            raise ValueError(f"continuation requires q_len > 1, got {q_len}")
        if self.kv_caches[0] is None:
            raise RuntimeError("continuation requires a completed initial prefill")
        if bsz != 1:
            raise NotImplementedError("sparse continuation supports batch_size == 1")

        # resident packing is deferred to _icecache_continuation, per layer, AFTER
        # the chunk-level DCI retrieval refreshes the semantic pages.
        self._cont_pack = [None] * self.n_layers
        self._cont_per_head = [None] * self.n_layers
        self._cont_run_start = [None] * self.n_layers
        self._cont_kv = [None] * self.n_layers

        # drain any backup a previous decode step left behind
        self._drain_pending_offload()

        # Advance the window token by token (same rotation as decode) and capture
        # the pages that leave it.  q_len is at most a few hundred and this is
        # pure bookkeeping -- the attention stays chunk-level (one forward).
        ps = self.page_size
        evicted = [[] for _ in range(self.n_layers)]  # per layer: [2, H, ps, D]
        for _ in range(q_len):
            for l, kvc in enumerate(self.kv_caches):
                if (kvc.budget is not None and kvc.n_real_pages >= kvc.budget
                        and kvc.last_page_len == ps):
                    # this token crosses a page boundary -> the allocation below
                    # rotates the window and destroys the page at next_evict_idx
                    evicted[l].append(
                        kvc.buffer[kvc.c2p[0, kvc.next_evict_idx]].clone()
                    )
            for kvc in self.kv_caches:
                kvc.decode_alloc_1_token(self.alloc_page)
            self.kv_last_page_len = utils.all_eq(
                kvc.last_page_len for kvc in self.kv_caches
            )
        self.kv_last_page_lens = torch.tensor([self.kv_last_page_len], **self._i32)
        self.n_dci_pages = (self.kv_caches[-1].budget - self.n_sink_pages
                            - self.kv_caches[-1].n_win_pages)

        # bulk append: one _DCI_add per layer with every page that left the
        # window, in eviction order (oldest first).  _DCI_add expects host
        # tensors (decode feeds it from the CPU offload cache), so copy down.
        # The same pages are appended to the ordered CPU page log (policy b's
        # retrieval source).
        H, D = self.n_kv_heads, self.head_dim
        for l in range(self.n_layers):
            if not evicted[l]:
                continue
            pages = torch.stack(evicted[l], dim=0).contiguous()  # [n, 2, H, ps, D]
            kv = (
                pages.permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4)
                .reshape(2, H, -1, D).contiguous().cpu()
            )
            self._DCI_add(0, l, kv[0], kv[1])
            self._page_log[l].extend([p.cpu() for p in evicted[l]])

        # if the chunk ends with a full tail, the NEXT rotation needs the page at
        # next_evict_idx preserved: defer it exactly the way decode does (backup
        # now; the next step's drain inserts it into the tree before its own
        # rotation destroys the slot)
        if (self.kv_caches[-1].n_real_pages >= self.kv_caches[-1].budget
                and self.kv_last_page_len == ps):
            for l in range(self.n_layers):
                self.offload_win_flag[l] = True
            with torch.cuda.stream(self.decode_backup_stream):
                [self.decode_backup_win_page(l) for l in range(self.n_layers)]

        self.n_kv_pages = utils.all_eq(kvc.n_real_pages for kvc in self.kv_caches)
        kv_indptr = torch.arange(0, self.n_kv_pages + 1, self.n_kv_pages, **self._i32)
        for b in self.budget2layers:
            self.kv_indptrs_tab[b] = self.kv_decode_indptrs_tab[b] = kv_indptr

    def _finish_continuation_sparse(self, bsz, q_len):
        # append_paged_kv_cache_prefill writes at the *tail* of the paged KV
        # (page.cuh: append_start = seq_len - append_seq_len), so the chunk lands
        # at positions [old_total, new_total) now that indptr/last_page_len
        # describe the post-chunk state.
        for l in range(self.n_layers):
            keys, vals = self._cont_kv[l]
            self.append_paged_kv_cache(l, keys, vals)
        self._cont_kv = [None] * self.n_layers
        self._cont_pack = [None] * self.n_layers
        self._cont_per_head = [None] * self.n_layers
        self._cont_run_start = [None] * self.n_layers
        for b, ls in self.budget2layers.items():
            n_kv_pages = utils.all_eq(self.kv_caches[l].n_real_pages for l in ls)
            self.kv_indptrs_tab[b] = self.kv_decode_indptrs_tab[b] = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )

    def _prepare_decode(self, bsz):
        if self.kv_last_page_len + 1 >= self.page_size:  # n_win_pages >= 2
            self.default_stream.wait_stream(self.decode_backup_stream)
            if self.offload_win_flag[-1]:
                for l in range(self.n_layers):
                    self.offload_win_page_to_DCI(l)
                    self.offload_win_flag[l] = False
        pre = [kvc.n_real_pages for kvc in self.kv_caches]
        n_new_kv_pages = utils.all_eq(
            kvc.decode_alloc_1_token(self.alloc_page) for kvc in self.kv_caches
        )
        self.kv_last_page_len = utils.all_eq(
            kvc.last_page_len for kvc in self.kv_caches
        )
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        # [kvc.decode_alloc_1_token() for kvc in self.cpu_kv_caches if kvc]

        if n_new_kv_pages > 0:
            assert n_new_kv_pages == 1

            # Update n_dci_pages
            self.n_dci_pages = self.kv_caches[-1].budget - \
                self.n_sink_pages - self.kv_caches[-1].n_win_pages

            cur = [kvc.n_real_pages for kvc in self.kv_caches]
            for b, ls in self.budget2layers.items():
                n_new_kv_real_pages = utils.all_eq(cur[l] - pre[l] for l in ls)
                if n_new_kv_real_pages > 0:
                    n_kv_pages = self.kv_caches[ls[0]].n_real_pages
                    self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = (
                        torch.arange(0, bsz * n_kv_pages + 1,
                                     n_kv_pages, **self._i32)
                    )

        if self.kv_caches[-1].n_real_pages >= self.kv_caches[-1].budget and self.kv_last_page_len == self.page_size:
            for l in range(self.n_layers):
                self.offload_win_flag[l] = True
            with torch.cuda.stream(self.decode_backup_stream):
                [self.decode_backup_win_page(l) for l in range(self.n_layers)]

        for b, h in self.decode_handler_tab.items():
            h.begin_forward(
                self.kv_decode_indptrs_tab[b],
                self.kv_last_page_lens,
                self.n_qo_heads,
                self.n_kv_heads,
                self.head_dim,
                self.page_size,
                data_type=self.dtype,
            )

    def _finish_decode(self, bsz):
        for handler in self.decode_handler_tab.values():
            handler.end_forward()

    def begin_forward(self, bsz, q_len):
        if q_len > 1:
            self._prepare_prefill(bsz, q_len)
        else:
            self._prepare_decode(bsz)

    def end_forward(self, bsz, q_len):
        if q_len > 1:
            self._finish_prefill(bsz, q_len)
        else:
            self._finish_decode(bsz)

    def _DCI_first_call(self, b, cur_id, query_states, key_states, value_states, projected):
        #######   DCI Inst Construction   #######
        if self.use_dci:
            dci_len = key_states.shape[1]

            _query_states = query_states.reshape(-1,
                                                 self.head_dim).float().numpy()
            _key_states = key_states.reshape(-1, self.head_dim).float().numpy()
            _value_states = value_states.reshape(-1,
                                                 self.head_dim).float().numpy()
            
            assert (_query_states.flags['C_CONTIGUOUS'])
            assert (_key_states.flags['C_CONTIGUOUS'])
            assert (_value_states.flags['C_CONTIGUOUS'])

            if self.check_reuse(cur_id) == 0:

                projected = projected.detach().cpu().numpy().astype(np.float32)
                assert (projected.flags['C_CONTIGUOUS'])

                num_neighbours = self.n_dci_pages - self.layer2topk[cur_id]
                query_field_of_view = 30
                construction_field_of_view = 30
                construction_prop_to_retrieve = 1.0
                query_prop_to_retrieve = 0.6

                num_to_visit = dci_len
                num_to_retrieve = -1
                prop_to_visit = 1.0

                padding_mask = np.ones(
                    [1, self.n_kv_heads, dci_len], dtype=np.bool_).reshape(-1)

                _, _ = self.dci_db[cur_id].add_query_at_end(_key_states, _query_states, _value_states,
                                                                padding_mask,
                                                                num_levels=-100,  # not used
                                                                num_points=dci_len,
                                                                num_neighbours=num_neighbours,
                                                                c_num_to_visit=num_to_visit,
                                                                c_num_to_retrieve=num_to_retrieve,
                                                                c_prop_to_visit=prop_to_visit,
                                                                c_prop_to_retrieve=construction_prop_to_retrieve,
                                                                c_field_of_view=construction_field_of_view,
                                                                q_num_to_visit=num_to_visit,
                                                                q_field_of_view=query_field_of_view,
                                                                q_num_to_retrieve=num_to_retrieve,
                                                                q_prop_to_visit=prop_to_visit,
                                                                q_prop_to_retrieve=query_prop_to_retrieve,
                                                                transform=True,
                                                                parallel_level=self.parallel_level,
                                                                causal=False,
                                                                random=False,
                                                                do_query=False,
                                                                track=False,
                                                                update_addr=False,
                                                                data_proj=projected,
                                                                ratio=self.ratio,
                                                                interval=120,
                                                                X=30,
                                                                anchor_threshold=0.9
                                                                )

                dci_db = self.dci_db[cur_id]
            else:
                reuse_id = self.check_reuse(cur_id)
                dci_db = self.dci_db[reuse_id]

            stride = self.cpu_n_bytes_per_page
            max_num_leaves = dci_db.num_leaves.max()
            cpu_cache = self.cpu_kv_caches[cur_id]
            kvc = self.kv_caches[cur_id]
            cpu_cache.prefill_alloc_n_tokens(max_num_leaves * self.page_size)
            _base = cpu_cache[b, 0].data_ptr()
            assert cpu_cache[b, -1].data_ptr() - _base == (max_num_leaves - 1) * stride  # check contiguous allocation

            # For Offloading
            ns = kvc.n_sink_pages

            ev_gpi = kvc.c2p.clone()
            ev_gpi[:, :ns] = -1
            ev_gpi[:, -kvc.budget + ns:] = -1

            self.prefill_evicted_pages[cur_id] = ev_gpi

            kvc.c2p = torch.cat(
                [kvc.c2p[:, :ns], kvc.c2p[:, -kvc.budget + ns:]], dim=-1)

            self.kvc_capacity[cur_id] = 1 << (int(max_num_leaves) - 1).bit_length()
            kvc.cc2gp = torch.full(
                [kvc.batch_size, self.n_kv_heads, self.kvc_capacity[cur_id]], -1,  **self._ci32)
            kvc.ccc = torch.ones(
                [kvc.batch_size, self.n_kv_heads, self.kvc_capacity[cur_id]], **self._cb)
            self.page_address_buffer[cur_id] = np.full(
                [kvc.batch_size, self.n_kv_heads, self.kvc_capacity[cur_id]], -1, dtype=np.uintp)

            
            if self.check_reuse(cur_id) == 0:
                # Use the new allocated pages to store the DCI tokens
                new_address = [None] * self.n_kv_heads
                # data rearrangement
                # !! Here assume batch size = 1
                offset = self.page_size * self.head_dim * self.cpu_dtype.itemsize

                for i in range(self.n_kv_heads):
                    base = _base + i * offset
                    tmp_addr = [cast(base + j * stride, c_void_p).value for j in range(dci_db.num_leaves[i])]
                    new_address[i] = tmp_addr
                    self.page_address_buffer[cur_id][b, i, :len(tmp_addr)] = np.array(tmp_addr, dtype=np.uintp)

                page_indices = np.array(np.tile(np.arange(max_num_leaves), (self.n_kv_heads, 1)), dtype=np.int32)
                dci_db.address_update(indices=page_indices, new_address=new_address,
                                               num_pages=dci_db.num_leaves, offset=self.n_kv_heads*self.page_size*self.head_dim)
            else:
                old_index, old_offset = self.dci_db[reuse_id].token2node
                layer_offset = _base - self.cpu_kv_caches[reuse_id][b, 0].data_ptr()
                head_offset = self.page_size * self.head_dim * self.cpu_dtype.itemsize
                for i in range(self.n_kv_heads):
                    tmp_addr = self.page_address_buffer[reuse_id][b, i, :dci_db.num_leaves[i]] + layer_offset
                    self.page_address_buffer[cur_id][b, i, :dci_db.num_leaves[i]] = np.array(tmp_addr, dtype=np.uintp)
                
                DCI.reuse_copy_node(p_index=old_index, p_offset=old_offset, keys=_key_states, values=_value_states, new_address=[cast(_base + i * head_offset, c_void_p).value for i in range(self.n_kv_heads)], kv_offset=self.n_kv_heads*self.page_size*self.head_dim)


    def _DCI_add(self, b, cur_id, key_states, value_states):
        if self.use_dci:

            dci_len = key_states.shape[1]

            _key_states = key_states.reshape(-1, self.head_dim).float().numpy()
            _value_states = value_states.reshape(-1,
                                                 self.head_dim).float().numpy()

            assert (_key_states.flags['C_CONTIGUOUS'])
            assert (_value_states.flags['C_CONTIGUOUS'])

            cpu_cache = self.cpu_kv_caches[cur_id]
            kvc = self.kv_caches[cur_id]

            if self.check_reuse(cur_id) == 0:

                num_neighbours = 1  # Not used
                query_field_of_view = max(
                    int((self.seq_len) * self.search_ratio), 30)
                construction_field_of_view = 20
                construction_prop_to_retrieve = 0.6
                query_prop_to_retrieve = 0.6

                prev_num_points = self.dci_db[cur_id].num_points[0]
                num_to_visit = prev_num_points
                num_to_retrieve = -1
                prop_to_visit = 1.0
                padding_mask = np.ones(
                    [1, self.n_kv_heads, dci_len], dtype=np.bool_).reshape(-1)

                prev_max_num_pages = self.dci_db[cur_id].num_leaves.max()

                self.prev_num_points = prev_num_points
                self.prev_num_pages = copy.deepcopy(self.dci_db[cur_id].num_leaves)
                self.prev_index, self.prev_offset = self.dci_db[cur_id].token2node

                ccc = kvc.ccc[b][:, :prev_max_num_pages].numpy().astype(
                    np.bool_).reshape(self.batch_size*self.n_kv_heads, -1)

                _, _ = self.dci_db[cur_id].add_query(_key_states, None, _value_states,
                                                    padding_mask,
                                                    num_levels=-100,  # not used
                                                    num_points=dci_len,
                                                    num_neighbours=num_neighbours,
                                                    c_num_to_visit=num_to_visit,
                                                    c_num_to_retrieve=num_to_retrieve,
                                                    c_prop_to_visit=prop_to_visit,
                                                    c_prop_to_retrieve=construction_prop_to_retrieve,
                                                    c_field_of_view=construction_field_of_view,
                                                    q_num_to_visit=num_to_visit,
                                                    q_field_of_view=query_field_of_view,
                                                    q_num_to_retrieve=num_to_retrieve,
                                                    q_prop_to_visit=prop_to_visit,
                                                    q_prop_to_retrieve=query_prop_to_retrieve,
                                                    transform=True,
                                                    parallel_level=self.parallel_level,
                                                    causal=True,
                                                    random=False,
                                                    do_query=False,
                                                    track=True,
                                                    update_addr=True,
                                                    changed_page_list=ccc,
                                                    )

                kvc.ccc[b][:, :prev_max_num_pages] = torch.tensor(
                    ccc, **self._cb).reshape(self.n_kv_heads, -1)
                
                dci_db = self.dci_db[cur_id]
            else:
                reuse_id = self.check_reuse(cur_id)
                dci_db = self.dci_db[reuse_id]
                kvc.ccc = self.kv_caches[reuse_id].ccc.clone()
                reuse_ccc = kvc.ccc[b][:, :dci_db.num_leaves.max()].numpy().astype(np.bool_).reshape(self.batch_size*self.n_kv_heads, -1)

            # For DCI tokens
            max_num_pages = dci_db.num_leaves.max()
            if max_num_pages > self.prev_num_pages.max():
                n_new_pages = cpu_cache.decode_alloc_n_tokens(
                    (max_num_pages - self.prev_num_pages.max()) * self.page_size)
                assert n_new_pages == max_num_pages - self.prev_num_pages.max()

            if b == 0:
                if max_num_pages > self.kvc_capacity[cur_id]:
                    prev_kvc_capacity = self.kvc_capacity[cur_id]
                    self.kvc_capacity[cur_id] = 1 << (
                        int(max_num_pages) - 1).bit_length()
                    kvc.cc2gp = utils.cat(
                        kvc.cc2gp,
                        torch.full([self.batch_size, self.n_kv_heads,
                                   self.kvc_capacity[cur_id] - prev_kvc_capacity], -1, **self._ci32),
                        dim=-1,
                    )
                    if self.check_reuse(cur_id) == 0:
                        kvc.ccc = utils.cat(
                            kvc.ccc,
                            torch.ones([self.batch_size, self.n_kv_heads,
                                    self.kvc_capacity[cur_id] - prev_kvc_capacity], **self._cb),
                            dim=-1,
                        )
                    self.page_address_buffer[cur_id] = np.concatenate(
                        [
                            self.page_address_buffer[cur_id],
                            np.full([self.batch_size, self.n_kv_heads,
                                   self.kvc_capacity[cur_id] - prev_kvc_capacity], -1, dtype=np.uintp)
                        ],
                        axis=-1,
                    )

            # Use the new allocated pages to store the DCI tokens
            new_num_leaves = dci_db.num_leaves - self.prev_num_pages
            new_address = [None] * self.n_kv_heads
            # !! Here assume batch size = 1
            offset = self.page_size * self.head_dim * self.cpu_dtype.itemsize
            new_indices = np.zeros(
                [self.n_kv_heads, new_num_leaves.max()], dtype=np.int32)
            for inst in range(self.n_kv_heads):
                if new_num_leaves[inst] == 0:
                    new_address[inst] = []
                    continue
                tmp_new_indices = np.arange(self.prev_num_pages[inst], dci_db.num_leaves[inst])
                new_indices[inst, :new_num_leaves[inst]] = tmp_new_indices
                tmp_addr = [cast(cpu_cache[b, j].data_ptr() + inst * offset, c_void_p).value for j in tmp_new_indices]
                new_address[inst] = tmp_addr
                self.page_address_buffer[cur_id][b, inst, tmp_new_indices] = np.array(tmp_addr, dtype=np.uintp)

            if self.check_reuse(cur_id) == 0:
                dci_db.address_update(indices=new_indices, new_address=new_address,
                                               num_pages=new_num_leaves, offset=self.n_kv_heads*self.page_size*self.head_dim)
            else:
                old_index, old_offset = self.prev_index, self.prev_offset
                new_index, new_offset = dci_db.token2node
                DCI.reuse_update_node(old_index=old_index, old_offset=old_offset, new_index=new_index, new_offset=new_offset, keys=_key_states, values=_value_states, new_address=self.page_address_buffer[cur_id][0], kv_offset=self.n_kv_heads*self.page_size*self.head_dim, ccc=reuse_ccc, num_leaves=dci_db.num_leaves)


    def _DCI_query(self, b, cur_id, query_states, nn_idx_override=None,
                   field_of_view_override=None):
        if self.use_dci:

            bsz = 1

            num_neighbours = self.n_dci_pages - self.layer2topk[cur_id]
            query_field_of_view = max(
                int((self.seq_len) * self.search_ratio), 30)
            if field_of_view_override is not None:
                query_field_of_view = int(field_of_view_override)
                if query_field_of_view < 1:
                    raise ValueError("field_of_view must be positive")
            query_prop_to_retrieve = 0.8

            prev_num_points = self.dci_db[cur_id].num_points[0]
            num_to_visit = prev_num_points
            num_to_retrieve = -1
            prop_to_visit = 1.0
            padding_mask = np.ones(
                [bsz, self.n_qo_heads, 1], dtype=np.bool_).reshape(-1)

            _query = query_states.reshape(-1, self.head_dim).float().numpy()

            # Use gc2cc and cc2gp for page indexing
            kvc = self.kv_caches[cur_id]

            assert (_query.flags['C_CONTIGUOUS'])

            if nn_idx_override is None:
                nn_idx, _ = self.dci_db[cur_id].query(_query,
                                                    padding_mask,
                                                    num_neighbours=num_neighbours,
                                                    field_of_view=query_field_of_view,
                                                    num_to_visit=num_to_visit,
                                                    num_to_retrieve=num_to_retrieve,
                                                    prop_to_visit=prop_to_visit,
                                                    prop_to_retrieve=query_prop_to_retrieve,
                                                    parallel_level=self.parallel_level,
                                                    ratio=self.ratio,
                                                )
            else:
                nn_idx = nn_idx_override
                if nn_idx.size != self.n_qo_heads * 2 * num_neighbours:
                    raise ValueError("native batch DCI result has the wrong size")

            nn_idx = nn_idx.reshape(self.n_qo_heads, 2, -1)
            nn_idx_0 = nn_idx[:, 0, :].reshape(self.n_kv_heads, self.ratio, -1)
            nn_idx_1 = nn_idx[:, 1, :].reshape(self.n_kv_heads, self.ratio, -1)
            if self.n_prefetch_layers > 1:
                self.nn_idx_all[b, cur_id] = nn_idx_1

            # Handle the case where there are duplicated integers in the same row of nn_idx_0
            if self.ratio > 1:
                nn_idx_0_interleaved = nn_idx_0.transpose(
                    0, 2, 1).reshape(self.n_kv_heads, -1)
                nn_idx_0 = np.vstack([utils.first_k_unique(
                    row, num_neighbours) for row in nn_idx_0_interleaved])
            else:
                nn_idx_0 = nn_idx_0.reshape(self.n_kv_heads, -1)

            nn_idx_0 = np.ascontiguousarray(nn_idx_0)
            nn_idx_1 = np.ascontiguousarray(nn_idx_1)

            padded_arrays = torch.tensor(nn_idx_0, **self._ci32)
            head_ids = torch.arange(
                self.n_kv_heads, device=padded_arrays.device).unsqueeze(1)

            if self.selected_page_idx[cur_id] is None:
                # evicted_idx, recall_idx, evict_num
                ns = kvc.n_sink_pages
                self.selected_page_idx[cur_id] = nn_idx_0
                recall_idx = torch.tensor(nn_idx_0, **self._i32)
                evicted_idx = kvc.c2p[b, torch.arange(ns, num_neighbours + ns, **self._i32)].unsqueeze(0).expand(self.n_kv_heads, -1)
                kvc.cc2gp[b, head_ids, padded_arrays] = evicted_idx.cpu()
            else:
                recall_idx, evicted_idx, out_idx = DCI.diff_pages_by_head(nn_idx_0, self.selected_page_idx[cur_id], kvc.ccc[b, head_ids, padded_arrays].numpy(), kvc.cc2gp[b].numpy())
                self.selected_page_idx[cur_id] = out_idx
                recall_idx = torch.tensor(recall_idx, **self._i32)
                evicted_idx = torch.tensor(evicted_idx, **self._i32)
            
            evict_num = (recall_idx >= 0).sum(1)
            kvc.ccc[b, head_ids, padded_arrays] = 0

            return evicted_idx.contiguous(), recall_idx.contiguous(), evict_num

    def select_pages(self, layer_idx):
        """Block-granular selection over the ordered CPU page log (policy b:
        recency).

        ``_page_log[layer][j]`` holds the j-th offloaded page in token order
        (appended at both ingestion points: prompt offload and chunk evictions).
        Recency selection is therefore a slice of log indices -- no tree query,
        no query vector.  The DCI tree's ``num_leaves`` are tree-internal nodes
        that differ per head and are NOT page ids, so the selection bypasses
        the leaf space entirely.

        Because the selection is query-independent, every decode step and every
        chunk token share the same resident set (the §6.1 chunk/tokenwise
        divergence under DCI does not exist here).  A relevance-based block
        policy can replace the slice later without touching anything else.
        """
        filled = len(self._page_log[layer_idx])
        K = self.n_dci_pages - self.layer2topk[layer_idx]
        k = min(K, filled)
        return list(range(filled - k, filled))

    def retrieve_blocks(self, layer_idx):
        """Block retrieval (policy b): select + copy CPU pages -> semantic slots.

        Replaces ``estimate_select_recall``: same return contract ``(eids, nr)``
        for ``scatter_pages``, no query vector.  The selected pages are copied
        CPU -> transit -> cast buffer with the same ``copy_to_buffer`` call
        ``recall`` uses -- addresses rebuilt directly from the CPU cache
        (``cpu_cache[b, j].data_ptr() + head * page_offset``), bypassing the
        per-head DCI leaf addressing.
        """
        kvc = self.kv_caches[layer_idx]
        ns = kvc.n_sink_pages
        K = self.n_dci_pages - self.layer2topk[layer_idx]
        page_ids = self.select_pages(layer_idx)
        k = len(page_ids)

        # skip the copy when the selection is unchanged (the common case between
        # window slides); page_valid_entries is refreshed either way
        prev = getattr(self, "_block_sel", None)
        unchanged = prev is not None and prev.get(layer_idx) == page_ids
        if not hasattr(self, "_block_sel"):
            self._block_sel = {}
        self._block_sel[layer_idx] = page_ids

        self.page_valid_entries[layer_idx][ns: ns + k] = self.page_size
        if k < K:
            self.page_valid_entries[layer_idx][ns + k: ns + K] = 0

        eids = kvc.c2p[0, ns: ns + k].unsqueeze(0).expand(
            self.n_kv_heads, -1).contiguous()
        if unchanged or k == 0:
            # slots already hold this selection's content: nr=0 makes the
            # caller's scatter_pages a no-op.  Returning the full nr here would
            # make it re-scatter from a STALE cast buffer and corrupt the slots.
            nr = torch.zeros((self.n_kv_heads,), dtype=torch.int64,
                             device=self.device)
            return eids, nr
        nr = torch.full((self.n_kv_heads,), k, dtype=torch.int64,
                        device=self.device)

        thread_id = threading.get_ident()
        if hasattr(self, '_thread_locals') and thread_id in self._thread_locals:
            c2g_stream = self._thread_locals[thread_id].c2g_stream
        else:
            c2g_stream = self.c2g_stream

        cpu_cache = self._page_log[layer_idx]
        b = 0
        head_page_offset = self.page_size * self.head_dim * self.cpu_dtype.itemsize
        n_transit_pages = k * self.n_kv_heads
        counter = 0
        for i in range(self.n_kv_heads):
            for j in page_ids:
                self._src_address_buffer[counter] = (
                    cpu_cache[j].data_ptr() + i * head_page_offset)
                counter += 1

        with torch.cuda.stream(c2g_stream):
            DCI.copy_to_buffer(
                self._src_address_buffer,
                ptr_dest=cast(self.cpu_transit_buffer[b].data_ptr(),
                              c_void_p).value,
                list_size=counter, update_num=self.page_size,
                offset_s=self.n_kv_heads * self.page_size * self.head_dim,
                offset_t=n_transit_pages * self.page_size * self.head_dim,
                dim=self.head_dim,
                page_size=self.page_size * self.head_dim, dtype=0)
            dst = self.cuda_transit_buffer[:, : 2 * n_transit_pages, :]
            src = self.cpu_transit_buffer[:, : 2 * n_transit_pages, :]
            dst.copy_(src, non_blocking=True)
            self.cuda_cast_buffer[:, : 2 * n_transit_pages, :].copy_(
                dst, non_blocking=True)
        c2g_stream.synchronize()

        return eids, nr

    def append_paged_kv_cache(self, layer_idx: int, keys: Tensor, vals: Tensor):
        kvc = self.kv_caches[layer_idx]
        kernels.append_paged_kv_cache(
            keys,
            vals,
            kvc.buffer,
            kvc.c2p,
            self.kv_indptrs_tab[self.layer2budget[layer_idx]],
            self.kv_last_page_lens,
            self.layout,
        )

    def recall(self, layer_idx: int, b: int, rids: Tensor, nr: Tensor):
        thread_id = threading.get_ident()
        if hasattr(self, '_thread_locals') and thread_id in self._thread_locals:
            thread_local = self._thread_locals[thread_id]

            c2g_stream = thread_local.c2g_stream
        else:
            # Fallback to main thread CUDA objects
            c2g_stream = self.c2g_stream

        n_transit_pages = torch.sum(nr).item()

        rids_cpu = rids.cpu()
        nr_cpu = nr.cpu()

        counter = 0
        for i in range(self.n_kv_heads):
            self._src_address_buffer[counter:counter+nr_cpu[i].item()] = self.page_address_buffer[layer_idx][b, i, rids_cpu[i, :nr_cpu[i]]]
            counter += nr_cpu[i].item()

        with torch.cuda.stream(c2g_stream):

            DCI.copy_to_buffer(self._src_address_buffer, ptr_dest=cast(self.cpu_transit_buffer[b].data_ptr(), c_void_p).value,
                               list_size=counter, update_num=self.page_size,
                               offset_s=self.n_kv_heads*self.page_size*self.head_dim,
                               offset_t=n_transit_pages*self.page_size*self.head_dim,
                               dim=self.head_dim, page_size=self.page_size*self.head_dim, dtype=0)
        ############################################################

        with torch.cuda.stream(c2g_stream):
            dst = self.cuda_transit_buffer[:, : 2 * n_transit_pages, :]
            src = self.cpu_transit_buffer[:, : 2 * n_transit_pages, :]
            dst.copy_(src, non_blocking=True)

            self.cuda_cast_buffer[:, : 2 * n_transit_pages, :].copy_(
                dst, non_blocking=True
            )

    async def estimate_select_recall_wrapper(self, layer_idx: int, query_states: Tensor):
        return await self._loop.run_in_executor(self._loop_executor, self.estimate_select_recall, layer_idx, query_states)

    def estimate_select_recall(self, layer_idx: int, query_states: Tensor):
        thread_id = threading.get_ident()
        if hasattr(self, '_thread_locals') and thread_id in self._thread_locals:
            thread_local = self._thread_locals[thread_id]

            c2g_stream = thread_local.c2g_stream
        else:
            # Fallback to main thread CUDA objects
            c2g_stream = self.c2g_stream

        eids, nr = None, None

        kvc = self.kv_caches[layer_idx]
        assert kvc.batch_size == 1

        if kvc.n_real_pages == kvc.budget and self.use_dci:
            ns = kvc.n_sink_pages
            for i in range(kvc.batch_size):

                if self.check_reuse(layer_idx) == 0:
                    eids, rids, nr = self._DCI_query(
                        i, layer_idx, query_states[i].cpu().detach().transpose(0, 1))
                
                    self.prev_nr = nr
                    self.prev_eids = eids
                    self.prev_rids = rids
                else:
                    reuse_id = self.check_reuse(layer_idx)
                    offset = self.kv_caches[layer_idx].c2p[0, 0] - self.kv_caches[reuse_id].c2p[0, 0]
                    assert ((self.kv_caches[layer_idx].c2p - self.kv_caches[reuse_id].c2p) == offset).all()
                    eids = self.prev_eids.clone()
                    mask = self.prev_eids != -1
                    eids[mask] += offset
                    nr = self.prev_nr.clone()
                    rids = self.prev_rids.clone()
                    assert eids is not None and nr is not None and rids is not None

                if eids is not None:

                    self.recall(layer_idx, i, rids, nr)

                    if self.check_reuse(layer_idx) == 0:
                        self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(
                            self.dci_db[layer_idx].get_valid_entries(self.selected_page_idx[layer_idx]), **self._i32).T
                    else:
                        self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(
                            self.dci_db[reuse_id].get_valid_entries(self.selected_page_idx[reuse_id]), **self._i32).T

                c2g_stream.synchronize()

        return eids, nr

    def scatter_pages(self, layer_idx, eids, nr):

        _cpp.scatter_pages(self.cuda_cast_buffer, 
                           self.kv_caches[layer_idx].pool.buffer,
                           eids, nr)

    def prefill_backup_pages(self, layer_idx: int):
            
        kvc = self.kv_caches[layer_idx]
        tmp_cpu_kvc = self.temp_cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        if kvc.n_real_pages > kvc.budget:
            # ns = max(2, self.n_sink_pages)
            ns = self.n_sink_pages
            nw = self.n_win_pages
            assert ns + nw <= kvc.budget

            # Calculate the number of pages to offload to DCI-CPU
            self.num_offload_pages = min((kvc.n_real_pages - kvc.budget) * self.offload_ratio,
                                         kvc.n_real_pages - ns - nw)  # !! 2 is the hyper-parameter
            self.n_dci_pages = min(
                self.num_offload_pages - (kvc.n_real_pages - kvc.budget), kvc.budget - ns - nw)
            kvc.n_win_pages = kvc.budget - self.n_dci_pages - ns

            assert (tmp_cpu_kvc.n_real_pages == 0)
            tmp_cpu_kvc.prefill_alloc_n_tokens(
                self.num_offload_pages * self.page_size)

            gpu_offset = kvc.c2p[0, ns]

            cpu_start = int(tmp_cpu_kvc.c2p[0, 0].item())
            gpu_start = int(gpu_offset)

            dst = tmp_cpu_kvc.buffer[cpu_start: cpu_start + self.num_offload_pages]
            src = kvc.buffer[gpu_start: gpu_start + self.num_offload_pages]

            torch.cuda.current_stream().wait_stream(self.default_stream)
            dst.copy_(src)
            torch.cuda.synchronize()

            current_stream = torch.cuda.current_stream()
            current_stream.synchronize()

    def decode_backup_win_page(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        win_kvc = self.offload_win_caches[layer_idx]
        if kvc.budget is None:
            return
        if kvc.n_win_pages == kvc.n_final_win_pages:
            self.num_evict_win = 1
            for i in range(kvc.batch_size):
                win_kvc[i, 0].copy_(
                    kvc[i, kvc.next_evict_idx], non_blocking=True)
        else:
            if kvc.n_win_pages - self.offload_ratio + 1 >= kvc.n_final_win_pages:
                self.num_evict_win = self.offload_ratio
            else:
                self.num_evict_win = kvc.n_win_pages - kvc.n_final_win_pages + 1

            if self.n_offloaded_win_caches < self.num_evict_win:
                win_kvc.decode_alloc_n_tokens(
                    self.page_size*(self.num_evict_win - self.n_offloaded_win_caches))
                self.n_offloaded_win_caches = self.num_evict_win
            for i in range(kvc.batch_size):
                for gi, ci in zip(win_kvc.c2p[i, :self.num_evict_win], np.arange(self.num_evict_win)):
                    win_kvc.pool[gi].copy_(
                        kvc[i, kvc.next_evict_idx+ci], non_blocking=True)

    def offload_win_page_to_DCI(self, layer_idx: int):
        win_kvc = self.offload_win_caches[layer_idx]
        kvc = self.kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for i in range(win_kvc.batch_size):
            sub_kv_states = win_kvc[i, :self.num_evict_win].permute(1, 0, 2, 3, 4).permute(
                0, 2, 1, 3, 4).reshape(2, self.n_kv_heads, -1, self.head_dim)
            self._DCI_add(i, layer_idx, sub_kv_states[0], sub_kv_states[1])


    async def prefill_evict_extra_pages_wrapper(self, layer_idx: int, query_states: Tensor, projected: Tensor):
        return await self._loop.run_in_executor(self._loop_executor, self.prefill_evict_extra_pages, layer_idx, query_states, projected)


    def prefill_evict_extra_pages(self, layer_idx: int, query_states: Tensor, projected: Tensor):
        thread_id = threading.get_ident()

        # reuse the thread local cuda stream at decode
        if hasattr(self, '_thread_locals') and thread_id in self._thread_locals:
            thread_local = self._thread_locals[thread_id]

            local_stream = thread_local.c2g_stream
        else:
            # Fallback to main thread CUDA objects
            local_stream = self.c2g_stream

        with torch.cuda.stream(local_stream):
            kvc = self.kv_caches[layer_idx]

            budget = self.layer2budget[layer_idx]

            if budget is not None:
                if kvc.n_real_pages > kvc.budget:
                    self.prefill_backup_pages(layer_idx)

            tmp_cpu_kvc = self.temp_cpu_kv_caches[layer_idx]
            bsz = kvc.batch_size
            ng = self.n_groups

            if kvc.budget is None or self.num_offload_pages == 0:
                return None
            
            if kvc.n_real_pages > kvc.budget:
                assert bsz == 1
                assert ng == 1

                local_stream.synchronize()
                
                for b in range(bsz):
                    offloaded_pages = tmp_cpu_kvc[b, :self.num_offload_pages]
                    offloaded_pages_cat = offloaded_pages.permute(1, 0, 2, 3, 4).permute(
                        0, 2, 1, 3, 4).reshape(2, self.n_kv_heads, -1, self.head_dim)
                    
                    self._DCI_first_call(b, layer_idx, query_states[b].cpu().detach().transpose(
                        # shape: [bsz, num_neighbours]
                        0, 1), offloaded_pages_cat[0], offloaded_pages_cat[1], projected)
                    
                # ordered page log: the prompt's offloaded pages, in token order
                # (must land in the log BEFORE tmp_cpu_kvc.clear() frees them)
                self._page_log[layer_idx].extend(
                    tmp_cpu_kvc[b, : self.num_offload_pages].clone())

                tmp_cpu_kvc.clear()
            else:
                self.use_dci = False

        return None

    def alloc_page(self):
        if len(self._pool._free_ids) <= 0:
            for i in range(self.n_layers):
                kvc = self.kv_caches[i]
                evt = self.prefill_backup_events[i]
                ev_gpi = self.prefill_evicted_pages[i]
                if ev_gpi is None:
                    continue
                if evt is not None:
                    self.default_stream.wait_event(evt)
                [
                    kvc.pool.free_page(pid)
                    for pid in ev_gpi.reshape(-1).tolist()
                    if pid >= 0
                ]
                self.prefill_backup_events[i] = None
                self.prefill_evicted_pages[i] = None
                break
        return self._pool.alloc_page()

    def prefill_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.prefill_handler.forward(q, kvc.buffer, page_ids.reshape(-1))

    def decode_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]

        if page_ids is None:
            page_ids = kvc.c2p
        if self.layer2budget[layer_idx] is None or self.use_dci is False:
            dci = False
            page_valid_entries = None
        else:
            dci = True
            # page_valid_entries = torch.ones_like(page_ids.reshape(-1)).repeat(self.n_kv_heads) * self.page_size - 5
            page_valid_entries = self.page_valid_entries[layer_idx][: page_ids.reshape(
                -1).shape[0]].reshape(-1)
            
        return self.decode_handler_tab[kvc.budget].forward(q, kvc.buffer, page_ids.reshape(-1), page_valid_entries=page_valid_entries, dci=dci)
