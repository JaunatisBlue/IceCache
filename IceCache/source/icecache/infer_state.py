from typing import List, Union, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
from time import time, perf_counter
from dciknn import DCI
from .page_scan import PageScan, greedy_packed_pages
from .pag_retrieval import InsufficientPagesError
from tqdm import tqdm
import copy
from ctypes import c_float, POINTER, cast, c_void_p

class DeprecatedError(NotImplementedError):
    pass


Digest = Tuple[Tensor, Tensor]


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
        retrieval_backend="dci",
        pag_ef_search=100,
        # PAG build cost is ~linear in max_search_k: measured 4.3 s @128 vs
        # 58.0 s @1024 per (layer, KV head) at N=16000 with no page-recall gain,
        # so 128 is the default.
        pag_max_search_k=128,
        pag_topm_initial_factor=4,
        pag_generation_reserve=4096,
        pag_ef_construction=200,
        pag_target_degree=16,
        pag_projection_levels=64,
        # Design A reserves the page-id space for decode up front (spec 6b, C1):
        # the CPU addresses are allocated at prefill and never grow, which is
        # what keeps the one-shot contiguity assert in _DCI_first_call true.
        #
        # page_scan_generation_reserve is the decode budget that reservation
        # must hold, in tokens per KV head. `PageScan` turns it into
        # `ceil(tokens / page_size) + DEFAULT_RESERVE_MARGIN_PAGES` pages -- the
        # margin is deliberate (D2): an earlier deployment reserved exactly
        # `ceil(4096 / 16) = 256` pages, so a generation of 4097 tokens died on
        # the flush that needed the 257th page. The pages are a hard cap: past
        # them `PageScan.insert` can only refuse the flush, it cannot grow. It
        # refuses *before* mutating, and a too-small declaration is rejected
        # here, at construction, rather than mid-generation.
        page_scan_generation_reserve=4096,
        # Batch every layer's page build into one greedy at the end of prefill
        # instead of one greedy per layer (~4.3x on a 16k prompt; see
        # _page_scan_flush). Off reproduces the per-layer behaviour, for A/B
        # timing in one process or as an escape hatch.
        page_scan_batch=True,
        # Threads for the deferred CPU page writes in _page_scan_flush. They are
        # independent per layer and numpy drops the GIL for the copy, so they
        # overlap: 5.2x on 8 threads against 2.77 GB/s serial. <=1 keeps the
        # serial loop, for A/B and as an escape hatch.
        page_scan_write_threads=8,
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
        if retrieval_backend not in ("dci", "pag_mips", "page_scan"):
            raise ValueError(f"Unknown retrieval backend: {retrieval_backend}")
        if (pag_generation_reserve < 0 or pag_max_search_k <= 0 or
                pag_ef_search <= 0 or pag_topm_initial_factor <= 0 or
                pag_ef_construction <= 0 or pag_target_degree <= 0 or
                pag_projection_levels <= 0 or pag_projection_levels % 8):
            raise ValueError("Invalid PAG search or capacity configuration")
        if page_scan_generation_reserve < 0:
            raise ValueError("Invalid page_scan generation reserve")
        if retrieval_backend == "page_scan" and page_scan_generation_reserve < page_size:
            # D2: a sub-page (or zero) declaration cannot serve a decode step --
            # with the old code it survived construction and died on the first
            # flush that ran out of page-id space.
            raise ValueError(
                "page_scan_generation_reserve must be at least one page "
                f"({page_size} decode tokens), got {page_scan_generation_reserve}")
        self.retrieval_backend = retrieval_backend
        self.page_scan_generation_reserve = page_scan_generation_reserve
        self.page_scan_batch = bool(page_scan_batch)
        self.page_scan_write_threads = int(page_scan_write_threads)
        if self.page_scan_write_threads < 0:
            raise ValueError("Invalid page_scan_write_threads")
        # Lazily built: most runs never offload anything, and a pool per
        # InferState is cheap but not free.
        self._page_scan_write_pool = None
        self.pag_config = (pag_generation_reserve, pag_max_search_k,
                           pag_ef_search, pag_topm_initial_factor,
                           pag_ef_construction, pag_target_degree, pag_projection_levels)
        self.pag_selectors = [None] * n_layers
        self.pag_disabled = set()
        self.pag_fallback_count = 0
        self.page_scans = [None] * n_layers
        # (b, layer, K, V, reuse_id) stashed during prefill by
        # _page_scan_first_call; the greedy and the CPU page write run as one
        # batch in _page_scan_flush. See that method for why they are deferred.
        self._page_scan_deferred = []
        self.query_seconds = {"dci": [], "pag_mips": [], "page_scan": []}
        self._pag_pool = None
        self._ensure_pag_pool()

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

        self._pool = KvPool(n_max_pages, page_size, n_kv_heads,
                            head_dim, dtype, device, (0, 2, 1, 3))
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
        # Decode-time staging caches, written by each reuse group's anchor
        # layer and read by that group's aliases within the same token:
        # `_recall_cpu` holds the anchor's rids/nr already copied to the host,
        # `page_valid_cache` its `get_valid_entries` result. Both mirror
        # `prev_eids`/`prev_rids`/`prev_nr` and describe the current token only
        # -- the anchor replaces them before any alias can read them.
        self._recall_cpu = {}
        self.page_valid_cache = {}
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

    def _ensure_pag_pool(self):
        """Return the one shared PAG pool, creating it on first use.

        Every layer's PagPageSelector shares this pool: pag.Index.search
        releases the GIL so the per-head searches overlap, while
        pag.Index.build holds it so index construction stays serial.
        """
        if self.retrieval_backend != "pag_mips":
            return None
        if self._pag_pool is None:
            self._pag_pool = ThreadPoolExecutor(max_workers=self.n_kv_heads)
        return self._pag_pool

    def _close_pag_pool(self):
        """Retire the shared PAG pool. Safe to call repeatedly."""
        pool = self._pag_pool
        if pool is not None:
            self._pag_pool = None
            pool.shutdown(wait=True)

    def _prepare_prefill(self, bsz, q_len):
        self.num_offload_pages = None
        self.n_dci_pages = None
        self.offload_win_flag = [False] * self.n_layers
        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.selected_page_idx = [None] * self.n_layers
        self.page_scans = [None] * self.n_layers
        self._page_scan_deferred = []
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
        # Decode-time staging caches, written by each reuse group's anchor
        # layer and read by that group's aliases within the same token:
        # `_recall_cpu` holds the anchor's rids/nr already copied to the host,
        # `page_valid_cache` its `get_valid_entries` result. Both mirror
        # `prev_eids`/`prev_rids`/`prev_nr` and describe the current token only
        # -- the anchor replaces them before any alias can read them.
        self._recall_cpu = {}
        self.page_valid_cache = {}
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
            # A new prompt starts here, so the previous prompt's selectors are
            # done (PAG queries run synchronously on the calling thread) and the
            # shared pool holds no in-flight work. Retire it; the next prefill
            # recreates it lazily in _DCI_first_call.
            self._close_pag_pool()
            for selector in self.pag_selectors:
                if selector is not None:
                    selector.close()
            self.pag_selectors = [None] * self.n_layers
            self.pag_disabled.clear()
            self.pag_fallback_count = 0
            self.page_scans = [None] * self.n_layers
            self._page_scan_deferred = []
            self.query_seconds = {"dci": [], "pag_mips": [], "page_scan": []}
            # page_scan builds its own structure in _DCI_first_call; the DCI
            # tree is not allocated at all for that backend.
            if self.retrieval_backend != "page_scan":
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
        # Before offloading below, and long before the first decode token reads a
        # page: pack every layer stashed by _page_scan_first_call in one batch.
        if self.retrieval_backend == "page_scan":
            self._page_scan_flush()
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

    # ------------------------------------------------------- page_scan (Design A)
    #
    # page_scan replaces DCI's tree build and tree retrieval with exact-kNN page
    # packing plus one matmul over page representatives (experiment/design_a_spec.md).
    # Everything below is gated on retrieval_backend == "page_scan"; the DCI path
    # above is untouched and stays runnable as the baseline arm.

    def _page_scan_write(self, b, cur_id, pages, slots, key_states, value_states):
        """Write one layer's own K/V into its CPU pages at ``(page, slot)``.

        ``pages``/``slots`` are ``int32 [H, m]``; the payloads are ``float32
        [H, m, head_dim]``. The CPU pages were allocated contiguously and never
        move (spec section 7, hazard 1), so page ``p``'s K plane for head ``h``
        lives at ``region[p, 0, h]`` of the frame allocated at prefill.
        """
        cpu_cache = self.cpu_kv_caches[cur_id]
        base = int(cpu_cache.c2p[b, 0])
        region = cpu_cache.pool.buffer.numpy()[base:]
        heads = np.arange(self.n_kv_heads)[:, None]
        region[pages, 0, heads, slots] = np.asarray(key_states)
        region[pages, 1, heads, slots] = np.asarray(value_states)

    def _page_scan_layout(self, b, cur_id, n_reserved):
        """Reserve the GPU-side bookkeeping for ``cur_id`` (mirrors DCI's part).

        Returns the CPU base address of the layer's page frame. All ``n_reserved``
        pages are allocated and addressed now, so a page emitted at decode already
        has a valid CPU address (spec section 6b, C1).
        """
        stride = self.cpu_n_bytes_per_page
        cpu_cache = self.cpu_kv_caches[cur_id]
        kvc = self.kv_caches[cur_id]
        cpu_cache.prefill_alloc_n_tokens(n_reserved * self.page_size)
        _base = cpu_cache[b, 0].data_ptr()
        assert cpu_cache[b, -1].data_ptr() - _base == (n_reserved - 1) * stride

        # For offloading (unchanged from DCI)
        ns = kvc.n_sink_pages
        ev_gpi = kvc.c2p.clone()
        ev_gpi[:, :ns] = -1
        ev_gpi[:, -kvc.budget + ns:] = -1
        self.prefill_evicted_pages[cur_id] = ev_gpi

        kvc.c2p = torch.cat(
            [kvc.c2p[:, :ns], kvc.c2p[:, -kvc.budget + ns:]], dim=-1)

        self.kvc_capacity[cur_id] = 1 << (int(n_reserved) - 1).bit_length()
        kvc.cc2gp = torch.full(
            [kvc.batch_size, self.n_kv_heads, self.kvc_capacity[cur_id]], -1, **self._ci32)
        kvc.ccc = torch.ones(
            [kvc.batch_size, self.n_kv_heads, self.kvc_capacity[cur_id]], **self._cb)
        self.page_address_buffer[cur_id] = np.full(
            [kvc.batch_size, self.n_kv_heads, self.kvc_capacity[cur_id]], -1, dtype=np.uintp)

        offset = self.page_size * self.head_dim * self.cpu_dtype.itemsize
        # One vector add per head, not n_reserved: the address of page j for head
        # i is just base + i*offset + j*stride, so the whole row is one numpy
        # add. Building it as a Python list of ctypes ints instead measured
        # 425 ms/row of TTFT (68 calls, ~10k ctypes calls each) for identical
        # values.
        page_strides = np.arange(n_reserved, dtype=np.uintp) * np.uintp(stride)
        for i in range(self.n_kv_heads):
            self.page_address_buffer[cur_id][b, i, :n_reserved] = (
                np.uintp(_base + i * offset) + page_strides)
        return _base

    def _page_scan_first_call(self, b, cur_id, key_states, value_states):
        """Design A prefill: the layer's CPU page layout, and a deferred build.

        The *layout* runs here and cannot move: it truncates ``kvc.c2p`` to
        sink+window, and ``prefill_sdpa`` reads ``kvc.c2p`` for this layer
        immediately after this call (``modeling.py:150``, with ``page_ids=None``).
        Deferring it would feed every later layer an untruncated ``c2p`` and
        silently change attention.

        The *greedy* and the CPU page write are deferred to
        :meth:`_page_scan_flush`, which runs them for every layer as one batch.
        Nothing in the prefill forward reads a page partition or the CPU pages it
        fills -- the first reader is decode-time ``_page_scan_add``.
        """
        H = self.n_kv_heads
        reuse_id = self.check_reuse(cur_id)
        k = key_states.reshape(H, -1, self.head_dim)
        v = value_states.reshape(H, -1, self.head_dim)
        if reuse_id == 0:
            # n_pages_reserved, not n_pages_built: the extra pages are address
            # space for decode-time page emission (spec section 6b, C1). The
            # count comes from the declared generation budget plus PageScan's
            # margin, and PageScan refuses a declaration its reservation cannot
            # hold -- D2's loud failure, before a single decode token is served.
            scan = PageScan(H, self.head_dim, self.page_size, device=self.device,
                            generation_reserve_tokens=self.page_scan_generation_reserve)
            self.page_scans[cur_id] = scan
            self._page_scan_layout(b, cur_id, scan.reserve_address_space(k.shape[1]))
        else:
            # A reuse layer aliases the source's partition and lays its own K/V
            # out the same way; it must not build (spec section 5, hazard 5).
            scan = self.page_scans[reuse_id]
            if scan is None:
                raise RuntimeError(f"page_scan source layer {reuse_id} has no structure")
            _base = self._page_scan_layout(b, cur_id, scan.n_pages)
            layer_offset = _base - self.cpu_kv_caches[reuse_id][b, 0].data_ptr()
            for i in range(self.n_kv_heads):
                tmp_addr = self.page_address_buffer[reuse_id][b, i, :scan.n_pages] + layer_offset
                self.page_address_buffer[cur_id][b, i, :scan.n_pages] = np.array(
                    tmp_addr, dtype=np.uintp)
        # clone(): the source is a temp_cpu_kv_cache that its caller frees as soon
        # as this returns (infer_state.py:1434), but the write waits for the flush.
        self._page_scan_deferred.append((b, cur_id, k.clone(), v.clone(), reuse_id))
        if not self.page_scan_batch:
            # One layer per flush: the pre-batching behaviour. Kept switchable so
            # the TTFT delta can be measured against it in the same process, and
            # so a batch that ever misbehaved has a way out. Both paths use the
            # same greedy (use_sim=False), which isolates batching itself.
            self._page_scan_flush()

    def _page_scan_flush(self):
        """Run every deferred greedy as ONE batch, then write each layer's pages.

        This is the whole TTFT fix. The greedy's per-page ops (``argmax``,
        ``masked_fill``, ``topk``, both scatters) are launch-latency-bound, not
        throughput-bound, so they cost about the same at M=8 as at M=96. The row
        bmm is not in that class -- it is bandwidth-bound and does scale with M
        -- so the loop as a whole is not head-independent. Measured on synthetic
        keys at the real shape (M=96, N=16072, page_size 16, use_sim=False,
        median of 3): 96 heads cost 796.5 ms against 291.8 ms for one 8-head
        layer, i.e. 2.73x in total and 0.227x per head. An earlier version of
        this comment read "1.04x in total, 0.09x per head", which generalised
        the latency-bound ops to the whole loop and conflated 12x heads with
        12x layers; it was wrong and is not worth inheriting. The build still
        goes from 3408 ms to 798 ms on a 16k prompt -- 12 x 291.8 = 3502 ms
        unbatched, a 4.4x win -- which is essentially the entire TTFT gap
        against DCI (3.0 s).

        Batching is bit-exact against per-layer builds: every op in the loop is
        row-independent, and measured on real keys packing 16 heads at once vs
        8+8 gave 0/188320 differing entries.
        """
        pending = self._page_scan_deferred
        if not pending:
            return
        self._page_scan_deferred = []
        H = self.n_kv_heads
        builders = [i for i, p in enumerate(pending) if p[4] == 0]
        if builders:
            # One greedy for every builder layer. use_sim=False because the
            # materialised [M, N, N] similarity would be 99 GB at 96 heads; the
            # row-on-demand path is still 4.27x ahead of building them serially.
            #
            # One greedy packs them all at a shared token count, so unequal
            # lengths would silently mis-slice the result. Every layer offloads
            # the same prompt under the same budget, so this holds -- but assert
            # it rather than trust it, since the failure is a wrong partition,
            # not a crash.
            lengths = {int(pending[i][2].shape[1]) for i in builders}
            if len(lengths) != 1:
                raise RuntimeError(
                    f"page_scan batch needs one token count across layers, got {sorted(lengths)}")
            keys = torch.cat([pending[i][2] for i in builders], dim=0).to(self.device)
            packed = greedy_packed_pages(keys, self.page_size, use_sim=False)
            for n, i in enumerate(builders):
                cur_id = pending[i][1]
                scan = self.page_scans[cur_id]
                # The GPU slice goes back in so the keys are not sent over PCIe
                # a second time; the CPU copy stays for the write below.
                scan.build_from_packed(
                    packed[n * H:(n + 1) * H], keys[n * H:(n + 1) * H],
                    n_reserved=scan.n_pages)
            del keys, packed
        # Writes run after every partition exists, so the reuse layers -- which
        # write their own K/V at their source's (page, slot) -- read a partition
        # that is already complete.
        #
        # They are independent -- each entry fills its own layer's CPU pool at a
        # partition that is already final -- and numpy releases the GIL for the
        # copy, so they run on a small pool. Serial this loop sustains only
        # 2.77 GB/s (it is a scattered 4-D fancy-index store, not a memcpy) and
        # cost 1070 ms/row of TTFT; 8 threads measured 5.2x on the same work.
        if self.page_scan_write_threads > 1 and len(pending) > 1:
            if self._page_scan_write_pool is None:
                self._page_scan_write_pool = ThreadPoolExecutor(
                    max_workers=min(self.page_scan_write_threads, len(pending)))
            # list() so an exception in a worker surfaces here, not at GC.
            list(self._page_scan_write_pool.map(self._page_scan_write_entry, pending))
        else:
            for entry in pending:
                self._page_scan_write_entry(entry)

    def _page_scan_write_entry(self, entry):
        """Write one stashed layer's K/V into its CPU pages (flush worker body)."""
        b, cur_id, k, v, reuse_id = entry
        scan = self.page_scans[cur_id if reuse_id == 0 else reuse_id]
        self._page_scan_write(b, cur_id, scan.token2page, scan.offset_in_page, k, v)

    def _page_scan_add(self, b, cur_id, key_states, value_states):
        """Design A decode insert (mirrors DCI's reuse gating exactly, spec 6b C4)."""
        H = self.n_kv_heads
        m = key_states.shape[1]
        k_np = key_states.reshape(H, m, self.head_dim).float().numpy()
        v_np = value_states.reshape(H, m, self.head_dim).float().numpy()
        reuse_id = self.check_reuse(cur_id)
        if reuse_id == 0:
            scan = self.page_scans[cur_id]
            if scan is None:
                raise RuntimeError(f"page_scan layer {cur_id} has no structure")
            pages, slots = scan.insert(k_np)
            # Pages that gained tokens are stale in the GPU copy and must be
            # re-recalled if selected (DCI's changed_page_list).
            kvc = self.kv_caches[cur_id]
            kvc.ccc[b, torch.arange(H).unsqueeze(1), torch.as_tensor(pages)] = True
        else:
            # The reuse layer reads the source's assignment instead of computing
            # its own: its keys differ, so recomputing would silently diverge.
            scan = self.page_scans[reuse_id]
            if scan is None or scan.last_insert is None:
                raise RuntimeError(f"page_scan source layer {reuse_id} has not inserted")
            pages, slots = scan.last_insert
        self._page_scan_write(b, cur_id, pages, slots, k_np, v_np)

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

            if self.retrieval_backend == "page_scan":
                self._page_scan_first_call(b, cur_id, key_states, value_states)
                return

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
                if self.retrieval_backend == "pag_mips":
                    try:
                        from .pag_retrieval import PagPageSelector
                        mapping = np.asarray(dci_db.token2node[0], dtype=np.int32).reshape(self.n_kv_heads, -1)[:, :dci_len].copy()
                        self.pag_selectors[cur_id] = PagPageSelector(
                            cur_id, key_states.float().numpy(), mapping,
                            *self.pag_config, executor=self._ensure_pag_pool())
                    except Exception as exc:
                        self.pag_disabled.add(cur_id)
                        print(f"PAG build failed on layer {cur_id}; using DCI: {exc}")
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
            if self.retrieval_backend == "page_scan":
                self._page_scan_add(b, cur_id, key_states, value_states)
                return

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
                if self.pag_selectors[cur_id] is not None and cur_id not in self.pag_disabled:
                    try:
                        mapping = np.asarray(dci_db.token2node[0], dtype=np.int32).reshape(self.n_kv_heads, -1)[:, :self.pag_selectors[cur_id].count + dci_len].copy()
                        self.pag_selectors[cur_id].insert(key_states.float().numpy(), mapping)
                    except Exception as exc:
                        self.pag_disabled.add(cur_id)
                        print(f"PAG insert failed on layer {cur_id}; using DCI: {exc}")
            else:
                old_index, old_offset = self.prev_index, self.prev_offset
                new_index, new_offset = dci_db.token2node
                DCI.reuse_update_node(old_index=old_index, old_offset=old_offset, new_index=new_index, new_offset=new_offset, keys=_key_states, values=_value_states, new_address=self.page_address_buffer[cur_id][0], kv_offset=self.n_kv_heads*self.page_size*self.head_dim, ccc=reuse_ccc, num_leaves=dci_db.num_leaves)


    def _DCI_query(self, b, cur_id, query_states):
        if self.use_dci:

            bsz = 1

            num_neighbours = self.n_dci_pages - self.layer2topk[cur_id]

            if self.retrieval_backend == "page_scan":
                # num_neighbours is per-layer, exactly as DCI computes it: the
                # shape assert in _apply_selected_pages uses this same
                # expression (spec section 6b, C3). No DCI state is touched.
                query_start = perf_counter()
                # Pass the tensor through: PageScan's device scan copies it to
                # the GPU itself, so a numpy() conversion here would be a no-op
                # view on an already-CPU tensor -- it buys nothing and is not
                # what the scan's cost consists of. The real transfers are the
                # host->device copy of `q` and the device->host copy of the
                # result, and the latter synchronises the stream, so
                # query_seconds["page_scan"] includes whatever was already
                # queued. (An earlier comment here cited "1.91 vs 0.61 ms" for
                # this conversion; that measurement is not reproducible from
                # this call path, because `query_states` arrives here already
                # on the CPU -- see the `.cpu()` at the _DCI_query call site.)
                _query = query_states.reshape(-1, self.head_dim).float()
                page_ids = self.page_scans[cur_id].query(_query, num_neighbours)
                self.query_seconds["page_scan"].append(perf_counter() - query_start)
                return self._apply_selected_pages(b, cur_id, page_ids)

            query_field_of_view = max(
                int((self.seq_len) * self.search_ratio), 30)
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
            query_start = perf_counter()

            # Gate the PAG path on n_prefetch_layers <= 1: with deeper layer
            # prefetching this query runs on the asyncio worker thread for a
            # different layer ahead of time, so keep PAG on the simple
            # synchronous path only.
            if (self.retrieval_backend == "pag_mips" and self.n_prefetch_layers <= 1
                    and cur_id not in self.pag_disabled and self.pag_selectors[cur_id] is not None):
                try:
                    page_ids = self.pag_selectors[cur_id].select(
                        _query, num_neighbours, self.page_size)
                except InsufficientPagesError:
                    self.pag_fallback_count += 1
                except Exception as exc:
                    self.pag_disabled.add(cur_id)
                    self.pag_fallback_count += 1
                    print(f"PAG query failed on layer {cur_id}; using DCI: {exc}")
                else:
                    self.query_seconds["pag_mips"].append(perf_counter() - query_start)
                    return self._apply_selected_pages(b, cur_id, page_ids)

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

            self.query_seconds["dci"].append(perf_counter() - query_start)
            return self._apply_selected_pages(b, cur_id, nn_idx_0)

    def _apply_selected_pages(self, b, cur_id, nn_idx_0):
        kvc = self.kv_caches[cur_id]
        num_neighbours = self.n_dci_pages - self.layer2topk[cur_id]
        if nn_idx_0.shape != (self.n_kv_heads, num_neighbours):
            raise ValueError('Page selector returned the wrong shape')
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

    def retrieval_stats(self):
        def latency(values):
            if not values:
                return {"count": 0}
            ms = np.asarray(values, dtype=np.float64) * 1000
            return {"count": len(values), "p50_ms": float(np.percentile(ms, 50)),
                    "p95_ms": float(np.percentile(ms, 95)),
                    "p99_ms": float(np.percentile(ms, 99))}

        selectors = {}
        for layer, selector in enumerate(self.pag_selectors):
            if selector is not None:
                selectors[layer] = {
                    "build_ms": selector.build_seconds * 1000,
                    "queries": selector.query_count,
                    "retries": selector.retry_count,
                    "shortfalls": selector.shortfall_count,
                    "inserted_tokens": selector.insert_count,
                    "search": latency(selector.search_seconds),
                    "aggregate": latency(selector.aggregate_seconds),
                    "insert": latency(selector.insert_seconds),
                }
        scans = {}
        for layer, scan in enumerate(self.page_scans):
            if scan is not None:
                scans[layer] = {
                    "build_ms": scan.build_seconds * 1000,
                    "n_built": [int(n) for n in scan.n_built],
                    "n_pages_reserved": int(scan.n_pages),
                    "queries": len(scan.query_seconds),
                    "inserts": len(scan.insert_seconds),
                    "query": latency(scan.query_seconds),
                    "insert": latency(scan.insert_seconds),
                }
        return {"configured_backend": self.retrieval_backend,
                "disabled_layers": sorted(self.pag_disabled),
                "fallback_queries": self.pag_fallback_count,
                "query": {name: latency(values) for name, values in self.query_seconds.items()},
                "pag_layers": selectors,
                "page_scan_layers": scans}

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

    def recall(self, layer_idx: int, b: int, rids: Tensor, nr: Tensor, source: int = 0):
        """Stage one layer's recalled pages into ``cpu_transit_buffer``.

        ``source`` is the layer whose retrieval produced ``rids``/``nr``: 0 for
        an anchor layer, and the anchor's layer id for an alias. Alias layers
        receive the anchor's arrays *cloned verbatim* (``estimate_select_recall``
        below), so the two D2H copies and the element count are identical across
        every call of a reuse group -- they are computed once at the anchor and
        read back here. That also removes the ``torch.sum(nr).item()`` device
        sync, whose value is exactly the CPU element count ``counter``, and the
        reader should note the two were already required to agree: the old code
        used the synced GPU sum for ``offset_t`` and the CPU count for
        ``list_size``.

        The address gather itself cannot be shared: ``page_address_buffer`` is
        per layer, so an alias's addresses genuinely differ from its anchor's.
        It is one vectorised numpy gather rather than a per-head Python loop --
        measured 369 us -> 31 us per call at H=8, since the loop's cost was
        numpy fancy-indexing overhead, not the 80 elements it copies.
        """
        thread_id = threading.get_ident()
        if hasattr(self, '_thread_locals') and thread_id in self._thread_locals:
            thread_local = self._thread_locals[thread_id]

            c2g_stream = thread_local.c2g_stream
        else:
            # Fallback to main thread CUDA objects
            c2g_stream = self.c2g_stream

        if source:
            rids_cpu, nr_cpu, counter = self._recall_cpu[source]
        else:
            rids_cpu = rids.cpu().numpy()
            nr_cpu = nr.cpu().numpy()
            counter = int(nr_cpu.sum())
            self._recall_cpu[layer_idx] = (rids_cpu, nr_cpu, counter)
        n_transit_pages = counter

        max_n = int(nr_cpu.max())
        if max_n > 0:
            # Head i contributes rids_cpu[i, :nr_cpu[i]], in head order. Gather
            # the full [H, max_n] block, then drop the columns past each head's
            # count -- masking them to 0 first, because the unused tail of a
            # rids row holds whatever the selector left there and would
            # otherwise index out of range.
            keep = np.arange(max_n)[None, :] < nr_cpu[:, None]
            flat = self.page_address_buffer[layer_idx][b][
                np.arange(self.n_kv_heads)[:, None], np.where(keep, rids_cpu[:, :max_n], 0)][keep]
            self._src_address_buffer[:counter] = flat

        with torch.cuda.stream(c2g_stream):

            DCI.copy_to_buffer(self._src_address_buffer, ptr_dest=cast(self.cpu_transit_buffer[b].data_ptr(), c_void_p).value,
                               list_size=counter, update_num=self.page_size,
                               offset_s=self.n_kv_heads*self.page_size*self.head_dim,
                               offset_t=n_transit_pages*self.page_size*self.head_dim,
                               dim=self.head_dim, page_size=self.page_size*self.head_dim, dtype=0)
        ############################################################

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
            reuse_id = self.check_reuse(layer_idx)
            for i in range(kvc.batch_size):

                if reuse_id == 0:
                    eids, rids, nr = self._DCI_query(
                        i, layer_idx, query_states[i].cpu().detach().transpose(0, 1))

                    self.prev_nr = nr
                    self.prev_eids = eids
                    self.prev_rids = rids
                else:
                    # Both the offset and the invariant it is derived from are
                    # re-read every token, deliberately. Caching them once per
                    # row is WRONG and changes the generated text: measured on
                    # hotpotqa row 1, a cached offset turned "Charles Laughton"
                    # into "Charles Dickens". `c2p[0, 0]` does move during decode
                    # (for the anchor and its aliases together, which is why the
                    # whole-row invariant below still holds either way), so the
                    # alias's page ids depend on the live value.
                    offset = self.kv_caches[layer_idx].c2p[0, 0] - self.kv_caches[reuse_id].c2p[0, 0]
                    assert ((self.kv_caches[layer_idx].c2p - self.kv_caches[reuse_id].c2p) == offset).all()
                    # where() instead of clone + masked index_add_: same values,
                    # two launches instead of four, no index tensors.
                    eids = torch.where(self.prev_eids != -1,
                                       self.prev_eids + offset, self.prev_eids)
                    nr = self.prev_nr.clone()
                    rids = self.prev_rids.clone()
                    assert eids is not None and nr is not None and rids is not None

                if eids is not None:

                    self.recall(layer_idx, i, rids, nr, reuse_id)

                    # `get_valid_entries` is a per-(layer, token) function of the
                    # selection, and an alias layer reads its anchor's selection
                    # (`selected_page_idx[reuse_id]`), so the value an alias needs
                    # is the one the anchor already computed this token. Anchors
                    # always recompute, which is what keeps this fresh; the
                    # difference is that an alias now copies its anchor's result
                    # device-to-device instead of re-running the selector and
                    # making a fresh pageable host->device transfer of it.
                    if reuse_id == 0:
                        if self.retrieval_backend == "page_scan":
                            # True per-(page, head) occupancy, never -1 (spec 6b, C2/C5).
                            entries = self.page_scans[layer_idx].get_valid_entries(self.selected_page_idx[layer_idx])
                        else:
                            entries = self.dci_db[layer_idx].get_valid_entries(self.selected_page_idx[layer_idx])
                        entries = torch.tensor(entries, **self._i32)
                        self.page_valid_cache[layer_idx] = entries
                    else:
                        entries = self.page_valid_cache[reuse_id]

                    self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = entries.T

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
