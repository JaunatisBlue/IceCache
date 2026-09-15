from typing import List, Union, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import atexit
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
try:
    from dciknn._dci import _dci_first_k_unique_by_head
except ImportError:
    _dci_first_k_unique_by_head = None
try:
    from dciknn._dci import _dci_copy_to_buffer_batched
except ImportError:
    _dci_copy_to_buffer_batched = None
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
        profile_dci=False,
        profile_warmup_tokens=0,
        group_size=None,
        n_groups=None,
        debug=False,
        ratio_1=0.01,
        ratio_2=0.2,
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
        # Optional layerwise M-DCI tree schedule.  Layers before the boundary
        # retain ratio_1; layers at/after it use the alternate promotion rate.
        # Disabled by default so existing runs are unchanged.
        self.promotion_fast_start_layer = int(os.environ.get(
            "ICECACHE_PROMOTION_FAST_START_LAYER", "-1"))
        self.promotion_fast_ratio = float(os.environ.get(
            "ICECACHE_PROMOTION_FAST_RATIO", str(ratio_1)))

        self.dtype = dtype
        self.device = device
        self.cpu_dtype = torch.float32
        # Requires the matching M-DCI dtype=2 gather extension.  It converts
        # CPU-resident FP32 KV pages into an FP16 pinned staging buffer before
        # H2D, halving PCIe traffic while keeping the DCI index in FP32.
        self.fp16_recall = bool(int(
            os.environ.get("ICECACHE_FP16_RECALL", "0")))
        if self.fp16_recall and self.dtype is not torch.float16:
            raise ValueError(
                "ICECACHE_FP16_RECALL=1 currently requires torch.float16 KV"
            )
        self.recall_dtype = self.dtype if self.fp16_recall else self.cpu_dtype
        if self.fp16_recall:
            self._validate_fp16_recall_backend()

        self.offload_ratio: int = 2  # >= 1
        self.search_ratio: float = 1e-3
        self.debug = debug
        self.use_dci = True
        # DCI's native parallel query can oversubscribe OpenMP on very long
        # contexts.  Keep the original default, while allowing launchers to
        # disable nested parallelism for stable long-running benchmarks.
        self.parallel_level = int(os.environ.get("ICECACHE_DCI_PARALLEL_LEVEL", "2"))

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
        # Experiment 10: layer-sensitivity skip list.  A comma-separated set of
        # anchor layer indices; those layers reuse the previous selection
        # (prev_eids) instead of running DCI+recall, to measure which layers
        # can be skipped without quality loss.  Empty by default (no-op).
        _skip_env = os.environ.get("ICECACHE_SKIP_DCI_LAYERS", "")
        self.skip_dci_layers = set(
            int(x) for x in _skip_env.split(",") if x.strip())
        self._in_estimate = False
        self._decode_phase = False
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
        self._recall_fp = dict(
            dtype=self.recall_dtype, device=torch.device("cpu"))
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
        # Page IDs are reused, but every layer still owns different KV data.
        # This opt-in path gathers a whole reuse group in one OpenMP region
        # and submits one larger H2D copy.
        self.batch_layer_recall = bool(int(
            os.environ.get("ICECACHE_BATCH_LAYER_RECALL", "0")))
        if self.batch_layer_recall:
            if self.n_reuse_layers <= 1:
                raise ValueError(
                    "ICECACHE_BATCH_LAYER_RECALL=1 requires n_reuse_layers > 1")
            if not self.fp16_recall:
                raise ValueError(
                    "ICECACHE_BATCH_LAYER_RECALL=1 currently requires "
                    "ICECACHE_FP16_RECALL=1")
            if self.n_prefetch_layers:
                raise ValueError(
                    "ICECACHE_BATCH_LAYER_RECALL=1 is incompatible with "
                    "layer prefetch in this prototype")
            if _dci_copy_to_buffer_batched is None:
                raise RuntimeError(
                    "ICECACHE_BATCH_LAYER_RECALL=1 requires the matching "
                    "M-DCI batched-gather extension")
        self._batched_recall_slices = {}
        # CUDA events alone would miss the CPU-resident DCI search.  This
        # wall-clock profiling path is deliberately opt-in.
        self.profile_dci = profile_dci
        self.profile_warmup_tokens = profile_warmup_tokens
        self.profile_decode_steps = 0
        self.profile_sequence_decode_steps = 0
        self.profile_measured_steps = 0
        self.profile_dci_calls = 0
        self.profile_dci_seconds = 0.0
        self.profile_decode_seconds = 0.0
        # Coarse system-boundary timers.  These deliberately avoid timing
        # individual M-DCI tree operations and instead expose where decode
        # crosses between GPU, CPU search, page metadata, and KV transfer.
        self.profile_query_d2h_seconds = 0.0
        self.profile_native_query_seconds = 0.0
        self.profile_native_query_seconds_by_layer = defaultdict(float)
        self.profile_native_query_calls_by_layer = defaultdict(int)
        self.profile_query_postprocess_seconds = 0.0
        self.profile_query_dedup_seconds = 0.0
        self.profile_query_mapping_seconds = 0.0
        self.profile_query_diff_seconds = 0.0
        self.profile_recall_gather_seconds = 0.0
        self.profile_recall_wait_seconds = 0.0
        self.profile_page_metadata_seconds = 0.0
        self.profile_index_update_seconds = 0.0
        # Fine-grained decode-side incremental DCI insertion timers.  These
        # are opt-in with profile_dci and split the coarse index_update timer.
        self.profile_index_pack_seconds = 0.0
        self.profile_index_numpy_seconds = 0.0
        self.profile_index_prepare_seconds = 0.0
        self.profile_index_native_insert_seconds = 0.0
        self.profile_index_ccc_writeback_seconds = 0.0
        self.profile_index_page_alloc_seconds = 0.0
        self.profile_index_address_update_seconds = 0.0
        self.profile_index_address_prepare_seconds = 0.0
        self.profile_index_native_address_update_seconds = 0.0
        self.profile_index_reuse_update_seconds = 0.0
        # Step-1 subdivision of index_address_prepare: metadata prep, the
        # per-leaf data_ptr() resolution loop, the list->ndarray conversion and
        # the page_address_buffer write.  Only touched when profile_dci=True.
        self.profile_index_addrprep_meta_seconds = 0.0
        self.profile_index_addrprep_leaf_seconds = 0.0
        self.profile_index_addrprep_np_seconds = 0.0
        self.profile_index_addrprep_write_seconds = 0.0
        # One record per layer-level index update, so the report can bucket
        # cost by tree size / anchor-vs-reuse instead of averaging it away.
        self.profile_index_call_records = []
        self.profile_index_call_records_cap = 4000
        # Optional raw dump of the per-call records (JSON), for fitting
        # T_insert(prev_num_points, ...) offline.
        self.profile_call_dump = os.environ.get(
            "ICECACHE_PROFILE_CALL_DUMP", "")
        if self.profile_call_dump:
            atexit.register(self._dump_index_call_records)
        self.profile_recall_calls = 0
        self.profile_recall_submissions = 0
        self.profile_recall_pages = 0
        self.profile_index_update_calls = 0
        self.profile_cross_token_reuses = 0
        self.profile_cross_token_reuse_calls = 0
        self.profile_cross_token_boundary_refreshes = 0
        self._profile_decode_start = None
        # [ICECACHE-PROFILE] Optional per-decode-step latency samples, so an
        # A/B can report TPOT spread (std / p50 / p95) rather than only a mean.
        # Default off: `profile_step_samples` stays None and costs one compare.
        self.profile_step_samples = (
            [] if bool(int(os.environ.get(
                "ICECACHE_PROFILE_STEP_SAMPLES", "0"))) else None)
        self.profile_step_samples_cap = int(os.environ.get(
            "ICECACHE_PROFILE_STEP_SAMPLES_CAP", "50000"))

        # === ICECACHE_DIAG: joint diagnostic (timing split + address mergeability) ===
        # Pure-additive, opt-in.  When ICECACHE_DIAG is unset this whole block
        # costs two bool checks per recall and changes no numerical behaviour.
        self.diag_enabled = bool(int(os.environ.get("ICECACHE_DIAG", "0")))
        self.diag_dump_path = os.environ.get("ICECACHE_DIAG_DUMP", "")
        self.diag_max_records = int(
            os.environ.get("ICECACHE_DIAG_MAX_RECORDS", "400"))
        # Bound the number of CUDA events created so a long run cannot
        # accumulate unbounded event objects.
        self.diag_event_budget = int(
            os.environ.get("ICECACHE_DIAG_MAX_EVENTS", "800"))
        self.diag_addr_prep_seconds = 0.0
        self.diag_copy_buffer_seconds = 0.0
        self.diag_h2d_ms = 0.0
        self.diag_cast_ms = 0.0
        self.diag_h2d_count = 0
        self.diag_cast_count = 0
        self.diag_records = []
        self.diag_pending = []
        self.diag_saved = False
        # [ICECACHE-FASTADDR] vectorised source-address construction (A/B gate)
        self.fast_addr = bool(int(os.environ.get("ICECACHE_FAST_ADDR", "1")))
        # [ICECACHE-VECADDR] decode-side incremental DCI address preparation.
        # `vec_addr` -> replace the per-leaf `data_ptr()` loop with NumPy
        #                integer arithmetic over `c2p`, then hand the result to
        #                the M-DCI binding as a plain Python list.
        #                (Feeding the binding a NumPy array instead of a list
        #                segfaults it -- see docs/DeepSeek_..._results.md.)
        #                DEFAULT ON since 2026-09-14: the addresses are
        #                bit-identical to the old path (unit + in-vivo
        #                element-wise checks), 20-sample Qasper F1 is
        #                unchanged (45.44 vs 45.48), and it removes ~71% of
        #                `index_address_prepare`.  Set ICECACHE_VEC_ADDR=0 to
        #                get the old loop back.
        # `addr_equiv_check` -> read-only in-vivo equivalence assertion: the
        #                vectorised formula must reproduce the per-leaf
        #                `data_ptr()` result element for element.
        self.vec_addr = bool(int(os.environ.get("ICECACHE_VEC_ADDR", "1")))
        self.addr_equiv_check = bool(
            int(os.environ.get("ICECACHE_ADDR_EQUIV_CHECK", "0")))
        self.addr_equiv_checks = 0
        self.addr_equiv_mismatch = 0
        self.addr_equiv_logical_mismatch = 0
        self.addr_equiv_by_layer = {}
        # Distinct physical page ids seen per (layer, head) -> proves whether
        # the CPU pool is actually fragmented during decode.
        self.addr_equiv_phys = set()
        if self.diag_enabled:
            atexit.register(self._save_diag)
        self.nn_idx_all = None
        self.attn_layers = [None] * n_layers

        # Optional, read-only DCI selection diagnostic.  It never changes the
        # selected pages; it only compares consecutive decode selections from
        # the same anchor layer and KV head.
        self.trace_dci_churn = bool(int(
            os.environ.get("ICECACHE_TRACE_DCI_CHURN", "0")))
        self._trace_prev_selection = [None] * n_layers
        self._trace_overlap = []
        self._trace_top25_overlap = []
        self._trace_top50_overlap = []
        self._trace_exact = []
        self._trace_by_layer = defaultdict(list)
        self._trace_by_head = defaultdict(list)

        # Oracle-only feasibility trace for query-adaptive early termination.
        # Partial-budget queries are compared with the unchanged full query;
        # only the full result is used by inference.
        self.trace_dci_adaptive = bool(int(
            os.environ.get("ICECACHE_TRACE_DCI_ADAPTIVE", "0")))
        self.trace_dci_adaptive_levels = tuple(
            float(x) for x in os.environ.get(
                "ICECACHE_TRACE_DCI_LEVELS", "0.125,0.25,0.5,1.0"
            ).split(",")
        )
        self.trace_dci_adaptive_thresholds = tuple(
            float(x) for x in os.environ.get(
                "ICECACHE_TRACE_DCI_THRESHOLDS", "0.8,0.9,0.95"
            ).split(",")
        )
        self._adaptive_fixed_recall = defaultdict(list)
        self._adaptive_stop_fraction = defaultdict(list)
        self._adaptive_oracle_recall = defaultdict(list)
        self._adaptive_stage_seconds = defaultdict(float)
        self._adaptive_stage_calls = defaultdict(int)

        # Decode-path DCI visit budget.  M-DCI caps the number of projected
        # points it expands per level with
        #   num_projs_to_visit = max(num_to_visit*num_simp,
        #                            prop_to_visit*num_points*num_simp)
        # IceCache historically pins prop_to_visit=1.0 (visit everything),
        # which makes M-DCI's budget parameter dead.  Exposing it here lets
        # us unlock partial-budget search (e.g. 0.25 = visit 25% of points).
        self.dci_prop_to_visit = float(
            os.environ.get("ICECACHE_DCI_PROP_TO_VISIT", "1.0"))
        assert 0.0 < self.dci_prop_to_visit <= 1.0

        # Cross-token event-driven DCI gate.  When the layer's query vector
        # hasn't moved much since the last DCI call, we skip the search and
        # reuse the previously-selected page set.  Four safety valves keep
        # this safe: max-consecutive-reuse, force-refresh cadence, cosine
        # threshold, and a structural-boundary override.  This remains an
        # experimental, opt-in path because it regresses long-form QA.
        self.enable_cross_token_dci = bool(int(
            os.environ.get("ICECACHE_CROSS_TOKEN_DCI", "0")))
        # 1 - cos(q_t, q_{t-1}) below this triggers a reuse.
        self.cosine_reuse_threshold = float(
            os.environ.get("ICECACHE_CROSS_TOKEN_COS", "0.05"))
        # Stop reusing after this many consecutive same-layer skips.
        self.max_consec_reuse = int(
            os.environ.get("ICECACHE_CROSS_TOKEN_MAX", "8"))
        # Always refresh after this many layers since the last DCI call.
        self.force_refresh_every = int(
            os.environ.get("ICECACHE_CROSS_TOKEN_REFRESH", "16"))
        # Per-layer tracking of the query signature that drove the last DCI
        # call.  Stored as a normalized head-dim vector on CPU.
        self.prev_query_sig = [None] * n_layers
        self.layer_reuse_count = [0] * n_layers
        self.layers_since_refresh = [0] * n_layers
        # Per-token boundary flag, set in _prepare_decode.
        self._token_is_boundary = False

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

    @staticmethod
    def _validate_fp16_recall_backend():
        """Fail closed when dtype=2 is unavailable in the M-DCI extension."""
        source = np.array([1.5, -2.0], dtype=np.float32)
        source_ptr = np.array([source.ctypes.data], dtype=np.uintp)
        destination = np.full(2, np.nan, dtype=np.float16)
        DCI.copy_to_buffer(
            source_ptr,
            ptr_dest=destination.ctypes.data,
            list_size=1,
            update_num=1,
            offset_s=1,
            offset_t=1,
            dim=1,
            page_size=1,
            dtype=2,
        )
        if not np.array_equal(
            destination, source.astype(np.float16), equal_nan=False
        ):
            raise RuntimeError(
                "ICECACHE_FP16_RECALL=1 requires the M-DCI dtype=2 "
                "FP32-to-FP16 gather patch"
            )
    
    def check_reuse(self, cur_id, start=2):
        # Experiment 10: layer on the skip list is forced to reuse the previous
        # anchor layer's selection (falls through the cross-layer reuse branch
        # below, which handles the c2p offset + recall safely).  Only applied
        # once the previous anchor's DCI tree exists (decode phase); during
        # prefill the trees are still being built, so fall back to normal.
        if cur_id in getattr(self, "skip_dci_layers", set()):
            if cur_id > start:
                reuse_id = cur_id - self.n_reuse_layers
                if (getattr(self, "_in_estimate", False)
                        and getattr(self, "_decode_phase", False)
                        and threading.get_ident() == getattr(
                            self, "_estimate_thread", None)):
                    return reuse_id
            return 0
        if self.n_reuse_layers == 0 or cur_id <= start:
            return 0
        position = cur_id - start
        cycle_length = self.n_reuse_layers
        position_in_cycle = position % cycle_length
        if position_in_cycle == 0:
            return 0
        else:
            return cur_id-position_in_cycle

    def _query_signature(self, query_states):
        """Cheap, head-mean, normalized signature for cosine-based gating.

        query_states has shape [bsz, 1, n_qo_heads, head_dim].  We average
        across heads to produce a single head_dim-vector, then L2-normalize
        so a dot-product becomes cosine similarity.  Stays on CPU to keep
        the gate off the GPU critical path.
        """
        sig = query_states.reshape(-1, self.head_dim).mean(dim=0)
        sig = sig.float().detach().cpu()
        norm = torch.linalg.vector_norm(sig)
        if norm.item() == 0.0:
            return sig
        return sig / norm

    def _can_reuse_cross_token(self, layer_idx, query_states):
        """Return True if we should reuse the previous DCI result for this
        layer instead of running a fresh search."""
        if not self.enable_cross_token_dci:
            return False
        if self.prev_query_sig[layer_idx] is None:
            return False
        if self._token_is_boundary:
            # A page boundary reshuffled the cache; refresh once.
            self.profile_cross_token_boundary_refreshes += 1
            return False
        if self.layers_since_refresh[layer_idx] >= self.force_refresh_every:
            return False
        if self.layer_reuse_count[layer_idx] >= self.max_consec_reuse:
            return False
        self.profile_cross_token_reuse_calls += 1
        cur_sig = self._query_signature(query_states)
        cos_sim = torch.dot(self.prev_query_sig[layer_idx], cur_sig).item()
        # Reuse when the query hasn't moved much (1 - cos is small).
        return (1.0 - cos_sim) < self.cosine_reuse_threshold

    def _update_query_signature(self, layer_idx, query_states):
        self.prev_query_sig[layer_idx] = self._query_signature(query_states)
        self.layer_reuse_count[layer_idx] = 0
        self.layers_since_refresh[layer_idx] = 0

    def _prepare_prefill(self, bsz, q_len):
        # Each prompt has its own cache fill and initial recalls, so warm-up
        # must restart per sequence while aggregate counters remain intact.
        self._decode_phase = False
        self.profile_sequence_decode_steps = 0
        # Cross-token gate state is per-sequence; reset on each new prompt.
        for i in range(self.n_layers):
            self.prev_query_sig[i] = None
            self.layer_reuse_count[i] = 0
            self.layers_since_refresh[i] = 0
        self._token_is_boundary = False
        # Do not compare the last decode selection of one sample with the
        # first selection of the next sample.
        self._trace_prev_selection = [None] * self.n_layers
        self.profile_cross_token_reuses = 0
        self.profile_cross_token_reuse_calls = 0
        self.profile_cross_token_boundary_refreshes = 0
        self.num_offload_pages = None
        self.n_dci_pages = None
        self.offload_win_flag = [False] * self.n_layers
        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.selected_page_idx = [None] * self.n_layers
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
                **self._recall_fp,
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
                **self._recall_fp,
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

        
        if self.dtype is not torch.float32 and not self.fp16_recall:
            self.cuda_cast_buffer = torch.empty(
                [self.batch_size, 2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages),
                    self.page_size * self.head_dim],
                dtype=self.dtype, device=self.device,
                pin_memory=False
            )
        else:
            self.cuda_cast_buffer = self.cuda_transit_buffer

        # [ICECACHE-DOUBLEBUF] ping-pong transit buffers; order with events
        # instead of c2g_stream.synchronize().  See apply_double_buffer_patch.py.
        self.double_buffer = bool(int(os.environ.get("ICECACHE_DOUBLE_BUFFER", "0")))
        self._db_cursor = {}
        self._db_used_slot = {}
        self._db_ev_read = [None, None]
        self._db_ev_cast = [None, None]
        self._db_ev_used = [None, None]
        if self.double_buffer:
            if self.batch_layer_recall or self.n_prefetch_layers:
                raise ValueError(
                    "ICECACHE_DOUBLE_BUFFER=1 is incompatible with batched "
                    "layer recall and with layer prefetch")
            _cast_alias = (self.cuda_cast_buffer is self.cuda_transit_buffer)
            if self.n_reuse_layers > 0:
                _w = 2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages) * self.n_reuse_layers
            else:
                _w = 2 * self.n_kv_heads * (self.layer2budget[-1] - self.n_sink_pages - self.n_win_pages)
            self.cpu_transit_buffer = [
                torch.empty([self.batch_size, _w, self.page_size * self.head_dim],
                            **self._recall_fp, pin_memory=True) for _ in range(2)]
            self.cuda_transit_buffer = [
                torch.empty([self.batch_size, _w, self.page_size * self.head_dim],
                            **self._fp, pin_memory=False) for _ in range(2)]
            if _cast_alias:
                self.cuda_cast_buffer = self.cuda_transit_buffer
            else:
                self.cuda_cast_buffer = [
                    self.cuda_cast_buffer,
                    torch.empty_like(self.cuda_cast_buffer)]

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
                    promotion_prob = self.ratio_1
                    if (self.promotion_fast_start_layer >= 0
                            and i >= self.promotion_fast_start_layer):
                        promotion_prob = self.promotion_fast_ratio
                    self.dci_db[i] = DCI(self.head_dim, 1, 1, promotion_prob=promotion_prob, promotion_prob_subseq=self.ratio_2, num_points=q_len, init=True, num_inst=self.n_kv_heads, debug=self.debug, transform=True, parallel_level=self.parallel_level, proj_vec=proj_vec)

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


    def _prepare_decode(self, bsz):
        self._decode_phase = True
        self._batched_recall_slices.clear()
        self.profile_decode_steps += 1
        self.profile_sequence_decode_steps += 1
        if self.profile_dci:
            # Synchronize only for profiling so one measured interval includes
            # the GPU work and CPU DCI work on the decode critical path.
            torch.cuda.synchronize(self.device)
            self._profile_decode_start = perf_counter()

        # A new page boundary lands on the token that fills a page slot.
        # Force-refresh the cross-token gate state so the just-bumped cache
        # layout is observed by DCI on the next call.
        self._token_is_boundary = (
            self.kv_last_page_len + 1 >= self.page_size
        )

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
        if self.profile_dci and self._profile_decode_start is not None:
            torch.cuda.synchronize(self.device)
            if self.profile_sequence_decode_steps > self.profile_warmup_tokens:
                _step_seconds = perf_counter() - self._profile_decode_start
                self.profile_decode_seconds += _step_seconds
                self.profile_measured_steps += 1
                # [ICECACHE-PROFILE] Opt-in per-step samples so TPOT can be
                # reported with a spread instead of only an aggregate mean.
                if (self.profile_step_samples is not None
                        and len(self.profile_step_samples)
                        < self.profile_step_samples_cap):
                    self.profile_step_samples.append(_step_seconds)
            self._profile_decode_start = None

    def _profile_is_measured_step(self):
        return self.profile_dci and self.profile_sequence_decode_steps > self.profile_warmup_tokens

    def _diag_collect(self, layer_idx, b, rids_cpu, nr_cpu):
        # [ICECACHE-DIAG] record the selected leaf ids + their real CPU
        # addresses, per (layer, head).  Only the first
        # diag_max_records recalls are kept, so the dump stays tiny.
        if len(self.diag_records) >= self.diag_max_records:
            return
        for i in range(self.n_kv_heads):
            cnt = int(nr_cpu[i])
            if cnt <= 0:
                continue
            leaves = np.asarray(rids_cpu[i, :cnt]).astype(np.int64)
            addrs = self.page_address_buffer[layer_idx][
                b, i, leaves].astype(np.uint64)
            self.diag_records.append((layer_idx, i, leaves, addrs))
        if (len(self.diag_records) >= self.diag_max_records
                and not self.diag_saved):
            self._save_diag()

    def _save_diag(self):
        if not self.diag_enabled:
            return
        path = self.diag_dump_path or os.path.join(
            os.getcwd(), "icecache_diag.npz")
        try:
            if self.diag_records:
                layer = np.asarray(
                    [r[0] for r in self.diag_records], dtype=np.int32)
                head = np.asarray(
                    [r[1] for r in self.diag_records], dtype=np.int32)
                flat_leaf = np.concatenate([r[2] for r in self.diag_records])
                flat_addr = np.concatenate([r[3] for r in self.diag_records])
                offs = np.zeros(len(self.diag_records) + 1, dtype=np.int64)
                for k, r in enumerate(self.diag_records):
                    offs[k + 1] = offs[k] + len(r[2])
            else:
                layer = np.zeros(0, np.int32)
                head = np.zeros(0, np.int32)
                flat_leaf = np.zeros(0, np.int64)
                flat_addr = np.zeros(0, np.uint64)
                offs = np.zeros(1, np.int64)
            np.savez(
                path, layer=layer, head=head, flat_leaf=flat_leaf,
                flat_addr=flat_addr, offsets=offs,
                addr_prep_seconds=self.diag_addr_prep_seconds,
                copy_buffer_seconds=self.diag_copy_buffer_seconds,
                h2d_ms=self.diag_h2d_ms, cast_ms=self.diag_cast_ms,
                h2d_count=self.diag_h2d_count,
                cast_count=self.diag_cast_count)
            self.diag_saved = True
            print("[ICECACHE-DIAG] records=%d -> %s" % (
                len(self.diag_records), path))
            print("[ICECACHE-DIAG] addr_prep_s=%.4f copy_buffer_s=%.4f "
                  "h2d_ms=%.3f cast_ms=%.3f h2d_n=%d cast_n=%d" % (
                      self.diag_addr_prep_seconds,
                      self.diag_copy_buffer_seconds,
                      self.diag_h2d_ms, self.diag_cast_ms,
                      self.diag_h2d_count, self.diag_cast_count))
        except Exception as exc:  # never let diagnostics break a run
            print("[ICECACHE-DIAG] save failed: %r" % (exc,))

    def _step_sample_stats(self):
        """Mean / spread of the per-decode-step latencies (ms).

        Only populated when ICECACHE_PROFILE_STEP_SAMPLES=1.  `n` should match
        `decode_steps_measured`; `mean_ms` should match `decode_tpot_ms`.
        """
        samples = self.profile_step_samples
        if not samples:
            return None
        arr = np.asarray(samples, dtype=np.float64) * 1e3
        return {
            "n": int(arr.size),
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "cv": float(arr.std(ddof=1) / arr.mean()) if arr.size > 1 else 0.0,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
        }

    def _dump_index_call_records(self):
        """atexit dump of the per-call index-update records (JSON)."""
        if not self.profile_call_dump:
            return
        try:
            import json

            with open(self.profile_call_dump, "w") as fh:
                json.dump(self.profile_index_call_records, fh)
            print("[ICECACHE-PROFILE] %d index-update call records -> %s"
                  % (len(self.profile_index_call_records),
                     self.profile_call_dump))
        except Exception as exc:  # never let diagnostics break a run
            print("[ICECACHE-PROFILE] call dump failed: %r" % (exc,))

    @staticmethod
    def _tree_size_bucket(prev_num_points):
        """Power-of-two bucket key for the pre-insertion tree size."""
        p = int(prev_num_points)
        if p < 0:
            return "unknown"
        if p == 0:
            return "0"
        lo = 1 << (p - 1).bit_length() - 1
        return "p%d-%d" % (lo, 2 * lo - 1)

    def _index_addrprep_buckets(self):
        """Aggregate the per-call index-update records by tree size.

        Answers "does the address path / native insert scale with the number of
        points already in the tree", which a per-token average hides.
        """
        _MS_FIELDS = (
            ("addr_prep_ms", "addr_prep_ms_sum"),
            ("addr_leaf_ms", "addr_leaf_ms_sum"),
            ("addr_np_ms", "addr_np_ms_sum"),
            ("addr_write_ms", "addr_write_ms_sum"),
            ("addr_meta_ms", "addr_meta_ms_sum"),
            ("native_insert_ms", "native_insert_ms_sum"),
            ("native_addr_update_ms", "native_addr_update_ms_sum"),
            ("reuse_update_ms", "reuse_update_ms_sum"),
        )
        by_tree = {}
        by_layer = {}
        for rec in self.profile_index_call_records:
            for target, mk in (
                (by_tree, self._tree_size_bucket(rec.get("prev_num_points", -1))),
                (by_layer, str(rec.get("layer", -1))),
            ):
                bucket = target.get(mk)
                if bucket is None:
                    bucket = {
                        "calls": 0,
                        "anchor_calls": 0,
                        "reuse_calls": 0,
                        "insert_tokens": 0,
                        "new_leaves_total": 0,
                        "new_leaves_max": 0,
                        "prev_num_points_min": None,
                        "prev_num_points_max": None,
                    }
                    for _, sum_key in _MS_FIELDS:
                        bucket[sum_key] = 0.0
                    target[mk] = bucket
                bucket["calls"] += 1
                if rec.get("anchor"):
                    bucket["anchor_calls"] += 1
                else:
                    bucket["reuse_calls"] += 1
                bucket["insert_tokens"] += int(rec.get("insert_tokens", 0))
                bucket["new_leaves_total"] += int(rec.get("new_leaves_total", 0))
                bucket["new_leaves_max"] = max(
                    bucket["new_leaves_max"], int(rec.get("new_leaves_max", 0)))
                pnp = int(rec.get("prev_num_points", -1))
                if pnp >= 0:
                    if bucket["prev_num_points_min"] is None:
                        bucket["prev_num_points_min"] = pnp
                        bucket["prev_num_points_max"] = pnp
                    else:
                        bucket["prev_num_points_min"] = min(
                            bucket["prev_num_points_min"], pnp)
                        bucket["prev_num_points_max"] = max(
                            bucket["prev_num_points_max"], pnp)
                for src, sum_key in _MS_FIELDS:
                    bucket[sum_key] += float(rec.get(src, 0.0))

        for target in (by_tree, by_layer):
            for bucket in target.values():
                calls = max(bucket["calls"], 1)
                for src, sum_key in _MS_FIELDS:
                    bucket[src.replace("_ms", "_ms_per_call")] = (
                        bucket.pop(sum_key) / calls)
        return {"by_tree_size": by_tree, "by_layer": by_layer}

    def get_profile_stats(self):
        """Return aggregate steady-state decode timing statistics in seconds."""
        measured = self.profile_measured_steps
        total_steps = max(self.profile_decode_steps, 1)
        return {
            "warmup_tokens": self.profile_warmup_tokens,
            "decode_steps_total": self.profile_decode_steps,
            "decode_steps_measured": measured,
            "decode_total_seconds": self.profile_decode_seconds,
            "dci_select_seconds": self.profile_dci_seconds,
            "dci_select_calls": self.profile_dci_calls,
            "decode_tpot_ms": 1000 * self.profile_decode_seconds / measured if measured else None,
            "decode_step_latency": self._step_sample_stats(),
            "dci_select_ms_per_token": 1000 * self.profile_dci_seconds / measured if measured else None,
            "dci_share": self.profile_dci_seconds / self.profile_decode_seconds if self.profile_decode_seconds else None,
            "dci_calls_per_token": self.profile_dci_calls / measured if measured else None,
            "query_d2h_ms_per_token": 1000 * self.profile_query_d2h_seconds / measured if measured else None,
            "native_query_ms_per_token": 1000 * self.profile_native_query_seconds / measured if measured else None,
            "native_query_ms_per_call_by_layer": {
                str(layer): 1000 * seconds
                / self.profile_native_query_calls_by_layer[layer]
                for layer, seconds in sorted(
                    self.profile_native_query_seconds_by_layer.items())
                if self.profile_native_query_calls_by_layer[layer]
            },
            "query_postprocess_ms_per_token": 1000 * self.profile_query_postprocess_seconds / measured if measured else None,
            "query_dedup_ms_per_token": 1000 * self.profile_query_dedup_seconds / measured if measured else None,
            "query_mapping_ms_per_token": 1000 * self.profile_query_mapping_seconds / measured if measured else None,
            "query_diff_ms_per_token": 1000 * self.profile_query_diff_seconds / measured if measured else None,
            "recall_gather_ms_per_token": 1000 * self.profile_recall_gather_seconds / measured if measured else None,
            "recall_wait_ms_per_token": 1000 * self.profile_recall_wait_seconds / measured if measured else None,
            "page_metadata_ms_per_token": 1000 * self.profile_page_metadata_seconds / measured if measured else None,
            "index_update_ms_per_token": 1000 * self.profile_index_update_seconds / measured if measured else None,
            "index_pack_ms_per_token": 1000 * self.profile_index_pack_seconds / measured if measured else None,
            "index_numpy_ms_per_token": 1000 * self.profile_index_numpy_seconds / measured if measured else None,
            "index_prepare_ms_per_token": 1000 * self.profile_index_prepare_seconds / measured if measured else None,
            "index_native_insert_ms_per_token": 1000 * self.profile_index_native_insert_seconds / measured if measured else None,
            "index_ccc_writeback_ms_per_token": 1000 * self.profile_index_ccc_writeback_seconds / measured if measured else None,
            "index_page_alloc_ms_per_token": 1000 * self.profile_index_page_alloc_seconds / measured if measured else None,
            "index_address_update_ms_per_token": 1000 * self.profile_index_address_update_seconds / measured if measured else None,
            "index_address_prepare_ms_per_token": 1000 * self.profile_index_address_prepare_seconds / measured if measured else None,
            "index_native_address_update_ms_per_token": 1000 * self.profile_index_native_address_update_seconds / measured if measured else None,
            "index_reuse_update_ms_per_token": 1000 * self.profile_index_reuse_update_seconds / measured if measured else None,
            "index_addrprep_meta_ms_per_token": 1000 * self.profile_index_addrprep_meta_seconds / measured if measured else None,
            "index_addrprep_leaf_ms_per_token": 1000 * self.profile_index_addrprep_leaf_seconds / measured if measured else None,
            "index_addrprep_np_ms_per_token": 1000 * self.profile_index_addrprep_np_seconds / measured if measured else None,
            "index_addrprep_write_ms_per_token": 1000 * self.profile_index_addrprep_write_seconds / measured if measured else None,
            "index_addrprep_buckets": self._index_addrprep_buckets(),
            "addr_equiv": (self.get_addr_equiv_stats()
                           if self.addr_equiv_check else None),
            "recall_calls_per_token": self.profile_recall_calls / measured if measured else None,
            "recall_submissions_per_token": self.profile_recall_submissions / measured if measured else None,
            "recall_pages_per_token": self.profile_recall_pages / measured if measured else None,
            "index_update_calls": self.profile_index_update_calls,
            "fp16_recall": self.fp16_recall,
            "batch_layer_recall": self.batch_layer_recall,
            "cross_token_reuses": self.profile_cross_token_reuses,
            "cross_token_reuse_calls": self.profile_cross_token_reuse_calls,
            "cross_token_reuse_share": (
                self.profile_cross_token_reuses
                / self.profile_cross_token_reuse_calls
                if self.profile_cross_token_reuse_calls else None
            ),
            "cross_token_boundary_refreshes": self.profile_cross_token_boundary_refreshes,
            "cross_token_skip_rate": (
                self.profile_cross_token_reuses / total_steps / self.n_layers
                if total_steps else None
            ),
        }

    @staticmethod
    def _selection_overlap(current, previous, prefix=None):
        """Set overlap normalized by the current selection width."""
        if prefix is not None:
            current = current[:prefix]
            previous = previous[:prefix]
        current_set = set(int(x) for x in current if x >= 0)
        previous_set = set(int(x) for x in previous if x >= 0)
        if not current_set:
            return 1.0
        return len(current_set & previous_set) / len(current_set)

    def _trace_dci_selection(self, layer_idx, selection):
        if not self.trace_dci_churn:
            return
        current = np.asarray(selection, dtype=np.int32).copy()
        previous = self._trace_prev_selection[layer_idx]
        self._trace_prev_selection[layer_idx] = current
        if previous is None or previous.shape != current.shape:
            return

        width = current.shape[1]
        top25 = max(1, width // 4)
        top50 = max(1, width // 2)
        for head_idx in range(current.shape[0]):
            overlap = self._selection_overlap(
                current[head_idx], previous[head_idx])
            overlap25 = self._selection_overlap(
                current[head_idx], previous[head_idx], top25)
            overlap50 = self._selection_overlap(
                current[head_idx], previous[head_idx], top50)
            exact = float(np.array_equal(
                current[head_idx], previous[head_idx]))
            self._trace_overlap.append(overlap)
            self._trace_top25_overlap.append(overlap25)
            self._trace_top50_overlap.append(overlap50)
            self._trace_exact.append(exact)
            self._trace_by_layer[layer_idx].append(overlap)
            self._trace_by_head[head_idx].append(overlap)

    def get_dci_churn_stats(self):
        """Return aggregate consecutive-token page-selection stability."""
        def summarize(values):
            if not values:
                return None
            values = np.asarray(values, dtype=np.float64)
            return {
                "count": int(values.size),
                "mean": float(values.mean()),
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }

        return {
            "all_topk_overlap": summarize(self._trace_overlap),
            "top25_overlap": summarize(self._trace_top25_overlap),
            "top50_overlap": summarize(self._trace_top50_overlap),
            "exact_same_share": (
                float(np.mean(self._trace_exact))
                if self._trace_exact else None
            ),
            "share_overlap_ge_90pct": (
                float(np.mean(np.asarray(self._trace_overlap) >= 0.9))
                if self._trace_overlap else None
            ),
            "share_overlap_ge_75pct": (
                float(np.mean(np.asarray(self._trace_overlap) >= 0.75))
                if self._trace_overlap else None
            ),
            "by_layer_mean": {
                str(k): float(np.mean(v))
                for k, v in sorted(self._trace_by_layer.items())
            },
            "by_kv_head_mean": {
                str(k): float(np.mean(v))
                for k, v in sorted(self._trace_by_head.items())
            },
        }

    def _raw_dci_to_pages(self, raw_indices, num_neighbours):
        """Convert raw per-Q-head results into unique per-KV-head pages."""
        reshaped = raw_indices.reshape(self.n_qo_heads, 2, -1)
        if self.ratio > 1:
            if _dci_first_k_unique_by_head is not None:
                pages = _dci_first_k_unique_by_head(
                    reshaped, self.n_kv_heads, self.ratio, num_neighbours)
            else:
                pages = reshaped[:, 0, :].reshape(
                    self.n_kv_heads, self.ratio, -1)
                interleaved = pages.transpose(0, 2, 1).reshape(
                    self.n_kv_heads, -1)
                pages = np.vstack([
                    utils.first_k_unique(row, num_neighbours)
                    for row in interleaved
                ])
        else:
            pages = reshaped[:, 0, :].reshape(self.n_kv_heads, -1)
        return np.ascontiguousarray(pages)

    def _trace_adaptive_queries(self, dci_db, query, padding_mask,
                                num_neighbours, field_of_view,
                                num_points, full_pages, full_query_seconds):
        """Measure per-head search convergence without changing inference."""
        if not self.trace_dci_adaptive:
            return

        staged_pages = []
        staged_fractions = []
        for fraction in self.trace_dci_adaptive_levels:
            if fraction >= 1.0:
                pages = full_pages
                elapsed = full_query_seconds
            else:
                visit = max(num_neighbours, int(num_points * fraction))
                stage_start = perf_counter()
                raw, _ = dci_db.query(
                    query,
                    padding_mask,
                    num_neighbours=num_neighbours,
                    field_of_view=field_of_view,
                    num_to_visit=visit,
                    num_to_retrieve=-1,
                    # A negative proportion tells the binding to honor the
                    # explicit per-stage num_to_visit budget.
                    prop_to_visit=-1.0,
                    prop_to_retrieve=0.8,
                    parallel_level=self.parallel_level,
                    ratio=self.ratio,
                )
                elapsed = perf_counter() - stage_start
                pages = self._raw_dci_to_pages(raw, num_neighbours)
            self._adaptive_stage_seconds[str(fraction)] += elapsed
            self._adaptive_stage_calls[str(fraction)] += 1
            staged_pages.append(pages)
            staged_fractions.append(fraction)

        for stage_idx, fraction in enumerate(staged_fractions[:-1]):
            for head_idx in range(self.n_kv_heads):
                self._adaptive_fixed_recall[str(fraction)].append(
                    self._selection_overlap(
                        staged_pages[stage_idx][head_idx],
                        full_pages[head_idx]))

        for threshold in self.trace_dci_adaptive_thresholds:
            key = str(threshold)
            for head_idx in range(self.n_kv_heads):
                chosen_idx = len(staged_pages) - 1
                # Each KV head independently stops when consecutive budgets
                # produce sufficiently similar page sets.
                for stage_idx in range(1, len(staged_pages)):
                    convergence = self._selection_overlap(
                        staged_pages[stage_idx][head_idx],
                        staged_pages[stage_idx - 1][head_idx])
                    if convergence >= threshold:
                        chosen_idx = stage_idx
                        break
                self._adaptive_stop_fraction[key].append(
                    staged_fractions[chosen_idx])
                self._adaptive_oracle_recall[key].append(
                    self._selection_overlap(
                        staged_pages[chosen_idx][head_idx],
                        full_pages[head_idx]))

    def get_dci_adaptive_stats(self):
        def summarize(values):
            values = np.asarray(values, dtype=np.float64)
            if values.size == 0:
                return None
            return {
                "count": int(values.size),
                "mean": float(values.mean()),
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }

        adaptive = {}
        for threshold, recalls in self._adaptive_oracle_recall.items():
            recall_array = np.asarray(recalls, dtype=np.float64)
            adaptive[threshold] = {
                "stop_fraction": summarize(
                    self._adaptive_stop_fraction[threshold]),
                "oracle_recall": summarize(recalls),
                "share_recall_ge_90pct": float(np.mean(
                    recall_array >= 0.9)),
                "share_recall_ge_95pct": float(np.mean(
                    recall_array >= 0.95)),
            }
        return {
            "levels": self.trace_dci_adaptive_levels,
            "stage_mean_ms_per_layer_query": {
                key: 1000 * seconds / self._adaptive_stage_calls[key]
                for key, seconds in self._adaptive_stage_seconds.items()
                if self._adaptive_stage_calls[key]
            },
            "fixed_fraction_oracle_recall": {
                key: summarize(values)
                for key, values in self._adaptive_fixed_recall.items()
            },
            "adaptive": adaptive,
        }

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
        # Experiment 10: tree construction must always happen, regardless of
        # the skip list (a skipped layer still needs its own DCI tree for the
        # cross-layer reuse of other layers to address pages).  Guard the
        # whole method so skip-forcing in check_reuse never applies here.
        _prev_in_est = getattr(self, "_in_estimate", False)
        self._in_estimate = False
        try:
            return self._DCI_first_call_impl(
                b, cur_id, query_states, key_states, value_states, projected)
        finally:
            self._in_estimate = _prev_in_est

    def _DCI_first_call_impl(self, b, cur_id, query_states, key_states, value_states, projected):
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


    @staticmethod
    def page_address_formula(pool_base, physical_ids, page_stride, head_offset):
        """Physical CPU-side byte address of a KV head inside a page.

        The CPU KV pool is a single contiguous pinned tensor and `page_stride`
        is `cpu_n_bytes_per_page`, so page `p` starts at
        `pool_base + p * page_stride`.  `physical_ids` must already be mapped
        through `c2p`: the pool is fragmented during decode (pages are handed
        out from a free-id set and reused in place), so a logical cache page id
        is NOT a physical page id.
        """
        return (np.uintp(pool_base)
                + np.asarray(physical_ids, dtype=np.uintp) * np.uintp(page_stride)
                + np.uintp(head_offset))

    def _addr_equiv_probe(self, cur_id, b, inst, c2p_np, pool_base,
                          logical_ids, head_offset, slow_addr):
        """Read-only equivalence check; raises on any mismatch."""
        physical_ids = c2p_np[b, logical_ids]
        fast = self.page_address_formula(
            pool_base, physical_ids, self.cpu_n_bytes_per_page, head_offset)
        # Deliberately *not* the production formula: this is the tempting
        # "logical page id == physical page id" shortcut we want to falsify.
        logical = self.page_address_formula(
            pool_base, logical_ids, self.cpu_n_bytes_per_page, head_offset)
        slow = np.asarray(slow_addr, dtype=np.uintp)
        n = int(slow.size)
        bad_fast = int(np.count_nonzero(fast != slow))
        bad_logical = int(np.count_nonzero(logical != slow))
        self.addr_equiv_checks += n
        self.addr_equiv_mismatch += bad_fast
        self.addr_equiv_logical_mismatch += bad_logical
        self.addr_equiv_phys.update(
            int(x) for x in np.unique(np.asarray(physical_ids, dtype=np.int64)))
        key = "%d" % int(cur_id)
        rec = self.addr_equiv_by_layer.get(key)
        if rec is None:
            rec = [0, 0, 0]
            self.addr_equiv_by_layer[key] = rec
        rec[0] += n
        rec[1] += bad_fast
        rec[2] += bad_logical
        if bad_fast:
            raise AssertionError(
                "[ICECACHE-ADDR-EQUIV] vectorised address mismatch: layer=%d "
                "head=%d n=%d (mismatched=%d)"
                % (cur_id, inst, n, bad_fast))

    def get_addr_equiv_stats(self):
        return {
            "vec_addr": self.vec_addr,
            "checks": self.addr_equiv_checks,
            "fast_mismatch": self.addr_equiv_mismatch,
            "logical_shortcut_mismatch": self.addr_equiv_logical_mismatch,
            "distinct_physical_pages": len(self.addr_equiv_phys),
            "by_layer": {
                layer: {"n": v[0], "fast_mismatch": v[1],
                        "logical_shortcut_mismatch": v[2]}
                for layer, v in sorted(
                    self.addr_equiv_by_layer.items(), key=lambda kv: int(kv[0]))
            },
        }

    def _DCI_add(self, b, cur_id, key_states, value_states):
        if self.use_dci:
            profile_stage = self._profile_is_measured_step()
            stage_start = perf_counter() if profile_stage else None

            # Per-call record, only materialised while profiling.  It is filled
            # in stage by stage and appended at the end of the function.
            _call_rec = None
            if profile_stage:
                _reuse_of = self.check_reuse(cur_id)
                _call_rec = {
                    "layer": int(cur_id),
                    "anchor": _reuse_of == 0,
                    "reuse_of": int(_reuse_of),
                }

            dci_len = key_states.shape[1]

            _key_states = key_states.reshape(-1, self.head_dim).float().numpy()
            _value_states = value_states.reshape(-1,
                                                 self.head_dim).float().numpy()

            assert (_key_states.flags['C_CONTIGUOUS'])
            assert (_value_states.flags['C_CONTIGUOUS'])
            if stage_start is not None:
                self.profile_index_numpy_seconds += perf_counter() - stage_start
                stage_start = perf_counter()

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

                if stage_start is not None:
                    self.profile_index_prepare_seconds += perf_counter() - stage_start
                    native_insert_start = perf_counter()

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

                if stage_start is not None:
                    _native_insert_s = perf_counter() - native_insert_start
                    self.profile_index_native_insert_seconds += _native_insert_s
                    writeback_start = perf_counter()
                    if _call_rec is not None:
                        _call_rec["native_insert_ms"] = 1e3 * _native_insert_s

                kvc.ccc[b][:, :prev_max_num_pages] = torch.tensor(
                    ccc, **self._cb).reshape(self.n_kv_heads, -1)
                if stage_start is not None:
                    self.profile_index_ccc_writeback_seconds += (
                        perf_counter() - writeback_start)
                    page_alloc_start = perf_counter()
                
                dci_db = self.dci_db[cur_id]
            else:
                reuse_id = self.check_reuse(cur_id)
                dci_db = self.dci_db[reuse_id]
                kvc.ccc = self.kv_caches[reuse_id].ccc.clone()
                reuse_ccc = kvc.ccc[b][:, :dci_db.num_leaves.max()].numpy().astype(np.bool_).reshape(self.batch_size*self.n_kv_heads, -1)

            if profile_stage:
                page_alloc_start = perf_counter()

            if _call_rec is not None:
                _call_rec["prev_num_points"] = (
                    int(self.prev_num_points)
                    if self.prev_num_points is not None else -1)
                _call_rec["insert_tokens"] = int(dci_len)
                _call_rec["n_heads"] = int(self.n_kv_heads)

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

            if profile_stage:
                self.profile_index_page_alloc_seconds += (
                    perf_counter() - page_alloc_start)
                address_update_start = perf_counter()

            # Use the new allocated pages to store the DCI tokens
            _ap_meta_start = perf_counter() if profile_stage else None
            new_num_leaves = dci_db.num_leaves - self.prev_num_pages
            new_address = [None] * self.n_kv_heads
            # !! Here assume batch size = 1
            offset = self.page_size * self.head_dim * self.cpu_dtype.itemsize
            new_indices = np.zeros(
                [self.n_kv_heads, new_num_leaves.max()], dtype=np.int32)
            # [ICECACHE-VECADDR] Zero-copy view of the logical->physical page
            # map.  `c2p` lives on the CPU cache and is never reallocated in
            # place (it is rebuilt by `utils.cat`), so a per-call view is safe.
            _vec_on = self.vec_addr or self.addr_equiv_check
            c2p_np = None
            pool_base = 0
            if _vec_on:
                c2p_np = cpu_cache.c2p.numpy()
                pool_base = cpu_cache.pool.buffer.data_ptr()
            if _ap_meta_start is not None:
                self.profile_index_addrprep_meta_seconds += (
                    perf_counter() - _ap_meta_start)
                _ap_leaf_seconds = 0.0
                _ap_np_seconds = 0.0
                _ap_write_seconds = 0.0
                _ap_leaves_total = 0
                _ap_leaves_max = 0
            for inst in range(self.n_kv_heads):
                if new_num_leaves[inst] == 0:
                    new_address[inst] = []
                    continue
                tmp_new_indices = np.arange(self.prev_num_pages[inst], dci_db.num_leaves[inst])
                new_indices[inst, :new_num_leaves[inst]] = tmp_new_indices
                _ap_t0 = perf_counter() if profile_stage else None
                if self.vec_addr:
                    _ap_arr = self.page_address_formula(
                        pool_base, c2p_np[b, tmp_new_indices],
                        self.cpu_n_bytes_per_page, inst * offset)
                    _ap_t1 = perf_counter() if profile_stage else None
                    # The M-DCI binding takes a plain list of ints here; a
                    # NumPy array is not interchangeable (it segfaults).
                    tmp_addr = _ap_arr.tolist()
                else:
                    tmp_addr = [cast(cpu_cache[b, j].data_ptr() + inst * offset, c_void_p).value for j in tmp_new_indices]
                    _ap_t1 = perf_counter() if profile_stage else None
                    _ap_arr = np.array(tmp_addr, dtype=np.uintp)
                if _ap_t0 is not None:
                    _ap_t2 = perf_counter()
                    _ap_leaf_seconds += _ap_t1 - _ap_t0
                    _ap_np_seconds += _ap_t2 - _ap_t1
                    _ap_leaves_total += len(tmp_addr)
                    if len(tmp_addr) > _ap_leaves_max:
                        _ap_leaves_max = len(tmp_addr)
                new_address[inst] = tmp_addr
                self.page_address_buffer[cur_id][b, inst, tmp_new_indices] = _ap_arr
                if self.addr_equiv_check:
                    # Reference must always be the true per-leaf data_ptr()
                    # path, even when the fast path is the one in production.
                    if self.vec_addr:
                        _slow_ref = [
                            cast(cpu_cache[b, j].data_ptr() + inst * offset,
                                 c_void_p).value
                            for j in tmp_new_indices]
                    else:
                        _slow_ref = tmp_addr
                    self._addr_equiv_probe(
                        cur_id, b, inst, c2p_np, pool_base,
                        tmp_new_indices, inst * offset, _slow_ref)
                if _ap_t0 is not None:
                    _ap_write_seconds += perf_counter() - _ap_t2

            if profile_stage:
                _ap_end = perf_counter()
                self.profile_index_addrprep_leaf_seconds += _ap_leaf_seconds
                self.profile_index_addrprep_np_seconds += _ap_np_seconds
                self.profile_index_addrprep_write_seconds += _ap_write_seconds
                self.profile_index_address_prepare_seconds += (
                    _ap_end - address_update_start)
                _call_rec["addr_prep_ms"] = 1e3 * (_ap_end - address_update_start)
                _call_rec["addr_meta_ms"] = 1e3 * (
                    _ap_end - address_update_start) - 1e3 * (
                    _ap_leaf_seconds + _ap_np_seconds + _ap_write_seconds)
                _call_rec["addr_leaf_ms"] = 1e3 * _ap_leaf_seconds
                _call_rec["addr_np_ms"] = 1e3 * _ap_np_seconds
                _call_rec["addr_write_ms"] = 1e3 * _ap_write_seconds
                _call_rec["new_leaves_total"] = int(_ap_leaves_total)
                _call_rec["new_leaves_max"] = int(_ap_leaves_max)

            if self.check_reuse(cur_id) == 0:
                native_address_start = perf_counter() if profile_stage else None
                dci_db.address_update(indices=new_indices, new_address=new_address,
                                               num_pages=new_num_leaves, offset=self.n_kv_heads*self.page_size*self.head_dim)
                if native_address_start is not None:
                    _native_addr_s = perf_counter() - native_address_start
                    self.profile_index_native_address_update_seconds += _native_addr_s
                    if _call_rec is not None:
                        _call_rec["native_addr_update_ms"] = 1e3 * _native_addr_s
            else:
                old_index, old_offset = self.prev_index, self.prev_offset
                new_index, new_offset = dci_db.token2node
                reuse_update_start = perf_counter() if profile_stage else None
                DCI.reuse_update_node(old_index=old_index, old_offset=old_offset, new_index=new_index, new_offset=new_offset, keys=_key_states, values=_value_states, new_address=self.page_address_buffer[cur_id][0], kv_offset=self.n_kv_heads*self.page_size*self.head_dim, ccc=reuse_ccc, num_leaves=dci_db.num_leaves)
                if reuse_update_start is not None:
                    _reuse_update_s = perf_counter() - reuse_update_start
                    self.profile_index_reuse_update_seconds += _reuse_update_s
                    if _call_rec is not None:
                        _call_rec["reuse_update_ms"] = 1e3 * _reuse_update_s

            if profile_stage:
                _addr_update_s = perf_counter() - address_update_start
                self.profile_index_address_update_seconds += _addr_update_s
                if _call_rec is not None:
                    _call_rec["addr_update_ms"] = 1e3 * _addr_update_s
                if len(self.profile_index_call_records) < self.profile_index_call_records_cap:
                    self.profile_index_call_records.append(_call_rec)


    def _DCI_query(self, b, cur_id, query_states):
        if self.use_dci:

            profile_stage = self._profile_is_measured_step()

            bsz = 1

            num_neighbours = self.n_dci_pages - self.layer2topk[cur_id]
            # M-DCI terminates the entire process when field_of_view is
            # smaller than the number of neighbour pages it must expand.
            # Budget 64 therefore cannot safely use the old minimum of 30.
            query_field_of_view = max(
                int(self.seq_len * self.search_ratio), num_neighbours, 30)
            query_prop_to_retrieve = 0.8

            prev_num_points = self.dci_db[cur_id].num_points[0]
            # NOTE: M-DCI takes max(num_to_visit, prop_to_visit*num_points)
            # as the visit budget, so num_to_visit MUST be <= the prop
            # budget or prop_to_visit is swallowed.  Keep both consistent.
            num_to_visit = max(
                int(prev_num_points * self.dci_prop_to_visit), 1)
            num_to_retrieve = -1
            prop_to_visit = self.dci_prop_to_visit
            padding_mask = np.ones(
                [bsz, self.n_qo_heads, 1], dtype=np.bool_).reshape(-1)

            _query = query_states.reshape(-1, self.head_dim).float().numpy()

            # Use gc2cc and cc2gp for page indexing
            kvc = self.kv_caches[cur_id]

            assert (_query.flags['C_CONTIGUOUS'])

            native_query_start = perf_counter() if (
                profile_stage or self.trace_dci_adaptive) else None
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
            native_query_elapsed = (
                perf_counter() - native_query_start
                if native_query_start is not None else 0.0
            )
            if profile_stage:
                self.profile_native_query_seconds += native_query_elapsed
                self.profile_native_query_seconds_by_layer[cur_id] += (
                    native_query_elapsed)
                self.profile_native_query_calls_by_layer[cur_id] += 1

            postprocess_start = perf_counter() if profile_stage else None

            nn_idx = nn_idx.reshape(self.n_qo_heads, 2, -1)
            nn_idx_1 = nn_idx[:, 1, :].reshape(self.n_kv_heads, self.ratio, -1)
            if self.n_prefetch_layers > 1:
                self.nn_idx_all[b, cur_id] = nn_idx_1

            # Handle the case where there are duplicated integers in the same row of nn_idx_0
            dedup_start = perf_counter() if profile_stage else None
            if self.ratio > 1:
                if _dci_first_k_unique_by_head is not None:
                    nn_idx_0 = _dci_first_k_unique_by_head(
                        nn_idx,
                        self.n_kv_heads,
                        self.ratio,
                        num_neighbours,
                    )
                else:
                    nn_idx_0 = nn_idx[:, 0, :].reshape(
                        self.n_kv_heads, self.ratio, -1)
                    nn_idx_0_interleaved = nn_idx_0.transpose(
                        0, 2, 1).reshape(self.n_kv_heads, -1)
                    nn_idx_0 = np.vstack([utils.first_k_unique(
                        row, num_neighbours)
                        for row in nn_idx_0_interleaved])
            else:
                nn_idx_0 = nn_idx[:, 0, :].reshape(self.n_kv_heads, -1)

            nn_idx_0 = np.ascontiguousarray(nn_idx_0)
            nn_idx_1 = np.ascontiguousarray(nn_idx_1)
            if dedup_start is not None:
                self.profile_query_dedup_seconds += (
                    perf_counter() - dedup_start)

            self._trace_dci_selection(cur_id, nn_idx_0)
            self._trace_adaptive_queries(
                self.dci_db[cur_id],
                _query,
                padding_mask,
                num_neighbours,
                query_field_of_view,
                prev_num_points,
                nn_idx_0,
                native_query_elapsed,
            )

            mapping_start = perf_counter() if profile_stage else None
            padded_arrays = torch.tensor(nn_idx_0, **self._ci32)
            head_ids = torch.arange(
                self.n_kv_heads, device=padded_arrays.device).unsqueeze(1)
            if mapping_start is not None:
                self.profile_query_mapping_seconds += (
                    perf_counter() - mapping_start)

            # The number of recall slots can change while the cache transitions
            # from prefill to steady-state decoding.  In that case an old page
            # set has a different width and cannot be diffed against the new
            # selection.  Treat it as a fresh resident-set initialization.
            diff_start = perf_counter() if profile_stage else None
            if (
                self.selected_page_idx[cur_id] is None
                or self.selected_page_idx[cur_id].shape != nn_idx_0.shape
            ):
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
            if diff_start is not None:
                self.profile_query_diff_seconds += (
                    perf_counter() - diff_start)
            
            evict_num = (recall_idx >= 0).sum(1)
            kvc.ccc[b, head_ids, padded_arrays] = 0

            if postprocess_start is not None:
                self.profile_query_postprocess_seconds += (
                    perf_counter() - postprocess_start)

            return evicted_idx.contiguous(), recall_idx.contiguous(), evict_num

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

        profile_stage = self._profile_is_measured_step()
        gather_start = perf_counter() if profile_stage else None

        # [ICECACHE-DOUBLEBUF] bind this recall to a buffer slot
        if self.double_buffer:
            _cursor = self._db_cursor.get(layer_idx, 0)
            _p = _cursor % 2
            self._db_cursor[layer_idx] = _cursor + 1
            self._db_used_slot[layer_idx] = _p
            _ptb = self.cpu_transit_buffer[_p]
            _ctb = self.cuda_transit_buffer[_p]
            _cbuf = self.cuda_cast_buffer[_p]
            _evr = self._db_ev_read[_p]
            if _evr is not None:
                _evr.synchronize()
        else:
            _p = 0
            _ptb = self.cpu_transit_buffer
            _ctb = self.cuda_transit_buffer
            _cbuf = self.cuda_cast_buffer

        n_transit_pages = torch.sum(nr).item()

        if self.fast_addr:
            # [ICECACHE-FASTADDR] Build the source-address list from NumPy
            # views.  The original loop re-converts a torch tensor into a
            # NumPy index array and calls .item() on every iteration, which
            # costs hundreds of microseconds per recall and is independent
            # of how many pages are actually transferred.
            rids_cpu = rids.cpu().numpy()
            nr_cpu = nr.cpu().numpy()
            counter = 0
            for i in range(self.n_kv_heads):
                c = int(nr_cpu[i])
                if c:
                    self._src_address_buffer[counter:counter + c] = (
                        self.page_address_buffer[layer_idx][
                            b, i, rids_cpu[i, :c]])
                    counter += c
        else:
            rids_cpu = rids.cpu()
            nr_cpu = nr.cpu()

            counter = 0
            for i in range(self.n_kv_heads):
                self._src_address_buffer[counter:counter+nr_cpu[i].item()] = self.page_address_buffer[layer_idx][b, i, rids_cpu[i, :nr_cpu[i]]]
                counter += nr_cpu[i].item()
        # [ICECACHE-DIAG] end of CPU address preparation
        diag_addr_prep_end = perf_counter() if self.diag_enabled else None
        if self.diag_enabled and gather_start is not None:
            self.diag_addr_prep_seconds += (diag_addr_prep_end - gather_start)
            self._diag_collect(layer_idx, b, rids_cpu, nr_cpu)

        with torch.cuda.stream(c2g_stream):

            DCI.copy_to_buffer(self._src_address_buffer, ptr_dest=cast(_ptb[b].data_ptr(), c_void_p).value,
                               list_size=counter, update_num=self.page_size,
                               offset_s=self.n_kv_heads*self.page_size*self.head_dim,
                               offset_t=n_transit_pages*self.page_size*self.head_dim,
                               dim=self.head_dim,
                               page_size=self.page_size*self.head_dim,
                               dtype=2 if self.fp16_recall else 0)
        if gather_start is not None:
            self.profile_recall_gather_seconds += perf_counter() - gather_start
            self.profile_recall_calls += 1
            self.profile_recall_submissions += 1
            self.profile_recall_pages += n_transit_pages
        # [ICECACHE-DIAG] copy_to_buffer segment (CPU-side scattered gather)
        if self.diag_enabled and diag_addr_prep_end is not None:
            self.diag_copy_buffer_seconds += (perf_counter() - diag_addr_prep_end)
        ############################################################

        with torch.cuda.stream(c2g_stream):
            # [ICECACHE-DOUBLEBUF] slot reuse must wait for the consumer
            if self.double_buffer and self._db_ev_used[_p] is not None:
                c2g_stream.wait_event(self._db_ev_used[_p])
            dst = _ctb[:, : 2 * n_transit_pages, :]
            src = _ptb[:, : 2 * n_transit_pages, :]
            # [ICECACHE-DIAG] CUDA-event brackets for H2D and cast segments
            diag_on = self.diag_enabled and self.diag_event_budget > 0
            diag_e0 = torch.cuda.Event(enable_timing=True) if diag_on else None
            diag_e1 = torch.cuda.Event(enable_timing=True) if diag_on else None
            diag_e2 = torch.cuda.Event(enable_timing=True) if diag_on else None
            if diag_e0 is not None:
                diag_e0.record(c2g_stream)
            dst.copy_(src, non_blocking=True)
            if diag_e1 is not None:
                diag_e1.record(c2g_stream)
            if self.double_buffer:
                if self._db_ev_read[_p] is None:
                    self._db_ev_read[_p] = torch.cuda.Event()
                self._db_ev_read[_p].record(c2g_stream)

            _cbuf[:, : 2 * n_transit_pages, :].copy_(
                dst, non_blocking=True
            )
            if diag_e2 is not None:
                diag_e2.record(c2g_stream)
            if diag_e0 is not None:
                self.diag_pending.append((diag_e0, diag_e1, diag_e2))
                self.diag_event_budget -= 1
            if self.double_buffer:
                if self._db_ev_cast[_p] is None:
                    self._db_ev_cast[_p] = torch.cuda.Event()
                self._db_ev_cast[_p].record(c2g_stream)

    def recall_layer_group(self, layer_idx: int, b: int,
                           rids: Tensor, nr: Tensor):
        """Gather one layer-reuse group into [L0 K,V | L1 K,V | ...]."""
        assert self.check_reuse(layer_idx) == 0
        profile_stage = self._profile_is_measured_step()
        gather_start = perf_counter() if profile_stage else None

        layers = [layer_idx]
        for candidate in range(layer_idx + 1,
                               min(layer_idx + self.n_reuse_layers,
                                   self.n_layers)):
            if self.check_reuse(candidate) != layer_idx:
                break
            if (self.layer2budget[candidate] != self.layer2budget[layer_idx]
                    or self.kv_caches[candidate].n_real_pages
                    != self.kv_caches[candidate].budget):
                break
            layers.append(candidate)

        rids_cpu = rids.cpu()
        nr_cpu = nr.cpu()
        pages_per_layer = int(torch.sum(nr_cpu).item())
        if pages_per_layer == 0:
            for member in layers:
                self._batched_recall_slices[member] = (0, 0)
            return

        total_pages = pages_per_layer * len(layers)
        dest_k = np.empty(total_pages, dtype=np.int32)
        dest_v = np.empty(total_pages, dtype=np.int32)
        counter = 0
        for group_idx, member in enumerate(layers):
            layer_start = 2 * group_idx * pages_per_layer
            self._batched_recall_slices[member] = (
                layer_start, 2 * pages_per_layer)
            local_page = 0
            for head_idx in range(self.n_kv_heads):
                count = int(nr_cpu[head_idx].item())
                if count:
                    end = counter + count
                    self._src_address_buffer[counter:end] = (
                        self.page_address_buffer[member][
                            b, head_idx, rids_cpu[head_idx, :count]])
                    page_ids = np.arange(
                        local_page, local_page + count, dtype=np.int32)
                    dest_k[counter:end] = layer_start + page_ids
                    dest_v[counter:end] = (
                        layer_start + pages_per_layer + page_ids)
                    counter = end
                    local_page += count

        with torch.cuda.stream(self.c2g_stream):
            _dci_copy_to_buffer_batched(
                self._src_address_buffer, dest_k, dest_v,
                cast(self.cpu_transit_buffer[b].data_ptr(), c_void_p).value,
                counter,
                self.n_kv_heads * self.page_size * self.head_dim,
                self.page_size * self.head_dim,
                self.page_size * self.head_dim,
                2,
            )
            width = 2 * total_pages
            self.cuda_transit_buffer[:, :width, :].copy_(
                self.cpu_transit_buffer[:, :width, :], non_blocking=True)

        if gather_start is not None:
            self.profile_recall_gather_seconds += perf_counter() - gather_start
            self.profile_recall_calls += len(layers)
            self.profile_recall_submissions += 1
            self.profile_recall_pages += total_pages

    async def estimate_select_recall_wrapper(self, layer_idx: int, query_states: Tensor):
        return await self._loop.run_in_executor(self._loop_executor, self.estimate_select_recall, layer_idx, query_states)

    def estimate_select_recall(self, layer_idx: int, query_states: Tensor):
        thread_id = threading.get_ident()
        self._in_estimate = True
        self._estimate_thread = thread_id
        try:
            return self._estimate_select_recall_impl(
                layer_idx, query_states, thread_id)
        finally:
            self._in_estimate = False

    def _estimate_select_recall_impl(self, layer_idx, query_states, thread_id):
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
            cross_token_reused = False
            for i in range(kvc.batch_size):

                if self.check_reuse(layer_idx) == 0:
                    if self._can_reuse_cross_token(layer_idx, query_states[i]):
                        # Same layer, query hasn't moved → reuse the prior
                        # selection.  Between two non-boundary decode tokens
                        # the GPU paged cache and DCI index haven't moved,
                        # so the previously-selected pages are still
                        # resident and no CPU→GPU recall is needed.  We
                        # also skip the diff_pages_by_head bookkeeping since
                        # the resident set is identical to last token's.
                        eids = self.prev_eids.clone()
                        nr = torch.zeros_like(self.prev_nr)
                        rids = self.prev_rids.clone()
                        cross_token_reused = True
                        self.profile_cross_token_reuses += 1
                        self.layer_reuse_count[layer_idx] += 1
                        self.layers_since_refresh[layer_idx] += 1
                    else:
                        dci_start = perf_counter() if self._profile_is_measured_step() else None
                        query_d2h_start = dci_start
                        query_states_cpu = query_states[i].cpu().detach().transpose(0, 1)
                        if query_d2h_start is not None:
                            self.profile_query_d2h_seconds += (
                                perf_counter() - query_d2h_start)
                        eids, rids, nr = self._DCI_query(
                            i, layer_idx, query_states_cpu)
                        if dci_start is not None:
                            self.profile_dci_seconds += perf_counter() - dci_start
                            self.profile_dci_calls += 1
                        self._update_query_signature(layer_idx, query_states[i])

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

                if eids is not None and not cross_token_reused:
                    if (self.batch_layer_recall
                            and self.check_reuse(layer_idx) == 0):
                        self.recall_layer_group(layer_idx, i, rids, nr)
                    elif (not self.batch_layer_recall
                          or layer_idx not in self._batched_recall_slices):
                        self.recall(layer_idx, i, rids, nr)

                    metadata_start = (
                        perf_counter() if self._profile_is_measured_step()
                        else None
                    )
                    if self.check_reuse(layer_idx) == 0:
                        self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(
                            self.dci_db[layer_idx].get_valid_entries(self.selected_page_idx[layer_idx]), **self._i32).T
                    else:
                        self.page_valid_entries[layer_idx][ns: ns + self.n_dci_pages - self.layer2topk[layer_idx]] = torch.tensor(
                            self.dci_db[reuse_id].get_valid_entries(self.selected_page_idx[reuse_id]), **self._i32).T
                    if metadata_start is not None:
                        self.profile_page_metadata_seconds += (
                            perf_counter() - metadata_start)

                recall_wait_start = (
                    perf_counter() if self._profile_is_measured_step()
                    else None
                )
                # [ICECACHE-DOUBLEBUF] with two slots the consumer stream
                # waits on an event, so the host does not block here
                if not self.double_buffer:
                    c2g_stream.synchronize()
                if recall_wait_start is not None:
                    self.profile_recall_wait_seconds += (
                        perf_counter() - recall_wait_start)
                # [ICECACHE-DIAG] read H2D/cast event timings now the stream drained
                if self.diag_enabled and self.diag_pending:
                    for _e0, _e1, _e2 in self.diag_pending:
                        self.diag_h2d_ms += _e0.elapsed_time(_e1)
                        self.diag_cast_ms += _e1.elapsed_time(_e2)
                    self.diag_h2d_count += len(self.diag_pending)
                    self.diag_cast_count += len(self.diag_pending)
                    self.diag_pending = []

        return eids, nr

    def scatter_pages(self, layer_idx, eids, nr):
        if self.double_buffer:
            # [ICECACHE-DOUBLEBUF] consume this layer's used slot
            _p = self._db_used_slot.get(layer_idx, 0)
            _cur = torch.cuda.current_stream()
            if self._db_ev_cast[_p] is not None:
                _cur.wait_event(self._db_ev_cast[_p])
            _cpp.scatter_pages(self.cuda_cast_buffer[_p],
                               self.kv_caches[layer_idx].pool.buffer,
                               eids, nr)
            if self._db_ev_used[_p] is None:
                self._db_ev_used[_p] = torch.cuda.Event()
            self._db_ev_used[_p].record(_cur)
            return
        transit = self.cuda_cast_buffer
        if self.batch_layer_recall and layer_idx in self._batched_recall_slices:
            start, width = self._batched_recall_slices[layer_idx]
            transit = transit[:, start:start + width, :]
        _cpp.scatter_pages(transit,
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
        update_start = (
            perf_counter() if self._profile_is_measured_step() else None
        )
        win_kvc = self.offload_win_caches[layer_idx]
        kvc = self.kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for i in range(win_kvc.batch_size):
            pack_start = (
                perf_counter() if self._profile_is_measured_step() else None
            )
            sub_kv_states = win_kvc[i, :self.num_evict_win].permute(1, 0, 2, 3, 4).permute(
                0, 2, 1, 3, 4).reshape(2, self.n_kv_heads, -1, self.head_dim)
            if pack_start is not None:
                self.profile_index_pack_seconds += perf_counter() - pack_start
            self._DCI_add(i, layer_idx, sub_kv_states[0], sub_kv_states[1])
        if update_start is not None:
            self.profile_index_update_seconds += perf_counter() - update_start
            self.profile_index_update_calls += 1


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
