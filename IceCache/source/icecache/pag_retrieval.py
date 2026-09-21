"""PAG token search with DCI's current token-to-page layout."""

import threading
from pathlib import Path
from time import perf_counter

import numpy as np


class InsufficientPagesError(RuntimeError):
    """The current PAG candidate limit cannot fill the page budget."""


class PagPageSelector:
    def __init__(self, layer, keys, token_to_page, reserve, max_search_k, ef_search,
                 initial_factor=4, ef_construction=200, target_degree=16,
                 projection_levels=64, executor=None):
        import pag
        from concurrent.futures import ThreadPoolExecutor

        if (reserve < 0 or max_search_k <= 0 or ef_search <= 0 or initial_factor <= 0 or
                ef_construction <= 0 or target_degree <= 0 or
                projection_levels <= 0 or projection_levels % 8):
            raise ValueError('Invalid PAG search or capacity configuration')
        self.layer = layer
        self.keys = np.ascontiguousarray(keys, dtype=np.float32)
        self.mapping = np.ascontiguousarray(token_to_page, dtype=np.int32)
        if (self.keys.ndim != 3 or not all(self.keys.shape) or
                self.mapping.shape != self.keys.shape[:2]):
            raise ValueError('PAG keys and token mapping have incompatible shapes')
        if np.any(self.mapping < 0) or not np.all(np.isfinite(self.keys)):
            raise ValueError('Invalid prefill keys or DCI token mapping')
        self.count = self.keys.shape[1]
        self.capacity = self.count + reserve
        self.max_search_k = max_search_k
        self.ef_search = ef_search
        self.initial_factor = initial_factor
        # Keep every search and its key/page lookup atomic with online inserts.
        self.layer_lock = threading.Lock()
        self.locks = [threading.Lock() for _ in self.keys]
        self.indexes = []
        self.query_count = 0
        self.retry_count = 0
        self.shortfall_count = 0
        self.insert_count = 0
        self.build_seconds = 0.0
        self.search_seconds = []
        self.aggregate_seconds = []
        self.insert_seconds = []
        for head, vectors in enumerate(self.keys):
            options = pag.BuildOptions()
            options.index_path = str(Path('/tmp') / f'icecache_pag_{id(self)}_{layer}_{head}')
            options.metric = pag.Metric.MaximumInnerProduct
            options.mode = pag.IndexMode.Online
            options.max_elements = self.capacity
            options.max_search_k = max_search_k
            options.ef_construction = ef_construction
            options.target_degree = target_degree
            options.projection_levels = projection_levels
            index = pag.Index()
            start = perf_counter()
            index.build(np.ascontiguousarray(vectors), options)
            self.build_seconds += perf_counter() - start
            self.indexes.append(index)
        # One task per KV head is submitted to this pool. A shared executor is
        # passed in by InferState (34 layers worth of private pools would be far
        # too many threads); only create one when running standalone.
        self._private_executor = None
        if executor is None:
            self._private_executor = ThreadPoolExecutor(
                max_workers=len(self.indexes))
            executor = self._private_executor
        self.executor = executor
        self._check_labels()

    def close(self):
        """Release the pool this selector owns, if any. Idempotent."""
        if self._private_executor is not None:
            self._private_executor.shutdown(wait=True)
            self._private_executor = None
            self.executor = None

    def _check_labels(self):
        for head, index in enumerate(self.indexes):
            with self.locks[head]:
                labels, _ = index.search(self.keys[head, :1], top_k=min(10, self.count),
                                         ef_search=self.ef_search)
            labels = np.asarray(labels)
            if (labels.size == 0 or not np.all(np.isfinite(labels)) or
                    np.any((labels < 0) | (labels >= self.count))):
                raise ValueError('PAG build returned invalid token labels')

    def select(self, queries, budget, page_size):
        with self.layer_lock:
            return self._select_locked(queries, budget, page_size)

    def _select_locked(self, queries, budget, page_size):
        queries = np.asarray(queries)
        if (queries.ndim != 2 or queries.shape[1] != self.keys.shape[2] or
                queries.shape[0] == 0 or queries.shape[0] % len(self.indexes) or
                not np.all(np.isfinite(queries)) or budget <= 0 or page_size <= 0):
            raise ValueError('Invalid PAG page budget or GQA group')
        self.query_count += 1
        group = queries.shape[0] // len(self.indexes)
        top_m = min(self.count, self.max_search_k,
                    max(self.initial_factor * budget, budget * page_size // 2))
        # Submit each KV head's full pipeline (search, key gather, scoring, page
        # aggregation) as one task. pag.Index.search releases the GIL, so the
        # heads overlap; pag.Index.build holds it, which is why builds stay
        # serial. Workers only take self.locks[head], never self.layer_lock,
        # which this thread holds, so there is no lock cycle.
        futures = [
            self.executor.submit(
                self._select_head, head,
                np.ascontiguousarray(
                    queries[head * group:(head + 1) * group], dtype=np.float32),
                budget, top_m)
            for head in range(len(self.indexes))
        ]
        result = np.empty((len(self.indexes), budget), dtype=np.int32)
        failure = None
        for head, future in enumerate(futures):
            try:
                page_ids, counts = future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if failure is None:
                    failure = exc
                continue
            # Counters are accumulated per head and summed here on the main
            # thread, so no worker mutates shared state.
            self.retry_count += counts['retries']
            self.shortfall_count += counts['shortfalls']
            self.search_seconds.extend(counts['search'])
            self.aggregate_seconds.extend(counts['aggregate'])
            if page_ids is None:
                if failure is None:
                    failure = InsufficientPagesError(
                        'PAG candidates cover fewer pages than the budget')
            else:
                result[head] = page_ids
        if failure is not None:
            raise failure
        return result

    def _select_head(self, head, q, budget, top_m):
        """Run one KV head's adaptive search and page aggregation to completion."""
        index = self.indexes[head]
        search_seconds = []
        aggregate_seconds = []
        retries = 0
        cap = min(self.count, self.max_search_k)
        while True:
            start = perf_counter()
            with self.locks[head]:
                labels, _ = index.search(q, top_k=top_m, ef_search=self.ef_search)
            search_seconds.append(perf_counter() - start)
            start = perf_counter()
            labels = np.asarray(labels, dtype=np.int64).reshape(q.shape[0], -1)
            if labels.shape[1] == 0 or np.any((labels < 0) | (labels >= self.count)):
                raise ValueError('PAG returned an invalid token label')
            pages = self.mapping[head, labels]
            scores = np.einsum('gmd,gd->gm', self.keys[head, labels], q)
            page_ids = self.aggregate(pages, scores, budget)
            aggregate_seconds.append(perf_counter() - start)
            if page_ids is not None:
                return page_ids, {'retries': retries, 'shortfalls': 0,
                                  'search': search_seconds,
                                  'aggregate': aggregate_seconds}
            if top_m >= cap:
                return None, {'retries': retries, 'shortfalls': 1,
                              'search': search_seconds,
                              'aggregate': aggregate_seconds}
            retries += 1
            top_m = min(top_m * 2, self.count, self.max_search_k)

    @staticmethod
    def aggregate(pages, scores, budget):
        """Best score per unique page, ordered by score descending then page id.

        Returns None when fewer than ``budget`` distinct pages were found, which
        is the signal to retry with a larger top_m.
        """
        flat_pages = pages.ravel()
        flat_scores = scores.ravel().astype(np.float64)
        valid = flat_pages >= 0
        if not valid.all():
            flat_pages = flat_pages[valid]
            flat_scores = flat_scores[valid]
        uniq, inv = np.unique(flat_pages, return_inverse=True)
        best = np.full(uniq.size, -np.inf, dtype=np.float64)
        np.maximum.at(best, inv, flat_scores)
        if uniq.size < budget:
            return None
        order = np.lexsort((uniq, -best))[:budget]
        return uniq[order].astype(np.int32)

    def insert(self, new_keys, token_to_page):
        with self.layer_lock:
            self._insert_locked(new_keys, token_to_page)

    def _insert_locked(self, new_keys, token_to_page):
        new_keys = np.ascontiguousarray(new_keys, dtype=np.float32)
        if new_keys.ndim != 3 or new_keys.shape[0] != len(self.indexes) or new_keys.shape[2] != self.keys.shape[2]:
            raise ValueError('New PAG keys have incompatible shape')
        if not np.all(np.isfinite(new_keys)):
            raise ValueError('New PAG keys contain NaN or infinity')
        n = new_keys.shape[1]
        if n == 0:
            return
        if self.count + n > self.capacity:
            raise ValueError('PAG online capacity exceeded')
        mapping = np.ascontiguousarray(token_to_page, dtype=np.int32)
        if mapping.shape != (len(self.indexes), self.count + n):
            raise ValueError('DCI token mapping does not match PAG labels')
        if np.any(mapping < 0):
            raise ValueError('DCI contains unmapped tokens')
        labels = np.arange(self.count, self.count + n, dtype=np.int64)
        start = perf_counter()
        # DCI page splits may remap old tokens. Publish the complete mapping only
        # after every PAG head has accepted the new labels, under the layer lock.
        for head, index in enumerate(self.indexes):
            with self.locks[head]:
                index.insert_batch(np.ascontiguousarray(new_keys[head]), labels)
        self.keys = np.concatenate((self.keys, new_keys), axis=1)
        self.mapping = mapping
        self.count += n
        self.insert_count += n
        self.insert_seconds.append(perf_counter() - start)
