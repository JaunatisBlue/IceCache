"""Multi-head M-DCI backend for one layer of semantic-page descriptors."""

from __future__ import annotations

import numpy as np

from .unified_page_index import PageCandidate, SessionMaterialization, UnifiedPageCatalog


class LayerDCIBackend:
    """Index equal-sized per-head catalogs in one native multi-instance DCI."""

    def __init__(
        self,
        catalogs: dict[int, UnifiedPageCatalog],
        *,
        seed: int = 0,
        promotion_prob: float = 0.01,
        promotion_prob_subseq: float = 0.2,
        parallel_level: int = 0,
    ) -> None:
        if len(catalogs) < 2:
            raise ValueError("native DCI backend requires at least two KV heads")
        from dciknn import DCI

        self.heads = sorted(catalogs)
        if self.heads != list(range(len(self.heads))):
            raise ValueError("head catalogs must be contiguous from zero")
        self.catalogs = catalogs
        if len({catalog.layer_idx for catalog in catalogs.values()}) != 1:
            raise ValueError("one DCI backend cannot mix transformer layers")
        for head, catalog in catalogs.items():
            if catalog.head_idx != head or not catalog._pages:
                raise ValueError("each head needs a non-empty matching catalog")
        counts = {len(catalog._pages) for catalog in catalogs.values()}
        if len(counts) != 1:
            raise ValueError("all heads must contain the same number of descriptors")
        self.num_points = counts.pop()
        self.page_ids = {head: list(catalogs[head]._pages) for head in self.heads}
        matrices = []
        for head in self.heads:
            matrix = np.stack(
                [catalogs[head]._pages[page_id].descriptor for page_id in self.page_ids[head]]
            ).astype(np.float32)
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            matrices.append(matrix)
        if len({matrix.shape[1] for matrix in matrices}) != 1:
            raise ValueError("all head descriptors must share one dimension")
        self.dimension = matrices[0].shape[1]
        self.descriptors = np.ascontiguousarray(np.concatenate(matrices, axis=0))
        self.parallel_level = parallel_level
        rng = np.random.default_rng(seed)
        projection = rng.normal(size=self.dimension + 1).astype(np.float32)
        projection /= np.linalg.norm(projection)
        self.projection = np.ascontiguousarray(projection)
        self.index = DCI(
            self.dimension,
            1,
            1,
            promotion_prob=promotion_prob,
            promotion_prob_subseq=promotion_prob_subseq,
            num_points=self.num_points,
            init=True,
            num_inst=len(self.heads),
            transform=True,
            parallel_level=parallel_level,
            proj_vec=self.projection,
        )
        projected = np.ascontiguousarray(
            (self.descriptors @ self.projection[:-1]).reshape(-1, 1).astype(np.float32)
        )
        dummy_queries = np.ascontiguousarray(
            self.descriptors.reshape(len(self.heads), self.num_points, self.dimension)[:, 0]
        )
        self.index.add_query_at_end(
            self.descriptors,
            dummy_queries,
            self.descriptors,
            np.ones(len(self.heads) * self.num_points, dtype=np.bool_),
            num_points=self.num_points,
            num_neighbours=1,
            c_num_to_visit=self.num_points,
            c_num_to_retrieve=-1,
            c_prop_to_visit=1.0,
            c_prop_to_retrieve=1.0,
            c_field_of_view=30,
            q_num_to_visit=self.num_points,
            q_field_of_view=30,
            q_num_to_retrieve=-1,
            q_prop_to_visit=1.0,
            q_prop_to_retrieve=0.8,
            transform=True,
            parallel_level=parallel_level,
            causal=False,
            random=False,
            do_query=False,
            track=False,
            update_addr=False,
            data_proj=projected,
            ratio=1,
            interval=120,
            X=30,
            anchor_threshold=0.9,
        )
        self.min_leaves = int(np.min(self.index.num_leaves))
        if self.min_leaves < 2:
            raise ValueError("DCI produced too few leaves for safe page retrieval")

    def query_page_ids(
        self,
        queries: np.ndarray,
        *,
        num_neighbours: int,
        field_of_view: int = 30,
        prop_to_retrieve: float = 0.8,
    ) -> dict[int, list[str]]:
        if (
            isinstance(num_neighbours, (bool, np.bool_))
            or not isinstance(num_neighbours, (int, np.integer))
        ):
            raise TypeError("num_neighbours must be an integer")
        if int(num_neighbours) <= 0:
            return {head: [] for head in self.heads}
        num_neighbours = int(num_neighbours)
        if not np.isfinite(prop_to_retrieve) or not 0 < prop_to_retrieve <= 1:
            raise ValueError("prop_to_retrieve must be finite and in (0, 1]")
        # Native result buffers are sized by prop_to_retrieve * num_leaves.
        # Asking for more neighbours corrupts memory instead of raising. Keep
        # the call within that ABI capacity and below the total leaf count.
        native_capacity = min(
            self.min_leaves - 1,
            int(np.floor(prop_to_retrieve * self.min_leaves)),
        )
        if native_capacity < 1:
            raise ValueError("prop_to_retrieve leaves no safe result capacity")
        k = min(num_neighbours, self.num_points, native_capacity)
        queries = np.asarray(queries, dtype=np.float32)
        if (
            queries.ndim != 2
            or queries.shape[0] == 0
            or queries.shape[1] != self.dimension
            or queries.shape[0] % len(self.heads) != 0
        ):
            raise ValueError(
                "queries must have shape [n_query_heads, dimension], with "
                "n_query_heads divisible by the number of KV heads"
            )
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        if not np.all(np.isfinite(queries)) or np.any(norms == 0):
            raise ValueError("queries must contain finite, nonzero vectors")
        query_ratio = queries.shape[0] // len(self.heads)
        queries = np.ascontiguousarray(queries / norms)
        raw, counts = self.index.query(
            queries,
            np.ones(queries.shape[0], dtype=np.bool_),
            num_neighbours=k,
            field_of_view=field_of_view,
            num_to_visit=self.num_points,
            num_to_retrieve=-1,
            prop_to_visit=1.0,
            prop_to_retrieve=prop_to_retrieve,
            parallel_level=self.parallel_level,
            ratio=1,
        )
        raw_array = np.asarray(raw)
        if not np.issubdtype(raw_array.dtype, np.integer):
            raise TypeError("native DCI result buffer must contain integer IDs")
        expected_size = queries.shape[0] * 2 * k
        if raw_array.size != expected_size:
            raise RuntimeError(
                f"native DCI returned {raw_array.size} IDs, expected {expected_size}"
            )
        rows = raw_array.reshape(queries.shape[0], 2, k)
        count_array = np.asarray(counts)
        if (
            not np.issubdtype(count_array.dtype, np.integer)
            or count_array.shape != (queries.shape[0],)
            or np.any(count_array < 0)
            or np.any(count_array > k)
        ):
            raise RuntimeError("native DCI returned invalid per-query result counts")

        # The extension ABI stores [leaf IDs | point IDs] for each query.
        # Validate both halves before translating point IDs.  Besides catching
        # corrupt/uninitialised slots, this fails closed if a future native ABI
        # swaps the two halves instead of silently treating leaves as pages.
        num_leaves = np.asarray(self.index.num_leaves).reshape(len(self.heads))
        point_rows: list[np.ndarray] = []
        for query_idx, raw_count in enumerate(count_array):
            count = int(raw_count)
            head = query_idx // query_ratio
            leaves = rows[query_idx, 0, :count]
            points = rows[query_idx, 1, :count]
            if count and (
                np.any(leaves < 0)
                or np.any(leaves >= num_leaves[head])
                or np.any(points < 0)
                or np.any(points >= self.num_points)
            ):
                raise RuntimeError(
                    "native DCI result violates [leaf IDs | point IDs] layout"
                )
            point_rows.append(points)
        result = {}
        for head in self.heads:
            seen: set[int] = set()
            merged = []
            for query_idx in range(
                head * query_ratio, (head + 1) * query_ratio
            ):
                for raw_idx in point_rows[query_idx]:
                    idx = int(raw_idx)
                    if idx not in seen:
                        seen.add(idx)
                        merged.append(self.page_ids[head][idx])
            result[head] = merged
        return result

    def search(
        self,
        queries: np.ndarray,
        *,
        session_id: str,
        candidate_limit: int,
        overfetch: int = 2,
        error_weight: float = 0.0,
        rag_bias: float = 0.0,
        materialization: SessionMaterialization | None = None,
    ) -> dict[int, list[PageCandidate]]:
        ids_by_head = self.query_page_ids(
            queries,
            num_neighbours=min(self.num_points, candidate_limit * overfetch),
            field_of_view=max(30, candidate_limit * overfetch),
        )
        resolver = materialization.resolve if materialization else None
        query_ratio = queries.shape[0] // len(self.heads)
        return {
            head: self.catalogs[head].search(
                queries[head * query_ratio : (head + 1) * query_ratio],
                session_id=session_id,
                candidate_limit=candidate_limit,
                error_weight=error_weight,
                rag_bias=rag_bias,
                cost_resolver=resolver,
                page_ids=ids_by_head[head],
            )
            for head in self.heads
        }

    def search_partitioned(
        self,
        queries: np.ndarray,
        *,
        session_id: str,
        session_catalogs: dict[int, UnifiedPageCatalog] | None,
        candidate_limit: int,
        overfetch: int = 2,
        error_weight: float = 0.0,
        rag_bias: float = 0.0,
        materialization: SessionMaterialization | None = None,
    ) -> dict[int, list[PageCandidate]]:
        """Merge global DCI candidates with one session's exact delta.

        Prompt pages are intentionally absent from the shared native tree.
        Otherwise foreign sessions can consume its bounded result buffer and
        leave too few visible candidates after filtering. The private delta is
        expected to be small and short-lived; a second ANN backend can replace
        its exact search without changing this merge contract.
        """
        global_candidates = self.search(
            queries,
            session_id=session_id,
            candidate_limit=candidate_limit,
            overfetch=overfetch,
            error_weight=error_weight,
            rag_bias=rag_bias,
            materialization=materialization,
        )
        if not session_catalogs:
            return global_candidates
        if set(session_catalogs) != set(self.heads):
            raise ValueError("session catalogs must cover the backend's KV heads")
        resolver = materialization.resolve if materialization else None
        query_ratio = queries.shape[0] // len(self.heads)
        merged: dict[int, list[PageCandidate]] = {}
        for head in self.heads:
            catalog = session_catalogs[head]
            if catalog.layer_idx != self.catalogs[head].layer_idx or catalog.head_idx != head:
                raise ValueError("session catalog layer/head does not match backend")
            private = catalog.search(
                queries[head * query_ratio : (head + 1) * query_ratio],
                session_id=session_id,
                candidate_limit=candidate_limit,
                error_weight=error_weight,
                rag_bias=rag_bias,
                cost_resolver=resolver,
            )
            candidates = global_candidates[head] + private
            candidates.sort(key=lambda item: (-item.utility, item.page.page_id))
            merged[head] = candidates[:candidate_limit]
        return merged
