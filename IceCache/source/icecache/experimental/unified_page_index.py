"""Reference control plane for a unified RAG and session-KV page index.

This deliberately uses exact cosine search rather than DCI.  It defines the
metadata, visibility, scoring, and empirical chance-constraint contracts that
the IceCache runtime can later implement with DCI and CUDA page movement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Literal

import numpy as np


PageSource = Literal["rag", "prompt"]


def _readonly_copy(value: np.ndarray, dtype) -> np.ndarray:
    """Own and freeze metadata arrays so cached derivatives stay valid."""
    array = np.array(value, dtype=dtype, copy=True, order="C")
    frozen = np.frombuffer(array.tobytes(), dtype=array.dtype)
    return frozen.reshape(array.shape)


@dataclass(frozen=True, eq=False)
class SemanticPage:
    page_id: str
    source: PageSource
    descriptor: np.ndarray
    error_radius: float
    load_cost_samples: np.ndarray
    document_id: str | None = None
    segment_id: str | None = None
    session_id: str | None = None
    materialized: bool = False
    layer_idx: int = 0
    head_idx: int = 0

    def __post_init__(self) -> None:
        descriptor = _readonly_copy(self.descriptor, np.float32)
        costs = _readonly_copy(self.load_cost_samples, np.float64)
        if descriptor.ndim != 1 or not descriptor.size:
            raise ValueError("descriptor must be a non-empty vector")
        if costs.ndim != 1 or not costs.size or np.any(costs < 0):
            raise ValueError("load_cost_samples must be a non-empty nonnegative vector")
        if self.error_radius < 0:
            raise ValueError("error_radius must be nonnegative")
        if self.source == "prompt" and not self.session_id:
            raise ValueError("prompt pages must carry session_id")
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "load_cost_samples", costs)


@dataclass(frozen=True, eq=False)
class PageCandidate:
    page: SemanticPage
    relevance: float
    utility: float
    load_cost_samples: np.ndarray
    materialization_state: str

    def __post_init__(self) -> None:
        costs = _readonly_copy(self.load_cost_samples, np.float64)
        if costs.ndim != 1 or not costs.size or np.any(costs < 0):
            raise ValueError("candidate load costs must be a non-empty nonnegative vector")
        object.__setattr__(self, "load_cost_samples", costs)


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[PageCandidate, ...]
    cost_quantile: float
    empirical_violation_probability: float
    total_utility: float
    source_counts: dict[str, int] = field(default_factory=dict)
    feasible: bool = True
    prompt_floor_satisfied: bool = True
    head_floor_satisfied: bool = True


class UnifiedPageCatalog:
    def __init__(self, layer_idx: int = 0, head_idx: int = 0) -> None:
        self._pages: dict[str, SemanticPage] = {}
        self._descriptor_norms: dict[str, float] = {}
        self._dimension: int | None = None
        self._n_scenarios: int | None = None
        self.layer_idx = layer_idx
        self.head_idx = head_idx

    def add(self, page: SemanticPage) -> None:
        if page.layer_idx != self.layer_idx:
            raise ValueError(
                f"page belongs to layer {page.layer_idx}, catalog is layer {self.layer_idx}"
            )
        if page.head_idx != self.head_idx:
            raise ValueError(
                f"page belongs to head {page.head_idx}, catalog is head {self.head_idx}"
            )
        if page.page_id in self._pages:
            raise ValueError(f"duplicate page_id: {page.page_id}")
        if self._dimension is None:
            self._dimension = page.descriptor.size
            self._n_scenarios = page.load_cost_samples.size
        if page.descriptor.size != self._dimension:
            raise ValueError("all descriptors must use one shared vector space")
        if page.load_cost_samples.size != self._n_scenarios:
            raise ValueError("cost samples must share aligned scenario IDs")
        norm = float(np.linalg.norm(page.descriptor))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("page descriptor must have a finite nonzero norm")
        self._pages[page.page_id] = page
        self._descriptor_norms[page.page_id] = norm

    def extend(self, pages: Iterable[SemanticPage]) -> None:
        for page in pages:
            self.add(page)

    def remove(self, page_id: str) -> None:
        del self._pages[page_id]
        del self._descriptor_norms[page_id]

    def remove_session(self, session_id: str) -> int:
        """Remove all private pages owned by a completed session."""
        doomed = [
            page_id
            for page_id, page in self._pages.items()
            if page.source == "prompt" and page.session_id == session_id
        ]
        for page_id in doomed:
            del self._pages[page_id]
            del self._descriptor_norms[page_id]
        return len(doomed)

    def source_counts(self) -> dict[str, int]:
        return {
            source: sum(page.source == source for page in self._pages.values())
            for source in ("prompt", "rag")
        }

    def search(
        self,
        query: np.ndarray,
        *,
        session_id: str,
        candidate_limit: int,
        error_weight: float = 0.0,
        rag_bias: float = 0.0,
        cost_resolver: Callable[[SemanticPage], tuple[str, np.ndarray]] | None = None,
        page_ids: Iterable[str] | None = None,
    ) -> list[PageCandidate]:
        """Run one mixed-source search while enforcing private-page visibility."""
        allowed_ids: tuple[str, ...] | None = None
        if page_ids is not None:
            if isinstance(page_ids, (str, bytes)):
                raise TypeError("page_ids must be an iterable of page IDs, not a string")
            requested = tuple(page_ids)
            if any(not isinstance(page_id, str) or not page_id for page_id in requested):
                raise ValueError("page_ids must contain non-empty strings")
            allowed_ids = tuple(dict.fromkeys(requested))
            missing = [page_id for page_id in allowed_ids if page_id not in self._pages]
            if missing:
                raise KeyError(f"unknown page_ids: {missing}")
        resolver_owner = getattr(cost_resolver, "__self__", None)
        resolver_session = getattr(resolver_owner, "session_id", None)
        if resolver_session is not None and resolver_session != session_id:
            raise ValueError(
                f"cost resolver belongs to session {resolver_session!r}, "
                f"but query belongs to {session_id!r}"
            )
        if not self._pages:
            return []
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != self._dimension:
            raise ValueError("query dimension does not match catalog")
        qnorm = np.linalg.norm(query, axis=1, keepdims=True)
        if np.any(qnorm == 0):
            raise ValueError("query must be nonzero")
        query = query / qnorm
        candidates = []
        pages = (
            self._pages.values()
            if allowed_ids is None
            else (self._pages[page_id] for page_id in allowed_ids)
        )
        for page in pages:
            if page.source == "prompt" and page.session_id != session_id:
                continue
            pnorm = self._descriptor_norms[page.page_id]
            # Under grouped-query attention, several Q heads share one KV
            # head. A page is useful when any member query retrieves it, which
            # matches IceCache's per-Q-head search followed by union/dedup.
            relevance = float(
                np.max(query @ (page.descriptor / max(pnorm, 1e-12)))
            )
            utility = relevance - error_weight * page.error_radius
            if page.source == "rag":
                utility += rag_bias
            if cost_resolver is None:
                state = "materialized" if page.materialized else "default"
                costs = page.load_cost_samples
            else:
                state, costs = cost_resolver(page)
            candidates.append(PageCandidate(page, relevance, utility, costs, state))
        candidates.sort(key=lambda item: (-item.utility, item.page.page_id))
        return candidates[:candidate_limit]


def empirical_quantile(samples: np.ndarray, confidence: float) -> float:
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    return float(np.quantile(samples, confidence, method="higher"))


def _summarize(
    selected: list[PageCandidate],
    budget: float,
    confidence: float,
    *,
    max_pages: int | None = None,
    min_prompt_pages: int = 0,
    min_pages_per_head: int = 0,
    required_heads: tuple[int, ...] = (),
) -> SelectionResult:
    if selected:
        costs = np.stack([item.load_cost_samples for item in selected]).sum(axis=0)
    else:
        costs = np.zeros(1, dtype=np.float64)
    source_counts = {
        source: sum(item.page.source == source for item in selected)
        for source in ("prompt", "rag")
    }
    cost_quantile = empirical_quantile(costs, confidence)
    prompt_floor_satisfied = source_counts["prompt"] >= min_prompt_pages
    head_counts = {
        head: sum(item.page.head_idx == head for item in selected)
        for head in required_heads
    }
    head_floor_satisfied = all(
        count >= min_pages_per_head for count in head_counts.values()
    )
    return SelectionResult(
        selected=tuple(selected),
        cost_quantile=cost_quantile,
        empirical_violation_probability=float(np.mean(costs > budget)),
        total_utility=float(sum(item.utility for item in selected)),
        source_counts=source_counts,
        feasible=(
            cost_quantile <= budget
            and (max_pages is None or len(selected) <= max_pages)
            and prompt_floor_satisfied
            and head_floor_satisfied
        ),
        prompt_floor_satisfied=prompt_floor_satisfied,
        head_floor_satisfied=head_floor_satisfied,
    )


def select_pages(
    candidates: list[PageCandidate],
    *,
    max_pages: int,
    resource_budget: float,
    confidence: float,
    min_prompt_pages: int = 0,
    min_pages_per_head: int = 0,
    local_search: bool = True,
) -> SelectionResult:
    """Greedily maximize mixed-source utility under an empirical chance bound.

    Cost samples are aligned scenarios rather than independent per-page draws,
    so congestion or shared-transfer tails remain correlated when pages are
    combined.  A candidate set is feasible when its requested cost quantile is
    no greater than ``resource_budget``.
    """
    if (
        max_pages < 0
        or min_prompt_pages < 0
        or min_prompt_pages > max_pages
        or min_pages_per_head < 0
    ):
        raise ValueError("invalid page limits")
    if resource_budget < 0:
        raise ValueError("resource_budget must be nonnegative")
    if not candidates or max_pages == 0:
        return _summarize([], resource_budget, confidence,
            max_pages=max_pages, min_prompt_pages=min_prompt_pages,
            min_pages_per_head=min_pages_per_head)
    n_scenarios = candidates[0].load_cost_samples.size
    if any(item.load_cost_samples.size != n_scenarios for item in candidates):
        raise ValueError("candidate costs must use aligned scenarios")

    selected: list[PageCandidate] = []
    selected_keys: set[tuple[object, ...]] = set()
    total_costs = np.zeros(n_scenarios, dtype=np.float64)
    required_heads = tuple(sorted({item.page.head_idx for item in candidates}))
    if len(required_heads) * min_pages_per_head > max_pages:
        raise ValueError("per-head floor exceeds max_pages")

    def feasible(item: PageCandidate) -> bool:
        return empirical_quantile(total_costs + item.load_cost_samples, confidence) <= resource_budget

    # Preserve a minimum width for every KV head before globally competing for
    # the remaining transfer budget. Current IceCache kernels require a dense
    # per-head page tensor even though the resource constraint is global.
    for head in required_heads:
        head_candidates = [item for item in candidates if item.page.head_idx == head]
        for item in sorted(
            head_candidates,
            key=lambda candidate: _priority(candidate, confidence),
            reverse=True,
        ):
            if sum(chosen.page.head_idx == head for chosen in selected) >= min_pages_per_head:
                break
            if len(selected) >= max_pages:
                break
            if (
                item.utility >= 0
                and _candidate_key(item) not in selected_keys
                and feasible(item)
            ):
                selected.append(item)
                selected_keys.add(_candidate_key(item))
                total_costs += item.load_cost_samples

    # Protect a configurable amount of native context before RAG pages consume
    # the shared budget. Within each phase, utility per tail-cost is the greedy
    # priority, while negative-utility pages are never forced in.
    prompt_candidates = [item for item in candidates if item.page.source == "prompt"]
    for item in sorted(
        prompt_candidates,
        key=lambda candidate: _priority(candidate, confidence),
        reverse=True,
    ):
        if sum(chosen.page.source == "prompt" for chosen in selected) >= min_prompt_pages:
            break
        if len(selected) >= max_pages:
            break
        if (
            item.utility >= 0
            and _candidate_key(item) not in selected_keys
            and feasible(item)
        ):
            selected.append(item)
            selected_keys.add(_candidate_key(item))
            total_costs += item.load_cost_samples

    remaining = [
        item for item in candidates if _candidate_key(item) not in selected_keys
    ]
    for item in sorted(
        remaining,
        key=lambda candidate: _priority(candidate, confidence),
        reverse=True,
    ):
        if len(selected) >= max_pages:
            break
        if item.utility >= 0 and feasible(item):
            selected.append(item)
            selected_keys.add(_candidate_key(item))
            total_costs += item.load_cost_samples
    if local_search:
        selected = _improve_selection(
            selected,
            candidates,
            max_pages=max_pages,
            resource_budget=resource_budget,
            confidence=confidence,
            min_prompt_pages=min_prompt_pages,
            min_pages_per_head=min_pages_per_head,
            required_heads=required_heads,
        )
    return _summarize(selected, resource_budget, confidence,
        max_pages=max_pages, min_prompt_pages=min_prompt_pages,
        min_pages_per_head=min_pages_per_head,
        required_heads=required_heads)


def _improve_selection(
    selected: list[PageCandidate],
    candidates: list[PageCandidate],
    *,
    max_pages: int,
    resource_budget: float,
    confidence: float,
    min_prompt_pages: int,
    min_pages_per_head: int,
    required_heads: tuple[int, ...],
) -> list[PageCandidate]:
    """Apply improving add and one-for-two exchange moves.

    The one-for-two neighborhood repairs the common knapsack failure where one
    attractive high-cost page blocks two jointly better medium-cost pages. Its
    worst-case work is cubic in the ANN candidate count, independent of the
    number of empirical cost scenarios except for feasibility evaluation.
    """
    current = list(selected)
    while True:
        current_summary = _summarize(
            current,
            resource_budget,
            confidence,
            max_pages=max_pages,
            min_prompt_pages=min_prompt_pages,
            min_pages_per_head=min_pages_per_head,
            required_heads=required_heads,
        )
        best = None
        best_utility = current_summary.total_utility if current_summary.feasible else -np.inf
        removals: list[PageCandidate | None] = [None, *current]
        for removed in removals:
            base = [item for item in current if item is not removed]
            base_ids = {_candidate_key(item) for item in base}
            available = [
                item for item in candidates if _candidate_key(item) not in base_ids
            ]
            if removed is not None:
                removed_key = _candidate_key(removed)
                available = [
                    item for item in available if _candidate_key(item) != removed_key
                ]
            room = max_pages - len(base)
            for add_count in range(1, min(2, room) + 1):
                for additions in combinations(available, add_count):
                    trial = base + list(additions)
                    summary = _summarize(
                        trial,
                        resource_budget,
                        confidence,
                        max_pages=max_pages,
                        min_prompt_pages=min_prompt_pages,
                        min_pages_per_head=min_pages_per_head,
                        required_heads=required_heads,
                    )
                    if summary.feasible and summary.total_utility > best_utility + 1e-12:
                        best, best_utility = trial, summary.total_utility
        if best is None:
            return current
        current = best


def select_pages_exact(
    candidates: list[PageCandidate],
    *,
    max_pages: int,
    resource_budget: float,
    confidence: float,
    min_prompt_pages: int = 0,
    min_pages_per_head: int = 0,
    max_candidates: int = 20,
) -> SelectionResult:
    """Return the utility-optimal feasible subset for a small candidate pool.

    This oracle is intentionally exponential and exists to evaluate scalable
    approximations. It must not be used on an unbounded ANN result set.
    """
    if len(candidates) > max_candidates:
        raise ValueError(
            f"exact selector accepts at most {max_candidates} candidates; "
            f"received {len(candidates)}"
        )
    if (
        max_pages < 0
        or min_prompt_pages < 0
        or min_prompt_pages > max_pages
        or min_pages_per_head < 0
    ):
        raise ValueError("invalid page limits")
    if resource_budget < 0:
        raise ValueError("resource_budget must be nonnegative")
    if candidates:
        n_scenarios = candidates[0].load_cost_samples.size
        if any(item.load_cost_samples.size != n_scenarios for item in candidates):
            raise ValueError("candidate costs must use aligned scenarios")

    best: list[PageCandidate] | None = None
    best_utility = -np.inf
    required_heads = tuple(sorted({item.page.head_idx for item in candidates}))
    if len(required_heads) * min_pages_per_head > max_pages:
        raise ValueError("per-head floor exceeds max_pages")
    upper = min(max_pages, len(candidates))
    for size in range(min_prompt_pages, upper + 1):
        for subset_tuple in combinations(candidates, size):
            subset = list(subset_tuple)
            if sum(item.page.source == "prompt" for item in subset) < min_prompt_pages:
                continue
            summary = _summarize(
                subset,
                resource_budget,
                confidence,
                max_pages=max_pages,
                min_prompt_pages=min_prompt_pages,
                min_pages_per_head=min_pages_per_head,
                required_heads=required_heads,
            )
            if not summary.feasible:
                continue
            # Empty/shorter subsets remain valid, but adding negative utility
            # pages can never displace a higher-utility feasible solution.
            if summary.total_utility > best_utility:
                best, best_utility = subset, summary.total_utility
    if best is None:
        return _summarize(
            [], resource_budget, confidence, min_prompt_pages=min_prompt_pages,
            max_pages=max_pages, min_pages_per_head=min_pages_per_head,
            required_heads=required_heads,
        )
    return _summarize(
        best,
        resource_budget,
        confidence,
        max_pages=max_pages, min_prompt_pages=min_prompt_pages,
        min_pages_per_head=min_pages_per_head,
        required_heads=required_heads,
    )


def _candidate_key(item: PageCandidate) -> tuple[object, ...]:
    """Identity of one layer/head page slice in the mixed candidate set."""
    page = item.page
    return (
        page.source,
        page.session_id,
        page.layer_idx,
        page.head_idx,
        page.page_id,
    )


def _priority(
    item: PageCandidate, confidence: float
) -> tuple[float, float, str]:
    tail = empirical_quantile(item.load_cost_samples, confidence)
    return (item.utility / max(tail, 1e-12), item.utility, item.page.page_id)


class SessionMaterialization:
    """Resolve one global page into session-local state and resource cost.

    Cost vectors for every state must share scenario IDs. This makes a common
    congestion scenario affect recompute, CPU-load, and GPU-hit candidates in
    a correlated way rather than pretending their tails are independent.
    """

    def __init__(
        self,
        cost_by_state: dict[str, np.ndarray],
        *,
        session_id: str,
        default_rag_state: str = "descriptor_only",
        default_prompt_state: str = "cpu_kv",
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not cost_by_state:
            raise ValueError("cost_by_state must not be empty")
        self.cost_by_state = {
            state: np.asarray(costs, dtype=np.float64)
            for state, costs in cost_by_state.items()
        }
        sizes = {costs.size for costs in self.cost_by_state.values()}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError("all materialization costs must use aligned scenarios")
        if any(costs.ndim != 1 or np.any(costs < 0) for costs in self.cost_by_state.values()):
            raise ValueError("materialization costs must be nonnegative vectors")
        if default_rag_state not in self.cost_by_state or default_prompt_state not in self.cost_by_state:
            raise ValueError("default states need cost vectors")
        self.default_rag_state = default_rag_state
        self.default_prompt_state = default_prompt_state
        self.session_id = session_id
        self.state_by_page: dict[str, str] = {}

    def set_state(self, page_id: str, state: str) -> None:
        if state not in self.cost_by_state:
            raise ValueError(f"unknown materialization state: {state}")
        self.state_by_page[page_id] = state

    def resolve(self, page: SemanticPage) -> tuple[str, np.ndarray]:
        if page.source == "prompt" and page.session_id != self.session_id:
            raise ValueError(
                f"prompt page belongs to session {page.session_id!r}, not "
                f"resolver session {self.session_id!r}"
            )
        default = self.default_rag_state if page.source == "rag" else self.default_prompt_state
        state = self.state_by_page.get(page.page_id, default)
        return state, self.cost_by_state[state]


class LayeredPartitionedIndex:
    """One logical mixed-source index over per-layer physical partitions.

    Every indexed transformer layer owns an immutable global RAG catalog.
    Prompt pages live in a separate per-session delta catalog. A layer query
    searches both partitions and merges candidates before selection; vectors
    from different layers are never compared.
    """

    def __init__(self, indexed_layers: Iterable[int], n_kv_heads: int = 1) -> None:
        layers = sorted(set(indexed_layers))
        if not layers:
            raise ValueError("indexed_layers must not be empty")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        self.n_kv_heads = n_kv_heads
        self.global_catalogs = {
            (layer, head): UnifiedPageCatalog(layer, head)
            for layer in layers
            for head in range(n_kv_heads)
        }
        self.session_catalogs: dict[
            str, dict[tuple[int, int], UnifiedPageCatalog]
        ] = {}

    @property
    def indexed_layers(self) -> tuple[int, ...]:
        return tuple(sorted({layer for layer, _head in self.global_catalogs}))

    def add(self, page: SemanticPage) -> None:
        key = (page.layer_idx, page.head_idx)
        if key not in self.global_catalogs:
            raise ValueError(f"layer {page.layer_idx} is not indexed")
        if page.source == "rag":
            self.global_catalogs[key].add(page)
            return
        session_layers = self.session_catalogs.setdefault(
            page.session_id,
            {
                (layer, head): UnifiedPageCatalog(layer, head)
                for layer, head in self.global_catalogs
            },
        )
        session_layers[key].add(page)

    def remove_session(self, session_id: str) -> int:
        catalogs = self.session_catalogs.pop(session_id, None)
        return sum(len(catalog._pages) for catalog in catalogs.values()) if catalogs else 0

    def search_head(
        self,
        layer_idx: int,
        head_idx: int,
        query: np.ndarray,
        *,
        session_id: str,
        candidate_limit: int,
        error_weight: float = 0.0,
        rag_bias: float = 0.0,
        cost_resolver: Callable[[SemanticPage], tuple[str, np.ndarray]] | None = None,
    ) -> list[PageCandidate]:
        key = (layer_idx, head_idx)
        if key not in self.global_catalogs:
            raise ValueError(f"layer/head {key} is not indexed")
        kwargs = dict(
            session_id=session_id,
            candidate_limit=candidate_limit,
            error_weight=error_weight,
            rag_bias=rag_bias,
            cost_resolver=cost_resolver,
        )
        mixed = self.global_catalogs[key].search(query, **kwargs)
        private = self.session_catalogs.get(session_id)
        if private and private[key]._pages:
            mixed.extend(private[key].search(query, **kwargs))
        mixed.sort(key=lambda item: (-item.utility, item.page.page_id))
        return mixed[:candidate_limit]

    def search_layer(
        self,
        layer_idx: int,
        queries: np.ndarray,
        **kwargs,
    ) -> dict[int, list[PageCandidate]]:
        queries = np.asarray(queries)
        if queries.ndim == 1 and self.n_kv_heads == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[0] != self.n_kv_heads:
            raise ValueError(
                f"layer queries must have shape [{self.n_kv_heads}, dimension]"
            )
        return {
            head: self.search_head(layer_idx, head, queries[head], **kwargs)
            for head in range(self.n_kv_heads)
        }

    def search_layers(
        self,
        queries: dict[int, np.ndarray],
        *,
        session_id: str,
        candidate_limit: int,
        error_weight: float = 0.0,
        rag_bias: float = 0.0,
        cost_resolver: Callable[[SemanticPage], tuple[str, np.ndarray]] | None = None,
    ) -> dict[int, dict[int, list[PageCandidate]]]:
        missing = set(queries) - set(self.indexed_layers)
        if missing:
            raise ValueError(f"queries contain unindexed layers: {sorted(missing)}")
        return {
            layer: self.search_layer(
                layer,
                query,
                session_id=session_id,
                candidate_limit=candidate_limit,
                error_weight=error_weight,
                rag_bias=rag_bias,
                cost_resolver=cost_resolver,
            )
            for layer, query in queries.items()
        }
