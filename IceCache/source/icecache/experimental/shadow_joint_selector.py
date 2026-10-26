"""Shadow-mode joint RAG/prompt selection on real IceCache decode queries."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .centroid_provider import ExactLeafCentroidProvider
from .unified_page_index import (
    PageCandidate,
    SemanticPage,
    UnifiedPageCatalog,
    select_pages,
)


@dataclass(frozen=True)
class ShadowSelectionEvent:
    layer_idx: int
    seq_len: int
    selected_page_ids: tuple[str, ...]
    prompt_pages: int
    rag_pages: int
    total_utility: float
    cost_quantile: float
    empirical_violation_probability: float
    feasible: bool


class ShadowJointSelector:
    """Evaluate a mixed-source policy without changing IceCache page recall.

    Prompt descriptors come from the current session's real DCI leaves. RAG
    descriptors are immutable metadata registered by the caller. Selected RAG
    pages remain descriptor-only: this observer records that they would need
    materialization but never presents them as native leaf IDs.
    """

    def __init__(
        self,
        prompt_provider: ExactLeafCentroidProvider,
        *,
        session_id: str,
        prompt_cost_samples: np.ndarray,
        max_pages: int,
        resource_budget: float,
        confidence: float = 0.95,
        min_prompt_pages: int = 0,
        min_pages_per_head: int = 1,
        candidate_limit_per_head: int = 16,
        error_weight: float = 0.0,
        rag_bias: float = 0.0,
    ) -> None:
        self.prompt_provider = prompt_provider
        self.session_id = session_id
        self.prompt_cost_samples = np.asarray(prompt_cost_samples, dtype=np.float64)
        if self.prompt_cost_samples.ndim != 1 or not self.prompt_cost_samples.size:
            raise ValueError("prompt_cost_samples must be a non-empty vector")
        self.max_pages = max_pages
        self.resource_budget = resource_budget
        self.confidence = confidence
        self.min_prompt_pages = min_prompt_pages
        self.min_pages_per_head = min_pages_per_head
        self.candidate_limit_per_head = candidate_limit_per_head
        self.error_weight = error_weight
        self.rag_bias = rag_bias
        self.rag_pages: dict[tuple[int, int], list[SemanticPage]] = {}
        self.events: list[ShadowSelectionEvent] = []
        self.lock = threading.Lock()

    def register_rag_page(self, page: SemanticPage) -> None:
        if page.source != "rag":
            raise ValueError("shadow RAG registration requires source='rag'")
        if page.load_cost_samples.size != self.prompt_cost_samples.size:
            raise ValueError("RAG and prompt costs must use aligned scenarios")
        self.rag_pages.setdefault((page.layer_idx, page.head_idx), []).append(page)

    def on_dci_event(self, event: str, **payload) -> None:
        if event != "query":
            return
        state = payload["state"]
        layer = int(payload["layer_idx"])
        if layer not in self.prompt_provider.descriptors:
            return
        query = payload["query_states"]
        if hasattr(query, "detach"):
            query = query.detach().cpu().numpy()
        query = np.asarray(query, dtype=np.float32).reshape(state.n_qo_heads, state.head_dim)
        ratio = state.n_qo_heads // state.n_kv_heads
        descriptors = self.prompt_provider.descriptors[layer]
        leaves = self.prompt_provider.num_leaves[layer]
        candidates: list[PageCandidate] = []
        for head in range(state.n_kv_heads):
            catalog = UnifiedPageCatalog(layer, head)
            for leaf in range(int(leaves[head])):
                catalog.add(
                    SemanticPage(
                        page_id=f"prompt:{layer}:{head}:{leaf}",
                        source="prompt",
                        descriptor=descriptors[head, leaf],
                        error_radius=0.0,
                        load_cost_samples=self.prompt_cost_samples,
                        session_id=self.session_id,
                        layer_idx=layer,
                        head_idx=head,
                    )
                )
            catalog.extend(self.rag_pages.get((layer, head), ()))
            head_query = query[head * ratio : (head + 1) * ratio]
            candidates.extend(
                catalog.search(
                    head_query,
                    session_id=self.session_id,
                    candidate_limit=self.candidate_limit_per_head,
                    error_weight=self.error_weight,
                    rag_bias=self.rag_bias,
                )
            )
        result = select_pages(
            candidates,
            max_pages=self.max_pages,
            resource_budget=self.resource_budget,
            confidence=self.confidence,
            min_prompt_pages=self.min_prompt_pages,
            min_pages_per_head=self.min_pages_per_head,
        )
        record = ShadowSelectionEvent(
            layer_idx=layer,
            seq_len=int(state.seq_len),
            selected_page_ids=tuple(item.page.page_id for item in result.selected),
            prompt_pages=result.source_counts["prompt"],
            rag_pages=result.source_counts["rag"],
            total_utility=result.total_utility,
            cost_quantile=result.cost_quantile,
            empirical_violation_probability=result.empirical_violation_probability,
            feasible=result.feasible,
        )
        with self.lock:
            self.events.append(record)
