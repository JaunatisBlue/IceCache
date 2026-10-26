#!/usr/bin/env python3
"""Synthetic comparison of independent and unified RAG/KV page selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icecache.experimental import SemanticPage, UnifiedPageCatalog, select_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--rag-pages", type=int, default=32)
    parser.add_argument("--prompt-pages", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--resource-budget", type=float, default=13.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def build_catalog(args: argparse.Namespace) -> tuple[UnifiedPageCatalog, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    query = rng.normal(size=args.dimension).astype(np.float32)
    query /= np.linalg.norm(query)
    # One shared multiplier preserves congestion correlation across page costs.
    congestion = rng.lognormal(mean=0.0, sigma=0.35, size=512)
    catalog = UnifiedPageCatalog()
    for source, count in (("rag", args.rag_pages), ("prompt", args.prompt_pages)):
        for idx in range(count):
            relevance_strength = rng.beta(2.0, 2.0)
            noise = rng.normal(size=args.dimension)
            descriptor = relevance_strength * query + (1 - relevance_strength) * noise
            base_cost = rng.uniform(0.6, 2.2) * (1.2 if source == "rag" else 1.0)
            idiosyncratic = rng.lognormal(mean=0.0, sigma=0.12, size=512)
            catalog.add(
                SemanticPage(
                    page_id=f"{source}-{idx}",
                    source=source,
                    descriptor=descriptor,
                    error_radius=float(rng.uniform(0.0, 0.2)),
                    load_cost_samples=base_cost * congestion * idiosyncratic,
                    document_id=f"doc-{idx // 4}" if source == "rag" else None,
                    session_id="session-a" if source == "prompt" else None,
                )
            )
    return catalog, query


def as_dict(result) -> dict:
    return {
        "selected": [item.page.page_id for item in result.selected],
        "source_counts": result.source_counts,
        "total_utility": result.total_utility,
        "cost_quantile": result.cost_quantile,
        "empirical_violation_probability": result.empirical_violation_probability,
        "feasible": result.feasible,
        "prompt_floor_satisfied": result.prompt_floor_satisfied,
    }


def main() -> None:
    args = parse_args()
    catalog, query = build_catalog(args)
    candidates = catalog.search(
        query,
        session_id="session-a",
        candidate_limit=args.candidate_limit,
        error_weight=0.5,
        rag_bias=0.05,
    )
    unified = select_pages(
        candidates,
        max_pages=args.max_pages,
        resource_budget=args.resource_budget,
        confidence=args.confidence,
        min_prompt_pages=2,
    )
    # A compositional baseline reserves half the slots and risk budget for each
    # source before either selector sees the other source's candidates.
    independent_selected = []
    for source in ("prompt", "rag"):
        arm = select_pages(
            [item for item in candidates if item.page.source == source],
            max_pages=args.max_pages // 2,
            resource_budget=args.resource_budget / 2,
            confidence=args.confidence,
        )
        independent_selected.extend(arm.selected)
    from icecache.experimental.unified_page_index import _summarize

    independent = _summarize(independent_selected, args.resource_budget, args.confidence)
    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "candidate_sources": {
            source: sum(item.page.source == source for item in candidates)
            for source in ("prompt", "rag")
        },
        "independent_fixed_split": as_dict(independent),
        "unified": as_dict(unified),
        "warning": "Synthetic control-plane experiment; it does not establish model quality or runtime speedup.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
