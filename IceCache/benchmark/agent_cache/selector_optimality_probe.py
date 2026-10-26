#!/usr/bin/env python3
"""Measure greedy unified-page selection against the exact small-pool oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icecache.experimental import select_pages, select_pages_exact
from unified_rag_sim import build_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=14)
    parser.add_argument("--max-pages", type=int, default=6)
    parser.add_argument("--resource-budget", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_limit > 20:
        raise ValueError("candidate-limit must be <= 20 for exact enumeration")
    records = []
    for seed in range(args.runs):
        config = argparse.Namespace(seed=seed, dimension=24, rag_pages=20, prompt_pages=20)
        catalog, query = build_catalog(config)
        candidates = catalog.search(
            query,
            session_id="session-a",
            candidate_limit=args.candidate_limit,
            error_weight=0.5,
            rag_bias=0.05,
        )
        kwargs = dict(
            max_pages=args.max_pages,
            resource_budget=args.resource_budget,
            confidence=args.confidence,
            min_prompt_pages=2,
        )
        greedy = select_pages(candidates, **kwargs, local_search=False)
        refined = select_pages(candidates, **kwargs)
        exact = select_pages_exact(candidates, **kwargs)
        gap = exact.total_utility - greedy.total_utility
        refined_gap = exact.total_utility - refined.total_utility
        records.append(
            {
                "seed": seed,
                "greedy_utility": greedy.total_utility,
                "exact_utility": exact.total_utility,
                "absolute_gap": gap,
                "relative_gap": gap / max(abs(exact.total_utility), 1e-12),
                "refined_utility": refined.total_utility,
                "refined_absolute_gap": refined_gap,
                "refined_relative_gap": refined_gap / max(abs(exact.total_utility), 1e-12),
                "greedy_feasible": greedy.feasible,
                "refined_feasible": refined.feasible,
                "exact_feasible": exact.feasible,
            }
        )
    feasible = [row for row in records if row["exact_feasible"]]
    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": {
            "runs": len(records),
            "exact_feasible_runs": len(feasible),
            "greedy_feasible_runs": sum(row["greedy_feasible"] for row in records),
            "refined_feasible_runs": sum(row["refined_feasible"] for row in records),
            "greedy_optimal_fraction": float(np.mean([abs(row["absolute_gap"]) < 1e-9 for row in feasible])) if feasible else 0.0,
            "refined_optimal_fraction": float(np.mean([abs(row["refined_absolute_gap"]) < 1e-9 for row in feasible])) if feasible else 0.0,
            "mean_absolute_gap": float(np.mean([row["absolute_gap"] for row in feasible])) if feasible else 0.0,
            "mean_relative_gap": float(np.mean([row["relative_gap"] for row in feasible])) if feasible else 0.0,
            "max_relative_gap": float(np.max([row["relative_gap"] for row in feasible])) if feasible else 0.0,
            "refined_mean_relative_gap": float(np.mean([row["refined_relative_gap"] for row in feasible])) if feasible else 0.0,
            "refined_max_relative_gap": float(np.max([row["refined_relative_gap"] for row in feasible])) if feasible else 0.0,
        },
        "records": records,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
