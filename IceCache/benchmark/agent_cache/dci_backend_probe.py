#!/usr/bin/env python3
"""Compare multi-head M-DCI page retrieval with exact cosine search."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from icecache.experimental import LayerDCIBackend, SemanticPage, UnifiedPageCatalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--query-heads", type=int,
        help="GQA query heads; defaults to the number of KV heads",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--overfetch", type=int, default=1)
    parser.add_argument("--prompt-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.query_heads is None:
        args.query_heads = args.heads
    if args.query_heads % args.heads:
        parser.error("--query-heads must be divisible by --heads")
    query_ratio = args.query_heads // args.heads
    rng = np.random.default_rng(args.seed)
    catalogs = {
        head: UnifiedPageCatalog(layer_idx=4, head_idx=head)
        for head in range(args.heads)
    }
    for head, catalog in catalogs.items():
        descriptors = rng.normal(size=(args.points, args.dimension)).astype(np.float32)
        for idx, descriptor in enumerate(descriptors):
            catalog.add(
                SemanticPage(
                    page_id=f"h{head}-rag-{idx}",
                    source="rag",
                    descriptor=descriptor,
                    error_radius=0.0,
                    load_cost_samples=np.ones(8),
                    layer_idx=4,
                    head_idx=head,
                )
            )
    session_catalogs = {
        head: UnifiedPageCatalog(layer_idx=4, head_idx=head)
        for head in range(args.heads)
    }
    for head, catalog in session_catalogs.items():
        descriptors = rng.normal(
            size=(args.prompt_points, args.dimension)
        ).astype(np.float32)
        for idx, descriptor in enumerate(descriptors):
            catalog.add(
                SemanticPage(
                    page_id=f"h{head}-prompt-{idx}",
                    source="prompt",
                    descriptor=descriptor,
                    error_radius=0.0,
                    load_cost_samples=np.ones(8),
                    session_id="probe",
                    layer_idx=4,
                    head_idx=head,
                )
            )
    build_start = time.perf_counter()
    backend = LayerDCIBackend(catalogs, seed=args.seed)
    build_ms = (time.perf_counter() - build_start) * 1000
    recalls, exact_top1 = [], []
    approx_ms, exact_ms = [], []
    for _ in range(args.queries):
        queries = rng.normal(size=(args.query_heads, args.dimension)).astype(np.float32)
        start = time.perf_counter()
        approx = backend.search_partitioned(
            queries,
            session_id="probe",
            session_catalogs=session_catalogs if args.prompt_points else None,
            candidate_limit=args.topk,
            overfetch=args.overfetch,
        )
        approx_ms.append((time.perf_counter() - start) * 1000)
        exact_start = time.perf_counter()
        exact_by_head = {
            head: sorted(
                catalog.search(
                    queries[head * query_ratio : (head + 1) * query_ratio],
                    session_id="probe", candidate_limit=args.topk
                )
                + session_catalogs[head].search(
                    queries[head * query_ratio : (head + 1) * query_ratio],
                    session_id="probe", candidate_limit=args.topk
                ),
                key=lambda item: (-item.utility, item.page.page_id),
            )[: args.topk]
            for head, catalog in catalogs.items()
        }
        exact_ms.append((time.perf_counter() - exact_start) * 1000)
        for head, catalog in catalogs.items():
            exact = exact_by_head[head]
            exact_ids = [item.page.page_id for item in exact]
            approx_ids = [item.page.page_id for item in approx[head]]
            recalls.append(len(set(exact_ids) & set(approx_ids)) / len(exact_ids))
            exact_top1.append(bool(approx_ids) and approx_ids[0] == exact_ids[0])
    result = {
        "config": vars(args) | {"json_out": str(args.json_out) if args.json_out else None},
        "head_queries": len(recalls),
        "mean_recall_at_k": float(np.mean(recalls)),
        "p05_recall_at_k": float(np.quantile(recalls, 0.05)),
        "min_recall_at_k": float(np.min(recalls)),
        "exact_top1_rate": float(np.mean(exact_top1)),
        "build_ms": build_ms,
        "approx_query_ms": {
            "mean": float(np.mean(approx_ms)),
            "p50": float(np.quantile(approx_ms, 0.50)),
            "p95": float(np.quantile(approx_ms, 0.95)),
        },
        "python_exact_query_ms": {
            "mean": float(np.mean(exact_ms)),
            "p50": float(np.quantile(exact_ms, 0.50)),
            "p95": float(np.quantile(exact_ms, 0.95)),
        },
        "warning": "Synthetic descriptor retrieval test; not a RAG quality result.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
