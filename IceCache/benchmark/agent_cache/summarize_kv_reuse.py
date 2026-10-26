#!/usr/bin/env python3
"""Aggregate layerwise outputs from repeated rag_kv_reuse_probe runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(payloads: list[dict]) -> dict:
    result = {"runs": len(payloads), "random_init_runs": 0, "comparisons": {}}
    result["random_init_runs"] = sum(bool(p["config"].get("random_init")) for p in payloads)
    for comparison in ("same_position_different_context", "different_position_and_context"):
        grouped = defaultdict(list)
        for payload in payloads:
            for row in payload[comparison]:
                grouped[int(row["layer"])].append(row)
        layers = []
        for layer, rows in sorted(grouped.items()):
            def values(path):
                first, second = path
                return np.asarray([row[first][second] for row in rows], dtype=np.float64)

            layers.append(
                {
                    "layer": layer,
                    "observations": len(rows),
                    "k_cosine_mean": float(values(("k", "cosine_mean")).mean()),
                    "v_cosine_mean": float(values(("v", "cosine_mean")).mean()),
                    "page_top1_identity_mean": float(
                        values(("key_page_retrieval", "page_top1_identity")).mean()
                    ),
                    "page_mrr_mean": float(
                        values(("key_page_retrieval", "page_mrr")).mean()
                    ),
                }
            )
        result["comparisons"][comparison] = layers
    result["validity_warning"] = (
        "Random-init runs establish architecture-level context dependence only. "
        "A pretrained-model gate is still required for the direct-KV claim."
        if result["random_init_runs"]
        else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    result = summarize(payloads)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
