#!/usr/bin/env python3
"""Calibrate empirical recall-cost scenarios from existing DCI_PROFILE logs.

Each profile is an aggregate from one complete run, not a per-page latency
trace.  The logs mix configurations, so the derived milliseconds-per-head-page
values are stress scenarios, not samples from one stationary distribution.
They are useful for sensitivity analysis but cannot validate a VaR claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PREFIX = "DCI_PROFILE "


def parse_profile_line(line: str) -> dict | None:
    marker = line.find(PREFIX)
    if marker < 0:
        return None
    return json.loads(line[marker + len(PREFIX) :])


def load_profiles(log_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(log_dir.rglob("*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            profile = parse_profile_line(line)
            if profile is not None:
                profile["log_path"] = str(path)
                rows.append(profile)
    return rows


def usable_recall_profiles(rows: list[dict]) -> list[dict]:
    needed = ("recall_pages_per_token", "recall_wait_ms_per_token", "decode_tpot_ms")
    return [
        row
        for row in rows
        if all(row.get(key) is not None and np.isfinite(row[key]) for key in needed)
        and row["recall_pages_per_token"] > 0
        and row["recall_wait_ms_per_token"] >= 0
        and row["decode_tpot_ms"] > 0
    ]


def per_head_page_cost_samples(rows: list[dict]) -> np.ndarray:
    usable = usable_recall_profiles(rows)
    if not usable:
        raise ValueError("no usable recall profiles")
    return np.asarray(
        [row["recall_wait_ms_per_token"] / row["recall_pages_per_token"] for row in usable],
        dtype=np.float64,
    )


def aligned_page_costs(rows: list[dict], head_pages: float) -> np.ndarray:
    """Return a correlated cross-configuration stress vector for a candidate."""
    if head_pages < 0:
        raise ValueError("head_pages must be nonnegative")
    return per_head_page_cost_samples(rows) * head_pages


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def summarize(rows: list[dict]) -> dict:
    usable = usable_recall_profiles(rows)
    page_cost = per_head_page_cost_samples(usable)
    pages = np.asarray([row["recall_pages_per_token"] for row in usable])
    waits = np.asarray([row["recall_wait_ms_per_token"] for row in usable])
    tpots = np.asarray([row["decode_tpot_ms"] for row in usable])
    submissions = np.asarray(
        [row.get("recall_submissions_per_token") or row.get("recall_calls_per_token") or 0 for row in usable]
    )
    design = np.column_stack((np.ones_like(pages), pages, submissions))
    coefficients, *_ = np.linalg.lstsq(design, waits, rcond=None)
    predicted = design @ coefficients
    residual = waits - predicted
    ss_total = float(np.sum((waits - waits.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_total if ss_total else 0.0
    return {
        "profiles_found": len(rows),
        "profiles_usable": len(usable),
        "units": {
            "recall_wait": "milliseconds per decode token",
            "recall_pages": "head-pages per decode token",
            "per_head_page_cost": "milliseconds per head-page",
        },
        "recall_pages_per_token": quantiles(pages),
        "recall_wait_ms_per_token": quantiles(waits),
        "decode_tpot_ms": quantiles(tpots),
        "per_head_page_wait_ms": quantiles(page_cost),
        "recall_wait_fraction_of_tpot": quantiles(waits / tpots),
        "correlations": {
            "pages_vs_wait": float(np.corrcoef(pages, waits)[0, 1]),
            "pages_vs_tpot": float(np.corrcoef(pages, tpots)[0, 1]),
            "wait_vs_tpot": float(np.corrcoef(waits, tpots)[0, 1]),
        },
        "ols_wait_model": {
            "formula": "wait_ms = intercept + page_slope * head_pages + submission_slope * submissions",
            "intercept_ms": float(coefficients[0]),
            "page_slope_ms": float(coefficients[1]),
            "submission_slope_ms": float(coefficients[2]),
            "r_squared": r2,
            "residual_ms": quantiles(residual),
            "warning": "Descriptive fit across heterogeneous runs; not a causal latency model.",
        },
        "scenario_samples_ms_per_head_page": page_cost.tolist(),
        "statistical_warning": (
            "Profiles aggregate heterogeneous configurations and are not IID "
            "samples from a deployment distribution. Use these values as a "
            "stress envelope, not to claim calibrated violation probability."
        ),
        "provenance": [row["log_path"] for row in usable],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("IceCache/experiment/experiment/logs/dci_opt"),
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(load_profiles(args.log_dir))
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
