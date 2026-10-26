#!/usr/bin/env python3
"""Split-conformal calibration for per-decode IceCache resource bounds.

Input files are produced by ``recall_cost_trace_probe.py``. Events are first
aggregated by decode step across layers because the chance constraint applies
to the whole token and layer costs within a token are correlated. A simple
nonnegative pages-to-cost slope is fit on training steps; calibration residuals
then provide a finite-sample upper correction. Test steps are never used to
choose the bound.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def aggregate_decode_steps(payloads: list[dict], cost_key: str) -> list[dict]:
    observations = []
    for trace_id, payload in enumerate(payloads):
        grouped: dict[int, dict] = {}
        for event in payload.get("events", []):
            step = int(event["decode_step"])
            row = grouped.setdefault(
                step,
                {
                    "trace_id": trace_id,
                    "decode_step": step,
                    "head_pages": 0.0,
                    "cost_ms": 0.0,
                    "layers": 0,
                },
            )
            row["head_pages"] += float(event["recalled_head_pages"])
            row["cost_ms"] += float(event[cost_key])
            row["layers"] += 1
        observations.extend(grouped.values())
    return observations


def conformal_upper_residual(residuals: np.ndarray, alpha: float) -> float:
    """Return the split-conformal upper residual, or inf if sample is too small."""
    if residuals.ndim != 1 or residuals.size == 0:
        raise ValueError("calibration residuals must be a non-empty vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = math.ceil((residuals.size + 1) * (1 - alpha))
    if rank > residuals.size:
        return math.inf
    return float(np.sort(residuals)[rank - 1])


def split_observations(
    observations: list[dict], *, seed: int, train_fraction: float, calibration_fraction: float
) -> tuple[list[dict], list[dict], list[dict]]:
    if not observations:
        raise ValueError("no decode-step observations")
    if train_fraction <= 0 or calibration_fraction <= 0 or train_fraction + calibration_fraction >= 1:
        raise ValueError("fractions must leave non-empty train/calibration/test regions")
    indices = np.random.default_rng(seed).permutation(len(observations))
    n_train = max(1, int(len(indices) * train_fraction))
    n_cal = max(1, int(len(indices) * calibration_fraction))
    if n_train + n_cal >= len(indices):
        raise ValueError("not enough observations for requested three-way split")
    take = lambda ids: [observations[int(idx)] for idx in ids]
    return take(indices[:n_train]), take(indices[n_train : n_train + n_cal]), take(indices[n_train + n_cal :])


def calibrate(
    observations: list[dict],
    *,
    alpha: float,
    seed: int,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
) -> dict:
    train, calibration, test = split_observations(
        observations,
        seed=seed,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
    )

    def arrays(rows):
        return (
            np.asarray([row["head_pages"] for row in rows], dtype=np.float64),
            np.asarray([row["cost_ms"] for row in rows], dtype=np.float64),
        )

    train_pages, train_cost = arrays(train)
    denominator = float(np.dot(train_pages, train_pages))
    slope = max(0.0, float(np.dot(train_pages, train_cost)) / denominator) if denominator else 0.0
    cal_pages, cal_cost = arrays(calibration)
    correction = conformal_upper_residual(cal_cost - slope * cal_pages, alpha)
    test_pages, test_cost = arrays(test)
    upper = slope * test_pages + correction
    violations = test_cost > upper
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "seed": seed,
        "split": {"train": len(train), "calibration": len(calibration), "test": len(test)},
        "slope_ms_per_head_page": slope,
        "conformal_correction_ms": correction,
        "vacuous_bound": not np.isfinite(correction),
        "test": {
            "violations": int(violations.sum()),
            "violation_rate": float(violations.mean()),
            "coverage": float(1 - violations.mean()),
            "mean_bound_ms": float(np.mean(upper)),
            "mean_actual_ms": float(np.mean(test_cost)),
        },
        "warning": (
            "Finite-sample marginal coverage requires exchangeable decode-step "
            "observations. Report per-trace and temporal-shift tests separately."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--cost-key",
        default="blocking_minus_query_submit_ms",
        choices=("blocking_total_ms", "blocking_minus_query_submit_ms"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.traces]
    observations = aggregate_decode_steps(payloads, args.cost_key)
    result = calibrate(
        observations,
        alpha=args.alpha,
        seed=args.seed,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    result["cost_key"] = args.cost_key
    result["traces"] = [str(path) for path in args.traces]
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
