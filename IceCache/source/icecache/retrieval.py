"""Stable extension contracts around IceCache candidate retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    # CandidatePlan lives with the materialization state machine because its
    # validated fast constructors are used there.  The runtime export below
    # is lazy to avoid retrieval -> materialization -> retrieval at import
    # time while still giving callers one canonical class.
    from .materialization import CandidatePlan

__all__ = [
    "CandidatePlan",
    "CandidateProvider",
    "CandidateQueryResult",
    "DCIObserver",
    "DCIObserverGroup",
    "PlannedCandidateProvider",
    "query_candidate_provider",
    "validate_leaf_ids",
]


def __getattr__(name: str):
    """Lazily re-export the canonical two-stage plan type."""
    if name == "CandidatePlan":
        from .materialization import CandidatePlan

        globals()[name] = CandidatePlan
        return CandidatePlan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class DCIObserver(Protocol):
    """Receive read-only lifecycle events from the DCI path.

    Implementations may copy tensors for tracing, but must not mutate payloads.
    Callbacks can run on IceCache's worker thread and therefore must provide
    their own synchronization when sharing state.
    """

    def on_dci_event(self, event: str, **payload: Any) -> None: ...


class CandidateProvider(Protocol):
    """Return semantic leaf IDs for one indexed layer and decode token."""

    def query_leaf_ids(
        self,
        *,
        layer_idx: int,
        query_states: np.ndarray,
        num_neighbours: int,
        state: Any,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class CandidateQueryResult:
    """One provider call's native input and optional two-stage plan.

    ``effective_leaf_ids`` is the exact array the runtime must validate and
    send to native bookkeeping.  It may be the plan's ready rows in live mode
    or its fallback rows in shadow mode, so consumers must never reconstruct
    it by merging plan fields a second time.
    """

    effective_leaf_ids: np.ndarray
    plan: CandidatePlan | None = None


class PlannedCandidateProvider(Protocol):
    """Atomically return the effective native rows and their control plan."""

    def query_candidate_result(
        self,
        *,
        layer_idx: int,
        query_states: np.ndarray,
        num_neighbours: int,
        state: Any,
    ) -> CandidateQueryResult: ...


def query_candidate_provider(provider: Any, **kwargs: Any) -> CandidateQueryResult:
    """Dispatch to a plan-aware provider while preserving legacy providers."""
    planned = getattr(provider, "query_candidate_result", None)
    if callable(planned):
        result = planned(**kwargs)
        if not isinstance(result, CandidateQueryResult):
            raise TypeError(
                "query_candidate_result must return CandidateQueryResult"
            )
        return result
    legacy = getattr(provider, "query_leaf_ids", None)
    if not callable(legacy):
        raise TypeError(
            "candidate provider must expose query_candidate_result or "
            "query_leaf_ids"
        )
    return CandidateQueryResult(effective_leaf_ids=legacy(**kwargs))


class DCIObserverGroup:
    """Deliver events to multiple observers in deterministic order."""

    def __init__(self, *observers: DCIObserver) -> None:
        self.observers = list(observers)

    def append(self, observer: DCIObserver) -> None:
        self.observers.append(observer)

    def on_dci_event(self, event: str, **payload: Any) -> None:
        for observer in tuple(self.observers):
            observer.on_dci_event(event, **payload)


def validate_leaf_ids(
    leaf_ids: np.ndarray,
    *,
    n_kv_heads: int,
    num_neighbours: int,
    num_leaves: np.ndarray,
) -> np.ndarray:
    """Validate a provider result before it reaches native bookkeeping.

    Bounds are checked in the source dtype before the int32 conversion.  This
    is essential because values such as ``2**32`` and ``-2**32`` otherwise
    alias leaf zero during the cast and pass all later range checks.
    """
    for name, value in (
        ("n_kv_heads", n_kv_heads),
        ("num_neighbours", num_neighbours),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    n_kv_heads = int(n_kv_heads)
    num_neighbours = int(num_neighbours)

    result = np.asarray(leaf_ids)
    expected = (n_kv_heads, num_neighbours)
    if result.shape != expected:
        raise ValueError(f"candidate provider returned {result.shape}, expected {expected}")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError("candidate provider must return integer leaf IDs")
    if result.dtype.kind == "i" and int(result.min()) < 0:
        raise ValueError("candidate provider returned a negative leaf ID")
    if int(result.max()) > np.iinfo(np.int32).max:
        raise ValueError("candidate provider returned a leaf ID outside int32")

    raw_limits = np.asarray(num_leaves)
    if not np.issubdtype(raw_limits.dtype, np.integer):
        raise TypeError("num_leaves must contain integers")
    if raw_limits.size != n_kv_heads:
        raise ValueError(
            f"num_leaves has {raw_limits.size} entries, expected {n_kv_heads}"
        )
    limits = raw_limits.reshape(n_kv_heads)
    if any(
        int(limit) <= 0 or int(limit) > np.iinfo(np.int32).max
        for limit in limits
    ):
        raise ValueError("num_leaves must contain positive int32 limits")
    if any(int(row.max()) >= int(limit) for row, limit in zip(result, limits)):
        raise ValueError("candidate provider returned an out-of-range leaf ID")
    if any(len(np.unique(row)) != num_neighbours for row in result):
        raise ValueError("candidate provider returned duplicate IDs within a KV head")
    return np.ascontiguousarray(result, dtype=np.int32)
