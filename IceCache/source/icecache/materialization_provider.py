"""CandidateProvider adapter for session-local RAG materialization.

This module is the runtime seam between the unified selector, the
session-local materializer and IceCache's existing ``CandidateProvider`` hook.
It deliberately does not change ``_DCI_query``: that hook still receives one
validated native leaf array.  The two-stage result is exposed as
``last_plan`` so a plan-aware runtime can submit ``deferred_rag_pages`` without
putting page IDs on the native path.

Stage-1 equivalence: with an empty RAG selection, ``query_leaf_ids`` returns
the prompt fallback verbatim.  Stage-3 behavior: ready RAG pages contribute
their per-head native leaves first; a page that is not resolvable for this
layer is deferred, the current token continues on the fallback, and
``on_deferred`` (if supplied) is invoked outside the native path.  Deferred
callback failures are recorded, never raised: materialization must not break
generation.

The adapter consumes a per-head selection, either as

* ``Sequence[Sequence[object]]`` (one ordered list per KV head),
* a ``Mapping[int, Sequence[object]]`` keyed by KV head, or
* a flat sequence of candidates carrying ``page.head_idx`` (grouped with
  :func:`group_selected_by_head`), including codex's ``SelectionResult``
  because its ``.selected`` attribute is unwrapped.

Selection policy stays outside this module; the caller supplies
``select_rag(layer_idx, query_states, state)``.
"""

from __future__ import annotations

import threading
from typing import Callable, Mapping

import numpy as np

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from icecache.materialization import (  # noqa: E402
        CandidatePlan,
        PlanPrefixCache,
        RAGMaterializer,
        build_candidate_plan,
        normalise_selected_page_ids,
    )
    from icecache.retrieval import CandidateQueryResult  # noqa: E402
else:
    from .materialization import (
        CandidatePlan,
        PlanPrefixCache,
        RAGMaterializer,
        build_candidate_plan,
        normalise_selected_page_ids,
    )
    from .retrieval import CandidateQueryResult

__all__ = [
    "DeferredPageQueue",
    "MaterializingCandidateProvider",
    "group_selected_by_head",
]


def _require_positive_int(value, *, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _is_builtin_method(obj: object, name: str) -> bool:
    """True only for an un-overridden ``RAGMaterializer`` method.

    Checks the underlying function rather than ``type(obj)`` so that an
    instance-level shadow (``materializer.head_resolver = custom``) also
    disqualifies the fast path.  Also requires the binding to target ``obj``:
    a foreign bound method (``m1.head_resolver = m2.head_resolver``) has the
    built-in ``__func__`` but reads ``m2``'s leaves, while the plan cache's
    epoch is tracked on ``m1`` -- stale rows would never be invalidated.
    """
    attr = getattr(obj, name, None)
    return (
        getattr(attr, "__func__", None) is getattr(RAGMaterializer, name)
        and getattr(attr, "__self__", None) is obj
    )


def group_selected_by_head(
    selected: object, n_kv_heads: int
) -> list[list[object]]:
    """Group a flat selector result into one ordered list per KV head.

    ``selected`` may be a ``SelectionResult``-like object with a ``.selected``
    attribute.  Every entry must expose ``page.head_idx`` (a
    ``PageCandidate`` does; a bare ``SemanticPage`` is also accepted because
    ``.page`` is optional).  Prompt entries are retained; ``build_candidate_plan``
    filters them out.
    """
    n_heads = _require_positive_int(n_kv_heads, name="n_kv_heads")
    if isinstance(selected, (str, bytes)):
        raise TypeError("selection must be a sequence, not a string")
    if hasattr(selected, "selected"):
        selected = selected.selected
    if selected is None:
        raise TypeError("selection must not be None")
    rows: list[list[object]] = [[] for _ in range(n_heads)]
    for item in selected:
        page = getattr(item, "page", item)
        head = getattr(page, "head_idx", None)
        if (
            not isinstance(head, (int, np.integer))
            or isinstance(head, bool)
        ):
            raise TypeError(
                "each selected entry must expose an integer page.head_idx"
            )
        head = int(head)
        if not 0 <= head < n_heads:
            raise ValueError(
                f"selected entry has head_idx={head}, expected [0, {n_heads})"
            )
        rows[head].append(item)
    return rows


class DeferredPageQueue:
    """Thread-safe, ordered collector for pages that need materialization.

    Use as ``on_deferred`` on :class:`MaterializingCandidateProvider`: decode
    only records the request, and the next request boundary drains it:

    .. code-block:: python

        queue = DeferredPageQueue()
        provider = MaterializingCandidateProvider(..., on_deferred=queue)
        # ... decode ...
        pending = queue.drain(session_id)
        # next boundary: re-run retrieval/prefill for ``pending`` at a safe
        # point (never inside the decode loop)

    Order is first-seen; duplicates are dropped.  A drained page is not
    remembered, so a page that stays unresolvable is re-queued on the next
    decode step.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: dict[str, dict[str, None]] = {}

    def __call__(self, session_id: str, page_ids) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if isinstance(page_ids, (str, bytes)):
            raise TypeError("page_ids must be a sequence, not a string")
        ids = list(page_ids)
        if not ids:
            return
        if any(not isinstance(page_id, str) or not page_id for page_id in ids):
            raise ValueError("page_ids must be non-empty strings")
        with self._lock:
            pending = self._pending.setdefault(session_id, {})
            for page_id in ids:
                pending.setdefault(page_id, None)

    def pending(self, session_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._pending.get(session_id, {}))

    def pending_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._pending.get(session_id, ()))

    def drain(self, session_id: str) -> tuple[str, ...]:
        with self._lock:
            pending = self._pending.pop(session_id, {})
            return tuple(pending)

    def sessions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._pending)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._pending.pop(session_id, None)


class MaterializingCandidateProvider:
    """Implement ``CandidateProvider`` on top of a ``RAGMaterializer``.

    Parameters
    ----------
    materializer:
        Session-local materializer.  ``head_resolver(session_id, layer_idx)``
        is required.  If it also exposes ``page_is_ready(session_id,
        page_id)``, the adapter uses it to tell "not materialized" (defer)
        from "materialized but no leaf map for this layer" (zero leaves,
        never defer).  A wrapper that exposes only ``head_resolver`` loses
        that protection and can defer an already-encoded page; such a wrapper
        must pass ``page_is_ready=`` explicitly.
    session_id:
        The current request's session.
    n_kv_heads:
        Number of KV heads.  The fallback provider must return
        ``[n_kv_heads, num_neighbours]``.
    fallback_provider:
        Existing ``CandidateProvider`` (for example the centroid provider or
        the native DCI path).  Its result is validated by
        ``build_candidate_plan``.
    select_rag:
        ``select_rag(layer_idx, query_states, state)`` returns the per-head
        selection described in the module docstring.  Selection policy stays
        caller-owned.
    num_leaves:
        ``num_leaves(layer_idx, state)`` returns the per-KV-head native leaf
        limits for this layer (scalar or 1-D array).  Usually
        ``lambda layer, state: state.dci_db[layer].num_leaves``.
    on_deferred:
        Optional ``on_deferred(session_id, page_ids)``.  This is the only place
        deferred page IDs may go; it must not touch native leaf bookkeeping.
        Ordinary exceptions are stored in ``last_deferred_error`` and do not
        fail the token; ``BaseException`` propagates.
    apply_rag:
        ``False`` runs the full selection/plan/deferred pipeline but returns
        the validated fallback leaf IDs.  This is the "RAG in shadow only,
        logits unchanged" baseline: metrics and materialization requests are
        still collected, so hit rate can be compared against a later live arm.
    page_is_ready:
        Optional ``page_is_ready(session_id, page_id) -> bool`` override.
        When omitted, ``materializer.page_is_ready`` is used if present.
    trusted_resolver:
        ``None`` (default) trusts the built-in, un-overridden
        ``RAGMaterializer`` resolver only.  A subclass that overrides
        ``head_resolver``, ``resolve_head_leaves`` or ``resolve_page_leaves``
        — or shadows any of them on the instance — must opt in explicitly.
        The trusted path still range-checks result arrays wider than int32 and
        the merged per-head leaves.
    use_prefix_cache:
        ``True`` (default) reuses the merged plan rows across decode tokens
        when the ready leaves fill every head's width (the steady state at
        production shape).  A materializer epoch invalidates entries on every
        state/leaf-map transition, and a hit still applies the current
        ``num_leaves`` range check, so results are identical to recomputing.
        Enabled only for the built-in resolver and readiness hook; set
        ``False`` for a live A/B or when wrapping a materializer whose leaf
        visibility can change outside its own mutators.
    skip_fallback_on_full_hit:
        ``True`` (default) skips the fallback query entirely on a plan-cache hit whose
        ready leaves fill every head's width (the fallback's values are
        provably unused, and its per-token DCI query is then pure cost).
        Only applies with ``apply_rag=True``; shadow mode always runs the
        fallback because it returns it.  Requires the plan cache, so it is
        inert with ``use_prefix_cache=False`` or a non-builtin resolver.
        On such a hit the recorded plan has ``fallback_queried=False`` and
        its ``fallback_leaf_ids`` mirrors the ready rows.

    Selection, fallback-provider and plan-building errors propagate (fail
    closed) rather than silently degrading to an unvalidated candidate set;
    ordinary ``Exception``\\s from ``on_deferred`` are stored instead (and do
    not fail the token), while ``BaseException`` (cancellation/interrupt)
    propagates so the request can abort.  A caller that wants a last-resort
    fallback should make the fallback provider itself robust.
    """

    def __init__(
        self,
        *,
        materializer,
        session_id: str,
        n_kv_heads: int,
        fallback_provider,
        select_rag: Callable,
        num_leaves: Callable,
        on_deferred: Callable | None = None,
        page_is_ready: Callable | None = None,
        trusted_resolver: bool | None = None,
        apply_rag: bool = True,
        use_prefix_cache: bool = True,
        skip_fallback_on_full_hit: bool = True,
    ) -> None:
        if not hasattr(materializer, "head_resolver"):
            raise TypeError("materializer must expose head_resolver(session_id, layer_idx)")
        if page_is_ready is not None and not callable(page_is_ready):
            raise TypeError("page_is_ready must be callable or None")
        builtin_ready = False
        if page_is_ready is None:
            discovered = getattr(materializer, "page_is_ready", None)
            if callable(discovered):
                page_is_ready = discovered
                builtin_ready = (
                    getattr(discovered, "__func__", None)
                    is RAGMaterializer.page_is_ready
                    and getattr(discovered, "__self__", None) is materializer
                )
        else:
            builtin_ready = (
                getattr(page_is_ready, "__func__", None)
                is RAGMaterializer.page_is_ready
                and getattr(page_is_ready, "__self__", None) is materializer
            )
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not hasattr(fallback_provider, "query_leaf_ids"):
            raise TypeError("fallback_provider must expose query_leaf_ids(...)")
        for name, value in (("select_rag", select_rag), ("num_leaves", num_leaves)):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        if on_deferred is not None and not callable(on_deferred):
            raise TypeError("on_deferred must be callable or None")
        # RAGMaterializer maps are validated at registration time.  Only
        # trust the built-in resolver: a subclass could override either
        # hook and return unvalidated leaves, including int64 IDs that
        # would alias when cast to int32.
        builtin_resolver = isinstance(materializer, RAGMaterializer) and all(
            _is_builtin_method(materializer, name)
            for name in (
                "head_resolver",
                "resolve_head_leaves",
                "resolve_page_leaves",
                # ``head_resolver`` delegates to ``_head_rows``; overriding
                # only that private helper must also disable auto-trust.
                "_head_rows",
            )
        )
        if trusted_resolver is None:
            trusted_resolver = builtin_resolver
        elif not isinstance(trusted_resolver, bool):
            raise TypeError("trusted_resolver must be a bool or None")
        if not isinstance(apply_rag, bool):
            raise TypeError("apply_rag must be a bool")
        if not isinstance(use_prefix_cache, bool):
            raise TypeError("use_prefix_cache must be a bool")
        if not isinstance(skip_fallback_on_full_hit, bool):
            raise TypeError("skip_fallback_on_full_hit must be a bool")
        self.trusted_resolver = trusted_resolver
        self.apply_rag = apply_rag
        self.use_prefix_cache = use_prefix_cache
        self.skip_fallback_on_full_hit = skip_fallback_on_full_hit
        # The epoch-invalidated merge cache is only safe with the built-in
        # resolver, the built-in readiness hook and the materializer's own
        # mutation methods (all three hold for an unmodified RAGMaterializer).
        # The cache's epoch lives on the materializer its factory was called
        # on.  A foreign bound factory returns a cache owned by another
        # materializer, so mutations of this one would never invalidate it.
        builtin_cache = (
            isinstance(materializer, RAGMaterializer)
            and _is_builtin_method(materializer, "plan_prefix_cache")
        )
        self._plan_cache_enabled = bool(
            use_prefix_cache
            and trusted_resolver
            and builtin_resolver
            and builtin_ready
            and builtin_cache
        )
        # Freeze the factory: a post-construction instance shadow must not be
        # able to redirect the cache to another materializer's epoch.
        self._plan_cache_factory = (
            materializer.plan_prefix_cache
            if self._plan_cache_enabled
            else None
        )
        # No cache registry here: ``_layer_plan_cache`` asks the frozen
        # factory each time, and the materializer owns the
        # ``(session, layer)`` registry (dropped by ``release_session``).  A
        # provider-held dict keyed by session would retain one cache object
        # -- with up to ``PlanPrefixCache.MAX_ENTRIES`` merged plan rows --
        # for every session a long-lived pooled provider ever served.
        self.materializer = materializer
        self.session_id = session_id
        self.n_kv_heads = _require_positive_int(n_kv_heads, name="n_kv_heads")
        # Freeze the resolver factory at construction.  Auto-trust was
        # decided above; fetching ``materializer.head_resolver`` per call
        # would let a post-construction instance shadow bypass the check.
        self._resolver_factory = materializer.head_resolver
        self._page_is_ready = page_is_ready
        self.fallback_provider = fallback_provider
        self.select_rag = select_rag
        self.num_leaves = num_leaves
        self.on_deferred = on_deferred
        self._lock = threading.RLock()
        self._last_plan: CandidatePlan | None = None
        self._plan_count = 0
        self._last_deferred_error: BaseException | None = None
        self._stats = {
            "calls": 0,
            "ready_rag_pages": 0.0,
            "deferred_rag_pages": 0.0,
            "rag_leaves_used": 0.0,
            "fallback_leaves_used": 0.0,
            "deferred_events": 0,
            "deferred_errors": 0,
            "shadow_calls": 0,
        }

    # -- introspection ---------------------------------------------------
    @property
    def last_plan(self) -> CandidatePlan | None:
        """The most recent plan.

        This is last-writer-wins and is not bound to the caller, so it is
        intended for a single-caller decode loop (inspection/logging).  It is
        not a per-thread result; do not read it from a concurrent request.
        """
        with self._lock:
            return self._last_plan

    @property
    def plan_count(self) -> int:
        with self._lock:
            return self._plan_count

    @property
    def last_deferred_error(self) -> BaseException | None:
        with self._lock:
            return self._last_deferred_error

    @property
    def stats(self) -> dict[str, float | int]:
        """Cumulative per-call plan metrics plus a materialization hit rate.

        Counts are summed over ``query_leaf_ids`` calls, so with one call per
        indexed layer per decode token they are "page-slots" (layer × token),
        not unique pages.  ``materialization_hit_rate`` is
        ``ready / (ready + deferred)`` and is ``0.0`` when nothing was
        selected.
        """
        with self._lock:
            snapshot = dict(self._stats)
        selected = snapshot["ready_rag_pages"] + snapshot["deferred_rag_pages"]
        snapshot["selected_rag_pages"] = selected
        snapshot["materialization_hit_rate"] = (
            snapshot["ready_rag_pages"] / selected if selected > 0 else 0.0
        )
        return snapshot

    def reset_stats(self) -> None:
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0.0 if isinstance(self._stats[key], float) else 0

    # -- CandidateProvider ----------------------------------------------
    def _layer_plan_cache(self, layer_idx: int) -> PlanPrefixCache:
        """Cache object for the current (materializer, session, layer).

        The factory is frozen at construction and re-queried on every call:
        the materializer's own registry is keyed by ``(session, layer)``, so
        repointing a pooled provider cannot replay another session's rows,
        and ``release_session`` drops the entries instead of leaking one
        cache object per session served.
        """
        factory = self._plan_cache_factory
        if factory is None:
            raise RuntimeError(
                "plan cache is not enabled; check _plan_cache_enabled"
            )
        cache = factory(self.session_id, layer_idx)
        if getattr(cache, "_owner", None) is not self.materializer:
            raise RuntimeError(
                "plan_prefix_cache returned a cache owned by another "
                "materializer; refusing to replay foreign rows"
            )
        return cache

    def _per_head_selection(self, selection: object) -> list[list[object]]:
        if isinstance(selection, Mapping):
            rows: list[list[object]] = [[] for _ in range(self.n_kv_heads)]
            for raw_head, items in selection.items():
                if not isinstance(raw_head, (int, np.integer)) or isinstance(
                    raw_head, bool
                ):
                    raise TypeError(
                        "selection mapping keys must be integer KV head indices"
                    )
                head = int(raw_head)
                if not 0 <= head < self.n_kv_heads:
                    raise ValueError(
                        f"selection head {head} outside [0, {self.n_kv_heads})"
                    )
                if isinstance(items, (str, bytes)):
                    raise TypeError("per-head selection must be a sequence")
                rows[head].extend(items)
            return rows
        if isinstance(selection, (str, bytes)):
            raise TypeError("selection must not be a string")
        if hasattr(selection, "selected"):
            selection = selection.selected
        if selection is None:
            raise TypeError("selection must not be None")
        items = list(selection)
        if len(items) == self.n_kv_heads and all(
            isinstance(row, (list, tuple)) for row in items
        ):
            return [list(row) for row in items]
        return group_selected_by_head(items, self.n_kv_heads)

    def _record_plan(self, plan: CandidatePlan) -> None:
        """Store ``last_plan``, accumulate stats and publish deferred pages."""
        metrics = plan.metrics
        with self._lock:
            self._last_plan = plan
            self._plan_count += 1
            stats = self._stats
            stats["calls"] += 1
            stats["ready_rag_pages"] += metrics.get("n_ready_rag_pages", 0.0)
            stats["deferred_rag_pages"] += metrics.get("n_deferred_rag_pages", 0.0)
            stats["rag_leaves_used"] += metrics.get("n_rag_leaves_used", 0.0)
            stats["fallback_leaves_used"] += metrics.get(
                "n_fallback_leaves_used", 0.0
            )
            if not self.apply_rag:
                stats["shadow_calls"] += 1

        if plan.deferred_rag_pages and self.on_deferred is not None:
            with self._lock:
                self._stats["deferred_events"] += 1
            try:
                self.on_deferred(self.session_id, plan.deferred_rag_pages)
            except Exception as exc:  # generation must continue on ordinary errors
                with self._lock:
                    self._last_deferred_error = exc
                    self._stats["deferred_errors"] += 1
            else:
                with self._lock:
                    self._last_deferred_error = None

    def _query_leaf_ids_with_plan(
        self,
        *,
        layer_idx: int,
        query_states,
        num_neighbours: int,
        state,
    ) -> tuple[np.ndarray, CandidatePlan]:
        # The error belongs to the call that produced it; clear it before any
        # validation, so even a call rejected on its arguments does not keep
        # reporting an older on_deferred failure.
        with self._lock:
            self._last_deferred_error = None
        if (
            not isinstance(layer_idx, (int, np.integer))
            or isinstance(layer_idx, bool)
            or int(layer_idx) < 0
        ):
            raise ValueError("layer_idx must be a non-negative integer")
        layer_idx = int(layer_idx)
        width = _require_positive_int(num_neighbours, name="num_neighbours")

        # Full-prefix-cache hit: every head is filled by ready leaves, so the
        # fallback's values are unused.  Probe before running it; the per-token
        # native DCI query is otherwise pure cost on this path.
        full_hit = (
            self.skip_fallback_on_full_hit
            and self.apply_rag
            and self._plan_cache_enabled
        )
        per_head = None
        limits = None
        if full_hit:
            # An empty cache cannot hit.  Checking that first skips the
            # selection parse during the deferred/encoding window, where every
            # call is a miss anyway (measured: it removed a ~40-90 us/layer
            # regression on the all-deferred path).
            cache = self._layer_plan_cache(layer_idx)
            entry = None
            if len(cache):
                limits = self.num_leaves(layer_idx, state)
                selection = self.select_rag(layer_idx, query_states, state)
                per_head = self._per_head_selection(selection)
                per_head_ids = normalise_selected_page_ids(
                    per_head, n_heads=self.n_kv_heads, layer_idx=layer_idx
                )
                entry = cache.probe(
                    per_head_ids,
                    width=width,
                    limits=limits,
                    n_heads=self.n_kv_heads,
                )
            if entry is not None:
                plan = CandidatePlan._from_frozen_ready(
                    ready_leaf_ids=entry.rows,
                    # Unused; mirrors ready rows only to keep the fixed shape.
                    fallback_leaf_ids=entry.rows,
                    deferred_rag_pages=entry.deferred,
                    selection_metrics=entry.plan_metrics(
                        n_heads=self.n_kv_heads, width=width
                    ),
                    fallback_queried=False,
                )
                self._record_plan(plan)
                return np.array(plan.ready_leaf_ids, copy=True), plan

        fallback = np.asarray(
            self.fallback_provider.query_leaf_ids(
                layer_idx=layer_idx,
                query_states=query_states,
                num_neighbours=width,
                state=state,
            )
        )
        expected = (self.n_kv_heads, width)
        if fallback.shape != expected:
            raise ValueError(
                f"fallback provider returned {fallback.shape}, expected {expected}"
            )
        if not np.issubdtype(fallback.dtype, np.integer):
            raise TypeError(
                f"fallback provider returned {fallback.dtype}; leaf IDs must be "
                "integers"
            )

        if per_head is None:
            limits = self.num_leaves(layer_idx, state)
            selection = self.select_rag(layer_idx, query_states, state)
            per_head = self._per_head_selection(selection)
        ready_hook = self._page_is_ready
        is_page_ready = (
            (lambda page_id: ready_hook(self.session_id, page_id))
            if ready_hook is not None
            else None
        )
        plan = build_candidate_plan(
            fallback_leaf_ids=fallback,
            selected_rag_by_head=per_head,
            resolve_leaves=self._resolver_factory(
                self.session_id, layer_idx
            ),
            num_leaves=limits,
            layer_idx=layer_idx,
            is_page_ready=is_page_ready,
            trusted_resolver=self.trusted_resolver,
            prefix_cache=(
                self._layer_plan_cache(layer_idx)
                if self._plan_cache_enabled
                else None
            ),
        )
        self._record_plan(plan)

        # Shadow mode keeps the model on the validated fallback while the
        # plan, its metrics and any deferred pages are still recorded.
        source = plan.ready_leaf_ids if self.apply_rag else fallback
        # A writable copy: the plan's arrays are immutable, but the native
        # path and some consumers expect a regular array.
        return np.array(source, copy=True), plan

    def query_candidate_result(
        self,
        *,
        layer_idx: int,
        query_states,
        num_neighbours: int,
        state,
    ) -> CandidateQueryResult:
        """Return this call's effective rows and plan as one result object."""
        effective, plan = self._query_leaf_ids_with_plan(
            layer_idx=layer_idx,
            query_states=query_states,
            num_neighbours=num_neighbours,
            state=state,
        )
        return CandidateQueryResult(effective_leaf_ids=effective, plan=plan)

    def query_leaf_ids(
        self,
        *,
        layer_idx: int,
        query_states,
        num_neighbours: int,
        state,
    ) -> np.ndarray:
        """Compatibility API for runtimes that do not consume plans yet."""
        return self.query_candidate_result(
            layer_idx=layer_idx,
            query_states=query_states,
            num_neighbours=num_neighbours,
            state=state,
        ).effective_leaf_ids


def _self_test() -> None:
    checks = 0

    def ok(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    def raises(exc_type, fn, *args, **kwargs) -> None:
        nonlocal checks
        try:
            fn(*args, **kwargs)
        except exc_type:
            checks += 1
            return
        raise AssertionError(f"expected {exc_type.__name__} from {fn}")

    from icecache.materialization import GlobalRAGPageStore, chunk_document

    pages = chunk_document(
        np.arange(8, dtype=np.int64),
        document_id="doc",
        chunk_size=4,
        descriptor_version="v1",
    )
    store = GlobalRAGPageStore()
    store.add_many(pages)
    mat = RAGMaterializer(store, n_kv_heads=2)
    page_id = pages[0].page_id
    mat.submit("s", [page_id])
    mat.begin_encoding("s", [page_id], token_start=0, token_end=4)
    nodes = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ],
        dtype=np.int32,
    )
    mat.register_layer_leaves("s", page_id, 0, nodes, np.array([4, 4]),
                              token_offset=0)
    mat.complete_encoding("s", [page_id])

    fallback_arr = np.array([[2, 3], [2, 3]], dtype=np.int32)
    limits = np.array([4, 4], dtype=np.int32)

    class _Fallback:
        def __init__(self) -> None:
            self.calls = 0

        def query_leaf_ids(self, *, layer_idx, query_states, num_neighbours,
                           state):
            self.calls += 1
            return fallback_arr

    fallback = _Fallback()
    calls: list[tuple] = []

    def selector(layer, query, state):
        calls.append((layer, query, state))
        return [[page_id], [page_id]]

    def num_leaves(layer, state):
        return limits

    provider = MaterializingCandidateProvider(
        materializer=mat,
        session_id="s",
        n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
    )
    result = provider.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(result.tolist() == [[0, 1], [0, 1]], "ready RAG leaves are used")
    ok(result.flags.writeable, "adapter returns a writable copy")
    ok(provider.plan_count == 1, "plan count")
    ok(provider.last_plan is not None, "last plan exposed")
    ok(provider.last_plan.deferred_rag_pages == (), "nothing deferred")
    ok(provider.last_plan.metrics["n_rag_leaves_used"] == 4.0,
       "metrics exposed")
    ok(fallback.calls == 1 and calls[0][0] == 0, "fallback/selector called once")
    stats = provider.stats
    ok(stats["calls"] == 1 and stats["ready_rag_pages"] == 1.0,
       "stats count a ready page")
    ok(stats["rag_leaves_used"] == 4.0 and stats["fallback_leaves_used"] == 0.0,
       "stats count leaves")
    ok(stats["materialization_hit_rate"] == 1.0, "stats hit rate")
    ok(stats["shadow_calls"] == 0, "stats shadow calls")

    # Decode-hot merge cache: the built-in resolver/hook enables it, a second
    # identical call hits it, and its output is bit-identical either way.
    # Layer 2 keeps this section off the counters used by later tests.
    mat.register_layer_leaves("s", page_id, 2, nodes, np.array([4, 4]),
                              token_offset=0)
    cached = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
    )
    ok(cached._plan_cache_enabled, "plan cache enabled for builtin materializer")
    cached0 = cached.query_leaf_ids(
        layer_idx=2, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    cached1 = cached.query_leaf_ids(
        layer_idx=2, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(np.array_equal(cached0, result) and np.array_equal(cached1, cached0),
       "cached decode result matches")
    cache = cached._layer_plan_cache(2)
    # The empty-cache gate skips the probe on the first call, so only
    # ``build_candidate_plan``'s lookup misses and stores the entry.
    ok(cache.hits == 1 and cache.misses == 1, "cache miss then hit")
    plain = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
        use_prefix_cache=False,
    )
    ok(not plain._plan_cache_enabled, "cache can be disabled for live A/B")
    ok(np.array_equal(
        plain.query_leaf_ids(
            layer_idx=2, query_states=np.zeros((2, 4)), num_neighbours=2,
            state=None,
        ),
        cached0,
    ), "uncached provider matches the cached result")
    # A custom readiness hook is not epoch-tracked, so the cache stays off.
    custom_ready = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
        page_is_ready=lambda page_id: mat.page_is_ready("s", page_id),
    )
    ok(not custom_ready._plan_cache_enabled,
       "custom readiness hook disables the cache")
    # Explicit conservative validation keeps the cache off: the cached path
    # requires trusted_resolver=True, so enabling it would raise per token.
    conservative = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
        trusted_resolver=False,
    )
    ok(not conservative._plan_cache_enabled,
       "explicit trusted_resolver=False disables the cache")
    ok(np.array_equal(
        conservative.query_leaf_ids(
            layer_idx=2, query_states=np.zeros((2, 4)), num_neighbours=2,
            state=None,
        ),
        cached0,
    ), "trusted_resolver=False still returns the same rows")

    # A subclass that only overrides _head_rows must not be auto-trusted.
    class _HeadRowsOverride(RAGMaterializer):
        def _head_rows(self, session_id, layer_idx):
            return super()._head_rows(session_id, layer_idx)

    shadowed_rows = MaterializingCandidateProvider(
        materializer=_HeadRowsOverride(store, n_kv_heads=2),
        session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
    )
    ok(not shadowed_rows.trusted_resolver
       and not shadowed_rows._plan_cache_enabled,
       "_head_rows override disables auto-trust and the cache")

    # A foreign bound method has the built-in __func__ but reads another
    # materializer's state, whose mutations never bump this one's epoch.
    mat2 = RAGMaterializer(store, n_kv_heads=2)
    foreign_resolver = RAGMaterializer(store, n_kv_heads=2)
    foreign_resolver.head_resolver = mat2.head_resolver
    foreign_provider = MaterializingCandidateProvider(
        materializer=foreign_resolver, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
    )
    ok(not foreign_provider.trusted_resolver
       and not foreign_provider._plan_cache_enabled,
       "foreign-bound head_resolver disables auto-trust and the cache")
    foreign_ready = RAGMaterializer(store, n_kv_heads=2)
    foreign_ready.page_is_ready = mat2.page_is_ready
    ready_provider = MaterializingCandidateProvider(
        materializer=foreign_ready, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
    )
    ok(not ready_provider._plan_cache_enabled,
       "foreign-bound readiness hook disables the cache")
    foreign_cache = RAGMaterializer(store, n_kv_heads=2)
    foreign_cache.plan_prefix_cache = mat2.plan_prefix_cache
    cache_provider = MaterializingCandidateProvider(
        materializer=foreign_cache, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
    )
    ok(not cache_provider._plan_cache_enabled,
       "foreign-bound plan_prefix_cache disables the cache")

    class _CacheOverride(RAGMaterializer):
        def plan_prefix_cache(self, session_id, layer_idx):
            return super().plan_prefix_cache(session_id, layer_idx)

    override_provider = MaterializingCandidateProvider(
        materializer=_CacheOverride(store, n_kv_heads=2), session_id="s",
        n_kv_heads=2, fallback_provider=fallback, select_rag=selector,
        num_leaves=num_leaves,
    )
    ok(not override_provider._plan_cache_enabled,
       "plan_prefix_cache override disables the cache")

    # The factory is frozen at construction: a later instance shadow cannot
    # redirect an already-enabled provider to another materializer's epoch.
    frozen_cache = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback, select_rag=selector, num_leaves=num_leaves,
    )
    ok(frozen_cache._plan_cache_enabled, "builtin materializer enables cache")
    mat.plan_prefix_cache = mat2.plan_prefix_cache
    try:
        owned = frozen_cache._layer_plan_cache(2)
    finally:
        del mat.plan_prefix_cache
    ok(getattr(owned, "_owner", None) is mat,
       "post-construction cache shadow cannot redirect the factory")

    # Repointing a pooled provider must not replay the old session's rows.
    # The same global page is materialized in both sessions with different
    # per-layer leaves, so the expected rows are unambiguous.
    page2 = pages[1].page_id
    mat.submit("s", [page2])
    mat.begin_encoding("s", [page2], token_start=0, token_end=4)
    mat.register_layer_leaves("s", page2, 2,
                              np.array([[0, 1, 1, 0], [0, 1, 0, 1]]),
                              np.array([4, 4]), token_offset=0)
    mat.complete_encoding("s", [page2])
    mat.submit("s2", [page2])
    mat.begin_encoding("s2", [page2], token_start=0, token_end=4)
    nodes2 = np.array([[3, 3, 2, 2], [3, 2, 3, 2]], dtype=np.int32)
    mat.register_layer_leaves("s2", page2, 2, nodes2, np.array([4, 4]),
                              token_offset=0)
    mat.complete_encoding("s2", [page2])
    repoint = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, q, s: [[page2], [page2]],
        num_leaves=num_leaves,
    )
    before = repoint.query_leaf_ids(
        layer_idx=2, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(before.tolist() == [[0, 1], [0, 1]], "session s returns s rows")
    s_cache = repoint._layer_plan_cache(2)
    repoint.session_id = "s2"
    after = repoint.query_leaf_ids(
        layer_idx=2, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    # register_layer_leaves sorts each head's unique leaves ascending.
    ok(after.tolist() == [[2, 3], [2, 3]],
       "repointed provider resolves the new session's leaves")
    ok(repoint._layer_plan_cache(2) is not s_cache
       and repoint._layer_plan_cache(2).misses == 1,
       "a repointed session gets its own cache object")

    # Every materializer mutation bumps the epoch; the next call recomputes.
    # ``repoint`` shares this (session, layer) cache object, so compare deltas.
    hits_before, misses_before = cache.hits, cache.misses
    mat.mark_gpu("s", [page_id])
    mat.mark_cpu("s", [page_id])
    again = cached.query_leaf_ids(
        layer_idx=2, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(np.array_equal(again, cached0), "epoch invalidation recomputes the rows")
    # The epoch bump empties the cache, so the gate skips the probe: one miss.
    ok(cache.hits == hits_before and cache.misses == misses_before + 1,
       f"epoch mutation forces a miss (hits={cache.hits}, misses={cache.misses})")

    # Full-hit fast path: once a full-ready plan is cached, the fallback's
    # values are unused, so the provider must not run its query at all.
    # Rows must still be bit-identical to the normal path.  Layer 3 is a
    # fresh cache, so the first call is a real miss.
    mat.register_layer_leaves("s", page_id, 3, nodes, np.array([4, 4]),
                              token_offset=0)
    skip = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
        skip_fallback_on_full_hit=True,
    )
    fb_before = fallback.calls
    skip0 = skip.query_leaf_ids(
        layer_idx=3, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(fallback.calls == fb_before + 1, "full-hit fast path runs the fallback on miss")
    skip1 = skip.query_leaf_ids(
        layer_idx=3, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(fallback.calls == fb_before + 1,
       "full-hit fast path skips the fallback query on a hit")
    ok(np.array_equal(skip0, cached0) and np.array_equal(skip1, skip0),
       "fast path returns the same rows")
    ok(skip.last_plan is not None and skip.last_plan.fallback_queried is False,
       "fast-path plan marks its fallback unqueried")
    ok(np.array_equal(skip.last_plan.fallback_leaf_ids, skip1),
       "fast-path fallback field mirrors the ready rows")
    # An epoch miss must run the fallback and return the native result.
    mat.mark_gpu("s", [page_id])
    mat.mark_cpu("s", [page_id])
    fb_before = fallback.calls
    skip2 = skip.query_leaf_ids(
        layer_idx=3, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(fallback.calls == fb_before + 1 and np.array_equal(skip2, cached0),
       "epoch miss runs the fallback and returns the same rows")
    # Shadow mode returns the fallback itself, so skip must be inert there.
    shadow_skip = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
        apply_rag=False,
        skip_fallback_on_full_hit=True,
    )
    fb_before = fallback.calls
    shadow_skip.query_leaf_ids(
        layer_idx=3, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    shadow_rows = shadow_skip.query_leaf_ids(
        layer_idx=3, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(fallback.calls == fb_before + 2
       and np.array_equal(shadow_rows, fallback_arr),
       "shadow mode always queries and returns the fallback")
    # Without the plan cache there is no hit to skip.
    skip_plain = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
        use_prefix_cache=False,
        skip_fallback_on_full_hit=True,
    )
    fb_before = fallback.calls
    skip_plain.query_leaf_ids(
        layer_idx=3, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    skip_plain.query_leaf_ids(
        layer_idx=3, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(fallback.calls == fb_before + 2,
       "skip is inert without the plan cache")

    # Shadow mode computes the same plan but keeps the model on fallback.
    shadow_calls: list[tuple] = []

    def shadow_deferred(session_id, page_ids):
        shadow_calls.append((session_id, tuple(page_ids)))

    shadow = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=selector,
        num_leaves=num_leaves,
        on_deferred=shadow_deferred,
        apply_rag=False,
    )
    shadow_result = shadow.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(shadow_result.tolist() == fallback_arr.tolist(),
       "shadow mode returns fallback")
    ok(shadow.last_plan.metrics["n_rag_leaves_used"] == 4.0,
       "shadow mode still records metrics")
    ok(shadow.last_plan.deferred_rag_pages == (),
       "shadow mode still records deferrals")
    ok(shadow.apply_rag is False and provider.apply_rag is True,
       "apply_rag exposed")
    ok(shadow.stats["shadow_calls"] == 1
       and shadow.stats["ready_rag_pages"] == 1.0,
       "shadow stats still record the plan")
    raises(TypeError, MaterializingCandidateProvider, materializer=mat,
           session_id="s", n_kv_heads=2, fallback_provider=fallback,
           select_rag=selector, num_leaves=num_leaves, apply_rag=1)

    # Pure prompt: no selection means the fallback is returned verbatim.
    pure = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[], []],
        num_leaves=num_leaves,
    )
    result = pure.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(result.tolist() == fallback_arr.tolist(), "pure prompt equals fallback")
    ok(pure.last_plan.metrics["n_rag_leaves_used"] == 0.0,
       "pure prompt uses no RAG leaves")

    # Unmaterialized pages are deferred, never scattered; the fallback fills.
    deferred_calls: list[tuple] = []

    def on_deferred(session_id, page_ids):
        deferred_calls.append((session_id, tuple(page_ids)))

    ghost = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [["ghost/page"], ["ghost/page"]],
        num_leaves=num_leaves,
        on_deferred=on_deferred,
    )
    result = ghost.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(result.tolist() == fallback_arr.tolist(), "deferred page falls back")
    ok(deferred_calls == [("s", ("ghost/page",))], "deferred callback")
    ok(ghost.last_deferred_error is None, "no deferred error")
    ghost_stats = ghost.stats
    ok(ghost_stats["deferred_rag_pages"] == 1.0
       and ghost_stats["deferred_events"] == 1, "stats count deferrals")
    ok(ghost_stats["materialization_hit_rate"] == 0.0,
       "deferred-only hit rate is zero")
    ok(ghost_stats["fallback_leaves_used"] == 4.0,
       "stats count fallback leaves")
    ghost.reset_stats()
    ok(ghost.stats["calls"] == 0 and ghost.stats["selected_rag_pages"] == 0.0,
       "reset_stats clears")

    # Deferred requests are recorded only; decode never starts encoding.
    queue = DeferredPageQueue()
    queued = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [["ghost/page"], ["ghost/page"]],
        num_leaves=num_leaves,
        on_deferred=queue,
    )
    ok(queued.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2,
        state=None,
    ).tolist() == fallback_arr.tolist(), "queued page still falls back")
    ok(queue.pending("s") == ("ghost/page",), "deferred page queued")
    ok(mat.records("s").get("ghost/page") is None,
       "deferred page has no record until the next boundary")
    queue("s", ("a", "b"))
    queue("s", ("b", "c"))
    ok(queue.pending("s") == ("ghost/page", "a", "b", "c"),
       "queue dedupes and preserves order")
    ok(queue.pending_count("s") == 4, "queue count")
    ok(queue.sessions() == ("s",), "queue sessions")
    ok(queue.drain("s") == ("ghost/page", "a", "b", "c"), "queue drain")
    ok(queue.drain("s") == (), "drain clears")
    ok(queue.pending("s") == (), "drain leaves nothing pending")
    raises(ValueError, queue, "", ("a",))
    raises(TypeError, queue, "s", "a")
    raises(ValueError, queue, "s", ("",))
    queue.clear("s")

    def queue_worker(worker_id: int) -> None:
        for step in range(50):
            queue(f"q{worker_id}", (f"p{worker_id}-{step}",))

    threads = [threading.Thread(target=queue_worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ok(all(queue.pending_count(f"q{i}") == 50 for i in range(4)),
       "queue is thread-safe")

    # A failing deferred callback records the error but never fails the token.
    def bad_callback(session_id, page_ids):
        raise RuntimeError("submit failed")

    ghost_fail = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [["ghost/page"], ["ghost/page"]],
        num_leaves=num_leaves,
        on_deferred=bad_callback,
    )
    result = ghost_fail.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(result.tolist() == fallback_arr.tolist(), "callback failure keeps fallback")
    ok(isinstance(ghost_fail.last_deferred_error, RuntimeError),
       "callback error recorded")

    # The recorded error belongs to its call: a later clean prompt-only call
    # must not keep reporting it.
    flag = {"defer": True}
    flaky = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: (
            [["ghost/page"], ["ghost/page"]] if flag["defer"] else [[], []]
        ),
        num_leaves=num_leaves,
        on_deferred=bad_callback,
    )
    flaky.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(isinstance(flaky.last_deferred_error, RuntimeError),
       "failing call records the deferred error")
    flag["defer"] = False
    clean = flaky.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(clean.tolist() == fallback_arr.tolist(), "clean call keeps fallback")
    ok(flaky.last_deferred_error is None,
       "a later call clears the stale deferred error")

    # A ready page with no map for the queried layer (e.g. after
    # invalidate_layer) contributes no leaves but is never deferred: the
    # deferred channel would try to re-encode an already encoded page.
    stale_queue: list[tuple] = []

    def stale_deferred(session_id, page_ids):
        stale_queue.append((session_id, tuple(page_ids)))

    stale = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[page_id], [page_id]],
        num_leaves=num_leaves,
        on_deferred=stale_deferred,
    )
    stale_result = stale.query_leaf_ids(
        layer_idx=1, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(stale_result.tolist() == fallback_arr.tolist(),
       "layer without a map falls back")
    ok(stale.last_plan.deferred_rag_pages == (),
       "ready page without a layer map is not deferred")
    ok(stale_queue == [], "no deferred event for an already-encoded page")
    ok(stale.stats["ready_rag_pages"] == 1.0
       and stale.stats["deferred_rag_pages"] == 0.0,
       "missing-map page counts as materialized")

    # Auto-trust is limited to the built-in resolver; a subclass that
    # overrides either hook must opt in explicitly.
    class _Overriding(RAGMaterializer):
        def head_resolver(self, session_id, layer_idx):
            return lambda page_id, head: None

    overridden = MaterializingCandidateProvider(
        materializer=_Overriding(store, n_kv_heads=2), session_id="s",
        n_kv_heads=2, fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[], []],
        num_leaves=num_leaves,
    )
    ok(overridden.trusted_resolver is False,
       "overriding materializer is not auto-trusted")

    # ``resolve_head_leaves`` delegates to ``resolve_page_leaves``, so
    # overriding only the latter is enough to bypass the trusted path's
    # validation and must also disable auto-trust.
    class _OverridingPageLeaves(RAGMaterializer):
        def resolve_page_leaves(self, session_id, page_id, layer_idx):
            return None

    page_leaves = MaterializingCandidateProvider(
        materializer=_OverridingPageLeaves(store, n_kv_heads=2), session_id="s",
        n_kv_heads=2, fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[], []],
        num_leaves=num_leaves,
    )
    ok(page_leaves.trusted_resolver is False,
       "overriding resolve_page_leaves is not auto-trusted")

    # An instance-level shadow of an otherwise built-in hook is just as
    # dangerous as a subclass override.
    shadowed_mat = RAGMaterializer(store, n_kv_heads=2)
    shadowed_mat.head_resolver = lambda session_id, layer_idx: (
        lambda page_id, head: None
    )
    shadowed_provider = MaterializingCandidateProvider(
        materializer=shadowed_mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[], []],
        num_leaves=num_leaves,
    )
    ok(shadowed_provider.trusted_resolver is False,
       "instance-shadowed resolver is not auto-trusted")

    # A wrapper that exposes only ``head_resolver`` cannot tell "not encoded"
    # from "encoded but no map for this layer".  That limitation is documented
    # and can be removed only by passing ``page_is_ready`` explicitly.
    class _ResolverOnly:
        def __init__(self, inner):
            self.inner = inner

        def head_resolver(self, session_id, layer_idx):
            return lambda page_id, head: None

    wrapped = _ResolverOnly(mat)
    no_hook = MaterializingCandidateProvider(
        materializer=wrapped, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[page_id], [page_id]],
        num_leaves=num_leaves,
    )
    no_hook_result = no_hook.query_leaf_ids(
        layer_idx=1, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(no_hook_result.tolist() == fallback_arr.tolist(),
       "wrapper without readiness hook falls back")
    ok(no_hook.last_plan.deferred_rag_pages == (page_id,),
       "wrapper without readiness hook defers an already-encoded page")

    with_hook = MaterializingCandidateProvider(
        materializer=wrapped, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[page_id], [page_id]],
        num_leaves=num_leaves,
        page_is_ready=lambda sid, pid: mat.page_is_ready(sid, pid),
    )
    hooked_result = with_hook.query_leaf_ids(
        layer_idx=1, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(hooked_result.tolist() == fallback_arr.tolist(),
       "explicit readiness hook falls back without leaves")
    ok(with_hook.last_plan.deferred_rag_pages == (),
       "explicit readiness hook does not defer an already-encoded page")
    ok(with_hook.stats["ready_rag_pages"] == 1.0,
       "explicit readiness hook counts the page as materialized")
    raises(TypeError, MaterializingCandidateProvider, materializer=mat,
           session_id="s", n_kv_heads=2, fallback_provider=fallback,
           select_rag=lambda layer, query, state: [[], []],
           num_leaves=num_leaves, page_is_ready=42)

    # The stale-error clear happens before argument validation, so even a call
    # rejected while binding its arguments clears the previous failure.
    flag["defer"] = True
    flaky.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(isinstance(flaky.last_deferred_error, RuntimeError),
       "call rejected later starts from a recorded error")
    raises(ValueError, flaky.query_leaf_ids, layer_idx=-1,
           query_states=np.zeros((2, 4)), num_neighbours=2, state=None)
    ok(flaky.last_deferred_error is None,
       "an argument-rejected call clears the stale deferred error")

    # Flat SelectionResult-like results are grouped by page.head_idx.
    class _Page:
        def __init__(self, page_id, head):
            self.page_id = page_id
            self.head_idx = head
            self.source = "rag"

    class _Candidate:
        def __init__(self, page_id, head):
            self.page = _Page(page_id, head)

    class _Result:
        selected = (_Candidate("a", 0), _Candidate("b", 1), _Candidate("c", 1))

    grouped = group_selected_by_head(_Result(), 2)
    ok([len(row) for row in grouped] == [1, 2], "group flat result")
    ok(grouped[0][0].page.page_id == "a", "group preserves order")
    raises(TypeError, group_selected_by_head, ["bare"], 2)
    raises(ValueError, group_selected_by_head, [_Candidate("a", 2)], 2)
    raises(ValueError, group_selected_by_head, [], 0)

    # A Mapping keyed by head is accepted directly.
    mapped = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: {
            0: [_Candidate(page_id, 0)], 1: [_Candidate(page_id, 1)]
        },
        num_leaves=num_leaves,
    )
    result = mapped.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(result.tolist() == [[0, 1], [0, 1]], "mapping selection")

    # Closed loop: unmaterialized at decode -> deferred -> boundary prefill ->
    # the next query resolves the same global page to native leaves.
    from icecache.prefill_materialization import PrefillMaterialization

    closed_queue = DeferredPageQueue()
    waiting = MaterializingCandidateProvider(
        materializer=mat, session_id="loop0", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[page_id], [page_id]],
        num_leaves=num_leaves,
        on_deferred=closed_queue,
    )
    before = waiting.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(before.tolist() == fallback_arr.tolist(),
       "closed loop starts on fallback")
    ok(closed_queue.drain("loop0") == (page_id,), "closed loop deferred page")

    boundary = PrefillMaterialization(mat, "loop")
    loop_prompt = boundary.prepare(pages=[pages[0]], prefix_token_ids=[7, 8])
    offset = loop_prompt.span_for(page_id).token_start
    ok(offset == 2, "closed loop span")
    loop_maps = boundary.register_layer(
        0, nodes, limits, token_offset=offset
    )
    boundary.finish()
    ok(loop_maps[page_id].shape == (2, 2), "closed loop registered")
    ok(mat.poll_ready("loop") == [page_id], "closed loop page ready")

    ready_provider = MaterializingCandidateProvider(
        materializer=mat, session_id="loop", n_kv_heads=2,
        fallback_provider=fallback,
        select_rag=lambda layer, query, state: [[page_id], [page_id]],
        num_leaves=num_leaves,
        on_deferred=closed_queue,
    )
    after = ready_provider.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(after.tolist() == [[0, 1], [0, 1]], "closed loop uses RAG leaves")
    ok(ready_provider.last_plan.deferred_rag_pages == (),
       "closed loop is not deferred")
    ok(closed_queue.pending("loop") == (), "closed loop did not defer again")

    # The other session is unaffected by loop0's deferral and can be released
    # independently.
    mat.release_session("loop")
    ok(mat.poll_ready("loop") == [], "released session is empty")
    relapsed = ready_provider.query_leaf_ids(
        layer_idx=0, query_states=np.zeros((2, 4)), num_neighbours=2, state=None
    )
    ok(relapsed.tolist() == fallback_arr.tolist(),
       "released session falls back")
    ok(closed_queue.pending("loop") == (page_id,),
       "released session re-queues the page")
    closed_queue.clear("loop")

    # Fallback width and layer arguments are validated.
    class _BadShape(_Fallback):
        def query_leaf_ids(self, *, layer_idx, query_states, num_neighbours,
                           state):
            return np.zeros((2, 3), dtype=np.int32)

    bad = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=_BadShape(),
        select_rag=lambda layer, query, state: [[], []],
        num_leaves=num_leaves,
    )
    raises(ValueError, bad.query_leaf_ids, layer_idx=0,
           query_states=np.zeros((2, 4)), num_neighbours=2, state=None)

    class _FloatFallback(_Fallback):
        def query_leaf_ids(self, *, layer_idx, query_states, num_neighbours,
                           state):
            return fallback_arr.astype(np.float32)

    float_fallback = MaterializingCandidateProvider(
        materializer=mat, session_id="s", n_kv_heads=2,
        fallback_provider=_FloatFallback(),
        select_rag=lambda layer, query, state: [[], []],
        num_leaves=num_leaves,
    )
    raises(TypeError, float_fallback.query_leaf_ids, layer_idx=0,
           query_states=np.zeros((2, 4)), num_neighbours=2, state=None)
    raises(ValueError, provider.query_leaf_ids, layer_idx=-1,
           query_states=np.zeros((2, 4)), num_neighbours=2, state=None)
    raises(TypeError, provider._per_head_selection, {0.5: []})
    raises(ValueError, provider._per_head_selection, {0: [], 2: []})

    # Concurrent queries keep a consistent last plan and count.
    def worker() -> None:
        for _ in range(20):
            provider.query_leaf_ids(
                layer_idx=0, query_states=np.zeros((2, 4)),
                num_neighbours=2, state=None,
            )

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ok(provider.plan_count == 81, "concurrent plan count")

    print(f"materialization_provider self-test: {checks} checks passed")


if __name__ == "__main__":
    _self_test()
