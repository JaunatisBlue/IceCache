"""Drain deferred RAG pages at a request boundary (stage 2B).

Decode must never encode.  :class:`~icecache.materialization_provider.
MaterializingCandidateProvider` therefore records selected-but-unmaterialized
pages through ``on_deferred``; this module processes those requests at a safe
point -- an agent pause, a tool boundary or a new turn -- and turns them into
real materializations:

.. code-block:: python

    queue = DeferredPageQueue()
    boundary = RequestBoundaryMaterializer(materializer, deferred=queue)
    provider = MaterializingCandidateProvider(
        ..., on_deferred=boundary.submit_deferred)
    # ... decode only queues ...

    report = boundary.materialize(
        session_id,
        encode=runtime.append_and_index,  # chunked prefill + DCI add
        position_base=state.n_prompt_tokens,      # current context length
        token_offset=n_sink_pages * page_size,    # absolute DCI token 0
    )

``encode`` receives the assembled append-only prompt and the page IDs that
were moved to ``encoding``; it must append those tokens to the session
(controlled chunked prefill), run ``_DCI_add``, and return
``{layer_idx: (token2node, num_leaves)}`` for the layers it indexed.  The
control plane then registers every returned layer through
:class:`~icecache.prefill_materialization.PrefillMaterialization` and marks
the pages ready.

Failure policy follows design doc §12 ("materializer 失败时 fallback 不崩溃"):
a failed encode is rolled back, re-queued for the next boundary and reported;
it does not escape as an exception.

A failure **after** ``encode`` returned is different: the append block is
already committed to the session's context.  Re-encoding it at the next
boundary would duplicate those tokens, so such pages are marked terminally
failed for this session (``skipped_doomed`` in later reports) instead of
being re-queued.  They stay unmaterialized and fall back; ``release_session``
clears the terminal mark.

``BaseException`` (cancellation, ``KeyboardInterrupt``) is treated as
*possibly committed* from the moment ``encode`` is called: a signal is
delivered at bytecode boundaries, so it can land after the encoder appended
the block but before it returned, and no caller-side flag can observe that
window.  Those pages are doomed and the exception propagates; only a
``BaseException`` raised before ``encode`` starts (in prompt assembly) is
safe to re-queue.

Only the control plane lives here.  Prompt insertion, the chunked prefill,
DCI mutations and releasing CPU/GPU KV stay with the runtime.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from icecache.materialization import (  # noqa: E402
        GlobalRAGPageStore,
        RAGMaterializer,
        chunk_document,
    )
    from icecache.materialization_provider import (  # noqa: E402
        DeferredPageQueue,
    )
    from icecache.prefill_materialization import (  # noqa: E402
        PrefillMaterialization,
    )
else:
    from .materialization import (
        GlobalRAGPageStore,
        RAGMaterializer,
        chunk_document,
    )
    from .materialization_provider import DeferredPageQueue
    from .prefill_materialization import PrefillMaterialization

__all__ = ["BoundaryReport", "RequestBoundaryMaterializer"]


@dataclass(frozen=True)
class BoundaryReport:
    """Outcome of one request-boundary materialization round.

    ``encoded`` pages are now ``cpu_kv`` (or were already ready and skipped);
    ``failed`` pages were rolled back and re-queued.  ``unknown`` pages are
    not in the global store and were dropped -- they can never materialize.
    ``skipped_doomed`` pages failed *after* their append was committed (so a
    retry would duplicate context); they are terminally failed for this
    session until ``release_session``.  ``deferred_next`` lists everything
    that must be retried at a later boundary, which includes failed, busy and
    batch-capped pages -- never doomed ones.
    """

    session_id: str
    requested: tuple[str, ...]
    encoded: tuple[str, ...]
    skipped_ready: tuple[str, ...]
    skipped_busy: tuple[str, ...]
    skipped_doomed: tuple[str, ...]
    unknown: tuple[str, ...]
    failed: tuple[str, ...]
    deferred_next: tuple[str, ...]
    errors: tuple[tuple[str, str], ...]
    layers: tuple[int, ...]
    position_base: int
    token_offset: int
    n_tokens: int = 0

    @property
    def ok(self) -> bool:
        """True when nothing failed and every requested page is accounted for."""
        return not self.failed and not self.unknown


class _SessionGate:
    """Per-session boundary mutex plus a release generation counter.

    ``busy`` serializes boundaries for the same session, but its condition
    lock is *not* held while user ``encode`` code runs.  Holding a lock across
    user code is what allows two sessions to deadlock (each encoder releasing
    the other session); the generation counter is how a boundary that was
    in flight across a ``release_session`` learns to discard its result.
    """

    __slots__ = ("cond", "busy", "generation")

    def __init__(self) -> None:
        self.cond = threading.Condition(threading.RLock())
        self.busy = False
        self.generation = 0


class RequestBoundaryMaterializer:
    """Turn :class:`DeferredPageQueue` requests into materializations.

    One instance serves every session; boundaries for the same session are
    serialized, boundaries for different sessions run independently.  Share
    the ``deferred`` queue with the provider's ``on_deferred`` hook
    (:meth:`submit_deferred` does exactly that), otherwise the provider and
    the drain would see different queues.

    The per-session condition lock is never held across user code, so an
    ``encode`` callback that releases another session cannot deadlock a
    boundary of that session.  A ``release_session`` concurrent with an
    in-flight boundary bumps the session's generation; the boundary then
    reports its pages as failed instead of registering into the released
    session.
    """

    def __init__(self, materializer, *, deferred=None) -> None:
        if not isinstance(materializer, RAGMaterializer):
            raise TypeError("materializer must be a RAGMaterializer")
        if deferred is None:
            deferred = DeferredPageQueue()
        if not isinstance(deferred, DeferredPageQueue):
            raise TypeError("deferred must be a DeferredPageQueue")
        self.materializer = materializer
        self.deferred = deferred
        self._gates: dict[str, _SessionGate] = {}
        self._locks_guard = threading.Lock()
        # Pages whose append was committed but whose registration failed:
        # retrying them would append the block twice.  Cleared by
        # release_session (a fresh context may encode them again).
        self._doomed: dict[str, set[str]] = {}
        self.stats = {
            "boundaries": 0,
            "encoded_pages": 0,
            "failed_pages": 0,
            "unknown_pages": 0,
            "doomed_pages": 0,
        }

    # -- queue seam ------------------------------------------------------
    def submit_deferred(self, session_id: str, page_ids) -> None:
        """``on_deferred`` hook: record pages for the next boundary.

        Takes the per-session lock so a submission cannot slip past
        ``release_session`` between its queue clear and its return.  A
        deferral that arrives *after* ``release_session`` returns still
        queues: the runtime must stop the session's decode loop (and its
        ``on_deferred`` calls) before releasing, as its cancel-in-flight
        contract requires.  The condition lock is not held across ``encode``,
        so a provider query made from ``encode`` on the boundary thread may
        submit without deadlocking.
        """
        _require_session_id(session_id)
        gate = self._session_gate(session_id)
        with gate.cond:
            self.deferred(session_id, page_ids)

    def pending(self, session_id: str) -> tuple[str, ...]:
        """Pages waiting for the next boundary (does not drain)."""
        return self.deferred.pending(session_id)

    def doomed_pages(self, session_id: str) -> tuple[str, ...]:
        """Pages whose append committed but whose registration failed.

        They are terminally failed for this session: re-encoding would append
        the block twice.  ``release_session`` clears the mark.
        """
        _require_session_id(session_id)
        return tuple(sorted(self._doomed_pages(session_id)))

    # -- boundary --------------------------------------------------------
    def materialize(
        self,
        session_id: str,
        *,
        encode: Callable[[object, tuple[str, ...]], Mapping],
        position_base,
        token_offset,
        page_ids: Sequence[str] | None = None,
        require_full_span: bool = True,
        max_pages: int | None = None,
    ) -> BoundaryReport:
        """Materialize drained (or explicitly selected) RAG pages now.

        ``encode(prompt, page_ids)`` must append ``prompt.token_ids`` to the
        session at ``position_base`` (position ids and attention mask equal to
        putting the chunks in context normally), run the DCI add, and return
        ``{layer_idx: (token2node, num_leaves)}``.  Returning ``{}`` is
        allowed: pages then finish as ready but unmapped, and join retrieval
        once a later ``token2node`` registers their leaves.

        **``encode`` must be atomic on failure.**  If it raises, the session's
        KV must be left as it was (discard a partially appended block): the
        pages are rolled back and re-queued, and the next boundary appends
        them again -- retrying after a committed append would duplicate the
        chunk in the context.

        ``page_ids=None`` drains the deferred queue atomically; an explicit
        list is useful for a caller that wants a different policy.  At most
        ``max_pages`` pages are encoded in this round; the rest stay queued.

        Never raises for an encode/registration failure.  A failure before the
        append commits rolls the pages back and re-queues them; a failure
        after it commits marks them doomed and reports ``n_tokens`` as the
        committed block length, because the session context grew even though
        the pages are unusable.  The caller must advance its context length by
        that amount.  ``BaseException`` is treated as *possibly committed*
        from the moment ``encode`` is called: a signal can land after the
        append but before ``encode`` returns, so those pages are doomed (never
        re-queued) and the exception propagates; the caller knows the block
        length from its own encoder and must account for it there.
        """
        _require_session_id(session_id)
        if not callable(encode):
            raise TypeError("encode must be callable")
        position_base = _require_count(position_base, "position_base")
        token_offset = _require_count(token_offset, "token_offset")
        if not isinstance(require_full_span, bool):
            raise TypeError("require_full_span must be a bool")
        if max_pages is not None:
            max_pages = _require_positive(max_pages, "max_pages")

        gate = self._session_gate(session_id)
        with _boundary_slot(gate) as generation:
            if page_ids is None:
                requested = list(self.deferred.drain(session_id))
            else:
                requested = _require_page_ids(page_ids)
            requested = _dedupe(requested)

            active, skipped_ready, skipped_busy, unknown = [], [], [], []
            skipped_doomed: list[str] = []
            doomed = self._doomed_pages(session_id)
            for page_id in requested:
                if page_id in doomed:
                    skipped_doomed.append(page_id)
                    continue
                if page_id not in self.materializer.pages:
                    unknown.append(page_id)
                    continue
                try:
                    record = self.materializer.record(session_id, page_id)
                except KeyError:
                    record = None
                if record is not None and record.is_ready:
                    skipped_ready.append(page_id)
                elif record is not None and record.state == "encoding":
                    skipped_busy.append(page_id)
                else:
                    active.append(page_id)

            overflow: list[str] = []
            if max_pages is not None and len(active) > max_pages:
                overflow = active[max_pages:]
                active = active[:max_pages]

            errors: list[tuple[str, str]] = [
                (page_id, "not a global RAG page") for page_id in unknown
            ]
            with self._locks_guard:
                self.stats["boundaries"] += 1
                self.stats["unknown_pages"] += len(unknown)
            if skipped_busy or overflow:
                self.deferred(session_id, skipped_busy + overflow)

            if not active:
                return BoundaryReport(
                    session_id=session_id,
                    requested=tuple(requested),
                    encoded=(),
                    skipped_ready=tuple(skipped_ready),
                    skipped_busy=tuple(skipped_busy),
                    skipped_doomed=tuple(skipped_doomed),
                    unknown=tuple(unknown),
                    failed=(),
                    deferred_next=tuple(skipped_busy + overflow),
                    errors=tuple(errors),
                    layers=(),
                    position_base=position_base,
                    token_offset=token_offset,
                )

            pages = [self.materializer.pages.get(page_id) for page_id in active]
            request = PrefillMaterialization(self.materializer, session_id)
            committed = False
            encode_started = False
            try:
                prompt = request.prepare_append(
                    pages=pages, position_base=position_base
                )
                if self._released(gate, generation):
                    # release_session ran while this boundary was assembling.
                    # Do not append into a context that no longer exists, and
                    # do not re-queue: the queue was cleared for good.
                    raise _SessionReleased("session released mid-boundary")
                begun = tuple(request.begun_page_ids)
                encode_started = True
                layer_maps = encode(prompt, begun)
                committed = True
                layers = _register_layers(
                    request,
                    layer_maps,
                    token_offset=token_offset,
                    require_full_span=require_full_span,
                )
                request.finish()
            except Exception as exc:  # noqa: BLE001 - fallback must not crash
                released = isinstance(exc, _SessionReleased) or self._released(
                    gate, generation
                )
                if released:
                    # The session was torn down under this boundary; neither
                    # dooming nor re-queuing is meaningful (the queue and the
                    # records are gone, and a re-queue would resurrect them).
                    errors.extend(
                        (page_id, f"{exc} (session released mid-boundary)")
                        for page_id in active
                    )
                elif committed:
                    # The append is already in the context; re-encoding would
                    # duplicate it.  Fail closed for this session instead.
                    self._doom(session_id, active)
                    errors.extend(
                        (page_id,
                         f"{exc} (append committed; not retryable in this "
                         "session)")
                        for page_id in active
                    )
                else:
                    errors.extend((page_id, str(exc)) for page_id in active)
                    self.deferred(session_id, active)
                _roll_back(request, active, exc)
                with self._locks_guard:
                    self.stats["failed_pages"] += len(active)
                return BoundaryReport(
                    session_id=session_id,
                    requested=tuple(requested),
                    encoded=(),
                    skipped_ready=tuple(skipped_ready),
                    skipped_busy=tuple(skipped_busy),
                    skipped_doomed=tuple(skipped_doomed),
                    unknown=tuple(unknown),
                    failed=tuple(active),
                    deferred_next=tuple(
                        skipped_busy
                        + overflow
                        + ([] if committed or released else active)
                    ),
                    errors=tuple(errors),
                    layers=(),
                    position_base=position_base,
                    token_offset=token_offset,
                    n_tokens=(
                        0 if released
                        else int(prompt.n_tokens) if committed
                        else 0
                    ),
                )
            except BaseException as exc:
                # Cancellation/interrupt: propagate instead of swallowing.
                # Unlike an ordinary exception (where the atomic-encode
                # contract lets us re-queue), a signal can be delivered after
                # ``encode`` appended the block but before its ``return`` --
                # a window no flag set by the caller can close.  Treat any
                # interrupt at or after the call as possibly committed, so the
                # block is never appended twice.  Dooming drops the pages for
                # this session; the caller accounts for the committed length
                # from its own encoder.  A released session is neither doomed
                # nor re-queued: its bookkeeping is already gone.
                if not self._released(gate, generation):
                    if committed or encode_started:
                        self._doom(session_id, active)
                    else:
                        self.deferred(session_id, active)
                _roll_back(request, active, exc)
                with self._locks_guard:
                    self.stats["failed_pages"] += len(active)
                raise

            with self._locks_guard:
                self.stats["encoded_pages"] += len(begun)
            return BoundaryReport(
                session_id=session_id,
                requested=tuple(requested),
                encoded=begun,
                skipped_ready=tuple(skipped_ready),
                skipped_busy=tuple(skipped_busy),
                skipped_doomed=tuple(skipped_doomed),
                unknown=tuple(unknown),
                failed=(),
                deferred_next=tuple(skipped_busy + overflow),
                errors=tuple(errors),
                layers=layers,
                position_base=position_base,
                token_offset=token_offset,
                n_tokens=int(prompt.n_tokens),
            )

    def release_session(self, session_id: str) -> int:
        """Drop this session's queue and materialization records.

        Control-plane metadata only; the runtime still releases the actual
        CPU/GPU KV pages and its DCI delta (design doc §6.4).

        The per-session gate stays registered: another thread may still refer
        to it, and handing out a fresh gate for the same session would let two
        boundaries run concurrently after a release.

        This does *not* wait for an in-flight boundary.  Waiting would
        deadlock whenever two encoders release each other's sessions; instead
        the generation bump tells the in-flight boundary to discard its result
        (it reports its pages failed and registers nothing).

        The queue clear runs under the same lock as ``submit_deferred``, so a
        deferral cannot interleave with the teardown.  A deferral that arrives
        *after* this returns still queues the ID: the runtime must stop the
        session's decode loop (and its ``on_deferred`` calls) before calling
        this, as its cancel-in-flight contract requires.
        """
        _require_session_id(session_id)
        gate = self._session_gate(session_id)
        with gate.cond:
            # Bump first: a boundary that finishes after any part of the
            # teardown below must observe the release and not register into
            # the session it just released.
            gate.generation += 1
            with self._locks_guard:
                self._doomed.pop(session_id, None)
            self.deferred.clear(session_id)
            return self.materializer.release_session(session_id)

    # -- internals -------------------------------------------------------
    def _doom(self, session_id: str, page_ids) -> None:
        """Permanently fail pages whose append was already committed."""
        with self._locks_guard:
            doomed = self._doomed.setdefault(session_id, set())
            for page_id in page_ids:
                if page_id not in doomed:
                    doomed.add(page_id)
                    self.stats["doomed_pages"] += 1

    def _doomed_pages(self, session_id: str) -> frozenset[str]:
        with self._locks_guard:
            return frozenset(self._doomed.get(session_id, ()))

    def _session_gate(self, session_id: str) -> _SessionGate:
        with self._locks_guard:
            gate = self._gates.get(session_id)
            if gate is None:
                gate = _SessionGate()
                self._gates[session_id] = gate
            return gate

    @staticmethod
    def _released(gate: _SessionGate, generation: int) -> bool:
        """True when ``release_session`` ran since ``generation`` was taken."""
        with gate.cond:
            return gate.generation != generation


class _SessionReleased(RuntimeError):
    """Internal: ``release_session`` raced this boundary's assembly."""


@contextlib.contextmanager
def _boundary_slot(gate: _SessionGate):
    """Serialize boundaries for one session without holding a lock in user code.

    Waits for the previous boundary to finish, marks the session busy, and
    yields the generation to check afterwards.  The condition lock is released
    before the caller runs ``encode``, so two encoders releasing each other's
    sessions cannot form an ABBA cycle.
    """
    with gate.cond:
        while gate.busy:
            gate.cond.wait()
        gate.busy = True
        generation = gate.generation
    try:
        yield generation
    finally:
        with gate.cond:
            gate.busy = False
            gate.cond.notify_all()


def _register_layers(
    request: PrefillMaterialization,
    layer_maps,
    *,
    token_offset: int,
    require_full_span: bool,
) -> tuple[int, ...]:
    if layer_maps is None:
        layer_maps = {}
    if not isinstance(layer_maps, Mapping):
        raise TypeError(
            "encode must return a mapping {layer_idx: (token2node, num_leaves)}"
        )
    normalised: list[tuple[int, object, object]] = []
    for key, value in layer_maps.items():
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
            raise TypeError(f"layer key {key!r} is not an integer")
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError(
                f"layer {key!r} must map to (token2node, num_leaves), "
                f"got {type(value).__name__}"
            )
        normalised.append((int(key), value[0], value[1]))
    layers: list[int] = []
    for layer_idx, token2node, num_leaves in sorted(normalised):
        request.register_layer(
            layer_idx,
            token2node,
            num_leaves,
            token_offset=token_offset,
            require_full_span=require_full_span,
        )
        layers.append(layer_idx)
    return tuple(layers)


def _roll_back(request: PrefillMaterialization, page_ids, exc: BaseException) -> None:
    reason = f"boundary materialization failed: {type(exc).__name__}: {exc}"
    try:
        request.abort(reason)
        return
    except RuntimeError:
        # Already finished or aborted; fall through to an explicit fail so a
        # stranding page cannot hide behind the request state.
        pass
    for page_id in page_ids:
        try:
            request.materializer.fail(request.session_id, page_id, reason)
        except KeyError:
            pass


def _require_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")


def _require_count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _require_positive(value, name: str) -> int:
    value = _require_count(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_page_ids(page_ids) -> list[str]:
    if isinstance(page_ids, (str, bytes)):
        raise TypeError("page_ids must be a sequence, not a string")
    ids = list(page_ids)
    if any(not isinstance(page_id, str) or not page_id for page_id in ids):
        raise ValueError("page_ids must be non-empty strings")
    return ids


def _dedupe(ids: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for page_id in ids:
        seen.setdefault(page_id, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    checks = 0

    def ok(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    def build(n_pages: int = 6, chunk_size: int = 8):
        tokens = np.arange(n_pages * chunk_size, dtype=np.int64)
        pages = chunk_document(
            tokens,
            document_id="doc",
            chunk_size=chunk_size,
            descriptor_version="v1",
            descriptor_fn=lambda chunk: np.ones(2, dtype=np.float32),
        )
        store = GlobalRAGPageStore()
        store.add_many(pages)
        materializer = RAGMaterializer(store, n_kv_heads=2)
        return pages, materializer

    def token_map(n_tokens: int, num_leaves: int = 64):
        # 4 tokens per leaf, distinct per index so a wrong offset shows up.
        nodes = np.array(
            [[(i // 4) % num_leaves for i in range(n_tokens)],
             [(i // 4) % num_leaves for i in range(n_tokens)]],
            dtype=np.int32,
        )
        return nodes, np.array([num_leaves, num_leaves], dtype=np.int32)

    # -- success path: append at an absolute position --------------------
    pages, materializer = build()
    queue = DeferredPageQueue()
    boundary = RequestBoundaryMaterializer(materializer, deferred=queue)
    selected = (pages[0].page_id, pages[1].page_id)
    boundary.submit_deferred("s1", selected)

    seen = {}

    def encode_ok(prompt, page_ids):
        seen["prompt"] = prompt
        seen["page_ids"] = page_ids
        n_tokens = 16 + int(prompt.n_tokens)  # absolute session positions
        return {0: token_map(n_tokens), 2: token_map(n_tokens)}

    report = boundary.materialize(
        "s1", encode=encode_ok, position_base=16, token_offset=0
    )
    ok(report.ok, f"success report not clean: {report}")
    ok(report.encoded == selected, report.encoded)
    ok(report.layers == (0, 2), report.layers)
    ok(report.n_tokens == 16, report.n_tokens)
    ok(queue.pending("s1") == (), queue.pending("s1"))
    ok(seen["page_ids"] == tuple(selected), seen["page_ids"])
    ok(seen["prompt"].n_tokens == 16, seen["prompt"].n_tokens)
    # Recorded spans are absolute session positions, not prompt-local.
    rec = materializer.record("s1", pages[0].page_id)
    ok(rec.state == "cpu_kv", rec.state)
    ok((rec.token_start, rec.token_end) == (16, 24), (rec.token_start, rec.token_end))
    rec1 = materializer.record("s1", pages[1].page_id)
    ok((rec1.token_start, rec1.token_end) == (24, 32), (rec1.token_start, rec1.token_end))
    # Leaf maps contain the DCI tokens of the appended span, layer 2 included.
    nodes, _limits = token_map(32)
    expected0 = np.unique(nodes[0, 16:24]).tolist()
    ok(rec.layer_leaf_ids[0][0].tolist() == expected0, "leaf map page0")
    expected1 = np.unique(nodes[0, 24:32]).tolist()
    ok(rec1.layer_leaf_ids[2][0].tolist() == expected1, "leaf map page1")

    # -- session isolation: the same global pages stay private -----------
    ok(not materializer.page_is_ready("s2", pages[0].page_id), "leaked ready")
    try:
        materializer.record("s2", pages[0].page_id)
    except KeyError:
        ok(True, "s2 has no record for s1's page")
    else:
        ok(False, "s2 inherited s1's materialization record")

    # -- ready pages are skipped, unknown pages dropped ------------------
    report = boundary.materialize(
        "s1",
        encode=encode_ok,
        position_base=32,
        token_offset=0,
        page_ids=(pages[0].page_id, "missing/page"),
    )
    ok(report.skipped_ready == (pages[0].page_id,), report.skipped_ready)
    ok(report.unknown == ("missing/page",), report.unknown)
    ok(report.encoded == (), report.encoded)
    ok(not report.ok, "unknown page should make the report not ok")
    ok(("missing/page", "not a global RAG page") in report.errors, report.errors)

    # -- encode failure rolls back, re-queues and never raises -----------
    boundary.submit_deferred("s3", (pages[2].page_id,))

    def encode_fail(prompt, page_ids):
        raise RuntimeError("prefill exploded")

    report = boundary.materialize(
        "s3", encode=encode_fail, position_base=0, token_offset=0
    )
    ok(report.failed == (pages[2].page_id,), report.failed)
    ok(queue.pending("s3") == (pages[2].page_id,), queue.pending("s3"))
    rec = materializer.record("s3", pages[2].page_id)
    ok(rec.state == "failed", rec.state)
    ok(rec.error is not None and "prefill exploded" in rec.error, rec.error)
    ok(any("prefill exploded" in message for _, message in report.errors), report.errors)

    # Retry at the next boundary succeeds from the failed state.
    report = boundary.materialize(
        "s3", encode=encode_ok, position_base=0, token_offset=0
    )
    ok(report.encoded == (pages[2].page_id,), report.encoded)
    ok(materializer.record("s3", pages[2].page_id).state == "cpu_kv", "not ready")

    # -- require_full_span failure is reported, not raised ---------------
    boundary.submit_deferred("s4", (pages[3].page_id,))

    def encode_short(prompt, page_ids):
        # Index only the first token: the page is not fully present.
        return {0: token_map(1)}

    report = boundary.materialize(
        "s4", encode=encode_short, position_base=0, token_offset=0
    )
    ok(report.failed == (pages[3].page_id,), report.failed)
    # encode returned: the append is committed, so a retry would duplicate it.
    # The context grew by the block even though the page failed.
    ok(report.n_tokens == pages[3].n_tokens, report.n_tokens)
    ok(report.deferred_next == (), report.deferred_next)
    ok(queue.pending("s4") == (), queue.pending("s4"))
    ok(any("append committed" in message for _, message in report.errors),
       report.errors)
    ok(materializer.record("s4", pages[3].page_id).state == "failed", "not failed")
    ok(boundary.stats["doomed_pages"] == 1, boundary.stats)

    # A re-deferral of a doomed page is dropped without calling encode again.
    called = []

    def encode_must_not_run(prompt, page_ids):
        called.append(page_ids)
        return {0: token_map(1)}

    boundary.submit_deferred("s4", (pages[3].page_id,))
    report = boundary.materialize(
        "s4", encode=encode_must_not_run, position_base=0, token_offset=0
    )
    ok(report.skipped_doomed == (pages[3].page_id,), report.skipped_doomed)
    ok(not called, "doomed page was encoded again")
    ok(report.failed == () and report.ok, report)
    ok(queue.pending("s4") == (), queue.pending("s4"))

    # The clipped mapping is explicit and works when chosen up front.
    boundary.submit_deferred("s4b", (pages[3].page_id,))
    report = boundary.materialize(
        "s4b",
        encode=encode_short,
        position_base=0,
        token_offset=0,
        require_full_span=False,
    )
    ok(report.ok, report)
    ok(report.encoded == (pages[3].page_id,), report.encoded)

    # release_session clears the terminal mark (a fresh context may retry).
    boundary.release_session("s4")
    ok(boundary.doomed_pages("s4") == (), "doomed mark survived")
    boundary.submit_deferred("s4", (pages[3].page_id,))
    report = boundary.materialize(
        "s4", encode=encode_ok, position_base=0, token_offset=0
    )
    ok(report.encoded == (pages[3].page_id,), report.encoded)

    # -- max_pages caps one round; the rest stay queued ------------------
    boundary.submit_deferred("s5", (pages[0].page_id, pages[1].page_id))
    report = boundary.materialize(
        "s5", encode=encode_ok, position_base=0, token_offset=0, max_pages=1
    )
    ok(report.encoded == (pages[0].page_id,), report.encoded)
    ok(report.deferred_next == (pages[1].page_id,), report.deferred_next)
    ok(queue.pending("s5") == (pages[1].page_id,), queue.pending("s5"))
    report = boundary.materialize(
        "s5", encode=encode_ok, position_base=8, token_offset=0
    )
    ok(report.encoded == (pages[1].page_id,), report.encoded)

    # -- a page another boundary is encoding is skipped and re-queued ----
    materializer.submit("s6", [pages[4].page_id])
    materializer.begin_encoding(
        "s6", [pages[4].page_id], token_start=0, token_end=pages[4].n_tokens
    )
    report = boundary.materialize(
        "s6",
        encode=encode_ok,
        position_base=0,
        token_offset=0,
        page_ids=(pages[4].page_id,),
    )
    ok(report.skipped_busy == (pages[4].page_id,), report.skipped_busy)
    ok(queue.pending("s6") == (pages[4].page_id,), queue.pending("s6"))
    ok(materializer.record("s6", pages[4].page_id).state == "encoding", "clobbered")
    materializer.fail("s6", pages[4].page_id, "cleanup")

    # -- BaseException during encode dooms (possibly committed) ----------
    # A signal delivered after encode appended the block cannot be told
    # apart from one delivered before, so the pages are terminally failed
    # for this session instead of re-queued.
    boundary.submit_deferred("s7", (pages[5].page_id,))

    def encode_interrupt(prompt, page_ids):
        raise KeyboardInterrupt

    try:
        boundary.materialize(
            "s7", encode=encode_interrupt, position_base=0, token_offset=0
        )
    except KeyboardInterrupt:
        ok(True, "KeyboardInterrupt propagated")
    else:
        ok(False, "KeyboardInterrupt was swallowed")
    ok(queue.pending("s7") == (), queue.pending("s7"))
    ok(boundary.doomed_pages("s7") == (pages[5].page_id,),
       boundary.doomed_pages("s7"))
    ok(materializer.record("s7", pages[5].page_id).state == "failed", "not failed")

    # An interrupt *after* encode returned (registration phase) must not
    # re-queue the page: the append is already committed.
    boundary.submit_deferred("s7b", (pages[5].page_id,))

    class InterruptingMap(dict):
        def items(self):
            raise KeyboardInterrupt

    def encode_then_interrupt(prompt, page_ids):
        return InterruptingMap()

    try:
        boundary.materialize(
            "s7b", encode=encode_then_interrupt, position_base=0, token_offset=0
        )
    except KeyboardInterrupt:
        ok(True, "post-commit KeyboardInterrupt propagated")
    else:
        ok(False, "post-commit KeyboardInterrupt was swallowed")
    ok(queue.pending("s7b") == (), queue.pending("s7b"))
    ok(boundary.doomed_pages("s7b") == (pages[5].page_id,),
       boundary.doomed_pages("s7b"))

    # -- releasing another session from encode must not deadlock ---------
    # d1's encoder releases d2 while d2 is inside its own encode.  With a
    # per-session lock held across ``encode`` this is an ABBA deadlock; the
    # gate is not held across user code, so d1 returns, unblocks d2, and d2
    # observes the generation bump instead of registering into a released
    # session.
    boundary.submit_deferred("d1", (pages[0].page_id,))
    boundary.submit_deferred("d2", (pages[1].page_id,))
    both_encoding = threading.Barrier(2, timeout=10.0)
    d2_go = threading.Event()
    cross: dict[str, object] = {}

    def encode_d1(prompt, page_ids):
        both_encoding.wait()
        ok(boundary.release_session("d2") == 1, "d1 released d2's record")
        d2_go.set()
        return {}

    def encode_d2(prompt, page_ids):
        both_encoding.wait()
        ok(d2_go.wait(10.0), "d2 unblocked")
        return {}

    def run_cross(session, encode):
        try:
            cross[session] = boundary.materialize(
                session, encode=encode, position_base=0, token_offset=0
            )
        except BaseException as exc:  # noqa: BLE001 - the deadlock is the point
            cross[session] = exc

    threads = [
        threading.Thread(target=run_cross, args=("d1", encode_d1)),
        threading.Thread(target=run_cross, args=("d2", encode_d2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    ok(not any(thread.is_alive() for thread in threads),
       "cross-session boundaries did not deadlock")
    ok(all(not isinstance(report, BaseException) for report in cross.values()),
       cross)
    ok(cross["d1"].encoded == (pages[0].page_id,), cross["d1"])
    ok(cross["d2"].failed == (pages[1].page_id,), cross["d2"])
    ok(cross["d2"].deferred_next == (), cross["d2"].deferred_next)

    # -- release_session during encode is observed, not registered into --
    boundary.submit_deferred("r1", (pages[2].page_id,))
    entered = threading.Event()
    proceed = threading.Event()
    released_report: dict[str, object] = {}

    def encode_wait(prompt, page_ids):
        entered.set()
        ok(proceed.wait(10.0), "release wait timed out")
        return {}

    def run_released():
        released_report["r"] = boundary.materialize(
            "r1", encode=encode_wait, position_base=0, token_offset=0
        )

    worker = threading.Thread(target=run_released)
    worker.start()
    ok(entered.wait(10.0), "encode did not start")
    ok(boundary.release_session("r1") == 1,
       "release removed the encoding record")
    proceed.set()
    worker.join(timeout=10.0)
    ok(not worker.is_alive(), "released boundary returned")
    report = released_report["r"]
    ok(report.failed == (pages[2].page_id,), report)
    ok(report.deferred_next == (), report.deferred_next)
    ok(report.n_tokens == 0, report.n_tokens)
    ok(not materializer.page_is_ready("r1", pages[2].page_id),
       "released boundary registered nothing")
    ok(queue.pending("r1") == (), queue.pending("r1"))

    # -- argument validation ---------------------------------------------
    def encodes_nothing(prompt, page_ids):
        return {}

    for kwargs, exc_type in (
        ({"position_base": -1, "token_offset": 0}, ValueError),
        ({"position_base": True, "token_offset": 0}, TypeError),
        ({"position_base": 0, "token_offset": -2}, ValueError),
        ({"position_base": 0, "token_offset": 0, "max_pages": 0}, ValueError),
        ({"position_base": 0, "token_offset": 0, "require_full_span": 1}, TypeError),
    ):
        try:
            boundary.materialize("s8", encode=encodes_nothing, **kwargs)
        except exc_type:
            ok(True, f"accepted {kwargs}")
        else:
            ok(False, f"accepted invalid {kwargs}")

    # -- release_session clears queue and records ------------------------
    ok(
        boundary.release_session("s1") >= 2,
        "release_session did not report records",
    )
    ok(queue.pending("s1") == (), "queue survived release_session")
    ok(not materializer.page_is_ready("s1", pages[0].page_id), "record survived")
    report = boundary.materialize(
        "s1", encode=encodes_nothing, position_base=0, token_offset=0
    )
    ok(report.requested == (), report.requested)

    # -- empty mapping: ready but unmapped, never deferred --------------
    boundary.submit_deferred("s9", (pages[0].page_id,))
    report = boundary.materialize(
        "s9", encode=encodes_nothing, position_base=0, token_offset=0
    )
    ok(report.ok and report.encoded == (pages[0].page_id,), report)
    ok(materializer.page_is_ready("s9", pages[0].page_id), "not ready")
    empty = materializer.resolve_head_leaves("s9", pages[0].page_id, 0, 0)
    ok(empty is not None and empty.size == 0, "unmapped page resolved leaves")

    print(f"boundary_materialization self-test: {checks} checks passed")


if __name__ == "__main__":
    _self_test()
