"""Request-boundary coordination for RAG page materialization (stage 2).

The control plane in :mod:`icecache.materialization` is deliberately unaware of
the runtime.  This module is the narrow piece that sits at the request
boundary, before decode:

1. retrieve global pages for the request (``prepare_from_query``) or accept
   a caller-selected list,
2. assemble them into the prompt (text side),
3. record each page's absolute token span and start encoding,
4. let the runtime prefill that prompt normally (no extra encoding during
   decode),
5. register the prefill's per-layer ``token2node`` arrays, which yields the
   per-KV-head native leaf IDs, and
6. complete (or abort) the pages.

The runtime still owns prompt insertion, the prefill itself and DCI
``_DCI_add``.  This helper only keeps the session materialization table in
step with what the prompt actually contains.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from icecache.materialization import (  # noqa: E402
        AssembledPrompt,
        RAGMaterializer,
        assemble_prompt,
    )
else:
    from .materialization import AssembledPrompt, RAGMaterializer, assemble_prompt

__all__ = ["PrefillMaterialization"]


class PrefillMaterialization:
    """One request's prompt-assembly + leaf-registration lifecycle.

    Typical use::

        request = PrefillMaterialization(materializer, session_id)
        prompt = request.prepare(pages=selected_pages,
                                 prefix_token_ids=prompt_ids,
                                 suffix_token_ids=suffix_ids)
        # ... run the normal prefill on prompt.token_ids ...
        # token_offset is the absolute prompt position of DCI token 0, NOT the
        # first RAG page's position.  IceCache's first prefill index starts at
        # page n_sink_pages, so it is n_sink_pages * page_size (verify against
        # the layer's dci_len).  Passing a page start instead silently maps the
        # wrong prompt tokens whenever the page is not the first indexed token.
        for layer_idx, (token2node, num_leaves) in indexed_layers.items():
            request.register_layer(layer_idx, token2node, num_leaves,
                                   token_offset=n_sink_pages * page_size)
        request.finish()

    One request gets one session (design doc §7).  Deep KV is only valid for
    the context that produced it, so a page that is already CPU/GPU-resident
    in this session is rejected rather than silently reused; start a new
    session to re-encode it.

    ``register_layer`` rolls the whole request back on error (``abort``): a
    partially registered layer would otherwise leave some pages resolvable and
    others not for the same prompt.  Failures are fail-closed to the prompt
    fallback, not propagated into the native leaf path.
    """

    def __init__(self, materializer, session_id: str) -> None:
        if not isinstance(materializer, RAGMaterializer):
            raise TypeError("materializer must be a RAGMaterializer")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        self.materializer = materializer
        self.session_id = session_id
        self._prompt: AssembledPrompt | None = None
        self._begun: list[str] = []
        self._retrieval_hits: tuple[tuple[str, float], ...] = ()
        self._finished = False
        self._aborted = False
        self._abort_reason: str | None = None
        # prepare/register_layer/finish/abort may be called from different
        # threads (the prefill worker vs. a cancellation path).  Serializing
        # them prevents abort from observing an empty ``_begun`` just before
        # prepare starts pages, which would strand them in ``encoding``.
        self._lock = threading.RLock()

    # -- state -----------------------------------------------------------
    @property
    def prompt(self) -> AssembledPrompt | None:
        with self._lock:
            return self._prompt

    @property
    def begun_page_ids(self) -> tuple[str, ...]:
        """Pages this request moved to ``encoding`` (excludes reused pages)."""
        with self._lock:
            return tuple(self._begun)

    @property
    def page_ids(self) -> tuple[str, ...]:
        """Every RAG page in the assembled prompt."""
        with self._lock:
            if self._prompt is None:
                return ()
            return tuple(span.page_id for span in self._prompt.page_spans)

    @property
    def retrieval_hits(self) -> tuple[tuple[str, float], ...]:
        """``(page_id, cosine score)`` pairs this request assembled.

        Empty unless ``prepare_from_query`` was used (or every hit was
        filtered by ``min_score``).
        """
        with self._lock:
            return self._retrieval_hits

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "PrefillMaterialization":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Abort a request that leaves the block without ``finish()``.

        ``prepare``/``register_layer``/``finish`` must all happen inside the
        block.  An exception is never swallowed; aborting here only ensures the
        session does not keep pages stuck in ``encoding``.
        """
        with self._lock:
            if not self._finished and not self._aborted:
                reason = (
                    f"request failed: {exc_type.__name__}"
                    if exc_type is not None
                    else "request exited without finish"
                )
                try:
                    self.abort(reason)
                except Exception:
                    if exc_type is None:
                        raise
        return False

    def prepare(
        self,
        *,
        pages: Sequence[object],
        prefix_token_ids,
        suffix_token_ids=None,
        page_ids: Sequence[str] | None = None,
    ) -> AssembledPrompt:
        """Assemble the prompt and start encoding its RAG pages.

        ``pages`` are the chunks selected by retrieval (global page objects).
        The returned ``AssembledPrompt.token_ids`` is what the runtime must
        prefill.  Calling ``prepare`` twice on one helper is an error; use a
        fresh helper per request.
        """
        with self._lock:
            if self._aborted:
                raise RuntimeError(f"request was aborted: {self._abort_reason}")
            self._require_unprepared()
            prompt = assemble_prompt(
                pages=pages,
                prefix_token_ids=prefix_token_ids,
                suffix_token_ids=suffix_token_ids,
            )
            ids = (
                [span.page_id for span in prompt.page_spans]
                if page_ids is None
                else list(page_ids)
            )
            return self._start(prompt, ids, position_base=0)

    def prepare_append(
        self,
        *,
        pages: Sequence[object],
        position_base: int,
    ) -> AssembledPrompt:
        """Assemble an append-only block at an absolute session position.

        Stage-B request-boundary path: at an agent pause or tool boundary the
        newly retrieved chunks are appended to the *current* context instead
        of rebuilding the prompt.  ``prompt.token_ids`` contains only the
        appended tokens (prefill exactly these, not the whole context); the
        recorded page spans are absolute session positions
        ``position_base + [start, end)``.

        ``position_base`` is the current context length, i.e. the absolute
        position of ``prompt.token_ids[0]``.  Pass the same ``token_offset``
        to :meth:`register_layer` as for the session's original prefill: DCI
        token 0 does not move when tokens are appended.

        The runtime still owns the chunked prefill and must make the appended
        positions, attention mask and KV page sequence equivalent to putting
        the chunks into the context normally (design doc §7, stage B).  This
        helper only keeps the materialization table in step with it.
        """
        with self._lock:
            if self._aborted:
                raise RuntimeError(f"request was aborted: {self._abort_reason}")
            self._require_unprepared()
            if isinstance(position_base, bool) or not isinstance(
                position_base, (int, np.integer)
            ):
                raise TypeError("position_base must be an integer")
            if int(position_base) < 0:
                raise ValueError("position_base must be non-negative")
            prompt = assemble_prompt(pages=pages)
            ids = [span.page_id for span in prompt.page_spans]
            return self._start(prompt, ids, position_base=int(position_base))

    def _start(
        self,
        prompt: AssembledPrompt,
        ids: Sequence[str],
        *,
        position_base: int,
    ) -> AssembledPrompt:
        """Record absolute spans and move the pages to ``encoding``."""
        # begin_prompt rolls back every page it started (newly submitted
        # or pre-queued) on its own failures, so nothing is left
        # queued/encoding.  The remaining window is a signal delivered at (or
        # after) the call's return, when the pages are already ``encoding``
        # but the return value cannot be bound.  ``begun_out`` is filled
        # inside begin_prompt, before it returns, so the cleanup below sees
        # them; ``fail`` is idempotent, so pages begin_prompt already rolled
        # back are harmless here.
        begun: list[str] = []
        try:
            begun_ids = self.materializer.begin_prompt(
                self.session_id,
                prompt,
                ids,
                position_base=position_base,
                begun_out=begun,
            )
            self._prompt = prompt
            self._begun = list(begun_ids)
        except Exception as exc:
            # begin_prompt rolled back its own state changes, so the helper
            # stays unprepared and retryable (an ordinary validation error is
            # not a cancellation).  ``begun`` is normally empty; fail
            # anything in it so a subclass that raises after starting pages
            # cannot strand them in ``encoding``.
            self._begun = []
            for page_id in begun:
                try:
                    self.materializer.fail(
                        self.session_id, page_id, f"begin_prompt failed: {exc!r}"
                    )
                except KeyError:
                    pass
            raise
        except BaseException as exc:
            self._begun = []
            self._prompt = None
            self._aborted = True
            self._abort_reason = f"interrupted in begin_prompt: {exc!r}"
            for page_id in begun:
                try:
                    self.materializer.fail(
                        self.session_id, page_id, self._abort_reason
                    )
                except KeyError:
                    pass
            raise
        if self._aborted:
            # A cancellation that arrived while begin_prompt was running
            # (e.g. from a callback on this thread) already returned an
            # empty scan; fail the pages it just started instead of
            # leaving them in ``encoding``.
            for page_id in self._begun:
                try:
                    self.materializer.fail(
                        self.session_id, page_id, self._abort_reason or "aborted"
                    )
                except KeyError:
                    pass
            self._begun = []
            raise RuntimeError(f"request was aborted: {self._abort_reason}")
        return prompt

    def prepare_from_query(
        self,
        *,
        query_descriptor,
        top_k: int,
        prefix_token_ids,
        suffix_token_ids=None,
        min_score: float | None = None,
        allowed_page_ids: Sequence[str] | None = None,
    ) -> AssembledPrompt:
        """Retrieve global RAG pages by text descriptor, then prepare them.

        Runs :meth:`GlobalRAGPageStore.search` over the materializer's global
        page store and feeds the hits to :meth:`prepare` -- the stage-2
        "text retrieval -> prompt assembly" half of the request boundary.
        The runtime still owns prompt insertion, prefill and ``_DCI_add``.

        The search happens before any state transition, so a bad descriptor,
        an unknown ``allowed_page_ids`` entry or a corpus/version error leaves
        the request untouched and retryable.  ``min_score`` drops hits below a
        cosine score after retrieval (``top_k`` still bounds the candidate
        list).  The assembled ``(page_id, score)`` pairs are exposed through
        :attr:`retrieval_hits`.
        """
        if min_score is not None:
            if isinstance(min_score, bool) or not isinstance(
                min_score, (int, float, np.integer, np.floating)
            ):
                raise TypeError("min_score must be a real number")
            if not np.isfinite(min_score):
                raise ValueError("min_score must be finite")
            min_score = float(min_score)
        hits = self.materializer.pages.search(
            query_descriptor,
            top_k=top_k,
            allowed_page_ids=allowed_page_ids,
        )
        if min_score is not None:
            hits = [
                (page, score) for page, score in hits if score >= min_score
            ]
        prompt = self.prepare(
            pages=[page for page, _ in hits],
            prefix_token_ids=prefix_token_ids,
            suffix_token_ids=suffix_token_ids,
        )
        with self._lock:
            self._retrieval_hits = tuple(
                (page.page_id, float(score)) for page, score in hits
            )
        return prompt

    def register_layer(
        self,
        layer_idx: int,
        token2node,
        num_leaves,
        *,
        token_offset: int,
        replace: bool = False,
        require_full_span: bool = True,
    ) -> dict[str, np.ndarray]:
        """Register one prefill layer for every newly begun page.

        ``token2node`` is the layer's DCI map (``[n_heads, n_dci_tokens]`` or
        the ``(token2node, ...)`` tuple the backend returns); ``token_offset``
        is the absolute prompt position of DCI token 0 for *this* registration.
        It is required on purpose: a wrong positive offset silently maps the
        wrong tokens, so the runtime must derive it (first prefill:
        ``n_sink_pages * page_size``) rather than relying on a 0 default.
        A page that is already ready is not re-registered.

        ``require_full_span=True`` rejects a chunk that is only partly present
        in this registration (for example a page crossing the sink/window
        boundary); pass ``False`` only when mapping the offloaded subset is
        intended.

        Returns ``{page_id: [n_heads, n_leaves] map}``.  On any error the
        whole request is aborted and the error is re-raised.
        """
        with self._lock:
            self._require_active()
            if not isinstance(layer_idx, (int, np.integer)) or int(layer_idx) < 0:
                raise ValueError("layer_idx must be a non-negative integer")
            layer_idx = int(layer_idx)
            if not self._begun:
                return {}
            try:
                maps: dict[str, np.ndarray] = {}
                for page_id in self._begun:
                    maps[page_id] = self.materializer.register_layer_leaves(
                        self.session_id,
                        page_id,
                        layer_idx,
                        token2node,
                        num_leaves,
                        token_offset=token_offset,
                        replace=replace,
                        require_full_span=require_full_span,
                    )
                return maps
            except BaseException as exc:
                # Includes KeyboardInterrupt / asyncio.CancelledError: a page
                # left in ``encoding`` by an interrupted prefill is unusable
                # until an explicit ``retry``, so roll back regardless.
                try:
                    self.abort(
                        f"layer {layer_idx} leaf registration failed: {exc}"
                    )
                finally:
                    raise

    def finish(self) -> list[str]:
        """Mark this request's newly encoded pages as CPU-resident.

        A failure (for example the session was released while it was
        encoding) aborts the whole request and re-raises, mirroring
        ``register_layer``; it never leaves some pages ready and others not.
        """
        with self._lock:
            self._require_active()
            pages = list(self._begun)
            try:
                if pages:
                    self.materializer.complete_encoding(self.session_id, pages)
                self._finished = True
            except BaseException as exc:
                # A signal can land after complete_encoding made every page
                # ready but before ``_finished`` is set.  Aborting then would
                # fail pages that are already CPU-resident; a retry cannot
                # recover them (they are ``failed`` and re-encoding would
                # duplicate the committed append).  Observe the commit
                # instead of trusting the flag.
                if pages and self._all_ready(pages):
                    self._finished = True
                    raise
                try:
                    self.abort(f"finish failed: {exc}")
                finally:
                    raise
            return pages

    def _all_ready(self, pages: Sequence[str]) -> bool:
        """True when every page is CPU/GPU-resident in this session."""
        for page_id in pages:
            try:
                record = self.materializer.record(self.session_id, page_id)
            except KeyError:
                return False
            if record.state not in ("cpu_kv", "gpu_kv"):
                return False
        return True

    def abort(self, reason: str) -> list[str]:
        """Roll back this request's newly begun pages (idempotent).

        Every call re-scans ``_begun`` instead of returning early once
        ``_aborted`` is set: a cancellation may race with ``prepare``, and an
        early return there would leave pages in ``encoding`` forever.  The
        first reason wins.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            if self._finished:
                raise RuntimeError("cannot abort after finish")
            if not self._aborted:
                self._aborted = True
                self._abort_reason = reason
            for page_id in self._begun:
                try:
                    self.materializer.fail(self.session_id, page_id, reason)
                except KeyError:
                    pass
            return list(self._begun)

    @property
    def abort_reason(self) -> str | None:
        with self._lock:
            return self._abort_reason

    # -- internals -------------------------------------------------------
    def _require_unprepared(self) -> None:
        if self._prompt is not None or self._finished or self._aborted:
            raise RuntimeError("prepare() may be called only once per request")

    def _require_active(self) -> None:
        if self._prompt is None:
            raise RuntimeError("call prepare() first")
        if self._finished:
            raise RuntimeError("request is already finished")
        if self._aborted:
            raise RuntimeError(f"request was aborted: {self._abort_reason}")


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

    from icecache.materialization import (
        GlobalRAGPageStore,
        RAGMaterializer,
        chunk_document,
    )

    pages = chunk_document(
        np.arange(12, dtype=np.int64),
        document_id="doc",
        chunk_size=4,
        descriptor_version="v1",
    )
    store = GlobalRAGPageStore()
    store.add_many(pages)
    mat = RAGMaterializer(store, n_kv_heads=2)
    node_index = np.array(
        [
            [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
            [5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8],
        ],
        dtype=np.int32,
    )
    limits = np.array([5, 9], dtype=np.int32)

    request = PrefillMaterialization(mat, "s")
    ok(request.prompt is None and request.page_ids == (), "starts unprepared")
    raises(RuntimeError, request.finish)
    raises(RuntimeError, request.register_layer, 0, node_index, limits,
           token_offset=0)

    prompt = request.prepare(
        pages=[pages[0], pages[2]],
        prefix_token_ids=[7, 8],
        suffix_token_ids=[9],
    )
    ok(prompt.token_ids.tolist() == [7, 8] + pages[0].token_ids.tolist()
       + pages[2].token_ids.tolist() + [9], "prompt assembled")
    ok(request.page_ids == (pages[0].page_id, pages[2].page_id), "prompt pages")
    ok(request.begun_page_ids == (pages[0].page_id, pages[2].page_id),
       "both pages begun")
    ok(request.prompt is prompt, "prompt exposed")
    ok(mat.record("s", pages[0].page_id).token_start == 2,
       "first span recorded")
    ok(mat.record("s", pages[2].page_id).token_end == 10,
       "second span recorded")
    raises(RuntimeError, request.prepare, pages=[], prefix_token_ids=[],
           suffix_token_ids=[])

    # Offset chosen so the first page starts at DCI token 0.  In production
    # the base is n_sink_pages * page_size, not the page's own start.
    maps = request.register_layer(
        0, node_index, limits,
        token_offset=prompt.span_for(pages[0].page_id).token_start,
    )
    ok(set(maps) == {pages[0].page_id, pages[2].page_id}, "layer maps returned")
    ok(maps[pages[0].page_id].shape == (2, 2), "first page map shape")
    ok(maps[pages[0].page_id][0].tolist() == [0, 1], "first page head 0")
    ok(maps[pages[0].page_id][1].tolist() == [5, 6], "first page head 1")
    ok(maps[pages[2].page_id][0].tolist() == [2, 3], "second page head 0")

    # A page in this prompt may be encoded before another layer registers;
    # resolution stays None until that layer is registered.
    ok(mat.resolve_head_leaves("s", pages[0].page_id, 1, 0) is None,
       "unregistered layer resolves None")

    finished = request.finish()
    ok(finished == [pages[0].page_id, pages[2].page_id], "finish returns pages")
    ok(request.finished and not request.aborted, "finished state")
    ok(mat.record("s", pages[0].page_id).state == "cpu_kv",
       "pages complete to cpu_kv")
    raises(RuntimeError, request.finish)
    raises(RuntimeError, request.register_layer, 1, node_index, limits,
           token_offset=0)
    raises(RuntimeError, request.abort, "late")

    # A ready page in the same session is rejected: its KV was produced under
    # a different context, so a second prompt must not silently reuse it.
    mat.submit("s2", [pages[0].page_id])
    mat.begin_encoding("s2", [pages[0].page_id], token_start=0, token_end=4)
    mat.register_layer_leaves("s2", pages[0].page_id, 0, node_index, limits,
                              token_offset=0)
    mat.complete_encoding("s2", [pages[0].page_id])
    again = PrefillMaterialization(mat, "s2")
    raises(ValueError, again.prepare, pages=[pages[0]],
           prefix_token_ids=[7], suffix_token_ids=[9])
    ok(mat.record("s2", pages[0].page_id).state == "cpu_kv",
       "rejected reuse keeps the ready page")
    ok(mat.resolve_head_leaves("s2", pages[0].page_id, 0, 0).tolist()
       == [0, 1], "rejected reuse keeps its leaves")

    # Abort rolls back only this request's newly begun pages.
    aborted = PrefillMaterialization(mat, "s3")
    aborted.prepare(pages=[pages[1]], prefix_token_ids=[7])
    ok(aborted.abort("prefill failed") == [pages[1].page_id], "abort returns")
    ok(aborted.aborted and aborted.abort_reason == "prefill failed", "abort state")
    ok(mat.record("s3", pages[1].page_id).state == "failed", "abort fails page")
    ok(mat.record("s3", pages[1].page_id).layer_leaf_ids == {},
       "abort clears leaf maps")
    ok(aborted.abort("again") == [pages[1].page_id], "abort is idempotent")

    # A registration error aborts the whole request and re-raises.
    failing = PrefillMaterialization(mat, "s4")
    failing.prepare(pages=[pages[0]], prefix_token_ids=[7], suffix_token_ids=[])
    raises(ValueError, failing.register_layer, 0, node_index, limits,
           token_offset=-1)
    ok(failing.aborted, "registration failure aborts")
    ok(mat.record("s4", pages[0].page_id).state == "failed",
       "registration failure fails the page")

    # Out-of-span registration also aborts rather than leaving a partial map.
    failing2 = PrefillMaterialization(mat, "s5")
    failing2.prepare(pages=[pages[0]], prefix_token_ids=[7], suffix_token_ids=[])
    raises(ValueError, failing2.register_layer, 0, node_index[:, :2], limits,
           token_offset=0)
    ok(failing2.aborted, "short token2node aborts")
    ok(mat.record("s5", pages[0].page_id).state == "failed",
       "short token2node fails page")

    # A chunk only partly present can be registered explicitly; the default
    # fails closed instead of silently mapping the wrong tokens.
    partial = PrefillMaterialization(mat, "s10")
    partial.prepare(pages=[pages[0]], prefix_token_ids=[7, 8],
                    suffix_token_ids=[], page_ids=[pages[0].page_id])
    # Span [2, 6) against only four indexed tokens: [2, 6) is out of range.
    raises(ValueError, partial.register_layer, 0, node_index[:, :4], limits,
           token_offset=0)
    ok(partial.aborted, "full-span rejection aborts the request")
    partial.abort("checked full-span rejection")

    partial2 = PrefillMaterialization(mat, "s11")
    partial2.prepare(pages=[pages[0]], prefix_token_ids=[7, 8],
                     suffix_token_ids=[], page_ids=[pages[0].page_id])
    maps = partial2.register_layer(
        0, node_index[:, :4], limits, token_offset=0, require_full_span=False
    )
    ok(partial2.finish() == [pages[0].page_id], "partial registration finishes")
    ok(
        mat.resolve_head_leaves("s11", pages[0].page_id, 0, 0).tolist() == [1],
        "partial span maps the offloaded subset",
    )
    ok(
        mat.resolve_head_leaves("s11", pages[0].page_id, 0, 1).tolist() == [5, 6],
        "partial span maps every offloaded head",
    )

    # Releasing the session mid-request makes finish() abort the request
    # instead of raising while leaving it half-alive.
    gone = PrefillMaterialization(mat, "s12")
    gone.prepare(pages=[pages[0]], prefix_token_ids=[7], suffix_token_ids=[])
    mat.release_session("s12")
    raises(KeyError, gone.finish)
    ok(gone.aborted and not gone.finished, "finish failure aborts")
    ok("finish failed" in (gone.abort_reason or ""), "finish failure reason")
    ok(mat.records("s12") == {}, "released session stays empty")

    # Context-manager use aborts requests that fail or forget to finish.
    blown = PrefillMaterialization(mat, "s13")
    try:
        with blown:
            blown.prepare(pages=[pages[0]], prefix_token_ids=[7],
                          suffix_token_ids=[])
            raise RuntimeError("prefill blew up")
    except RuntimeError:
        pass
    ok(blown.aborted and "RuntimeError" in (blown.abort_reason or ""),
       "context exit aborts after an exception")
    ok(mat.record("s13", pages[0].page_id).state == "failed",
       "context exit fails begun pages")

    forgotten = PrefillMaterialization(mat, "s14")
    with forgotten:
        forgotten.prepare(pages=[pages[0]], prefix_token_ids=[7],
                          suffix_token_ids=[])
    ok(forgotten.aborted and not forgotten.finished,
       "context exit without finish aborts")

    completed = PrefillMaterialization(mat, "s15")
    with completed:
        completed.prepare(pages=[], prefix_token_ids=[1],
                          suffix_token_ids=[])
        completed.finish()
    ok(completed.finished and not completed.aborted,
       "finished context does not abort")

    # A failed prepare leaves no page queued behind.
    unknown = PrefillMaterialization(mat, "s6")
    raises(TypeError, unknown.prepare, pages=["missing/page"],
           prefix_token_ids=[7], suffix_token_ids=[])

    # Pure prompt: no RAG pages, every lifecycle call is a no-op.
    pure = PrefillMaterialization(mat, "s7")
    pure_prompt = pure.prepare(pages=[], prefix_token_ids=[1, 2],
                               suffix_token_ids=[3])
    ok(pure_prompt.token_ids.tolist() == [1, 2, 3], "pure prompt tokens")
    ok(pure.begun_page_ids == (), "pure prompt begins nothing")
    ok(pure.register_layer(0, node_index, limits, token_offset=0) == {},
       "pure register no-op")
    ok(pure.finish() == [], "pure finish no-op")
    ok(mat.records("s7") == {}, "pure prompt creates no records")

    # A subset selection only begins the requested pages.
    subset = PrefillMaterialization(mat, "s8")
    subset.prepare(pages=[pages[0], pages[1]], prefix_token_ids=[7],
                   suffix_token_ids=[], page_ids=[pages[1].page_id])
    ok(subset.begun_page_ids == (pages[1].page_id,), "subset begins one page")
    ok(pages[0].page_id not in mat.records("s8"),
       "unselected page is left untouched")
    subset.abort("done")

    # A failed begin_encoding is rolled back by prepare.  A page already in
    # ``encoding`` makes the batch fail *after* submit queued the other page,
    # which exercises the rollback path rather than assembly validation.
    mat.submit("s9", [pages[2].page_id])
    mat.begin_encoding("s9", [pages[2].page_id], token_start=0, token_end=4)
    wedged = PrefillMaterialization(mat, "s9")
    raises(ValueError, wedged.prepare, pages=[pages[2], pages[1]],
           prefix_token_ids=[7], suffix_token_ids=[])
    ok(mat.record("s9", pages[2].page_id).state == "encoding",
       "pre-existing encoding page is left untouched")
    ok(mat.record("s9", pages[1].page_id).state == "failed",
       "newly queued page is rolled back by a failed begin_prompt")
    ok(wedged.begun_page_ids == () and not wedged.aborted,
       "failed prepare records no begun pages")

    # token_offset is required: relying on a 0 default silently mis-maps.
    required = PrefillMaterialization(mat, "s16")
    required.prepare(pages=[pages[0]], prefix_token_ids=[7],
                     suffix_token_ids=[])
    raises(TypeError, required.register_layer, 0, node_index, limits)
    required.abort("token_offset check")

    # abort() before prepare() is not lost: prepare must refuse to start pages
    # that the cancellation path already believes are rolled back.
    cancelled = PrefillMaterialization(mat, "s17")
    cancelled.abort("cancelled before prepare")
    raises(RuntimeError, cancelled.prepare, pages=[pages[0]],
           prefix_token_ids=[7], suffix_token_ids=[])
    ok(mat.records("s17") == {}, "abort-before-prepare creates no records")

    # BaseException (KeyboardInterrupt / asyncio.CancelledError) must also
    # roll back: a page stranded in ``encoding`` would need an explicit retry.
    class _InterruptingRegister(RAGMaterializer):
        def register_layer_leaves(self, *args, **kwargs):
            raise KeyboardInterrupt("cancelled during prefill")

    intr_mat = _InterruptingRegister(store, n_kv_heads=2)
    interrupted = PrefillMaterialization(intr_mat, "k1")
    interrupted.prepare(pages=[pages[0]], prefix_token_ids=[7],
                        suffix_token_ids=[])
    try:
        interrupted.register_layer(0, node_index, limits, token_offset=0)
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt from register_layer")
    ok(interrupted.aborted, "BaseException aborts the request")
    ok(intr_mat.record("k1", pages[0].page_id).state == "failed",
       "BaseException during register leaves no page in encoding")

    class _InterruptingBegin(RAGMaterializer):
        def begin_encoding(self, session_id, page_ids, **kwargs):
            super().begin_encoding(session_id, page_ids, **kwargs)
            raise KeyboardInterrupt("cancelled during begin_encoding")

    begin_mat = _InterruptingBegin(store, n_kv_heads=2)
    interrupted_begin = PrefillMaterialization(begin_mat, "k3")
    try:
        interrupted_begin.prepare(pages=[pages[0]], prefix_token_ids=[7],
                                  suffix_token_ids=[])
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt from prepare")
    ok(begin_mat.record("k3", pages[0].page_id).state == "failed",
       "BaseException during begin_encoding is rolled back")
    ok(interrupted_begin.begun_page_ids == (),
       "interrupted prepare records no begun pages")

    # An interrupt between begin_prompt() returning and the helper recording
    # ``_begun`` used to strand the started pages in ``encoding``: abort()
    # scanned an empty list.  Inject exactly that window.
    class _InterruptingBookkeeping(PrefillMaterialization):
        def __setattr__(self, name, value):
            if name == "_prompt" and getattr(self, "_armed", False):
                object.__setattr__(self, "_armed", False)
                raise KeyboardInterrupt("cancelled after begin_prompt")
            object.__setattr__(self, name, value)

    book_mat = RAGMaterializer(store, n_kv_heads=2)
    interrupted_book = _InterruptingBookkeeping(book_mat, "k4")
    interrupted_book._armed = True
    try:
        interrupted_book.prepare(pages=[pages[0]], prefix_token_ids=[7],
                                 suffix_token_ids=[])
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt after begin_prompt")
    ok(interrupted_book.aborted,
       "interrupt after begin_prompt marks the request aborted")
    ok(book_mat.record("k4", pages[0].page_id).state == "failed",
       "interrupt after begin_prompt fails the started pages")
    ok(interrupted_book.begun_page_ids == (),
       "interrupted bookkeeping leaves no begun pages")

    # A signal at begin_prompt's *return* boundary (the call itself raises
    # after doing its work) looks identical: begun_out must let the cleanup
    # fail the pages it just started.
    class _InterruptingBeginPromptReturn(RAGMaterializer):
        def begin_prompt(self, *args, **kwargs):
            super().begin_prompt(*args, **kwargs)
            raise KeyboardInterrupt("cancelled at begin_prompt return")

    ret_mat = _InterruptingBeginPromptReturn(store, n_kv_heads=2)
    interrupted_ret = PrefillMaterialization(ret_mat, "k5")
    try:
        interrupted_ret.prepare(pages=[pages[0]], prefix_token_ids=[7],
                                suffix_token_ids=[])
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt at begin_prompt return")
    ok(interrupted_ret.aborted, "interrupt at begin_prompt return aborts")
    ok(ret_mat.record("k5", pages[0].page_id).state == "failed",
       "interrupt at begin_prompt return fails the started pages")

    # A signal after complete_encoding committed every page must not abort:
    # the pages are already CPU-resident and re-encoding would duplicate the
    # committed append.
    class _InterruptingFinishAfterCommit(RAGMaterializer):
        def complete_encoding(self, *args, **kwargs):
            super().complete_encoding(*args, **kwargs)
            raise KeyboardInterrupt("cancelled after complete_encoding")

    post_mat = _InterruptingFinishAfterCommit(store, n_kv_heads=2)
    post = PrefillMaterialization(post_mat, "k6")
    post.prepare(pages=[pages[0]], prefix_token_ids=[7], suffix_token_ids=[])
    try:
        post.finish()
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt from finish")
    ok(post.finished and not post.aborted,
       "interrupt after complete_encoding keeps the request finished")
    ok(post_mat.record("k6", pages[0].page_id).state == "cpu_kv",
       "interrupt after complete_encoding leaves the page ready")

    class _InterruptingFinish(RAGMaterializer):
        def complete_encoding(self, *args, **kwargs):
            raise KeyboardInterrupt("cancelled at finish")

    fin_mat = _InterruptingFinish(store, n_kv_heads=2)
    fin = PrefillMaterialization(fin_mat, "k2")
    fin.prepare(pages=[pages[0]], prefix_token_ids=[7], suffix_token_ids=[])
    fin.register_layer(0, node_index, limits, token_offset=0)
    try:
        fin.finish()
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt from finish")
    ok(fin.aborted, "BaseException during finish aborts the request")
    ok(fin_mat.record("k2", pages[0].page_id).state == "failed",
       "BaseException during finish leaves no page encoding")

    # --- stage-2 retrieval: prepare_from_query ---------------------------
    def one_hot(chunk: np.ndarray) -> np.ndarray:
        vector = np.zeros(3, dtype=np.float64)
        vector[(int(chunk[0]) // 4) % 3] = 1.0
        return vector

    rag_pages = chunk_document(
        np.arange(12, dtype=np.int64),
        document_id="rag",
        chunk_size=4,
        descriptor_version="v1",
        descriptor_fn=one_hot,
    )
    rag_store = GlobalRAGPageStore()
    rag_store.add_many(rag_pages)
    rag_mat = RAGMaterializer(rag_store, n_kv_heads=2)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    retrieved = PrefillMaterialization(rag_mat, "r1")
    ok(retrieved.retrieval_hits == (), "no retrieval hits before prepare")
    prompt_r = retrieved.prepare_from_query(
        query_descriptor=query, top_k=2,
        prefix_token_ids=[7], suffix_token_ids=[9],
    )
    hits = rag_store.search(query, top_k=2)
    ok(retrieved.retrieval_hits == tuple(
        (page.page_id, float(score)) for page, score in hits),
       "retrieval hits match the store search")
    ok(prompt_r.page_spans[0].page_id == rag_pages[0].page_id,
       "query retrieval puts the matching page first")
    expected_tokens = [7]
    for page, _ in hits:
        expected_tokens.extend(page.token_ids.tolist())
    expected_tokens.append(9)
    ok(prompt_r.token_ids.tolist() == expected_tokens,
       "retrieved prompt is assembled from the hits in score order")
    ok(retrieved.begun_page_ids == tuple(page.page_id for page, _ in hits),
       "retrieved pages are begun")
    ok(rag_mat.record("r1", rag_pages[0].page_id).state == "encoding",
       "retrieved page is encoding before prefill")
    ok(retrieved.finish() == [page.page_id for page, _ in hits],
       "retrieved request finishes")

    # min_score filters hits; an empty result is a valid no-RAG prompt.
    filtered = PrefillMaterialization(rag_mat, "r2")
    prompt_f = filtered.prepare_from_query(
        query_descriptor=query, top_k=3, min_score=1.5,
        prefix_token_ids=[7], suffix_token_ids=[9],
    )
    ok(prompt_f.token_ids.tolist() == [7, 9], "filtered prompt has no pages")
    ok(filtered.page_ids == () and filtered.begun_page_ids == (),
       "filtered request begins nothing")
    ok(filtered.retrieval_hits == (), "filtered hits are empty")
    ok(filtered.finish() == [], "no-RAG request finishes empty")

    # Search failures happen before any state transition: the request stays
    # unprepared and retryable, not half-begun.
    untouched = PrefillMaterialization(rag_mat, "r3")
    raises(KeyError, untouched.prepare_from_query, query_descriptor=query,
           top_k=1, prefix_token_ids=[7],
           allowed_page_ids=["missing"])
    raises(ValueError, untouched.prepare_from_query,
           query_descriptor=np.ones(2), top_k=1, prefix_token_ids=[7])
    raises(TypeError, untouched.prepare_from_query, query_descriptor=query,
           top_k=1, prefix_token_ids=[7], min_score=True)
    raises(ValueError, untouched.prepare_from_query, query_descriptor=query,
           top_k=1, prefix_token_ids=[7], min_score=float("nan"))
    ok(untouched.prompt is None and untouched.begun_page_ids == (),
       "failed retrieval leaves the request unprepared")
    ok(rag_mat.records("r3") == {}, "failed retrieval creates no records")
    retried = untouched.prepare_from_query(
        query_descriptor=query, top_k=1, prefix_token_ids=[7])
    ok(retried.page_spans[0].page_id == rag_pages[0].page_id,
       "request is retryable after a failed retrieval")
    raises(RuntimeError, untouched.prepare_from_query, query_descriptor=query,
           top_k=1, prefix_token_ids=[7])
    untouched.abort("retrieval test done")

    # top_k=0 is a valid no-RAG request.
    empty_k = PrefillMaterialization(rag_mat, "r4")
    ok(empty_k.prepare_from_query(query_descriptor=query, top_k=0,
                                  prefix_token_ids=[7]).token_ids.tolist()
       == [7], "top_k zero yields the prefix-only prompt")
    empty_k.abort("top_k zero")

    # Stage B: an append-only block at an absolute session position.  The
    # prompt carries only the appended tokens, but the recorded spans are
    # absolute so they can be mapped with the session's DCI token_offset.
    appended = PrefillMaterialization(mat, "a1")
    append_prompt = appended.prepare_append(
        pages=[pages[1]], position_base=100)
    ok(append_prompt.token_ids.tolist() == pages[1].token_ids.tolist(),
       "append prompt holds only the appended tokens")
    ok(appended.page_ids == (pages[1].page_id,), "append page list")
    ok((mat.record("a1", pages[1].page_id).token_start,
        mat.record("a1", pages[1].page_id).token_end) == (100, 104),
       "append spans are absolute session positions")
    raises(RuntimeError, appended.prepare_append, pages=[pages[0]],
           position_base=104)
    append_maps = appended.register_layer(
        0, node_index, limits, token_offset=96)
    ok(append_maps[pages[1].page_id][0].tolist() == [2, 3],
       "append head 0 maps the absolute span")
    ok(append_maps[pages[1].page_id][1].tolist() == [6, 7],
       "append head 1 maps the absolute span")
    ok(appended.finish() == [pages[1].page_id], "append finishes")

    invalid_append = PrefillMaterialization(mat, "a2")
    raises(ValueError, invalid_append.prepare_append, pages=[pages[0]],
           position_base=-1)
    raises(TypeError, invalid_append.prepare_append, pages=[pages[0]],
           position_base=True)
    ok(invalid_append.prompt is None, "invalid append leaves the request empty")

    # Deep KV is context-bound: re-encoding a ready page in the same session
    # is rejected instead of silently reusing a stale map.
    again = PrefillMaterialization(mat, "a1")
    raises(ValueError, again.prepare_append, pages=[pages[1]],
           position_base=104)
    again.abort("ready page rejected")

    print(f"prefill_materialization self-test: {checks} checks passed")


if __name__ == "__main__":
    _self_test()
