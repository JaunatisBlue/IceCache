"""Session-local materialization for the unified RAG x IceCache control plane.

This module implements Milestones 1-4 of
``docs/UNIFIED_RAG_ICECACHE_IMPLEMENTATION.md``:

* ``GlobalRAGPage`` / ``GlobalRAGPageStore``: immutable global document-page
  metadata (tokens, descriptor, hash).  A global page never owns cross-session
  deep KV.
* ``SessionPageMaterialization`` / ``RAGMaterializer``: per-``(session_id,
  page_id)`` state machine with fail-closed transitions, rollback and session
  cleanup.  The first version only supports materialization *before* the main
  prefill; asynchronous mid-decode encoding is deliberately out of scope.
* ``map_span_to_leaves``: maps a RAG chunk's token span to per-KV-head DCI
  leaf IDs using that layer's ``token2node`` mapping.
* ``CandidatePlan`` / ``build_candidate_plan``: closed-loop two-stage provider
  result.  Unmaterialized RAG pages can only appear in ``deferred_rag_pages``;
  ``ready_leaf_ids`` and ``fallback_leaf_ids`` are validated native leaves.

Correctness boundaries mirrored from the design doc:

1. The global store is metadata-only; it is shared, immutable and never holds
   session KV.
2. Prompt pages and encoded RAG KV are session-private.  Materialization state
   is keyed by ``(session_id, page_id)`` and two sessions citing the same
   global page are fully independent.
3. Different layers / KV heads are never mixed.  Leaf maps are stored per
   layer and per KV head.
4. Unmaterialized RAG pages never become native leaf IDs.
5. Descriptor versions and dimensions may not be mixed in one store.

``token2node`` coordinate trap: IceCache's DCI index covers only the offloaded
middle of the prefill, not the whole prompt.  Its token axis starts at the
first offloaded token (after the sink pages).  Callers must pass
``token_offset`` = absolute prompt position of DCI token 0 when the record's
``token_start``/``token_end`` are prompt-absolute positions.  The argument is
required (no default) because a wrong offset silently maps a RAG chunk to
unrelated leaves; full-span membership is checked by default.

A page that is materialized but has no leaf map for the queried layer (for
example after ``invalidate_layer``) contributes zero leaves; it is never sent
to the deferred channel, because deferral means "encode this at the next
request boundary" and an already-encoded page cannot be encoded twice.

The module is numpy-only (plus a local ``validate_leaf_ids`` import), so it can
be imported in CPU-only analysis environments.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    # Direct execution of the in-module self-test.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from icecache.retrieval import validate_leaf_ids
else:
    from .retrieval import validate_leaf_ids

__all__ = [
    "AssembledPrompt",
    "CandidatePlan",
    "GlobalRAGPage",
    "GlobalRAGPageStore",
    "MaterializationState",
    "PageTokenSpan",
    "RAGMaterializer",
    "SessionPageMaterialization",
    "assemble_prompt",
    "build_candidate_plan",
    "chunk_document",
    "map_span_to_leaves",
]


MaterializationState = Literal[
    "descriptor_only",
    "queued",
    "encoding",
    "cpu_kv",
    "gpu_kv",
    "failed",
]

_STATES = frozenset(
    {"descriptor_only", "queued", "encoding", "cpu_kv", "gpu_kv", "failed"}
)
_READY_STATES = frozenset({"cpu_kv", "gpu_kv"})
_ENCODED_STATES = frozenset({"encoding", "cpu_kv", "gpu_kv"})

# Token-to-leaf mappings use a negative sentinel for unmapped tokens.
_UNMAPPED = -1

# Thresholds for the vectorized fallback fill in build_candidate_plan, chosen
# on the measured crossover (CUDA-index-free CPU microbenchmark, 2026-10-22):
# below ~128 missing slots the Python set loop wins; above it one bool mask is
# 1.5-5x faster.  The mask is O(num_leaves), so cap that to avoid allocating a
# fresh multi-megabyte mask per head per layer per decode token.
_VECTOR_FILL_MIN = 128
_VECTOR_FILL_MAX_LEAVES = 1 << 20

# ``GlobalRAGPageStore.search`` switches from one ``np.dot`` per page to a
# cached stacked matvec above this many matrix elements.  Measured at
# 20k x 1024 on a quiet, NUMA-bound CPU: 72-74 ms/query (loop) vs ~6 ms
# (matvec + lexsort), identical top-k.  Below the threshold the loop avoids
# allocating a descriptor-sized copy for tiny stores.
_SEARCH_MATRIX_MIN_ELEMENTS = 65_536


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return a genuinely immutable array.

    ``array.setflags(write=False)`` is not enough: an array that owns its
    buffer can be made writable again by any consumer, which would mutate an
    otherwise "immutable" global page.  A bytes-backed view cannot: NumPy
    refuses to set ``WRITEABLE`` back to True on it.
    """
    contiguous = np.ascontiguousarray(array)
    frozen = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


_INT32_MAX = np.iinfo(np.int32).max


def _check_leaf_id_bounds(
    array: np.ndarray, *, minimum: int, name: str
) -> None:
    """Reject values that would alias when cast to int32.

    ``validate_leaf_ids`` casts to int32 *before* checking signs and limits,
    so ``2**32`` becomes leaf ``0`` and ``-2**32`` likewise.  Call this on the
    original integer array first; every value that survives is exactly
    representable.
    """
    if not array.size:
        return
    if array.dtype.kind == "i" and int(array.min()) < minimum:
        raise ValueError(f"{name} contains a leaf ID below {minimum}")
    # Every value of a signed integer dtype no wider than int32 is already
    # <= INT32_MAX, so the upper check only matters for wider/unsigned input.
    if not (array.dtype.kind == "i" and array.dtype.itemsize <= 4):
        if int(array.max()) > _INT32_MAX:
            raise ValueError(f"{name} contains an ID that does not fit in int32")


def _as_numpy(value, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _require_token_ids(
    token_ids, *, name: str = "token_ids", allow_empty: bool = False
) -> np.ndarray:
    array = _as_numpy(token_ids, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        # ``np.asarray([])`` is float64; an explicitly empty sequence is still
        # a legitimate no-op prefix/suffix.
        if not allow_empty:
            raise ValueError(f"{name} must not be empty")
        return _readonly(np.empty(0, dtype=np.int64))
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer token IDs")
    if np.any(array < 0):
        raise ValueError(f"{name} must not contain negative token IDs")
    if np.issubdtype(array.dtype, np.unsignedinteger) and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{name} exceeds the int64 token range")
    return _readonly(array.astype(np.int64, copy=False))


def _require_index(value, name: str) -> int:
    """Reject bools: ``isinstance(True, int)`` would silently mean 1.

    Used for layer/head indices, steps and offsets, where a leaked boolean
    would address the wrong layer or shift a token span by one.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_span(token_start: int, token_end: int) -> tuple[int, int]:
    if (
        isinstance(token_start, bool)
        or isinstance(token_end, bool)
        or not isinstance(token_start, (int, np.integer))
        or not isinstance(token_end, (int, np.integer))
    ):
        raise TypeError("token span bounds must be integers")
    token_start, token_end = int(token_start), int(token_end)
    if token_start < 0 or token_end <= token_start:
        raise ValueError(
            f"invalid token span [{token_start}, {token_end}); "
            "expected 0 <= start < end"
        )
    return token_start, token_end


def _content_hash(token_ids: np.ndarray) -> str:
    payload = np.ascontiguousarray(token_ids, dtype="<i8").tobytes()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Global document pages (Milestone 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GlobalRAGPage:
    """Immutable global RAG page metadata.

    ``token_ids`` is the authoritative materialization input: re-tokenizing
    decoded text can change special tokens and chunk boundaries.  ``descriptor``
    is optional only to support id/token-only pages; text retrieval skips it.

    The descriptor is text-level and layer-agnostic.  It is *not* a K centroid
    and must not be compared with native per-layer K descriptors without the
    explicit dual-path score merge described in the design doc.
    """

    page_id: str
    document_id: str
    segment_id: str
    token_ids: np.ndarray
    descriptor: np.ndarray | None
    descriptor_version: str
    text_hash: str
    error_radius: float = 0.0

    def __post_init__(self) -> None:
        for name in ("page_id", "document_id", "segment_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.descriptor_version, str) or not self.descriptor_version:
            raise ValueError("descriptor_version must be a non-empty string")
        if not isinstance(self.text_hash, str) or not self.text_hash:
            raise ValueError("text_hash must be a non-empty string")
        object.__setattr__(self, "token_ids", _require_token_ids(self.token_ids))
        if self.descriptor is not None:
            raw_descriptor = _as_numpy(self.descriptor, name="descriptor")
            if np.iscomplexobj(raw_descriptor):
                raise TypeError("descriptor must be real-valued")
            descriptor = raw_descriptor.astype(np.float64, copy=False)
            if descriptor.ndim != 1 or descriptor.size == 0:
                raise ValueError("descriptor must be a non-empty vector")
            if not np.all(np.isfinite(descriptor)):
                raise ValueError("descriptor must be finite")
            if np.linalg.norm(descriptor) <= 0.0:
                raise ValueError("descriptor must have non-zero norm")
            object.__setattr__(self, "descriptor", _readonly(descriptor))
        radius = float(self.error_radius)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("error_radius must be finite and non-negative")
        object.__setattr__(self, "error_radius", radius)

    @property
    def n_tokens(self) -> int:
        return int(self.token_ids.shape[0])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobalRAGPage):
            return NotImplemented
        return _pages_equal(self, other)

    def __hash__(self) -> int:
        # page_id is part of content equality, so this stays consistent.
        return hash(self.page_id)

    @classmethod
    def from_tokens(
        cls,
        *,
        page_id: str,
        document_id: str,
        segment_id: str,
        token_ids,
        descriptor,
        descriptor_version: str,
        error_radius: float = 0.0,
        text_hash: str | None = None,
    ) -> "GlobalRAGPage":
        tokens = _require_token_ids(token_ids)
        return cls(
            page_id=page_id,
            document_id=document_id,
            segment_id=segment_id,
            token_ids=tokens,
            descriptor=descriptor,
            descriptor_version=descriptor_version,
            text_hash=text_hash or _content_hash(tokens),
            error_radius=error_radius,
        )


class GlobalRAGPageStore:
    """Immutable-by-identity global RAG page directory.

    The store pins ``descriptor_version`` and the descriptor dimension on the
    first insert and rejects any later mismatch.  Re-adding the exact same page
    is idempotent; reusing a ``page_id`` with different content is an error.
    """

    def __init__(self, *, descriptor_version: str | None = None) -> None:
        if descriptor_version is not None and (
            not isinstance(descriptor_version, str) or not descriptor_version
        ):
            raise ValueError("descriptor_version must be a non-empty string")
        self._pages: dict[str, GlobalRAGPage] = {}
        # Norms are per-page constants (descriptors are immutable once added),
        # so computing them at insert turns each query's per-page norm from an
        # O(dim) reduction into one dict lookup.  Measured at 20k pages x 1024
        # dims: 110 of 170 ms per query was norm recomputation.
        self._page_norms: dict[str, float] = {}
        # Stacked search matrix, built lazily for large stores and dropped on
        # every mutation.  ``search`` used to do one ``np.dot`` per page: at
        # 20k x 1024 that is 72-74 ms/query on a quiet CPU, versus ~6 ms for
        # one BLAS matvec plus a lexsort over the same descriptors (measured
        # with identical top-k).  The copy costs one descriptor-sized buffer.
        self._search_cache: (
            tuple[
                np.ndarray, np.ndarray, np.ndarray, list[GlobalRAGPage],
                dict[str, int],
            ]
            | None
        ) = None
        self._lock = threading.RLock()
        self._descriptor_version = descriptor_version
        self._descriptor_dim: int | None = None

    @property
    def descriptor_version(self) -> str | None:
        with self._lock:
            return self._descriptor_version

    @property
    def descriptor_dim(self) -> int | None:
        with self._lock:
            return self._descriptor_dim

    def __len__(self) -> int:
        with self._lock:
            return len(self._pages)

    def __contains__(self, page_id: object) -> bool:
        with self._lock:
            return page_id in self._pages

    def page_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._pages))

    def pages(self) -> tuple[GlobalRAGPage, ...]:
        with self._lock:
            return tuple(self._pages[key] for key in sorted(self._pages))

    def get(self, page_id: str) -> GlobalRAGPage:
        with self._lock:
            try:
                return self._pages[page_id]
            except KeyError:
                raise KeyError(f"unknown global RAG page {page_id!r}") from None

    def remove(self, page_id: str) -> GlobalRAGPage | None:
        with self._lock:
            self._page_norms.pop(page_id, None)
            self._search_cache = None
            return self._pages.pop(page_id, None)

    def add(self, page: GlobalRAGPage) -> GlobalRAGPage:
        return self.add_many([page])[0]

    def add_many(self, pages: Sequence[GlobalRAGPage]) -> list[GlobalRAGPage]:
        """Validate the whole batch before committing any of it."""
        batch = list(pages)
        if not batch:
            return []
        with self._lock:
            version = self._descriptor_version
            dim = self._descriptor_dim
            seen: set[str] = set()
            for page in batch:
                if not isinstance(page, GlobalRAGPage):
                    raise TypeError("add_many expects GlobalRAGPage instances")
                if page.page_id in seen:
                    raise ValueError(f"duplicate page_id {page.page_id!r} in batch")
                seen.add(page.page_id)
                existing = self._pages.get(page.page_id)
                if existing is not None:
                    if not _pages_equal(existing, page):
                        raise ValueError(
                            f"page_id {page.page_id!r} already exists with "
                            "different content"
                        )
                    continue
                if version is None:
                    version = page.descriptor_version
                elif page.descriptor_version != version:
                    raise ValueError(
                        f"descriptor_version mismatch: store has {version!r}, "
                        f"page has {page.descriptor_version!r}"
                    )
                if page.descriptor is not None:
                    page_dim = int(page.descriptor.shape[0])
                    if dim is None:
                        dim = page_dim
                    elif page_dim != dim:
                        raise ValueError(
                            f"descriptor dimension mismatch: store has {dim}, "
                            f"page has {page_dim}"
                        )
            self._descriptor_version = version
            self._descriptor_dim = dim
            inserted = False
            for page in batch:
                stored = self._pages.setdefault(page.page_id, page)
                if stored is page:
                    inserted = True
                    if page.descriptor is not None:
                        self._page_norms[page.page_id] = float(
                            np.linalg.norm(page.descriptor)
                        )
            if inserted:
                # Idempotent re-adds leave the directory unchanged, so they
                # keep the stacked matrix.
                self._search_cache = None
            return [self._pages[page.page_id] for page in batch]

    def search(
        self,
        query_descriptor,
        *,
        top_k: int,
        allowed_page_ids: Sequence[str] | None = None,
    ) -> list[tuple[GlobalRAGPage, float]]:
        """Exact cosine text retrieval over global descriptors.

        Ties are broken by ``page_id`` so ingestion order does not affect the
        baseline.  Pages without a descriptor are skipped.  The query must be
        produced by the same embedding (the store's pinned descriptor version
        guards pages, not the query).
        """
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, (int, np.integer))
            or top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer")
        raw_query = _as_numpy(query_descriptor, name="query_descriptor")
        if np.iscomplexobj(raw_query):
            raise TypeError("query descriptor must be real-valued")
        query = raw_query.astype(np.float64, copy=False)
        if query.ndim != 1 or query.size == 0:
            raise ValueError("query descriptor must be a non-empty vector")
        if not np.all(np.isfinite(query)):
            raise ValueError("query descriptor must be finite")
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 0.0:
            raise ValueError("query descriptor must have non-zero norm")
        if top_k == 0:
            return []
        with self._lock:
            if self._descriptor_dim is None:
                return []
            if query.shape[0] != self._descriptor_dim:
                raise ValueError(
                    f"query descriptor has dimension {query.shape[0]}, "
                    f"store expects {self._descriptor_dim}"
                )
            allowed = None
            if allowed_page_ids is not None:
                if isinstance(allowed_page_ids, (str, bytes)):
                    raise TypeError(
                        "allowed_page_ids must be a sequence, not a string"
                    )
                allowed = set(allowed_page_ids)
                unknown = allowed.difference(self._pages)
                if unknown:
                    raise KeyError(
                        f"unknown allowed page ids: {sorted(unknown)[:5]}"
                    )
            if (
                len(self._pages) * int(self._descriptor_dim)
                >= _SEARCH_MATRIX_MIN_ELEMENTS
            ):
                return self._search_stacked(
                    query, query_norm, allowed, int(top_k)
                )
            scored: list[tuple[GlobalRAGPage, float]] = []
            for page in self._pages.values():
                if page.descriptor is None:
                    continue
                if allowed is not None and page.page_id not in allowed:
                    continue
                page_norm = self._page_norms.get(page.page_id)
                if page_norm is None:
                    # Defensive: a page whose norm was never cached (or was
                    # dropped with an earlier remove and re-added elsewhere).
                    page_norm = float(np.linalg.norm(page.descriptor))
                    self._page_norms[page.page_id] = page_norm
                denominator = query_norm * page_norm
                score = float(np.dot(query, page.descriptor) / denominator)
                scored.append((page, score))
        scored.sort(key=lambda item: (-item[1], item[0].page_id))
        return scored[: int(top_k)]

    def _search_stacked(
        self,
        query: np.ndarray,
        query_norm: float,
        allowed: set[str] | None,
        top_k: int,
    ) -> list[tuple[GlobalRAGPage, float]]:
        """Vectorized ``search`` for large stores (lock held).

        Scores may differ from the per-page loop in the last ulp because BLAS
        chooses a different accumulation order for a matvec than for ``dot``;
        every call on one store state takes this same path, so repeated
        queries and the retrieval/reference comparison stay exactly equal.
        """
        cache = self._search_cache
        if cache is None:
            cache = self._build_search_matrix()
        matrix, norms, ids, pages, index = cache
        rows: np.ndarray | None = None
        if allowed is None:
            scores = (matrix @ query) / (norms * query_norm)
        else:
            rows = np.fromiter(
                (index[page_id] for page_id in allowed if page_id in index),
                dtype=np.intp,
            )
            if rows.size == 0:
                return []
            scores = (matrix[rows] @ query) / (norms[rows] * query_norm)
            ids = ids[rows]
        # ``lexsort`` is stable and sorts by the last key first: score
        # descending, then page_id ascending, matching the Python sort key.
        order = np.lexsort((ids, -scores))[:top_k]
        if rows is None:
            return [(pages[int(i)], float(scores[i])) for i in order]
        return [(pages[int(rows[i])], float(scores[i])) for i in order]

    def _build_search_matrix(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, list[GlobalRAGPage], dict[str, int]
    ]:
        """Stack descriptors once for the vectorized path (lock held)."""
        pages = [
            page for page in self._pages.values() if page.descriptor is not None
        ]
        if pages:
            matrix = np.stack([page.descriptor for page in pages])
            ids = np.array([page.page_id for page in pages])
        else:
            matrix = np.empty((0, int(self._descriptor_dim or 0)), dtype=np.float64)
            ids = np.empty(0, dtype="<U1")
        norms = np.empty(len(pages), dtype=np.float64)
        for row, page in enumerate(pages):
            page_norm = self._page_norms.get(page.page_id)
            if page_norm is None:
                page_norm = float(np.linalg.norm(page.descriptor))
                self._page_norms[page.page_id] = page_norm
            norms[row] = page_norm
        index = {page.page_id: row for row, page in enumerate(pages)}
        self._search_cache = (matrix, norms, ids, pages, index)
        return self._search_cache


def _pages_equal(left: GlobalRAGPage, right: GlobalRAGPage) -> bool:
    if left.page_id != right.page_id:
        return False
    if left.document_id != right.document_id or left.segment_id != right.segment_id:
        return False
    if left.descriptor_version != right.descriptor_version:
        return False
    if left.text_hash != right.text_hash:
        return False
    if left.error_radius != right.error_radius:
        return False
    if not np.array_equal(left.token_ids, right.token_ids):
        return False
    if (left.descriptor is None) != (right.descriptor is None):
        return False
    if left.descriptor is not None and not np.array_equal(
        left.descriptor, right.descriptor
    ):
        return False
    return True


def chunk_document(
    token_ids,
    *,
    document_id: str,
    chunk_size: int,
    descriptor_version: str,
    descriptor_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    overlap: int = 0,
    page_id_prefix: str | None = None,
    error_radius: float = 0.0,
) -> list[GlobalRAGPage]:
    """Split a document into stable token-boundary pages.

    Segment IDs encode their token boundaries (``seg<start>_<end>``), so the
    identity of a chunk that keeps the same boundaries is stable across
    re-ingestion.  ``descriptor_fn`` receives a read-only int64 token array and
    must return a 1-D descriptor; it may be ``None`` for id/token-only pages.
    """
    tokens = _require_token_ids(token_ids)
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, (int, np.integer))
        or overlap < 0
    ):
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    prefix = page_id_prefix or document_id
    step = int(chunk_size) - int(overlap)
    pages: list[GlobalRAGPage] = []
    total = int(tokens.shape[0])
    previous_end: int | None = None
    for start in range(0, total, step):
        end = min(start + int(chunk_size), total)
        if previous_end is not None and end <= previous_end:
            # With overlap > 0 the final window can be fully contained in its
            # predecessor; it adds no new token boundary.
            break
        chunk = tokens[start:end]
        descriptor = None
        if descriptor_fn is not None:
            descriptor = descriptor_fn(chunk)
        segment_id = f"seg{start:08d}_{end:08d}"
        pages.append(
            GlobalRAGPage(
                page_id=f"{prefix}/{segment_id}",
                document_id=document_id,
                segment_id=segment_id,
                token_ids=chunk,
                descriptor=descriptor,
                descriptor_version=descriptor_version,
                text_hash=_content_hash(chunk),
                error_radius=error_radius,
            )
        )
        previous_end = end
        if end == total:
            break
    return pages


@dataclass(frozen=True)
class PageTokenSpan:
    page_id: str
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.page_id, str) or not self.page_id:
            raise ValueError("page_id must be a non-empty string")
        start, end = _require_span(self.token_start, self.token_end)
        object.__setattr__(self, "token_start", start)
        object.__setattr__(self, "token_end", end)

    @property
    def n_tokens(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True, eq=False)
class AssembledPrompt:
    token_ids: np.ndarray
    page_spans: tuple[PageTokenSpan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", _require_token_ids(self.token_ids))
        spans = tuple(self.page_spans)
        seen: set[str] = set()
        for span in spans:
            if not isinstance(span, PageTokenSpan):
                raise TypeError("page_spans must contain PageTokenSpan instances")
            if span.page_id in seen:
                raise ValueError(f"duplicate page span {span.page_id!r}")
            if span.token_end > self.token_ids.shape[0]:
                raise ValueError("page span exceeds assembled prompt length")
            seen.add(span.page_id)
        object.__setattr__(self, "page_spans", spans)

    @property
    def n_tokens(self) -> int:
        return int(self.token_ids.shape[0])

    def span_for(self, page_id: str) -> PageTokenSpan:
        for span in self.page_spans:
            if span.page_id == page_id:
                return span
        raise KeyError(page_id)


def assemble_prompt(
    *,
    pages: Sequence[GlobalRAGPage],
    prefix_token_ids=None,
    suffix_token_ids=None,
) -> AssembledPrompt:
    """Concatenate prefix + RAG pages + suffix and record page spans.

    The returned spans are absolute positions in the assembled prompt.  They
    are the values later translated to DCI coordinates with ``token_offset``.
    """
    prefix = (
        np.empty(0, dtype=np.int64)
        if prefix_token_ids is None
        else _require_token_ids(
            prefix_token_ids, name="prefix_token_ids", allow_empty=True
        )
    )
    suffix = (
        np.empty(0, dtype=np.int64)
        if suffix_token_ids is None
        else _require_token_ids(
            suffix_token_ids, name="suffix_token_ids", allow_empty=True
        )
    )
    pieces = [prefix]
    spans: list[PageTokenSpan] = []
    seen: set[str] = set()
    offset = int(prefix.shape[0])
    for page in pages:
        if not isinstance(page, GlobalRAGPage):
            raise TypeError("pages must contain GlobalRAGPage instances")
        if page.page_id in seen:
            raise ValueError(f"page {page.page_id!r} appears more than once")
        seen.add(page.page_id)
        pieces.append(page.token_ids)
        spans.append(
            PageTokenSpan(
                page_id=page.page_id,
                token_start=offset,
                token_end=offset + page.n_tokens,
            )
        )
        offset += page.n_tokens
    pieces.append(suffix)
    token_ids = np.concatenate(pieces).astype(np.int64, copy=False)
    return AssembledPrompt(token_ids=token_ids, page_spans=tuple(spans))


# ---------------------------------------------------------------------------
# Per-session materialization state machine (Milestone 2)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SessionPageMaterialization:
    """State of one global page inside one session.

    The record is mutable but only through its transition methods, which
    enforce the legal state graph.  ``layer_leaf_ids`` maps ``layer_idx`` to a
    ``[n_kv_heads, max_leaves]`` int32 array padded with ``-1``.
    """

    session_id: str
    page_id: str
    state: MaterializationState = "descriptor_only"
    token_start: int | None = None
    token_end: int | None = None
    layer_leaf_ids: dict[int, np.ndarray] = field(default_factory=dict)
    last_access_step: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.page_id, str) or not self.page_id:
            raise ValueError("page_id must be a non-empty string")
        if self.state not in _STATES:
            raise ValueError(f"unknown materialization state {self.state!r}")
        if (self.token_start is None) != (self.token_end is None):
            raise ValueError("token_start and token_end must both be set or both be None")
        if self.token_start is not None:
            start, end = _require_span(self.token_start, self.token_end)
            self.token_start, self.token_end = start, end
            if self.state == "descriptor_only":
                raise ValueError("descriptor_only pages cannot carry a token span")
        if self.state in _ENCODED_STATES and self.token_start is None:
            raise ValueError(f"{self.state} pages require a token span")
        self.last_access_step = _require_index(
            self.last_access_step, "last_access_step"
        )
        cleaned: dict[int, np.ndarray] = {}
        for layer_idx, mapping in dict(self.layer_leaf_ids).items():
            layer_key = _require_index(layer_idx, "layer_leaf_ids key")
            cleaned[layer_key] = _validate_leaf_map(mapping)
        if cleaned and self.state not in _ENCODED_STATES:
            raise ValueError(
                f"{self.state} pages cannot own layer leaf maps"
            )
        self.layer_leaf_ids = cleaned
        if self.state == "failed":
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("failed pages must record a non-empty error")
        else:
            self.error = None

    # -- queries ---------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self.state in _READY_STATES

    @property
    def n_tokens(self) -> int:
        if self.token_start is None:
            return 0
        return int(self.token_end) - int(self.token_start)

    def snapshot(self) -> "SessionPageMaterialization":
        return SessionPageMaterialization(
            session_id=self.session_id,
            page_id=self.page_id,
            state=self.state,
            token_start=self.token_start,
            token_end=self.token_end,
            layer_leaf_ids={
                layer: np.array(mapping, copy=True)
                for layer, mapping in self.layer_leaf_ids.items()
            },
            last_access_step=self.last_access_step,
            error=self.error,
        )

    # -- transitions -----------------------------------------------------
    def _clear_encoding(self) -> None:
        self.token_start = None
        self.token_end = None
        self.layer_leaf_ids = {}
        self.error = None

    def queue(self) -> None:
        if self.state == "queued":
            return
        if self.state == "descriptor_only":
            self.state = "queued"
            return
        if self.state == "failed":
            self._clear_encoding()
            self.state = "queued"
            return
        raise ValueError(
            f"cannot queue page {self.page_id!r} from state {self.state!r}"
        )

    def begin_encoding(self, token_start: int, token_end: int) -> None:
        if self.state != "queued":
            raise ValueError(
                f"cannot begin encoding page {self.page_id!r} from state "
                f"{self.state!r}"
            )
        start, end = _require_span(token_start, token_end)
        self.token_start, self.token_end = start, end
        self.state = "encoding"

    def complete_encoding(self, *, target: Literal["cpu_kv"] = "cpu_kv") -> None:
        if self.state != "encoding":
            raise ValueError(
                f"cannot complete page {self.page_id!r} from state {self.state!r}"
            )
        if target != "cpu_kv":
            raise ValueError("encoding must complete into 'cpu_kv' first")
        if self.token_start is None:
            raise ValueError("encoding page has no token span")
        self.state = "cpu_kv"

    def promote_gpu(self) -> None:
        if self.state == "gpu_kv":
            return
        if self.state != "cpu_kv":
            raise ValueError(
                f"cannot promote page {self.page_id!r} from state {self.state!r}"
            )
        self.state = "gpu_kv"

    def demote_cpu(self) -> None:
        if self.state == "cpu_kv":
            return
        if self.state != "gpu_kv":
            raise ValueError(
                f"cannot demote page {self.page_id!r} from state {self.state!r}"
            )
        self.state = "cpu_kv"

    def fail(self, error: str) -> None:
        if not isinstance(error, str) or not error:
            raise ValueError("error must be a non-empty string")
        self.state = "failed"
        self.token_start = None
        self.token_end = None
        self.layer_leaf_ids = {}
        self.error = error

    def touch(self, step: int) -> None:
        self.last_access_step = _require_index(step, "step")

    def register_leaf_map(
        self, layer_idx: int, mapping, *, replace: bool = False
    ) -> np.ndarray:
        validated = _validate_leaf_map(mapping)
        return self._store_leaf_map(layer_idx, validated, replace=replace)

    def _register_leaf_map_validated(
        self, layer_idx: int, mapping, *, replace: bool = False
    ) -> np.ndarray:
        """Internal: commit a map built by :func:`map_span_to_leaves`.

        The caller guarantees int32 values, per-head uniqueness, trailing
        ``-1`` padding, leaf bounds, and a read-only (bytes-backed) buffer.
        Re-validating here costs ~0.18 ms per page/layer at production shape
        (16 tokens x 8 heads) and dominates request-boundary registration.
        """
        array = np.ascontiguousarray(mapping, dtype=np.int32)
        if array.flags.writeable:
            raise ValueError("validated leaf map must be read-only")
        return self._store_leaf_map(layer_idx, array, replace=replace)

    def _store_leaf_map(
        self, layer_idx: int, validated: np.ndarray, *, replace: bool
    ) -> np.ndarray:
        layer_idx = _require_index(layer_idx, "layer_idx")
        if self.state not in _ENCODED_STATES:
            raise ValueError(
                f"page {self.page_id!r} is {self.state!r}; leaf maps require "
                "an encoded page"
            )
        if layer_idx in self.layer_leaf_ids and not replace:
            raise ValueError(
                f"layer {layer_idx} already has a leaf map for page "
                f"{self.page_id!r}; pass replace=True to re-register"
            )
        for layer, existing in self.layer_leaf_ids.items():
            if existing.shape[0] != validated.shape[0]:
                raise ValueError(
                    f"layer {layer} has {existing.shape[0]} KV heads, new map "
                    f"has {validated.shape[0]}"
                )
        self.layer_leaf_ids[layer_idx] = validated
        return np.array(validated, copy=True)


def _validate_leaf_map(mapping) -> np.ndarray:
    array = np.asarray(mapping)
    if array.ndim != 2:
        raise ValueError("leaf map must be 2-D [n_kv_heads, max_leaves]")
    if array.shape[0] <= 0:
        raise ValueError("leaf map must contain at least one KV head")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("leaf map must contain integer leaf IDs")
    # Check the source dtype before the int32 cast: a wrapped ID would
    # otherwise pass as a legitimate leaf.
    _check_leaf_id_bounds(array, minimum=_UNMAPPED, name="leaf map")
    array = np.ascontiguousarray(array, dtype=np.int32)
    for row in array:
        valid = row[row >= 0]
        if len(np.unique(valid)) != valid.size:
            raise ValueError("leaf map contains duplicate IDs within a KV head")
        # Padding must be a suffix so callers can slice by count.
        if valid.size and np.any(row[: valid.size] < 0):
            raise ValueError("leaf map padding must be a trailing suffix")
    return _readonly(array)


def map_span_to_leaves(
    token2node,
    *,
    token_start: int,
    token_end: int,
    num_leaves,
) -> np.ndarray:
    """Map a DCI token span to unique leaf IDs per KV head.

    ``token2node`` may be the ``(node_index, node_offset)`` tuple returned by
    the DCI database; only ``node_index`` is used.  ``node_index`` may be
    ``[n_kv_heads, n_tokens]`` or a flat vector.  Negative entries are treated
    as unmapped tokens and skipped; any non-negative entry outside
    ``[0, num_leaves[head])`` is rejected (it indicates a stale or inconsistent
    mapping, not a legitimate candidate).

    Returns an int32 ``[n_kv_heads, max_leaves]`` array padded with ``-1``.
    Each returned leaf contains at least one token of the span by construction.
    An empty map (``shape[1] == 0``) means no span token was indexed.
    """
    node_index = token2node
    if isinstance(node_index, tuple) and len(node_index) == 2:
        # The DCI backend returns ``(node_index, node_offset)``; only the
        # first entry is used.  A tuple of two *per-head rows* has the same
        # shape, and taking row 0 would silently drop every other head.
        # Fail closed on the ambiguous 1-D/1-D case rather than guess.
        first, second = node_index
        try:
            first_array = np.asarray(first)
            second_array = np.asarray(second)
        except (TypeError, ValueError):
            first_array = second_array = None
        if (
            first_array is not None
            and second_array is not None
            and first_array.ndim == 1
            and second_array.shape == first_array.shape
        ):
            raise ValueError(
                "ambiguous token2node 2-tuple: two same-length 1-D arrays are "
                "indistinguishable from (node_index, node_offset); pass "
                "token2node[0] explicitly or a 2-D [n_kv_heads, n_tokens] map"
            )
        node_index = first
    nodes = _as_numpy(node_index, name="token2node")
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    if nodes.ndim != 2:
        raise ValueError("token2node must be 1-D or 2-D")
    if not np.issubdtype(nodes.dtype, np.integer):
        raise TypeError("token2node must contain integer leaf IDs")
    if np.issubdtype(nodes.dtype, np.unsignedinteger) and nodes.size:
        if int(nodes.max()) > np.iinfo(np.int64).max:
            raise ValueError("token2node contains IDs outside the int64 range")
    n_heads, n_tokens = nodes.shape
    if n_heads <= 0 or n_tokens <= 0:
        raise ValueError("token2node must be non-empty on both axes")
    limits = _normalise_num_leaves(num_leaves, n_heads)
    start, end = _require_span(token_start, token_end)
    if end > n_tokens:
        raise ValueError(
            f"token span [{start}, {end}) exceeds token2node length {n_tokens}; "
            "check token_offset (DCI token 0 is not prompt token 0)"
        )
    # Compare in int64 so an unsigned token2node cannot mix with the signed
    # limits (NumPy would otherwise promote both to float64).
    span = nodes[:, start:end].astype(np.int64, copy=False)
    over = span >= limits.reshape(n_heads, 1)
    if np.any(over):
        head, pos = np.argwhere(over)[0]
        raise ValueError(
            f"head {int(head)}: leaf {int(span[head, pos])} is outside "
            f"num_leaves={int(limits[head])}"
        )
    # One sort per head, then a shifted comparison: duplicates land next to
    # each other and negatives lead, so a single mask yields the unique
    # non-negative leaves.  This is ~4x faster than np.unique per head on the
    # production span (16 tokens x 8 heads), where call overhead dominates.
    values = np.sort(span, axis=1)
    keep = np.empty(values.shape, dtype=bool)
    keep[:, 0] = True
    if values.shape[1] > 1:
        keep[:, 1:] = values[:, 1:] != values[:, :-1]
    keep &= values >= 0
    counts = keep.sum(axis=1)
    max_len = int(counts.max())
    result = np.full((n_heads, max_len), _UNMAPPED, dtype=np.int32)
    for head in range(n_heads):
        count = int(counts[head])
        if count:
            result[head, :count] = values[head][keep[head]]
    return _readonly(result)


class RAGMaterializer:
    """Per-session materialization table for global RAG pages.

    Prefill-before-request usage::

        hits = page_store.search(query_descriptor, top_k=k)
        prompt = assemble_prompt(pages=[p for p, _ in hits],
                                 prefix_token_ids=prefix,
                                 suffix_token_ids=query)
        mat.submit(session_id, [p.page_id for p, _ in hits])
        mat.begin_encoding(session_id, [p.page_id for p, _ in hits],
                           token_spans={s.page_id:
                                        (s.token_start, s.token_end)
                                        for s in prompt.page_spans})
        # Run the normal prefill on prompt.token_ids.  After _DCI_add, for each
        # indexed layer and page:
        #   mat.register_layer_leaves(session_id, page_id, layer, token2node,
        #                             num_leaves, token_offset=dci_start)
        mat.complete_encoding(session_id, [p.page_id for p, _ in hits])

    No background encoding is started here; the caller owns the encode step and
    must release the actual CPU/GPU KV pages and session index on cleanup (this
    class only owns the ``(session_id, page_id)`` metadata).  All public
    methods are thread-safe because DCI observer callbacks may run on worker
    threads.
    """

    def __init__(
        self,
        pages: GlobalRAGPageStore,
        *,
        n_kv_heads: int | None = None,
    ) -> None:
        if n_kv_heads is not None:
            if isinstance(n_kv_heads, bool) or not isinstance(
                n_kv_heads, (int, np.integer)
            ):
                # ``int(0.5) == 0`` would otherwise build a materializer in
                # which every multi-head registration fails later.
                raise TypeError("n_kv_heads must be an integer")
            if int(n_kv_heads) <= 0:
                raise ValueError("n_kv_heads must be positive")
        self.pages = pages
        self._n_kv_heads = None if n_kv_heads is None else int(n_kv_heads)
        self._records: dict[tuple[str, str], SessionPageMaterialization] = {}
        self._lock = threading.RLock()
        # Bumped by every state/map mutation; plan caches reject entries from
        # an older epoch.  ``touch`` is deliberately excluded (it is a
        # decode-hot access marker and does not affect resolution).
        self._epoch = 0
        self._prefix_caches: dict[tuple[str, int], "PlanPrefixCache"] = {}

    @property
    def n_kv_heads(self) -> int | None:
        with self._lock:
            return self._n_kv_heads

    # -- lifecycle -------------------------------------------------------
    def submit(self, session_id: str, page_ids: Sequence[str]) -> list[str]:
        """Queue pages for materialization; returns newly queued page IDs.

        Idempotent: already queued/encoding/ready pages are left untouched.
        Re-submitting a failed page retries it and discards its rolled-back
        span/leaf maps.
        """
        _require_session_id(session_id)
        ids = _require_page_ids(page_ids)
        queued: list[str] = []
        with self._lock:
            for page_id in ids:
                self.pages.get(page_id)  # fail before mutating anything
            for page_id in ids:
                record = self._records.get((session_id, page_id))
                if record is None:
                    record = SessionPageMaterialization(
                        session_id=session_id, page_id=page_id
                    )
                    self._records[(session_id, page_id)] = record
                if record.state in _READY_STATES or record.state in (
                    "queued",
                    "encoding",
                ):
                    continue
                record.queue()
                queued.append(page_id)
            self._epoch += 1
        return queued

    def poll_ready(self, session_id: str) -> list[str]:
        _require_session_id(session_id)
        with self._lock:
            return sorted(
                page_id
                for (sid, page_id), record in self._records.items()
                if sid == session_id and record.is_ready
            )

    def begin_encoding(
        self,
        session_id: str,
        page_ids: Sequence[str],
        *,
        token_start: int | None = None,
        token_end: int | None = None,
        token_spans: Mapping[str, tuple[int, int]] | None = None,
    ) -> list[str]:
        """Move queued pages into ``encoding`` with prompt-absolute spans.

        Either pass one ``token_start``/``token_end`` shared by every page, or
        pass ``token_spans`` mapping each page to its own span.  The latter is
        required when several retrieved chunks sit at different prompt
        positions, e.g. ``assemble_prompt(...).page_spans``.
        """
        _require_session_id(session_id)
        ids = _require_page_ids(page_ids)
        if token_spans is not None:
            if token_start is not None or token_end is not None:
                raise ValueError(
                    "pass either token_spans or token_start/token_end, not both"
                )
            spans = dict(token_spans)
            if set(spans) != set(ids):
                raise ValueError(
                    "token_spans keys must exactly match page_ids"
                )
            normalised = {
                page_id: _require_span(*spans[page_id]) for page_id in ids
            }
        else:
            if token_start is None or token_end is None:
                raise ValueError(
                    "begin_encoding requires token_spans or token_start/token_end"
                )
            shared = _require_span(token_start, token_end)
            normalised = {page_id: shared for page_id in ids}
        with self._lock:
            records = [self._require_record(session_id, page_id) for page_id in ids]
            # The recorded span must be exactly this page's token range.  A
            # shared span across chunks of unequal length would otherwise map
            # unrelated prompt tokens into the page's native leaves.
            for record in records:
                page = self.pages.get(record.page_id)
                start, end = normalised[record.page_id]
                if end - start != page.n_tokens:
                    raise ValueError(
                        f"page {record.page_id!r}: span [{start}, {end}) covers "
                        f"{end - start} tokens, but the global page has "
                        f"{page.n_tokens}"
                    )
            # Validate the whole batch before mutating any record.
            for record in records:
                if record.state != "queued":
                    raise ValueError(
                        f"cannot begin encoding page {record.page_id!r} from "
                        f"state {record.state!r}"
                    )
            for record in records:
                start, end = normalised[record.page_id]
                record.begin_encoding(start, end)
            self._epoch += 1
        return list(ids)

    def begin_prompt(
        self,
        session_id: str,
        prompt: AssembledPrompt,
        page_ids: Sequence[str] | None = None,
        *,
        position_base: int = 0,
        begun_out: list[str] | None = None,
    ) -> list[str]:
        """Submit and begin the RAG pages of an already assembled prompt.

        This is the stage-A request-boundary helper: the caller assembles the
        prompt and will prefill it normally.  This method only records each
        page's absolute prompt span and moves the page to ``encoding``.  After
        the prefill / ``_DCI_add``, register each indexed layer with
        ``register_layer_leaves`` and then call ``complete_encoding``.

        ``position_base`` is the absolute session position of
        ``prompt.token_ids[0]``.  It is 0 when this prompt is the whole request
        (stage A) and the current context length when the prompt is an
        *append* at an agent/tool boundary (stage B).  Recorded spans must be
        absolute session positions because ``register_layer_leaves``
        subtracts the absolute DCI ``token_offset`` from them; ``prompt``
        itself keeps local coordinates and only the records are shifted.

        Deep KV is valid only for the context that produced it, and a request
        gets its own session (design doc §7).  Passing a page that is already
        ``cpu_kv``/``gpu_kv`` therefore raises instead of silently reusing a
        map that may point at DCI leaves from another context; start a new
        session for a re-encoded page.

        ``begun_out``, when given, is extended with the page IDs this call
        moved to ``encoding`` *before* the call returns.  A caller that must
        clean up after a ``BaseException`` delivered at the return boundary
        (when it cannot bind the return value) should pass a list it already
        holds and fail every ID in it, so pages cannot strand in ``encoding``
        with no owning request.
        """
        _require_session_id(session_id)
        if not isinstance(prompt, AssembledPrompt):
            raise TypeError("prompt must be an AssembledPrompt")
        if isinstance(position_base, bool) or not isinstance(
            position_base, (int, np.integer)
        ):
            raise TypeError("position_base must be an integer")
        if int(position_base) < 0:
            raise ValueError("position_base must be non-negative")
        position_base = int(position_base)
        available = {span.page_id: span for span in prompt.page_spans}
        if page_ids is None:
            ids = [span.page_id for span in prompt.page_spans]
        else:
            ids = list(page_ids)
            if not ids:
                return []
            ids = _require_page_ids(ids)
            unknown = [page_id for page_id in ids if page_id not in available]
            if unknown:
                raise ValueError(
                    f"pages {unknown!r} have no span in the assembled prompt"
                )
        if not ids:
            return []
        spans = {
            page_id: (
                available[page_id].token_start + position_base,
                available[page_id].token_end + position_base,
            )
            for page_id in ids
        }
        # Snapshot the pre-call state so the rollback touches only pages whose
        # state this call changed.  A page that was already ``queued`` (for a
        # retry or an earlier submit) is moved queued->encoding by
        # ``begin_encoding`` and must be rolled back too; a page another
        # request is already encoding must not be.  The outer lock (reentrant)
        # makes snapshot+submit+begin_encoding atomic, so no other thread can
        # queue/create an id in between and be mistaken for this call's page.
        with self._lock:
            previous = {
                page_id: (
                    None
                    if (record := self._records.get((session_id, page_id)))
                    is None
                    else record.state
                )
                for page_id in ids
            }
            try:
                self.submit(session_id, ids)
                started = self.begin_encoding(
                    session_id, ids, token_spans=spans
                )
                if begun_out is not None:
                    # Recorded while still under the lock, immediately after
                    # the state change: the caller's cleanup can then see
                    # every page even if a signal lands on ``return``.
                    begun_out.extend(started)
            except BaseException:
                # Under the lock, any record that now exists in
                # queued/encoding was moved by this call.  Deriving that from
                # the snapshot (not from submit's return value) also covers a
                # BaseException delivered *inside* submit after it queued
                # pages: that call never returns, so its list is lost.
                for page_id in ids:
                    record = self._records.get((session_id, page_id))
                    if record is None:
                        continue
                    was = previous[page_id]
                    if was is None or was in (
                        "descriptor_only",
                        "failed",
                    ):
                        # This call created/requeued the page; whether the
                        # interrupt landed before, inside or after submit, it
                        # has no owning request now.
                        if record.state in ("queued", "encoding"):
                            record.fail("begin_prompt failed")
                    elif was == "queued" and record.state == "encoding":
                        # Pre-queued, moved to encoding by this call.  If it is
                        # still ``queued`` the interrupt came before its turn
                        # and it belongs to no request here, so leave it.
                        record.fail("begin_prompt failed")
                self._epoch += 1
                raise
        return ids

    def complete_encoding(
        self, session_id: str, page_ids: Sequence[str]
    ) -> list[str]:
        """Mark encoded pages as CPU-resident after the prefill finished."""
        _require_session_id(session_id)
        ids = _require_page_ids(page_ids)
        with self._lock:
            records = [self._require_record(session_id, page_id) for page_id in ids]
            for record in records:
                if record.state != "encoding":
                    raise ValueError(
                        f"cannot complete page {record.page_id!r} from state "
                        f"{record.state!r}"
                    )
            for record in records:
                record.complete_encoding()
            self._epoch += 1
        return list(ids)

    def mark_gpu(self, session_id: str, page_ids: Sequence[str]) -> list[str]:
        _require_session_id(session_id)
        ids = _require_page_ids(page_ids)
        with self._lock:
            records = [self._require_record(session_id, page_id) for page_id in ids]
            for record in records:
                if record.state not in _READY_STATES:
                    raise ValueError(
                        f"cannot promote page {record.page_id!r} from state "
                        f"{record.state!r}"
                    )
            for record in records:
                record.promote_gpu()
            self._epoch += 1
        return list(ids)

    def mark_cpu(self, session_id: str, page_ids: Sequence[str]) -> list[str]:
        _require_session_id(session_id)
        ids = _require_page_ids(page_ids)
        with self._lock:
            records = [self._require_record(session_id, page_id) for page_id in ids]
            for record in records:
                if record.state not in _READY_STATES:
                    raise ValueError(
                        f"cannot demote page {record.page_id!r} from state "
                        f"{record.state!r}"
                    )
            for record in records:
                record.demote_cpu()
            self._epoch += 1
        return list(ids)

    def fail(self, session_id: str, page_id: str, error: str) -> None:
        """Roll back a page: drop its span and leaf maps, record the error."""
        _require_session_id(session_id)
        with self._lock:
            self._require_record(session_id, page_id).fail(error)
            self._epoch += 1

    def retry(self, session_id: str, page_id: str) -> None:
        """Return a failed -- or interrupted ``encoding`` -- page to ``queued``.

        An ``encoding`` page whose prefill was interrupted would otherwise be
        unrecoverable: ``submit`` skips it and ``queue`` rejects it.  Retrying
        drops its span and leaf maps atomically, so the next ``begin_prompt``
        starts from a clean state.  Any encoder still holding the old page must
        not call ``complete_encoding`` afterwards; that call now fails closed.
        """
        _require_session_id(session_id)
        with self._lock:
            record = self._require_record(session_id, page_id)
            if record.state == "encoding":
                record.fail("retry after interrupted encoding")
            record.queue()
            self._epoch += 1

    def touch(self, session_id: str, page_ids: Sequence[str], *, step: int) -> None:
        _require_session_id(session_id)
        ids = _require_page_ids(page_ids)
        with self._lock:
            records = [self._require_record(session_id, page_id) for page_id in ids]
            for record in records:
                record.touch(step)

    def release_page(self, session_id: str, page_id: str) -> bool:
        """Drop one metadata record; the caller must cancel any in-flight encode."""
        _require_session_id(session_id)
        with self._lock:
            key = (session_id, page_id)
            if key not in self._records:
                return False
            # Bump *before* deleting.  An interrupt between the delete and
            # the bump would leave an external plan cache at the current
            # epoch serving rows for a page that no longer exists, and a
            # retry returns False without ever bumping.
            self._epoch += 1
            del self._records[key]
            return True

    def release_session(self, session_id: str) -> int:
        """Drop every ``(session_id, page_id)`` record; never touches globals.

        This clears control-plane metadata only.  The caller must also release
        the session's CPU/GPU KV pages and delete its prompt/delta index; those
        resources are not owned by this table.
        """
        _require_session_id(session_id)
        with self._lock:
            keys = [key for key in self._records if key[0] == session_id]
            cache_keys = [
                cache_key
                for cache_key in self._prefix_caches
                if cache_key[0] == session_id
            ]
            # Bump *before* any deletion: an interrupt between the deletes
            # and the bump would leave external plan caches at the current
            # epoch after the records are gone, and a retry (with nothing
            # left to remove) would never bump.
            if keys or cache_keys:
                self._epoch += 1
            for key in keys:
                del self._records[key]
            for key in cache_keys:
                del self._prefix_caches[key]
            return len(keys)

    # -- introspection ---------------------------------------------------
    def record(self, session_id: str, page_id: str) -> SessionPageMaterialization:
        _require_session_id(session_id)
        with self._lock:
            return self._require_record(session_id, page_id).snapshot()

    def records(self, session_id: str) -> dict[str, SessionPageMaterialization]:
        _require_session_id(session_id)
        with self._lock:
            return {
                page_id: record.snapshot()
                for (sid, page_id), record in self._records.items()
                if sid == session_id
            }

    def state_counts(self, session_id: str) -> dict[str, int]:
        _require_session_id(session_id)
        counts: dict[str, int] = {}
        with self._lock:
            for (sid, _), record in self._records.items():
                if sid == session_id:
                    counts[record.state] = counts.get(record.state, 0) + 1
        return counts

    # -- leaf resolution (Milestone 3) -----------------------------------
    def register_layer_leaves(
        self,
        session_id: str,
        page_id: str,
        layer_idx: int,
        token2node,
        num_leaves,
        *,
        token_offset: int,
        replace: bool = False,
        require_full_span: bool = True,
    ) -> np.ndarray:
        """Register the leaf map for one page/layer from ``token2node``.

        ``token_offset`` is the absolute prompt position of DCI token 0; the
        record's prompt span is shifted into DCI coordinates before mapping,
        i.e. ``dci_index = prompt_position - token_offset`` (the offset is
        *subtracted*, not added).
        It has no default on purpose: the runtime must derive it (first prefill:
        ``n_sink_pages * page_size``).  A wrong positive offset does not raise
        -- it silently maps the wrong prompt tokens -- so this argument is
        required rather than defaulting to 0.  It is per registration: after a
        staged add, a page appended by a later add may need a different offset
        than a page from the first prefill.
        With ``require_full_span=False`` a partially offloaded chunk is clipped
        and only its indexed tokens contribute; a fully outside chunk yields an
        empty ``[n_heads, 0]`` map.  On failure the record keeps its previously
        registered maps; call ``fail`` to roll the whole page back.
        """
        _require_session_id(session_id)
        layer_idx = _require_index(layer_idx, "layer_idx")
        token_offset = _require_index(token_offset, "token_offset")
        with self._lock:
            record = self._require_record(session_id, page_id)
            if record.state not in _ENCODED_STATES or record.token_start is None:
                raise ValueError(
                    f"page {page_id!r} is {record.state!r}; register leaves only "
                    "while encoding or after prefill"
                )
            page = self.pages.get(page_id)
            if record.n_tokens != page.n_tokens:
                raise ValueError(
                    f"page {page_id!r}: recorded span has {record.n_tokens} "
                    f"tokens, but the global page has {page.n_tokens}"
                )
            nodes = token2node
            if isinstance(nodes, tuple) and len(nodes) == 2:
                nodes = nodes[0]
            nodes_array = _as_numpy(nodes, name="token2node")
            if nodes_array.ndim == 1:
                # A flat vector could be a single KV head or one row of a
                # multi-head map; inferring one head would pin the
                # materializer to the wrong shape for good.
                if self._n_kv_heads is None:
                    raise ValueError(
                        "flat token2node is ambiguous without n_kv_heads; pass "
                        "a 2-D [n_kv_heads, n_tokens] map or construct "
                        "RAGMaterializer(n_kv_heads=...)"
                    )
                nodes_array = nodes_array.reshape(1, -1)
            if nodes_array.ndim != 2:
                raise ValueError("token2node must be 1-D or 2-D")
            n_tokens = int(nodes_array.shape[1])
            start = int(record.token_start) - int(token_offset)
            end = int(record.token_end) - int(token_offset)
            if require_full_span:
                if start < 0 or end > n_tokens:
                    raise ValueError(
                        f"prompt span [{record.token_start}, {record.token_end}) "
                        f"with token_offset={token_offset} maps to DCI span "
                        f"[{start}, {end}) outside [0, {n_tokens}); the chunk is "
                        "not fully indexed (check the sink/window split)"
                    )
            else:
                start = max(0, start)
                end = min(n_tokens, end)
                if end <= start:
                    limits = _normalise_num_leaves(num_leaves, nodes_array.shape[0])
                    empty = _readonly(
                        np.full((limits.size, 0), _UNMAPPED, dtype=np.int32)
                    )
                    self._commit_leaf_map(
                        record, layer_idx, empty, replace, validated=True
                    )
                    return np.array(empty, copy=True)
            mapping = map_span_to_leaves(
                nodes_array,
                token_start=start,
                token_end=end,
                num_leaves=num_leaves,
            )
            self._commit_leaf_map(
                record, layer_idx, mapping, replace, validated=True
            )
            return np.array(mapping, copy=True)

    def _commit_leaf_map(
        self,
        record: SessionPageMaterialization,
        layer_idx: int,
        mapping: np.ndarray,
        replace: bool,
        *,
        validated: bool = False,
    ) -> None:
        heads = mapping.shape[0]
        if self._n_kv_heads is not None and self._n_kv_heads != heads:
            raise ValueError(
                f"leaf map has {heads} KV heads, materializer expects "
                f"{self._n_kv_heads}"
            )
        for layer, existing in record.layer_leaf_ids.items():
            if existing.shape[0] != heads:
                raise ValueError(
                    f"layer {layer} has {existing.shape[0]} KV heads, new map "
                    f"has {heads}"
                )
        # Commit the head count only after a successful registration, so a
        # rejected map cannot pin the materializer to the wrong shape.
        if validated:
            record._register_leaf_map_validated(layer_idx, mapping, replace=replace)
        else:
            record.register_leaf_map(layer_idx, mapping, replace=replace)
        if self._n_kv_heads is None:
            self._n_kv_heads = heads
        self._epoch += 1

    def page_is_ready(self, session_id: str, page_id: str) -> bool:
        """True when the page's KV is materialized (``cpu_kv``/``gpu_kv``).

        A ready page may still lack a leaf map for the queried layer (for
        example after ``invalidate_layer``).  Such a page contributes zero
        leaves for that layer; it must **not** be routed to the deferred
        channel, because deferral re-queues a page that is already encoded and
        the next ``begin_prompt`` would fail.  Pass this as
        ``build_candidate_plan(is_page_ready=...)`` to distinguish the two
        ``None`` cases a head resolver returns.
        """
        _require_session_id(session_id)
        with self._lock:
            record = self._get_record(session_id, page_id)
            return record is not None and record.is_ready

    def resolve_native_leaves(
        self, session_id: str, page_id: str, layer_idx: int
    ) -> np.ndarray | None:
        """Return the padded ``[n_heads, max_leaves]`` map or ``None``.

        ``None`` means the page is not materialized or the layer mapping has not
        been registered yet.  An empty ``[n_heads, 0]`` array means the page is
        materialized but contributes no indexed leaves for that layer.
        """
        _require_session_id(session_id)
        layer_idx = _require_index(layer_idx, "layer_idx")
        with self._lock:
            record = self._get_record(session_id, page_id)
            if record is None or not record.is_ready:
                return None
            mapping = record.layer_leaf_ids.get(layer_idx)
            if mapping is None:
                return None
            return np.array(mapping, copy=True)

    def resolve_page_leaves(
        self, session_id: str, page_id: str, layer_idx: int
    ) -> np.ndarray | None:
        """Read-only ``[n_heads, max_leaves]`` map without a defensive copy.

        This is the decode-hot-path accessor: the returned array is backed by
        bytes and cannot be made writeable.  Use ``resolve_native_leaves``
        when the caller might mutate the result.
        """
        _require_session_id(session_id)
        layer_idx = _require_index(layer_idx, "layer_idx")
        with self._lock:
            record = self._get_record(session_id, page_id)
            if record is None or not record.is_ready:
                return None
            return record.layer_leaf_ids.get(layer_idx)

    def resolve_head_leaves(
        self, session_id: str, page_id: str, layer_idx: int, head_idx: int
    ) -> np.ndarray | None:
        """1-D valid leaf IDs for one KV head, or ``None`` if not ready.

        A page that is materialized but has no map for this layer (for example
        after ``invalidate_layer``) returns an empty array, not ``None``: it is
        a legitimate zero-leaf contribution and must not be deferred.  ``None``
        therefore means "the page is not materialized yet".
        """
        head_idx = _require_index(head_idx, "head_idx")
        mapping = self.resolve_page_leaves(session_id, page_id, layer_idx)
        if mapping is None:
            if self.page_is_ready(session_id, page_id):
                return np.empty(0, dtype=np.int32)
            return None
        if head_idx >= mapping.shape[0]:
            raise ValueError(
                f"head {head_idx} is outside the map's {mapping.shape[0]} KV heads"
            )
        row = mapping[head_idx]
        # ``mapping`` normally comes from a validated registration, but a
        # subclass can override ``resolve_page_leaves``; check the source
        # dtype before the int32 cast so an int64 ID cannot alias to a
        # different leaf here.
        if row.dtype.kind not in "iu":
            raise TypeError("resolved leaf map must contain integer IDs")
        if row.dtype.itemsize > 4:
            _check_leaf_id_bounds(row, minimum=0, name="resolved leaf map")
        return np.ascontiguousarray(row[row >= 0], dtype=np.int32)

    def invalidate_layer(self, session_id: str, layer_idx: int) -> int:
        """Drop recorded leaf maps for one layer across a session.

        A later ``_DCI_add`` (or a reuse pass) may reassign native leaves, so
        maps recorded from the previous ``token2node`` can become stale even
        though the page's KV is still resident.  Call this when the runtime
        detects reorganization, then re-register from the new ``token2node``.
        Returns the number of pages whose map was dropped.  Page state and
        token spans are untouched; resolution returns ``None`` for the layer
        until a new map is registered.
        """
        _require_session_id(session_id)
        layer_idx = _require_index(layer_idx, "layer_idx")
        dropped = 0
        with self._lock:
            for (sid, _), record in self._records.items():
                if sid != session_id:
                    continue
                if record.layer_leaf_ids.pop(layer_idx, None) is not None:
                    dropped += 1
            self._epoch += 1
        return dropped

    def head_resolver(
        self, session_id: str, layer_idx: int
    ) -> Callable[[str, int], np.ndarray | None]:
        """Bind a ``(page_id, head) -> leaf IDs`` callable for plans.

        The bound callable fetches each page's complete ``[n_heads,
        max_leaves]`` map once and caches its per-head rows for the callable's
        lifetime (one plan), so resolving every head of a page costs one table
        lookup instead of one per head.  A later registration or
        ``invalidate_layer`` is visible to the next resolver, not to this one,
        which keeps a single plan internally consistent.  A subclass that
        overrides ``resolve_head_leaves`` or ``resolve_page_leaves`` keeps the
        plain per-head path (its accessor may not be a pure lookup).  The
        batched rows are read-only views of the page map: consume them, do not
        mutate; use ``resolve_native_leaves`` when a writable copy is needed.
        """
        _require_session_id(session_id)
        layer_idx = _require_index(layer_idx, "layer_idx")
        if (
            getattr(self.resolve_head_leaves, "__func__", None)
            is not RAGMaterializer.resolve_head_leaves
            or getattr(self.resolve_page_leaves, "__func__", None)
            is not RAGMaterializer.resolve_page_leaves
        ):
            def resolve_custom(page_id: str, head_idx: int) -> np.ndarray | None:
                return self.resolve_head_leaves(
                    session_id, page_id, layer_idx, head_idx
                )

            return resolve_custom
        cache: dict[str, tuple[np.ndarray, ...] | None] = {}

        def resolve(page_id: str, head_idx: int) -> np.ndarray | None:
            head_idx = _require_index(head_idx, "head_idx")
            if page_id not in cache:
                cache[page_id] = self._head_rows(session_id, page_id, layer_idx)
            rows = cache[page_id]
            if rows is None:
                return None
            if not rows:
                return np.empty(0, dtype=np.int32)
            if head_idx >= len(rows):
                raise ValueError(
                    f"head {head_idx} is outside the map's {len(rows)} KV heads"
                )
            return rows[head_idx]

        return resolve

    def _head_rows(
        self, session_id: str, page_id: str, layer_idx: int
    ) -> tuple[np.ndarray, ...] | None:
        """Per-head compact leaf rows for one page, from a single map lookup.

        ``None`` means not materialized; an empty tuple means materialized but
        without a map for this layer (every head contributes zero leaves).
        """
        mapping = self.resolve_page_leaves(session_id, page_id, layer_idx)
        if mapping is None:
            return () if self.page_is_ready(session_id, page_id) else None
        if mapping.dtype.kind not in "iu":
            raise TypeError("resolved leaf map must contain integer IDs")
        if mapping.dtype.itemsize > 4:
            _check_leaf_id_bounds(mapping, minimum=0, name="resolved leaf map")
        if mapping.shape[1] == 0:
            # A registered-but-empty map (no span token indexed): every head
            # contributes zero leaves.  ``argmax`` on a zero-length axis raises,
            # so this must precede the padding scan.
            return tuple(
                np.empty(0, dtype=np.int32) for _ in range(mapping.shape[0])
            )
        if int(mapping.min()) >= 0:
            # No -1 padding: registration validated every entry, so the rows
            # can be returned as views (one scan of the page instead of one
            # per head, which dominated a cold plan).
            return tuple(mapping)
        neg = mapping < 0
        counts = np.where(
            neg.any(axis=1), neg.argmax(axis=1), mapping.shape[1]
        )
        return tuple(
            np.ascontiguousarray(row[: int(count)], dtype=np.int32)
            for row, count in zip(mapping, counts)
        )

    def plan_prefix_cache(
        self, session_id: str, layer_idx: int
    ) -> PlanPrefixCache:
        """Return the epoch-versioned merge cache for one session layer.

        Pass it as ``build_candidate_plan(prefix_cache=...)`` together with
        this materializer's ``head_resolver``, ``page_is_ready`` and
        ``trusted_resolver=True``.  Any state or leaf-map mutation through
        this materializer invalidates every entry; do not use the cache with
        a resolver that can change leaf visibility behind the materializer's
        back.  ``release_session`` drops the session's caches.
        """
        _require_session_id(session_id)
        if not isinstance(layer_idx, (int, np.integer)) or int(layer_idx) < 0:
            raise ValueError("layer_idx must be a non-negative integer")
        key = (session_id, int(layer_idx))
        with self._lock:
            cache = self._prefix_caches.get(key)
            if cache is None:
                cache = self._prefix_caches[key] = PlanPrefixCache(self, key)
            return cache

    # -- internals -------------------------------------------------------
    def _require_record(
        self, session_id: str, page_id: str
    ) -> SessionPageMaterialization:
        record = self._get_record(session_id, page_id)
        if record is None:
            raise KeyError(
                f"page {page_id!r} has no materialization record in session "
                f"{session_id!r}; submit it first"
            )
        return record

    def _get_record(
        self, session_id: str, page_id: str
    ) -> SessionPageMaterialization | None:
        return self._records.get((session_id, page_id))


def _require_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")


def _require_page_ids(page_ids: Sequence[str]) -> list[str]:
    if isinstance(page_ids, str):
        raise TypeError("page_ids must be a sequence of strings, not a string")
    ids = list(page_ids)
    if not ids:
        raise ValueError("page_ids must not be empty")
    if any(not isinstance(page_id, str) or not page_id for page_id in ids):
        raise ValueError("page_ids must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("page_ids must not contain duplicates")
    return ids


def _normalise_num_leaves(num_leaves, n_heads: int) -> np.ndarray:
    raw = np.asarray(num_leaves)
    if raw.dtype.kind not in "iu":
        raise TypeError("num_leaves must contain integers")
    limits = raw.astype(np.int64, copy=False).reshape(-1)
    if limits.size == 1:
        limits = np.repeat(limits, n_heads)
    if limits.shape != (n_heads,):
        raise ValueError(
            f"num_leaves has {limits.size} entries for {n_heads} KV heads"
        )
    # ``min``/``max`` dispatch once instead of twice through ``np.any``; the
    # semantics are identical for a non-empty 1-D array.
    if int(limits.min()) <= 0:
        raise ValueError("every KV head must expose at least one leaf")
    if int(limits.max()) > _INT32_MAX:
        raise ValueError("num_leaves must fit in int32 native leaf IDs")
    return limits


# ---------------------------------------------------------------------------
# Closed-loop candidate plan (Milestone 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CandidatePlan:
    """Two-stage provider result.

    ``ready_leaf_ids`` and ``fallback_leaf_ids`` are both fixed-width
    ``[n_kv_heads, width]`` native leaves.  ``deferred_rag_pages`` are selected
    RAG pages that were not materialized; they never appear in either array.

    ``fallback_queried`` is ``False`` only on the provider's full-prefix-cache
    hit path, where the ready leaves fill every head and the fallback is
    provably unused.  ``fallback_leaf_ids`` then mirrors ``ready_leaf_ids``
    purely to keep the fixed shape; it is not a native result and no consumer
    may use it (``n_fallback_leaves_used`` is ``0``).

    Direct construction checks shape, dtype, int32 representability and
    within-head uniqueness.  Per-head range limits need ``num_leaves`` and are
    therefore enforced only by :func:`build_candidate_plan`; prefer that
    constructor (or call ``validate_leaf_ids`` yourself) in runtime code.
    """

    ready_leaf_ids: np.ndarray
    deferred_rag_pages: tuple[str, ...]
    fallback_leaf_ids: np.ndarray
    selection_metrics: dict[str, float]
    fallback_queried: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.fallback_queried, bool):
            raise TypeError("fallback_queried must be a bool")
        ready = np.asarray(self.ready_leaf_ids)
        fallback = np.asarray(self.fallback_leaf_ids)
        if ready.ndim != 2 or fallback.ndim != 2:
            raise ValueError("leaf arrays must be 2-D [n_kv_heads, width]")
        if ready.shape != fallback.shape:
            raise ValueError(
                f"ready {ready.shape} and fallback {fallback.shape} shapes differ"
            )
        if ready.shape[0] <= 0 or ready.shape[1] <= 0:
            raise ValueError("leaf arrays must be non-empty")
        if not np.issubdtype(ready.dtype, np.integer) or not np.issubdtype(
            fallback.dtype, np.integer
        ):
            raise TypeError("leaf arrays must contain integer leaf IDs")
        if np.any(ready < 0) or np.any(fallback < 0):
            raise ValueError("leaf arrays may not contain padding/negative IDs")
        if np.any(ready > _INT32_MAX) or np.any(fallback > _INT32_MAX):
            raise ValueError("leaf arrays contain IDs that do not fit in int32")
        for name, array in (("ready_leaf_ids", ready), ("fallback_leaf_ids", fallback)):
            for row in array:
                if row.size > 1:
                    ordered = np.sort(row)
                    if np.any(ordered[1:] == ordered[:-1]):
                        raise ValueError(
                            f"{name} contains duplicate leaf IDs within a KV head"
                        )
        if isinstance(self.deferred_rag_pages, str):
            raise TypeError("deferred_rag_pages must be a sequence, not a string")
        deferred = tuple(self.deferred_rag_pages)
        if len(set(deferred)) != len(deferred) or any(
            not isinstance(page_id, str) or not page_id for page_id in deferred
        ):
            raise ValueError("deferred_rag_pages must be unique non-empty strings")
        metrics = {
            str(key): float(value) for key, value in dict(self.selection_metrics).items()
        }
        object.__setattr__(
            self, "ready_leaf_ids", _readonly(ready.astype(np.int32, copy=False))
        )
        object.__setattr__(
            self, "fallback_leaf_ids", _readonly(fallback.astype(np.int32, copy=False))
        )
        object.__setattr__(self, "deferred_rag_pages", deferred)
        object.__setattr__(self, "selection_metrics", metrics)

    @classmethod
    def _from_validated(
        cls,
        *,
        ready_leaf_ids: np.ndarray,
        fallback_leaf_ids: np.ndarray,
        deferred_rag_pages: Sequence[str],
        selection_metrics: dict,
        fallback_queried: bool = True,
    ) -> "CandidatePlan":
        """Internal fast path for ``build_candidate_plan``.

        The caller guarantees both arrays are int32-representable, in range,
        non-negative, within-head distinct and shape-identical.  Only the
        cheap field normalization is repeated; this avoids a hot-path
        re-validation on every decode token.
        """
        plan = object.__new__(cls)
        ready = np.ascontiguousarray(ready_leaf_ids, dtype=np.int32)
        fallback = np.ascontiguousarray(fallback_leaf_ids, dtype=np.int32)
        if ready.shape != fallback.shape or ready.ndim != 2:
            raise ValueError("validated plan arrays must be matching 2-D shapes")
        object.__setattr__(plan, "ready_leaf_ids", _readonly(ready))
        object.__setattr__(plan, "fallback_leaf_ids", _readonly(fallback))
        object.__setattr__(plan, "deferred_rag_pages", tuple(deferred_rag_pages))
        object.__setattr__(plan, "selection_metrics", dict(selection_metrics))
        object.__setattr__(plan, "fallback_queried", fallback_queried)
        return plan

    @classmethod
    def _from_frozen_ready(
        cls,
        *,
        ready_leaf_ids: np.ndarray,
        fallback_leaf_ids: np.ndarray,
        deferred_rag_pages: Sequence[str],
        selection_metrics: dict,
        fallback_queried: bool = True,
    ) -> "CandidatePlan":
        """Fast path for a ready row that is already ``_readonly``-frozen.

        :class:`PlanPrefixCache` uses this: it stores the exact frozen array
        it built, so re-freezing it on every decode token is pure copy.  The
        fallback still gets its own immutable snapshot unless
        ``fallback_queried`` is ``False``, in which case ``fallback_leaf_ids``
        is the cached ready row and exists only to keep the fixed shape (the
        provider did not call the fallback at all).
        """
        plan = object.__new__(cls)
        fallback = _readonly(
            np.ascontiguousarray(fallback_leaf_ids, dtype=np.int32)
        )
        if ready_leaf_ids.shape != fallback.shape or ready_leaf_ids.ndim != 2:
            raise ValueError("validated plan arrays must be matching 2-D shapes")
        object.__setattr__(plan, "ready_leaf_ids", ready_leaf_ids)
        object.__setattr__(plan, "fallback_leaf_ids", fallback)
        object.__setattr__(plan, "deferred_rag_pages", tuple(deferred_rag_pages))
        object.__setattr__(plan, "selection_metrics", dict(selection_metrics))
        object.__setattr__(plan, "fallback_queried", fallback_queried)
        return plan

    @property
    def metrics(self) -> dict[str, float]:
        """Alias for the shorter name used by the retrieval.py contract."""
        return self.selection_metrics


@dataclass(frozen=True)
class _PlanPrefixEntry:
    """One reusable merge result (all heads filled by ready leaves)."""

    rows: np.ndarray
    deferred: tuple[str, ...]
    ready_pages: int
    rag_leaves_used: int
    bounds: tuple[tuple[int, int] | None, ...]

    def plan_metrics(self, *, n_heads: int, width: int) -> dict[str, float]:
        """The metrics of replaying this entry (no fallback was used)."""
        return {
            "n_ready_rag_pages": float(self.ready_pages),
            "n_deferred_rag_pages": float(len(self.deferred)),
            "n_rag_leaves_used": float(self.rag_leaves_used),
            "n_fallback_leaves_used": 0.0,
            "n_kv_heads": float(n_heads),
            "width": float(width),
        }


class PlanPrefixCache:
    """Per-``(session, layer)`` cache of merge results for trusted resolvers.

    A plan whose ready leaves fill every head's ``width`` contains no fallback
    value, so its rows, deferred list and metrics are identical on every
    decode token with the same per-head selection and ``width``.  Caching that
    result removes the merge (the dominant provider cost at production shape)
    from the steady-state decode path.

    Correctness relies on two properties, both supplied by
    ``RAGMaterializer``:

    * every state or leaf-map mutation bumps the materializer epoch, and a
      lookup under a different epoch drops all entries;
    * entries store the min/max of *all* ready leaves per head (not only the
      chosen ones), so a shrunken ``num_leaves`` still fails closed on a hit.

    The cache is only valid for the built-in resolver and readiness hook.  A
    caller with a custom accessor can mutate leaf visibility without touching
    the epoch; do not pass this cache to :func:`build_candidate_plan` in that
    case.  ``MAX_ENTRIES`` bounds memory when the selection changes often; an
    overflowing cache is cleared rather than evicted one by one.
    """

    MAX_ENTRIES = 32

    def __init__(self, owner, key: tuple[str, int]) -> None:
        self._owner = owner
        self._key = key
        self._epoch = -1
        self._entries: dict[tuple, _PlanPrefixEntry] = {}
        self.hits = 0
        self.misses = 0

    def _sync_locked(self) -> None:
        """Drop entries from an older materialization epoch (lock held)."""
        if self._epoch != self._owner._epoch:
            self._entries.clear()
            self._epoch = self._owner._epoch

    @property
    def session_id(self) -> str:
        return self._key[0]

    @property
    def layer_idx(self) -> int:
        return self._key[1]

    def epoch(self) -> int:
        """Current materializer epoch; pass to ``store`` on a miss."""
        with self._owner._lock:
            return self._owner._epoch

    def lookup(self, key: tuple) -> _PlanPrefixEntry | None:
        with self._owner._lock:
            self._sync_locked()
            entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def probe(
        self,
        per_head_ids: Sequence[Sequence[str]],
        *,
        width: int,
        limits,
        n_heads: int,
    ) -> _PlanPrefixEntry | None:
        """Look up an already-parsed selection before running the fallback.

        ``per_head_ids`` must be the output of
        :func:`normalise_selected_page_ids`.  The entry is returned only when
        every head was filled by ready leaves, so the fallback's values are
        provably unused; the caller may skip the fallback query entirely.
        Applies the same epoch sync and closed-range re-check as the hit path
        in :func:`build_candidate_plan`.
        """
        normalized = _normalise_num_leaves(limits, n_heads)
        key = (width, tuple(tuple(ids) for ids in per_head_ids))
        entry = self.lookup(key)
        if entry is None:
            return None
        for head, bound in enumerate(entry.bounds):
            if bound is not None and (
                bound[0] < 0 or bound[1] >= normalized[head]
            ):
                raise ValueError(
                    f"head {head}: trusted leaf outside [0, {normalized[head]})"
                )
        return entry

    def store(self, key: tuple, entry: _PlanPrefixEntry, *, epoch: int) -> bool:
        """Insert ``entry`` when it was computed within the current epoch.

        A mismatch means a materialization transition interleaved with the
        computation, so the entry may mix two states; it is dropped rather
        than cached.  Returns whether the entry was stored.
        """
        with self._owner._lock:
            if epoch != self._owner._epoch:
                return False
            self._sync_locked()
            if len(self._entries) >= self.MAX_ENTRIES:
                self._entries.clear()
            self._entries[key] = entry
        return True

    def clear(self) -> None:
        with self._owner._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._owner._lock:
            self._sync_locked()
            return len(self._entries)


def _plan_page_id(
    item, layer_idx: int | None, head_idx: int | None = None
) -> tuple[str, str | None]:
    if isinstance(item, str):
        return item, None
    page = getattr(item, "page", item)
    page_id = getattr(page, "page_id", None)
    if not isinstance(page_id, str) or not page_id:
        raise TypeError("selected entries must be page IDs or page-like objects")
    source = getattr(page, "source", None)
    if source is not None and not isinstance(source, str):
        raise TypeError("page source must be a string")
    if source == "prompt":
        # Prompt-source entries are ignored by the caller; do not validate
        # their layer/head metadata.
        return page_id, source
    if layer_idx is not None:
        # A selector bug can return another layer's candidates; the resulting
        # leaves are still in range, so only an explicit check catches it.
        exposed = getattr(page, "layer_idx", None)
        if exposed is not None and exposed != layer_idx:
            raise ValueError(
                f"page {page_id!r} belongs to layer {exposed}, but layer "
                f"{layer_idx} is being queried"
            )
    if head_idx is not None:
        exposed_head = getattr(page, "head_idx", None)
        if exposed_head is not None and exposed_head != head_idx:
            raise ValueError(
                f"page {page_id!r} belongs to KV head {exposed_head}, but it "
                f"is listed for head {head_idx}"
            )
    return page_id, source


def normalise_selected_page_ids(
    selected_rag_by_head: Sequence[Sequence[object]],
    *,
    n_heads: int,
    layer_idx: int | None = None,
) -> list[list[str]]:
    """Parse the selector's per-head entries into page IDs.

    Prompt-source entries are dropped: the prompt side is represented by the
    runtime's fallback leaves, never by a RAG page ID.  Shared by
    :func:`build_candidate_plan` and the provider's pre-fallback cache probe,
    so both derive exactly the same cache key.
    """
    if len(selected_rag_by_head) != n_heads:
        raise ValueError(
            f"selected_rag_by_head has {len(selected_rag_by_head)} heads, "
            f"expected {n_heads}"
        )
    per_head_ids: list[list[str]] = []
    for head, selection in enumerate(selected_rag_by_head):
        if isinstance(selection, (str, bytes)):
            # A flat list of page IDs whose length happens to equal n_heads
            # would otherwise be iterated character by character, silently
            # deferring one-character "page IDs".
            raise TypeError(
                f"head {head}: expected a sequence of candidates, got a "
                "string; pass one list of page IDs per KV head"
            )
        ids: list[str] = []
        for item in selection:
            # Plain page-ID strings are the common path and skip the per-item
            # ``_plan_page_id`` call (it dominated the cached plan's profile
            # at production width: 224 items/call).
            if type(item) is str:
                if not item:
                    # The public CandidatePlan constructor and the deferred
                    # queue both reject empty page IDs; without this check the
                    # builder emitted a plan they would not accept.
                    raise ValueError(
                        f"head {head}: selected page ID must be non-empty"
                    )
                ids.append(item)
                continue
            page_id, source = _plan_page_id(item, layer_idx, head)
            if source == "prompt":
                continue
            ids.append(page_id)
        per_head_ids.append(ids)
    return per_head_ids


def build_candidate_plan(
    *,
    fallback_leaf_ids,
    selected_rag_by_head: Sequence[Sequence[object]],
    resolve_leaves: Callable[[str, int], np.ndarray | None],
    num_leaves,
    layer_idx: int | None = None,
    is_page_ready: Callable[[str], bool] | None = None,
    trusted_resolver: bool = False,
    prefix_cache: "PlanPrefixCache | None" = None,
) -> CandidatePlan:
    """Combine selected RAG pages with a validated prompt fallback.

    ``selected_rag_by_head[h]`` is the selector's ordered page list for KV head
    ``h``; entries may be page IDs or ``PageCandidate``-like objects.  Entries
    with ``source == "prompt"`` are ignored here: the prompt side is
    represented by ``fallback_leaf_ids``, the runtime's own validated native
    prompt result.  Ready RAG leaves displace fallback slots, so a selector
    enforcing a prompt-page floor must account for that displacement itself.

    ``resolve_leaves(page_id, head) -> 1-D leaf IDs | None`` must return
    ``None`` while the page has no native leaves for the layer (not
    materialized, or its layer map was never registered) and an array
    (possibly empty) once the page is resolvable.  Readiness is a per-page
    property; a page is deferred if *any* head reports ``None``.

    ``is_page_ready(page_id)`` disambiguates "not materialized" from
    "materialized but no map for this layer" (for example after
    ``invalidate_layer``).  A ready page whose resolver returns ``None``
    contributes zero leaves instead of being deferred: the deferred channel
    re-queues pages, and an already-encoded page cannot be re-encoded.
    ``RAGMaterializer`` exposes ``page_is_ready`` for this.  A page reported
    not ready is deferred without calling ``resolve_leaves`` at all, and when
    no selected page is ready the merge is skipped and the validated fallback
    is returned as the plan's ready rows (same result, decode-hot shortcut).

    ``layer_idx``, when given, rejects a candidate that exposes a different
    ``layer_idx`` (a selector that mixed layers); likewise a candidate exposing
    a different ``head_idx`` than its list position is rejected.  Ready pages
    contribute
    leaves first, and the prompt fallback fills the remaining fixed-width slots
    without duplicates; ready leaves past ``width`` are dropped.

    ``trusted_resolver=True`` declares that every array from ``resolve_leaves``
    was already validated against these per-head ``num_leaves`` (1-D, integer,
    non-negative, in range, duplicate-free) -- true for
    ``RAGMaterializer.head_resolver``, whose maps are built by
    ``map_span_to_leaves``.  It skips the per-page checks and the final ready
    re-validation, because the merged rows are deduplicated here and their
    pieces are certified.  Arrays wider than int32 are still range-checked
    before the cast (otherwise ``2**32 + 5`` would silently become leaf 5).
    The fallback is always validated.  This is a decode-hot-path option; keep
    the default for arbitrary resolvers.

    ``prefix_cache`` (a :class:`PlanPrefixCache`, only with a trusted
    resolver) reuses the merged rows of a selection whose ready leaves fill
    every head's width, because that result contains no fallback value and is
    therefore identical on every decode token until the materialization state
    changes.  Selections that need fallback leaves are not cached.  Cache
    entries are epoch-versioned and still apply the closed-range check above;
    a cache hit is semantically identical to recomputing the plan -- it is an
    optimization knob (disable it for a live A/B), not a different policy.
    """
    fallback = np.asarray(fallback_leaf_ids)
    if fallback.ndim != 2:
        raise ValueError("fallback_leaf_ids must be 2-D [n_kv_heads, width]")
    n_heads, width = fallback.shape
    if width <= 0:
        # ``CandidatePlan`` rejects zero-width leaf arrays; reject them here
        # instead of returning a plan its own constructor would refuse.
        raise ValueError("fallback_leaf_ids must have a positive width")
    limits = _normalise_num_leaves(num_leaves, n_heads)
    # Inline equivalent of retrieval.validate_leaf_ids, with the source-dtype
    # wraparound check (2**32 -> 0) done before the int32 cast and the
    # within-head duplicate check vectorized across heads.
    if not np.issubdtype(fallback.dtype, np.integer):
        raise TypeError("fallback_leaf_ids must contain integer leaf IDs")
    _check_leaf_id_bounds(fallback, minimum=0, name="fallback_leaf_ids")
    fallback = np.ascontiguousarray(fallback, dtype=np.int32)
    if fallback.size:
        # Negatives were rejected above (signed) or are impossible (unsigned).
        if np.any(fallback >= limits.reshape(n_heads, 1)):
            raise ValueError("fallback_leaf_ids contains an out-of-range leaf ID")
        if width > 1:
            ordered = np.sort(fallback, axis=1)
            if np.any(ordered[:, 1:] == ordered[:, :-1]):
                raise ValueError(
                    "fallback_leaf_ids contains duplicate leaf IDs within a "
                    "KV head"
                )
    per_head_ids = normalise_selected_page_ids(
        selected_rag_by_head, n_heads=n_heads, layer_idx=layer_idx
    )

    cache_key = None
    if prefix_cache is not None:
        if not trusted_resolver:
            raise ValueError("prefix_cache requires trusted_resolver=True")
        cache_epoch = prefix_cache.epoch()
        cache_key = (width, tuple(tuple(ids) for ids in per_head_ids))
        cached = prefix_cache.lookup(cache_key)
        if cached is not None:
            # Same closed-range check as the slow path: ``limits`` comes from
            # the live DCI index and can shrink without a materializer
            # mutation.
            for head, bound in enumerate(cached.bounds):
                if bound is not None and (
                    bound[0] < 0 or bound[1] >= limits[head]
                ):
                    raise ValueError(
                        f"head {head}: trusted leaf outside [0, {limits[head]})"
                    )
            metrics = cached.plan_metrics(n_heads=n_heads, width=width)
            return CandidatePlan._from_frozen_ready(
                ready_leaf_ids=cached.rows,
                deferred_rag_pages=cached.deferred,
                fallback_leaf_ids=fallback,
                selection_metrics=metrics,
            )

    # Resolve readiness once per (page, head).  A page is usable only if it is
    # ready for every head; inconsistent resolvers fail closed into deferred.
    all_page_ids: list[str] = []
    seen_pages: set[str] = set()
    for ids in per_head_ids:
        for page_id in ids:
            if page_id not in seen_pages:
                seen_pages.add(page_id)
                all_page_ids.append(page_id)
    resolved: dict[tuple[str, int], np.ndarray] = {}
    page_ready: dict[str, bool] = {}
    empty = np.empty(0, dtype=np.int32)
    for page_id in all_page_ids:
        if is_page_ready is not None and not is_page_ready(page_id):
            # Known-unmaterialized: it can only be deferred.  Skipping the
            # per-head resolver calls is the decode-cold path (the common one
            # before the boundary materializes a page) and is equivalent:
            # the built-in resolver returns None for such a page.
            page_ready[page_id] = False
            continue
        missing = 0
        for head in range(n_heads):
            leaves = resolve_leaves(page_id, head)
            if leaves is None:
                missing += 1
                continue
            array = np.asarray(leaves)
            if array.ndim != 1:
                raise ValueError(
                    f"page {page_id!r}, head {head}: leaves must be 1-D"
                )
            # Checked even on the trusted path: the int32 cast below would
            # otherwise alias 2**32 + 5 to leaf 5.
            if array.dtype.kind not in "iu":
                raise TypeError(
                    f"page {page_id!r}, head {head}: leaves must be integers"
                )
            if array.dtype.itemsize > 4:
                _check_leaf_id_bounds(
                    array, minimum=0, name=f"page {page_id!r} head {head}"
                )
            if not trusted_resolver:
                if array.size and np.any(array < 0):
                    raise ValueError(
                        f"page {page_id!r}, head {head}: negative leaf ID"
                    )
                if array.size and np.any(array >= limits[head]):
                    raise ValueError(
                        f"page {page_id!r}, head {head}: leaf ID outside "
                        f"num_leaves={limits[head]}"
                    )
                if np.unique(array).size != array.size:
                    raise ValueError(
                        f"page {page_id!r}, head {head}: duplicate leaf IDs in "
                        "one page result"
                    )
            resolved[(page_id, head)] = np.ascontiguousarray(
                array, dtype=np.int32
            )
        if missing and is_page_ready is not None and is_page_ready(page_id):
            # Materialized, but this layer has no map (e.g. invalidated).  The
            # page contributes nothing here; deferring it would try to encode
            # an already-encoded page.
            for head in range(n_heads):
                resolved.setdefault((page_id, head), empty)
            missing = 0
        page_ready[page_id] = missing == 0

    if not any(page_ready.values()):
        # Nothing ready: the merged result is exactly the validated fallback,
        # with every selected page deferred.  Skip the merge loop (the decode
        # hot path while a page waits for its boundary materialization).
        metrics = {
            "n_ready_rag_pages": 0.0,
            "n_deferred_rag_pages": float(len(all_page_ids)),
            "n_rag_leaves_used": 0.0,
            "n_fallback_leaves_used": float(n_heads * width),
            "n_kv_heads": float(n_heads),
            "width": float(width),
        }
        return CandidatePlan._from_validated(
            ready_leaf_ids=fallback,
            deferred_rag_pages=tuple(all_page_ids),
            fallback_leaf_ids=fallback,
            selection_metrics=metrics,
        )

    leaf_bounds: tuple[tuple[int, int] | None, ...] | None = None
    if trusted_resolver:
        # Maps are duplicate-free by construction, but their recorded limits
        # can be stale if the DCI index was rebuilt smaller.  Check each head
        # once for all ready pages instead of once per page.  The bounds are
        # kept for the prefix cache, which re-checks them against the current
        # limits on every hit (a DCI rebuild does not touch the materializer
        # epoch).
        bounds: list[tuple[int, int] | None] = []
        for head in range(n_heads):
            arrays = [
                resolved[(page_id, head)]
                for page_id in all_page_ids
                if page_ready[page_id] and (page_id, head) in resolved
            ]
            if not arrays:
                bounds.append(None)
                continue
            joined = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
            if joined.size and (
                int(joined.min()) < 0 or int(joined.max()) >= limits[head]
            ):
                raise ValueError(
                    f"head {head}: trusted leaf outside [0, {limits[head]})"
                )
            bounds.append(
                None if not joined.size else (int(joined.min()), int(joined.max()))
            )
        leaf_bounds = tuple(bounds)

    deferred: list[str] = []
    deferred_seen: set[str] = set()
    ready_rows = np.empty((n_heads, width), dtype=np.int32)
    # Built lazily: wide heads take the vectorized fill and never need these.
    fallback_lists: list[list[int]] | None = None
    rag_leaves_used = 0
    fallback_leaves_used = 0
    # True while every head is filled by ready leaves alone: only then is the
    # merged row independent of this token's fallback and cacheable.
    all_full_ready = True
    for head in range(n_heads):
        ready_arrays: list[np.ndarray] = []
        for page_id in per_head_ids[head]:
            if not page_ready[page_id]:
                if page_id not in deferred_seen:
                    deferred_seen.add(page_id)
                    deferred.append(page_id)
                continue
            ready_arrays.append(resolved[(page_id, head)])
        total_ready = sum(array.size for array in ready_arrays)
        # The Python loop stops once ``width`` uniques are collected, so the
        # vectorized path only pays off while the ready maps stay close to the
        # output width; far above it the loop visits roughly ``width`` leaves
        # and wins.  Measured crossover ~1.5x width (n=1024 vs width=512 is a
        # tie, n=512 vs 512 is ~2x for the vector).
        if _VECTOR_FILL_MIN <= total_ready <= (3 * width) // 2:
            # Same ordered first-occurrence de-duplication as the set loop:
            # ``unique`` gives first indices in flattened (page) order, so
            # sorting by them restores page order before the width truncation.
            if ready_arrays:
                joined = np.concatenate(ready_arrays)
                unique, first = np.unique(joined, return_index=True)
                chosen = unique[np.argsort(first, kind="stable")][:width].tolist()
            else:
                chosen = []
            chosen_set = set(chosen)
        else:
            chosen = []
            chosen_set = set()
            for array in ready_arrays:
                for leaf in array.tolist():
                    if len(chosen) >= width:
                        break
                    if leaf not in chosen_set:
                        chosen_set.add(leaf)
                        chosen.append(leaf)
        rag_leaves_used += len(chosen)
        before = len(chosen)
        need = width - before
        if need > 0:
            all_full_ready = False
        if need > 0 and need >= _VECTOR_FILL_MIN:
            # Wide heads (num_neighbours grows with the page budget) make the
            # Python fill loop the dominant plan cost.  Fallback rows are
            # duplicate-free (validated above), so removing the ready leaves
            # with one bool mask is exactly the ordered first-`need` fill.
            # The mask costs O(num_leaves); cap it so a huge DCI index does
            # not allocate a fresh mask per head per layer per token.
            limit = int(limits[head])
            if limit <= _VECTOR_FILL_MAX_LEAVES:
                seen = np.zeros(limit, dtype=bool)
                seen[np.asarray(chosen, dtype=np.int32)] = True
                row = fallback[head]
                chosen.extend(row[~seen[row]][:need].tolist())
            else:
                if fallback_lists is None:
                    fallback_lists = [row.tolist() for row in fallback]
                for leaf in fallback_lists[head]:
                    if len(chosen) >= width:
                        break
                    if leaf not in chosen_set:
                        chosen_set.add(leaf)
                        chosen.append(leaf)
        elif need > 0:
            if fallback_lists is None:
                fallback_lists = [row.tolist() for row in fallback]
            for leaf in fallback_lists[head]:
                if len(chosen) >= width:
                    break
                if leaf not in chosen_set:
                    chosen_set.add(leaf)
                    chosen.append(leaf)
        if len(chosen) < width:
            raise ValueError(
                f"head {head}: only {len(chosen)} distinct leaves available for "
                f"width {width}; fallback must contain enough unique leaves"
            )
        fallback_leaves_used += len(chosen) - before
        ready_rows[head] = chosen[:width]

    if not trusted_resolver:
        ready_rows = validate_leaf_ids(
            ready_rows,
            n_kv_heads=n_heads,
            num_neighbours=width,
            num_leaves=limits,
        )
    ready_pages = sum(1 for page_id in all_page_ids if page_ready[page_id])
    if cache_key is not None and all_full_ready:
        # Store the same immutable representation the hit path reuses; a
        # plain ``writeable=False`` array could be re-enabled by a consumer.
        rows = _readonly(np.asarray(ready_rows, dtype=np.int32))
        prefix_cache.store(
            cache_key,
            _PlanPrefixEntry(
                rows=rows,
                deferred=tuple(deferred),
                ready_pages=ready_pages,
                rag_leaves_used=rag_leaves_used,
                bounds=leaf_bounds,
            ),
            epoch=cache_epoch,
        )
    metrics = {
        "n_ready_rag_pages": float(ready_pages),
        "n_deferred_rag_pages": float(len(deferred)),
        "n_rag_leaves_used": float(rag_leaves_used),
        "n_fallback_leaves_used": float(fallback_leaves_used),
        "n_kv_heads": float(n_heads),
        "width": float(width),
    }
    return CandidatePlan._from_validated(
        ready_leaf_ids=ready_rows,
        deferred_rag_pages=tuple(deferred),
        fallback_leaf_ids=fallback,
        selection_metrics=metrics,
    )


# ---------------------------------------------------------------------------
# In-module self-test (kept in the source tree by project convention)
# ---------------------------------------------------------------------------


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

    # -- Global pages and store -----------------------------------------
    def descriptor(tokens: np.ndarray) -> np.ndarray:
        # Deterministic fake embedding with no learned semantics.
        vec = np.array(
            [tokens.sum() % 7 + 1.0, (tokens.size % 5) + 1.0, 2.0],
            dtype=np.float64,
        )
        return vec

    tokens = np.arange(10, dtype=np.int64) + 100
    pages = chunk_document(
        tokens,
        document_id="docA",
        chunk_size=4,
        descriptor_version="v1",
        descriptor_fn=descriptor,
    )
    ok(len(pages) == 3, "chunk count")
    ok([p.n_tokens for p in pages] == [4, 4, 2], "chunk lengths")
    ok(pages[0].segment_id == "seg00000000_00000004", "stable segment id")
    ok(pages[0].page_id == "docA/seg00000000_00000004", "stable page id")
    ok(np.array_equal(pages[1].token_ids, tokens[4:8]), "chunk boundaries")
    ok(pages[0].text_hash != pages[1].text_hash, "distinct chunk hashes")
    ok(not pages[0].token_ids.flags.writeable, "token ids frozen")
    ok(not pages[0].descriptor.flags.writeable, "descriptor frozen")
    # A read-only view of a writable buffer can be re-opened; the stored arrays
    # are bytes-backed so the store's metadata cannot be mutated in place.
    raises(ValueError, setattr, pages[0].token_ids.flags, "writeable", True)
    raises(ValueError, setattr, pages[0].descriptor.flags, "writeable", True)

    store = GlobalRAGPageStore()
    added = store.add_many(pages)
    ok(len(added) == 3 and len(store) == 3, "store add_many")
    ok(store.descriptor_version == "v1", "store pins descriptor version")
    ok(store.descriptor_dim == 3, "store pins descriptor dim")
    ok(store.add(pages[0]) is pages[0], "re-add identical page is idempotent")
    ok(store.get(pages[0].page_id) == pages[0], "store get")
    ok(store.page_ids() == tuple(sorted(p.page_id for p in pages)), "sorted ids")

    conflicting = GlobalRAGPage(
        page_id=pages[0].page_id,
        document_id="docA",
        segment_id=pages[0].segment_id,
        token_ids=np.arange(4),
        descriptor=pages[0].descriptor,
        descriptor_version="v1",
        text_hash="other",
    )
    raises(ValueError, store.add, conflicting)
    bad_version = GlobalRAGPage(
        page_id="docB/seg",
        document_id="docB",
        segment_id="seg",
        token_ids=np.array([1, 2]),
        descriptor=np.ones(3),
        descriptor_version="v2",
        text_hash="h",
    )
    raises(ValueError, store.add, bad_version)
    bad_dim = GlobalRAGPage(
        page_id="docC/seg",
        document_id="docC",
        segment_id="seg",
        token_ids=np.array([1, 2]),
        descriptor=np.ones(4),
        descriptor_version="v1",
        text_hash="h",
    )
    raises(ValueError, store.add, bad_dim)
    raises(ValueError, GlobalRAGPage, page_id="", document_id="d", segment_id="s",
           token_ids=np.array([1]), descriptor=None, descriptor_version="v1",
           text_hash="h")
    raises(TypeError, GlobalRAGPage, page_id="d/seg", document_id="d",
           segment_id="seg", token_ids=np.array([1]),
           descriptor=np.ones(3, dtype=np.complex128), descriptor_version="v1",
           text_hash="h")
    raises(ValueError, GlobalRAGPage, page_id="d/seg", document_id="d",
           segment_id="seg", token_ids=np.array([2**63], dtype=np.uint64),
           descriptor=None, descriptor_version="v1", text_hash="h")

    # Search: page 0 (sum 406) and page 1 (sum 422) produce distinct vectors.
    query = descriptor(tokens[:4])
    hits = store.search(query, top_k=3)
    ok(len(hits) == 3, "search returns k")
    ok(hits[0][0].page_id == pages[0].page_id, "search finds query chunk first")
    ok(all(-1.0 - 1e-9 <= score <= 1.0 + 1e-9 for _, score in hits), "cosine range")
    ok(store.search(query, top_k=0) == [], "top_k zero")
    raises(ValueError, store.search, np.ones(2), top_k=1)
    raises(ValueError, store.search, np.zeros(3), top_k=1)
    raises(KeyError, store.search, query, top_k=1,
           allowed_page_ids=["missing/page"])
    only = store.search(query, top_k=3, allowed_page_ids=[pages[2].page_id])
    ok(len(only) == 1 and only[0][0].page_id == pages[2].page_id, "search filter")

    # Descriptor norms are cached once per immutable page.  The cache must
    # agree exactly with a fresh computation, a missing entry must recompute
    # to the same scores, and remove/re-add must keep it in step.
    ok(len(store._page_norms) == len(store),
       "descriptor norms cached for every page")
    ok(all(store._page_norms[page.page_id]
           == float(np.linalg.norm(page.descriptor)) for page in pages),
       "cached norms equal a direct computation")
    baseline = [(page.page_id, score) for page, score in store.search(query, top_k=3)]
    store._page_norms.pop(pages[0].page_id)
    ok([(page.page_id, score) for page, score in store.search(query, top_k=3)]
       == baseline, "missing cached norm recomputes identical scores")
    ok(pages[0].page_id in store._page_norms, "recomputed norm is cached")
    removed = store.remove(pages[2].page_id)
    ok(removed is not None and pages[2].page_id not in store._page_norms,
       "remove drops the cached norm")
    store.add(pages[2])
    ok(store._page_norms[pages[2].page_id]
       == float(np.linalg.norm(pages[2].descriptor)), "re-add recreates the norm")

    # Large stores take the stacked-matvec path.  Its ranking must match an
    # independent per-page computation, the allowed-ID filter must agree, and
    # add/remove must drop the cached matrix.
    big_rng = np.random.default_rng(99)
    big_dim = 256
    big_pages = [
        GlobalRAGPage(
            page_id=f"big/{index:04d}",
            document_id="big",
            segment_id=f"seg{index}",
            token_ids=np.array([index, index + 1], dtype=np.int64),
            descriptor=big_rng.standard_normal(big_dim),
            descriptor_version="v1",
            text_hash=f"h{index}",
        )
        for index in range(300)
    ]
    big = GlobalRAGPageStore()
    big.add_many(big_pages)
    ok(len(big) * big_dim >= _SEARCH_MATRIX_MIN_ELEMENTS,
       "big store qualifies for the stacked path")
    big_query = big_rng.standard_normal(big_dim)
    big_norm = float(np.linalg.norm(big_query))

    def big_reference(allowed_ids=None):
        scored_ref = [
            (
                page,
                float(np.dot(big_query, page.descriptor))
                / (big_norm * big._page_norms[page.page_id]),
            )
            for page in big.pages()
            if page.descriptor is not None
            and (allowed_ids is None or page.page_id in allowed_ids)
        ]
        scored_ref.sort(key=lambda item: (-item[1], item[0].page_id))
        return scored_ref

    got = big.search(big_query, top_k=5)
    want = big_reference()[:5]
    ok([page.page_id for page, _ in got] == [page.page_id for page, _ in want],
       "stacked search ranking")
    ok(all(abs(a[1] - b[1]) <= 1e-12 for a, b in zip(got, want)),
       "stacked search scores")
    ok(big._search_cache is not None, "stacked matrix is cached")
    allowed_ids = [big_pages[index].page_id for index in range(0, 300, 7)]
    got_allowed = big.search(big_query, top_k=4, allowed_page_ids=allowed_ids)
    want_allowed = big_reference(set(allowed_ids))[:4]
    ok([page.page_id for page, _ in got_allowed]
       == [page.page_id for page, _ in want_allowed],
       "stacked search allowed filter")
    big.remove(big_pages[0].page_id)
    ok(big._search_cache is None, "remove invalidates the stacked matrix")
    ok(all(page.page_id != big_pages[0].page_id
           for page, _ in big.search(big_query, top_k=5)),
       "removed page is gone from stacked search")
    newest = big.search(big_pages[1].descriptor, top_k=1)
    ok(newest[0][0].page_id == big_pages[1].page_id, "stacked exact hit")
    added_descriptor = big_rng.standard_normal(big_dim)
    big.add(GlobalRAGPage(
        page_id="big/new",
        document_id="big",
        segment_id="segnew",
        token_ids=np.array([999], dtype=np.int64),
        descriptor=added_descriptor,
        descriptor_version="v1",
        text_hash="hnew",
    ))
    ok(big._search_cache is None, "add invalidates the stacked matrix")
    ok(big.search(added_descriptor, top_k=1)[0][0].page_id == "big/new",
       "added page is found through a rebuilt matrix")

    # Prompt assembly records absolute page spans.
    assembled = assemble_prompt(
        pages=pages,
        prefix_token_ids=np.array([7, 8]),
        suffix_token_ids=np.array([9]),
    )
    ok(assembled.n_tokens == 13, "assembled length")
    ok(assembled.span_for(pages[0].page_id).token_start == 2, "first span start")
    ok(assembled.span_for(pages[0].page_id).token_end == 6, "first span end")
    ok(assembled.span_for(pages[2].page_id).token_start == 10, "last span start")
    ok(assembled.token_ids[10] == tokens[8], "assembled content")
    raises(ValueError, assemble_prompt, pages=[pages[0], pages[0]])
    raises(KeyError, assembled.span_for, "nope")

    # Overlapping chunks remain stable and avoid a fully contained tail.
    overlap_pages = chunk_document(
        tokens, document_id="docO", chunk_size=4, overlap=2,
        descriptor_version="v1",
    )
    ok([p.n_tokens for p in overlap_pages] == [4, 4, 4, 4], "overlap chunks")
    ok(overlap_pages[0].page_id != overlap_pages[1].page_id, "overlap ids unique")

    # -- State machine ---------------------------------------------------
    mat = RAGMaterializer(store, n_kv_heads=2)
    q = mat.submit("s1", [pages[0].page_id, pages[1].page_id])
    ok(q == [pages[0].page_id, pages[1].page_id], "submit queues both")
    ok(mat.submit("s1", [pages[0].page_id]) == [], "submit is idempotent")
    ok(mat.poll_ready("s1") == [], "queued pages are not ready")
    ok(mat.state_counts("s1") == {"queued": 2}, "queued counts")

    mat.begin_encoding("s1", [pages[0].page_id], token_start=2, token_end=6)
    rec = mat.record("s1", pages[0].page_id)
    ok(rec.state == "encoding" and rec.token_start == 2 and rec.token_end == 6,
       "begin encoding records span")
    raises(ValueError, mat.begin_encoding, "s1", [pages[0].page_id],
           token_start=2, token_end=6)

    # token2node: 2 heads, 12 DCI tokens.  Chunk 0 is DCI tokens [0, 4).
    node_index = np.array(
        [
            [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
            [5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8],
        ],
        dtype=np.int32,
    )
    leaves = np.array([5, 9], dtype=np.int32)
    mapping = mat.register_layer_leaves(
        "s1", pages[0].page_id, 0, node_index, leaves, token_offset=2
    )
    ok(mapping.shape == (2, 2), "leaf map shape")
    ok(sorted(mapping[0].tolist()) == [0, 1], "head 0 leaves")
    ok(sorted(mapping[1].tolist()) == [5, 6], "head 1 leaves")
    ok(set(mapping[0].tolist()).issubset({0, 1}), "head 0 membership")
    ok(set(mapping[1].tolist()).issubset({5, 6}), "head 1 membership")
    # Acceptance: every returned leaf's members contain a chunk token.
    for head in range(2):
        for leaf in mapping[head]:
            members = np.flatnonzero(node_index[head, 0:4] == leaf)
            ok(members.size > 0, f"head {head} leaf {leaf} has a chunk token")
    raises(ValueError, mat.register_layer_leaves, "s1", pages[0].page_id, 0,
           node_index, leaves, token_offset=2)
    # prompt span [2, 6) shifted by 3 maps to [-1, 3): not fully indexed.
    raises(ValueError, mat.register_layer_leaves, "s1", pages[0].page_id, 1,
           node_index, leaves, token_offset=3, require_full_span=True)
    empty_map = mat.register_layer_leaves(
        "s1", pages[0].page_id, 2, node_index, leaves, token_offset=6,
        require_full_span=False,
    )
    ok(empty_map.shape == (2, 0), "fully outside span yields empty map")
    # A map with the wrong number of KV heads is rejected, and the failure does
    # not replace the already-registered layer 0 map.
    raises(ValueError, mat.register_layer_leaves, "s1", pages[0].page_id, 1,
           node_index[0], 1, token_offset=2)
    ok(mat.resolve_native_leaves("s1", pages[0].page_id, 1) is None,
       "rejected head-count map is not stored")
    # Internal fast path: stores a map built by map_span_to_leaves verbatim,
    # but refuses a mutable array (a caller could otherwise edit a registered
    # map in place).
    validated_map = _readonly(np.array([[0, 1], [5, 6]], dtype=np.int32))
    stored = mat.record("s1", pages[0].page_id)._register_leaf_map_validated(
        7, validated_map
    )
    ok(stored.tolist() == [[0, 1], [5, 6]], "validated fast path stores map")
    raises(ValueError, mat.record("s1", pages[0].page_id)
           ._register_leaf_map_validated, 8,
           np.array([[0], [5]], dtype=np.int32))
    raises(TypeError, map_span_to_leaves, node_index,
           token_start=0, token_end=4, num_leaves=np.array([5.0, 9.0]))
    # A partially indexed chunk can be clipped explicitly: prompt span [2,6)
    # shifted by 4 is [-2,2), so only DCI tokens [0,2) contribute.
    clipped = mat.register_layer_leaves(
        "s1", pages[0].page_id, 3, node_index, leaves, token_offset=4,
        require_full_span=False,
    )
    ok(clipped.shape == (2, 1), "clipped leaf map shape")
    ok(clipped[0, 0] == 0 and clipped[1, 0] == 5, "clipped leaf values")

    raises(ValueError, mat.register_layer_leaves, "s1", pages[1].page_id, 0,
           node_index, leaves, token_offset=2)  # not yet encoding
    mat.fail("s1", pages[0].page_id, "synthetic encode failure")
    failed = mat.record("s1", pages[0].page_id)
    ok(failed.state == "failed" and failed.error == "synthetic encode failure",
       "failure recorded")
    ok(failed.token_start is None and failed.layer_leaf_ids == {},
       "failure rolls back span and leaves")
    ok(mat.resolve_native_leaves("s1", pages[0].page_id, 0) is None,
       "failed page has no native leaves")

    # Retry is explicit and re-queues from a clean record.
    mat.retry("s1", pages[0].page_id)
    ok(mat.record("s1", pages[0].page_id).state == "queued", "retry requeues")
    ok(mat.record("s1", pages[0].page_id).error is None, "retry clears error")

    # Complete a page and promote/demote it.
    mat.begin_encoding("s1", [pages[0].page_id], token_start=2, token_end=6)
    mat.register_layer_leaves(
        "s1", pages[0].page_id, 0, node_index, leaves, token_offset=2
    )
    mat.complete_encoding("s1", [pages[0].page_id])
    ok(mat.poll_ready("s1") == [pages[0].page_id], "ready after completion")
    ok(mat.record("s1", pages[0].page_id).state == "cpu_kv", "cpu_kv state")
    mat.mark_gpu("s1", [pages[0].page_id])
    ok(mat.record("s1", pages[0].page_id).state == "gpu_kv", "gpu_kv state")
    mat.mark_cpu("s1", [pages[0].page_id])
    ok(mat.record("s1", pages[0].page_id).state == "cpu_kv", "demote cpu_kv")
    resolved = mat.resolve_native_leaves("s1", pages[0].page_id, 0)
    ok(resolved is not None and np.array_equal(resolved, mapping),
       "resolve after completion")
    ok(resolved.flags.writeable, "resolve_native_leaves returns a copy")
    direct = mat.resolve_page_leaves("s1", pages[0].page_id, 0)
    ok(direct is not None and np.array_equal(direct, mapping),
       "resolve_page_leaves values")
    ok(not direct.flags.writeable, "resolve_page_leaves is read-only")
    ok(mat.resolve_page_leaves("s1", pages[0].page_id, 99) is None,
       "unregistered layer has no direct map")
    ok(mat.resolve_native_leaves("s1", pages[0].page_id, 99) is None,
       "unregistered layer resolves None")
    head0 = mat.resolve_head_leaves("s1", pages[0].page_id, 0, 0)
    ok(sorted(head0.tolist()) == [0, 1], "head resolver leaves")
    raises(ValueError, mat.resolve_head_leaves, "s1", pages[0].page_id, 0, 2)
    raises(ValueError, mat.resolve_head_leaves, "s1", pages[0].page_id, 0, -1)
    raises(ValueError, mat.resolve_native_leaves, "s1", pages[0].page_id, -1)
    # The plan-bound resolver batches a page's map into per-head rows.  It
    # must agree with the per-head accessor, including the empty/bounds cases.
    bound = mat.head_resolver("s1", 0)
    for head_idx in range(2):
        ok(np.array_equal(
            bound(pages[0].page_id, head_idx),
            mat.resolve_head_leaves("s1", pages[0].page_id, 0, head_idx),
        ), f"bound resolver head {head_idx} matches")
    ok(bound("missing/page", 0) is None, "bound resolver unknown page")
    raises(ValueError, bound, pages[0].page_id, 2)
    raises(ValueError, bound, pages[0].page_id, -1)
    raises(ValueError, mat.head_resolver, "s1", -1)
    ok(mat.head_resolver("s1", 9)(pages[0].page_id, 0).size == 0,
       "bound resolver without a layer map contributes zero leaves")
    # The cache is per plan: a re-registration is picked up by the next
    # resolver, while an already-bound plan stays internally consistent.
    stale_resolver = mat.head_resolver("s1", 0)
    stale_head0 = stale_resolver(pages[0].page_id, 0).tolist()
    mat.register_layer_leaves(
        "s1", pages[0].page_id, 0, node_index, leaves, token_offset=4,
        require_full_span=False, replace=True,
    )
    ok(stale_resolver(pages[0].page_id, 0).tolist() == stale_head0,
       "bound resolver keeps the map it first saw")
    fresh = mat.head_resolver("s1", 0)
    ok(fresh(pages[0].page_id, 0).tolist() == [0],
       "a new resolver sees the re-registered map")
    # Restore the original layer-0 map for the following checks.
    mat.register_layer_leaves(
        "s1", pages[0].page_id, 0, node_index, leaves, token_offset=2,
        replace=True,
    )
    ok(np.array_equal(
        mat.resolve_head_leaves("s1", pages[0].page_id, 0, 0),
        np.array([0, 1], dtype=np.int32),
    ), "restored layer-0 map")
    # A subclass that customizes per-head resolution keeps the plain path:
    # batching bypasses its accessor and could change the leaves.
    class _Counting(RAGMaterializer):
        calls = 0

        def resolve_head_leaves(self, *args):
            type(self).calls += 1
            return super().resolve_head_leaves(*args)

    counting = _Counting(store, n_kv_heads=2)
    counting.submit("cnt", [pages[0].page_id])
    counting.begin_encoding("cnt", [pages[0].page_id], token_start=2,
                            token_end=6)
    counting.register_layer_leaves("cnt", pages[0].page_id, 0, node_index,
                                   leaves, token_offset=2)
    counting.complete_encoding("cnt", [pages[0].page_id])
    bound_counting = counting.head_resolver("cnt", 0)
    bound_counting(pages[0].page_id, 0)
    bound_counting(pages[0].page_id, 1)
    ok(_Counting.calls == 2,
       "overridden resolve_head_leaves keeps the per-head path")
    counting.release_session("cnt")
    raises(ValueError, mat.complete_encoding, "s1", [pages[0].page_id])
    mat.touch("s1", [pages[0].page_id], step=7)
    ok(mat.record("s1", pages[0].page_id).last_access_step == 7, "touch step")

    # A DCI add may reassign leaves: dropping one layer's map must leave the
    # page's KV state and the other layers intact.
    mat.submit("inval", [pages[0].page_id])
    mat.begin_encoding("inval", [pages[0].page_id], token_start=2, token_end=6)
    mat.register_layer_leaves(
        "inval", pages[0].page_id, 0, node_index, leaves, token_offset=2
    )
    mat.register_layer_leaves(
        "inval", pages[0].page_id, 1, node_index, leaves, token_offset=2
    )
    mat.complete_encoding("inval", [pages[0].page_id])
    ok(mat.invalidate_layer("inval", 0) == 1, "invalidate drops one layer")
    ok(mat.resolve_native_leaves("inval", pages[0].page_id, 0) is None,
       "invalidated layer resolves None")
    ok(mat.resolve_native_leaves("inval", pages[0].page_id, 1) is not None,
       "other layers survive invalidation")
    ok(mat.record("inval", pages[0].page_id).state == "cpu_kv",
       "invalidation keeps the KV state")
    ok(mat.record("inval", pages[0].page_id).token_start == 2,
       "invalidation keeps the span")
    ok(mat.invalidate_layer("inval", 0) == 0, "double invalidation is a no-op")
    ok(mat.invalidate_layer("other", 0) == 0, "other sessions unaffected")
    raises(ValueError, mat.invalidate_layer, "inval", -1)
    mat.register_layer_leaves(
        "inval", pages[0].page_id, 0, node_index, leaves, token_offset=2
    )
    ok(np.array_equal(
        mat.resolve_native_leaves("inval", pages[0].page_id, 0), mapping
    ), "re-registration restores the layer")
    mat.release_session("inval")

    # Snapshot copies are defensive: distinct objects, read-only, equal values.
    snap_a = mat.record("s1", pages[0].page_id)
    snap_b = mat.record("s1", pages[0].page_id)
    ok(snap_a is not snap_b, "each record call returns a new snapshot")
    ok(
        snap_a.layer_leaf_ids[0] is not snap_b.layer_leaf_ids[0],
        "snapshot leaf arrays are copies",
    )
    ok(
        np.array_equal(snap_a.layer_leaf_ids[0], snap_b.layer_leaf_ids[0]),
        "snapshot values agree",
    )
    ok(not snap_a.layer_leaf_ids[0].flags.writeable, "snapshot arrays are read-only")

    # Duplicate IDs, mid-row padding, and int32 overflow inside a leaf map are
    # rejected.
    raises(ValueError, _validate_leaf_map, np.array([[1, 1], [2, 3]]))
    raises(ValueError, _validate_leaf_map, np.array([[1, -1, 2], [2, 3, -1]]))
    raises(ValueError, _validate_leaf_map, np.array([[2**31]], dtype=np.int64))
    # int64 values that wrap into a valid int32 leaf must be rejected pre-cast.
    raises(ValueError, _validate_leaf_map, np.array([[2**32]], dtype=np.int64))
    raises(ValueError, _validate_leaf_map, np.array([[-2**32]], dtype=np.int64))
    raises(ValueError, map_span_to_leaves, node_index,
           token_start=0, token_end=4,
           num_leaves=np.array([2**31, 9], dtype=np.int64))
    raises(ValueError, map_span_to_leaves, node_index,
           token_start=0, token_end=4, num_leaves=np.array([1, 9]))
    raises(ValueError, map_span_to_leaves, node_index,
           token_start=0, token_end=13, num_leaves=leaves)
    raises(ValueError, map_span_to_leaves, node_index,
           token_start=3, token_end=3, num_leaves=leaves)
    raises(ValueError, map_span_to_leaves,
           np.array([[2**63]], dtype=np.uint64),
           token_start=0, token_end=1, num_leaves=leaves)
    unmapped = np.full((2, 6), -1, dtype=np.int32)
    ok(map_span_to_leaves(unmapped, token_start=1, token_end=5,
                          num_leaves=leaves).shape == (2, 0),
       "unmapped span yields empty map")
    tuple_form = (node_index, np.zeros(12, dtype=np.int32))
    ok(np.array_equal(
        map_span_to_leaves(tuple_form, token_start=0, token_end=4,
                           num_leaves=leaves), mapping
    ), "token2node tuple form")
    # Two same-length 1-D arrays are indistinguishable from a pair of
    # per-head rows, so the flat tuple form fails closed.
    raises(ValueError, map_span_to_leaves,
           (node_index[0], np.zeros(12, dtype=np.int32)),
           token_start=0, token_end=4, num_leaves=leaves)

    # A DCI add can reorganize leaves; re-registration must be explicit.
    new_nodes = node_index.copy()
    new_nodes[0, 0:4] = 3
    replaced = mat.register_layer_leaves(
        "s1", pages[0].page_id, 0, new_nodes, leaves, token_offset=2,
        replace=True,
    )
    ok(replaced[0, 0] == 3 and replaced[0, 1] == -1,
       "replace re-registers layer leaves")
    replacement = mat.resolve_native_leaves("s1", pages[0].page_id, 0)
    ok(replacement is not None and replacement[0, 0] == 3,
       "replacement is visible to resolution")

    # -- Session isolation ----------------------------------------------
    q2 = mat.submit("s2", [pages[0].page_id])
    ok(q2 == [pages[0].page_id], "second session queues independently")
    ok(mat.record("s1", pages[0].page_id).state == "cpu_kv", "s1 still ready")
    ok(mat.record("s2", pages[0].page_id).state == "queued", "s2 independent")
    mat.begin_encoding("s2", [pages[0].page_id], token_start=2, token_end=6)
    mat.register_layer_leaves(
        "s2", pages[0].page_id, 0, node_index, leaves, token_offset=2
    )
    mat.complete_encoding("s2", [pages[0].page_id])
    ok(mat.resolve_native_leaves("s2", pages[0].page_id, 0) is not None,
       "s2 resolves its own KV")
    ok(mat.record("s1", pages[0].page_id).state == "cpu_kv",
       "s1 unaffected by s2 materialization")
    mat.fail("s2", pages[0].page_id, "s2 failure")
    ok(mat.record("s1", pages[0].page_id).state == "cpu_kv",
       "s1 unaffected by s2 failure")
    ok(mat.resolve_native_leaves("s2", pages[0].page_id, 0) is None,
       "s2 failure is local")
    ok(len(store) == 3, "materialization never mutates the global store")

    released = mat.release_session("s1")
    ok(released == 2, "release_session drops all records")
    ok(mat.records("s1") == {}, "s1 records cleared")
    ok(pages[0].page_id in store, "global page survives session release")
    ok(mat.resolve_native_leaves("s1", pages[0].page_id, 0) is None,
       "released session cannot resolve")
    ok(mat.record("s2", pages[0].page_id).state == "failed",
       "other session survives release")
    mat.release_session("s2")
    ok(len(store) == 3, "global store untouched after all releases")
    raises(KeyError, mat.record, "s1", pages[0].page_id)
    raises(ValueError, mat.submit, "", [pages[0].page_id])
    raises(ValueError, mat.submit, "sX", [])
    raises(KeyError, mat.submit, "sX", ["missing/page"])
    raises(TypeError, mat.submit, "sX", pages[0].page_id)
    raises(ValueError, mat.submit, "sX", [pages[0].page_id, pages[0].page_id])

    # Invalid transitions.
    mat.submit("s3", [pages[0].page_id])
    raises(ValueError, mat.complete_encoding, "s3", [pages[0].page_id])
    raises(ValueError, mat.mark_gpu, "s3", [pages[0].page_id])
    mat.begin_encoding("s3", [pages[0].page_id], token_start=2, token_end=6)
    raises(ValueError, mat.begin_encoding, "s3", [pages[0].page_id],
           token_start=2, token_end=6)
    mat.complete_encoding("s3", [pages[0].page_id])
    raises(ValueError, mat.retry, "s3", pages[0].page_id)
    mat.release_session("s3")

    # An interrupted encoding must be recoverable: retry clears its stale span
    # and leaf maps and returns it to ``queued``.
    mat.submit("s3b", [pages[0].page_id])
    mat.begin_encoding("s3b", [pages[0].page_id], token_start=2, token_end=6)
    mat.retry("s3b", pages[0].page_id)
    interrupted = mat.record("s3b", pages[0].page_id)
    ok(interrupted.state == "queued", "retry re-queues interrupted encoding")
    ok(interrupted.token_start is None and interrupted.layer_leaf_ids == {},
       "retry clears the stale span and maps")
    ok(mat.page_is_ready("s3b", pages[0].page_id) is False,
       "queued page is not ready")
    mat.release_session("s3b")
    # token_offset has no default: relying on 0 would silently mis-map.
    raises(TypeError, mat.register_layer_leaves, "sX", pages[0].page_id, 0,
           node_index, leaves)

    # Per-page spans support several retrieved chunks at different positions.
    mat.submit("spans", [pages[0].page_id, pages[1].page_id])
    mat.begin_encoding(
        "spans",
        [pages[0].page_id, pages[1].page_id],
        token_spans={
            pages[0].page_id: (0, 4),
            pages[1].page_id: (4, 8),
        },
    )
    ok(mat.record("spans", pages[0].page_id).token_start == 0, "span page 0")
    ok(mat.record("spans", pages[0].page_id).token_end == 4, "span page 0 end")
    ok(mat.record("spans", pages[1].page_id).token_start == 4, "span page 1")
    raises(ValueError, mat.begin_encoding, "spans",
           [pages[0].page_id, pages[1].page_id],
           token_spans={pages[0].page_id: (0, 4)})
    raises(ValueError, mat.begin_encoding, "spans",
           [pages[0].page_id, pages[1].page_id], token_start=0, token_end=4,
           token_spans={pages[0].page_id: (0, 4),
                        pages[1].page_id: (4, 8)})
    raises(ValueError, mat.begin_encoding, "spans", [pages[0].page_id],
           token_start=0, token_end=4)  # already encoding
    mat.release_session("spans")

    # The recorded span must equal the global page's token count.  A shared
    # span is invalid as soon as the retrieved chunks have unequal lengths
    # (e.g. the final, shorter chunk from chunk_document).
    mat.submit("badspan", [pages[0].page_id, pages[2].page_id])
    raises(ValueError, mat.begin_encoding, "badspan",
           [pages[0].page_id, pages[2].page_id], token_start=0, token_end=4)
    ok(mat.record("badspan", pages[0].page_id).state == "queued",
       "span mismatch does not mutate page 0")
    ok(mat.record("badspan", pages[2].page_id).state == "queued",
       "span mismatch does not mutate page 2")
    raises(ValueError, mat.begin_encoding, "badspan", [pages[0].page_id],
           token_start=0, token_end=10)
    mat.begin_encoding(
        "badspan",
        [pages[0].page_id, pages[2].page_id],
        token_spans={pages[0].page_id: (0, 4), pages[2].page_id: (4, 6)},
    )
    ok(mat.record("badspan", pages[2].page_id).token_end == 6,
       "per-page spans accept unequal chunk lengths")
    mat.release_session("badspan")

    # A batch transition validates every record before mutating any of them.
    mat.submit("atomic", [pages[0].page_id, pages[1].page_id])
    mat.begin_encoding("atomic", [pages[0].page_id], token_start=2, token_end=6)
    raises(ValueError, mat.begin_encoding, "atomic",
           [pages[1].page_id, pages[0].page_id], token_start=2, token_end=6)
    ok(mat.record("atomic", pages[1].page_id).state == "queued",
       "failed batch does not partially encode")
    mat.release_session("atomic")

    # token_offset is per registration, not per session: staged DCI adds can
    # index different chunks at different offsets.  Span [4,8) at offset 1 is
    # DCI [3,7), so head 0 contributes leaves 1 and 2 (head 1: 6 and 7).
    mat.submit("offsets", [pages[0].page_id, pages[1].page_id])
    mat.begin_encoding(
        "offsets",
        [pages[0].page_id, pages[1].page_id],
        token_spans={pages[0].page_id: (0, 4), pages[1].page_id: (4, 8)},
    )
    off0 = mat.register_layer_leaves(
        "offsets", pages[0].page_id, 0, node_index, leaves, token_offset=0
    )
    off1 = mat.register_layer_leaves(
        "offsets", pages[1].page_id, 0, node_index, leaves, token_offset=1
    )
    ok(sorted(off0[0].tolist()) == [0, 1], "per-page offset zero")
    ok(sorted(off1[0].tolist()) == [1, 2], "per-page offset respects second page")
    ok(off1[1].tolist() == [6, 7], "per-page offset head 1")
    mat.complete_encoding("offsets", [pages[0].page_id, pages[1].page_id])
    ok(np.array_equal(
        mat.resolve_native_leaves("offsets", pages[1].page_id, 0), off1
    ), "per-page offset map resolves")
    mat.release_session("offsets")

    # A ready page with an empty (but registered) map is still ready.
    mat.submit("empty", [pages[0].page_id])
    mat.begin_encoding("empty", [pages[0].page_id], token_start=2, token_end=6)
    empty_ready = mat.register_layer_leaves(
        "empty", pages[0].page_id, 0, node_index, leaves, token_offset=6,
        require_full_span=False,
    )
    ok(empty_ready.shape == (2, 0), "empty map registered")
    mat.complete_encoding("empty", [pages[0].page_id])
    resolved_empty = mat.resolve_native_leaves("empty", pages[0].page_id, 0)
    ok(resolved_empty is not None and resolved_empty.shape == (2, 0),
       "ready page with empty map resolves")
    ok(mat.resolve_head_leaves("empty", pages[0].page_id, 0, 0).size == 0,
       "empty head leaves resolve")
    empty_resolver = mat.head_resolver("empty", 0)
    ok(empty_resolver(pages[0].page_id, 0).size == 0,
       "batched resolver handles an empty map (head 0)")
    ok(empty_resolver(pages[0].page_id, 1).size == 0,
       "batched resolver handles an empty map (head 1)")
    raises(ValueError, empty_resolver, pages[0].page_id, 2)
    mat.release_session("empty")

    # A flat token2node cannot tell one KV head from one row of a multi-head
    # map.  Refuse to infer (and permanently pin) a head count.
    flat_mat = RAGMaterializer(store)
    flat_mat.submit("flat", [pages[0].page_id])
    flat_mat.begin_encoding("flat", [pages[0].page_id], token_start=2, token_end=6)
    raises(ValueError, flat_mat.register_layer_leaves, "flat",
           pages[0].page_id, 0, node_index[0], leaves, token_offset=2)
    ok(flat_mat.n_kv_heads is None, "ambiguous flat map does not pin head count")
    flat_mat.release_session("flat")
    # With an explicit single head the same flat map is valid.
    one_mat = RAGMaterializer(store, n_kv_heads=1)
    one_mat.submit("one", [pages[0].page_id])
    one_mat.begin_encoding("one", [pages[0].page_id], token_start=2, token_end=6)
    one_map = one_mat.register_layer_leaves(
        "one", pages[0].page_id, 0, node_index[0], 5, token_offset=2
    )
    ok(one_map.shape == (1, 2), "explicit single-head flat map")
    ok(sorted(one_map[0].tolist()) == [0, 1], "single-head flat leaf values")
    one_mat.release_session("one")

    # Stage-A request-boundary helper: assemble first, then record spans.
    started = mat.begin_prompt(
        "promptq", assembled, [pages[0].page_id, pages[2].page_id]
    )
    ok(started == [pages[0].page_id, pages[2].page_id], "begin_prompt ids")
    ok(mat.record("promptq", pages[0].page_id).token_start == 2,
       "begin_prompt first span")
    ok(mat.record("promptq", pages[2].page_id).token_end == 12,
       "begin_prompt last span")
    ok(mat.record("promptq", pages[2].page_id).state == "encoding",
       "begin_prompt moves to encoding")
    raises(ValueError, mat.begin_prompt, "promptq2", assembled,
           ["missing/page"])
    raises(ValueError, mat.begin_prompt, "promptq2", assembled,
           [pages[0].page_id, pages[0].page_id])
    raises(TypeError, mat.begin_prompt, "promptq2", None)
    ok(mat.begin_prompt("promptq3", assembled, []) == [],
       "empty page selection is a no-op")
    ok(mat.records("promptq3") == {}, "empty begin_prompt creates no records")
    pure_prompt = assemble_prompt(pages=[], prefix_token_ids=[7, 8])
    ok(mat.begin_prompt("pureq", pure_prompt) == [],
       "prompt without RAG pages is a no-op")
    ok(mat.records("pureq") == {}, "pure prompt creates no records")

    # Stage-B append: the prompt stays in local coordinates, but the recorded
    # spans are absolute session positions so a later registration can use the
    # session's absolute DCI token_offset.
    append_started = mat.begin_prompt(
        "appendq", assembled, [pages[0].page_id], position_base=100
    )
    ok(append_started == [pages[0].page_id], "append begin ids")
    append_record = mat.record("appendq", pages[0].page_id)
    ok((append_record.token_start, append_record.token_end) == (102, 106),
       "position_base shifts recorded spans")
    raises(ValueError, mat.begin_prompt, "appendq2", assembled,
           [pages[0].page_id], position_base=-1)
    raises(TypeError, mat.begin_prompt, "appendq2", assembled,
           [pages[0].page_id], position_base=1.5)
    raises(TypeError, mat.begin_prompt, "appendq2", assembled,
           [pages[0].page_id], position_base=True)
    ok(mat.records("appendq2") == {},
       "invalid position_base creates no records")
    mat.release_session("appendq")

    # A ready page must not be silently reused for a second prompt: deep KV is
    # context-bound, and its leaf map may point at stale DCI leaves.
    ok(mat.complete_encoding("promptq", started) == started,
       "complete the first prompt")
    raises(ValueError, mat.begin_prompt,
           "promptq", assembled, [pages[0].page_id, pages[2].page_id])

    # A failed begin_prompt is atomic: pages it newly queued are rolled back,
    # while a page that was already encoding keeps its state.
    mat.begin_prompt("atomic", assembled, [pages[0].page_id])
    raises(ValueError, mat.begin_prompt, "atomic", assembled,
           [pages[0].page_id, pages[2].page_id])
    ok(mat.record("atomic", pages[0].page_id).state == "encoding",
       "atomic failure keeps the pre-existing encoding page")
    ok(mat.record("atomic", pages[2].page_id).state == "failed",
       "atomic failure rolls back the newly queued page")
    mat.release_session("atomic")
    mat.release_session("promptq")

    # A non-Exception (KeyboardInterrupt / CancelledError) landing during the
    # begin_encoding mutation loop must also roll back the pages this call
    # queued; otherwise they stay in ``encoding`` with no owning request.
    class _InterruptingBeginPrompt(RAGMaterializer):
        def begin_encoding(self, session_id, page_ids, **kwargs):
            super().begin_encoding(session_id, page_ids, **kwargs)
            raise KeyboardInterrupt("interrupted during begin_encoding")

    interrupting = _InterruptingBeginPrompt(store, n_kv_heads=2)
    try:
        interrupting.begin_prompt("ib", assembled, [pages[0].page_id])
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt from begin_prompt")
    ok(interrupting.record("ib", pages[0].page_id).state == "failed",
       "BaseException during begin_prompt is rolled back")

    # A page that was already ``queued`` before the call is moved
    # queued->encoding by begin_encoding; the interrupt must roll that back
    # too, or it stays encoding with no owning request and no helper state
    # (prepare raised before recording it) to abort.
    pre = _InterruptingBeginPrompt(store, n_kv_heads=2)
    pre.submit("ib2", [pages[1].page_id])
    ok(pre.record("ib2", pages[1].page_id).state == "queued",
       "pre-queued page starts queued")
    try:
        pre.begin_prompt("ib2", assembled, [pages[1].page_id])
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt from begin_prompt")
    ok(pre.record("ib2", pages[1].page_id).state == "failed",
       "pre-queued page is rolled back when begin_prompt is interrupted")

    # A BaseException delivered *inside* submit (after it queued pages but
    # before it returned) loses submit's return value.  The rollback must use
    # the pre-call snapshot instead, or those pages stay queued with no owner.
    class _InterruptingSubmit(RAGMaterializer):
        armed = False

        def submit(self, session_id, page_ids):
            out = super().submit(session_id, page_ids)
            if self.armed:
                raise KeyboardInterrupt("interrupted inside submit")
            return out

    inside = _InterruptingSubmit(store, n_kv_heads=2)
    inside.submit("ib4", [pages[2].page_id])
    inside.armed = True
    try:
        inside.begin_prompt(
            "ib4", assembled, [pages[1].page_id, pages[2].page_id]
        )
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - the interrupt is the point
        raise AssertionError("expected KeyboardInterrupt from submit")
    ok(inside.record("ib4", pages[1].page_id).state == "failed",
       "page queued by an interrupted submit is rolled back")
    ok(inside.record("ib4", pages[2].page_id).state == "queued",
       "pre-queued page is untouched when submit is interrupted")

    # A page this call did not start (already encoding for its session) is
    # left alone by the rollback; the newly queued page is still failed.
    owner = RAGMaterializer(store, n_kv_heads=2)
    owner.begin_prompt("ib3", assembled, [pages[0].page_id])
    raises(
        ValueError,
        owner.begin_prompt,
        "ib3",
        assembled,
        [pages[0].page_id, pages[1].page_id],
    )
    ok(owner.record("ib3", pages[0].page_id).state == "encoding",
       "rollback leaves a pre-existing encoding page untouched")
    ok(owner.record("ib3", pages[1].page_id).state == "failed",
       "rollback still fails the newly queued page")

    # -- CandidatePlan ---------------------------------------------------
    fallback = np.array([[0, 1], [5, 6]], dtype=np.int32)
    lim = np.array([5, 9], dtype=np.int32)

    def resolver(page_id: str, head: int):
        if page_id == pages[0].page_id:
            return np.array([1, 2], dtype=np.int32) if head == 0 else np.array(
                [6, 7], dtype=np.int32
            )
        if page_id == pages[1].page_id:
            return np.array([], dtype=np.int32)
        return None

    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[[pages[0].page_id], [pages[0].page_id]],
        resolve_leaves=resolver,
        num_leaves=lim,
    )
    ok(plan.ready_leaf_ids.shape == (2, 2), "plan shape")
    ok(plan.ready_leaf_ids[0].tolist() == [1, 2], "ready head 0")
    ok(plan.ready_leaf_ids[1].tolist() == [6, 7], "ready head 1")
    ok(plan.deferred_rag_pages == (), "nothing deferred")
    ok(plan.selection_metrics["n_ready_rag_pages"] == 1.0, "ready metric")
    ok(np.array_equal(plan.fallback_leaf_ids, fallback), "fallback preserved")
    validate_leaf_ids(plan.ready_leaf_ids, n_kv_heads=2, num_neighbours=2,
                      num_leaves=lim)
    validate_leaf_ids(plan.fallback_leaf_ids, n_kv_heads=2, num_neighbours=2,
                      num_leaves=lim)

    # The trusted fast path used by MaterializingCandidateProvider produces
    # the same ready/fallback arrays for materializer-built maps.
    trusted_plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[[pages[0].page_id], [pages[0].page_id]],
        resolve_leaves=resolver,
        num_leaves=lim,
        trusted_resolver=True,
    )
    ok(np.array_equal(trusted_plan.ready_leaf_ids, plan.ready_leaf_ids),
       "trusted ready matches untrusted")
    ok(np.array_equal(trusted_plan.fallback_leaf_ids, plan.fallback_leaf_ids),
       "trusted fallback matches untrusted")
    ok(trusted_plan.deferred_rag_pages == plan.deferred_rag_pages,
       "trusted deferred matches untrusted")

    # A resized index can shrink num_leaves below a recorded map; the trusted
    # path still checks every ready head against the current limits.
    raises(
        ValueError,
        build_candidate_plan,
        fallback_leaf_ids=np.array([[0], [0]], dtype=np.int32),
        selected_rag_by_head=[[pages[0].page_id], [pages[0].page_id]],
        resolve_leaves=resolver,
        num_leaves=np.array([1, 1], dtype=np.int32),
        trusted_resolver=True,
    )

    # Epoch-invalidated merge cache (decode hot path).  The ready leaves fill
    # width 2, so the merged rows do not depend on the fallback and may be
    # reused until the materialization state changes.
    pc = RAGMaterializer(store, n_kv_heads=2)
    pc.submit("pc", [pages[0].page_id])
    pc.begin_encoding("pc", [pages[0].page_id], token_start=2, token_end=6)
    pc.register_layer_leaves(
        "pc", pages[0].page_id, 0, node_index, leaves, token_offset=2
    )
    pc.complete_encoding("pc", [pages[0].page_id])
    pc_cache = pc.plan_prefix_cache("pc", 0)
    pc_sel = [[pages[0].page_id], [pages[0].page_id]]

    def pc_build(fallback, *, limits=lim, selection=pc_sel, use_cache=True):
        return build_candidate_plan(
            fallback_leaf_ids=fallback,
            selected_rag_by_head=selection,
            resolve_leaves=pc.head_resolver("pc", 0),
            num_leaves=limits,
            is_page_ready=lambda page_id: pc.page_is_ready("pc", page_id),
            trusted_resolver=True,
            prefix_cache=pc_cache if use_cache else None,
        )

    pc_fb = np.array([[0, 1], [5, 6]], dtype=np.int32)
    first = pc_build(pc_fb)
    ok(len(pc_cache) == 1 and pc_cache.misses == 1, "prefix cache miss stores")
    second = pc_build(pc_fb)
    ok(pc_cache.hits == 1, "prefix cache hit")
    ok(np.array_equal(first.ready_leaf_ids, second.ready_leaf_ids),
       "cached rows match the miss")
    ok(second.deferred_rag_pages == first.deferred_rag_pages,
       "cached deferred matches")
    ok(second.selection_metrics == first.selection_metrics,
       "cached metrics match")
    # A changed fallback still lands in the plan's fallback field on a hit;
    # the ready rows do not (they contain no fallback leaves).
    other_fb = np.array([[2, 3], [2, 3]], dtype=np.int32)
    hit = pc_build(other_fb)
    ok(np.array_equal(hit.ready_leaf_ids, first.ready_leaf_ids),
       "cache hit keeps ready rows")
    ok(np.array_equal(hit.fallback_leaf_ids, other_fb),
       "cache hit uses the new fallback")
    ok(hit.selection_metrics["n_fallback_leaves_used"] == 0.0,
       "full ready rows use no fallback slots")
    ok(np.array_equal(
        hit.ready_leaf_ids, pc_build(other_fb, use_cache=False).ready_leaf_ids
    ), "cache hit equals an uncached rebuild")
    # Ready leaves that do not fill width need fallback leaves, so the merged
    # row is token-specific and must not be cached.
    wide_fb = np.array([[0, 1, 2], [5, 6, 7]], dtype=np.int32)
    partial = pc_build(wide_fb, limits=np.array([8, 9], dtype=np.int32))
    ok(len(pc_cache) == 1, "partial prefix is not cached")
    ok(np.array_equal(
        partial.ready_leaf_ids,
        pc_build(wide_fb, limits=np.array([8, 9], dtype=np.int32),
                 use_cache=False).ready_leaf_ids,
    ), "uncached partial plan matches")
    # ``touch`` is a decode-hot marker and must not invalidate the cache.
    pc.touch("pc", [pages[0].page_id], step=3)
    hits_before = pc_cache.hits
    pc_build(pc_fb)
    ok(pc_cache.hits == hits_before + 1, "touch does not invalidate")
    # Any state/map mutation drops entries; a replace registration is visible
    # to the next plan, and a shrunken limit fails closed on a hit.
    pc.mark_gpu("pc", [pages[0].page_id])
    ok(len(pc_cache) == 0, "state mutation invalidates")
    pc.register_layer_leaves(
        "pc", pages[0].page_id, 0, node_index, leaves, token_offset=0,
        replace=True,
    )
    replaced = pc_build(pc_fb)
    ok(replaced.ready_leaf_ids[0].tolist() == [1, 2],
       "cache sees a replaced map")
    raises(ValueError, pc_build, pc_fb,
           limits=np.array([2, 9], dtype=np.int32))
    pc.release_session("pc")
    ok(len(pc_cache) == 0 and ("pc", 0) not in pc._prefix_caches,
       "release_session drops the cache")
    raises(ValueError, build_candidate_plan,
           fallback_leaf_ids=pc_fb, selected_rag_by_head=pc_sel,
           resolve_leaves=resolver, num_leaves=lim,
           prefix_cache=pc_cache)

    # Pure-prompt selection (stage 1) must reproduce the validated fallback
    # exactly, with no RAG leaves and no deferred pages.
    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[[], []],
        resolve_leaves=resolver,
        num_leaves=lim,
    )
    ok(np.array_equal(plan.ready_leaf_ids, fallback),
       "pure-prompt plan equals fallback")
    ok(plan.deferred_rag_pages == (), "pure-prompt has no deferred pages")
    ok(plan.metrics["n_rag_leaves_used"] == 0.0, "pure-prompt uses no RAG leaves")
    ok(plan.metrics["n_fallback_leaves_used"] == 4.0,
       "pure-prompt uses every fallback slot")

    # Wide heads take the vectorized fallback fill (live ``num_neighbours``
    # grows with the page budget, so hundreds of slots are realistic).  It must
    # reproduce the ordered set-loop exactly for both the masked path and the
    # wide-limit loop path.
    def _reference_fill(rag_groups, fb_row, width):
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for group in rag_groups:
            for leaf in group.tolist():
                if len(chosen) >= width:
                    break
                if leaf not in chosen_set:
                    chosen_set.add(leaf)
                    chosen.append(leaf)
        before = len(chosen)
        for leaf in fb_row.tolist():
            if len(chosen) >= width:
                break
            if leaf not in chosen_set:
                chosen_set.add(leaf)
                chosen.append(leaf)
        return chosen[:width], len(chosen) - before

    rng = np.random.default_rng(7)
    wide_width = _VECTOR_FILL_MIN + 256
    wide_fb = np.stack(
        [
            rng.choice(4096, wide_width, replace=False),
            rng.choice(4096, wide_width, replace=False),
        ]
    ).astype(np.int32)
    wide_rag = {
        # Duplicates across pages are legitimate (the merge de-duplicates);
        # duplicates inside one page result are rejected below.
        0: (np.array([3, 100, 4095], np.int32),
            np.array([3, 11, 12, 13], np.int32)),
        1: (np.array([7, 8, 9], np.int32),
            np.array([9, 10], np.int32)),
    }

    def wide_resolver(page_id: str, head: int):
        if page_id == "warm/wide/a":
            return wide_rag[head][0]
        if page_id == "warm/wide/b":
            return wide_rag[head][1]
        return None

    wide = build_candidate_plan(
        fallback_leaf_ids=wide_fb,
        selected_rag_by_head=[
            ["warm/wide/a", "warm/wide/b"],
            ["warm/wide/a", "warm/wide/b"],
        ],
        resolve_leaves=wide_resolver,
        num_leaves=np.array([4096, 4096]),
    )
    wide_fallback_used = 0
    for head in range(2):
        expected, used = _reference_fill(
            list(wide_rag[head]), wide_fb[head], wide_width
        )
        ok(wide.ready_leaf_ids[head].tolist() == expected,
           f"vectorized fill matches set loop (head {head})")
        wide_fallback_used += used
    ok(wide.selection_metrics["n_fallback_leaves_used"]
       == float(wide_fallback_used), "vectorized fill fallback metric")

    # Above _VECTOR_FILL_MAX_LEAVES the per-head mask is skipped for the loop.
    huge_limit = _VECTOR_FILL_MAX_LEAVES + 1
    huge_fb = np.stack(
        [
            rng.choice(huge_limit - 1, wide_width, replace=False),
            rng.choice(huge_limit - 1, wide_width, replace=False),
        ]
    ).astype(np.int32)
    huge = build_candidate_plan(
        fallback_leaf_ids=huge_fb,
        selected_rag_by_head=[
            ["warm/wide/a", "warm/wide/b"],
            ["warm/wide/a", "warm/wide/b"],
        ],
        resolve_leaves=wide_resolver,
        num_leaves=np.array([huge_limit, huge_limit]),
    )
    for head in range(2):
        expected, _ = _reference_fill(
            list(wide_rag[head]), huge_fb[head], wide_width
        )
        ok(huge.ready_leaf_ids[head].tolist() == expected,
           f"wide-limit loop fill matches (head {head})")

    # Ready-leaf collection is vectorized when the ready maps are large
    # (>= _VECTOR_FILL_MIN leaves and <= 2x the output width).  It must match
    # the ordered set loop across pages, including cross-page duplicates and
    # truncation past ``width``.
    vec_width = 192
    vec_n_pages = 16
    vec_maps = {}
    for p in range(vec_n_pages):
        base = p * 15  # 16 leaves per page, 15 new -> deliberate overlap
        leaves = np.arange(base, base + 16, dtype=np.int32)
        for h in range(2):
            vec_maps[(f"vec/p{p}", h)] = (leaves + h) % 4000
    vec_fb = np.stack(
        [rng.choice(4000, vec_width, replace=False) for _ in range(2)]
    ).astype(np.int32)

    def vec_resolver(page_id: str, head: int):
        return vec_maps[(page_id, head)]

    vec = build_candidate_plan(
        fallback_leaf_ids=vec_fb,
        selected_rag_by_head=[
            [f"vec/p{p}" for p in range(vec_n_pages)] for _ in range(2)
        ],
        resolve_leaves=vec_resolver,
        num_leaves=np.array([4000, 4000]),
    )
    for head in range(2):
        expected, _ = _reference_fill(
            [vec_maps[(f"vec/p{p}", head)] for p in range(vec_n_pages)],
            vec_fb[head],
            vec_width,
        )
        ok(vec.ready_leaf_ids[head].tolist() == expected,
           f"vectorized ready-leaf collection matches (head {head})")

    # Unmaterialized pages are deferred and fallback fills every slot.
    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["ghost/page"], ["ghost/page"]],
        resolve_leaves=resolver,
        num_leaves=lim,
    )
    ok(plan.deferred_rag_pages == ("ghost/page",), "deferred page recorded")
    ok(np.array_equal(plan.ready_leaf_ids, fallback), "fallback used verbatim")
    ok(plan.selection_metrics["n_deferred_rag_pages"] == 1.0, "deferred metric")

    # Decode-cold fast path: a known-unmaterialized page is deferred without
    # calling the resolver at all, and with nothing ready the merge is skipped.
    def exploding_resolver(page_id: str, head: int):
        raise AssertionError("resolver must not be called for an unready page")

    fast = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["cold/a"], ["cold/b"]],
        resolve_leaves=exploding_resolver,
        num_leaves=lim,
        is_page_ready=lambda page_id: False,
    )
    ok(fast.deferred_rag_pages == ("cold/a", "cold/b"),
       "fast path defers in first-seen order")
    ok(np.array_equal(fast.ready_leaf_ids, fallback), "fast path uses fallback")
    ok(fast.selection_metrics == {
        "n_ready_rag_pages": 0.0,
        "n_deferred_rag_pages": 2.0,
        "n_rag_leaves_used": 0.0,
        "n_fallback_leaves_used": 4.0,
        "n_kv_heads": 2.0,
        "width": 2.0,
    }, "fast path metrics match the slow path")
    validate_leaf_ids(fast.ready_leaf_ids, n_kv_heads=2, num_neighbours=2,
                      num_leaves=lim)

    # Mixed readiness: only the cold page skips the resolver.
    def mixed_resolver(page_id: str, head: int):
        if page_id == "cold/page":
            raise AssertionError("resolver called for a known-cold page")
        return (
            np.array([1, 2], dtype=np.int32)
            if head == 0
            else np.array([6, 7], dtype=np.int32)
        )

    mixed = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["cold/page", "warm/page"], ["warm/page"]],
        resolve_leaves=mixed_resolver,
        num_leaves=lim,
        is_page_ready=lambda page_id: page_id == "warm/page",
    )
    ok(mixed.deferred_rag_pages == ("cold/page",),
       "only the cold page is deferred")
    ok(mixed.ready_leaf_ids[0].tolist() == [1, 2], "warm head 0 used")
    ok(mixed.ready_leaf_ids[1].tolist() == [6, 7], "warm head 1 used")

    # Ready page with no leaves contributes nothing but is not deferred.
    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[[pages[1].page_id], [pages[1].page_id]],
        resolve_leaves=resolver,
        num_leaves=lim,
    )
    ok(plan.deferred_rag_pages == (), "empty ready page is not deferred")
    ok(np.array_equal(plan.ready_leaf_ids, fallback), "empty ready falls back")

    # Partial readiness across heads fails closed into deferred.
    def half_ready(page_id: str, head: int):
        return np.array([1], dtype=np.int32) if head == 0 else None

    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["half/page"], ["half/page"]],
        resolve_leaves=half_ready,
        num_leaves=lim,
    )
    ok(plan.deferred_rag_pages == ("half/page",), "inconsistent readiness defers")
    ok(np.array_equal(plan.ready_leaf_ids, fallback), "half-ready falls back")

    # A ready page whose layer map was invalidated resolves None, but it must
    # not be deferred: the deferred channel re-queues pages, and an already
    # encoded page cannot begin encoding again.  It contributes zero leaves.
    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["stale/page"], ["stale/page"]],
        resolve_leaves=lambda page_id, head: None,
        is_page_ready=lambda page_id: page_id == "stale/page",
        num_leaves=lim,
    )
    ok(plan.deferred_rag_pages == (),
       "ready page without a layer map is not deferred")
    ok(np.array_equal(plan.ready_leaf_ids, fallback),
       "invalidated layer contributes no RAG leaves")
    ok(plan.selection_metrics["n_ready_rag_pages"] == 1.0,
       "invalidated layer still counts as a materialized page")

    # A candidate from another layer is a selector bug: the leaves are in
    # range, so only an explicit layer check catches it.
    class _LayerPage:
        page_id = "layer/page"
        layer_idx = 5

    class _LayerCandidate:
        page = _LayerPage()

    raises(ValueError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[[_LayerCandidate()], []],
           resolve_leaves=resolver, num_leaves=lim, layer_idx=3)

    # Likewise, a candidate listed under the wrong KV head is rejected when it
    # exposes head_idx (per-head selection lists are not regrouped here).
    class _HeadPage:
        page_id = "head/page"
        head_idx = 1

    class _HeadCandidate:
        page = _HeadPage()

    raises(ValueError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[[_HeadCandidate()], []],
           resolve_leaves=resolver, num_leaves=lim)

    # Prompt-source entries are ignored, so their metadata is not validated.
    class _PromptMismatch:
        page = type(
            "P",
            (),
            {
                "page_id": "prompt/mismatch",
                "source": "prompt",
                "layer_idx": 9,
                "head_idx": 9,
            },
        )()

    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[
            [_PromptMismatch()], [_PromptMismatch()]
        ],
        resolve_leaves=lambda page_id, head: None,
        num_leaves=lim,
        layer_idx=0,
    )
    ok(plan.deferred_rag_pages == (), "prompt metadata is not validated")

    # Even on the trusted path, an int64 ID must not alias through the cast.
    raises(ValueError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[["wide/page"], ["wide/page"]],
           resolve_leaves=lambda page_id, head: np.array(
               [2**32 + 5], dtype=np.int64
           ),
           num_leaves=lim, trusted_resolver=True)

    # RAG leaves that overlap the fallback are de-duplicated before filling.
    def overlapping(page_id: str, head: int):
        return np.array([0], dtype=np.int32) if head == 0 else np.array(
            [5], dtype=np.int32
        )

    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["over/page"], ["over/page"]],
        resolve_leaves=overlapping,
        num_leaves=lim,
    )
    ok(plan.ready_leaf_ids[0].tolist() == [0, 1], "dedup head 0")
    ok(plan.ready_leaf_ids[1].tolist() == [5, 6], "dedup head 1")

    # Multiple ready pages fill in selection order.
    def by_page(page_id: str, head: int):
        return {"a/page": 2, "b/page": 3}[page_id] + np.zeros(1, dtype=np.int32)

    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["a/page", "b/page"], ["a/page", "b/page"]],
        resolve_leaves=by_page,
        num_leaves=lim,
    )
    ok(plan.ready_leaf_ids[0].tolist() == [2, 3], "multi-page head 0 order")
    ok(plan.ready_leaf_ids[1].tolist() == [2, 3], "multi-page head 1 order")
    ok(plan.selection_metrics["n_ready_rag_pages"] == 2.0, "two ready pages")

    # Extra RAG leaves are truncated to the fixed page width.
    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[["wide/page"], ["wide/page"]],
        resolve_leaves=lambda page_id, head: (
            np.array([2, 3, 4]) if head == 0 else np.array([7, 8, 9])
        ),
        num_leaves=np.array([5, 10], dtype=np.int32),
    )
    ok(plan.ready_leaf_ids[0].tolist() == [2, 3], "width truncates head 0")
    ok(plan.ready_leaf_ids[1].tolist() == [7, 8], "width truncates head 1")
    ok(plan.selection_metrics["n_fallback_leaves_used"] == 0.0,
       "no fallback leaves when RAG fills width")

    # Prompt-source selections are ignored; the fallback already encodes them.
    class _FakePage:
        page_id = "prompt/page"

    class _FakeCandidate:
        page = _FakePage()
        source = "prompt"

    _FakePage.source = "prompt"
    plan = build_candidate_plan(
        fallback_leaf_ids=fallback,
        selected_rag_by_head=[[_FakeCandidate()], [_FakeCandidate()]],
        resolve_leaves=lambda page_id, head: None,
        num_leaves=lim,
    )
    ok(plan.deferred_rag_pages == (), "prompt entries ignored")

    raises(ValueError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[["ghost/page"]], resolve_leaves=resolver,
           num_leaves=lim)
    # The plain-string fast path used to accept "" and emit it as a deferred
    # page ID, which the public CandidatePlan constructor and the deferred
    # queue reject.
    raises(ValueError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[[""], [""]],
           resolve_leaves=lambda page_id, head: None,
           num_leaves=lim, is_page_ready=lambda page_id: False)
    raises(TypeError, build_candidate_plan, fallback_leaf_ids=fallback.astype(float),
           selected_rag_by_head=[[], []], resolve_leaves=resolver,
           num_leaves=lim)
    raises(ValueError, build_candidate_plan,
           fallback_leaf_ids=np.array([[0, 0], [5, 6]], dtype=np.int32),
           selected_rag_by_head=[[], []], resolve_leaves=resolver,
           num_leaves=lim)
    raises(ValueError, build_candidate_plan,
           fallback_leaf_ids=np.array([[0, 9], [5, 6]], dtype=np.int32),
           selected_rag_by_head=[[], []], resolve_leaves=resolver,
           num_leaves=lim)
    # int64 values that wrap to a valid int32 leaf must not alias through the
    # validate_leaf_ids cast (2**32 -> 0, -2**32 -> 0).
    raises(ValueError, build_candidate_plan,
           fallback_leaf_ids=np.array([[2**32, 1], [5, 6]], dtype=np.int64),
           selected_rag_by_head=[[], []], resolve_leaves=resolver,
           num_leaves=lim)
    raises(ValueError, build_candidate_plan,
           fallback_leaf_ids=np.array([[-2**32, 1], [5, 6]], dtype=np.int64),
           selected_rag_by_head=[[], []], resolve_leaves=resolver,
           num_leaves=lim)
    raises(ValueError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[["bad/page"], ["bad/page"]],
           resolve_leaves=lambda page_id, head: np.array([99]),
           num_leaves=lim)
    raises(ValueError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[["dup/page"], ["dup/page"]],
           resolve_leaves=lambda page_id, head: np.array([1, 1]),
           num_leaves=lim)
    raises(TypeError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=[[], []], resolve_leaves=resolver,
           num_leaves=np.array([5.0, 9.0]))
    # A per-head selection given as a bare string would be iterated
    # character-by-character, scheduling one-letter "pages".
    raises(TypeError, build_candidate_plan, fallback_leaf_ids=fallback,
           selected_rag_by_head=["page/a", "page/b"],
           resolve_leaves=resolver, num_leaves=lim,
           is_page_ready=lambda page_id: False)
    # CandidatePlan rejects a zero-width leaf array; the builder must not
    # return a plan its own constructor would refuse.
    raises(ValueError, build_candidate_plan,
           fallback_leaf_ids=np.zeros((2, 0), dtype=np.int32),
           selected_rag_by_head=[[], []], resolve_leaves=resolver,
           num_leaves=lim)
    raises(ValueError, CandidatePlan, ready_leaf_ids=fallback,
           deferred_rag_pages=("x", "x"), fallback_leaf_ids=fallback,
           selection_metrics={})
    raises(ValueError, CandidatePlan,
           ready_leaf_ids=np.array([[0, 0], [5, 6]], dtype=np.int32),
           deferred_rag_pages=(), fallback_leaf_ids=fallback,
           selection_metrics={})
    raises(ValueError, CandidatePlan, ready_leaf_ids=fallback,
           deferred_rag_pages=(),
           fallback_leaf_ids=np.array([[5, 5], [5, 6]], dtype=np.int32),
           selection_metrics={})
    raises(TypeError, CandidatePlan, ready_leaf_ids=fallback,
           deferred_rag_pages="x", fallback_leaf_ids=fallback,
           selection_metrics={})

    print(f"materialization self-test: {checks} checks passed")


if __name__ == "__main__":
    _self_test()
