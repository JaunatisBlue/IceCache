"""Session-local KV materialization for unified semantic pages.

Why this module exists
----------------------
``unified_page_index`` ranks global RAG pages and session-private prompt pages
in one descriptor space.  A descriptor is not attention state: before a
selected page can enter IceCache's page table it needs K/V for every KV head,
in the physical page layout, with a valid-entry count.  This module is the
storage/materialization half of that bridge.

Exactness contract
------------------
RoPE is applied after the key projection (and, for Qwen3, after ``k_norm``).
For one fixed pre-RoPE key, the positional transform is exactly invertible and
composable:

    K(p') = R(p') R(p)^T K(p)        V has no positional transform

where ``R(p)`` is the model's rotary rotation at position ``p``.  This module
stores pre-RoPE keys plus their source positions, so the rotation is not baked
into the artifact.

That statement is about one key, not about a whole page.  At layer 0 the
pre-RoPE key is the projected token embedding, so re-positioning a page only
changes the rotation.  At deeper layers the pre-RoPE key is produced by causal
attention over the prefix, whose scores depend on *relative* positions.  It
stays valid only when every relative position is preserved, i.e. when the
entire encoded span receives a common position shift (pure-RoPE models; no
ALiBi).  Moving one page while the rest of the context stays put, or reordering
pages, changes the deeper hidden states and is an ablation, not an exact
relocation.  ``materialize`` therefore accepts ``shift=<uniform delta>`` as the
exact path and rejects non-uniform ``positions`` for ``layer_idx > 0`` unless
``allow_position_reorder=True`` is set explicitly.

Re-rotation also does *not* make a page valid in a different context.  A page
encoded outside the target session (``scope="global"``) is cross-context by
default and is refused unless it carries ``verified_cross_context=True`` or the
caller passes ``allow_unverified=True``.  Session pages are never visible to a
different session.  This mirrors the feasibility gate in
``benchmark/agent_cache/UNIFIED_RAG_KV_DESIGN.md``: shared retrieval metadata
may remain useful even when deep K/V does not, so the unsafe path has to be
explicit.  ``context_fingerprint`` should cover both the prefix tokens and the
position layout that produced the pre-RoPE keys.

A uniform shift is exact only under these preconditions: the same delta is
applied to every token in the context *and* to the queries, the model uses
pure RoPE (no ALiBi or other absolute-position term outside ``apply_rope``),
the stored ``inv_freq`` matches the live rotary embedding (including any
``rope_scaling``), and the pre-RoPE keys came from the same prefix token ids.
Shifting positions does not retroactively condition a cached prefix on newly
inserted tokens: prepending a system prompt and then shifting the old cache is
a context change, not a relocation, even though every position moves by the
same delta.  The materializer can check the last two; the runtime owns the
first two.

Capturing artifacts
-------------------
Capture pre-RoPE keys immediately before ``apply_rotary_pos_emb`` costs one
extra tensor reference per layer: the model computes ``key_states`` as
``[batch, n_kv_heads, tokens, head_dim]`` there, while ``value_states`` is
still ``[batch, tokens, n_kv_heads, head_dim]``.  Transpose both to
``[n_kv_heads, tokens, head_dim]`` and pass them with the position ids and
``rotary_emb.inv_freq`` to :func:`split_into_pages`.  Do *not* capture from
``kvc.buffer``: that copy is already post-RoPE, so ``from_post_rope`` would
add the storage dtype's rounding error to every later relocation.

Descriptor caveat: a page descriptor is computed from rotated keys.  Even a
uniform relocation rotates different tokens by different amounts, so the mean
descriptor changes; materializing a page at new positions while ranking it by
its source-position descriptor is a mismatch.  Recompute the descriptor at the
positions actually used, or carry that mismatch explicitly in the fidelity
gate rather than assuming the source vector still represents the page.


Physical layout
---------------
IceCache builds ``KvPool`` with ``layout_map=(0, 2, 1, 3)``, so one pool page
is ``(2, n_kv_heads, page_size, head_dim)`` (K, V; HND).  ``PageKV`` stores one
page with all KV heads because IceCache's page table is shared across heads;
per-head catalogs only produce per-head descriptors.  ``pack_pages`` emits
exactly the pool shape for a copy into a ``KvPool`` slice.

Numerical note
--------------
Rotary tables are regenerated from the stored ``inv_freq`` in float64 and stay
float64: ``apply_rope`` casts them to float64, never to the key dtype.  Rotation
arithmetic runs in float64 and the result rounds once to the key dtype.  That is more accurate than, but *not bit-identical to*, a live
model that rounds every multiply/add in its own dtype: with the same table,
roughly a third of fp16 elements differ by one ulp.  The exactness contract
above is therefore about the rotation identity in float64
(``remove_rope(apply_rope(x)) == x``), not about reproducing the model's
per-operation rounding.  Prefer capturing pre-RoPE keys *before* RoPE:
``from_post_rope`` can recover them from already-rotated storage, but it then
inherits the rounding error of that storage (in particular IceCache's
reduced-precision keys).

Limitations
-----------
This implements the full-rotary, half-split convention used by Llama and Qwen
in this repository (``rotate_half = cat(-x[d/2:], x[:d/2])``).  A model with
``partial_rotary_factor < 1`` leaves some channels unrotated and needs a
masked table; an interleaved-pair model needs a different ``rotate_half``.

Usage
-----
>>> store = KVPageStore()
>>> store.add(PageKV(..., scope="session", session_id="s1"))
>>> batch = store.pack(["page-1"], session_id="s1", positions=target_pos)
>>> batch.kv.shape
(1, 2, n_kv_heads, page_size, head_dim)

This module imports only NumPy, so it stays usable in CPU-only analyses.  It
is intentionally *not* wired into ``InferState`` yet: the runtime hook that
copies a packed batch into the page table is a separate increment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

__all__ = [
    "KVPageStore",
    "MaterializedPage",
    "PageKV",
    "PackedPageBatch",
    "apply_rope",
    "materialize_page",
    "pack_pages",
    "pages_from_model_kv",
    "remove_rope",
    "rerope",
    "rope_cos_sin",
    "rotate_half",
    "split_into_pages",
]

PageScope = Literal["global", "session"]


# ---------------------------------------------------------------------------
# RoPE primitives
# ---------------------------------------------------------------------------


def _as_array(value: Any, *, name: str) -> np.ndarray:
    """Convert numpy/torch input without importing torch.

    CUDA tensors must be moved by the caller; a bf16 torch tensor cannot be
    converted by NumPy, so the caller should cast it to float32 first.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except (TypeError, RuntimeError) as exc:  # bf16 and other non-numpy dtypes
            raise TypeError(
                f"{name}: tensor dtype cannot be converted to NumPy; "
                "cast to float32 on CPU first"
            ) from exc
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name}: expected a numeric array")
    return array


def _validate_rope_inputs(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> None:
    if x.ndim < 1 or x.shape[-1] % 2 != 0:
        raise ValueError(
            f"RoPE input's last dimension must be even, got shape {x.shape}"
        )
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos and sin shapes differ: {cos.shape} vs {sin.shape}"
        )
    if cos.ndim < 2:
        raise ValueError(
            "cos/sin need a sequence axis; expected [..., tokens, head_dim]"
        )
    if cos.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"rotary width {cos.shape[-1]} does not match head_dim {x.shape[-1]}"
        )
    if cos.shape[-2] != x.shape[-2]:
        raise ValueError(
            f"rotary table has {cos.shape[-2]} positions but input has "
            f"{x.shape[-2]} tokens"
        )
    try:
        np.broadcast_shapes(x.shape, cos.shape)
    except ValueError as exc:
        raise ValueError(
            f"rotary table shape {cos.shape} does not broadcast to {x.shape}"
        ) from exc


def _compute_dtype(x: np.ndarray) -> np.dtype:
    if np.issubdtype(x.dtype, np.floating):
        return x.dtype
    return np.dtype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Llama/Qwen intra-head rotation: ``cat(-x[d/2:], x[:d/2])``."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"rotate_half needs an even last dimension, got {x.shape}")
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply ``R(p)``: ``x * cos + rotate_half(x) * sin``.

    ``cos``/``sin`` have shape ``[..., T, head_dim]`` and broadcast over any
    leading head/batch axes of ``x``.
    """
    x = _as_array(x, name="x")
    cos = _as_array(cos, name="cos")
    sin = _as_array(sin, name="sin")
    _validate_rope_inputs(x, cos, sin)
    x64 = x.astype(np.float64, copy=False)
    out = x64 * cos.astype(np.float64, copy=False)
    out += rotate_half(x64) * sin.astype(np.float64, copy=False)
    return out.astype(_compute_dtype(x), copy=False)


def remove_rope(y: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply ``R(p)^T``, the inverse of :func:`apply_rope` for the same table."""
    y = _as_array(y, name="y")
    cos = _as_array(cos, name="cos")
    sin = _as_array(sin, name="sin")
    _validate_rope_inputs(y, cos, sin)
    y64 = y.astype(np.float64, copy=False)
    out = y64 * cos.astype(np.float64, copy=False)
    out -= rotate_half(y64) * sin.astype(np.float64, copy=False)
    return out.astype(_compute_dtype(y), copy=False)


def rerope(
    keys: np.ndarray,
    source_cos: np.ndarray,
    source_sin: np.ndarray,
    target_cos: np.ndarray,
    target_sin: np.ndarray,
) -> np.ndarray:
    """Move rotated keys from source to target positions.

    Equivalent to ``R(target) R(source)^T keys``.  This is exact for the key
    projection when the prefix (and hence the pre-RoPE key) is unchanged.
    """
    unrotated = remove_rope(keys, source_cos, source_sin)
    return apply_rope(unrotated, target_cos, target_sin)


def rope_cos_sin(
    positions: Sequence[int] | np.ndarray,
    inv_freq: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the Llama-style rotary table for ``positions``.

    ``inv_freq`` has shape ``[head_dim // 2]``; the half-vector is duplicated
    so the result has shape ``[positions, head_dim]`` and matches
    ``rotate_half``'s half-split convention.

    Pass the model's *effective* ``inv_freq`` (``rotary_emb.inv_freq``), not
    ``1 / base**(arange/head_dim)`` recomputed from the config.  Scaled
    variants (for example Llama-3.1's ``rope_scaling`` with
    ``rope_type="llama3"``) transform the frequencies before use, and a table
    rebuilt from the base alone would silently mis-rotate every page.

    Returns float64 tables.  Operations in this module compute in float64 and
    cast back to the key dtype.
    """
    positions = _as_array(positions, name="positions")
    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if np.issubdtype(positions.dtype, np.floating):
        if not np.all(positions == np.floor(positions)):
            raise TypeError("positions must be integral")
        positions = positions.astype(np.int64)
    elif not np.issubdtype(positions.dtype, np.integer):
        raise TypeError("positions must be integral")
    inv_freq = np.asarray(inv_freq, dtype=np.float64).reshape(-1)
    if inv_freq.size == 0:
        raise ValueError("inv_freq must not be empty")
    if not np.all(np.isfinite(inv_freq)) or np.any(inv_freq < 0):
        raise ValueError("inv_freq must be finite and nonnegative")
    frequencies = np.outer(positions.astype(np.float64), inv_freq)
    emb = np.concatenate((frequencies, frequencies), axis=-1)
    return np.cos(emb), np.sin(emb)


# ---------------------------------------------------------------------------
# Page storage
# ---------------------------------------------------------------------------


def _freeze(array: np.ndarray) -> np.ndarray:
    array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PageKV:
    """One physical page of pre-RoPE keys and values for all KV heads.

    ``pre_rope_keys`` / ``values`` have shape ``[n_kv_heads, n_tokens,
    head_dim]``.  ``source_positions`` is the absolute position of every token
    when the page was encoded, and ``inv_freq`` is the rotary table used for
    that encoding, so a later session can re-rotate the page without knowing
    the encoder's model object.  For ``layer_idx > 0`` these keys are valid
    only under the prefix *and* relative-position layout that produced them;
    see the module docstring for the relocation contract.

    ``scope="global"`` pages are immutable, cross-session artifacts (RAG
    documents).  ``scope="session"`` pages belong to exactly one session.  A
    global page must opt in to cross-context reuse with
    ``verified_cross_context=True``; the gate is enforced by
    :func:`materialize_page`.
    """

    physical_page_id: str
    layer_idx: int
    pre_rope_keys: np.ndarray
    values: np.ndarray
    source_positions: np.ndarray
    inv_freq: np.ndarray
    page_size: int
    scope: PageScope = "session"
    session_id: str | None = None
    document_id: str | None = None
    segment_id: str | None = None
    context_fingerprint: str | None = None
    verified_cross_context: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.physical_page_id, str) or not self.physical_page_id:
            raise ValueError("physical_page_id must be a nonempty string")
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be nonnegative")

        keys = _as_array(self.pre_rope_keys, name="pre_rope_keys")
        values = _as_array(self.values, name="values")
        positions = _as_array(self.source_positions, name="source_positions").reshape(-1)
        inv_freq = np.asarray(_as_array(self.inv_freq, name="inv_freq"), dtype=np.float64).reshape(-1)

        if keys.ndim != 3:
            raise ValueError(
                f"pre_rope_keys must be [n_kv_heads, n_tokens, head_dim], got {keys.shape}"
            )
        if values.shape != keys.shape:
            raise ValueError(
                f"values shape {values.shape} differs from keys shape {keys.shape}"
            )
        n_heads, n_tokens, head_dim = keys.shape
        if n_heads <= 0 or n_tokens <= 0 or head_dim <= 0:
            raise ValueError("empty head, token, or head_dim axis")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if not np.issubdtype(keys.dtype, np.floating) or not np.issubdtype(values.dtype, np.floating):
            raise TypeError("keys and values must be floating point")
        if keys.dtype != values.dtype:
            raise TypeError(
                f"keys and values must share a dtype, got {keys.dtype} and {values.dtype}"
            )
        if self.page_size <= 0 or n_tokens > self.page_size:
            raise ValueError(
                f"page holds {n_tokens} tokens but page_size is {self.page_size}"
            )
        if positions.shape != (n_tokens,):
            raise ValueError(
                f"source_positions must have shape [{n_tokens}], got {positions.shape}"
            )
        if not np.issubdtype(positions.dtype, np.integer):
            if np.issubdtype(positions.dtype, np.floating) and np.all(
                positions == np.floor(positions)
            ):
                positions = positions.astype(np.int64)
            else:
                raise TypeError("source_positions must be integral")
        if np.any(positions < 0):
            raise ValueError("source_positions must be nonnegative")
        # Adjacent comparison instead of ``np.diff``: unsigned dtypes wrap on
        # subtraction, so ``[5, 0]`` as uint8 would pass a ``diff > 0`` test.
        if positions.size > 1 and np.any(positions[1:] <= positions[:-1]):
            raise ValueError("source_positions must be strictly increasing")
        if inv_freq.shape != (head_dim // 2,):
            raise ValueError(
                f"inv_freq must have shape [{head_dim // 2}], got {inv_freq.shape}"
            )
        if not np.all(np.isfinite(inv_freq)) or np.any(inv_freq < 0):
            raise ValueError("inv_freq must be finite and nonnegative")

        if self.scope not in ("global", "session"):
            raise ValueError(f"unknown scope: {self.scope!r}")
        if self.scope == "session":
            if not isinstance(self.session_id, str) or not self.session_id:
                raise ValueError("session pages require a nonempty session_id")
        else:
            if self.session_id is not None:
                raise ValueError("global pages must not carry a session_id")
        if self.context_fingerprint is not None and not self.context_fingerprint:
            raise ValueError("context_fingerprint must be nonempty when provided")

        object.__setattr__(self, "pre_rope_keys", _freeze(keys))
        object.__setattr__(self, "values", _freeze(values))
        object.__setattr__(self, "source_positions", _freeze(positions))
        object.__setattr__(self, "inv_freq", _freeze(inv_freq))

    @classmethod
    def from_post_rope(
        cls,
        *,
        post_rope_keys: Any,
        values: Any,
        source_positions: Any,
        inv_freq: Any,
        **kwargs: Any,
    ) -> "PageKV":
        """Recover pre-RoPE keys from already-rotated storage.

        Prefer capturing keys *before* RoPE.  Undoing rotation on a reduced
        precision copy adds that copy's rounding error, and re-rotation cannot
        remove it.
        """
        keys = _as_array(post_rope_keys, name="post_rope_keys")
        positions = _as_array(source_positions, name="source_positions").reshape(-1)
        cos, sin = rope_cos_sin(positions, inv_freq)
        unrotated = remove_rope(keys, cos, sin)
        return cls(
            pre_rope_keys=unrotated,
            values=values,
            source_positions=positions,
            inv_freq=inv_freq,
            **kwargs,
        )

    @property
    def n_kv_heads(self) -> int:
        return int(self.pre_rope_keys.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.pre_rope_keys.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.pre_rope_keys.shape[2])

    @property
    def dtype(self) -> np.dtype:
        return self.pre_rope_keys.dtype

    @property
    def nbytes(self) -> int:
        return int(
            self.pre_rope_keys.nbytes
            + self.values.nbytes
            + self.source_positions.nbytes
            + self.inv_freq.nbytes
        )

    def head_keys(self, head: int) -> np.ndarray:
        """Read-only pre-RoPE keys for one head, shape ``[n_tokens, head_dim]``."""
        if not 0 <= head < self.n_kv_heads:
            raise IndexError(f"head {head} out of range for {self.n_kv_heads} heads")
        return self.pre_rope_keys[head]

    def head_values(self, head: int) -> np.ndarray:
        """Read-only values for one head, shape ``[n_tokens, head_dim]``."""
        if not 0 <= head < self.n_kv_heads:
            raise IndexError(f"head {head} out of range for {self.n_kv_heads} heads")
        return self.values[head]


@dataclass(frozen=True)
class MaterializedPage:
    """Post-RoPE K/V ready to be written into a destination page table."""

    physical_page_id: str
    layer_idx: int
    keys: np.ndarray  # [n_kv_heads, n_tokens, head_dim], post-RoPE
    values: np.ndarray  # [n_kv_heads, n_tokens, head_dim]
    positions: np.ndarray  # [n_tokens]
    page_size: int
    scope: PageScope
    session_id: str | None
    document_id: str | None
    segment_id: str | None

    def __post_init__(self) -> None:
        # ``pack_pages`` accepts any duck-typed page, but instances built here
        # must not carry a key/value dtype mismatch (the packing assignment
        # would silently cast one side).
        keys = _as_array(self.keys, name="keys")
        values = _as_array(self.values, name="values")
        if keys.ndim != 3:
            raise ValueError(
                f"keys must be [n_kv_heads, n_tokens, head_dim], got {keys.shape}"
            )
        if values.shape != keys.shape:
            raise ValueError(
                f"values shape {values.shape} differs from keys shape {keys.shape}"
            )
        if not np.issubdtype(keys.dtype, np.floating) or not np.issubdtype(
            values.dtype, np.floating
        ):
            raise TypeError("keys and values must be floating point")
        if keys.dtype != values.dtype:
            raise TypeError(
                f"keys and values must share a dtype, got {keys.dtype} and "
                f"{values.dtype}"
            )
        positions = _as_array(self.positions, name="positions")
        if positions.shape != (keys.shape[1],):
            raise ValueError(
                f"positions must have shape [{keys.shape[1]}], got {positions.shape}"
            )

    @property
    def n_kv_heads(self) -> int:
        return int(self.keys.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.keys.shape[1])

    @property
    def valid_entries(self) -> np.ndarray:
        """Per-head valid token count, shape ``[n_kv_heads]``, int32."""
        return np.full(self.n_kv_heads, self.n_tokens, dtype=np.int32)

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes + self.positions.nbytes)


def _positions_array(value: Any, *, name: str) -> np.ndarray:
    positions = _as_array(value, name=name).reshape(-1)
    if not np.issubdtype(positions.dtype, np.integer):
        if np.issubdtype(positions.dtype, np.floating) and np.all(
            positions == np.floor(positions)
        ):
            positions = positions.astype(np.int64)
        else:
            raise TypeError(f"{name} must be integral")
    if np.any(positions < 0):
        raise ValueError(f"{name} must be nonnegative")
    # ``np.diff`` on an unsigned dtype wraps: [5, 0] as uint8 gives 251 > 0.
    if positions.size > 1 and np.any(positions[1:] <= positions[:-1]):
        raise ValueError(f"{name} must be strictly increasing")
    return positions


def split_into_pages(
    *,
    pre_rope_keys: Any,
    values: Any,
    source_positions: Any,
    inv_freq: Any,
    page_size: int,
    physical_id_prefix: str,
    layer_idx: int,
    scope: PageScope = "session",
    session_id: str | None = None,
    document_id: str | None = None,
    segment_id: str | None = None,
    context_fingerprints: str | Sequence[str] | None = None,
    verified_cross_context: bool = False,
) -> list[PageKV]:
    """Split one encoded span into page-aligned ``PageKV`` objects.

    A RAG chunk is usually longer than one IceCache page (a 512-token chunk at
    ``page_size=16`` is 32 pages), so page-aligned splitting has to happen
    before the runtime can address it.  The last page may be partial.

    ``context_fingerprints`` may be one string for the whole span or one entry
    per output page.  In a causal encoding, page *k*'s values depend on all
    earlier pages, so per-page fingerprints that chain the prefix are the
    precise choice; a single string is only correct for a standalone encoding
    where all pages share the same prefix.
    """
    keys = _as_array(pre_rope_keys, name="pre_rope_keys")
    vals = _as_array(values, name="values")
    if keys.ndim != 3 or vals.shape != keys.shape:
        raise ValueError(
            "pre_rope_keys and values must share shape [n_kv_heads, n_tokens, head_dim]"
        )
    positions = _positions_array(source_positions, name="source_positions")
    n_tokens = keys.shape[1]
    if positions.size != n_tokens:
        raise ValueError(
            f"source_positions has {positions.size} entries for {n_tokens} tokens"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if not isinstance(physical_id_prefix, str) or not physical_id_prefix:
        raise ValueError("physical_id_prefix must be a nonempty string")

    n_pages = (n_tokens + page_size - 1) // page_size
    if context_fingerprints is None:
        fingerprints: Sequence[str | None] = [None] * n_pages
    elif isinstance(context_fingerprints, str):
        fingerprints = [context_fingerprints] * n_pages
    else:
        fingerprints = list(context_fingerprints)
        if len(fingerprints) != n_pages:
            raise ValueError(
                f"expected {n_pages} context fingerprints, got {len(fingerprints)}"
            )

    prefix = physical_id_prefix.rstrip("/")
    pages: list[PageKV] = []
    for index in range(n_pages):
        start = index * page_size
        end = min(start + page_size, n_tokens)
        pages.append(
            PageKV(
                physical_page_id=f"{prefix}/p{index:05d}",
                layer_idx=layer_idx,
                pre_rope_keys=keys[:, start:end],
                values=vals[:, start:end],
                source_positions=positions[start:end],
                inv_freq=inv_freq,
                page_size=page_size,
                scope=scope,
                session_id=session_id,
                document_id=document_id,
                segment_id=segment_id,
                context_fingerprint=fingerprints[index],
                verified_cross_context=verified_cross_context,
            )
        )
    return pages


def pages_from_model_kv(
    key_states: Any,
    value_states: Any,
    position_ids: Any,
    inv_freq: Any,
    *,
    page_size: int,
    physical_id_prefix: str,
    layer_idx: int,
    **page_kwargs: Any,
) -> list[PageKV]:
    """Split the tensors captured immediately before RoPE into pages.

    This accepts the exact layouts at the capture point in
    ``adapter/modeling.py``: keys are ``[batch, n_kv_heads, tokens,
    head_dim]`` and values are ``[batch, tokens, n_kv_heads, head_dim]``.
    Only batch size one is supported.  ``position_ids`` may be ``[tokens]`` or
    ``[1, tokens]``.  Extra keyword arguments are forwarded to
    :func:`split_into_pages` (scope, session id, fingerprints, and so on).
    """
    keys = _as_array(key_states, name="key_states")
    vals = _as_array(value_states, name="value_states")
    if keys.ndim != 4:
        raise ValueError(
            "key_states must be [batch, n_kv_heads, tokens, head_dim]; "
            f"got {keys.shape}"
        )
    if vals.ndim != 4:
        raise ValueError(
            "value_states must be [batch, tokens, n_kv_heads, head_dim]; "
            f"got {vals.shape}"
        )
    if keys.shape[0] != 1 or vals.shape[0] != 1:
        raise ValueError("pages_from_model_kv supports batch size one only")
    batch, n_heads, n_tokens, head_dim = keys.shape
    if vals.shape[1:] != (n_tokens, n_heads, head_dim):
        raise ValueError(
            "key/value layouts disagree: expected values "
            f"[1, {n_tokens}, {n_heads}, {head_dim}], got {vals.shape}"
        )
    positions = _as_array(position_ids, name="position_ids")
    if positions.ndim == 2:
        if positions.shape[0] != 1:
            raise ValueError("position_ids must have batch size one")
        positions = positions[0]
    if positions.shape != (n_tokens,):
        raise ValueError(
            f"position_ids must have shape [{n_tokens}], got {positions.shape}"
        )

    return split_into_pages(
        pre_rope_keys=keys[0],  # [1, H, T, D] -> [H, T, D]
        values=vals[0].transpose(1, 0, 2),  # [1, T, H, D] -> [H, T, D]
        source_positions=positions,
        inv_freq=inv_freq,
        page_size=page_size,
        physical_id_prefix=physical_id_prefix,
        layer_idx=layer_idx,
        **page_kwargs,
    )


def materialize_page(
    page: PageKV,
    *,
    session_id: str,
    shift: int | None = None,
    positions: Any | None = None,
    cos: Any | None = None,
    sin: Any | None = None,
    target_context_fingerprint: str | None = None,
    allow_unverified: bool = False,
    allow_position_reorder: bool = False,
) -> MaterializedPage:
    """Produce post-RoPE K/V for ``page`` in ``session_id``.

    ``shift`` applies one common offset to the page's source positions; this
    is the exact relocation path (all relative positions are preserved).
    ``positions`` gives explicit per-token targets and is accepted for deeper
    layers only when the resulting offsets are uniform, unless
    ``allow_position_reorder=True`` is set for a controlled ablation.  Values
    are never transformed.

    ``cos``/``sin`` override the regenerated target table.  Pass the model's
    live ``position_embeddings`` here when bit-level closeness matters or when
    the model applies a rotary variant (for example a YaRN attention factor)
    that ``inv_freq`` alone does not capture.

    Passing a target fingerprint that differs from the page's source
    fingerprint requires the page to be marked ``verified_cross_context`` or
    ``allow_unverified=True``.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a nonempty string")
    if shift is not None and positions is not None:
        raise ValueError("pass either shift or positions, not both")
    if (cos is None) != (sin is None):
        raise ValueError("pass cos and sin together")

    if page.scope == "session" and page.session_id != session_id:
        raise ValueError(
            f"page {page.physical_page_id!r} belongs to session "
            f"{page.session_id!r}, not {session_id!r}"
        )
    if page.scope == "global" and not (
        page.verified_cross_context or allow_unverified
    ):
        raise ValueError(
            f"global page {page.physical_page_id!r} has no verified cross-context "
            "fidelity; pass allow_unverified=True only for a controlled ablation"
        )
    # A page with no stored fingerprint (for example one built without
    # ``context_fingerprints``) is *unverifiable*, not auto-matching: an
    # explicitly asserted target context must not silently skip the gate.
    if (
        target_context_fingerprint is not None
        and target_context_fingerprint != page.context_fingerprint
        and not (page.verified_cross_context or allow_unverified)
    ):
        raise ValueError(
            f"page {page.physical_page_id!r} was encoded under a different "
            "context prefix; re-rotation is exact only when the prefix is unchanged"
        )

    if shift is not None:
        if not isinstance(shift, (int, np.integer)) or isinstance(shift, bool):
            raise TypeError("shift must be an integer")
        # Add in int64: unsigned position arrays would wrap modulo 2**bits
        # (uint8 [250, 251, 252] + 10 becomes [4, 5, 6]) instead of relocating.
        source_positions = page.source_positions.astype(np.int64, copy=False)
        target_positions = source_positions + int(shift)
        if np.any(target_positions < 0):
            raise ValueError("shift would produce negative positions")
    elif positions is not None:
        target_positions = _positions_array(positions, name="positions")
        if target_positions.shape != (page.n_tokens,):
            raise ValueError(
                f"target positions must have shape [{page.n_tokens}], "
                f"got {target_positions.shape}"
            )
        # Same int64 normalization: int64 - uint64 would promote to float64.
        offsets = target_positions - page.source_positions.astype(
            np.int64, copy=False
        )
        if (
            page.layer_idx > 0
            and not allow_position_reorder
            and np.any(offsets != offsets[0])
        ):
            raise ValueError(
                "non-uniform repositioning changes deeper layers' hidden states; "
                "use shift=<delta> for exact relocation or set "
                "allow_position_reorder=True for a controlled ablation"
            )
    else:
        target_positions = page.source_positions

    if cos is None:
        target_cos, target_sin = rope_cos_sin(target_positions, page.inv_freq)
    else:
        target_cos = _as_array(cos, name="cos")
        target_sin = _as_array(sin, name="sin")
        if target_cos.shape != target_sin.shape:
            raise ValueError(
                f"cos and sin shapes differ: {target_cos.shape} vs {target_sin.shape}"
            )
        try:
            target_cos = np.broadcast_to(
                target_cos, (page.n_kv_heads, page.n_tokens, page.head_dim)
            )
            target_sin = np.broadcast_to(
                target_sin, (page.n_kv_heads, page.n_tokens, page.head_dim)
            )
        except ValueError as exc:
            raise ValueError(
                "provided cos/sin must broadcast to "
                f"[{page.n_kv_heads}, {page.n_tokens}, {page.head_dim}]"
            ) from exc
    # The store holds pre-RoPE keys, so relocation is a single rotation at the
    # destination.  ``rerope`` is for already-rotated post-RoPE keys.
    keys = apply_rope(page.pre_rope_keys, target_cos, target_sin)

    return MaterializedPage(
        physical_page_id=page.physical_page_id,
        layer_idx=page.layer_idx,
        keys=keys,
        values=page.values,
        positions=np.array(target_positions, dtype=np.int64, copy=True),
        page_size=page.page_size,
        scope=page.scope,
        session_id=page.session_id,
        document_id=page.document_id,
        segment_id=page.segment_id,
    )


@dataclass(frozen=True)
class PackedPageBatch:
    """Materialized pages packed in ``KvPool`` physical layout.

    ``kv`` has shape ``[n_pages, 2, n_kv_heads, page_size, head_dim]`` and is
    zero-padded after each page's valid tokens.  ``valid_entries`` has shape
    ``[n_pages, n_kv_heads]`` and belongs to destination page-table entries
    (``InferState.page_valid_entries``), not to pool slot ids.
    """

    kv: np.ndarray
    valid_entries: np.ndarray
    page_ids: tuple[str, ...]
    layer_idx: int
    page_size: int

    @property
    def n_pages(self) -> int:
        return len(self.page_ids)

    @property
    def nbytes(self) -> int:
        return int(self.kv.nbytes + self.valid_entries.nbytes)

    def describe(self) -> dict[str, Any]:
        return {
            "n_pages": self.n_pages,
            "layer_idx": self.layer_idx,
            "page_size": self.page_size,
            "n_kv_heads": int(self.kv.shape[2]),
            "head_dim": int(self.kv.shape[4]),
            "dtype": str(self.kv.dtype),
            "nbytes": self.nbytes,
            "valid_entries": self.valid_entries.tolist(),
        }


def pack_pages(
    pages: Iterable[MaterializedPage],
    *,
    page_size: int | None = None,
) -> PackedPageBatch:
    """Pack materialized pages into IceCache's ``(2, H, page_size, D)`` layout."""
    pages = tuple(pages)
    if not pages:
        raise ValueError("cannot pack an empty page list")
    first = pages[0]
    layer_idx = first.layer_idx
    n_heads = first.n_kv_heads
    head_dim = int(first.keys.shape[2])
    dtype = first.keys.dtype
    page_size = first.page_size if page_size is None else int(page_size)
    if page_size < first.n_tokens:
        raise ValueError(
            f"page_size {page_size} is smaller than {first.n_tokens} tokens"
        )

    kv = np.zeros((len(pages), 2, n_heads, page_size, head_dim), dtype=dtype)
    valid_entries = np.zeros((len(pages), n_heads), dtype=np.int32)
    page_ids: list[str] = []
    for index, page in enumerate(pages):
        if page.layer_idx != layer_idx:
            raise ValueError(
                f"cannot pack layers {layer_idx} and {page.layer_idx} together"
            )
        if page.n_kv_heads != n_heads or int(page.keys.shape[2]) != head_dim:
            raise ValueError("packed pages must share head count and head_dim")
        if page.keys.dtype != dtype:
            raise ValueError(
                f"packed pages must share a dtype, got {dtype} and {page.keys.dtype}"
            )
        if page.values.dtype != dtype:
            # ``MaterializedPage`` validates this too, but ``pack_pages`` is a
            # public entry point and the assignment below would silently cast.
            raise TypeError(
                f"page {page.physical_page_id!r} has keys dtype {dtype} but "
                f"values dtype {page.values.dtype}"
            )
        if page.page_size != page_size:
            raise ValueError(
                f"page {page.physical_page_id!r} was built for page_size "
                f"{page.page_size}, not {page_size}"
            )
        if page.n_tokens > page_size:
            raise ValueError(
                f"page {page.physical_page_id!r} holds {page.n_tokens} tokens, "
                f"more than page_size {page_size}"
            )
        kv[index, 0, :, : page.n_tokens, :] = page.keys
        kv[index, 1, :, : page.n_tokens, :] = page.values
        valid_entries[index, :] = page.n_tokens
        page_ids.append(page.physical_page_id)

    return PackedPageBatch(
        kv=kv,
        valid_entries=valid_entries,
        page_ids=tuple(page_ids),
        layer_idx=layer_idx,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Store and semantic-page bridge
# ---------------------------------------------------------------------------


class KVPageStore:
    """Registry of physical pages plus a per-session semantic binding table.

    The store never mutates a ``PageKV`` after insertion.  ``bind_semantic``
    connects an entry from ``experimental.unified_page_index`` (a per-layer,
    per-head ``SemanticPage``) to the physical page that carries all KV heads
    for that layer and segment.  Session-scoped bindings are keyed by
    ``(semantic page id, session id)`` so two sessions can materialize the same
    logical RAG page into independent KV with independent costs.
    """

    def __init__(
        self,
        *,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        page_size: int | None = None,
    ) -> None:
        if n_kv_heads is not None and n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if head_dim is not None and head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if page_size is not None and page_size <= 0:
            raise ValueError("page_size must be positive")
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._pages: dict[str, PageKV] = {}
        # (semantic_page_id, session_id_or_None) -> physical_page_id
        self._semantic_bindings: dict[tuple[str, str | None], str] = {}

    def __len__(self) -> int:
        return len(self._pages)

    def __contains__(self, page_id: object) -> bool:
        return page_id in self._pages

    def add(self, page: PageKV) -> PageKV:
        if page.physical_page_id in self._pages:
            raise KeyError(f"duplicate physical page id {page.physical_page_id!r}")
        if self.n_kv_heads is not None and page.n_kv_heads != self.n_kv_heads:
            raise ValueError(
                f"page has {page.n_kv_heads} heads, store expects {self.n_kv_heads}"
            )
        if self.head_dim is not None and page.head_dim != self.head_dim:
            raise ValueError(
                f"page head_dim {page.head_dim}, store expects {self.head_dim}"
            )
        if self.page_size is not None and page.page_size != self.page_size:
            raise ValueError(
                f"page page_size {page.page_size}, store expects {self.page_size}"
            )
        self._pages[page.physical_page_id] = page
        return page

    def add_pages(self, pages: Iterable[PageKV]) -> None:
        for page in pages:
            self.add(page)

    def get(self, page_id: str) -> PageKV:
        try:
            return self._pages[page_id]
        except KeyError as exc:
            raise KeyError(f"unknown physical page id {page_id!r}") from exc

    def remove(self, page_id: str) -> PageKV:
        page = self.get(page_id)
        del self._pages[page_id]
        for key, bound in list(self._semantic_bindings.items()):
            if bound == page_id:
                del self._semantic_bindings[key]
        return page

    def remove_session(self, session_id: str) -> int:
        """Drop every session-scoped KV page and binding owned by a session."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        doomed = [
            page_id
            for page_id, page in self._pages.items()
            if page.scope == "session" and page.session_id == session_id
        ]
        for page_id in doomed:
            self.remove(page_id)
        for key in list(self._semantic_bindings):
            if key[1] == session_id:
                del self._semantic_bindings[key]
        return len(doomed)

    def page_ids(
        self,
        *,
        session_id: str | None = None,
        scope: PageScope | None = None,
        document_id: str | None = None,
    ) -> list[str]:
        result = []
        for page_id, page in self._pages.items():
            if scope is not None and page.scope != scope:
                continue
            if document_id is not None and page.document_id != document_id:
                continue
            if session_id is not None:
                if page.scope == "session" and page.session_id != session_id:
                    continue
                if page.scope == "global":
                    # Global pages are visible, but only a verified page can be
                    # materialized; listing them here is a visibility query.
                    pass
            result.append(page_id)
        return sorted(result)

    def _resolve_position_spec(
        self,
        page_ids: Sequence[str],
        positions: Any | None,
    ) -> list[np.ndarray | None]:
        if positions is None:
            return [None] * len(page_ids)
        if isinstance(positions, Mapping):
            resolved = []
            for page_id in page_ids:
                if page_id not in positions:
                    raise ValueError(f"positions mapping is missing {page_id!r}")
                resolved.append(_positions_array(positions[page_id], name="positions"))
            return resolved
        array = _as_array(positions, name="positions")
        if array.ndim == 1:
            shared = _positions_array(array, name="positions")
            for page_id in page_ids:
                if self.get(page_id).n_tokens != shared.size:
                    raise ValueError(
                        f"shared positions have {shared.size} entries but page "
                        f"{page_id!r} holds {self.get(page_id).n_tokens} tokens"
                    )
            return [shared] * len(page_ids)
        if array.ndim == 2:
            if array.shape[0] != len(page_ids):
                raise ValueError(
                    f"positions has {array.shape[0]} rows for {len(page_ids)} pages"
                )
            return [
                _positions_array(array[index], name="positions")
                for index in range(len(page_ids))
            ]
        raise ValueError("positions must be 1D, 2D, or a mapping from page id")

    def _fingerprint_for(
        self,
        page_id: str,
        context_fingerprints: str | Mapping[str, str] | None,
    ) -> str | None:
        if context_fingerprints is None:
            return None
        if isinstance(context_fingerprints, str):
            return context_fingerprints
        if page_id not in context_fingerprints:
            # ``.get()`` returned None, which made ``materialize_page`` skip
            # the cross-context check entirely: a typo'd key silently disabled
            # the safety gate.  Match the positions mapping, which raises.
            raise ValueError(
                f"context_fingerprints mapping is missing {page_id!r}"
            )
        return context_fingerprints[page_id]

    def materialize(
        self,
        page_ids: Iterable[str],
        *,
        session_id: str,
        shift: int | None = None,
        positions: Any | None = None,
        cos: Any | None = None,
        sin: Any | None = None,
        context_fingerprints: str | Mapping[str, str] | None = None,
        allow_unverified: bool = False,
        allow_position_reorder: bool = False,
    ) -> list[MaterializedPage]:
        """Materialize pages; ``shift`` is the exact whole-span relocation path.

        Materialize every page of one encoding in a single call with one
        ``shift``.  The per-page check cannot see that two calls used different
        deltas, and mixing deltas across pages of the same context breaks the
        deeper-layer exactness precondition even though each page individually
        looks like a uniform shift.
        """
        if shift is not None and positions is not None:
            raise ValueError("pass either shift or positions, not both")
        if (cos is None) != (sin is None):
            raise ValueError("pass cos and sin together")
        page_ids = list(page_ids)
        if not page_ids:
            return []
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("page_ids contains duplicates")
        pages = [self.get(page_id) for page_id in page_ids]
        if shift is not None:
            if not isinstance(shift, (int, np.integer)) or isinstance(shift, bool):
                raise TypeError("shift must be an integer")
            position_specs = [None] * len(page_ids)
        else:
            position_specs = self._resolve_position_spec(page_ids, positions)
        return [
            materialize_page(
                page,
                session_id=session_id,
                shift=shift,
                positions=position_spec,
                cos=cos,
                sin=sin,
                target_context_fingerprint=self._fingerprint_for(
                    page_id, context_fingerprints
                ),
                allow_unverified=allow_unverified,
                allow_position_reorder=allow_position_reorder,
            )
            for page, page_id, position_spec in zip(pages, page_ids, position_specs)
        ]

    def pack(
        self,
        page_ids: Iterable[str],
        *,
        session_id: str,
        shift: int | None = None,
        positions: Any | None = None,
        cos: Any | None = None,
        sin: Any | None = None,
        context_fingerprints: str | Mapping[str, str] | None = None,
        allow_unverified: bool = False,
        allow_position_reorder: bool = False,
        page_size: int | None = None,
    ) -> PackedPageBatch:
        materialized = self.materialize(
            page_ids,
            session_id=session_id,
            shift=shift,
            positions=positions,
            cos=cos,
            sin=sin,
            context_fingerprints=context_fingerprints,
            allow_unverified=allow_unverified,
            allow_position_reorder=allow_position_reorder,
        )
        return pack_pages(materialized, page_size=page_size)

    def bind_semantic(
        self,
        semantic_page: Any,
        physical_page_id: str,
        *,
        session_id: str | None = None,
    ) -> None:
        """Bind a ``SemanticPage``-like object to a physical page.

        ``session_id`` is required when the physical page is session-scoped and
        must be omitted for global physical pages.  A prompt ``SemanticPage``
        must bind to a session page from its own session; a RAG page may bind
        either to a global artifact or to a session-local materialization.
        """
        page = self.get(physical_page_id)
        semantic_id = getattr(semantic_page, "page_id", None)
        if not isinstance(semantic_id, str) or not semantic_id:
            raise ValueError("semantic_page.page_id must be a nonempty string")
        layer_idx = getattr(semantic_page, "layer_idx", None)
        head_idx = getattr(semantic_page, "head_idx", 0)
        if layer_idx is not None and int(layer_idx) != page.layer_idx:
            raise ValueError(
                f"semantic page layer {layer_idx} differs from KV page layer "
                f"{page.layer_idx}"
            )
        if not 0 <= int(head_idx) < page.n_kv_heads:
            raise ValueError(
                f"semantic page head {head_idx} out of range for "
                f"{page.n_kv_heads} KV heads"
            )

        source = getattr(semantic_page, "source", None)
        if page.scope == "global":
            if session_id is not None:
                raise ValueError("global physical pages have no session binding")
        else:
            if session_id is None:
                session_id = getattr(semantic_page, "session_id", None)
            if session_id != page.session_id:
                raise ValueError(
                    f"binding session {session_id!r} differs from physical page "
                    f"session {page.session_id!r}"
                )
        if source == "prompt":
            if page.scope != "session":
                raise ValueError("prompt pages must bind to a session-scoped KV page")
            if getattr(semantic_page, "session_id", None) != page.session_id:
                raise ValueError(
                    "prompt semantic page and physical KV page have different owners"
                )

        key = (semantic_id, session_id)
        existing = self._semantic_bindings.get(key)
        if existing is not None and existing != physical_page_id:
            raise ValueError(
                f"semantic page {semantic_id!r} is already bound to {existing!r}"
            )
        self._semantic_bindings[key] = physical_page_id

    def resolve_semantic(
        self,
        semantic_page: Any,
        *,
        session_id: str,
    ) -> PageKV:
        """Resolve a semantic page to physical KV for ``session_id``.

        A session-specific binding wins over a global binding, which lets one
        RAG catalog entry point at a shared artifact or at a session-local
        re-encoding without mutating global metadata.
        """
        semantic_id = getattr(semantic_page, "page_id", None)
        if not isinstance(semantic_id, str) or not semantic_id:
            raise ValueError("semantic_page.page_id must be a nonempty string")
        page_id = self._semantic_bindings.get((semantic_id, session_id))
        if page_id is None:
            page_id = self._semantic_bindings.get((semantic_id, None))
        if page_id is None:
            raise KeyError(
                f"semantic page {semantic_id!r} has no KV binding for "
                f"session {session_id!r}"
            )
        page = self.get(page_id)
        if page.scope == "session" and page.session_id != session_id:
            raise ValueError(
                f"semantic page {semantic_id!r} resolved to a page owned by "
                f"session {page.session_id!r}"
            )
        return page

    def storage_report(self) -> dict[str, Any]:
        pages = list(self._pages.values())
        sessions = sorted(
            {page.session_id for page in pages if page.session_id is not None}
        )
        return {
            "n_pages": len(pages),
            "n_global_pages": sum(page.scope == "global" for page in pages),
            "n_session_pages": sum(page.scope == "session" for page in pages),
            "n_verified_global_pages": sum(
                page.scope == "global" and page.verified_cross_context
                for page in pages
            ),
            "n_tokens": sum(page.n_tokens for page in pages),
            "n_bytes": sum(page.nbytes for page in pages),
            "sessions": sessions,
            "n_semantic_bindings": len(self._semantic_bindings),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test(verbose: bool = False) -> int:
    """Exercise rotation identities, page packing, and the safety gates.

    Deliberately lives in the library module (not ``tests/``) so the
    materialization contract can be checked without installing the package.
    """
    rng = np.random.default_rng(0)
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    def close(a: np.ndarray, b: np.ndarray, atol: float = 1e-5) -> bool:
        return np.allclose(a, b, atol=atol, rtol=1e-5)

    def raises(exc_type, fn, *args, **kwargs) -> None:
        nonlocal checks
        checks += 1
        try:
            fn(*args, **kwargs)
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}, call returned")

    n_heads, n_tokens, head_dim = 2, 5, 8
    page_size = 8
    positions = np.arange(10, 10 + n_tokens, dtype=np.int64)
    target_positions = positions + 100
    base = 10_000.0
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))

    pre_rope = rng.normal(size=(n_heads, n_tokens, head_dim)).astype(np.float32)
    values = rng.normal(size=(n_heads, n_tokens, head_dim)).astype(np.float32)

    src_cos, src_sin = rope_cos_sin(positions, inv_freq)
    dst_cos, dst_sin = rope_cos_sin(target_positions, inv_freq)

    # 1. Llama half-split convention and round trip.
    rotated = apply_rope(pre_rope, src_cos, src_sin)
    recovered = remove_rope(rotated, src_cos, src_sin)
    check(close(pre_rope, recovered), "RoPE round trip")
    check(
        np.allclose(
            rotate_half(np.array([[1.0, 2.0, 3.0, 4.0]])),
            np.array([[-3.0, -4.0, 1.0, 2.0]]),
        ),
        "rotate_half half-split convention",
    )

    # 2. Re-rotation equals direct rotation of the pre-RoPE key.
    moved = rerope(rotated, src_cos, src_sin, dst_cos, dst_sin)
    direct = apply_rope(pre_rope, dst_cos, dst_sin)
    check(close(moved, direct, atol=2e-5), "re-rotation equals direct rotation")

    # 3. Same-position materialization matches a fresh rotation.
    page = PageKV(
        physical_page_id="doc/p0/l0",
        layer_idx=0,
        pre_rope_keys=pre_rope,
        values=values,
        source_positions=positions,
        inv_freq=inv_freq,
        page_size=page_size,
        scope="session",
        session_id="s1",
        context_fingerprint="prefix-a",
    )
    store = KVPageStore(n_kv_heads=n_heads, head_dim=head_dim, page_size=page_size)
    store.add(page)
    same = store.materialize(["doc/p0/l0"], session_id="s1")[0]
    check(close(same.keys, rotated, atol=2e-5), "same-position materialization")
    check(np.array_equal(same.positions, positions), "same-position indices")
    check(np.array_equal(same.values, values), "values are not transformed")
    check(np.array_equal(same.valid_entries, [n_tokens, n_tokens]), "valid entries")

    # 4. Shifted materialization matches encoding the page at the target.
    shifted = store.materialize(
        ["doc/p0/l0"], session_id="s1", positions=target_positions
    )[0]
    check(close(shifted.keys, direct, atol=2e-5), "shifted materialization")
    check(np.array_equal(shifted.positions, target_positions), "shifted indices")

    via_shift = store.materialize(["doc/p0/l0"], session_id="s1", shift=100)[0]
    check(close(via_shift.keys, shifted.keys), "shift=100 equals explicit +100 positions")
    try:
        store.materialize(
            ["doc/p0/l0"], session_id="s1", shift=1, positions=target_positions
        )
    except ValueError:
        pass
    else:
        raise AssertionError("shift and positions were accepted together")

    exact_cos = dst_cos.astype(np.float32)
    exact_sin = dst_sin.astype(np.float32)
    with_table = store.materialize(
        ["doc/p0/l0"],
        session_id="s1",
        positions=target_positions,
        cos=exact_cos,
        sin=exact_sin,
    )[0]
    check(
        close(with_table.keys, apply_rope(pre_rope, exact_cos, exact_sin), atol=1e-7),
        "caller-provided rotary table",
    )
    try:
        store.materialize(
            ["doc/p0/l0"], session_id="s1", cos=exact_cos
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cos without sin was accepted")
    try:
        store.materialize(
            ["doc/p0/l0"],
            session_id="s1",
            cos=np.ones((n_tokens, head_dim // 2)),
            sin=np.zeros((n_tokens, head_dim // 2)),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-width rotary table was accepted")

    # 4b. Uniform relocation is exact at every layer; non-uniform reordering is
    # not, because deeper hidden states depend on relative positions.
    layer_one = PageKV(
        physical_page_id="doc/p0/l1",
        layer_idx=1,
        pre_rope_keys=pre_rope,
        values=values,
        source_positions=positions,
        inv_freq=inv_freq,
        page_size=page_size,
        scope="session",
        session_id="s1",
        context_fingerprint="prefix-a",
    )
    store.add(layer_one)
    uniform_l1 = store.materialize(
        ["doc/p0/l1"], session_id="s1", positions=target_positions
    )[0]
    check(close(uniform_l1.keys, direct, atol=2e-5), "uniform layer-1 relocation")
    reordered = target_positions + np.arange(n_tokens)
    try:
        store.materialize(["doc/p0/l1"], session_id="s1", positions=reordered)
    except ValueError as exc:
        check("non-uniform repositioning" in str(exc), "non-uniform layer-1 gate")
    else:
        raise AssertionError("non-uniform layer-1 relocation escaped the gate")
    layer_zero_reorder = store.materialize(
        ["doc/p0/l0"], session_id="s1", positions=reordered
    )[0]
    reorder_cos, reorder_sin = rope_cos_sin(reordered, inv_freq)
    check(
        close(
            layer_zero_reorder.keys,
            apply_rope(pre_rope, reorder_cos, reorder_sin),
            atol=2e-5,
        ),
        "layer-0 reordering is just a rotation",
    )
    ablation = store.materialize(
        ["doc/p0/l1"],
        session_id="s1",
        positions=reordered,
        allow_position_reorder=True,
    )[0]
    check(
        close(ablation.keys, layer_zero_reorder.keys),
        "explicit reorder ablation at layer 1",
    )

    # 5. Packing pads the partial page and keeps the HND pool layout.
    packed = store.pack(["doc/p0/l0"], session_id="s1")
    check(packed.kv.shape == (1, 2, n_heads, page_size, head_dim), "packed shape")
    check(close(packed.kv[0, 0, :, :n_tokens], same.keys, atol=2e-5), "packed K")
    check(close(packed.kv[0, 1, :, :n_tokens], values), "packed V")
    check(np.all(packed.kv[0, 0, :, n_tokens:] == 0), "K padding is zero")
    check(np.all(packed.valid_entries == n_tokens), "packed valid entries")

    # 6. Recovering pre-RoPE keys from rotated storage.
    recovered_page = PageKV.from_post_rope(
        post_rope_keys=rotated,
        values=values,
        source_positions=positions,
        inv_freq=inv_freq,
        physical_page_id="doc/p1/l0",
        layer_idx=0,
        page_size=page_size,
        scope="session",
        session_id="s1",
    )
    check(
        close(recovered_page.pre_rope_keys, pre_rope, atol=2e-5),
        "from_post_rope inverse",
    )

    # 7. Session isolation.
    try:
        store.materialize(["doc/p0/l0"], session_id="s2")
    except ValueError as exc:
        check("belongs to session" in str(exc), "session isolation error message")
    else:
        raise AssertionError("session page leaked to another session")

    # 8. Cross-context gates for global pages.
    global_page = PageKV(
        physical_page_id="rag/p0/l0",
        layer_idx=0,
        pre_rope_keys=pre_rope,
        values=values,
        source_positions=positions,
        inv_freq=inv_freq,
        page_size=page_size,
        scope="global",
        document_id="doc-0",
    )
    store.add(global_page)
    try:
        store.materialize(["rag/p0/l0"], session_id="s1", allow_unverified=False)
    except ValueError as exc:
        check("verified cross-context" in str(exc), "global page default gate")
    else:
        raise AssertionError("unverified global page escaped the fidelity gate")
    unverified = store.materialize(
        ["rag/p0/l0"], session_id="s1", allow_unverified=True
    )[0]
    check(close(unverified.keys, rotated, atol=2e-5), "explicit unverified materialization")

    # 9. A changed prefix is refused even for a session page.
    try:
        store.materialize(
            ["doc/p0/l0"], session_id="s1", context_fingerprints="prefix-b"
        )
    except ValueError as exc:
        check("different context prefix" in str(exc), "prefix gate")
    else:
        raise AssertionError("changed-prefix materialization was accepted")
    store.materialize(
        ["doc/p0/l0"], session_id="s1", context_fingerprints="prefix-a"
    )

    # 10. Semantic-page bridge, including per-session RAG bindings.
    class _SemanticPage:
        def __init__(self, page_id, source, session_id=None, head_idx=0):
            self.page_id = page_id
            self.layer_idx = 0
            self.head_idx = head_idx
            self.source = source
            self.session_id = session_id

    prompt = _SemanticPage("prompt:7", "prompt", session_id="s1", head_idx=1)
    store.bind_semantic(prompt, "doc/p0/l0")
    check(store.resolve_semantic(prompt, session_id="s1") is page, "prompt binding")
    rag = _SemanticPage("rag:doc-0:seg0", "rag")
    store.bind_semantic(rag, "rag/p0/l0")
    check(
        store.resolve_semantic(rag, session_id="s1") is global_page,
        "global RAG binding",
    )
    session_rag = PageKV(
        physical_page_id="rag/s1/p0/l0",
        layer_idx=0,
        pre_rope_keys=pre_rope,
        values=values,
        source_positions=positions,
        inv_freq=inv_freq,
        page_size=page_size,
        scope="session",
        session_id="s1",
        document_id="doc-0",
    )
    store.add(session_rag)
    store.bind_semantic(rag, "rag/s1/p0/l0", session_id="s1")
    check(
        store.resolve_semantic(rag, session_id="s1") is session_rag,
        "session-specific binding wins",
    )
    check(
        store.resolve_semantic(rag, session_id="s2") is global_page,
        "other sessions fall back to the global binding, not the session page",
    )
    try:
        store.bind_semantic(
            _SemanticPage("prompt:8", "prompt", session_id="s2"), "doc/p0/l0"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("foreign prompt page bound to a session KV page")

    # 11. Long spans split into page-aligned pages, including a partial tail.
    long_tokens = 20
    long_positions = np.arange(1000, 1000 + long_tokens, dtype=np.int64)
    long_keys = rng.normal(size=(n_heads, long_tokens, head_dim)).astype(np.float32)
    long_values = rng.normal(size=(n_heads, long_tokens, head_dim)).astype(np.float32)
    split_store = KVPageStore(n_kv_heads=n_heads, head_dim=head_dim, page_size=page_size)
    split_pages = split_into_pages(
        pre_rope_keys=long_keys,
        values=long_values,
        source_positions=long_positions,
        inv_freq=inv_freq,
        page_size=page_size,
        physical_id_prefix="doc/long/seg0",
        layer_idx=0,
        scope="session",
        session_id="s3",
        document_id="doc-long",
        segment_id="seg0",
        context_fingerprints=["prefix-0", "prefix-1", "prefix-2"],
    )
    split_store.add_pages(split_pages)
    check(len(split_pages) == 3, "split page count")
    check([page.n_tokens for page in split_pages] == [8, 8, 4], "split page sizes")
    check(
        np.array_equal(
            np.concatenate([page.pre_rope_keys for page in split_pages], axis=1),
            long_keys,
        ),
        "split preserves keys",
    )
    check(
        np.array_equal(
            np.concatenate([page.source_positions for page in split_pages]),
            long_positions,
        ),
        "split preserves positions",
    )
    check(
        [page.physical_page_id for page in split_pages]
        == ["doc/long/seg0/p00000", "doc/long/seg0/p00001", "doc/long/seg0/p00002"],
        "split page ids",
    )
    shifted_long = long_positions + 300
    shift_mapping = {
        page.physical_page_id: shifted_long[index * page_size : index * page_size + page.n_tokens]
        for index, page in enumerate(split_pages)
    }
    split_materialized = split_store.materialize(
        [page.physical_page_id for page in split_pages],
        session_id="s3",
        positions=shift_mapping,
        context_fingerprints={
            page.physical_page_id: page.context_fingerprint for page in split_pages
        },
    )
    for index, (page, materialized) in enumerate(zip(split_pages, split_materialized)):
        start = index * page_size
        end = start + page.n_tokens
        shift_cos, shift_sin = rope_cos_sin(shifted_long[start:end], inv_freq)
        check(
            close(
                materialized.keys,
                apply_rope(long_keys[:, start:end], shift_cos, shift_sin),
                atol=2e-5,
            ),
            f"split page {index} shifted materialization",
        )
    split_batch = split_store.pack(
        [page.physical_page_id for page in split_pages],
        session_id="s3",
        positions=shift_mapping,
        context_fingerprints={
            page.physical_page_id: page.context_fingerprint for page in split_pages
        },
    )
    check(
        np.array_equal(split_batch.valid_entries[:, 0], [8, 8, 4]),
        "split packed valid entries",
    )
    check(
        np.all(split_batch.kv[2, :, :, 4:] == 0),
        "partial split page is zero-padded",
    )

    # 12. Capture helper accepts the adapter's key/value layouts.
    model_keys = long_keys[None]  # [1, H, T, D]
    model_values = long_values.transpose(1, 0, 2)[None]  # [1, T, H, D]
    model_pages = pages_from_model_kv(
        model_keys,
        model_values,
        long_positions[None],
        inv_freq,
        page_size=page_size,
        physical_id_prefix="doc/model/seg0",
        layer_idx=0,
        scope="session",
        session_id="s4",
        document_id="doc-model",
        segment_id="seg0",
    )
    check(len(model_pages) == 3, "model capture page count")
    check(
        np.array_equal(
            np.concatenate([page.pre_rope_keys for page in model_pages], axis=1),
            long_keys,
        ),
        "model capture keys",
    )
    check(
        np.array_equal(
            np.concatenate([page.values for page in model_pages], axis=1),
            long_values,
        ),
        "model capture values",
    )
    check(
        np.array_equal(
            np.concatenate([page.source_positions for page in model_pages]),
            long_positions,
        ),
        "model capture positions",
    )
    try:
        pages_from_model_kv(
            np.concatenate([model_keys, model_keys], axis=0),
            np.concatenate([model_values, model_values], axis=0),
            np.stack([long_positions, long_positions]),
            inv_freq,
            page_size=page_size,
            physical_id_prefix="doc/model/bad",
            layer_idx=0,
            scope="session",
            session_id="s4",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("batched model capture was accepted")

    # 13. Store lifecycle and accounting.
    report = store.storage_report()
    check(report["n_pages"] == 4, "storage report page count")
    check(report["n_global_pages"] == 1 and report["n_session_pages"] == 3, "scopes")
    check(report["n_bytes"] > 0 and report["n_semantic_bindings"] == 3, "accounting")
    check(store.remove_session("s1") == 3, "remove_session removes owned pages")
    check(
        "doc/p0/l0" not in store
        and "doc/p0/l1" not in store
        and "rag/s1/p0/l0" not in store,
        "removed pages",
    )
    check(store.resolve_semantic(rag, session_id="s1") is global_page, "global survives")

    # 14. Input validation should fail closed.
    try:
        PageKV(
            physical_page_id="bad",
            layer_idx=0,
            pre_rope_keys=pre_rope,
            values=values,
            source_positions=positions,
            inv_freq=inv_freq,
            page_size=4,
            scope="session",
            session_id="s1",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("overfull page accepted")
    try:
        PageKV(
            physical_page_id="bad2",
            layer_idx=0,
            pre_rope_keys=pre_rope,
            values=values,
            source_positions=positions,
            inv_freq=inv_freq,
            page_size=page_size,
            scope="global",
            session_id="s1",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("global page with session id accepted")

    # 5. Fail-closed edges found by an independent review.
    raises(
        ValueError,
        rope_cos_sin,
        np.zeros((2, 3), dtype=np.int64),
        inv_freq,
    )
    raises(
        ValueError,
        lambda: PageKV(
            physical_page_id="bad/uint",
            layer_idx=0,
            pre_rope_keys=pre_rope,
            values=values,
            source_positions=np.array([5, 0, 1, 2, 3], dtype=np.uint8),
            inv_freq=inv_freq,
            page_size=page_size,
            scope="session",
            session_id="s1",
        ),
    )
    # ``remove_session`` above emptied the original store, and an earlier loop
    # rebound ``page``; build a fresh page/store so the mapping checks reach
    # the gate they exercise.
    gate_page = PageKV(
        physical_page_id="doc/p0/l0",
        layer_idx=0,
        pre_rope_keys=pre_rope,
        values=values,
        source_positions=positions,
        inv_freq=inv_freq,
        page_size=page_size,
        scope="session",
        session_id="s1",
        context_fingerprint="prefix-a",
    )
    fresh = KVPageStore(
        n_kv_heads=n_heads, head_dim=head_dim, page_size=page_size
    )
    fresh.add(gate_page)
    raises(
        ValueError,
        fresh.materialize,
        ["doc/p0/l0"],
        session_id="s1",
        positions=np.array([5, 0, 1, 2, 3], dtype=np.uint8),
    )
    raises(
        ValueError,
        fresh.materialize,
        ["doc/p0/l0"],
        session_id="s1",
        context_fingerprints={"typo-key": "prefix-b"},
    )
    # An unknown *source* fingerprint is unverifiable, not auto-matching: an
    # explicitly asserted target context must still fail closed.
    no_fingerprint = PageKV(
        physical_page_id="doc/no-fingerprint",
        layer_idx=0,
        pre_rope_keys=pre_rope,
        values=values,
        source_positions=positions,
        inv_freq=inv_freq,
        page_size=page_size,
        scope="session",
        session_id="s1",
    )
    raises(
        ValueError,
        materialize_page,
        no_fingerprint,
        session_id="s1",
        target_context_fingerprint="prefix-b",
    )
    # A positive shift on an unsigned position array must relocate, not wrap
    # modulo 2**bits (uint8 [250..254] + 10 would become [4..8]).
    uint_page = PageKV(
        physical_page_id="doc/uint",
        layer_idx=0,
        pre_rope_keys=pre_rope,
        values=values,
        source_positions=np.array([250, 251, 252, 253, 254], dtype=np.uint8),
        inv_freq=inv_freq,
        page_size=page_size,
        scope="session",
        session_id="s1",
    )
    relocated = materialize_page(uint_page, session_id="s1", shift=10)
    check(
        np.array_equal(
            relocated.positions, np.arange(260, 265, dtype=np.int64)
        ),
        "unsigned source_positions relocate without wrapping",
    )
    raises(
        TypeError,
        lambda: MaterializedPage(
            physical_page_id="mixed",
            layer_idx=0,
            keys=same.keys,
            values=same.values.astype(np.float64),
            positions=same.positions,
            page_size=page_size,
            scope="session",
            session_id="s1",
            document_id=None,
            segment_id=None,
        ),
    )

    class _DuckPage:
        pass

    duck = _DuckPage()
    duck.physical_page_id = "duck/mixed"
    duck.layer_idx = 0
    duck.keys = np.array(same.keys)
    duck.values = np.array(same.values, dtype=np.float64)
    duck.positions = np.array(same.positions)
    duck.page_size = page_size
    duck.n_kv_heads = n_heads
    duck.n_tokens = n_tokens
    raises(TypeError, pack_pages, [duck])

    if verbose:
        print(f"kv_materialization self-test: {checks} checks passed")
    return checks


if __name__ == "__main__":
    _self_test(verbose=True)
