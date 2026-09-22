"""Design A: exact-kNN page packing with a matmul scan over page representatives.

Self-contained module (no import of ``infer_state``/``dciknn``). Specification:
``experiment/design_a_spec.md`` (including the corrections in section 6b, which
this file follows where they override sections 1-6).

Per (layer, KV head) the structure is

* ``token2page`` / ``offset_in_page`` -- the partition of the indexed prefill
  tokens into pages of ``page_size``, built by capacity-constrained greedy
  expansion along an exact kNN (seed order = descending ``||k||``, expansion =
  argmax inner product against the *seed*);
* ``reps`` -- the float32 mean of each page's members, the vector the decode-time
  scan scores against.

Two page counts, never conflated (spec section 6b, C1):

* ``n_built`` -- pages that exist and are selectable. Starts at ``ceil(N/P)`` and
  grows only when :meth:`PageScan.insert` emits a page. ``query`` never returns
  an id ``>= n_built`` (per head), so no unbuilt page can win a slot and no
  address-less page can reach ``recall``.
* ``n_pages`` -- the pre-reserved CPU address space, fixed after :meth:`build`.

The query scan has two implementations that must agree: numpy on the CPU (spec
section 6b, C7) and, when ``_reps_t`` exists, a device ``bmm``+``topk`` scan.
The device path is the one the deployed backend uses, and it is NOT free of
transfers: ``_DCI_query`` receives the query as a CPU tensor (the caller moves
``query_states`` to the host before dispatch), so every query pays a host->device
copy of ``q`` plus a device->host copy of the ``[H, budget]`` result. That result
copy is an implicit stream synchronisation, which means ``query_seconds``
includes whatever was already queued on the stream and is therefore
queue-dependent rather than a pure scan latency.

The two paths implement the same selection rule up to ties: with exactly equal
scores ``torch.topk`` and ``numpy.argpartition``+``argsort`` may choose different
members at the budget boundary, and because ``_apply_selected_pages`` fills
``evicted_idx`` positionally a tie can also perturb the row order.

Known deviation from the literal pseudocode of spec section 2: where two scores
are *exactly* equal, the collapse (C6) and the sequential greedy can disagree,
because ``torch.topk`` does not define the order of equal keys while ``argmax``
takes the lowest index. The partition is still valid, and the two agree token for
token on every tie-free case (``--self-test`` compares them on random and
structured inputs). Exact ties need near-identical keys, which the greedy keeps
apart in practice.

Reservation contract (spec section 6b, C1). The page-id space is allocated once,
by :meth:`PageScan.build`, and never grows -- every reserved page already has a
CPU address and a ``kvc_capacity`` slot (``infer_state._page_scan_layout``) --
so the space left for :meth:`PageScan.insert` is a **hard cap**. A caller
declares what it needs per head through ``generation_reserve_tokens`` and gets
``reserve_pages_for``'s answer, ``ceil(tokens / page_size)`` pages plus
``DEFAULT_RESERVE_MARGIN_PAGES``, so the cap is never exactly the declared
budget. The declaration is checked when the object is constructed, and a batch
that the remaining reservation provably cannot hold is refused by
:meth:`PageScan.insert` *before* it mutates anything, instead of surfacing
part-way through a later flush deep into a generation. Nothing grows the space
mid-generation; the only recovery from a refused insert is to reserve more at
the next prefill.

:meth:`PageScan.insert` is atomic: it validates the whole batch first and
journals the pre-image of everything it touches, so a failure *or an interrupt*
anywhere in the per-key loop unwinds the batch exactly and leaves the object
byte-identical to what it was when the call started.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

DEFAULT_PAGE_SIZE = 16
# Extra pages reserved by default for decode-time inserts (64 pages = 1024
# decode tokens per head before the reservation is exhausted).
DEFAULT_RESERVE_PAGES = 64
# Slack added on top of a declared decode budget, so that the hard cap is not
# exactly the budget (D2: the old deployment reserved precisely
# `ceil(4096 / 16) = 256` pages and died on the 257th). Sized to the largest
# single flush the deployment issues -- `offload_ratio` (2) window pages, i.e.
# 32 keys per head, which a full page space turns into `ceil(32 / 16) = 2`
# emitted pages -- rounded up to a comfortable 8 pages = 128 tokens per head.
DEFAULT_RESERVE_MARGIN_PAGES = 8


def reserve_pages_for(generation_reserve_tokens, page_size,
                      margin_pages=DEFAULT_RESERVE_MARGIN_PAGES):
    """Pages to reserve for ``generation_reserve_tokens`` of decode, plus margin.

    ``ceil(tokens / page_size)`` is exactly what the declared tokens occupy, so
    that count alone *is* the declared budget: 0% margin, with the flush that
    carries the first token past the declaration landing on the cap. The margin
    keeps the cap off that boundary -- the flushes in flight, and a deployment
    that inserts a page more than once (the window path flushes pages out of a
    ring), both draw on the same page space. Raises :class:`PageScanError` for a
    negative budget or margin; a negative budget reserves *less* than the
    generation it declares, which no margin can make work.
    """
    if page_size <= 0:
        raise PageScanError("page_size must be positive")
    if generation_reserve_tokens < 0:
        raise PageScanError("generation_reserve_tokens must be non-negative")
    if margin_pages < 0:
        raise PageScanError("margin_pages must be non-negative")
    return -(-int(generation_reserve_tokens) // int(page_size)) + int(margin_pages)


def first_k_unique(row, k):
    """First ``k`` unique values of ``row``, in order of first occurrence.

    Byte-for-byte the same semantics as ``icecache.utils.first_k_unique`` (which
    ``infer_state._DCI_query`` applies to the interleaved per-q-head page ids).
    """
    _, idx = np.unique(row, return_index=True)
    return row[np.sort(idx)[:k]]


class PageScanError(RuntimeError):
    """Invalid configuration, or the pre-reserved page space was exhausted."""


def greedy_packed_pages(k, page_size, use_sim=True):
    """Greedy-pack ``k`` into pages of ``page_size`` (spec section 2).

    ``k`` is ``[M, N, D]`` float32. Returns ``packed`` ``[M, N]`` int32 holding
    ``page * page_size + slot`` per token, ``-1`` for a token never assigned.

    The greedy is one seed per page plus ``page_size - 1`` expansions:

    * seed -- highest ``||k||`` still unassigned (equals the next entry of the
      descending-norm seed order);
    * expansion -- argmax inner product against the *seed*, over the whole
      unassigned set. The seed's score row is fixed for the page and the masked
      entry set only shrinks, so the sequential "argmax, mask, argmax, ..."
      picks exactly the descending order of one ``topk`` over that row (spec
      section 6b, C6). Exact, not approximate.

    Every operation is row-independent, so packing ``M`` heads together --
    including heads belonging to *different layers* -- gives bit-identical
    per-head results to packing them one at a time. Measured on two real
    prefill layers (M=16, N=11770): 0/188320 differing entries.

    ``use_sim`` materialises the exact ``[M, N, N]`` similarity once and gathers
    each seed row out of it. That is the fastest route for a single layer (8
    heads -> 8.27 GB) but impossible for a cross-layer batch (96 heads -> 99 GB),
    so a batched caller sets ``use_sim=False`` and recomputes just the one row it
    needs per iteration as an ``[M, 1, D] @ [M, D, N]`` bmm. That trades away
    holding ``N**2`` floats per head for ~4x more memory traffic (~790 GB at
    M=96, so bandwidth-bound at ~1.4 TB/s) -- and still beats ``M`` separate
    builds 4.27x, because every other op in the loop is launch-latency-bound and
    costs the same at M=8 as at M=96 (316 ms/layer -> 66 ms/layer).

    ``use_sim=False`` on CUDA is served by :func:`_greedy_packed_pages_live`,
    which updates the row scan to read only the tokens that can still be chosen
    and is required to be output-identical to :func:`_greedy_packed_pages_rescan`
    (the batched loop as it stood before, kept here as the reference the
    self-test holds the live path to).

    On CPU the live path is *not* taken. Its bit-identity rests on one property
    of the row gemv -- that the accumulator for a column does not depend on which
    other columns are in the operand -- which the production CUDA kernel has (one
    thread per column, a fixed loop over ``D``) but CPU BLAS does not: blocking
    is chosen from the operand shape, so a compacted scan rounds differently.
    Measured on the fuzz case that first exposed it (M=5, N=584, P=9, D=14, the
    page whose live set has L=403 columns): gathering those exact columns out of
    the full ``bmm`` row and recomputing them against the compacted operand gives
    6 of 2015 entries off by one ULP (max relative error 1.0e-07), and 1275 of
    2015 when the transposed operand is materialised instead of strided -- enough
    to flip a tied ``topk``. Not a bookkeeping bug: same column set, same order,
    same seed, gathered from the full row. So CPU keeps the full-width rescan and
    pays the old cost.
    """
    if not use_sim:
        return (_greedy_packed_pages_live(k, page_size) if k.is_cuda
                else _greedy_packed_pages_rescan(k, page_size))
    M, N, _ = k.shape
    n_built = -(-N // page_size)
    heads = torch.arange(M, device=k.device)
    norms = (k * k).sum(-1)                                   # [M, N]
    packed = torch.full((M, N), -1, dtype=torch.int32, device=k.device)
    assigned = torch.zeros((M, N), dtype=torch.bool, device=k.device)
    neg = float("-inf")
    # The similarity is what allows the gather; without it every iteration pays
    # a full [M, D, N] pass over the keys (see the docstring).
    sim = torch.bmm(k, k.transpose(1, 2)) if use_sim else None
    kT = None if use_sim else k.transpose(1, 2)

    # Only the pages that exist are built. Iterating the reserved count would run
    # argmax on an all-assigned row and silently re-assign tokens (6b, C5).
    for page in range(n_built):
        n_members = min(page_size, N - page * page_size)
        n_expand = n_members - 1                              # slots 1..n_members-1
        seed = torch.argmax(norms.masked_fill(assigned, neg), dim=1)
        assigned[heads, seed] = True
        packed[heads, seed] = page * page_size
        if n_expand <= 0:
            continue
        if use_sim:
            row = sim[heads, seed]
        else:
            row = torch.bmm(k[heads, seed].unsqueeze(1), kT).squeeze(1)
        # Masked with `assigned`, so every candidate is still unassigned --
        # `N - page * page_size - 1 >= n_expand` unassigned tokens always remain
        # and the topk is full.
        row = row.masked_fill(assigned, neg)
        _, cand = torch.topk(row, n_expand, dim=1)
        slots = torch.arange(1, n_members, dtype=torch.int32,
                             device=k.device).expand_as(cand)
        packed.scatter_(1, cand, page * page_size + slots)
        assigned.scatter_(1, cand, True)

    if not bool(assigned.all()):
        raise PageScanError("greedy packing left tokens unassigned")
    return packed


def _greedy_packed_pages_rescan(k, page_size):
    """The ``use_sim=False`` greedy as it was before live-set compaction.

    Kept for two reasons: it is the reference :func:`_self_test` holds
    :func:`_greedy_packed_pages_live` to on every shape, including the ones with
    forced exact ties, and it is the escape hatch if the live path ever has to be
    turned off. It is byte-for-byte the loop that shipped in ``2ea0cab`` (only
    the ``use_sim`` conditional, which is constant here, is folded out).
    """
    M, N, _ = k.shape
    n_built = -(-N // page_size)
    heads = torch.arange(M, device=k.device)
    norms = (k * k).sum(-1)                                   # [M, N]
    packed = torch.full((M, N), -1, dtype=torch.int32, device=k.device)
    assigned = torch.zeros((M, N), dtype=torch.bool, device=k.device)
    neg = float("-inf")
    kT = k.transpose(1, 2)

    for page in range(n_built):
        n_members = min(page_size, N - page * page_size)
        n_expand = n_members - 1                              # slots 1..n_members-1
        seed = torch.argmax(norms.masked_fill(assigned, neg), dim=1)
        assigned[heads, seed] = True
        packed[heads, seed] = page * page_size
        if n_expand <= 0:
            continue
        row = torch.bmm(k[heads, seed].unsqueeze(1), kT).squeeze(1)
        row = row.masked_fill(assigned, neg)
        _, cand = torch.topk(row, n_expand, dim=1)
        slots = torch.arange(1, n_members, dtype=torch.int32,
                             device=k.device).expand_as(cand)
        packed.scatter_(1, cand, page * page_size + slots)
        assigned.scatter_(1, cand, True)

    if not bool(assigned.all()):
        raise PageScanError("greedy packing left tokens unassigned")
    return packed


def live_compact_span(n_built, span=None):
    """Pages between live-set compactions in the batched greedy.

    Sized against the model in :func:`_greedy_packed_pages_live`: a compaction
    costs two scans of the live set (it reads the live keys and writes them
    dense) and is amortised over ``span`` pages that then scan the *span-start*
    live size instead of all ``N``. With ``n = N / page_size`` pages in the
    build the fraction of the row-scan cost that survives is
    ``(n + span)(span + 2) / (2 n span)``, minimised at ``span = sqrt(2 n)``
    (45% saved at n = 1005, and flat within a point or two either side, so the
    exact constant is not delicate).
    """
    if span is not None:
        return max(1, int(span))
    return max(1, int((2.0 * n_built) ** 0.5))


def _live_token_index(assigned, n_live):
    """Per row, the ascending indices of the ``False`` (unassigned) entries.

    ``[M, n_live]`` int64. Built by prefix-summing the mask and scattering each
    token's own index into its rank slot, which is idempotent under any order --
    unlike ``nonzero``, whose row-major output order is the only thing that makes
    a reshape safe. Assigned tokens are ranked into one scratch slot past the
    end of the returned view, so the last writer there does not matter.
    """
    M, N = assigned.shape
    rank = (~assigned).cumsum(1, dtype=torch.int64) - 1
    rank.masked_fill_(assigned, n_live)
    tokens = torch.arange(N, device=assigned.device).expand(M, N)
    buf = torch.empty((M, n_live + 1), dtype=torch.int64, device=assigned.device)
    buf.scatter_(1, rank, tokens)
    return buf[:, :n_live]


def _greedy_packed_pages_live(k, page_size, span=None):
    """``use_sim=False`` greedy that rescans only the tokens it can still use.

    Output-identical to the shared loop above -- same seed order, same rows, same
    ``topk``, same scatters -- and it changes only *how much of ``k`` the
    per-page row scan reads.

    The scan is the whole cost of the batched build: ``N / page_size`` sequential
    ``[M, 1, D] @ [M, D, N]`` bmms, each reading all ``M * N * D`` keys, and
    nothing in the build is reused between pages, so it is DRAM-bound. Measured
    at M=96, N=16072 (one A100, one build, torch.profiler): 518 ms of the 823 ms
    total, 793 GB at 1.53 TB/s, which is the card. The only lever left is to read
    fewer bytes.

    And it reads twice what it can use. ``row`` is masked with ``assigned``
    before the ``topk``, so at page ``p`` the ``page_size * p`` columns already
    assigned are read, scored, and thrown away; only the live columns are
    candidates. Scanning the live keys instead halves the reads, and it is
    exact: a gemv accumulates ``sum_d A[d, n] * x[d]`` independently per column
    ``n``, so removing columns cannot move a bit of the columns that stay -- but
    only for a kernel whose per-column accumulator does not depend on the
    operand shape. That is true of the CUDA kernel this runs on (measured:
    ``bmm(s, k_live^T)`` reproduces the full row's columns exactly, 0 of 1542912
    entries at L from 16072 down to 16) and false of CPU BLAS, which is why the
    dispatch in `greedy_packed_pages` is CUDA-only.

    Rebuilding the live set is a gather of ``L`` keys (read + write == two
    scans), so it is amortised over a *span* of pages instead of done per page.
    Inside a span the live array is a superset of the unassigned set -- tokens
    assigned during the span are still in it and are masked out of the compact
    row, which is exactly what the shared loop does to them in the full row.
    ``span`` defaults to :func:`live_compact_span`.

    The ``topk`` still sees a *full* ``[M, N]`` row: the compact row is scattered
    back to its tokens' own columns in ``row_buf``, with ``-inf`` everywhere
    else. That keeps the tensor handed to ``torch.topk`` identical to the shared
    loop's, so equal-valued candidates break the same way they always did --
    which a compacted ``topk`` would not guarantee, and which the forced-tie
    cases in the checks below exist to catch.
    """
    M, N, _ = k.shape
    n_built = -(-N // page_size)
    heads = torch.arange(M, device=k.device)
    head_col = heads.unsqueeze(1)                             # [M, 1]
    norms = (k * k).sum(-1)                                   # [M, N]
    packed = torch.full((M, N), -1, dtype=torch.int32, device=k.device)
    assigned = torch.zeros((M, N), dtype=torch.bool, device=k.device)
    neg = float("-inf")
    kT = k.transpose(1, 2)
    row_buf = torch.empty((M, N), dtype=k.dtype, device=k.device)
    # One arange for the whole build: the shared loop rebuilds it every page.
    slots_all = torch.arange(1, page_size, dtype=torch.int32, device=k.device)
    span = live_compact_span(n_built, span)
    live = span < n_built                                      # else never compact
    live_idx = None
    live_k = None

    for page in range(n_built):
        # The live set at the top of a span is every token no earlier page took,
        # which is exactly `N - page_size * page` of them. (Recomputed rather
        # than counted, so a span boundary is not a synchronisation point.)
        if live and page and page % span == 0:
            live_idx = _live_token_index(assigned, N - page * page_size)
            live_k = k[head_col, live_idx]                     # [M, L, D]
        n_members = min(page_size, N - page * page_size)
        n_expand = n_members - 1                              # slots 1..n_members-1
        seed = torch.argmax(norms.masked_fill(assigned, neg), dim=1)
        assigned[heads, seed] = True
        packed[heads, seed] = page * page_size
        if n_expand <= 0:
            continue
        s = k[heads, seed].unsqueeze(1)
        if live_idx is None:
            row = torch.bmm(s, kT).squeeze(1)
            row.masked_fill_(assigned, neg)
        else:
            # Compact row first, then back into the full [M, N] buffer at the
            # tokens' own columns. Everything outside `live_idx` was already
            # assigned when the span opened, and the mask below re-kills those
            # plus whatever this span has taken since -- so the buffer's stale
            # values never survive a page and it needs no fill.
            row_buf.scatter_(1, live_idx,
                             torch.bmm(s, live_k.transpose(1, 2)).squeeze(1))
            row_buf.masked_fill_(assigned, neg)
            row = row_buf
        _, cand = torch.topk(row, n_expand, dim=1)
        # slots_all is arange(1, page_size); the page's slots are its first
        # n_members - 1 entries, which is arange(1, n_members).
        slots = slots_all[:n_expand].expand_as(cand)
        packed.scatter_(1, cand, page * page_size + slots)
        assigned.scatter_(1, cand, True)

    if not bool(assigned.all()):
        raise PageScanError("greedy packing left tokens unassigned")
    return packed


class PageScan:
    """Capacity-constrained greedy pages plus an exact scan over their means.

    ``reps``/``page_sizes``/``_free`` are numpy and authoritative; ``_reps_t`` is
    the device mirror of ``reps``. Its presence is what selects the device query
    scan, and it also backs the decode-time insert scoring, so it is on the
    query path, not merely an insert optimisation.

    Arguments are per (layer, KV head):
        n_kv_heads: number of KV heads sharing one layer structure.
        head_dim / page_size: vector width and page capacity.
        device: torch device used for the build and for insert scoring.
        reserve_pages: pages reserved beyond ``ceil(N / page_size)`` for decode.
            Defaults to ``DEFAULT_RESERVE_PAGES``, or -- when
            ``generation_reserve_tokens`` is given -- to that declaration's
            ``reserve_pages_for`` answer.
        generation_reserve_tokens: decode tokens per head the reservation must
            hold. When set, the reservation becomes
            ``ceil(tokens / page_size) + DEFAULT_RESERVE_MARGIN_PAGES`` pages
            (via :meth:`build`'s ``n_reserved`` default), and the constructor
            raises :class:`PageScanError` if an explicitly passed
            ``reserve_pages`` cannot hold that -- D2's loud, pre-serving failure
            instead of a mid-generation ``PageScanError`` on a later flush.
    """

    def __init__(self, n_kv_heads, head_dim, page_size=DEFAULT_PAGE_SIZE,
                 device=None, reserve_pages=None, generation_reserve_tokens=None):
        if n_kv_heads <= 0 or head_dim <= 0 or page_size <= 0:
            raise PageScanError("n_kv_heads, head_dim and page_size must be positive")
        if reserve_pages is None:
            reserve_pages = (DEFAULT_RESERVE_PAGES if generation_reserve_tokens is None
                             else reserve_pages_for(generation_reserve_tokens, page_size))
        if reserve_pages < 0:
            raise PageScanError("reserve_pages must be non-negative")
        if generation_reserve_tokens is not None:
            required = reserve_pages_for(generation_reserve_tokens, page_size)
            if reserve_pages < required:
                raise PageScanError(
                    f"reserve_pages={reserve_pages} cannot hold the declared "
                    f"generation_reserve_tokens={generation_reserve_tokens} "
                    f"({required} pages including the "
                    f"{DEFAULT_RESERVE_MARGIN_PAGES}-page margin)")
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.reserve_pages = int(reserve_pages)
        self.generation_reserve_tokens = (
            None if generation_reserve_tokens is None else int(generation_reserve_tokens))
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Filled by build().
        self.count = 0
        self.n_pages = 0                          # reserved CPU address space
        self._n_built = np.zeros(self.n_kv_heads, dtype=np.int32)
        self.token2page = None                    # np.int32 [H, N]
        self.offset_in_page = None                # np.int32 [H, N]
        self.reps = None                          # np.float32 [H, n_pages, D]
        self.page_sizes = None                    # np.int32 [H, n_pages]
        self._free = None                         # np.bool_ [H, n_pages]
        self._reps_t = None                       # torch mirror on self.device
        self._bias_t = None                       # [H, n_pages] 0 built / -inf unbuilt
        self._bias_ver = 0                        # bumped on every in-place _bias_t write
        self._query_scores = None                 # reused [H*ratio, n_pages] scan output
        self._qd_cache = None                     # hoisted constants for _query_device
        self.last_insert = None                   # (np.int32 [H,m], np.int32 [H,m])

        self.build_seconds = 0.0
        self.query_seconds = []
        self.insert_seconds = []

    # ------------------------------------------------------------------ build

    def reserve_address_space(self, n_tokens):
        """Fix and return ``n_pages`` for ``n_tokens``, before any greedy runs.

        The reserved space depends only on the token count, never on the greedy,
        so a caller can lay a layer out -- ``infer_state._page_scan_layout``,
        which truncates ``kvc.c2p`` that ``prefill_sdpa`` reads straight after --
        while deferring the greedy that fills the same space. ``build`` and
        ``build_from_packed`` take the result back through ``n_reserved``.
        """
        self.n_pages = -(-int(n_tokens) // self.page_size) + self.reserve_pages
        return self.n_pages

    def build(self, keys, n_reserved=None):
        """Greedy pack ``keys`` into pages of ``page_size`` (spec section 2).

        ``keys`` is a ``[n_kv_heads, N, head_dim]`` array (numpy or torch). The
        loop runs over ``ceil(N / page_size)`` seeds with all heads batched, and
        each head's pages come out in descending-``||k||`` seed order, so the
        per-head page id space is aligned.
        """
        k = torch.as_tensor(keys, dtype=torch.float32, device=self.device)
        if k.ndim != 3 or k.shape[0] != self.n_kv_heads or k.shape[2] != self.head_dim:
            raise PageScanError(
                f"keys must be [{self.n_kv_heads}, N, {self.head_dim}], got {tuple(k.shape)}")
        if not torch.isfinite(k).all():
            raise PageScanError("keys contain NaN or infinity")
        H, N, D = k.shape
        P = self.page_size
        if N == 0:
            raise PageScanError("build requires at least one token")
        n_built = -(-N // P)
        if n_reserved is None:
            n_reserved = self.reserve_address_space(N)
        if n_reserved < n_built:
            raise PageScanError(
                f"n_reserved={n_reserved} cannot hold the {n_built} built pages")

        start = perf_counter()
        with torch.no_grad():
            packed = greedy_packed_pages(k, P, use_sim=True)
            self._adopt_partition(packed, k, n_reserved)
        self.build_seconds = perf_counter() - start
        return self

    def build_from_packed(self, packed, keys, n_reserved=None):
        """``build`` with the partition handed in, for a cross-layer batch.

        ``packed`` is this layer's ``[H, N]`` slice of
        :func:`greedy_packed_pages` run once over several layers' keys; ``keys``
        is the same ``[H, N, D]`` this layer would have passed to ``build``.
        The greedy is bit-identical to building this layer alone, so this exists
        only to avoid paying the greedy 12 times over.
        """
        k = torch.as_tensor(keys, dtype=torch.float32, device=self.device)
        if k.ndim != 3 or k.shape[0] != self.n_kv_heads or k.shape[2] != self.head_dim:
            raise PageScanError(
                f"keys must be [{self.n_kv_heads}, N, {self.head_dim}], got {tuple(k.shape)}")
        packed = torch.as_tensor(packed, dtype=torch.int32, device=self.device)
        H, N, D = k.shape
        if packed.shape != (H, N):
            raise PageScanError(
                f"packed must be [{H}, {N}] for these keys, got {tuple(packed.shape)}")
        if N == 0:
            raise PageScanError("build requires at least one token")
        n_built = -(-N // self.page_size)
        if n_reserved is None:
            n_reserved = self.reserve_address_space(N)
        if n_reserved < n_built:
            raise PageScanError(
                f"n_reserved={n_reserved} cannot hold the {n_built} built pages")
        start = perf_counter()
        with torch.no_grad():
            self._adopt_partition(packed, k, n_reserved)
        self.build_seconds = perf_counter() - start
        return self

    def _adopt_partition(self, packed, k, n_reserved):
        """Validate a greedy partition and make it this scan's state.

        Shared by ``build`` and ``build_from_packed``: those differ only in where
        ``packed`` came from, and everything downstream -- ``token2page``,
        ``offset_in_page``, ``page_sizes``, ``reps``, the device mirror and the
        unbuilt-page bias -- must be derived identically, or a batched layer
        would silently disagree with a singly-built one.
        """
        H, N, D = k.shape
        P = self.page_size
        n_built = -(-N // P)
        page_of = torch.where(packed >= 0, packed // P, packed)
        slot_of = torch.where(packed >= 0, packed % P, packed)
        if not bool((packed >= 0).all()):
            raise PageScanError("greedy packing left tokens unassigned")
        if not bool((page_of[:, :].amax(dim=1) < n_built).all()):
            raise PageScanError("greedy packing assigned a token out of range")

        # Representatives: mean of the members (spec section 2 -- the mean, not
        # the medoid, measured to be the better ranker).
        reps = torch.zeros((H, n_reserved, D), dtype=torch.float32, device=self.device)
        sizes = torch.zeros((H, n_reserved), dtype=torch.int32, device=self.device)
        ones = torch.ones((N,), dtype=torch.int32, device=self.device)
        for h in range(H):
            reps[h].index_add_(0, page_of[h], k[h])
            sizes[h].index_add_(0, page_of[h], ones)
        reps /= sizes.clamp(min=1).unsqueeze(-1).float()

        self.token2page = page_of.cpu().numpy().astype(np.int32)
        self.offset_in_page = slot_of.cpu().numpy().astype(np.int32)
        self.page_sizes = sizes.cpu().numpy().astype(np.int32)
        self.reps = reps.cpu().numpy()
        if self.device.type != "cpu":
            self._reps_t = reps
            # Unbuilt pages have an all-zero representative (index_add_ over an
            # empty set, then divide by clamp(min=1)), and 0 is not a low score.
            # query() masks this in so they can never win a slot (spec section
            # 6b, C1). Entries are cleared as pages are emitted by insert().
            self._bias_t = torch.full((H, n_reserved), float("-inf"),
                                      dtype=torch.float32, device=self.device)
            self._bias_t[:, :n_built] = 0.0

        self.count = N
        self.n_pages = n_reserved
        self._n_built = np.full(H, n_built, dtype=np.int32)
        self._free = (self.page_sizes < P) & (
            np.arange(self.n_pages, dtype=np.int32)[None, :] < self._n_built[:, None])
        self.last_insert = None

    # ------------------------------------------------------------------ query

    def query(self, q, budget):
        """Score every built page by ``q @ reps.T`` and return the page budget.

        ``q`` is ``[n_qo_heads, head_dim]``. For each KV head the ``ratio`` query
        vectors are scored independently over ``reps[:n_built]``, each contributes
        its top ``budget`` pages, and the same ``first_k_unique`` dedup that
        ``infer_state._DCI_query`` applies (lines 935-939) collapses them to
        ``budget`` distinct pages. Returns ``int32 [n_kv_heads, budget]`` with
        every id ``< n_built`` for that head.
        """
        if self.reps is None:
            raise PageScanError("query before build")
        # Validate on the incoming object: converting a device tensor to numpy
        # just to measure it would force the sync this path exists to avoid.
        if q.ndim != 2 or q.shape[1] != self.head_dim:
            raise PageScanError(f"q must be [n_qo_heads, {self.head_dim}], got {tuple(q.shape)}")
        H = self.n_kv_heads
        if q.shape[0] % H:
            raise PageScanError("n_qo_heads must be a multiple of n_kv_heads")
        ratio = q.shape[0] // H
        n_live = int(self._n_built.min())
        if not 0 < budget <= n_live:
            raise PageScanError(f"budget {budget} outside the {n_live} built pages")

        start = perf_counter()
        if self._reps_t is not None:
            out = self._query_device(q, budget, ratio)
            self.query_seconds.append(perf_counter() - start)
            return out
        q = q.detach().cpu().numpy() if isinstance(q, torch.Tensor) else np.asarray(q)
        q = np.ascontiguousarray(q, dtype=np.float32)
        qr = q.reshape(H, ratio, self.head_dim)
        # -inf beyond n_built: unbuilt pages have no representative and no CPU
        # address, so they must never win a slot (spec section 6b, C1).
        # `_n_built` only ever grows, so the buffer's tail stays -inf across
        # calls and is allocated once.
        scores = self._query_scores
        if scores is None or scores.shape[0] != H * ratio or scores.shape[1] != self.n_pages:
            scores = np.full((H * ratio, self.n_pages), -np.inf, dtype=np.float32)
            self._query_scores = scores
        for h in range(H):
            nb = int(self._n_built[h])
            np.matmul(qr[h], self.reps[h, :nb].T, out=scores[h * ratio:(h + 1) * ratio, :nb])
        scores = scores.reshape(H, ratio, self.n_pages)
        if ratio == 1:
            top = np.argsort(-scores[:, 0, :], axis=1)[:, :budget]
            interleaved = top
        else:
            # The top budget of each row, in descending score order. One
            # argpartition for the whole layer instead of one per head; the
            # sort of the budget survivors keeps torch.topk's tie order out of
            # the result, since the caller sees an ordered row.
            part = np.argpartition(scores, self.n_pages - budget, axis=2)[:, :, -budget:]
            part_scores = np.take_along_axis(scores, part, axis=2)
            order = np.argsort(-part_scores, axis=2)
            top = np.take_along_axis(part, order, axis=2)            # [H, ratio, budget]
            interleaved = top.transpose(0, 2, 1).reshape(H, budget * ratio)

        out = np.empty((H, budget), dtype=np.int32)
        for h in range(H):
            row = first_k_unique(interleaved[h], budget)
            if row.size < budget:
                # Only reachable when the ratio q-heads agree so strongly that
                # their union is smaller than the budget; fill from the head's
                # best pages by group max so the caller's shape assert holds.
                seen = set(row.tolist())
                order = np.argsort(-scores[h].max(axis=0))[:budget]
                extra = [int(p) for p in order if int(p) not in seen][:budget - row.size]
                row = np.concatenate([row, np.asarray(extra, dtype=row.dtype)])
            out[h] = row[:budget]
        self.query_seconds.append(perf_counter() - start)
        return out

    def _query_constants(self, H, budget, ratio):
        """Hoisted per-call constants for :meth:`_query_device`, rebuilt on staleness.

        ``ar`` and ``first`` depend only on ``(H, budget, ratio, n_pages)``, all
        of which are fixed once the address space is reserved; ``_query_device``
        rebuilt both on every call, and the query runs 12 anchor layers per
        token, so the allocation shows up. ``first`` is *not* constant across
        calls -- ``scatter_reduce_("amin")`` reduces *into* the buffer, so a
        stale smaller value from an earlier call would survive and the caller
        re-arms it with the sentinel instead. The built/unbuilt mask is not
        cached here: it is a derived view of ``_bias_t``, which :meth:`insert`
        writes in place, so caching it makes correctness depend on every write
        site remembering to bump ``_bias_ver``. ``_scan_ops`` derives it per
        call instead -- one comparison on an ``[H, 1, n_pages]`` broadcast
        against tensors already in register -- and cannot go stale.
        """
        c = self._qd_cache
        W = budget * ratio
        if (c is not None and c["H"] == H and c["W"] == W
                and c["n_pages"] == self.n_pages and c["bias"] is self._bias_t
                and c["ver"] == self._bias_ver):
            return c
        c = {
            "H": H, "W": W, "n_pages": self.n_pages, "bias": self._bias_t,
            "ver": self._bias_ver,
            "ar": torch.arange(W, device=self.device).expand(H, W),
            "first": torch.zeros((H, self.n_pages), dtype=torch.long,
                                 device=self.device),
        }
        self._qd_cache = c
        return c

    def _query_device(self, q, budget, ratio):
        """Device-side scan: ``bmm`` + ``topk`` + an order-preserving dedup.

        Same selection rule as the numpy path, including the candidate order
        that ``DCI`` uses: the top ``budget`` of each of the ``ratio`` query
        rows per KV head, interleaved rank-major then q-head (the
        ``transpose(0, 2, 1)`` below), then ``first_k_unique``.

        All (layer, KV head) structures are the same shape, so the whole layer
        is one batched ``bmm``. Transfers are the host->device copy of ``q`` (the
        caller supplies it on the CPU) and the device->host copy of the
        ``[H, budget]`` result; the latter synchronises the stream. See the module
        docstring for the tie-breaking caveat against the numpy path.

        The op sequence lives in :meth:`_scan_ops`, which is kept free of host
        round trips and of `torch.nonzero` so that it *can* be captured as a
        CUDA graph. It should not be, at these row lengths: a capture of this
        sequence costs ~107 ms on this box (measured with a warm allocator, and
        unchanged by sharing one graph pool), while the replay it replaces is
        0.186 ms/call against 0.772 ms/call eager -- 0.59 ms saved per call, so
        a capture needs ~180 calls on the *same* ``_reps_t``/``n_pages`` to pay
        for itself. Both change every LongBench row, so a capture is thrown away
        long before then. Revisit for generations in the thousands of tokens.
        """
        H = self.n_kv_heads
        if isinstance(q, torch.Tensor):
            qr = q.detach().to(self.device, torch.float32).reshape(H, ratio, self.head_dim)
        else:
            qr = torch.as_tensor(np.ascontiguousarray(q, dtype=np.float32),
                                 device=self.device).reshape(H, ratio, self.head_dim)
        c = self._query_constants(H, budget, ratio)
        with torch.no_grad():
            return self._scan_ops(qr, c, budget, ratio).cpu().numpy()

    def _scan_ops(self, qr, c, budget, ratio):
        """The scan as a straight-line op sequence, with no host round trip.

        ``qr`` is the float32 ``[H, ratio, head_dim]`` query already on the
        device; the result is an int32 ``[H, budget]`` device tensor.
        """
        H = self.n_kv_heads
        W = c["W"]
        scores = torch.bmm(qr, self._reps_t.transpose(1, 2))    # [H, ratio, n_pages]
        # Mask, never add: unbuilt columns have an all-zero representative,
        # so the bmm scores them 0.0, and `scores + -inf` erases that only
        # while the score is finite. A non-finite q makes the product NaN
        # (inf * 0), `NaN + -inf` is still NaN, and topk ranks NaN first --
        # which returned page ids >= n_built, i.e. pages whose K/V was never
        # written. masked_fill overwrites unconditionally, so unbuilt columns
        # are -inf for any input. The numpy path is structurally immune (it
        # never computes the unbuilt columns at all), so the two backends
        # could disagree only when this one was wrong. `_bias_ver` is not
        # consulted: `insert` writes `_bias_t` in place, and deriving the mask
        # from it here is both cheaper than a cached copy that has to be
        # invalidated per write and impossible to leave stale.
        scores = scores.masked_fill(self._bias_t.unsqueeze(1) != 0, float("-inf"))
        top = scores.topk(budget, dim=-1).indices               # [H, ratio, budget]
        flat = top.transpose(1, 2).reshape(H, W)                # [H, budget*ratio]

        # Order-preserving first-unique: mark each page id's first
        # occurrence across the interleaved candidates, then gather them in
        # candidate order. torch.unique would sort and lose the ranking.
        ar = c["ar"]
        first = c["first"]
        # scatter_reduce_("amin") *reduces into* `first`, it does not write
        # it, so the buffer must be re-armed with the sentinel every call --
        # otherwise an earlier call's smaller rank survives at a page id
        # this call did not rank, and `keep` marks a non-first occurrence.
        first.fill_(W)
        first.scatter_reduce_(1, flat, ar, reduce="amin")
        keep = first.gather(1, flat) == ar          # True only at a first occurrence
        pos = keep.cumsum(1) - 1                    # meaningful only where keep
        # Write ONLY the genuine first occurrences, and only those landing
        # inside the budget. A repeat occurrence shares an output slot with
        # the next first occurrence, and scattering two values to the same
        # (row, slot) is undefined on CUDA -- it silently emitted a duplicate
        # page id and dropped a distinct one. `pos[keep]` is strictly
        # increasing per row, so the surviving indices are distinct.
        #
        # The rejects are aimed at a padding column one past the budget and
        # sliced off, rather than filtered with `out[rows[m], pos[m]] = flat[m]`.
        # That boolean form lowered to three `torch.nonzero` calls, and a
        # `nonzero` on CUDA sizes its output with a host synchronisation -- it
        # was ~40% of the scan's kernels, and it is also the one op here that a
        # CUDA graph cannot capture.
        idx = pos.masked_fill(~keep, budget).clamp_(max=budget)
        src = flat.masked_fill(idx >= budget, 0)
        out = torch.zeros((H, budget + 1), dtype=torch.long, device=self.device)
        out.scatter_(1, idx, src)
        return out[:, :budget].to(torch.int32)

    # ----------------------------------------------------------------- insert

    def insert(self, new_keys):
        """Decode-time assignment of ``m`` new keys per head (spec section 4).

        Each new key goes to the built page with a free slot whose representative
        has the highest inner product; when no such page exists a page is emitted
        from the pre-reserved id space (``n_pages`` never changes; only
        ``n_built`` grows). Representatives are updated online, so the 16 keys of
        one flush are placed one at a time. Returns ``(pages, slots)`` as
        ``int32 [H, m]`` numpy arrays, also kept in :attr:`last_insert`.

        The reservation is a hard cap (see the module docstring): a batch the
        spare pages provably cannot hold is refused up front, and *any* failure
        unwinds the batch, so an interrupted ``insert`` -- exhausted reservation,
        ``Ctrl-C``, anything -- leaves the object byte-identical to what it was
        when the call started.
        """
        if self.reps is None:
            raise PageScanError("insert before build")
        k = torch.as_tensor(new_keys, dtype=torch.float32, device=self.device)
        if k.ndim != 3 or k.shape[0] != self.n_kv_heads or k.shape[2] != self.head_dim:
            raise PageScanError(
                f"new_keys must be [{self.n_kv_heads}, m, {self.head_dim}], got {tuple(k.shape)}")
        if not torch.isfinite(k).all():
            raise PageScanError("new_keys contain NaN or infinity")
        H, m, _ = k.shape
        if m == 0:
            return (np.zeros((H, 0), np.int32), np.zeros((H, 0), np.int32))
        # One device->host copy of the flush; every per-key step below then runs
        # on numpy, so the loop costs no GPU synchronisation. Decode calls this
        # from the main thread, where a sync per key would stall the stream.
        kn = k.cpu().numpy()
        P = self.page_size
        heads = np.arange(H)
        pages = np.full((H, m), -1, dtype=np.int32)
        slots = np.full((H, m), -1, dtype=np.int32)
        touched = []

        # Validate the capacity of the whole batch before mutating anything.
        # Every key takes exactly one slot, and a page is emitted exactly when
        # its head has no free slot left, so `m` keys emit exactly
        # `ceil((m - free_slots) / page_size)` pages per head -- a key that finds
        # no free slot opens a page with `page_size` slots and later keys fill
        # it, so the count is exact, not an estimate. If the reserved pages that
        # are still spare cannot cover it, this flush is *guaranteed* to walk off
        # the end of the page-id space (D2); say so here, and point at the knob,
        # rather than half-way through the loop on some later flush.
        built = np.arange(self.n_pages, dtype=np.int32)[None, :] < self._n_built[:, None]
        free_slots = np.where(built, P - self.page_sizes, 0).sum(axis=1)      # [H]
        needed = np.maximum(-(-(m - free_slots) // P), 0)                     # [H]
        spare = self.n_pages - self._n_built
        short = np.flatnonzero(needed > spare)
        if short.size:
            h = int(short[0])
            raise PageScanError(
                f"page reservation exhausted: {m} keys need {int(needed[h])} more "
                f"page(s) on head {h} but only {int(spare[h])} of the "
                f"{self.n_pages} reserved pages are left (n_built="
                f"{int(self._n_built[h])}); reserve more decode pages up front "
                f"(see reserve_pages_for)")

        # D3: the pre-image of everything the flush changes is journaled before
        # it changes, so the loop below is all-or-nothing. `page_sizes`/`_free`
        # are small, so they are snapshotted whole; `reps` is [H, n_pages, D] and
        # only the `m * H` rows a flush can touch are journaled, one key at a
        # time. Nothing outside this method is touched before the loop ends
        # (`last_insert`, the device mirror and `insert_seconds` are set after).
        old_sizes = self.page_sizes.copy()
        old_free = self._free.copy()
        old_n_built = self._n_built.copy()
        old_reps = np.empty((m, H, self.head_dim), dtype=np.float32)

        start = perf_counter()
        with torch.no_grad():
            # One batched inner product for the whole flush: the per-key loop
            # below then only refreshes the columns it actually changes, rather
            # than rescoring every page 16 times.
            scores = self._insert_scores(k)                       # [H, m, n_pages]
            try:
                for j in range(m):
                    # Both guards below are unreachable while the capacity check
                    # above passes; they stay as the backstop if it ever drifts.
                    if not (self._free.any() or (self._n_built < self.n_pages).any()):
                        raise PageScanError(
                            "page reservation exhausted: no free slot and no spare page")
                    masked = np.where(self._free, scores[:, j], -np.inf)
                    best = masked.argmax(axis=1)
                    # No free page for this head -> emit the next reserved page
                    # (argmax of an all -inf row would otherwise return page 0).
                    chosen = np.where(self._free.any(axis=1), best, self._n_built)
                    if (chosen >= self.n_pages).any():
                        raise PageScanError("page reservation exhausted")
                    size = self.page_sizes[heads, chosen]
                    # Journal this key's pre-image before its first mutation.
                    old_reps[j] = self.reps[heads, chosen]
                    pages[:, j] = chosen
                    slots[:, j] = size
                    # Emitting a page makes it selectable from this point on.
                    emitted = chosen >= self._n_built
                    self._n_built[emitted] = chosen[emitted] + 1
                    # Selectability is published after the loop, not here: while
                    # `_reps_t` still holds the all-zero placeholder the page had
                    # before it was built, a concurrent query would score it 0.0
                    # -- a plausible rank, on a page the caller has not yet given
                    # K/V to. See the publish step at the end of the method (F2).
                    # Online mean update of the chosen page.
                    self.reps[heads, chosen] = (
                        old_reps[j] * size[:, None] + kn[:, j]) / (size + 1)[:, None]
                    self.page_sizes[heads, chosen] = size + 1
                    self._free[heads, chosen] = (size + 1) < P
                    touched.append(np.stack([heads, chosen], axis=1))
                    # Later keys of this flush must see the updated rep of the page
                    # this key landed in (its column is the only one that changed).
                    if j + 1 < m:
                        scores[heads, j + 1:, chosen] = np.einsum(
                            "hd,hkd->hk", self.reps[heads, chosen], kn[:, j + 1:])
            except BaseException:
                # Unwind keys 0..j. Key j's pre-image was journaled before its
                # first mutation, and `pages[:, jj] == -1` means key jj was never
                # placed -- it failed on one of the guards above, which run
                # before the journal -- so there is nothing to undo for it.
                self.page_sizes[:] = old_sizes
                self._free[:] = old_free
                self._n_built[:] = old_n_built
                for jj in range(j, -1, -1):
                    placed = pages[:, jj] >= 0
                    if not placed.any():
                        continue
                    rows = heads[placed]
                    self.reps[rows, pages[rows, jj]] = old_reps[jj][placed]
                if self._bias_t is not None:
                    # Pages this flush emitted went from -inf to 0; unbuilt is
                    # -inf, exactly what they held before the call.
                    r, kk = np.nonzero(pages >= old_n_built[:, None])
                    if r.size:
                        self._bias_t[r, pages[r, kk]] = float("-inf")
                        self._bias_ver += 1
                raise

        # Refresh the device mirror for the rows this flush changed.
        if self._reps_t is not None and touched:
            rows = np.unique(np.concatenate(touched), axis=0)
            self._reps_t[rows[:, 0], rows[:, 1]] = torch.as_tensor(
                self.reps[rows[:, 0], rows[:, 1]], device=self.device)
        # Publish selectability last, mirror first: a page emitted by this flush
        # must not be able to win a query slot before `_reps_t` holds its real
        # representative. (Pages re-used from earlier flushes keep the bias they
        # already had, so only the newly emitted ones are touched here.)
        if self._bias_t is not None:
            published = pages >= old_n_built[:, None]
            if published.any():
                r, c = np.nonzero(published)
                self._bias_t[r, pages[r, c]] = 0.0
                self._bias_ver += 1
        self.last_insert = (pages, slots)
        self.insert_seconds.append(perf_counter() - start)
        return pages, slots

    def _insert_scores(self, keys):
        """Inner product of every new key with every page rep: [H, m, n_pages]."""
        if self._reps_t is not None:
            kt = torch.as_tensor(keys, dtype=torch.float32, device=self.device)
            return torch.bmm(kt, self._reps_t.transpose(1, 2)).cpu().numpy()
        return np.einsum("hkd,hpd->hkp", keys.numpy(), self.reps)

    # -------------------------------------------------------------- accessors

    def get_valid_entries(self, page_index):
        """True occupancy of each selected page, ``int32 [H, n_selected]``.

        Mirrors ``DCI.get_valid_entries``: ``infer_state`` transposes it into
        ``page_valid_entries[layer, ns : ns + budget]``, which the flashinfer
        decode kernel reads as ``page_offset < page_valid_entries[page, head]``.
        A page's ``m`` members occupy slots ``0..m-1`` with no holes, so this is
        a prefix count (spec section 6b, C5).

        An id that is negative or beyond the built pages returns **0**, never
        ``-1``: the tensor is int32 but the kernel parameter is
        ``const uint32_t*``, so -1 would become 0xFFFFFFFF and mark all 16 slots
        valid, making the kernel attend uninitialised KV (spec section 6b, C2).
        """
        if self.page_sizes is None:
            raise PageScanError("get_valid_entries before build")
        sel = np.asarray(page_index, dtype=np.int32)
        if sel.ndim != 2 or sel.shape[0] != self.n_kv_heads:
            raise PageScanError(f"page_index must be [{self.n_kv_heads}, n], got {sel.shape}")
        bad = (sel < 0) | (sel >= self.n_pages)
        rows = np.arange(self.n_kv_heads, dtype=np.int32)[:, None]
        out = self.page_sizes[rows, np.where(bad, 0, sel)].astype(np.int32)
        out[bad] = 0
        return out

    @property
    def n_built(self):
        """Selectable pages per head; grows only when ``insert`` emits one."""
        return self._n_built.copy()

    @property
    def num_leaves(self):
        """CPU page-id space per head -- fixed after build (spec section 4)."""
        return np.full(self.n_kv_heads, self.n_pages, dtype=np.int32)

    @property
    def num_points(self):
        """Per-head indexed token count (``num_points[0]`` is the scalar DCI API)."""
        return np.full(self.n_kv_heads, self.count, dtype=np.int32)

    @property
    def token2node(self):
        """``(token2page, offset_in_page)`` -- the DCI ``token2node`` analogue."""
        return self.token2page, self.offset_in_page

    def stats(self):
        return {
            "n_pages_reserved": self.n_pages,
            "n_pages_built": int(self._n_built.max()),
            "n_tokens": self.count,
            "build_ms": self.build_seconds * 1000,
            "queries": len(self.query_seconds),
            "inserts": len(self.insert_seconds),
            "query_p50_ms": (float(np.percentile(self.query_seconds, 50)) * 1000
                             if self.query_seconds else None),
            "insert_p50_ms": (float(np.percentile(self.insert_seconds, 50)) * 1000
                              if self.insert_seconds else None),
        }


def _self_test():
    """Invariants on synthetic data (spec sections 2, 3, 4, 6b, 7)."""
    rng = np.random.default_rng(0)
    H, D, P = 8, 32, 16
    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  ' + detail}")
        if not ok:
            failures.append(name)

    budget, ratio = 12, 4
    for N in (16, 100, 16 * 9, 11712):
        keys = rng.standard_normal((H, N, D)).astype(np.float32)
        keys *= rng.uniform(0.2, 3.0, size=(H, N, 1)).astype(np.float32)
        n_built = -(-N // P)
        ps = PageScan(H, D, P, device="cpu", reserve_pages=4).build(
            keys, n_reserved=n_built + 4)

        # Every token assigned exactly once == the page id histogram is the page
        # sizes, and the ids are all within the built range.
        hist = np.stack([np.bincount(ps.token2page[h], minlength=ps.n_pages)
                         for h in range(H)])
        check(f"N={N} every token assigned exactly once",
              ps.token2page.shape == (H, N) and (ps.token2page >= 0).all()
              and (ps.offset_in_page >= 0).all()
              and np.array_equal(hist, ps.page_sizes)
              and int(hist.sum()) == H * N)
        check(f"N={N} exactly ceil(N/P) built pages",
              ps.n_built.min() == ps.n_built.max() == n_built and ps.n_pages == n_built + 4
              and (ps.page_sizes[:, :n_built] > 0).all()
              and (ps.page_sizes[:, n_built:] == 0).all(),
              f"n_built={ps.n_built}")
        check(f"N={N} page sizes <= page_size",
              int(ps.page_sizes.max()) <= P and int(ps.page_sizes.sum()) == H * N)
        check(f"N={N} slots are a hole-free prefix 0..size-1 per page",
              all(sorted(ps.offset_in_page[h][ps.token2page[h] == p].tolist()) ==
                  list(range(int(ps.page_sizes[h, p])))
                  for h in range(H) for p in range(n_built)))
        # reps are the true per-page means
        ref = np.zeros((H, ps.n_pages, D), np.float32)
        for h in range(H):
            for p in range(n_built):
                ref[h, p] = keys[h][ps.token2page[h] == p].mean(0)
        err = np.abs(ref[:, :n_built] - ps.reps[:, :n_built]).max()
        check(f"N={N} reps are the true means", err < 1e-4, f"max err {err:g}")

        # query shape, no -1, distinct, only built pages, exact tops
        q = rng.standard_normal((H * ratio, D)).astype(np.float32)
        if n_built < budget:
            try:
                ps.query(q, budget)
                raised = False
            except PageScanError:
                raised = True
            check(f"N={N} query refuses a budget larger than the built pages", raised)
            sel = np.stack([np.arange(n_built, dtype=np.int32) for _ in range(H)])
            check(f"N={N} occupancy matches reality",
                  np.array_equal(ps.get_valid_entries(sel),
                                 np.stack([ps.page_sizes[h, sel[h]] for h in range(H)])))
            check(f"N={N} get_valid_entries never returns -1",
                  (ps.get_valid_entries(np.full((H, 2), -1, np.int32)) == 0).all()
                  and (ps.get_valid_entries(np.full((H, 2), ps.n_pages + 5, np.int32)) == 0).all())
            continue
        pages = ps.query(q, budget)
        check(f"N={N} query shape/dtype/no -1",
              pages.shape == (H, budget) and pages.dtype == np.int32
              and (pages >= 0).all())
        check(f"N={N} query only returns built pages",
              all((pages[h] < ps.n_built[h]).all() for h in range(H)))
        check(f"N={N} query rows distinct",
              all(len(set(pages[h].tolist())) == budget for h in range(H)))
        ok = True
        for h in range(H):
            sc = ps.reps[h, :n_built] @ q[h * ratio:(h + 1) * ratio].T
            per_q_head = set(np.argsort(-sc, axis=0)[:budget].ravel().tolist())
            group_max = set(np.argsort(-sc.max(1))[:budget].tolist())
            ok &= set(pages[h].tolist()) <= (per_q_head | group_max)
        check(f"N={N} query pages come from the exact per-q-head tops", ok)

        sel = pages
        ve = ps.get_valid_entries(sel)
        ref_occ = np.stack([ps.page_sizes[h, sel[h]] for h in range(H)])
        check(f"N={N} occupancy matches reality",
              ve.shape == (H, budget) and np.array_equal(ve, ref_occ)
              and (ve > 0).all() and (ve <= P).all())

        # insert: P new keys per head, reserved space fixed, n_built grows
        reserved = ps.n_pages
        old_sizes = ps.page_sizes.copy()
        old_reps = ps.reps.copy()
        old_built = ps.n_built.copy()
        new = rng.standard_normal((H, P, D)).astype(np.float32)
        pages_i, slots_i = ps.insert(new)
        check(f"N={N} insert returns [H,P] int32",
              pages_i.shape == (H, P) and slots_i.shape == (H, P)
              and pages_i.dtype == np.int32 and (slots_i >= 0).all() and (slots_i < P).all())
        check(f"N={N} insert keeps the reserved page space",
              ps.n_pages == reserved and (ps.num_leaves == reserved).all())
        check(f"N={N} insert only emits pages inside the reservation",
              (pages_i < reserved).all()
              and (ps.n_built >= old_built).all()
              and (ps.n_built <= pages_i.max(1) + 1).all())
        grown = np.stack([np.bincount(pages_i[h], minlength=reserved) for h in range(H)])
        check(f"N={N} insert occupancy grows by exactly the placed keys",
              np.array_equal(ps.page_sizes, old_sizes + grown)
              and int(ps.page_sizes.sum()) == H * (N + P))
        ok = True
        for h in range(H):
            for p in np.unique(pages_i[h]):
                p = int(p)
                if old_sizes[h, p] == 0:
                    continue                                   # emitted page
                tot = new[h][pages_i[h] == p].sum(0)
                n_new = int(grown[h, p])
                want = (old_reps[h, p] * old_sizes[h, p] + tot) / (old_sizes[h, p] + n_new)
                ok &= np.allclose(ps.reps[h, p], want, atol=1e-4)
        check(f"N={N} insert reps are the true means", ok)
        check(f"N={N} insert slots continue the page prefix",
              all(sorted(slots_i[h][pages_i[h] == p].tolist()) ==
                  list(range(int(old_sizes[h, p]), int(old_sizes[h, p]) + int(grown[h, p])))
                  for h in range(H) for p in np.unique(pages_i[h])))
        check(f"N={N} occupancy after insert matches reality",
              np.array_equal(ps.get_valid_entries(pages_i),
                             np.stack([ps.page_sizes[h, pages_i[h]] for h in range(H)])))
        # the scan must still be able to fill its budget and stay in range
        pages2 = ps.query(q, budget)
        check(f"N={N} query after insert stays in range",
              (pages2 >= 0).all()
              and all((pages2[h] < ps.n_built[h]).all() for h in range(H)))

    # D2: the declared generation reserve becomes
    # `ceil(tokens / page_size) + DEFAULT_RESERVE_MARGIN_PAGES` pages, and a
    # reservation that cannot hold the declaration is refused when the structure
    # is constructed -- not on the flush that walks off the end of the page-id
    # space, deep into a generation.
    check("reserve_pages_for adds a non-zero margin",
          DEFAULT_RESERVE_MARGIN_PAGES > 0
          and reserve_pages_for(4096, 16) == 256 + DEFAULT_RESERVE_MARGIN_PAGES
          and reserve_pages_for(1, 16) == 1 + DEFAULT_RESERVE_MARGIN_PAGES)
    try:
        PageScan(2, 4, P, device="cpu", reserve_pages=1, generation_reserve_tokens=4096)
        refused = "no exception"
    except PageScanError:
        refused = True
    check("a reservation below the declared budget fails at construction", refused is True)
    ps_decl = PageScan(2, 4, P, device="cpu", generation_reserve_tokens=4096)
    ps_decl.build(rng.standard_normal((2, 4 * P, 4)).astype(np.float32))
    check("the declared budget sizes the reservation, margin included",
          ps_decl.reserve_pages == 256 + DEFAULT_RESERVE_MARGIN_PAGES
          and ps_decl.generation_reserve_tokens == 4096
          and ps_decl.n_pages == 4 + 256 + DEFAULT_RESERVE_MARGIN_PAGES,
          f"reserve={ps_decl.reserve_pages} n_pages={ps_decl.n_pages}")

    # D3: an insert is all-or-nothing. Everything observable about the page
    # space -- which pages are selectable, their representatives and sizes, and
    # the ids handed back to the caller -- must be identical after a failed call.
    H, D = 4, 8
    keys = rng.standard_normal((H, 3 * P, D)).astype(np.float32)

    def scan_state(scan):
        """The observable page space: build state, occupancy, reps, last_insert."""
        return (scan.n_built, scan.page_sizes.copy(), scan._free.copy(),
                scan.reps.copy(), scan.last_insert)

    def same_state(a, b):
        if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                and np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])):
            return False
        if a[4] is None or b[4] is None:
            return (a[4] is None) == (b[4] is None)
        return all(np.array_equal(x, y) for x, y in zip(a[4], b[4]))

    # A flush whose *later* keys cannot fit: 3 full pages per head plus 1 spare
    # page cannot take 3 pages' worth of keys, and the old code placed the first
    # 16 before raising on the 17th.
    ps = PageScan(H, D, P, device="cpu", reserve_pages=1).build(keys)
    before = scan_state(ps)
    try:
        ps.insert(rng.standard_normal((H, 3 * P, D)).astype(np.float32))
        raised = "no exception"
    except PageScanError:
        raised = True
    check("insert refuses a flush the spare pages cannot hold", raised is True)
    check("the refused flush left the page space untouched",
          same_state(before, scan_state(ps)))

    class _FaultAfterWrite(np.ndarray):
        """An array view that raises once its ``n``-th in-place store has run.

        The capacity check refuses an over-large batch before the loop, so the
        rollback is driven by a fault *inside* the per-key loop instead -- the
        shape of failure an interrupt produces, where some keys are already
        placed and the rest are not.
        """

        def __new__(cls, base, exc, n):
            obj = np.asarray(base).view(cls)
            obj._exc, obj._n, obj._count = exc, n, 0
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self._exc = getattr(obj, "_exc", None)
            self._n = getattr(obj, "_n", 0)
            self._count = getattr(obj, "_count", 0)

        def __setitem__(self, key, value):
            self._count += 1
            if self._exc is not None and self._count >= self._n:
                exc, self._exc = self._exc, None       # fire exactly once
                raise exc
            super().__setitem__(key, value)

    # 4 spare pages hold this flush, so the capacity check passes and the loop
    # runs; the fault lands in the 3rd key, after 2 keys are fully placed.
    ps = PageScan(H, D, P, device="cpu", reserve_pages=4).build(keys)
    flush = rng.standard_normal((H, 3 * P, D)).astype(np.float32)
    before = scan_state(ps)
    fault = RuntimeError("injected mid-flush fault")
    reps = ps.reps
    ps.reps = _FaultAfterWrite(reps, fault, 3)
    try:
        ps.insert(flush)
        outcome = "no exception"
    except BaseException as exc:                       # the unwind is the point
        outcome = exc
    finally:
        ps.reps = reps
    check("a fault mid-flush propagates", outcome is fault,
          f"got {outcome!r}")
    check("the interrupted flush left the page space untouched",
          same_state(before, scan_state(ps)))
    # Untouched has to mean usable, not merely equal: the same object still
    # places the next flush exactly as it would have before the fault.
    pages3, slots3 = ps.insert(np.ascontiguousarray(flush[:, :P]))
    check("the unwound page space still places the next flush",
          (pages3 >= 0).all() and (slots3 >= 0).all()
          and np.array_equal(ps.n_built, np.full(H, 4))
          and int(ps.page_sizes.sum()) == H * 3 * P + H * P,
          f"n_built={ps.n_built} total={int(ps.page_sizes.sum())}")

    # The collapsed expansion (C6) must be token-for-token the literal greedy.
    # Random keys hide the case that matters -- a seed whose nearest neighbours
    # are already owned by an earlier page, so the row must be masked before the
    # topk. The "hub" keys force it: 16 tokens are built to be every later
    # seed's nearest neighbours, and page 0 takes all 16 of them.
    #
    # A near-parallel "cone" case is deliberately NOT compared here: near-parallel
    # keys make float32 scores collide (measured minimum top-2 gap of exactly 0),
    # and tied scores are the one place the collapse can differ -- numpy's argmax
    # takes the lowest index, torch.topk's order among equal scores is
    # unspecified. The hub keys are tie-free at every decision point: all inner
    # products are integers below 2**24, so they are exact in float32 and
    # distinct unless mathematically equal.
    ok_hubs = False
    for N, hub in ((200, False), (401, False), (977, True), (2048, True)):
        H, D, P = 2, 6, 16
        if hub:
            keys = np.zeros((H, N, D), np.float32)
            keys[:, :16, 0] = N + 1.0                     # hubs: 16 largest norms
            keys[:, 16:, 0] = 10.0
            keys[:, 16:, 1] = np.arange(1, N - 15, dtype=np.float32)
            ok_hubs = N > 16
        else:
            keys = rng.standard_normal((H, N, D)).astype(np.float32)
            keys[0] *= rng.uniform(0.05, 1.0, size=(N, 1)).astype(np.float32)
        ps = PageScan(H, D, P, device="cpu", reserve_pages=0).build(keys)
        if hub:
            # page 0 owns all 16 hubs, so every later seed's nearest neighbours
            # in the raw score row are already assigned
            ok_hubs &= set(np.nonzero(ps.token2page[0] == 0)[0].tolist()) == set(range(16))
        ok = True
        for h in range(H):
            assigned = np.zeros(N, bool)
            expected = []
            for seed in np.argsort(-np.linalg.norm(keys[h], axis=1), kind="stable"):
                if assigned[seed]:
                    continue
                page = [int(seed)]
                assigned[seed] = True
                while len(page) < P and not assigned.all():
                    scores = keys[h] @ keys[h, seed]
                    scores[assigned] = -np.inf
                    cand = int(np.argmax(scores))
                    page.append(cand)
                    assigned[cand] = True
                expected.append(page)
            got = [sorted(np.nonzero(ps.token2page[h] == p)[0].tolist())
                   for p in range(-(-N // P))]
            ok &= got == [sorted(e) for e in expected]
            ok &= np.array_equal(ps.page_sizes[h][:len(expected)],
                                 [len(e) for e in expected])
        check(f"collapsed expansion == literal greedy, N={N} hub={int(hub)}", ok)
    check("the hub case puts a taken neighbour in front of every later seed",
          ok_hubs)

    # Exhaustive check of the collapsed expansion against the literal greedy on a
    # hand-built four-cluster case. The perturbation keeps every inner product
    # distinct: torch.topk breaks exact ties in an unspecified order, so a test
    # with ties would only be testing topk, not the greedy.
    H, D, P, N = 1, 4, 4, 11
    idx = np.arange(N)
    cluster = np.zeros((N, 4), np.float32)
    cluster[idx, (idx // 3) % 4] = 3.0 - 0.1 * (idx % 3)
    keys = (cluster + 0.017 * (idx + 1)[:, None]).astype(np.float32)[None]
    ps = PageScan(H, D, P, device="cpu", reserve_pages=0).build(keys)
    assigned = np.zeros(N, bool)
    expected = []
    for seed in np.argsort(-np.linalg.norm(keys[0], axis=1), kind="stable"):
        if assigned[seed]:
            continue
        page = [int(seed)]
        assigned[seed] = True
        while len(page) < P and not assigned.all():
            scores = keys[0] @ keys[0, seed]
            scores[assigned] = -np.inf
            cand = int(np.argmax(scores))
            page.append(cand)
            assigned[cand] = True
        expected.append(page)
    got = [sorted(np.nonzero(ps.token2page[0] == p)[0].tolist()) for p in range(len(expected))]
    check("collapsed expansion reproduces the literal greedy token-for-token",
          got == [sorted(e) for e in expected]
          and np.array_equal(ps.page_sizes[0][:len(expected)], [len(e) for e in expected]),
          f"{got} != {[sorted(e) for e in expected]}")

    # Device-side scan (the path the decode loop actually takes): it must be the
    # same selection rule as the numpy fallback, not merely a similar one --
    # `_apply_selected_pages` fills evicted_idx by POSITION, so page order is
    # load-bearing. Also covers the `_bias_t` mask and the `_reps_t` refresh that
    # `insert` performs, neither of which the cpu-only cases above reach.
    if torch.cuda.is_available():
        H, D, P = 8, 64, 16
        for N in (512, 4096):
            keys = (rng.standard_normal((H, N, D)) * 2.0).astype(np.float32)
            ps = PageScan(H, D, P, device="cuda:0", reserve_pages=8).build(keys)
            # keys are scaled well apart, so no top-`budget` cut is a tie and the
            # comparison is not testing either library's tie order.
            check(f"N={N} device scan is enabled",
                  ps._reps_t is not None and ps._bias_t is not None)
            for stage in ("after build", "after insert"):
                if stage == "after insert":
                    ps.insert((rng.standard_normal((H, P, D)) * 2.0).astype(np.float32))
                same, bad = True, None
                for _ in range(4):
                    q = (rng.standard_normal((H * ratio, D)) * 2.0).astype(np.float32)
                    dev_pages = ps.query(torch.as_tensor(q, device="cuda:0"), budget)
                    reps_t = ps._reps_t
                    ps._reps_t = None                  # force the numpy fallback
                    try:
                        cpu_pages = ps.query(q, budget)
                    finally:
                        ps._reps_t = reps_t
                    if not np.array_equal(dev_pages, cpu_pages):
                        same, bad = False, (dev_pages, cpu_pages)
                check(f"N={N} device scan == numpy scan {stage}", same,
                      "" if same else f"{bad[0][0][:4]} != {bad[1][0][:4]}")
                check(f"N={N} device scan stays inside the built pages {stage}",
                      all(p.max() < ps.n_built[h] and len(set(p.tolist())) == budget
                          for h, p in enumerate(dev_pages)))
        # Overlapping q-head tops: the `ratio` query rows of one KV head sharing
        # pages is the NORMAL case on real keys (they attend to similar things),
        # and it is where the dedup has to drop repeats. Random queries almost
        # never collide, which is why the check above cannot see it. Near-parallel
        # rows make the collision heavy: an earlier CUDA scatter to a duplicated
        # (row, slot) silently emitted one page twice and dropped another.
        H, D, P = 4, 64, 16
        keys = (rng.standard_normal((H, 2048, D)) * 2.0).astype(np.float32)
        ps = PageScan(H, D, P, device="cuda:0", reserve_pages=8).build(keys)
        base = rng.standard_normal((H, D)).astype(np.float32)
        worst = 0
        for _ in range(6):
            # one direction per KV head, shared by all its q-heads (ratio=4)
            q = np.repeat(base[:H] + 0.01 * rng.standard_normal((H, D)).astype(np.float32),
                          4, axis=0)
            dev_pages = ps.query(torch.as_tensor(q, device="cuda:0"), budget)
            reps_t = ps._reps_t
            ps._reps_t = None
            try:
                cpu_pages = ps.query(q, budget)
            finally:
                ps._reps_t = reps_t
            worst = max(worst, int((dev_pages != cpu_pages).sum()))
            check("near-parallel q-heads: device scan == numpy scan",
                  np.array_equal(dev_pages, cpu_pages),
                  f"{dev_pages[0][:6]} != {cpu_pages[0][:6]}")
            check("near-parallel q-heads: no duplicated page in a row",
                  all(len(set(p.tolist())) == budget for p in dev_pages),
                  f"{dev_pages[0]}")

        # pages beyond n_built have a zero representative, and 0 is not a low
        # score, so the -inf bias is the only thing keeping them out (spec 6b C1)
        check("device scan never selects an unbuilt page",
              all(ps.query(torch.as_tensor(
                  (rng.standard_normal((H * ratio, D)) * 2.0).astype(np.float32),
                  device="cuda:0"), budget)[h].max() < ps.n_built[h] for h in range(H))
              and ps.n_built[0] < ps.n_pages)

        # ...and the mask must survive a NON-finite q. `scores + -inf` is NaN
        # when the score itself is NaN (or inf * 0 over an unbuilt all-zero
        # representative), and topk ranks NaN above everything, so adding the
        # bias used to hand back page ids >= n_built -- addresses whose K/V was
        # never written (review F1). Masking is unconditional, so it cannot.
        for bad in (np.inf, -np.inf, np.nan):
            q_bad = np.zeros((H * ratio, ps.head_dim), dtype=np.float32)
            q_bad[0, 0] = bad
            dev = ps.query(torch.as_tensor(q_bad, device="cuda:0"), budget)
            check(f"non-finite q ({bad}) still returns only built pages",
                  all(dev[h].max() < ps.n_built[h] for h in range(H)),
                  f"{dev[0]}")
    else:
        print("SKIP  device-side scan (no CUDA available)")

    # The batched greedy (`use_sim=False`) served by `_greedy_packed_pages_live`
    # is required to be BIT-identical to `_greedy_packed_pages_rescan`, the loop
    # it replaces: the partition is what the accuracy of the whole design rides
    # on, and no recall metric on this box can price a partition change (round 2
    # measured a -0.0158 accuracy move from a change `pq` scored at -0.003). So
    # the contract is `torch.equal`, not a score.
    #
    # Every case runs at several spans, including span=1 (a compaction boundary
    # at every page) and spans that do not divide the page count. The tie cases
    # are the point of the exercise: duplicated keys give a row with exactly
    # equal scores, and duplicated tokens give exactly equal norms, so the seed
    # argmax and the topk both have to break a tie. (No page count is large
    # enough here for the default span to engage, which is why the span is
    # forced rather than left to `live_compact_span`.)
    def _greedy_cases(device):
        for (M, N, P) in ((1, 16, 16), (2, 5, 16), (4, 32, 1), (4, 33, 8),
                          (3, 97, 7), (2, 200, 16), (5, 512, 16)):
            k = torch.as_tensor(rng.standard_normal((M, N, 8)) * 2.0,
                                dtype=torch.float32, device=device)
            yield f"M={M} N={N} P={P}", k, P
        # exact norm ties AND exact score ties: whole tokens duplicated, so the
        # seed's norms collide and its row has repeated values at repeated keys
        M, N, P = 3, 128, 16
        k = torch.as_tensor(rng.standard_normal((M, N, 8)) * 2.0,
                            dtype=torch.float32, device=device)
        k[:, 40] = k[:, 3]
        k[:, 41] = k[:, 3]
        k[:, 90] = k[:, 90]                       # untouched
        k[:, 100] = 0.0
        k[:, 101] = 0.0                           # equal zero norms, equal rows
        yield f"M={M} N={N} P={P} duplicate tokens", k, P
        # every key identical within a head: every unassigned score is a tie
        k = torch.ones((2, 64, 8), dtype=torch.float32, device=device)
        yield "M=2 N=64 P=8 all keys equal", k, 8

    # On CPU the live path is not the one that runs (see `greedy_packed_pages`:
    # CPU BLAS does not keep a column's accumulator independent of the operand
    # shape), so what is checked here is the public entry point -- which must
    # still hand back the shipped loop's bytes.
    for label, k, P in _greedy_cases("cpu"):
        want = _greedy_packed_pages_rescan(k, P)
        got = greedy_packed_pages(k, P, use_sim=False)
        check(f"batched greedy bit-identical (cpu), {label}",
              torch.equal(got, want),
              f"{int((got != want).sum())}/{got.numel()} entries differ")
    # On CUDA the live path is the one that runs, so it is held to the reference
    # directly, at several spans, on the tie cases above.
    if torch.cuda.is_available():
        for label, k, P in _greedy_cases("cuda:0"):
            want = _greedy_packed_pages_rescan(k, P)
            for span in (1, 3):
                got = _greedy_packed_pages_live(k, P, span=span)
                check(f"batched greedy bit-identical (cuda), {label}, span={span}",
                      torch.equal(got, want),
                      f"{int((got != want).sum())}/{got.numel()} entries differ")
            check(f"batched greedy bit-identical (cuda dispatch), {label}",
                  torch.equal(greedy_packed_pages(k, P, use_sim=False), want),
                  "dispatch did not reach the live path")

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


def _bench(N=11712, H=8, D=128, P=16, Q=32, budget=12, iters=20):
    import sys
    import time
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((H, N, D)).astype(np.float32)
    ps = PageScan(H, D, P, device="cuda:0", reserve_pages=512)
    ps.build(keys)
    print(f"build: {ps.build_seconds * 1000:.0f} ms/layer for {H} heads, N={N} "
          f"({ps.n_built[0]} pages)", flush=True)
    torch.cuda.synchronize()
    q = rng.standard_normal((Q, D)).astype(np.float32)
    for _ in range(3):
        ps.query(q, budget)
    t = time.perf_counter()
    for _ in range(iters):
        ps.query(q, budget)
    per_layer = (time.perf_counter() - t) / iters
    print(f"query: {per_layer * 1000:.3f} ms/layer "
          f"({per_layer / H * 1000:.4f} ms/head) numpy-CPU, budget={budget}", flush=True)
    new = rng.standard_normal((H, P, D)).astype(np.float32)
    ps.insert(new)
    t = time.perf_counter()
    for _ in range(iters):
        ps.insert(new)
    print(f"insert: {((time.perf_counter() - t) / iters) * 1000:.3f} ms per "
          f"{P}-token flush/layer", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        sys.exit(_bench(int(sys.argv[2]) if len(sys.argv) > 2 else 11712))
    sys.exit(_self_test())
