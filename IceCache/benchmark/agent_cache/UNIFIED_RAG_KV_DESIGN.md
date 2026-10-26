# Unified RAG/KV semantic paging: feasibility gate

## Claim that must be tested first

The attractive version of unified semantic paging stores external documents as
persistent KV pages and retrieves them directly into IceCache.  It removes both
the external embedding index and online document prefill.  That claim is valid
only if an offline document encoding remains usable under a later request's
context and positions.

Transformer KV is generally contextual.  At layer zero, K/V is derived from
token embeddings and is comparatively stable, but at deeper layers it depends
on every earlier layer's causal attention.  K is also rotated by RoPE according
to its online position.  Consequently, inserting an offline page into an
arbitrary request is not equivalent to placing the document text in that
request's prompt.  This is a correctness question, not merely an indexing
optimization.

`rag_kv_reuse_probe.py` is the first feasibility gate.  It holds document token
IDs fixed, changes the preceding context and position, and measures:

1. token/head cosine similarity and relative error for pre-RoPE K and V;
2. page descriptor stability; and
3. same-page top-1 retrieval accuracy across encodings.

Example:

```bash
python rag_kv_reuse_probe.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --document-file document.txt \
  --layers 0,8,16,24,31 \
  --json-out reuse-probe.json
```

## Decision rule

Direct persistent KV injection should proceed only if deep-layer K and V remain
stable and an end-to-end logit/quality equivalence test subsequently passes.
Stable page identity alone is insufficient: it proves that shared retrieval
metadata may work, but not that the recalled values are valid attention input.

If page identity is stable while deep K/V is not, the viable architecture is a
shared semantic page catalog plus session-local KV materialization.  One ANN
query can still jointly rank global document-page descriptors and private
prompt pages, but selected external pages must be re-encoded in the request
context.  Prefix caching and batched/chunked prefill can amortize this cost.

If both K/V and page identities are unstable, the unified component should be
a separate learned page descriptor aligned with attention relevance.  IceCache
can still use a single candidate/selection interface and a joint resource
budget, but claiming one raw-K DCI index would not be justified.

## Prototype boundary

The first runtime prototype should keep two physical namespaces even if the
logical query interface is unified:

- global immutable document descriptors and optional prefill artifacts;
- session-private prompt/DCI pages with ordinary causal visibility.

Candidates need `source`, `document_id`, `segment_id`, `session_id`,
`visibility`, `materialization_state`, and measured load/recompute cost.  The
selector can then compare prompt and RAG candidates under one page budget
without mixing their lifetimes or exposing one session's KV to another.

This separation also makes the necessary baselines precise: independent RAG +
IceCache, unified candidate ranking with session-local materialization, and
direct persistent-KV injection only if the feasibility and quality gates pass.

## Executable control-plane reference

`source/icecache/experimental/unified_page_index.py` now defines the first
executable contract. A global RAG
page is visible to every session, while a prompt page is visible only to its
owner. One cosine search ranks both sources in a shared descriptor space using
relevance, an error-radius penalty, and an optional RAG bias. The selector then
uses aligned load-cost scenarios, so correlated PCIe congestion is retained
when checking the empirical resource quantile. `unified_rag_sim.py` compares
this policy against a deliberately separate fixed budget split; the simulator
is a control-plane check and must not be cited as a model-quality result.

The three catalog/selector tests pass when invoked directly in the current
environment (which does not install pytest). In a 50-seed synthetic sweep with
24 mixed candidates, eight page slots, and a 95% empirical cost bound, unified
selection had higher utility in 90% of seeds. Its mean utility delta over the
fixed 50/50 source split was +0.428 pages-worth of the configured score, it
selected 0.48 more pages on average, and its mean measured violation rate was
3.67%. These numbers validate the implementation behavior only; synthetic
descriptors and costs cannot establish an application or systems improvement.

An attempted real-model gate used Qwen2.5-0.5B with its cache rooted inside the
repository. The model host delivered the 1 GB checkpoint at roughly 18 KB/s,
so the download was stopped rather than treated as an experiment. A meaningful
K/V stability conclusion still requires running the probe with local pretrained
weights; the tiny random-model smoke test proves only that instrumentation and
metrics execute correctly.

## Layer boundary and defensible system claim

IceCache does not have one model-wide DCI space. It creates a DCI instance per
indexed transformer layer, and only explicitly configured layer groups reuse a
topology/result. Keys and queries from different layers are not directly
comparable. Therefore, “one ANN query for RAG and KV” must mean one
mixed-source query at each indexed layer, not one ANN invocation for an entire
decode step. `n_reuse_layers` can amortize searches across nearby layers, but
that approximation needs its own quality ablation.

The global/private lifetime mismatch also argues against one monolithic mutable
tree as the first implementation. Global document pages are immutable and
shared, whereas session pages are frequently inserted and deleted and require
visibility filtering. A defensible implementation can expose one logical
per-layer query while probing a persistent global partition and a session-local
delta in parallel, then merge candidates before the chance-constrained
selector. A truly monolithic DCI tree remains an experimental arm whose scan
pollution, synchronization, deletion, and tenant-isolation costs must beat the
partitioned implementation.

## Selector optimality evidence

An exact subset enumerator now serves as the small-pool oracle. Across 100
synthetic candidate pools, ratio-greedy selection reached the oracle in only
42% of runs, with a 2.79% mean and 17.81% maximum relative utility gap. Adding
improving add/one-for-two exchange moves reached the oracle in 97% of runs;
its mean gap was 0.046% and maximum gap was 2.78%. All runs satisfied the
configured prompt floor and empirical resource constraint. The exact solver is
exponential and is only an evaluation oracle; the refined selector is the
runtime candidate.

## Resource-cost evidence and missing statistical gate

`dci_profile_cost_calibration.py` parsed 69 existing DCI profiles, of which 58
contain usable recall cost fields. Across this heterogeneous experiment set,
median effective recall wait was 0.00864 ms per head-page and P95 was 0.00933
ms; recall wait had 0.826 correlation with TPOT and consumed a median 31.5% of
TPOT. This confirms that recall blocking is material, but not that its
distribution is calibrated: the logs mix synchronous, asynchronous, NUMA,
batching, top-k, and other configurations. A linear fit explains only 27.7% of
wait variance and even yields an invalid negative submission coefficient.

Consequently these aggregate profiles are used only as correlated stress
scenarios. They cannot support a claimed 95% deployment guarantee.
`recall_cost_trace_probe.py` supplies the missing measurement: it wraps an
unmodified `InferState` and records each layer/decode event's recalled
head-pages, DCI query time, recall submission time, and effective blocking time.
The chance constraint must be calibrated and held out on these event-level
samples from one fixed configuration before it is described as probabilistic.

`chance_constraint_calibration.py` implements that gate with split conformal
calibration. It aggregates all indexed layers belonging to one decode token,
randomly separates train/calibration/test token groups, fits a nonnegative
head-page cost slope on train only, and obtains an upper residual correction
from calibration only. Test coverage is reported without retuning. At 95%
target coverage, fewer than 19 calibration tokens cannot produce a finite
split-conformal order statistic; the implementation returns an infinite,
explicitly vacuous bound instead of silently claiming a guarantee. Because
decode tokens can still be temporally dependent, final evidence must include
whole-trace and temporal-shift holdouts, not only a random within-trace split.

## Architecture-level KV reuse result

The pretrained checkpoint could not be downloaded through either Xet or plain
HTTP, but its cached Qwen2.5-0.5B configuration and tokenizer enabled a clearly
labelled random-initialization architecture test. Across three seeds, layer-0
K/V for identical document tokens remained identical across prefixes. Context
dependence appeared immediately: for equal-length unrelated prefixes, mean K
cosine was 0.433 at layer 1, 0.305 at layer 2, 0.125 at layer 4, and roughly
0.08--0.13 in the sampled deeper layers. Deep same-page top-1 identity was only
14.6%--20.8%. Changing both prefix content and document position produced the
same qualitative decay.

Random weights do not predict pretrained-model accuracy, so this is not the
final reuse result. It does establish that context-independent deep KV is not
an architectural invariant and therefore cannot be assumed by system design.
The first implementation should keep global document descriptors immutable and
track descriptor-only, CPU-KV, and GPU-KV materialization per session. Direct
global deep-KV injection remains behind the pretrained quality gate.

The core experimental implementation now follows this rule: `SemanticPage` stores
global identity and descriptor metadata, while each `PageCandidate` carries
session-resolved materialization state and cost samples. Two sessions can see
the same RAG page as a GPU hit and an online recompute respectively, without
mutating global state.

## Per-head M-DCI backend gate

IceCache's native DCI uses one instance per KV head inside each indexed layer,
so the executable backend mirrors that boundary rather than flattening heads
or layers into one vector space. `LayerDCIBackend` accepts only catalogs from a
single transformer layer, requires contiguous head IDs, and maps every native
point ID back through the corresponding head's page table. It uses DCI only to
generate candidates; the shared catalog then recomputes cosine relevance and
applies visibility, error, source-bias, and session-materialization logic.
`search_partitioned` keeps prompt pages out of the shared native tree, searches
the current session's short-lived delta separately, and merges both ranked
candidate streams. This is necessary because foreign-session pages could
otherwise fill DCI's bounded result buffer and disappear only after visibility
filtering, leaving an arbitrarily underfilled result.

A separate partition-contract smoke test used 256 global pages and 32 private
prompt pages per head across 4 heads. Over 25 query batches, overfetch 2 matched
the exact mixed-source top-8 with 1.0 mean, P05, and minimum recall, and with
1.0 exact top-1 rate. This small synthetic run validates merge and isolation
semantics only; the larger global-only sweep below remains the useful scaling
measurement.

The reusable mechanisms now live under `source/icecache/experimental`; the
benchmark directory contains only drivers, analyses, tests, and evidence. The
package root lazily imports the CUDA/DCI adapter, so the control plane can be
imported for CPU-only analysis without weakening the existing
`from icecache import adapter` runtime API.

`InferState` now exposes two default-off runtime contracts in
`source/icecache/retrieval.py`. A `DCIObserver` receives build, incremental-add,
and query events for trace collection. A `CandidateProvider` may replace only
candidate generation while retaining IceCache's existing page diff, recall,
and scatter path. Provider output is rejected before native bookkeeping unless
it has exactly the required per-head shape, integer in-range leaf IDs, and no
head-local duplicates. Custom providers are currently incompatible with layer
prefetch because that path also consumes native token IDs; failing explicitly
prevents a superficially working but semantically incomplete configuration.

The first runtime wiring check used a random-initialized, four-layer Qwen2
configuration chosen to match the compiled CUDA kernel's supported GQA group
and 128-dimensional heads. With a 192-token prefill, all four layers built
native DCI topology, the exact centroid provider observed 9--10 leaves per KV
head, and a provider-backed query returned two recalled leaves per head through
the unchanged IceCache diff path. This proves the extension wiring, not model
quality. Shorter trials also exposed two baseline constraints: the checked
kernel rejects Qwen2.5-0.5B's group size 7 and head dimension 64, and a short
overflow can yield `n_dci_pages - page_topk <= 0`. The latter now raises a
descriptive IceCache error before entering the native library.

The runtime now also supports shadow-mode joint selection. A deterministic
observer group first updates real prompt-leaf centroids, then evaluates those
leaves together with registered immutable RAG descriptors under the empirical
cost constraint. RAG hits remain materialization requests and never masquerade
as native leaf IDs. The first real-query run revealed that a purely global page
limit could allocate every page to one KV head, which is incompatible with
IceCache's dense per-head page tensor. The selector and exact oracle therefore
now share a `min_pages_per_head` constraint. After this correction, the same
192-token runtime check selected three prompt leaves spanning both KV heads,
with a cost quantile of 6.0 under a budget of 6.0. A deterministic unit case
also verifies mixed RAG/prompt selection while preserving every head's floor.

The native extension has an important unchecked ABI precondition. Its result
buffer is bounded by `prop_to_retrieve * num_leaves`, and requesting more
neighbors can corrupt the process instead of raising an exception. The backend
therefore caps native `k` at both `floor(prop_to_retrieve * min_leaves)` and
`min_leaves - 1`. A synthetic probe also found the single-instance path
unstable under repeated queries, so this prototype deliberately requires at
least two heads, matching IceCache's multi-head use rather than advertising a
more general wrapper than was tested.

On 8 heads with 4,096 random unit-normalized descriptors per head, 128
dimensions, 30 query batches, and exact top-8 as the oracle, the measured
candidate trade-off was:

| Overfetch | Mean recall@8 | P05 | Exact top-1 | Mean query ms | P95 query ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.859 | 0.750 | 0.929 | 7.30 | 11.67 |
| 2 | 0.900 | 0.750 | 0.929 | 10.20 | 14.54 |
| 4 | 0.920 | 0.750 | 0.942 | 15.48 | 20.27 |
| 8 | 1.000 | 1.000 | 1.000 | 24.82 | 29.74 |

These are process-level timings for native candidate generation plus Python
reranking on synthetic descriptors; they are not kernel-only measurements and
do not establish RAG relevance quality. They do establish that overfetch must
be an explicit accuracy/latency parameter and that a fixed value of two is not
enough for lossless retrieval at this scale. The next runtime step is to feed
real per-layer, per-head IceCache descriptors through this backend and compare
generation quality and TPOT against exact catalog search and unmodified DCI.
