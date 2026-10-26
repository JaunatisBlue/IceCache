# Agent cache feasibility experiments

This directory contains the first, model-independent experiment for combining
exact prefix caching with IceCache-style semantic page organization.

The first question is deliberately answered before changing DCI:

> In an agent workload, how much KV is exactly reusable, and how much cache is
> occupied by branches that are later abandoned?

## Trace format

Each JSONL row represents one complete model input at the time an inference
call starts:

```json
{"request_id":"root","parent_id":null,"token_ids":[1,2,3],"outcome":"kept"}
{"request_id":"branch-a","parent_id":"root","token_ids":[1,2,3,4,5],"outcome":"abandoned"}
```

`outcome` is one of `kept`, `abandoned`, or `unknown`. The trace must be in
request arrival order. Token IDs are used rather than text so chat-template
rewrites and special tokens are visible in the measurement.

## Run

Use the built-in branching trace to check the analysis pipeline:

```bash
/home/yx/miniconda3/envs/icecache/bin/python trace_prefix_profile.py \
  --synthetic --capacities 4,8,16
```

Analyze a collected trace:

```bash
/home/yx/miniconda3/envs/icecache/bin/python trace_prefix_profile.py \
  trace.jsonl --block-size 16 \
  --capacities 256,512,1024,2048 --json-out result.json
```

The main outputs are:

- `exact_parent_reuse_ratio`: exact token prefix shared with the parent call;
- `unbounded_prefix_hit_ratio`: upper bound for exact prefix caching;
- `lru_hit_ratio`: attainable hit ratio under each page capacity;
- `abandoned_unique_token_ratio`: tokens belonging only to abandoned branches;
- `abandoned_branch_ratio`: fraction of branches marked abandoned.

The next experiment will hold DCI-selected token IDs fixed and compare logical
pages with semantic pages. That isolates page-layout benefit from retrieval
quality.

`rag_kv_reuse_probe.py` tests a separate prerequisite for a unified RAG/KV
index: whether document K/V and page identity survive changes in request
context and token position. See `UNIFIED_RAG_KV_DESIGN.md` for the decision
rule. Run it before implementing direct persistent document-KV injection.

The unified control-plane prototype lives in
`../../source/icecache/experimental/unified_page_index.py`.
`selector_optimality_probe.py` compares its scalable selector with an exact
small-pool oracle. `dci_profile_cost_calibration.py` converts existing aggregate
profiles into cross-configuration stress scenarios, while
`recall_cost_trace_probe.py` collects the per-event samples required for an
actual chance-constraint calibration.

After collecting traces, calibrate and evaluate the resource bound without
reusing test events:

```bash
python chance_constraint_calibration.py trace-*.json \
  --alpha 0.05 --json-out chance-calibration.json
```

`../../source/icecache/experimental/dci_page_backend.py` is the first native
candidate backend for the unified catalog. It preserves IceCache's
one-DCI-instance-per-KV-head organization and
refuses to mix transformer layers. Because `dciknn` is installed in the
IceCache environment, run its synthetic exact-recall gate with:

```bash
PYTHONPATH=../../source /home/yx/miniconda3/envs/icecache/bin/python \
  dci_backend_probe.py --points 4096 --queries 30 --dimension 128 \
  --heads 8 --topk 8 --overfetch 2 --prompt-points 64 \
  --json-out dci-backend.json
```

The probe measures candidate recall and end-to-end Python call latency; it does
not measure RAG quality. The backend also guards the native result-buffer
capacity because the extension can corrupt memory when asked for more results
than its leaf allocation supports.

`materialization_boundary_probe.py` is the durable CPU interop check for the
two-stage plan. It runs two sessions against the same global RAG page, drains
their deferred queues at different absolute context positions, verifies that
they resolve to different native leaves, releases one session, and confirms
the other remains ready. Its encode callback synthesizes `token2node`, so its
reported microseconds are control-plane overhead only and must never be cited
as TTFT, TPOT, or real-prefill measurements:

```bash
PYTHONPATH=../../source /home/yx/miniconda3/envs/icecache/bin/python \
  materialization_boundary_probe.py
```

`../../source/icecache/experimental/runtime_descriptor_trace.py` attaches to
the optional `InferState` observer without changing its decisions and captures the post-RoPE K
tensors, grouped Q tensors, native leaf
membership, and selected leaf IDs needed for offline replay. Its exported
normalized leaf centroids are explicitly proxy page descriptors rather than
IceCache's native token-level score. Install the tracer before generation and
call `trace.export(output_dir)` after generation; output should remain under
the repository's experiment directory.

For non-invasive policy evaluation, combine an
`ExactLeafCentroidProvider` and `ShadowJointSelector` with
`DCIObserverGroup`, pass the group as `dci_observer`, and pass the centroid
provider as `dci_candidate_provider`. The shadow selector jointly considers
real prompt leaves and registered RAG descriptors while enforcing the resource,
prompt, and per-KV-head floors. It records decisions only; selected RAG pages
are not valid attention inputs until a session-local materializer has encoded
and registered their KV.

`semantic_layout_probe.py` implements that comparison on an unmodified
IceCache baseline. Counts are reported in **head-pages**: one 16-token slice for
one KV head. IceCache builds and recalls semantic leaves per KV head, so this is
the relevant byte-transfer unit. It is also a conservative comparison with a
system that can gather logical pages at head-slice granularity.

## vLLM online baseline

`vllm_prefix_baseline.py` replays a controlled five-call branching workload.
Each call generates one token, so the reported latency mostly measures prompt
processing. Run enabled and disabled arms in separate processes:

```bash
/home/yx/miniconda3/envs/vllm_dev/bin/python vllm_prefix_baseline.py \
  --enable-prefix-caching --json-out apc-on.json
/home/yx/miniconda3/envs/vllm_dev/bin/python vllm_prefix_baseline.py \
  --json-out apc-off.json
```

Use the same model, seed, prompt, request order, and GPU settings in both arms.
Repeat in interleaved order (`off/on/on/off`) before claiming a latency gain.
The script warms up the runtime with an unrelated prompt and clears the prefix
cache before recording the five requests.

Capacity-pressure example: keep one useful branch, run four abandoned branches,
then revisit the useful branch. With 16-token blocks, 768 blocks hold the 8k
shared prefix plus roughly two 2k branch suffixes:

```bash
/home/yx/miniconda3/envs/vllm_dev/bin/python vllm_prefix_baseline.py \
  --enable-prefix-caching --workload pressure --prefix-token-sweep 8192 \
  --branch-suffix-tokens 2048 --wrong-branches 4 \
  --num-gpu-blocks-override 768 --max-model-len 12288
```

Environment rule for this project: IceCache and offline trace experiments use
`/home/yx/miniconda3/envs/icecache`; only the vLLM baseline uses
`/home/yx/miniconda3/envs/vllm_dev`.
