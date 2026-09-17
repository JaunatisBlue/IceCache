"""B=2 IceCache decode probe with independently prefilled requests.

Run with the icecache environment and ``PYTHONPATH=IceCache/source``.  The
probe requires a real DCI query in both rows and fails if either row lacks one.
It measures a fixed batch, not an online scheduler or prefill throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icecache.adapter import enable_icecache
from icecache.adapter.modeling import set_icecache_infer_state
from icecache.batch import BatchInferState
from icecache.infer_state import ForwardMode, InferState
from icecache.kv_cache import KvPool


def make_prompt(tokenizer, length: int, seed: int, device: torch.device):
    rng = random.Random(seed)
    words = (
        "analysis retrieval memory computer database inference token attention "
        "network system document answer reasoning research science algorithm "
        "history scheduling process cache model sequence page information"
    ).split()
    paragraphs = [" ".join(rng.choices(words, k=96)) for _ in range(length // 64 + 2)]
    ids = tokenizer("\n".join(paragraphs), add_special_tokens=True).input_ids
    if len(ids) < length:
        raise RuntimeError(f"prompt tokenizer only produced {len(ids)} tokens")
    return torch.tensor([ids[:length]], dtype=torch.long, device=device)


def native_points(state):
    return [
        None if db is None else int(db.num_points[0])
        for db in state.dci_db
    ]


def allocated_pages(state):
    result = set()
    for cache in state.kv_caches:
        result.update(int(v) for v in cache.c2p.reshape(-1).tolist() if int(v) >= 0)
    return result


def compare_native_query_with_serial(states, query_threads, repeats=0, seed=919):
    """Compare candidate IDs on frozen real indexes before changing any state."""
    from icecache import _mdci_batch

    rng = np.random.default_rng(seed)
    checked = 0
    replay = {"serial_ms": [], "native_ms": []}
    for layer in (0, states[0].n_layers - 1):
        queries, capsules, neighbours, fields = [], [], [], []
        expected = []
        for request_id, state in enumerate(states):
            db = state.dci_db[layer]
            if db._orig_indices is not None:
                raise AssertionError("native raw comparison requires no original-index map")
            query = rng.standard_normal((state.n_qo_heads, state.head_dim)).astype(np.float32)
            k = state.n_dci_pages - state.layer2topk[layer]
            fov = max(int(state.seq_len * state.search_ratio), 30)
            print(f"raw parity serial layer={layer} request={request_id}", flush=True)
            serial, _ = db.query(query, np.ones(state.n_qo_heads, dtype=np.bool_),
                                 num_neighbours=k, field_of_view=fov,
                                 num_to_visit=int(db.num_points[0]),
                                 num_to_retrieve=-1, prop_to_visit=1.0,
                                 prop_to_retrieve=0.8,
                                 parallel_level=0, ratio=state.ratio)
            expected.append(np.asarray(serial).reshape(-1))
            queries.append(query)
            capsules.append(db._dci_inst)
            neighbours.append(k)
            fields.append(fov)
        print(f"raw parity native layer={layer}", flush=True)
        observed = _mdci_batch.batch_query(capsules, queries, neighbours,
                                            fields, states[0].ratio, query_threads)
        for i in range(2):
            actual = np.asarray(observed[i]).reshape(-1)
            if not np.array_equal(actual, expected[i]):
                different = int(np.count_nonzero(actual != expected[i]))
                raise AssertionError(
                    f"native candidate mismatch at layer {layer}, request {i}: "
                    f"{different}/{actual.size} positions"
                )
            checked += actual.size
        if repeats:
            def serial_call():
                return [state.dci_db[layer].query(
                    query, np.ones(state.n_qo_heads, dtype=np.bool_),
                    num_neighbours=k, field_of_view=fov,
                    num_to_visit=int(state.dci_db[layer].num_points[0]),
                    num_to_retrieve=-1, prop_to_visit=1.0,
                    prop_to_retrieve=0.8, parallel_level=2, ratio=state.ratio,
                ) for state, query, k, fov in zip(states, queries, neighbours, fields)]

            def native_call():
                return _mdci_batch.batch_query(capsules, queries, neighbours,
                                               fields, states[0].ratio, query_threads)

            for _ in range(3):
                serial_call()
                native_call()
            for repeat in range(repeats):
                arms = [("serial_ms", serial_call), ("native_ms", native_call)]
                if repeat % 2:
                    arms.reverse()
                for name, call in arms:
                    start = time.perf_counter()
                    call()
                    replay[name].append(1000.0 * (time.perf_counter() - start))
    return checked, replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--page-budget", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--gpu-pages", type=int, default=4096)
    parser.add_argument("--cpu-pages-per-request", type=int, default=4096)
    parser.add_argument("--query-backend", choices=("serial", "native"), default="serial")
    parser.add_argument("--query-threads", type=int, default=16)
    parser.add_argument("--compare-native-raw", action="store_true")
    parser.add_argument("--cpu-replay-repeats", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_tokens <= args.page_budget * args.page_size:
        raise ValueError("prompt must exceed the resident GPU page budget")
    torch.manual_seed(19)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    cfg = model.config
    head_dim = cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
    gpu_pool = KvPool(
        args.gpu_pages, args.page_size, cfg.num_key_value_heads,
        head_dim, torch.float16, device, (0, 2, 1, 3),
    )
    states = [
        InferState(
            n_layers=cfg.num_hidden_layers,
            n_qo_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            page_size=args.page_size,
            dtype=torch.float16,
            device=device,
            page_budgets=args.page_budget,
            page_topks=0,
            n_sink_pages=2,
            n_win_pages=2,
            n_prefetch_layers=0,
            n_reuse_layers=0,
            n_max_pages=args.gpu_pages,
            n_max_cpu_pages=args.cpu_pages_per_request,
            gpu_pool=gpu_pool,
        )
        for _ in range(2)
    ]
    enable_icecache(model, dtype=torch.float16, device=device,
                    infer_state=states[0])

    tokens = []
    prompts = []
    prefill_seconds = []
    # Query tasks run on worker threads.  InferenceMode is thread-local, while
    # DCI mutates cache tensors across those threads; no_grad avoids creating
    # inference tensors that reject such updates outside the main thread.
    with torch.no_grad():
        for i, state in enumerate(states):
            set_icecache_infer_state(model, state)
            ids = make_prompt(tokenizer, args.prompt_tokens, 101 + i, device)
            prompts.append(ids)
            state.forward_mode = ForwardMode.INITIAL_PREFILL
            start = time.perf_counter()
            try:
                out = model(input_ids=ids,
                            position_ids=torch.arange(ids.shape[1], device=device)[None],
                            cache_position=torch.arange(ids.shape[1], device=device),
                            use_cache=False, return_dict=True)
            finally:
                state.forward_mode = None
            torch.cuda.synchronize()
            prefill_seconds.append(time.perf_counter() - start)
            tokens.append(out.logits[:, -1].argmax(dim=-1))
            if not state.use_dci or any(db is None for db in state.dci_db):
                raise AssertionError(f"request {i} did not build every DCI tree")
            print(f"prefill request={i} tokens={ids.shape[1]} seconds={prefill_seconds[-1]:.3f}", flush=True)

        overlap = allocated_pages(states[0]) & allocated_pages(states[1])
        if overlap:
            raise AssertionError(f"requests own overlapping GPU pages: {sorted(overlap)[:8]}")

        raw_equal_elements, cpu_replay = (
            compare_native_query_with_serial(states, args.query_threads,
                                             repeats=args.cpu_replay_repeats)
            if args.compare_native_raw else (None, None)
        )

        batch = BatchInferState(states, query_backend=args.query_backend,
                                query_threads=args.query_threads)
        set_icecache_infer_state(model, batch)
        before_points = [native_points(state) for state in states]
        step_ms = []
        generated_ids = [[], []]
        try:
            torch.cuda.reset_peak_memory_stats(device)
            for step in range(args.steps):
                ids = torch.stack(tokens, dim=0)
                pos = torch.tensor(batch.seq_lens, device=device, dtype=torch.long)[:, None]
                start = time.perf_counter()
                out = model(input_ids=ids, position_ids=pos,
                            cache_position=pos[0], use_cache=False,
                            return_dict=True)
                torch.cuda.synchronize()
                step_ms.append(1000.0 * (time.perf_counter() - start))
                tokens = [out.logits[i, -1].argmax(dim=-1)[None] for i in range(2)]
                for i in range(2):
                    generated_ids[i].append(int(tokens[i].item()))
                if (step + 1) % 8 == 0:
                    print(f"decode step={step+1}/{args.steps} ms={step_ms[-1]:.2f} queries={batch.native_query_counts}", flush=True)
        finally:
            batch.close()

    after_points = [native_points(state) for state in states]
    if any(count == 0 for count in batch.native_query_counts):
        raise AssertionError(f"both requests must query DCI: {batch.native_query_counts}")
    for i, (before, after) in enumerate(zip(before_points, after_points)):
        if not any(end is not None and start is not None and end > start
                   for start, end in zip(before, after)):
            raise AssertionError(f"request {i} did not incrementally insert into DCI")
        if states[i].seq_len != args.prompt_tokens + args.steps:
            raise AssertionError(f"request {i} has incorrect final sequence length")
    if batch.decode_steps != args.steps:
        raise AssertionError("batch forward count disagrees with requested steps")
    overlap = allocated_pages(states[0]) & allocated_pages(states[1])
    if overlap:
        raise AssertionError(f"requests own overlapping GPU pages after decode: {sorted(overlap)[:8]}")

    measured = step_ms[min(8, len(step_ms)) :]
    import dciknn._dci as installed_dci
    dci_path = Path(installed_dci.__file__)
    summary = {
        "config": vars(args) | {"output": str(args.output)},
        "prefill_seconds": prefill_seconds,
        "runtime": {
            "torch": torch.__version__,
            "dci_path": str(dci_path),
            "dci_sha256": hashlib.sha256(dci_path.read_bytes()).hexdigest(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "native_module": getattr(getattr(batch, "_native", None), "__file__", None),
        },
        "native_query_counts": batch.native_query_counts,
        "native_scheduler": batch._native.last_query_stats()
            if args.query_backend == "native" else None,
        "raw_equal_elements": raw_equal_elements,
        "cpu_query_replay": cpu_replay,
        "native_points_before": before_points,
        "native_points_after": after_points,
        "decode_steps": batch.decode_steps,
        "step_ms": step_ms,
        "generated_token_ids": generated_ids,
        "mean_step_ms_after_warmup": sum(measured) / len(measured) if measured else None,
        "output_tokens_per_second_after_warmup":
            2000.0 * len(measured) / sum(measured) if measured else None,
        "batch_query_seconds": batch.batch_query_seconds,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ("native_points_before", "native_points_after", "step_ms")},
                     default=str, indent=2), flush=True)


if __name__ == "__main__":
    main()
