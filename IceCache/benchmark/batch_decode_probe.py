"""IceCache decode probe over an independently prefilled, variable-size batch.

Run with the icecache environment and ``PYTHONPATH=IceCache/source``.  The probe
requires a real DCI query in every active row and fails if any row lacks one.  It
measures a batch of ``--batch-size`` requests of *different* prompt lengths, runs
``--steps`` decode steps, then demonstrates request exit/reuse (``retire`` one
slot, ``admit`` a freshly prefilled request into it) followed by ``--extra-steps``
more steps.  There is no online scheduler or continuous batching here.
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

from icecache.adapter import enable_icecache, icecache_state
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
    n = len(states)
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
        for i in range(n):
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
    parser.add_argument("--extra-steps", type=int, default=24,
                        help="decode steps run while the retired slot stays free")
    parser.add_argument("--post-admit-steps", type=int, default=8,
                        help="decode steps run after a fresh request is admitted")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="number of independent requests (any B >= 1)")
    parser.add_argument("--page-budget", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--gpu-pages", type=int, default=0,
                        help="GPU page pool size; 0 = size it from the prompt lengths")
    parser.add_argument("--cpu-pages-per-request", type=int, default=4096)
    parser.add_argument("--query-backend", choices=("serial", "native"), default="serial")
    parser.add_argument("--query-threads", type=int, default=16)
    parser.add_argument("--prefill-mode", choices=("sequential", "batched"), default="sequential",
                        help="sequential = per-request model(...) prefill (default, unchanged); "
                             "batched = one padded model(...) forward over all B prompts, trees built per request")
    parser.add_argument("--compare-native-raw", action="store_true")
    parser.add_argument("--cpu-replay-repeats", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
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
    n = args.batch_size

    def prompt_len(i):
        """Distinct prompt length per request, each above the GPU page budget."""
        return args.prompt_tokens + i * args.page_size

    def pages_for(tokens):
        return max(1, -(-tokens // args.page_size))

    initial_lens = [prompt_len(i) for i in range(n)]
    # ``prefill_alloc_n_tokens`` needs one *contiguous* run per layer, and a
    # prefill keeps every prompt page resident, so the pool must hold
    # n_layers * sum(pages) pages at once.  Keep the admitted request no longer
    # than the slot it replaces: retiring a request only frees runs the size of
    # that request, so a longer newcomer would not find a big enough run even
    # with enough total free pages.
    admit_len = (initial_lens[0] - 4 * args.page_size) if n >= 2 else None
    needed_pages = cfg.num_hidden_layers * (
        sum(pages_for(L) for L in initial_lens)
        + (pages_for(admit_len) if admit_len is not None else 0)
    )
    gpu_pages = args.gpu_pages if args.gpu_pages > 0 else needed_pages + 2 * args.page_size
    if args.gpu_pages <= 0:
        print(f"auto gpu_pages={gpu_pages} (needed={needed_pages})", flush=True)
    gpu_pool = KvPool(
        gpu_pages, args.page_size, cfg.num_key_value_heads,
        head_dim, torch.float16, device, (0, 2, 1, 3),
    )

    def make_state():
        return InferState(
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
            n_max_pages=gpu_pages,
            n_max_cpu_pages=args.cpu_pages_per_request,
            gpu_pool=gpu_pool,
        )

    n = args.batch_size
    states = [make_state() for _ in range(n)]
    enable_icecache(model, dtype=torch.float16, device=device,
                    infer_state=states[0])

    tokens = []
    prompts = []
    prompt_lens = []
    prefill_seconds = []
    # Query tasks run on worker threads.  InferenceMode is thread-local, while
    # DCI mutates cache tensors across those threads; no_grad avoids creating
    # inference tensors that reject such updates outside the main thread.
    with torch.no_grad():
        if args.prefill_mode == "batched":
            # One padded model(...) forward over all B prompts.  Each member is
            # prefilled independently (its own KvCache / DCI tree); only the
            # FlashInfer prefill attention is batched.  pad positions are
            # excluded from KV write, attention and tree building.
            batch = BatchInferState(
                states, query_backend=args.query_backend,
                query_threads=args.query_threads, prefilled=False)
            # Prompts are consumed in ``active_indices`` order.  Nothing has been
            # retired yet, so active order == slot order, and ``prompt_lens`` is
            # appended in the same order so it stays slot-indexed.  ``make_prompt``
            # returns [1, L] while ``prefill_batch`` takes 1-D token ids.
            for slot in batch.active_indices:
                length_slot = prompt_len(slot)
                prompts.append(make_prompt(tokenizer, length_slot, 101 + slot, device)[0])
                prompt_lens.append(length_slot)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out, next_tokens = batch.prefill_batch(model, prompts)
            torch.cuda.synchronize()
            prefill_seconds = [time.perf_counter() - start]
            tokens = list(next_tokens)
            for slot in batch.active_indices:
                # A prompt within the page budget legitimately stays non-sparse.
                if states[slot].use_dci and any(db is None for db in states[slot].dci_db):
                    raise AssertionError(f"request {slot} did not build every DCI tree")
                print(f"batched prefill slot={slot} tokens={prompt_lens[slot]} "
                      f"seconds={prefill_seconds[-1]:.3f}", flush=True)
        else:
            # Distinct prompt length per request (still above the GPU budget).
            for i, state in enumerate(states):
                length_i = prompt_len(i)
                ids = make_prompt(tokenizer, length_i, 101 + i, device)
                prompts.append(ids)
                prompt_lens.append(length_i)
                state.forward_mode = ForwardMode.INITIAL_PREFILL
                start = time.perf_counter()
                try:
                    with icecache_state(model, state):
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

        # No two active requests may own the same physical GPU page.
        seen_pages = set()
        for i, state in enumerate(states):
            overlap = allocated_pages(state) & seen_pages
            if overlap:
                raise AssertionError(f"requests own overlapping GPU pages: {sorted(overlap)[:8]}")
            seen_pages |= allocated_pages(state)

        raw_equal_elements, cpu_replay = (
            compare_native_query_with_serial(states, args.query_threads,
                                             repeats=args.cpu_replay_repeats)
            if args.compare_native_raw else (None, None)
        )

        if args.prefill_mode != "batched":
            batch = BatchInferState.from_prefilled(
                states, query_backend=args.query_backend,
                query_threads=args.query_threads)
        batch.validate_ready()
        before_points = [native_points(state) for state in states]
        generated_ids = [[] for _ in range(n)]
        step_ms = []
        retire_step_ms = []
        admit_step_ms = []
        # Retire a NON-trailing slot whenever there are two, so the active set
        # stops being 0..B-1.  That is exactly the case in which a slot index and
        # a batch row index diverge, so this exercises the general path rather
        # than the contiguous prefix the main loop happens to use.
        retire_idx = 0 if n >= 2 else None
        admitted_index = None
        admitted_prompt_len = None
        retired_free_gain = None
        post_retire_steps = args.extra_steps if retire_idx is not None else 0

        def run_steps(count, phase, bucket):
            """Decode ``count`` steps over the *active* slots only.

            ``tokens`` is indexed by slot; the forward consumes active slots in
            ``batch.active_indices`` order, so rows must be mapped back by slot.
            """
            for step in range(count):
                active = batch.active_indices
                ids = torch.stack([tokens[i] for i in active], dim=0)
                start = time.perf_counter()
                out = batch.step(model, ids, return_dict=True)
                torch.cuda.synchronize()
                bucket.append(1000.0 * (time.perf_counter() - start))
                for row, slot in enumerate(active):
                    tokens[slot] = out.logits[row, -1].argmax(dim=-1)[None]
                    generated_ids[slot].append(int(tokens[slot].item()))
                if (step + 1) % 8 == 0:
                    print(f"{phase} step={step+1}/{count} ms={bucket[-1]:.2f} "
                          f"active={active} queries={batch.native_query_counts}", flush=True)

        try:
            torch.cuda.reset_peak_memory_stats(device)
            run_steps(args.steps, "decode", step_ms)

            if retire_idx is not None:
                # ---- Request exit: free the slot's GPU pages, keep decoding ---
                free_before = len(batch._pool._free_ids)
                import threading as _th
                threads_before = _th.active_count()
                retired_state = batch.retire(retire_idx)
                # retire() shuts the request's asyncio loop down; grab that before
                # we drop our reference to the state.
                loop_stopped = getattr(retired_state, "_loop", None) is None
                states[retire_idx] = None
                del retired_state
                import gc as _gc
                _gc.collect()
                retired_free_gain = len(batch._pool._free_ids) - free_before
                print(f"retired slot={retire_idx} active={batch.active_indices} "
                      f"gpu_pages_returned={retired_free_gain} "
                      f"loop_stopped={loop_stopped} "
                      f"threads {threads_before}->{_th.active_count()}", flush=True)
                run_steps(post_retire_steps, "post-retire", retire_step_ms)

                # ---- Request reuse: prefill a fresh request, admit it ---------
                new_state = make_state()
                new_ids = make_prompt(tokenizer, admit_len, 707 + n, device)
                new_state.forward_mode = ForwardMode.INITIAL_PREFILL
                try:
                    with icecache_state(model, new_state):
                        new_out = model(input_ids=new_ids,
                                        position_ids=torch.arange(new_ids.shape[1], device=device)[None],
                                        cache_position=torch.arange(new_ids.shape[1], device=device),
                                        use_cache=False, return_dict=True)
                finally:
                    new_state.forward_mode = None
                torch.cuda.synchronize()
                if not new_state.use_dci or any(db is None for db in new_state.dci_db):
                    raise AssertionError("admitted request did not build every DCI tree")
                batch.admit(retire_idx, new_state)
                states[retire_idx] = new_state
                prompt_lens[retire_idx] = admit_len
                tokens[retire_idx] = new_out.logits[:, -1].argmax(dim=-1)
                generated_ids[retire_idx] = []
                admitted_index = retire_idx
                admitted_prompt_len = admit_len
                print(f"admit slot={retire_idx} admit_tokens={admit_len} "
                      f"active={batch.active_indices}", flush=True)
                run_steps(args.post_admit_steps, "post-admit", admit_step_ms)
        finally:
            batch.close()

    post_admit_steps = args.post_admit_steps if admitted_index is not None else 0
    if any(state is None for state in states):
        raise AssertionError(
            "a retired slot was never re-admitted; the per-request final checks "
            "below need every slot populated")
    after_points = [native_points(state) for state in states]
    if any(batch.query_counts[i] == 0 for i in batch.active_indices):
        raise AssertionError(f"every active request must query DCI: {batch.native_query_counts}")
    for i in range(n):
        if i == admitted_index:
            # The admitted request only decoded after it was admitted.
            expected = admitted_prompt_len + post_admit_steps
        else:
            expected = prompt_lens[i] + args.steps + post_retire_steps + post_admit_steps
        if states[i].seq_len != expected:
            raise AssertionError(f"request {i} has incorrect final sequence length")
    if any(i != admitted_index and not any(
            end is not None and start is not None and end > start
            for start, end in zip(before_points[i], after_points[i]))
           for i in range(n)):
        raise AssertionError("an original request did not incrementally insert into DCI")
    if batch.decode_steps != args.steps + post_retire_steps + post_admit_steps:
        raise AssertionError("batch forward count disagrees with requested steps")
    # Re-check page isolation across the (now reused) active set.
    seen_pages = set()
    for i, state in enumerate(states):
        overlap = allocated_pages(state) & seen_pages
        if overlap:
            raise AssertionError(f"requests own overlapping GPU pages after decode: {sorted(overlap)[:8]}")
        seen_pages |= allocated_pages(state)

    measured = step_ms[min(8, len(step_ms)) :]
    import dciknn._dci as installed_dci
    dci_path = Path(installed_dci.__file__)
    summary = {
        "config": vars(args) | {"output": str(args.output)},
        "prefill_seconds": prefill_seconds,
        "prefill_mode": args.prefill_mode,
        "runtime": {
            "torch": torch.__version__,
            "dci_path": str(dci_path),
            "dci_sha256": hashlib.sha256(dci_path.read_bytes()).hexdigest(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "native_module": getattr(getattr(batch, "_native", None), "__file__", None),
        },
        "native_query_counts": batch.native_query_counts,
        "query_counts_by_layer": batch.query_counts_by_layer,
        "query_backend": batch.query_backend,
        "workspace_bytes": batch.workspace_bytes,
        "retire_gpu_pages_returned": retired_free_gain,
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
            1000.0 * n * len(measured) / sum(measured) if measured else None,
        "retire_step_ms": retire_step_ms,
        "admit_step_ms": admit_step_ms,
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
