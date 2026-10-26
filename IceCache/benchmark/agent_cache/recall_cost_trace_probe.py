#!/usr/bin/env python3
"""Collect per-event IceCache recall cost without modifying the runtime.

The probe wraps an ``InferState`` instance at runtime. ``estimate_select_recall``
already synchronizes the recall CUDA stream, so its wall duration measures the
effective blocking path. Nested wrappers separately expose DCI query and CPU
gather/submission time. Results retain decode step and layer identity rather
than collapsing an entire benchmark into one profile.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import types
from collections import defaultdict
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset", default="IceCache/benchmark/LongBench/qasper"
    )
    parser.add_argument("--sample-index", type=int, default=57)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--page-budget", type=int, default=16)
    parser.add_argument("--page-topk", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--n-unlimited-layers", type=int, default=2)
    parser.add_argument("--n-max-pages", type=int, default=50000)
    parser.add_argument("--n-max-cpu-pages", type=int, default=50000)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


class RecallTrace:
    def __init__(self, state) -> None:
        self.state = state
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.decode_step_by_layer = defaultdict(int)
        self.local = threading.local()

    @staticmethod
    def _timed(original, sink_name):
        def wrapper(trace, *args, **kwargs):
            start = time.perf_counter_ns()
            result = original(*args, **kwargs)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            setattr(trace.local, sink_name, elapsed)
            return result

        return wrapper

    def install(self) -> None:
        state = self.state
        original_query = state._DCI_query
        original_recall = state.recall
        original_estimate = state.estimate_select_recall

        def query_wrapper(_state, *args, **kwargs):
            return RecallTrace._timed(original_query, "query_ms")(self, *args, **kwargs)

        def recall_wrapper(_state, *args, **kwargs):
            return RecallTrace._timed(original_recall, "recall_submit_ms")(self, *args, **kwargs)

        def estimate_wrapper(_state, layer_idx, query_states):
            self.local.query_ms = 0.0
            self.local.recall_submit_ms = 0.0
            start = time.perf_counter_ns()
            eids, nr = original_estimate(layer_idx, query_states)
            total_ms = (time.perf_counter_ns() - start) / 1e6
            if nr is not None:
                with self.lock:
                    step = self.decode_step_by_layer[layer_idx]
                    self.decode_step_by_layer[layer_idx] += 1
                    recalled = int(nr.detach().cpu().sum().item())
                    self.events.append(
                        {
                            "decode_step": step,
                            "layer": int(layer_idx),
                            "seq_len": int(state.seq_len),
                            "recalled_head_pages": recalled,
                            "query_ms": float(self.local.query_ms),
                            "recall_submit_ms": float(self.local.recall_submit_ms),
                            "blocking_total_ms": total_ms,
                            "blocking_minus_query_submit_ms": max(
                                0.0,
                                total_ms
                                - float(self.local.query_ms)
                                - float(self.local.recall_submit_ms),
                            ),
                        }
                    )
            return eids, nr

        state._DCI_query = types.MethodType(query_wrapper, state)
        state.recall = types.MethodType(recall_wrapper, state)
        state.estimate_select_recall = types.MethodType(estimate_wrapper, state)


def summarize(events: list[dict]) -> dict:
    if not events:
        return {"events": 0}
    import numpy as np

    def stats(key):
        values = np.asarray([event[key] for event in events], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    pages = np.asarray([event["recalled_head_pages"] for event in events])
    blocking = np.asarray([event["blocking_total_ms"] for event in events])
    return {
        "events": len(events),
        "recalled_head_pages": stats("recalled_head_pages"),
        "query_ms": stats("query_ms"),
        "recall_submit_ms": stats("recall_submit_ms"),
        "blocking_total_ms": stats("blocking_total_ms"),
        "blocking_minus_query_submit_ms": stats("blocking_minus_query_submit_ms"),
        "pages_vs_blocking_correlation": float(np.corrcoef(pages, blocking)[0, 1]) if len(events) > 1 else None,
    }


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from icecache import adapter
    from icecache.infer_state import InferState

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device
    ).eval()
    config = model.config
    state = InferState(
        n_layers=config.num_hidden_layers,
        n_qo_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
        page_size=args.page_size,
        dtype=torch.float16,
        device=device,
        page_budgets=args.page_budget,
        page_topks=args.page_topk,
        n_unlimited_layers=args.n_unlimited_layers,
        n_max_pages=args.n_max_pages,
        n_max_cpu_pages=args.n_max_cpu_pages,
        n_sink_pages=2,
        n_win_pages=2,
    )
    trace = RecallTrace(state)
    trace.install()
    adapter.enable_icecache(model, dtype=torch.float16, device=device, infer_state=state)

    row = load_from_disk(args.dataset)[args.sample_index]
    prompt = (
        "You are given a scientific article and a question. Answer concisely.\n\n"
        f"Article: {row['context']}\n\nQuestion: {row['input']}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = {
        "config": {
            "model": args.model,
            "sample_index": args.sample_index,
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "page_size": args.page_size,
            "page_budget": args.page_budget,
            "page_topk": args.page_topk,
        },
        "summary": summarize(trace.events),
        "events": trace.events,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
