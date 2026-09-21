"""Small, paired LongBench comparison of DCI and PAG MIPS.

Run from any directory; all model and dataset reads are local. Each backend
uses the same model, prompts, generation settings, and per-sample seed.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source"))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Cached Hugging Face model ID or local path")
    parser.add_argument("--dataset", default="hotpotqa", help="Local LongBench task name")
    parser.add_argument("--dataset-root", type=Path, default=HERE / "LongBench")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1], help="Fixed dataset row indices")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="Use one literal prompt instead of LongBench rows")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 file containing one prompt")
    parser.add_argument("--output", type=Path, required=True, help="JSONL result path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backends", nargs="+", choices=["dci", "pag_mips"],
                        default=["dci", "pag_mips"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=32760)
    parser.add_argument("--max-new-tokens", type=int, help="Override task generation length")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--page-budget", type=int, default=16)
    parser.add_argument("--page-topk", type=int, default=0)
    parser.add_argument("--n-unlimited-layers", type=int, default=2)
    parser.add_argument("--n-max-bytes", type=int, default=40 * (1 << 28))
    parser.add_argument("--n-max-cpu-bytes", type=int, default=80 * (1 << 28))
    parser.add_argument("--pag-ef-search", type=int, default=100)
    parser.add_argument("--pag-max-search-k", type=int, default=128)
    parser.add_argument("--pag-topm-initial-factor", type=int, default=4)
    parser.add_argument("--pag-generation-reserve", type=int, default=4096)
    parser.add_argument("--pag-ef-construction", type=int, default=200)
    parser.add_argument("--pag-target-degree", type=int, default=16)
    parser.add_argument("--pag-projection-levels", type=int, default=64)
    args = parser.parse_args()
    if (args.threads < 1 or args.max_input_tokens < 1 or
            args.page_size < 1 or args.page_budget < 1 or args.page_topk < 0 or
            args.n_unlimited_layers < 0 or args.n_max_bytes < 1 or
            args.n_max_cpu_bytes < 1 or args.max_new_tokens is not None and args.max_new_tokens < 1):
        parser.error("Thread, length, page, and memory settings must be valid positive values")
    if args.prompt is None and args.prompt_file is None and (not args.indices or min(args.indices) < 0):
        parser.error("--indices must contain nonnegative row numbers")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(args):
    if args.prompt is not None or args.prompt_file is not None:
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
        return [{"id": "prompt", "prompt": prompt, "answers": None, "all_classes": None}]

    from datasets import load_from_disk

    with (HERE / "longbench_config" / "dataset2prompt.json").open(encoding="utf-8") as f:
        formats = json.load(f)
    if args.dataset not in formats:
        raise ValueError(f"Unknown LongBench task: {args.dataset}")
    data = load_from_disk(str(args.dataset_root / args.dataset))
    if max(args.indices) >= len(data):
        raise IndexError(f"Dataset has {len(data)} rows; requested {max(args.indices)}")
    return [{"id": index, "prompt": formats[args.dataset].format(**data[index]),
             "answers": data[index].get("answers"),
             "all_classes": data[index].get("all_classes")}
            for index in args.indices]


def prepare_input(sample, tokenizer, args):
    prompt = sample["prompt"]
    tokens = tokenizer(prompt, add_special_tokens=True).input_ids
    if len(tokens) > args.max_input_tokens:
        left = args.max_input_tokens // 2
        right = args.max_input_tokens - left
        prompt = (tokenizer.decode(tokens[:left], skip_special_tokens=True) +
                  tokenizer.decode(tokens[-right:], skip_special_tokens=True))
    if sample["id"] != "prompt" and args.dataset not in {
            "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    return tokenizer(prompt, return_tensors="pt")


def score_prediction(prediction, sample, dataset):
    if not sample["answers"]:
        return None
    from longbench_eval import dataset2metric

    metric = dataset2metric.get(dataset)
    if metric is None:
        return None
    if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip("\n").split("\n")[0]
    return max(float(metric(prediction, answer, all_classes=sample["all_classes"]))
               for answer in sample["answers"])


class TokenTimer(StoppingCriteria):
    def __init__(self, device):
        self.device = device
        self.times = []

    def __call__(self, input_ids, scores, **kwargs):
        torch.cuda.synchronize(self.device)
        self.times.append(time.perf_counter())
        return False


def statistics(state):
    stats = state.retrieval_stats()
    pag_queries = stats["query"]["pag_mips"]["count"]
    dci_queries = stats["query"]["dci"]["count"]
    if pag_queries and dci_queries:
        actual = "mixed_pag_dci"
    elif pag_queries:
        actual = "pag_mips"
    elif dci_queries:
        actual = "dci"
    else:
        actual = "no_retrieval"
    return actual, stats


def run_backend(args, backend, samples, tokenizer, device, max_new_tokens, output):
    from icecache import adapter
    from icecache.infer_state import InferState

    seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
    config = model.config
    if config.model_type != "qwen3":
        raise ValueError("This comparison currently supports Qwen3 checkpoints only")
    # The current IceCache adapter detects Qwen3 normalization from this name.
    # A local checkpoint directory may not contain "qwen3" in its path.
    if "qwen3" not in config._name_or_path.lower():
        config._name_or_path = "qwen3"
    if args.n_unlimited_layers >= config.num_hidden_layers:
        raise ValueError("n-unlimited-layers must leave at least one retrieval layer")
    state = InferState(
        n_layers=config.num_hidden_layers,
        n_qo_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim or config.hidden_size // config.num_attention_heads,
        page_size=args.page_size, dtype=torch.float16, device=device,
        page_budgets=args.page_budget, page_topks=args.page_topk,
        n_unlimited_layers=args.n_unlimited_layers,
        n_max_bytes=args.n_max_bytes, n_max_cpu_bytes=args.n_max_cpu_bytes,
        n_prefetch_layers=0, n_reuse_layers=0,
        retrieval_backend=backend, pag_ef_search=args.pag_ef_search,
        pag_max_search_k=args.pag_max_search_k,
        pag_topm_initial_factor=args.pag_topm_initial_factor,
        pag_generation_reserve=args.pag_generation_reserve,
        pag_ef_construction=args.pag_ef_construction,
        pag_target_degree=args.pag_target_degree,
        pag_projection_levels=args.pag_projection_levels)
    adapter.enable_icecache(model, dtype=torch.float16, device=device, infer_state=state)

    for sample in samples:
        seed_all(args.seed + (sample["id"] if isinstance(sample["id"], int) else 0))
        inputs = prepare_input(sample, tokenizer, args).to(device)
        prompt_tokens = inputs.input_ids.shape[-1]
        if prompt_tokens < 4 * args.page_size + 100:
            raise ValueError(f"Sample {sample['id']} has only {prompt_tokens} input tokens; "
                             "PAG needs at least 100 offloaded tokens after sink/window pages")
        if prompt_tokens + max_new_tokens > config.max_position_embeddings:
            raise ValueError(f"Sample {sample['id']} exceeds model context length after chat formatting")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timer = TokenTimer(device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([timer]))[0]
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        prediction = tokenizer.decode(generated[prompt_tokens:], skip_special_tokens=True)
        actual, retrieval = statistics(state)
        record = {
            "sample_id": sample["id"], "dataset": None if sample["id"] == "prompt" else args.dataset,
            "model": args.model, "seed": args.seed + (sample["id"] if isinstance(sample["id"], int) else 0),
            "threads": args.threads, "page_size": args.page_size, "page_budget": args.page_budget,
            "page_topk": args.page_topk, "backend": backend, "actual_backend": actual,
            "pag_ef_search": args.pag_ef_search,
            "pag_ef_construction": args.pag_ef_construction,
            "pag_target_degree": args.pag_target_degree,
            "pag_projection_levels": args.pag_projection_levels,
            "retrieval_stats": retrieval, "prediction": prediction, "answers": sample["answers"],
            "score": score_prediction(prediction, sample, args.dataset),
            "prompt_tokens": prompt_tokens, "generated_tokens": len(generated) - prompt_tokens,
            "ttft_s": timer.times[0] - start if timer.times else None,
            "tpot_s": (timer.times[-1] - timer.times[0]) / (len(timer.times) - 1)
            if len(timer.times) > 1 else None,
            "total_s": end - start,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{backend} sample={sample['id']} actual={record['actual_backend']} "
              f"tokens={record['generated_tokens']} fallback={retrieval['fallback_queries']}", flush=True)
        del inputs, generated
    del model, state
    torch.cuda.empty_cache()


def main():
    args = arguments()
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count() or args.gpu < 0:
        raise RuntimeError(f"CUDA device {args.gpu} is unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(f"cuda:{args.gpu}")
    samples = load_samples(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    if args.max_new_tokens is None:
        with (HERE / "longbench_config" / "dataset2maxlen.json").open(encoding="utf-8") as f:
            lengths = json.load(f)
        max_new_tokens = lengths[args.dataset] if samples[0]["id"] != "prompt" else 32
    else:
        max_new_tokens = args.max_new_tokens
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {args.output}")
    print(f"Writing paired results to {args.output}", flush=True)
    for backend in args.backends:
        run_backend(args, backend, samples, tokenizer, device, max_new_tokens, args.output)


if __name__ == "__main__":
    main()
