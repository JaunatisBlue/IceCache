"""Measure IceCache TT2T with the repository's LongBench timing convention."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sequence-length", type=int, default=36_000)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--page-budgets", type=int, default=16)
    parser.add_argument("--n-unlimited-layers", type=int, default=2)
    parser.add_argument("--n-sink-pages", type=int, default=2)
    parser.add_argument("--n-win-pages", type=int, default=2)
    parser.add_argument("--n-reuse-layers", type=int, default=0)
    parser.add_argument("--n-prefetch-layers", type=int, default=0)
    parser.add_argument("--page-topks", type=int, default=0)
    parser.add_argument("--ratio-1", type=float, default=0.01)
    parser.add_argument("--ratio-2", type=float, default=0.2)
    parser.add_argument("--n-max-bytes", type=int, default=40 * (1 << 28))
    parser.add_argument("--n-max-cpu-bytes", type=int, default=80 * (1 << 28))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dataset", default="narrativeqa")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument(
        "--longbench-dir", type=Path, default=Path(__file__).parent / "LongBench"
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path(__file__).parent / "longbench_config/dataset2prompt.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def llama_chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def truncate_middle(input_ids: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if input_ids.shape[-1] < sequence_length:
        raise ValueError(
            f"selected LongBench prompt has only {input_ids.shape[-1]} tokens; "
            f"at least {sequence_length} are required"
        )
    left = sequence_length // 2
    right = sequence_length - left
    return torch.cat((input_ids[:, :left], input_ids[:, -right:]), dim=-1)


def build_longbench_input(tokenizer, args: argparse.Namespace):
    prompt_formats = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    if args.dataset not in prompt_formats:
        raise KeyError(f"no prompt template for LongBench dataset {args.dataset!r}")
    data = load_from_disk(str(args.longbench_dir / args.dataset))
    prompt_format = prompt_formats[args.dataset]

    if args.sample_index is None:
        candidates = range(len(data))
    else:
        if not 0 <= args.sample_index < len(data):
            raise IndexError(
                f"sample index {args.sample_index} is outside [0, {len(data)})"
            )
        candidates = [args.sample_index]

    for sample_index in candidates:
        prompt = prompt_format.format(**data[sample_index])
        prompt = llama_chat_prompt(tokenizer, prompt)
        input_ids = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids
        original_length = input_ids.shape[-1]
        if original_length >= args.sequence_length:
            input_ids = truncate_middle(input_ids, args.sequence_length).to("cuda")
            return (
                {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
                sample_index,
                original_length,
            )

    raise ValueError(
        f"no sample in {args.dataset!r} reaches {args.sequence_length} tokens"
    )


class TimingCriteria(StoppingCriteria):
    """Match the callback-based timer used by longbench_pred.py."""

    def __init__(self) -> None:
        self.timings = [time.time()]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.timings.append(time.time())
        return False


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True,
    ).eval()

    from icecache import adapter

    adapter.enable_icecache(
        model,
        dtype=torch.float16,
        device=device,
        page_size=args.page_size,
        page_budgets=args.page_budgets,
        n_unlimited_layers=args.n_unlimited_layers,
        n_sink_pages=args.n_sink_pages,
        n_win_pages=args.n_win_pages,
        n_reuse_layers=args.n_reuse_layers,
        n_prefetch_layers=args.n_prefetch_layers,
        page_topks=args.page_topks,
        ratio_1=args.ratio_1,
        ratio_2=args.ratio_2,
        n_max_bytes=args.n_max_bytes,
        n_max_cpu_bytes=args.n_max_cpu_bytes,
        debug=args.debug,
    )

    inputs, sample_index, original_length = build_longbench_input(tokenizer, args)
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"dataset={args.dataset} sample_index={sample_index} "
        f"original_length={original_length} "
        f"sequence_length={inputs['input_ids'].shape[1]} "
        f"page_budget={args.page_budgets}x{args.page_size} "
        f"reuse_layers={args.n_reuse_layers} repeats={args.repeats}",
        flush=True,
    )

    measurements = []
    generated_ids = []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            timer = TimingCriteria()
            output = model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
            if len(timer.timings) < 3:
                raise RuntimeError("the timing callback did not observe two generated tokens")
            elapsed = (timer.timings[1] - timer.timings[0]) + (
                timer.timings[2] - timer.timings[1]
            )
            new_ids = output[0, args.sequence_length :].tolist()
            if len(new_ids) != 2:
                raise RuntimeError(f"expected two generated tokens, got {len(new_ids)}")
            measurements.append(elapsed)
            generated_ids.append(new_ids)
            print(f"repeat={repeat + 1} tt2t_seconds={elapsed:.6f}", flush=True)

    result = {
        "method": "IceCache(reuse)" if args.n_reuse_layers else "IceCache",
        "model_path": os.path.abspath(args.model_path),
        "gpu": torch.cuda.get_device_name(device),
        "dataset": args.dataset,
        "sample_index": sample_index,
        "original_token_length": original_length,
        "sequence_length": args.sequence_length,
        "generated_tokens": 2,
        "page_size": args.page_size,
        "page_budgets": args.page_budgets,
        "token_budget": args.page_size * args.page_budgets,
        "n_unlimited_layers": args.n_unlimited_layers,
        "n_sink_pages": args.n_sink_pages,
        "n_win_pages": args.n_win_pages,
        "n_reuse_layers": args.n_reuse_layers,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "timing": "longbench_stopping_criteria_tt2t",
        "tt2t_seconds": measurements,
        "median_tt2t_seconds": statistics.median(measurements),
        "generated_token_ids": generated_ids,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False)
            output_file.write("\n")


if __name__ == "__main__":
    main()
