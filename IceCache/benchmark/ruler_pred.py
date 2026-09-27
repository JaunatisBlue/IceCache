#!/usr/bin/env python3
"""RULER evaluation at long context (150k-250k tokens) for two arms:

  * ``fullkv``   -- plain HuggingFace full-KV generation
  * ``icecache`` -- the IceCache page_scan path (``adapter.enable_icecache``)

PROMPT CONVENTION -- copied from RULER, not invented here
---------------------------------------------------------
RULER splits the prompt across two fields at data-prep time
(``scripts/data/prepare.py``): ``input`` is the templated task text and
``answer_prefix`` is the tail that opens the answer.  Its inference client
concatenates them at call time, with no separator:

    input_list=[data_point['input'] + data_point.get('answer_prefix', '')
                for data_point in batch]        # scripts/pred/call_api.py:305

so the raw (``--chat_template``-less) prompt here is exactly

    row["input"] + row.get("answer_prefix", "")

which is byte-identical to RULER's own convention.  ``--chat_template``
additionally wraps that string in the tokenizer's own chat template, using the
same idiom as ``gsm8k_pred.py``'s ``build_chat`` (Qwen: try
``enable_thinking=False`` first, fall back without it).

METRICS -- transplanted verbatim
--------------------------------
``string_match_all`` / ``string_match_part`` and the task -> metric mapping come
from RULER's ``scripts/eval/synthetic/constants.py``; ``postprocess_pred`` comes
from RULER's ``scripts/eval/evaluate.py``.  The Apache-2.0 header is kept intact
above them.  Nothing is imported from NeMo (``nemo_toolkit`` is never needed).

CRASH MODE
----------
At these lengths the prefill can take a CUDA illegal-memory-access (an upstream
worker-vs-main-thread race on ``kvc.c2p``, reachable for any row with L > 256).
That is NOT worked around here.  The mitigation is per-row resumability: every
row is appended and flushed before the next one starts, so a crash costs at most
the row in flight and re-running the same command continues where it stopped.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

import jinja2
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Metrics -- transplanted verbatim from RULER
# --------------------------------------------------------------------------- #
# The two metric functions below are copied unchanged from RULER's
# scripts/eval/synthetic/constants.py (github.com/NVIDIA/RULER), together with
# its licence header.  They take (predictions: [str], references: [[str]]).
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


# RULER's synthetic.yaml maps each *task directory* to one of the five
# `TASKS` entries in scripts/eval/synthetic/constants.py:
#   niah_* -> niah (string_match_all), vt -> variable_tracking (string_match_all),
#   cwe -> common_words_extraction (string_match_all),
#   fwe -> freq_words_extraction (string_match_all), qa_* -> qa (string_match_part).
# The values here are that file's `TASKS[...]['metric_fn']`, flattened.
TASK_METRIC_FN = {
    "niah": string_match_all,
    "variable_tracking": string_match_all,
    "common_words_extraction": string_match_all,
    "freq_words_extraction": string_match_all,
    "qa": string_match_part,
}

# task directory name -> RULER task name (mirrors scripts/synthetic.yaml).
TASK_DIR_TO_TASK = {
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
}

# RULER's own generation budget per task, from the `tokens_to_generate` field of
# scripts/data/synthetic/constants.py.  It is baked into the `length` metadata of
# every data row (niah.py: `length = tokens(input_text) + tokens_to_generate`),
# and a smaller --max_new_tokens than this truncates the answer and depresses
# the score.  Advisory only: --max_new_tokens is used as given.
RULER_TOKENS_TO_GENERATE = {
    "niah": 128,
    "variable_tracking": 30,
    "common_words_extraction": 120,
    "freq_words_extraction": 50,
    "qa": 32,
}


def task_name_for_dir(task_dir):
    """Map an on-disk task directory (e.g. ``niah_single_1``) to a RULER task.

    Mirrors RULER's ``scripts/synthetic.yaml``: the yaml key is the directory
    name and its ``task:`` field names the metric family.  Unknown names raise
    instead of silently scoring 0.
    """
    if task_dir in TASK_DIR_TO_TASK:
        return TASK_DIR_TO_TASK[task_dir]
    # niah_single_1 / niah_multikey_2 / qa_1 / qa_2 / ...
    head = task_dir.split("_")[0]
    if head in TASK_METRIC_FN:
        return head
    raise KeyError(
        f"no RULER metric for task directory {task_dir!r}; known families are "
        f"{sorted(TASK_METRIC_FN)} and directory aliases {sorted(TASK_DIR_TO_TASK)}"
    )


def metric_fn_for_task_dir(task_dir):
    return TASK_METRIC_FN[task_name_for_dir(task_dir)]


# postprocess_pred is copied unchanged from RULER's scripts/eval/evaluate.py
# (its unused `task_config` argument is kept so the transplant stays verbatim).
def postprocess_pred(predict_str: str, task_config: dict):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def score_row(pred, outputs, metric_fn):
    """Score one row with RULER's metric, as a 0-100 float.

    RULER scores a whole file at once (``metric_fn(preds, refs)``); calling the
    identical function on a single-element list gives that row's share of the
    aggregate.  The file-level number is recomputed from all rows in
    :func:`write_summary`, so the reported task score is bit-identical to
    RULER's ``evaluate.py``.
    """
    if not outputs:
        return 0.0
    return float(metric_fn([postprocess_pred(pred, {})], [outputs]))


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def build_chat(tokenizer, prompt, model_name):
    """Wrap a raw prompt in the model's own chat template.

    Same idiom as ``gsm8k_pred.py``'s ``build_chat``: prefer the tokenizer's
    jinja template, try ``enable_thinking=False`` first for Qwen (Qwen3
    Instruct-2507 checkpoints have no thinking mode), fall back to a bare
    apply_chat_template, and only then to a hardcoded ``[INST]`` wrapper.
    """
    messages = [{"role": "user", "content": prompt}]
    if "qwen" in model_name.lower():
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError, AttributeError, jinja2.TemplateError):
            pass
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (TypeError, ValueError, AttributeError, jinja2.TemplateError):
        return f"[INST]{prompt}[/INST]"


def build_prompt(tokenizer, row, model_name, chat_template):
    """Assemble the prompt for one RULER row.

    Raw convention (byte-identical to RULER's ``scripts/pred/call_api.py:305``)::

        row["input"] + row.get("answer_prefix", "")
    """
    prompt = row["input"] + row.get("answer_prefix", "")
    if chat_template:
        prompt = build_chat(tokenizer, prompt, model_name)
    return prompt


# --------------------------------------------------------------------------- #
# Model loading -- mirrors longbench_pred.py's load_model_and_tokenizer
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    if args.icecache:
        from icecache import adapter

        model = AutoModelForCausalLM.from_pretrained(
            path, device_map=device, torch_dtype=torch.float16
        ).to(device)
        adapter.enable_icecache(
            model, dtype=torch.float16, device=device, **args.__dict__
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map=device,
        )
    model = model.eval()
    return model, tokenizer


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser(
        description="Evaluate a HF causal LM on RULER at long context "
        "(plain Full-KV and IceCache page_scan arms)."
    )
    ap.add_argument(
        "--model",
        type=str,
        default="qwen3-4b-2507",
        help="Model label; only used to pick the chat-template family and to "
        "resolve --model-path via longbench_config/model2path.json.",
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Checkpoint path. Overrides longbench_config/model2path.json.",
    )
    ap.add_argument("--icecache", action="store_true", help="Enable IceCache.")
    ap.add_argument(
        "--arm",
        type=str,
        default=None,
        help="Arm label used in the output path. Defaults to 'icecache' when "
        "--icecache is given and 'fullkv' otherwise; set it explicitly to tag a "
        "run (e.g. 'icecache_page_scan').",
    )
    # IceCache knobs (same names/defaults as longbench_pred.py, so the same
    # command line works). They are forwarded to InferState via **args.__dict__.
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--page-budgets", type=int, default=16)
    ap.add_argument("--n-unlimited-layers", type=int, default=2)
    ap.add_argument("--n-max-bytes", type=int, default=40 * (1 << 28))
    ap.add_argument("--n-max-cpu-bytes", type=int, default=80 * (1 << 28))
    ap.add_argument("--page-topks", type=int, default=0)
    ap.add_argument("--n-win-pages", type=int, default=2)
    ap.add_argument("--n-sink-pages", type=int, default=2)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ratio_1", type=float, default=0.01)
    ap.add_argument("--ratio_2", type=float, default=0.2)
    ap.add_argument("--n_prefetch_layers", type=int, default=0)
    ap.add_argument("--n_reuse_layers", type=int, default=0)
    ap.add_argument(
        "--retrieval-backend",
        choices=["dci", "pag_mips", "page_scan"],
        default="dci",
    )
    ap.add_argument("--pag-ef-search", type=int, default=100)
    ap.add_argument("--pag-max-search-k", type=int, default=128)
    ap.add_argument("--pag-topm-initial-factor", type=int, default=4)
    ap.add_argument("--pag-generation-reserve", type=int, default=4096)
    ap.add_argument("--pag-ef-construction", type=int, default=200)
    ap.add_argument("--pag-target-degree", type=int, default=16)
    ap.add_argument("--pag-projection-levels", type=int, default=64)
    # RULER data and generation
    ap.add_argument(
        "--data_dir",
        type=str,
        default="/home/yx/ruler-data",
        help="Root holding <length>/<task>/validation.jsonl.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="pred_ruler",
        help="Predictions land in <out_dir>/<arm>/<length>/<task>.jsonl.",
    )
    ap.add_argument(
        "--lengths",
        nargs="+",
        default=None,
        help="Context lengths to run (directory names under --data_dir). "
        "Default: every length directory found under --data_dir.",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task directories to run (e.g. niah_single_1 qa_1). "
        "Default: every task directory found under each length.",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Run only the first N rows of each file; 0 means all.",
    )
    ap.add_argument(
        "--chat_template",
        action="store_true",
        help="Wrap each prompt in the model's own chat template. Off by "
        "default, which reproduces RULER's raw prompt exactly.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Greedy generation budget per row.",
    )
    ap.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Record a failing row in <task>.errors.jsonl and keep going "
        "instead of aborting the run. Off by default: a CUDA illegal-memory-"
        "access poisons the context, so failing loudly + resuming is safer.",
    )
    args = ap.parse_args(cmd_args)
    if args.page_budgets < 0:
        args.page_budgets = None
    if args.max_samples < 0:
        ap.error("--max_samples must be nonnegative")
    if args.arm is None:
        args.arm = "icecache" if args.icecache else "fullkv"
    return args


def resolve_model_path(args):
    """--model-path, else longbench_config/model2path.json next to this file."""
    if args.model_path:
        return args.model_path
    cfg = Path(__file__).resolve().parent / "longbench_config" / "model2path.json"
    if cfg.exists():
        with cfg.open("r") as handle:
            model2path = json.load(handle)
        if args.model in model2path:
            return model2path[args.model]
    raise SystemExit(
        f"no checkpoint path: pass --model-path (or add {args.model!r} to {cfg})"
    )


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def model_max_position_embeddings(model_path):
    """The checkpoint's real context limit, read from its config.json on disk.

    The tokenizer's own `model_max_length` is not usable here: Qwen3 reports
    1010000, while the model's actual limit is `max_position_embeddings`
    (262144 for Qwen3-4B-Instruct-2507).  Returns None if unreadable.
    """
    try:
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as handle:
            return int(json.load(handle).get("max_position_embeddings"))
    except (OSError, ValueError, TypeError):
        return None


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resume_offset(out_path):
    """Number of complete rows already written to ``out_path``.

    A crash can leave a half-written final line.  Such a tail is truncated here
    so the row it belongs to is re-run rather than silently skipped (a plain
    line count, as ``longbench_pred.get_pred`` uses, would skip it), and a
    resumed run never appends to a corrupt file.
    """
    if not os.path.exists(out_path):
        return 0
    with open(out_path, "rb") as handle:
        raw = handle.read()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]  # well-formed file: drop the trailing empty piece
    good = []
    for line in lines:
        if not line.strip():
            continue
        try:
            json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            break  # partial line from a crash: everything after it is redone
        good.append(line)
    if len(good) != len(lines):
        with open(out_path, "wb") as handle:
            handle.write(b"".join(line + b"\n" for line in good))
    return len(good)


def append_jsonl(out_path, record):
    with open(out_path, "a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_summary(summary_path, records, length, task):
    """(Re)write the per-arm summary after each row.

    The task score is RULER's own aggregate over the whole file --
    ``metric_fn(predicts, references)`` -- not a mean of per-row scores, so it
    matches RULER's ``scripts/eval/evaluate.py`` exactly.  It is recomputed from
    the rows on disk, so it is correct after a resume too.
    """
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except json.JSONDecodeError:
            summary = {}
    entry = summary.setdefault(str(length), {})
    preds = [postprocess_pred(r["pred"], {}) for r in records]
    refs = [r["outputs"] for r in records]
    if refs and refs[0] and refs[0][0] is not None:
        score = float(metric_fn_for_task_dir(task)(preds, refs))
    else:
        score = 0.0
    entry[task] = {
        "score": score,
        "n": len(records),
        "nulls": f"{sum([len(x) == 0 for x in preds])}/{len(preds)}",
        "mean_generated_tokens": (
            float(np.mean([r["generated_tokens"] for r in records])) if records else 0.0
        ),
        "mean_token_length": (
            float(np.mean([r["token_length"] for r in records])) if records else 0.0
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Job discovery
# --------------------------------------------------------------------------- #
def discover_jobs(data_dir, lengths, tasks):
    """Expand (length, task, path) triples, skipping what is not on disk."""
    root = Path(data_dir)
    if not root.is_dir():
        raise SystemExit(f"--data_dir {root} is not a directory")
    if lengths:
        length_names = [str(x) for x in lengths]
    else:
        length_names = sorted(
            [p.name for p in root.iterdir() if p.is_dir()],
            key=lambda s: (len(s), s),
        )
    jobs = []
    for length in length_names:
        length_dir = root / length
        if not length_dir.is_dir():
            print(f"[discover] skipping {length}: no such directory under {root}")
            continue
        if tasks:
            task_names = list(tasks)
        else:
            task_names = sorted([p.name for p in length_dir.iterdir() if p.is_dir()])
        for task in task_names:
            path = length_dir / task / "validation.jsonl"
            if not path.exists():
                print(f"[discover] skipping {length}/{task}: no validation.jsonl")
                continue
            jobs.append((length, task, path))
    return jobs


# --------------------------------------------------------------------------- #
# The per-file loop
# --------------------------------------------------------------------------- #
def run_file(
    model,
    tokenizer,
    rows,
    length,
    task,
    device,
    model_name,
    out_path,
    summary_path,
    args,
    max_positions=None,
):
    start = resume_offset(out_path)
    if start >= len(rows):
        print(f"[{task}@{length}] already complete ({start}/{len(rows)}), skipping")
        return
    print(f"[{task}@{length}] resuming at row {start}/{len(rows)} -> {out_path}")

    metric_fn = metric_fn_for_task_dir(task)
    error_path = out_path[: -len(".jsonl")] + ".errors.jsonl" if out_path.endswith(".jsonl") else out_path + ".errors.jsonl"

    ruler_budget = RULER_TOKENS_TO_GENERATE[task_name_for_dir(task)]
    if args.max_new_tokens < ruler_budget:
        print(
            f"[{task}@{length}] WARNING: --max_new_tokens={args.max_new_tokens} is "
            f"below RULER's own budget for this task ({ruler_budget}); answers may "
            f"be truncated, which depresses the score."
        )

    # Rows already on disk are needed for the whole-file (RULER-exact) score.
    records = read_jsonl(out_path) if os.path.exists(out_path) else []

    for idx in range(start, len(rows)):
        row = rows[idx]
        prompt = build_prompt(tokenizer, row, model_name, args.chat_template)
        inputs = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).to(device)
        context_length = int(inputs.input_ids.shape[-1])
        if max_positions and context_length >= max_positions:
            print(
                f"[{task}@{length}] WARNING: prompt is {context_length} tokens, at or "
                f"beyond this model's max_position_embeddings={max_positions}; "
                f"nothing is truncated, so this row may fail or extrapolate."
            )

        # Per-row audit line: prompt and generated token counts (spec 8).
        try:
            with torch.no_grad():
                from transformers import StoppingCriteria, StoppingCriteriaList

                class TimingCriteria(StoppingCriteria):
                    def __init__(self):
                        self.timings = [time.time()]

                    def __call__(self, input_ids, scores, **kwargs):
                        self.timings.append(time.time())
                        return False  # never stop early

                timer = TimingCriteria()
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([timer]),
                )[0]
            output_ids = output
            pred = tokenizer.decode(output_ids[context_length:], skip_special_tokens=True)
            generated_tokens = int(len(output_ids) - context_length)
            # Diagnostic only (never used for scoring or for the written row).
            # Guarded because a row that stops after one token would make
            # longbench_pred.py's unguarded `latencies[0] + latencies[1]` raise.
            token_latencies = [
                t2 - t1 for t1, t2 in zip(timer.timings, timer.timings[1:])
            ]
            if len(token_latencies) >= 3:
                print(
                    f"  prefill {token_latencies[0] + token_latencies[1]:.3f}s / "
                    f"decode {np.mean(token_latencies[2:]):.3f}s per token"
                )
            else:
                print(
                    f"  prefill+decode {sum(token_latencies):.3f}s over "
                    f"{len(token_latencies)} generation step(s)"
                )
        except Exception as error:  # noqa: BLE001 - re-raised unless opted in
            if not args.continue_on_error:
                raise
            print(
                f"[{task}@{length}] row {idx} (index={row.get('index')}) failed: "
                f"{type(error).__name__}: {error}"
            )
            append_jsonl(
                error_path,
                {
                    "index": row.get("index"),
                    "row": idx,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            continue

        score = score_row(pred, row.get("outputs"), metric_fn)
        record = {
            "index": row.get("index"),
            "length": row.get("length"),
            "task": task,
            "arm": args.arm,
            "token_length": context_length,
            "generated_tokens": generated_tokens,
            "pred": pred,
            "outputs": row.get("outputs"),
            "score": score,
            "token_position_answer": row.get("token_position_answer"),
            "chat_template": bool(args.chat_template),
        }
        append_jsonl(out_path, record)
        records.append(record)
        write_summary(summary_path, records, length, task)

        print(
            f"[{task}@{length}] row {idx + 1}/{len(rows)} index={row.get('index')} "
            f"prompt_tokens={context_length} generated_tokens={generated_tokens} "
            f"score={score:.2f}"
        )


def main():
    seed_everything(42)
    args = parse_args()

    model_path = resolve_model_path(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    print(f"model={model_name} path={model_path} device={device}")
    print(f"arm={args.arm} icecache={args.icecache} chat_template={args.chat_template}")

    out_root = Path(args.out_dir) / args.arm
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = str(out_root / "summary.json")
    print(f"output root: {out_root.resolve()}")

    max_positions = model_max_position_embeddings(model_path)
    if max_positions:
        print(f"model max_position_embeddings: {max_positions} (prompts are NOT truncated)")
    else:
        print("model max_position_embeddings: unknown (prompts are NOT truncated)")

    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device, args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    jobs = discover_jobs(args.data_dir, args.lengths, args.tasks)
    if not jobs:
        raise SystemExit(f"no runnable (length, task) pairs under {args.data_dir}")
    print(f"{len(jobs)} (length, task) file(s) to run")

    for length, task, path in jobs:
        try:
            metric_fn_for_task_dir(task)
        except KeyError as error:
            print(f"[discover] {error}")
            continue
        rows = read_jsonl(path)
        if args.max_samples:
            rows = rows[: args.max_samples]
        out_dir = out_root / str(length)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{task}.jsonl")
        run_file(
            model,
            tokenizer,
            rows,
            length,
            task,
            device,
            model_name,
            out_path,
            summary_path,
            args,
            max_positions=max_positions,
        )

    # Final summary: mean over the tasks of each length, plus an overall mean,
    # so a single number per task/arm is available without re-walking jsonl.
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        # Drop the derived keys written by a previous run: they are floats, not
        # length -> task stats maps, and would otherwise be re-aggregated.
        summary.pop("mean_score_per_task", None)
        summary.pop("mean_score", None)
        per_task = {}
        for length, tasks in summary.items():
            if not isinstance(tasks, dict):  # not a length entry; skip
                continue
            for task, stats in tasks.items():
                if isinstance(stats, dict) and "score" in stats:
                    per_task.setdefault(task, []).append(stats["score"])
        summary["mean_score_per_task"] = {
            task: round(float(np.mean(scores)), 2) for task, scores in per_task.items()
        }
        summary["mean_score"] = (
            round(float(np.mean([s for v in per_task.values() for s in v])), 2)
            if per_task
            else 0.0
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary.get("mean_score_per_task", {}), indent=2))
        print(f"overall mean score: {summary['mean_score']}")
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
