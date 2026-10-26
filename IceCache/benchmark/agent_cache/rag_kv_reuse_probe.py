#!/usr/bin/env python3
"""Test whether document KV can be reused across unrelated request contexts.

The proposed unified RAG/KV index assumes that a document encoded offline has
keys and values that remain useful when the document is recalled by a later
request.  This probe measures that assumption before any runtime integration.
It captures pre-RoPE K and V for the exact same document token IDs under three
contexts:

* context A and B have equal length but different token content;
* context Shift has a different prefix length.

In addition to token-level cosine similarity, it reports whether a page
descriptor from one encoding retrieves the page with the same logical identity
from the other encoding.  High page identity with low token K/V similarity
supports sharing ANN metadata, but not directly injecting persistent KV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local HF model path or model ID")
    parser.add_argument("--document-file", type=Path)
    parser.add_argument(
        "--document",
        default=(
            "IceCache stores old key-value pages in CPU memory and retrieves "
            "semantically relevant pages during autoregressive decoding."
        ),
    )
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--shift-prefix-tokens", type=int, default=192)
    parser.add_argument("--document-tokens", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Instantiate the architecture from config without pretrained weights",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _repeat_to_length(ids: torch.Tensor, length: int) -> torch.Tensor:
    if ids.numel() == 0:
        raise ValueError("tokenizer produced an empty token sequence")
    repeats = (length + ids.numel() - 1) // ids.numel()
    return ids.repeat(repeats)[:length]


def _cosine_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a32, b32 = a.float(), b.float()
    cosine = F.cosine_similarity(a32, b32, dim=-1)
    relative_l2 = (a32 - b32).norm(dim=-1) / a32.norm(dim=-1).clamp_min(1e-12)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(torch.quantile(cosine.flatten(), 0.05)),
        "relative_l2_mean": float(relative_l2.mean()),
    }


def _page_descriptors(keys: torch.Tensor, page_size: int) -> torch.Tensor:
    """Return normalized [head, page, dim] descriptors from [token, head, dim]."""
    n_full_pages = keys.shape[0] // page_size
    if n_full_pages == 0:
        raise ValueError("document must contain at least one complete page")
    keys = keys[: n_full_pages * page_size]
    pages = keys.reshape(n_full_pages, page_size, keys.shape[1], keys.shape[2])
    descriptors = pages.float().mean(dim=1).permute(1, 0, 2)
    return F.normalize(descriptors, dim=-1)


def _page_identity(a: torch.Tensor, b: torch.Tensor, page_size: int) -> dict[str, float]:
    da, db = _page_descriptors(a, page_size), _page_descriptors(b, page_size)
    similarities = torch.einsum("hpd,hqd->hpq", da, db)
    predicted = similarities.argmax(dim=-1)
    expected = torch.arange(da.shape[1], device=predicted.device).expand_as(predicted)
    reciprocal_ranks = 1.0 / (
        similarities.argsort(dim=-1, descending=True)
        .eq(expected.unsqueeze(-1))
        .float()
        .argmax(dim=-1)
        .add(1)
    )
    return {
        "page_top1_identity": float((predicted == expected).float().mean()),
        "page_mrr": float(reciprocal_ranks.mean()),
        "matching_page_cosine": float(similarities.diagonal(dim1=-2, dim2=-1).mean()),
    }


def _find_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    )
    for layers in candidates:
        if layers is not None:
            return list(layers)
    raise TypeError("unsupported model: could not locate decoder layers")


@torch.no_grad()
def _capture(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    wanted_layers: list[int],
    doc_start: int,
    doc_len: int,
) -> dict[int, dict[str, torch.Tensor]]:
    layers = _find_layers(model)
    captured: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in wanted_layers}
    handles = []

    def save_projection(layer: int, name: str):
        def hook(_module, _inputs, output):
            captured[layer][name] = output[:, doc_start : doc_start + doc_len].detach().cpu()

        return hook

    for layer in wanted_layers:
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"layer {layer} is outside [0, {len(layers) - 1}]")
        attn = layers[layer].self_attn
        # Qwen3 normalizes each projected head before RoPE; capturing k_norm
        # keeps the probe aligned with the key representation used by attention.
        key_capture_module = getattr(attn, "k_norm", attn.k_proj)
        handles.append(key_capture_module.register_forward_hook(save_projection(layer, "k")))
        handles.append(attn.v_proj.register_forward_hook(save_projection(layer, "v")))
    try:
        model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    config = model.config
    n_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    for layer in wanted_layers:
        for name in ("k", "v"):
            tensor = captured[layer].get(name)
            if tensor is None:
                raise RuntimeError(f"projection hook did not fire for layer {layer} {name}")
            captured[layer][name] = tensor[0].reshape(doc_len, n_kv_heads, head_dim)
    return captured


def _compare(
    left: dict[int, dict[str, torch.Tensor]],
    right: dict[int, dict[str, torch.Tensor]],
    page_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for layer in sorted(left):
        row: dict[str, Any] = {"layer": layer}
        for name in ("k", "v"):
            row[name] = _cosine_stats(left[layer][name], right[layer][name])
        row["key_page_retrieval"] = _page_identity(
            left[layer]["k"], right[layer]["k"], page_size
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=args.local_files_only
    )
    if args.random_init:
        config = AutoConfig.from_pretrained(
            args.model, local_files_only=args.local_files_only
        )
        model = AutoModelForCausalLM.from_config(config, dtype=dtype).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, local_files_only=args.local_files_only
        ).to(device).eval()
    layers = _find_layers(model)
    wanted = sorted({int(item) for item in args.layers.split(",") if item.strip()})
    wanted = [layer for layer in wanted if layer < len(layers)]
    if not wanted:
        raise ValueError("none of the requested layers exists in this model")

    document = args.document_file.read_text(encoding="utf-8") if args.document_file else args.document
    encode = lambda text: tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    doc_ids = _repeat_to_length(encode(document), args.document_tokens)
    prefix_a = _repeat_to_length(encode("A factual technical discussion. "), args.prefix_tokens)
    prefix_b = _repeat_to_length(encode("A fictional dialogue with unrelated vocabulary. "), args.prefix_tokens)
    prefix_shift = _repeat_to_length(encode("A shifted context about mathematics. "), args.shift_prefix_tokens)
    bos = torch.tensor([tokenizer.bos_token_id], dtype=torch.long) if tokenizer.bos_token_id is not None else torch.empty(0, dtype=torch.long)

    def make_input(prefix: torch.Tensor) -> tuple[torch.Tensor, int]:
        ids = torch.cat((bos, prefix, doc_ids)).unsqueeze(0).to(device)
        return ids, bos.numel() + prefix.numel()

    inputs = {name: make_input(prefix) for name, prefix in (("a", prefix_a), ("b", prefix_b), ("shift", prefix_shift))}
    captures = {
        name: _capture(model, ids, wanted, start, doc_ids.numel())
        for name, (ids, start) in inputs.items()
    }
    result = {
        "config": {
            "model": args.model,
            "document_tokens": int(doc_ids.numel()),
            "prefix_tokens": args.prefix_tokens,
            "shift_prefix_tokens": args.shift_prefix_tokens,
            "page_size": args.page_size,
            "layers": wanted,
            "random_init": args.random_init,
            "seed": args.seed,
            "note": "K is captured before RoPE; V has no positional rotation.",
            "validity_warning": (
                "Random weights test architecture and instrumentation only; "
                "they do not establish pretrained-model reuse quality."
                if args.random_init
                else None
            ),
        },
        "same_position_different_context": _compare(captures["a"], captures["b"], args.page_size),
        "different_position_and_context": _compare(captures["a"], captures["shift"], args.page_size),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
