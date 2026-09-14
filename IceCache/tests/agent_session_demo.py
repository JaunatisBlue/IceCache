"""Minimal agent / tool-call loop on top of TokenwiseContinuationReference.

Demonstrates the phase A contract end to end:

    start(prompt) -> greedy decode a tool call -> pause
    -> append the tool result to the existing KV/DCI tree (token-wise)
    -> resume greedy decode -> final answer

The whole point is that ``_prepare_prefill()`` runs exactly once, so the DCI
tree built during the initial prompt survives the tool round-trip.

Run::

    cd /home/yx/IceCache/IceCache
    PYTHONPATH=/home/yx/IceCache/IceCache/source \\
      /home/yx/miniconda3/envs/icecache/bin/python tests/agent_session_demo.py
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get(
    "AGENT_SESSION_MODEL", "/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Available tools:
<tools>
{"name": "word_count", "description": "Count the words in a text",
 "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                "required": ["text"]}}
</tools>

To call a tool, reply with a JSON object wrapped in <tool_call></tool_call> tags.
The result will be returned to you as a tool message."""

USER_PROMPT = "How many words are in the sentence 'ice cache keeps long contexts cheap'?"

CONTEXT_PARAGRAPH = (
    "IceCache is a CPU KV cache offloading system for long-sequence inference. "
    "It keeps only sink pages, a small window of recent pages, and the pages "
    "retrieved by a dynamic continuous index on the GPU, while the rest of the "
    "key/value cache lives in pinned CPU memory. "
) * 6


def word_count(text: str) -> int:
    return len(text.split())


TOOLS = {"word_count": lambda **kw: {"words": word_count(**kw)}}


def build_prompt(tokenizer) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        # a long context so the DCI tree actually gets built (needs > 64 tokens)
        {"role": "user", "content": CONTEXT_PARAGRAPH + "\n\n" + USER_PROMPT},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids


def make_stop_condition(tokenizer, delimiters):
    encoded = {d: tokenizer.encode(d, add_special_tokens=False) for d in delimiters}

    def stop(committed: torch.Tensor) -> bool:
        ids = committed[0].tolist()
        return any(
            len(toks) <= len(ids) and ids[-len(toks) :] == toks
            for toks in encoded.values()
        )

    return stop


def find_tool_call(text: str):
    start = text.find(TOOL_CALL_OPEN)
    if start < 0:
        return None
    end = text.find(TOOL_CALL_CLOSE, start)
    if end < 0:
        return None
    return text[start + len(TOOL_CALL_OPEN) : end].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--page-budgets", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    from icecache.adapter import enable_icecache
    from icecache.adapter.agent_session import IceCacheAgentSession
    from icecache.infer_state import InferState

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        .to("cuda")
        .eval()
    )

    # --- spy on _prepare_prefill: it must fire exactly once, at start() ---
    calls = []
    original = InferState._prepare_prefill

    def spy(self, bsz, q_len):
        calls.append(int(q_len))
        return original(self, bsz, q_len)

    InferState._prepare_prefill = spy

    enable_icecache(
        model,
        dtype=torch.float16,
        device=torch.device("cuda"),
        page_size=args.page_size,
        page_budgets=args.page_budgets,
        n_sink_pages=2,
        n_win_pages=2,
        page_topks=0,
        n_max_bytes=4 << 28,
        n_max_cpu_bytes=8 << 28,
    )
    session = IceCacheAgentSession(model)

    prompt_ids = build_prompt(tokenizer)
    print(f"initial prompt: {prompt_ids.shape[1]} tokens")

    # ---------------------------------------------------------------- turn 1
    session.start(prompt_ids)
    print(f"after start(): seq_len={session.seq_len} dci_active={session.dci_active}")

    stop = make_stop_condition(tokenizer, [TOOL_CALL_CLOSE])
    session.generate_greedy(max_new_tokens=args.max_new_tokens, stop_condition=stop)

    assistant_text = tokenizer.decode(
        session.committed_ids[0, prompt_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"\nassistant turn 1:\n{assistant_text}\n")

    # the last tool-call token must already be inside the KV cache
    assert session.seq_len == session.committed_ids.shape[1], "tail token not committed"
    print(f"tool-call tail committed: committed_ids={session.committed_ids.shape[1]} "
          f"== seq_len={session.seq_len}")

    call = find_tool_call(assistant_text)
    if call is None:
        print("model did not emit a tool call; stopping here")
        return

    import json

    payload = json.loads(call)
    name = payload["name"]
    result = TOOLS[name](**payload.get("arguments", payload.get("parameters", {})))
    print(f"tool {name} -> {result}")

    # ---------------------------------------------------------------- turn 2
    # Fold the tool result into the transcript and append only the new suffix.
    tool_text = (
        f"\n<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n"
        f"{json.dumps(result)}<|eot_id|>"
    )
    tool_ids = tokenizer(tool_text, add_special_tokens=False, return_tensors="pt").input_ids
    tool_ids = tool_ids.to(session.committed_ids.device)

    full_transcript = torch.cat([session.committed_ids, tool_ids], dim=-1)
    appended = session.append_transcript(full_transcript)
    print(f"\nappended {appended} tool-result tokens token-wise")
    print(f"seq_len={session.seq_len} committed={session.committed_ids.shape[1]} "
          f"pending_offload={session.has_pending_offload}")

    session.generate_greedy(max_new_tokens=args.max_new_tokens)
    answer = tokenizer.decode(
        session.committed_ids[0, full_transcript.shape[1] :], skip_special_tokens=True
    )

    # ---------------------------------------------------------------- report
    print(f"\nfinal answer:\n{answer}\n")
    print("=" * 60)
    print(f"committed_ids        = {session.committed_ids.shape[1]}")
    print(f"state.seq_len        = {session.seq_len}")
    print(f"_prepare_prefill     = {calls}  (must have length 1)")
    print(f"dci points per layer = {session.n_dci_points[:6]} ...")
    print(f"dci object ids       = {[i is not None for i in session.dci_object_ids]}")
    assert len(calls) == 1, f"the DCI tree was rebuilt: {calls}"
    assert session.seq_len == session.committed_ids.shape[1]
    print("OK: one prefill, tree preserved, transcript and KV in sync")


if __name__ == "__main__":
    main()
