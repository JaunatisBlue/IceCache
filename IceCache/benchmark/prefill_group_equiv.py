"""Same-process equivalence check for length-grouped batched prefill.

`prefill_batch(..., token_budget=N)` splits the batch into several smaller
forwards so each one pads to its own group's Lmax.  The padding must not be
observable: every request attends only its own real tokens, writes only its own
real tokens and builds its own tree, so the last-token logits must not depend on
which other requests shared its forward.

Comparing two *processes* cannot show that: DCI leaf selection is not
reproducible across processes (see the plan document), and cuBLAS may pick a
different split-k kernel for a different batch shape, so token ids can differ
without any bug.  This script therefore runs the plans back to back in ONE
process and reports

  * the noise floor  -- budget 0 vs budget 0 (identical configuration), and
  * the plan delta   -- budget 0 vs budget N,

as max |delta logits| plus the argmax agreement, so a real regression is
distinguishable from run-to-run noise.
"""

import gc
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/yx/IceCache/IceCache/source")

from icecache.adapter.modeling import enable_icecache, icecache_state  # noqa: E402
from icecache.batch import BatchInferState  # noqa: E402
from icecache.infer_state import InferState  # noqa: E402
from icecache.kv_cache import KvPool  # noqa: E402

MODEL = "/opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PAGE_SIZE = 16
PAGE_BUDGET = 16
BATCH = 8
BASE_TOKENS = 1024
SKEW_RATIO = 4
BUDGET = 8192

DEV = torch.device("cuda:0")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def make_prompt(length, seed):
    rng = __import__("random").Random(seed)
    words = (
        "analysis retrieval memory computer database inference token attention "
        "network system document answer reasoning research science algorithm"
    ).split()
    paragraphs = [" ".join(rng.choices(words, k=96)) for _ in range(length // 64 + 2)]
    ids = TOKENIZER("\n".join(paragraphs), add_special_tokens=True).input_ids
    return torch.tensor([ids[:length]], dtype=torch.long, device=DEV)


def pages_for(tokens):
    return max(1, -(-tokens // PAGE_SIZE))


def run(model, cfg, prompts, page_budget, budget, enable=False):
    gpu_pool = KvPool(cfg["gpu_pages"], PAGE_SIZE, cfg["nkv"], cfg["hd"],
                      torch.float16, DEV, (0, 2, 1, 3))

    def make_state():
        return InferState(
            n_layers=cfg["n_layers"], n_qo_heads=cfg["nqo"], n_kv_heads=cfg["nkv"],
            head_dim=cfg["hd"], page_size=PAGE_SIZE, dtype=torch.float16, device=DEV,
            page_budgets=page_budget, page_topks=0, n_sink_pages=2, n_win_pages=2,
            n_prefetch_layers=0, n_reuse_layers=0, n_max_pages=cfg["gpu_pages"],
            n_max_cpu_pages=16384, gpu_pool=gpu_pool)

    states = [make_state() for _ in range(BATCH)]
    if enable:
        enable_icecache(model, dtype=torch.float16, device=DEV, infer_state=states[0])
    batch = BatchInferState(states, query_backend="native", query_threads=16,
                            prefilled=False)
    with torch.no_grad():
        logits, tokens = batch.prefill_batch(
            model, [p[0].clone() for p in prompts], token_budget=budget)
    result = (logits.float().clone(),
              [int(t.item()) for t in tokens],
              batch.prefill_groups,
              batch.prefill_padded_rows,
              batch.prefill_real_rows)
    del batch, states, gpu_pool
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True).to(DEV).eval()
    c = model.config
    hd = c.head_dim or c.hidden_size // c.num_attention_heads
    lens = [BASE_TOKENS * SKEW_RATIO] + [
        BASE_TOKENS + i * PAGE_SIZE for i in range(1, BATCH)]
    prompts = [make_prompt(L, 101 + i) for i, L in enumerate(lens)]
    cfg = {
        "n_layers": c.num_hidden_layers, "nqo": c.num_attention_heads,
        "nkv": c.num_key_value_heads, "hd": hd,
        "gpu_pages": c.num_hidden_layers * sum(pages_for(L) for L in lens) + 512,
    }
    print(f"lens={lens} gpu_pages={cfg['gpu_pages']}", flush=True)

    runs = []
    for index, (tag, budget) in enumerate((
            ("budget=0    #1", None), ("budget=0    #2", None),
            (f"budget={BUDGET}     ", BUDGET))):
        logits, tokens, groups, padded, real = run(
            model, cfg, prompts, PAGE_BUDGET, budget, enable=(index == 0))
        runs.append((tag, logits, tokens, groups, padded, real))
        print(f"{tag}: groups={groups} padded={padded} real={real} "
              f"pad={1 - real / padded:.1%} tokens={tokens}", flush=True)

    ref_tag, ref_logits, ref_tokens = runs[0][0], runs[0][1], runs[0][2]
    for tag, logits, tokens, *_ in runs:
        delta = (logits - ref_logits).abs().max().item()
        logit_scale = ref_logits.abs().max().item()
        agree = sum(a == b for a, b in zip(ref_tokens, tokens))
        print(f"{ref_tag} vs {tag}: max|dlogits|={delta:.3e} "
              f"(scale {logit_scale:.2f}) argmax agree={agree}/{len(tokens)}",
              flush=True)


if __name__ == "__main__":
    main()
