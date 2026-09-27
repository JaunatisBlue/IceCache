# IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yuzhenmao/IceCache/blob/main/LICENSE)

### [Project](https://yuzhenmao.github.io/IceCache/) | [Paper](https://arxiv.org/abs/2604.10539)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuzhenmao/IceCache/blob/main/demo_colab.ipynb)

IceCache is a CPU KV cache offloading system for long-sequence inference with large language models. It uses **[Multi-level Dynamic Continuous Indexing (M-DCI)](https://github.com/yuzhenmao/M-DCI)** to efficiently retrieve the most relevant KV pages from CPU memory during autoregressive decoding, enabling long-context inference without keeping the full KV cache on GPU.

![IceCache overview](ice_teas.png)

## How It Works

IceCache splits the KV cache into three regions:

- **Sink pages** — the first few tokens (always retained on GPU)
- **Window pages** — the most recent tokens (always retained on GPU)
- **Offloaded pages** — the rest, stored on CPU and retrieved by DCI-based approximate nearest neighbor search

During each decoding step, IceCache uses the query vector to search the CPU-resident DCI index and fetches the top-k most relevant KV pages back to GPU for attention computation. The DCI index is built during prefill and updated incrementally during decoding.

## Notes

- **Float32 KV cache:** If both the CPU and GPU support bf16 and you intend to use bf16, please refer to the [bf16 branch](https://github.com/yuzhenmao/IceCache/tree/bf16). Otherwise, use this branch.


- **CPU threads:** The [M-DCI](https://github.com/yuzhenmao/M-DCI) index runs on CPU and benefits significantly from parallelism. We strongly recommend running IceCache on a machine with **≥ 64 CPU threads** for best performance. Fewer threads will work but will increase TTFT (time-to-first-token) noticeably, as the DCI indexing during prefill is the bottleneck.

  You can check and set the number of threads available to OpenMP:
  ```bash
  # Check available cores
  nproc

  # Optionally pin the thread count
  export OMP_NUM_THREADS=64
  ```

## Repository Structure

```
IceCache/
├── IceCache/
│   ├── source/
│   │   └── icecache/          # Core library
│   │       ├── adapter/       # Model patching (Llama, Mistral, etc.)
│   │       ├── infer_state.py # Per-sequence state, DCI management
│   │       ├── kv_cache.py    # CPU/GPU KV cache pool
│   │       └── kernels.py     # Custom CUDA kernels
│   ├── benchmark/
│   │   ├── run_longbench.sh   # LongBench evaluation script
│   │   ├── longbench_pred.py  # LongBench inference
│   │   ├── passkey_pred.py    # Passkey retrieval benchmark
│   │   └── gsm8k_pred.py      # GSM8K reasoning benchmark
│   └── requirements.txt
```

The DCI nearest neighbor index used by IceCache lives in a separate repository: **[M-DCI](https://github.com/yuzhenmao/M-DCI)**. See its README for details on the C extension and Python bindings.

## Installation

**Requirements:** CUDA-capable GPU, GCC, OpenBLAS

### 1. Clone

```bash
git clone https://github.com/yuzhenmao/IceCache.git --recursive
cd IceCache
```

### 2. Install PyTorch

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
cd IceCache
pip install -r requirements.txt
pip install flash_attn==2.6.1 --no-build-isolation
```

### 4. Install the IceCache Python package

```bash
cd source
pip install -e . --no-build-isolation
```

### 5. Build and install M-DCI (the DCI index)

M-DCI is maintained in its own repository. Clone it anywhere outside the IceCache tree and install it into the same Python environment:

```bash
git clone https://github.com/yuzhenmao/M-DCI.git
cd M-DCI
python3 setup.py install
```

See the [M-DCI README](https://github.com/yuzhenmao/M-DCI) for build prerequisites (OpenBLAS / Apple Accelerate, OpenMP) and troubleshooting.

## Running Benchmarks

All benchmark scripts are in `IceCache/benchmark/`.

### LongBench

```bash
cd IceCache/benchmark
bash run_longbench.sh
```

Or run a single dataset:

```bash
python longbench_pred.py \
    --icecache \
    --datasets narrativeqa \
    --page-budgets 16 \
    --page-size 16 \
    --n-win-pages 2 \
    --n-sink-pages 2 \
    --model llama-3.1 \
    --name my_run
```

### Passkey Retrieval

```bash
python passkey_pred.py \
    --icecache \
    --page-budgets 16 \
    --page-size 16 \
    --n-win-pages 2 \
    --n-sink-pages 2 \
    --model llama-3.1
```

### GSM8K

```bash
python gsm8k_pred.py \
    --icecache \
    --page-budgets 16 \
    --page-size 16 \
    --n-win-pages 2 \
    --n-sink-pages 2 \
    --model mistral-7b-inst
```

### page_scan Decode-Time Breakdown

```bash
cd IceCache/benchmark        # run it where page_scan_compare.py is run
python page_scan_breakdown.py \
    --model <hf-id-or-path> \
    --backends page_scan \
    --prompt-file prompt.txt \
    --max-input-tokens 36864 \
    --max-new-tokens 64 \
    --page-size 16 \
    --page-budget 16 \
    --page-topk 0 \
    --n-reuse-layers 0 \
    --inject-pages off \
    --threads 64 \
    --output row.jsonl \
    --probe-out breakdown.json
```

That is the full flag set of the arm this probe is used for — pass the flags of the
arm being decomposed, since the breakdown is only meaningful for the configuration
that was actually run. `--max-input-tokens` in particular: the harness default
(32760) silently truncates the 36k prompt this probe is used with. The `cd` is part
of the recipe, not decoration: the probe is run where `page_scan_compare.py` is run,
from `IceCache/benchmark` — from the repo root the invocation is a plain
file-not-found.

`page_scan_breakdown.py` runs `page_scan_compare.py` in-process (same model, prompt,
seed and generation settings as a normal run) with CUDA-event wrappers on the live
decode path, and splits a page_scan decode token into four buckets:

- **Query** — the GPU page scan (`bmm` / mask / `topk` / dedup), the selected-page H2D and the result D2H.
- **Loading** — staging the retrieved pages: host address gather, `copy_to_buffer`, and the H2D/D2D into the KV pool on its own CUDA stream.
- **Decoding** — attention over the stitched cache plus each layer's MLP.
- **Others** — the residual of the token interval: projections, rope, norms, the KV scatter, lm_head, sampling and host Python.

The four buckets are **CUDA-event intervals on the device timeline, not host
timings**. Host timing is misleading on this path: `PageScan.query` ends in a
`.cpu()` copy that drains the default stream, so a host clock around it also absorbs
the previous layer's attention and MLP, and the attention/MLP kernels themselves are
pure launches whose host time is only launch overhead. The event pairs add no
synchronisation, but they are not free: at a few hundred event records per token the
probe perturbs the host-timed `tpot_s` it also reports. Measured on this box, a probe
row's host `tpot_s` is 112.2–113.6 ms against the **clean 4-pass mean of 89.53 ms**
(range 87.4–94.5) for the same config with the probe off, i.e. **+26%**; the control
that shows the device work itself is unperturbed is the probe's own device frame
`d:attn_frame`, which reads 88.9–90.2 ms in those same runs, within 1% of clean. A
probe row's own `tpot_s` / `ttft_s` must therefore not be quoted as a clean number,
and the number to use is the frame-denominated share set below, not the
interval-denominated one.

Two share sets are emitted, on two denominators, and only one of them is usable:

- `rows[].means.frame_shares_pct` — **the one to plot.** Printed as `share of the
  device frame d:attn_frame`. Numerator buckets and denominator are all CUDA device
  intervals, so the probe's own host cost cannot inflate either.
- `rows[].means.shares_pct` — printed as `share of the mean interval`. The
  denominator is the harness host interval, which contains the probe's own wrapper
  cost: inflated, do not plot.

The wrappers also measure their own host cost (`rows[].wrapper_host` in the JSON,
the `probe overhead (MEASURED ...)` lines in the log), split into the part charged
inside the `d:attn_frame` interval and the part outside it — that split is what
bounds the frame denominator's own inflation.

## Key Parameters

| Parameter | Description |
|---|---|
| `--page-budgets` | Number of KV pages kept on GPU per layer |
| `--page-size` | Number of tokens per page |
| `--n-sink-pages` | Pages always kept (first tokens) |
| `--n-win-pages` | Pages always kept (most recent tokens) |
| `--model` | Model alias (e.g. `llama-3.1`, `mistral-7b-inst`) |
| `--n_reuse_layers` | Reuse DCI index across N consecutive layers (IceCache-reuse) |

## Acknowledgements

This codebase builds upon [ArkVale](https://github.com/pku-liang/ArkVale). We thank the authors for their excellent work.

## Citation

```bibtex
@inproceedings{
mao2024iceformer,
title={IceFormer: Accelerated Inference with Long-Sequence Transformers on {CPU}s},
author={Yuzhen Mao and Martin Ester and Ke Li},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=6RR3wU4mSZ}
}
```
```bibtex
@inproceedings{
mao2026icecache,
title={IceCache: Memory-Efficient {KV}-cache Management for Long-Sequence {LLM}s},
author={Yuzhen Mao and Qitong Wang and Martin Ester and Ke Li},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=yHxSKM9kdr}
}
```
