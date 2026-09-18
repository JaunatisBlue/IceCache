# IceCache batch 并行化方案（借鉴 vLLM / SGLang）

日期：2026-09-18 ｜ 基线提交：`142657a batch v3.0` ｜ 机型：A100 80GB PCIe，Llama-3.1-8B-Instruct，fp16
上游参考：`code_ref/vllm`（V1 架构）、`code_ref/sglang`（行号对应当前 checkout）
相关：`BATCH_CODE_FLOW.md`（调用链走读）、`BATCH_DECODE.md`、`BATCH_B8_SERIAL_VS_PARALLEL.md`

---

## 0. 结论摘要

1. **当前 batch 的并行度确实不好**：B=1 → B=8，每 token 成本只从 124 ms 降到 84.5 ms（**1.47×**），远不是 8×。
2. **瓶颈不在 GPU，而在 CPU 侧的「逐请求 × 逐层」操作**：批量 decode 每步约有 **3000 次强制 host 往返**（显式 `synchronize` + `.item()/.cpu()/.tolist()` 隐式同步 + 小张量 H↔D）。GPU 每个 kernel 之间被 Python 级同步反复饿死，所以 8 行几乎白加。
3. **prefill 侧的批量在长度均匀时赚不到**：1k token 的单请求 prefill 本身已经打满 A100，批量化只额外增加 padding（`[8,1136]`=9088 行 vs 真实 8640 行），实测 3.005 s（批量）vs 2.881 s（串行 8 次）——**没有收益**。
   **但这条只在长度均匀时成立**：一旦有长尾（agent 场景里一大片 tool result），整批会被 padding 吃掉大半 —— 实测 `[4096,1024..1136]` 是 64.3% padding、5.396 s，而**按长度分组后降到 21.5% / 3.968 s（−26.5%）并追平串行**（见「阶段 3 实施结果」）。
4. **范式差距（最关键）**：vLLM / SGLang 是「**单线程每步做一次整批操作 + 并行全在 GPU 内**」；IceCache 是「**每请求一个 CPU 线程 + 每请求一棵独立树 + 只有 query 内核并行**」。前者把并发放在 GPU，后者把并发放在 CPU 线程——这就是差距的根。
5. 因此下一步并行化的**第一优先级不是"再并行"**，而是**把逐请求的 CPU 操作整批化（去同步）**；第二优先级才是把 `retire/admit` 与建树收敛到**一个共享工作池**（任务粒度 = `(请求, 层)`，线程服务不同 prompt）；第三是 **chunked prefill + prefill/decode 混批**。

---

## 0.5 阶段 1 实施结果（2026-09-18 已落地、已验证）

**改了什么**（保持 serial(B=1) 行为不变）：
1. `validate_ready` 移出每步热路径 → 新增 `_validate_step()`（主机侧零同步的廉价检查），深度页归属扫描改由 `ICECACHE_BATCH_DEEP_VALIDATE=1` 或构造/`admit`/显式调用触发。
2. `build_attention_metadata` 全程 GPU：页重叠检查改成设备侧 `torch.unique(...).numel()`（`numel()` 是元数据，不触发同步）；越界检查由「每请求一次」改为「整批一次 `.item()`」。
3. `InferState.recall` 拆成「设备张量入口 + `recall_np`（吃主机数组）」；`_DCI_query` 额外留下 `_last_recall_np/_last_evict_np/_last_nr_np` 主机副本 ⇒ 批量 recall 不再为每请求做 D2H/`.item()`。
4. 批量 recall 的拷贝统一走 **batch 级共享 stream**（`BatchInferState._c2g_stream`），显式同步从「每请求一次」降到 **每层一次**。
5. `page_valid_entries` 由「每请求一次 H2D」改为**整层一次**（把各请求的有效条目块 `np.concatenate` 后一次性上传，再 GPU→GPU 切片写入）。
6. query 侧：每层一次 D2H（`q_cpu = query_states.detach().cpu()`），serial 与 native 回灌都复用；`_query_one` 接受主机张量。

**实测（B=8，serial 后端，8+4+4 步）**：

| 指标 | 阶段 1 前 | 阶段 1 后 | 变化 |
|---|---:|---:|---:|
| 每步耗时 | 676.16 ms | **552.23 ms** | **−18.3%** |
| 聚合吞吐 | 11.83 tok/s | **14.49 tok/s** | **+22.5%** |
| 相对 B=1 的每 token 成本 | 1.47× | **1.66×** | — |

B=1/B=2 同步受益：124.26→114.47 ms（−7.9%）、195.89→170.55 ms（−12.9%）。
**数值中立性**：`generated_token_ids` 在 B=1/2/8 三档与改动前**逐 token 完全一致** ⇒ 本次是纯主机侧重排，数学未变。

**顺带的关键发现：native 批量查询首次反超 serial**（因为它用同一份 `q_cpu`，回灌不再逐请求 D2H）：

| B=8 | 每步 | 吞吐 | `batch_query`/步 | parity |
|---|---:|---:|---:|---:|
| serial | 552.23 ms | 14.49 tok/s | 292.7 ms | — |
| **native** | **531.28 ms** | **15.06 tok/s** | 272.3 ms | 12288 ✓ |

（改动前 native 是 4.829 s vs serial 4.484 s —— **更慢**；阶段 1 后反超，且与 serial 逐 token 一致。）

**剩余瓶颈已被收窄**：`batch_query` 仍占步长 **51–55%**，且成本**线性于 B**（约 1.06 ms/(请求·层)，B=1 时 1.23 ms 同量级）⇒ 阶段 1 省下的 124 ms 全部来自 recall/valid/sync 那一段，**剩下的就是「逐请求的 DCI 查询块」本身**。这正是阶段 2 要打的（把 per-request 工作放进共享工作池 / 向量化其记账）。

### 阶段 2 实施结果（2026-09-18）

**先测量再动手。** 新增环境变量门控的查询分段计时器（`ICECACHE_QPROF=1`，默认零开销）以及
`native_search_seconds` / `native_bookkeep_seconds` / `serial_query_seconds` 三个计数（均已进 probe 输出）。

B=8 native 的每步构成（步长 535.6 ms）：

| 段 | ms/步 | 占步长 |
|---|---:|---:|
| `batch_query` 合计 | 277.9 | 51.9% |
| ├ native 搜索（已 16 线程并行） | 89.6 | 16.7% |
| └ 逐请求记账（单线程） | **182.4** | **34.1%** |
| 非 query 部分 | 257.8 | 48.1% |

`_DCI_query` 每次调用 **0.643 ms**（3968 次），拆分为 `pre` 0.205 / `diff` 0.168 / `upload` 0.209 / `other` 0.061。

**关键判断（修正了原计划的一部分）**：逐请求记账是 Python/numpy 为主、受 GIL 约束 ⇒ **开线程池收益有限**；
正确做法是 vLLM/SGLang 那一套——**把它从设备往返里解放出来、并按整批处理**。本阶段先落地第一步：

> `ccc` / `cc2gp` 从 GPU 张量迁到**主机 numpy**（与紧邻的 `page_address_buffer` 一致）。这两张表本来就只在主机侧使用
> （传给 DCI 绑定、做记账索引），放在设备上每 (请求,层) 要付 **2 次 D2H + 1 次 H2D**。

**结果（配对同场比较）**：

| | 每步 | 吞吐 | `batch_query`/步 | qprof `diff` |
|---|---:|---:|---:|---:|
| 表在 GPU | 572.78 ms | 13.97 tok/s | 282.7 ms | 0.168 ms/调用 |
| **表在主机** | **482.71 ms** | **16.57 tok/s** | 250.0 ms | **0.070 ms/调用（−58%）** |

逐 token 与改前一致 ✓、native parity 12288 ✓、8 棵树全建 ✓。

**阶段 1+2 累计（B 扫描，native；旧口径，修正见阶段 5 实施结果第 4 条）**：

| B | 每步 | 每 token | 吞吐 | 相对 B=1 |
|---:|---:|---:|---:|---:|
| 1 | 114.22 ms | 114.22 | 8.75 tok/s | 1.00× |
| 2 | 167.20 ms | 83.60 | 11.96 tok/s | 1.37× |
| 8 | **483.43 ms** | **60.43** | **16.55 tok/s** | **1.89×** |

对比阶段 1 之前（B=8：676.16 ms / 84.5 ms/token / 11.83 tok/s / 1.47×）：
**每步 −28.5%、吞吐 +39.9%、批量扩展性 1.47× → 1.89×**。

**阶段 2 剩余的大杠杆**：query 块仍占约 **52%**，其中 `pre`(0.190) + `upload`(0.186) + `other`(0.058) = **0.434 ms/调用仍是逐请求单线程**。
要再上一个台阶，必须把整批 8 个请求的记账**合并成一次**（batched `_DCI_query`：一次 reshape/去重、一次 `diff_pages_by_head`、一次上传），
即 vLLM/SGLang 的「每步一次整批操作」范式。**线程池则留给 prefill 建树**（那里的工作是 C++ 拷贝，能真并行、且可与 forward 重叠）。

### 阶段 2b 实施结果（2026-09-18，记账合并）

**先证伪一条**：把 `first_k_unique` 向量化（一次处理整批的行）在 200 组随机用例上**完全等价（0 mismatch，含重复值与 -1 哨兵）**，
但只快 **1.2×**（0.349 → 0.292 ms/批）⇒ `pre` 桶是固有 numpy 工作量，**合并去重不划算，放弃**。

**落地的是 `upload` 桶的合并**（真正可合并的部分）：让 `_DCI_query(..., return_host=True)` 在批量路径下**不建每请求的设备张量、也不做每请求的 GPU 归约**，
改由 `_batch_query_impl` 在整层末尾**一次性上传**所有请求的 `eids` / `nr`（`np.stack` → 1 次 H2D），再按请求切片交给 `scatter_pages`。
（踩坑：`scatter_pages` 断言 `n_evicts` 必须是 **Long**——设备路径是 `torch.sum` 在 int32 上自动提升来的，主机数组必须显式 `int64`。）

**结果（配对同场，B=8）**：

| | 每步 | 吞吐 | `batch_query`/步 | qprof total | `upload` |
|---|---:|---:|---:|---:|---:|
| 合并前 | 482.71 ms | 16.57 tok/s | 250.0 ms | 0.504 | 0.186 |
| **合并后** | **431.45 ms** | **18.54 tok/s** | **203.0 ms** | **0.333** | **0.031（−83%）** |

逐 token 与合并前一致 ✓、native 与 serial 一致 ✓、8 棵树全建 ✓。

**阶段 1+2+2b 累计（B 扫描，native；旧口径，修正见阶段 5 实施结果第 4 条）**：

| B | 每步 | 每 token | 吞吐 | 相对 B=1 |
|---:|---:|---:|---:|---:|
| 1 | 110.55 ms | 110.55 | 9.05 tok/s | 1.00× |
| 2 | 152.52 ms | 76.26 | 13.11 tok/s | 1.45× |
| 8 | **449.12 ms** | **56.14** | **17.81 tok/s** | **1.97×** |

对比起点（B=8：676.16 ms / 84.5 ms/token / 11.83 tok/s / **1.47×**）：**每步 −34%、每 token −34%、吞吐 +51%、扩展性 1.47× → 1.97×**。

> ⚠️ **本表的绝对值与「−34%」是旧口径（含冷启动的全 8 步均值）下的数字，已被阶段 5 修正**。
> 用稳态口径（丢弃前 2 步）对基线 `142657a` 重测后的正确数字是 **B=8 655.17 → 370.63 ms（−43.4%）、12.21 → 21.59 tok/s、扩展性 1.40× → 2.09×**，
> 见「阶段 5 实施结果」第 4 条。旧口径之所以低估，是因为冷启动那一步在各 revision 里占比不同。

**这一阶段的结论**：记账侧还剩下 `pre` 0.181 + `diff` 0.067 + `other` 0.053 ≈ 0.30 ms/调用（约 75 ms/步）是**固有工作**（numpy 去重、C++ diff、调用开销），
而**搜索本身就是 90 ms/步的硬成本**（256 次 × ~1 ms CPU，已 16 线程并行）。⇒ 要继续提升，方向已不在「记账合并」，
而在 ① 减少搜索频次/层数（`layer2topk`/`n_reuse_layers` 这类算法杠杆）② **阶段 3 chunked prefill**（prefill 目前 3 s 且有 padding 浪费）③ 阶段 4 混批。

### 阶段 3 实施结果（2026-09-18，长度分组 prefill + LM head 只算末位）

**改了什么**：

1. **`prefill_batch(model, prompts, token_budget=None)` 支持长度分组**：`token_budget=None` 保持原样（整批 padding 到同一个 `Lmax`，一次 forward）；
   给定预算时用 `_plan_prefill_groups` 按长度「最长优先」装箱——只有当**分组后的 padded 行数比单次整批优 1.15×** 时才真的拆
   （`_PREFILL_GROUP_MIN_GAIN`），所以**长度均匀的批不受影响、不会白付额外 forward**。每个组一次 forward，各自的 `Lmax` 与 `_prefill_active` 独立，
   ragged attention / KV 写入 / 建树仍按请求独立，padding 依旧全程被排除。
2. **LM head 只投影每条请求的末位真实 token**：原实现为了读「最后一个 token 的 logits」把整个 padded grid 投到 vocab
   （B=8、Lmax=1136 时是 9088 行 × 128256，logits 张量 2.3 GB）；现在 patch 成 `gather` 末位再投影。
   `ICECACHE_PREFILL_FULL_LOGITS=1` 可还原旧行为，便于配对测量。

**结果 A —— 长度分组（B=8，slot0 = 4096 token，其余 1024–1136）**：

| prefill 方案 | 总耗时 | forward 数 | padded 行 | padding |
|---|---:|---:|---:|---:|
| **基线 `142657a` 整批**（padding 到 4096） | **5.667 s** | 1 | 32768 | 64.3% |
| 现在 整批（不分组，只换了 LM head） | 5.396 s | 1 | 32768 | 64.3% |
| **现在 长度分组（budget 8192）** | **3.968 s** | 2 | **14912** | **21.5%** |
| 基线 串行（逐请求 8 次） | 3.966 s | 8 | 11712 | 0 |
| 现在 串行 | 4.031 s | 8 | 11712 | 0 |

⇒ **分组后整批 prefill −30.0%（对基线 5.667 s）**，并把「整批比串行慢」这个长期结论彻底翻转成**打平**（3.968 vs 3.966 s）。
这修正了 §0 结论 3 的「prefill 批量没有收益」——那句只在**长度均匀**时成立（`[8,1136]` 只浪费 4.9%，此时分组会被 1.15× 的阈值自动拒绝、退回单次 forward）；
一旦有长尾（agent 场景里一大片 tool result），整批就会被 padding 吃掉大半，而长度分组正是针对这个形状的。

**结果 B —— LM head 只算末位（均匀长度、单组，配对同场）**：

| | prefill 总耗时 | logits 张量 |
|---|---:|---:|
| 全量 logits（旧） | 3.223 s | 2.3 GB |
| **只算末位（新）** | **3.173 s** | **2 MB** |

⇒ 省约 50 ms（−1.6%）与 2.3 GB 显存。**收益很小，记录在此以免高估**：prefill 的 3.2 s 主体是 32 层的投影与 attention，128256 词表的 head 只占这一小截。

**等价性（同进程，见 `benchmark/prefill_group_equiv.py`）**：

```
budget=0 #1 vs budget=0 #1 : max|dlogits| = 0.000e+00   argmax 8/8   ← 同进程噪声基线
budget=0 #1 vs budget=0 #2 : max|dlogits| = 0.000e+00   argmax 8/8
budget=0 #1 vs budget=8192 : max|dlogits| = 1.953e-02   argmax 8/8   (logits 量级 18.97 ⇒ 0.10%)
```

⇒ **同进程完全可复现（噪声基线恰为 0）**，分组只带来 0.10% 的 GEMM 形状舍入，argmax 全一致。
**注意**：跨进程比对 `generated_token_ids` **不是**有效判据——本次实测里「基线整批 vs 串行」这种纯 padding 差异在 8 个 slot 中有 6 个也分叉了，
与既有的「DCI 选择状态跨进程不可复现」一致。要判等价必须在同一进程内比 logits。

### 阶段 5 实施结果（2026-09-18，测量协议）

1. **修了口径**：原 `mean_step_ms_after_warmup = mean(step_ms[min(8,len):])` 与结果表里用的「主相位 8 步均值」**不是同一个量**，
   且 `--steps <= warmup` 时直接变 `None`。现在统一为「主 decode 相位、丢弃前 2 步」，并新增 `step_stats`：
   `decode` / `decode_including_cold` / `retire` / `admit` 四组，每组给 `n/mean/std/p50/p95/min/max`。
   单看均值不足以判 A/B（既有的 ±5% 抖动就藏在这里），p50/p95 一并输出。
2. **新增 `--length-profile {uniform,skewed}`**：`skewed` 让 request 0 带 `--skewed-ratio`（默认 4）倍 token，复现「长 tool result + 若干短请求」的形状。
3. **新增 `--prefill-token-budget`**，并把 `prefill_groups` / `prefill_padded_rows` / `prefill_real_rows` 写进 probe 输出，
   使 padding 浪费成为可读数字而不是估算。
4. **修口径顺带修正了一个被低估的结论**：原来的「全 8 步均值」含**冷启动第 1 步**（B=8 时 1028.5 ms vs 稳态 370 ms），
   把 B=1/B=2/B=8 的每步耗时分别抬高 13%/15%/24%。因此**阶段 1/2/2b 的累计收益是在污染口径下算的，被低估了**。
   为了给出可比的数字，把基线 `142657a` 用 `git archive` 重新展开到 `/tmp`（拷贝已编译的 `.so`，**未在共享工作区做任何 stash/checkout**），
   用同一个探针、同一协议重测：

| B | 基线 `142657a` ms/步 | 现在 ms/步 | 变化 | 基线 tok/s | 现在 tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 114.49 | **96.93** | **−15.3%** | 8.73 | 10.32 |
| 2 | 176.99 | **133.70** | **−24.5%** | 11.30 | 14.96 |
| **8** | **655.17** | **370.63** | **−43.4%** | 12.21 | **21.59** |

- **批量扩展性：基线 1.40× → 现在 2.09×**；B=8 每 token **81.90 → 46.33 ms**。
- 对照旧口径（含冷启动的全 8 步均值）：基线 131.97 / 194.81 / 708.23 → 现在 109.52 / 152.24 / 458.40，
  即表面上只有 −17.0% / −21.9% / −35.3% / 1.97×。**两者都对，但只有稳态口径能反映真实的每 token 成本**，
  因为冷启动那一步在不同 revision 里占的比重不同。
- 稳态下的方差其实很小（B=8 std **4.46 ms** / 370.63 ms ≈ 1.2%），此前记的「TPOT 抖动 ±5%」主要就是冷启动那一步。

---

## 1. 现状诊断（全部实测/取证）

### 1.1 decode 侧的扩展性

同 8 个 prompt、`--prefill-mode batched`、serial 查询后端，仅变 batch：

| batch | 每步耗时(mean, main 相位) | 每 token 成本 | 聚合吞吐 | 相对 B=1 |
|---:|---:|---:|---:|---:|
| 1 | 124.3 ms | 124.0 ms | 8.05 tok/s | 1.00× |
| 2 | 195.9 ms | 98.0 ms | 10.21 tok/s | 1.27× |
| **8** | **676.2 ms** | **84.5 ms** | **11.83 tok/s** | **1.47×** |

（数据源：`/tmp/cmp_E.json`、`/tmp/cmp_D.json`、`/tmp/cmp_A.json` 的 `step_ms`；`mean_step_ms_after_warmup` 字段因 `--steps 8` 恰好等于 warmup 裁剪长度而为 `None`，**这是探针的口径缺陷，必须修**。）

### 1.2 每步的 host 往返账（B=8 × 32 层，取证自 `142657a`）

| 操作 | 位置 | 每步次数 | 为什么慢 |
|---|---|---:|---|
| `validate_ready` 的 `c2p.reshape(-1).tolist()` | `batch.py:605` | **256** | 逐(请求×层)把 GPU 页表拉主机，每次强制流排干 |
| `_query_one` 的 `q.cpu()` | `batch.py:674` | 256 | 逐请求把 query 拷回 CPU 做树查询 |
| `_DCI_query` 的 `evicted_idx.cpu()` + 赋给 `cc2gp` | `infer_state.py:1273` | 256 D→H + 256 H→D | 页索引拉回 CPU 再写回 GPU |
| `_DCI_query` 的 `torch.tensor(recall/evicted_idx, device)` | `infer_state.py:1277,1278` | 512 | 逐请求上传候选页索引 |
| `recall` 的 `torch.sum(nr).item()` | `infer_state.py:1404` | 256 | GPU 标量拉主机 |
| `recall` 的 `rids.cpu()` / `nr.cpu()` | `infer_state.py:1406,1407` | 256 + 256 | 逐请求把候选页号拉 CPU 填地址表 |
| `_recall_one` 的 `page_valid_entries` H→D | `batch.py:827-830` | 256 | 逐请求把 CPU 侧有效条目上传 GPU |
| `_recall_one` 的 `c2g_stream.synchronize()`（**唯一显式同步点**） | `batch.py:831` | **256** | 等 recall 的异步拷贝落地 |
| `_recall_one` 的 `int(nr.sum())` 守卫 | `batch.py:832` | 256 | GPU 标量拉主机 |
| `build_attention_metadata` 的 `p.tolist()` | `batch.py:634` | 256 | 逐请求把页表拉主机拼 CSR |
| `build_attention_metadata` 的 indptr H→D | `batch.py:638-641` | 32 | 每层一次 |
| `for request × for layer` 双层 Python 循环 | `batch.py:842` + `724` | 256 + 256 次迭代 | CPU 逐请求调度 |
| **FlashInfer `forward`（唯一真批量的 GPU 核）** | `batch.py:858` | **32** | — |

**合计：每步约 3000 次强制 host 往返 + 等量的小张量 H↔D 拷贝。**

> 更正：早前我把 `infer_state.py:1378` 的同步算进了热路径——它属于 `retrieve_blocks`（estimate/预取路径），**不在**批量 decode 热路径。热路径上的显式同步只有 `batch.py:831` 一处（256 次/步）；其余全是隐式同步。

### 1.3 「哪些必须串行？」——没有一个是语义上必须的

| 操作 | 判定 | 依据 |
|---|---|---|
| `recall`（填地址 → `copy_to_buffer`） | **可整批** | 地址填进 `_src_address_buffer[counter:counter+nr[i]]`（`infer_state.py:1409-1412`），`copy_to_buffer(list_size=counter)` 本来就吃「一串指针」（`:1416-1430`）⇒ 8 个请求可拼成一次拷 |
| `scatter_pages` | **可整批**（需先改缓冲布局） | `_cpp.scatter_pages(cast_buffer, pool.buffer, eids, nr)`（`:1487-1491`）接受任意 `eids/nr`；障碍是每请求各有独立 `cuda_cast_buffer` |
| `page_valid_entries` 更新 | **取数必须逐树，上传可整批** | `get_valid_entries` 是 per-tree C++ 调用（`batch.py:828`）；结果可拼成 1 个张量一次 H→D |
| `build_attention_metadata` | 已是整批一次 | `batch.py:620-641` 一次遍历即出 CSR |
| `_DCI_query` 的 diff/回灌 | **可向量化** | 目前逐请求 reshape/去重/diff（`infer_state.py:1245-1283`） |
| `validate_ready` | **可去掉/增量化** | 每步全量扫 8×32 层页表只为断言；应改为 admit/retire 时增量维护 |

### 1.4 prefill 侧：为什么批量没有收益，以及建树能否解耦

- **没有收益的原因**：`padding 到 Lmax` 后计算量 = `8×1136` 行，而真实只有 8640 行；单请求 1k-token prefill 在 A100 上已接近打满，批量只多花 5% 算力却没有吞吐收益（3.005 s vs 2.881 s）。
- **建树与 forward 的耦合**：`prefill_backup_pages`（`infer_state.py:1493-1529`）**只做 GPU→CPU 拷贝，不释放 GPU 页**；`c2p` 收缩发生在同 future 内的 `_DCI_first_call`（`:1025-1026`）；真正把页还回 `_free_ids` 的是 `alloc_page` 的**延迟回收**（`:1625-1642`，池空时才回收 `ev_gpi` 标记的页）。
- 因此 `_dci_future` 的 await（提交于 `batch.py:453-456`，同层循环内 await 于 `:441-443`，末层统一 await 于 `:336-341`）是**必要**的：既有「free-before-alloc 的延迟回收」容量约束，也有「offload 拷贝必须完成」的正确性约束。
- **⇒「把 32 层 evict/建树全部推迟到 forward 之后并行跑」不是白拿的**：它要求池子在 forward 期间**同时装下 8 请求 × 32 层 × 各自全部 prompt 页**（当前探测脚本的定容公式 `n_layers × Σpages` 恰好是这个量级，但**是否真的能全程不回收**需要实测确认，见 §5）。

---

## 2. vLLM / SGLang 到底怎么批处理（取证）

### 2.1 vLLM（V1）

| 问题 | 答案 | 证据 |
|---|---|---|
| 一个 step 怎么组织 | `Scheduler.schedule()` 一次产出 `SchedulerOutput`，**running(decode) 与 waiting(prefill) 同一批**，`GPUModelRunner.execute_model` 整批一次 forward | `v1/core/scheduler.py:99`；`v1/worker/gpu_worker.py:223,227`；`v1/worker/gpu_model_runner.py:274` |
| chunked prefill | 全局 **token budget**：长 prompt 每步只调度 `min(剩余未算 token, 剩余 budget)`，余下留到下一步；decode 与 prefill 共用同一 budget ⇒ 天然混批 | `scheduler.py:118,139,189,263,299` |
| 每层元数据构造 | **整批一次向量化**：`np.cumsum(num_scheduled_tokens)` → `query_start_loc_np` → 一次 `.to(device)`；`block_tables`/`slot_mapping` 同样整批 | `gpu_model_runner.py:459,511-515,532,534,568,508` |
| CPU 侧有「每请求线程」吗 | **没有**。块分配 / 块表更新 / 前缀命中都在**单个 scheduler 线程**里逐请求循环，每步各一次 | `kv_cache_manager.py:157`、`scheduler.py:158,275`、`gpu_model_runner.py:382`、`scheduler.py:245` |
| CUDA graph | decode 与 prefill **都可捕获**（按 padding 后的固定尺寸）；要求元数据是**可捕获张量** | `gpu_model_runner.py:150,157-159,885-892,1314,1328` |
| 前缀缓存 | 全局 **hash 块级**缓存（非树）：`cached_block_hash_to_block` | `kv_cache_manager.py:70,82,112,135` |

### 2.2 SGLang

| 问题 | 答案 | 证据 |
|---|---|---|
| batch 容器 | `ScheduleBatch` → `ForwardBatch.init_new` 借用其张量 | `schedule_batch.py:2257`；`forward_batch_info.py:399,758` |
| 分配 | `alloc_for_extend` / `alloc_for_decode` **整批一次**：一次 `alloc_token_slots` + triton kernel 写页表 | `mem_cache/allocation.py:344,53,587,16,78` |
| chunked prefill / 混批 | `--chunked-prefill-size` 切块；`--enable-mixed-chunk` 时 `new_batch.mix_with_running(running_batch)` 把 decode 与新 prefill 合成一个 batch | `managers/scheduler.py:1291,1313,4014-4017` |
| radix 树缓存 | 全局 `RadixCache`，跑在**单个 scheduler 事件循环**，每步一次；`lock_ref` 是节点引用计数**不是线程锁**（无 threading.Lock） | `radix_cache.py:321,397,457,638,264`；`scheduler.py:1906` |
| 并行来自哪 | TP/DP/EP + CUDA graph（prefill 与 decode 都捕获）+ torch.compile + kernel 内 warp/CTA；**没有任何「把请求分配 CPU 线程」的做法** | `model_runner.py:1474,1489,1100`；`flashattention_backend.py:673,675` |

### 2.3 差距表

| 环节 | vLLM | SGLang | IceCache 现在 | 差距 |
|---|---|---|---|---|
| ① prefill 批组成 | 与 decode 同批 + token budget 切块 | chunked prefill + `mix_with_running` | 一次 forward 过 B 个 padded prompt，再逐请求建树 | 有 padding 浪费；建树未批量化；无混批 |
| ② decode 批组成 | 同一 `InputBatch` 整批 | `running_batch` 持续 decode | `for request × for layer` 双层 Python 循环 | **热路径逐请求** |
| ③ 每层元数据 | 整批一次向量化 | 整批一次（`cumsum`） | 逐 `(请求,层)` 做 DCI query/回灌/CSR | **未跨请求向量化** |
| ④ KV 页分配 | 逐请求循环（但单线程） | **整批一次** + triton 写页表 | 单请求 Python `set` 操作 | 未批量化 |
| ⑤ 语义/前缀缓存 | 全局 hash 缓存，单线程 | 全局 `RadixCache`，单事件循环 | **每请求一棵 DCI 树 + 每请求独立 asyncio 线程** | 范式相反 |
| ⑥ 请求退出回收 | scheduler 标记 → 统一 free | `filter_batch` / `free_kv_row` | 每请求 `shutdown()` 2 个线程 | 开销随并发线性 |
| ⑦ CPU 并行方式 | 单线程每步整批 | 单线程每步整批 | 每请求 CPU 线程 + 仅 query 内核并行 | **根本分歧** |

**一句话**：vLLM/SGLang 把并行放在 **GPU 内**，CPU 侧每步只做一次整批操作；IceCache 把并行放在 **CPU 线程**，热路径仍是逐请求。因此「一个线程服务不同 prompt」这个要求，在 vLLM/SGLang 里的等价物**不是**「把请求分给线程」，而是「**每步把所有请求拼成一个整批张量交给 kernel**」。

---

## 3. 下一步并行化方案（分阶段，按 ROI 排序）

### 阶段 1（最高 ROI）：消灭每步 ~3000 次 host 往返——**整批化，而不是多开线程**
> 对应 §1.2 表格的每一行；依据 vLLM `gpu_model_runner.py:459-568`、SGLang `allocation.py:53,78`。

1. **`validate_ready` 从热路径移出**：改为在 `admit`/`retire`/页分配变更时**增量维护** `seen` 页集合（或只在断言开关打开时全量扫）。**去掉 256 次 `tolist()`/步**。
2. **`build_attention_metadata` 全程留在 GPU**：`indices` 用 `torch.cat`（本就是张量），`indptr`/`last`/`valid` 用张量运算构造；**去掉 256 次 `tolist()` + 32 次 H→D**。
3. **`recall` 整批化**：把 8 个请求的候选页地址拼进**同一个** `_src_address_buffer`、一次 `copy_to_buffer`；相应地 `rids.cpu()`/`nr.cpu()`/`sum(nr).item()` 只在整批层面做一次（或直接在 GPU 上算出地址）。**去掉 256×3 次 D→H + 256 次 sync**。
4. **`_recall_one` 的显式同步降到每层 1 次**：8 个请求的 recall 提交完后统一 `c2g_stream.synchronize()` 一次（配 3），而不是每请求一次（`batch.py:831`）。
5. **`scatter_pages` 整批化**：把 per-request `cuda_cast_buffer` 改成**共享缓冲**，一次 scatter 覆盖所有请求的 `eids/nr`（C++ 接口已接受任意集合）。**去掉 256 次提交与守卫同步**。
6. **`page_valid_entries` 一次上传**：8 个请求的 `get_valid_entries` 结果（CPU）拼成一个张量，一次 H→D。**去掉 256 次小上传**。
7. **query 侧整批**：`q` 的 D2H 从每请求一次改为**整批一次** `query_states[active].cpu()`；`_DCI_query` 的 `diff_pages_by_head` 与 `cc2gp` 更新改成接受 `[B, ...]` 的批张量（`infer_state.py:1245-1283` → batched 版本）。**去掉 256×4 次往返**。

**预期**：host 往返从 ~3000/步 → ~10/步 量级（每层：1 次 batch query + 1 次整批 recall + 1 次 sync + 1 次 CSR + 1 次 forward）。decode 步长保守估计压到 1/2–1/5（下限受 FlashInfer forward 本身约束）。

### 阶段 2：把 per-request 线程收敛为**一个共享工作池**（满足「一个线程服务不同 prompt」）
> 依据 vLLM/SGLang「单线程每步整批」+ 现有 native `_mdci_batch` 的动态任务队列范式（`mdci_batch.c:110`）。

8. 用**一个 batch 级线程池**替换「每请求 1 个 asyncio loop + 1 个 executor」（`infer_state.py:244-259`）：任务粒度 = `(请求, 层)`（或 `(请求, head)`），**动态领取**，因此同一线程会先后服务不同 prompt。
   - 收益：线程数从 `2B+1` 降到固定池大小（B=8 时 17 → 例如 4–8）；建树/淘汰跨请求并行且负载均衡。
   - 风险：现有 `_dci_future`/`_loop` 的 await 语义要重写为「按批 barrier」；需要给 DCI 树加**读写互斥**（现在完全靠阶段分离，见 `BATCH_B8_SERIAL_VS_PARALLEL.md` §2.4）。
9. 顺带把 `offload_win_page_to_DCI`（decode 侧 layer-0 的写树）也挪进该池，避免主线程被建树阻塞。

### 阶段 3：chunked prefill（去掉 padding 浪费、为混批铺路）
> 依据 vLLM `scheduler.py:118,263`、SGLang `scheduler.py:1291,1313`。

10. 引入 **token budget**：`prefill_batch` 不再把 B 个 prompt padding 到 `Lmax` 一次做完，而是按「总 token 预算」把**不同请求的 chunk 拼成一个扁平批**（`Σ tokens ≤ budget`），每步推进各请求的已算长度。收益：padding 计算归零（当前 9088 → 8640 行，且长尾差距更大时收益更明显）；为「prefill 与 decode 混批」提供结构。
11. 配套：`qo_indptr` 的构造从「每请求一次」改为「每请求每 chunk 一次」，`last_page_len` 与 `kv_indptr` 随之更新（现有 ragged CSR 机制已支持）。

### 阶段 4：prefill/decode 混批（SGLang `mix_with_running`）
> 依据 SGLang `scheduler.py:4014-4017`。

12. 同一 forward 内同时含「新请求的 prefill chunk」与「老请求的 decode token」——这正是 agent 场景（tool result 一大片 token 到达时，其它请求仍在 decode）。需要阶段 1/3 的地基。

**阶段 4 的具体改造点（已定位，未实现）**——「长度分组」已经把 prefill 拆成**多次 forward**，混批是把这些 forward 与 decode 的 forward 交替/合并：

- **为什么不冲突**：`BatchPrefillWithPagedKVCacheWrapper` 本身就接受 `qo_indptr`，而 **q_len == 1 只是它的退化情形**。
  所以「decode 的 1 token 行 + prefill 的 k token 行」可以放进**同一个 ragged 调用**，只需把两类行的 `qo_indptr` / `kv_indptr` 拼在一起。
  真正的障碍不在 attention，而在**同一个 forward 里每一层要跑两套 per-layer CPU 逻辑**：
- **改造点 1（分派）**：`BatchInferState.attention_forward` / `prefill_attention_forward` 目前是整体二选一（`forward_mode`）。
  混批需要一个按行分派：decode 行走 `decode_attention` 的 recall→query→scatter 链，prefill 行走 KV 写入→建树。
- **改造点 2（页表）**：两类行共用 `self._pool`，但 `prefill_alloc_n_tokens` 需要**连续 run**，而 decode 的 `alloc_page` 是单页；
  阶段 4 之前必须先解决 §5.2 的碎片问题（本次已经实测撞到一次：`--length-profile skewed` 用 4096-token prompt 时
  CPU 页池报 `Not enough contiguous free pages in pool`，因为 240 页/层 × 32 层远超默认 4096 页）。
- **改造点 3（LM head）**：混批里 prefill 行要「末位 token 的 logits」、decode 行要「唯一 token 的 logits」——
  两者在 `_last_token_logits` 这个 patch 里是同一件事（`last_pos` 对 decode 行就是 0），所以阶段 3 的改动正好为混批铺好了路。
- **改造点 4（DCI barrier）**：`_dci_future` 现在是「整批一个 barrier」，混批下必须按行各等各的，否则新请求的建树会拖住老请求的 decode。

**明确不做（附理由）**：把单请求路径的**跨层复用**（`n_reuse_layers` / `check_reuse`）搬进 batch。
它在单请求路径是靠 `DCI.reuse_copy_node` / `reuse_update_node`、`ccc` 复制、以及 **`c2p` 偏移别名**（`kv_caches[layer].c2p`
与 `kv_caches[reuse_id].c2p` 恒差一个常数）实现的，即**多层共享同一份 KV 页**；而 batch 的 `_check_compatible`
明确要求每层独立页，`validate_ready` 的「一页只属于一个 (请求,层)」不变量也建立在此之上。
这是一次 KV 布局级的改动，不是一次重构，**收益上限也只是搜索那 90 ms/步**（256 → ~86 次调用，约 −60 ms/步），
风险与「阶段 4 混批」同级却收益更低，故排在混批之后。

### 阶段 5：测量协议（必须与代码同步做）
13. **修探针口径**：`mean_step_ms_after_warmup` 在 `--steps == warmup` 时变 `None`（`step_ms[min(8, len):]` 的空数组）——改为「固定丢弃前 2 步」或在 steps 不足时明确报 `None` 并同时输出 `step_ms` 的分位统计。**已做（见「阶段 5 实施结果」）**。
14. **建立 B 扫描基准**：B ∈ {1,2,4,8} 的「每步延迟 / 每 token 成本 / 聚合吞吐」，作为阶段 1/2 的验收基线（当前只有 §1.1 这一组）。
15. **prefill 的公平对照**：批量 vs 串行在「真实长度分布」（长 prompt、混合长度）下重测——当前 1k×8 是中性的，不能代表长上下文场景。

---

## 4. 预期收益与风险

| 阶段 | 预期收益 | 主要风险 |
|---|---|---|
| 1 整批化去同步 | decode 步长 −50%~−80%；是**唯一能立刻见效**的一项 | 需要改 `recall/scatter` 的缓冲布局与 `_DCI_query` 的 batched 版本；`scatter_pages` 的 `nr=0` 契约（C++ 不尊重零）必须继续由调用方守护 |
| 2 共享工作池 | 线程数 2B+1 → 固定池；建树跨请求并行 | 需给 DCI 树加读写互斥；await 语义改写 — **最后没做这条（改成整批化更划算）** |
| 3 chunked prefill | 去掉 padding 浪费；长 prompt 才显著 | 改动面大（页分配 + CSR + 长度推进）— **已做「长度分组」版本，实测长尾下 −30%** |
| 4 混批 | 长 tool result 到达时不阻塞其它请求的 decode | 依赖 1/3；调度逻辑复杂度上升；**必须先解决 CPU 页池碎片（§5.2）** |
| 5 测量 | 让「有没有进步」变成可判定 | 无 — **已做完，并因此修正了阶段 1/2 被低估的结论** |

**不建议做的事**：为「让 8 个 prompt 并行」而增加 CPU 线程/进程——vLLM/SGLang 都没有这么做，且我们已实测：线程不是瓶颈，**host 往返次数**才是。

---

## 5. 待实测确认的开放问题

1. **池容量能否支撑「推迟 32 层 evict」**：`prefill_backup_pages` 不释放页、`_DCI_first_call` 只收缩 `c2p`、真正回收在 `alloc_page` 池空时（`infer_state.py:1625-1642`）。要确认「整个 forward 期间不回收」是否可行（这决定阶段 3 能否顺带把建树并行化）。
2. **`prefill_alloc_n_tokens` 的连续块需求**在多请求交错推进下是否更容易碎片失败。
   **已实测：会，而且很容易踩到。** `--length-profile skewed`（slot0 = 4096 token）在默认 `--cpu-pages-per-request 4096` 下直接崩
   `RuntimeError: Not enough contiguous free pages in pool`——该池是**每层**一套，4096 token 的 prompt 每层要约 240 页，×32 层远超 4096。
   探针需 `--cpu-pages-per-request 16384` 才跑得起来。**这是阶段 4 混批的前置障碍**（§3 阶段 4 改造点 2）。
3. **CUDA graph 可行性**：阶段 1/2 把元数据张量化后，decode 是否可捕获（`_step_impl` 目前有 Python 控制流）。
4. **DCI 树的读写互斥**：阶段 2 让建树跨请求并行后，必须显式加锁（现在靠阶段分离）。**注意：实测后阶段 2 走了「整批化」而不是「共享工作池」，所以这条暂不阻塞。**
5. **跨进程 `generated_token_ids` 不能当等价性判据**：本次实测中「基线整批 vs 串行」这种纯 padding 差异，8 个 slot 里 6 个也分叉了。
   判等价必须在**同进程内比 logits**（见 `benchmark/prefill_group_equiv.py`，噪声基线恰为 0）。这条应写进任何后续等价性检查的默认流程。

---

## 6. 参考（行号）索引

- 本文所有 IceCache 行号基于 `142657a`；vLLM/SGLang 行号基于 `code_ref/` 当前 checkout。
- 早期走读与不变量：`BATCH_CODE_FLOW.md`；B=8 对照数据：`BATCH_B8_SERIAL_VS_PARALLEL.md`；实现与限制：`BATCH_DECODE.md`。
