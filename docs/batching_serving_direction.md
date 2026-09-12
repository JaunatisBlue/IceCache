# IceCache 吞吐方向思考：Batch 之下的语义检索 KV Offload

> 日期：2026-09-11
> 定位：不是实验记录，是方向分析——回答"把 IceCache 从单请求算法系统升级为可 serving 的吞吐系统，有没有说法、说法在哪、先做什么"。
> 依据：`experiment/05_dci_system_optimization.md`（TPOT 105.6ms / DCI 34.12ms）、`experiment/06`（叶子退化结论）、代码证据（`source/icecache/infer_state.py` @1837 行）、continuous batching 文献。

---

## 0. TL;DR

**有说法，而且是空档，但不是"把 batch_size==1 的 assert 改掉"那么简单。**

核心判断一句话：**batch 之下，IceCache 先撞的是 CPU 选择墙（B≈2-3 就饱和），而不是 H2D 带宽墙（B≈8-16 才到）**。单请求优化（exp 05 的 8.3%）救不了这个——它优化的是每请求成本，batch 把成本乘以 B。所以吞吐方向的研究叙事必须是把**每请求选择成本压缩 ~10×**（接 exp 06 的叶子退化结论）+ **跨请求页面共享**（接 agent/RAG 负载的公共前缀）+ **批级流水线调度**，三者缺一不可。

这构成一个完整的论文级命题：**"Serving-aware 语义检索 KV offload"**——目前所有语义/检索式 KV offload 系统（IceCache、FreeKV、OmniKV、ShadowKV、ArkVale、Quest）全部只在单请求下评测，而 vLLM/SGLang 这类 serving 系统里又没有任何语义检索式 offload。交叉处是空的。

---

## 1. 现状：代码里的三条单请求硬约束

（代码证据，`~/IceCache/IceCache/source/icecache/infer_state.py`，1837 行版本）

| # | 位置 | 内容 | 含义 |
|---|---|---|---|
| 1 | L1571 | `assert kvc.batch_size == 1` | decode 选择路径只支持单请求，是硬闸门 |
| 2 | L356-357 | `ThreadPoolExecutor(max_workers=1)` | 调度用单线程 worker（DCI 计算内部的 32-worker 并行是另一层，见 exp 05） |
| 3 | L638 | 每请求 × 每层一棵 `DCI(... num_inst=n_kv_heads ...)` | 索引结构 per-request，天然无法跨请求共享或合并查询 |

已有的批式原语：`_dci_copy_to_buffer_batched`（L1536）——但它 batch 的是**层**（把 reuse-3 的三层 KV 排成一个 staging transfer，30 次 gather/H2D 合并为 10 次），不是**请求**。pinned staging buffer、`non_blocking` 拷贝（5 处）、async recall ring 也都是单请求语境下的。

结论：IceCache 今天是一台**单请求流水线机器**。exp 05 把这台单机调到了当前结构的合理极限（TPOT 115.2 → 105.6ms）。

---

## 2. 定量推演：batch 之下先撞哪堵墙

以 exp 05 的固定配置估算（Llama-3.1-8B、40k token 上下文、64 GPU pages、page 16、reuse 3、单张 A100 80G PCIe）：

### 2.1 基础数字

- 单请求 TPOT：105.6ms，其中 DCI 选择 34.12ms（吃满 32 物理核）、GPU decode 计算约 70ms
- 每次召回的 H2D 流量：query 层数 ≈ 32/3 = 11 层 × (64 pages × 16 tokens) × (2 × 8 KV heads × 128 dim × 2B fp16) = **约 46 MB / token / 请求**
- PCIe 4.0 x16 有效带宽按 ~22 GB/s 算：单请求传输 ≈ **2.1 ms/token**（目前完全可藏）

### 2.2 三堵墙随 B 的变化

| 墙 | 随 B 增长 | 到达点 | 说明 |
|---|---|---|---|
| **CPU 选择墙** | B × 34ms 核时/步 | **B ≈ 2-3** | 34ms/请求是吃满 32 核的墙钟时间。batch 一步的 GPU 时间约 70-90ms（权重量主导，对 B 不敏感），CPU 必须在这段时间内算完 B 个请求的选择——B×34ms 超过它，DCI 从"被藏住"变成"在关键路径上" |
| **H2D 带宽墙** | B × 2.1ms/步 | **B ≈ 8-16** | B=8 → 17ms；B=16 → 34ms；B=32 → 68ms。超过 GPU 步时间（~15-25ms）后传输变成关键路径，overlap 结构性失效 |
| **GPU 计算墙** | 缓慢增长 | B ≈ 64+ | decode 是访存主导（16GB 权重 / ~2TB/s ≈ 8ms 基线），varlen/ragged attention 内核现成（FlashAttention/FlashInfer），**这堵墙离得最远，不是问题** |

### 2.3 这组数字给出的研究议程

墙的顺序决定了做事的顺序：

1. **先压 CPU 选择成本**（34 → 2-4ms/请求，~10×）——不解决这个，serving 无从谈起；
2. **再打 H2D 流量**（跨请求共享 + 自适应 k，把 46MB/tok/req 的基数打下来）；
3. **最后才是调度器/流水线工程**（把前两步的成果组织成 continuous batching）。

顺序反了就是白干：在每请求 34ms 的成本下做任何调度器，吞吐上限都是 B=2。

---

## 3. Serving 文献框架：IceCache 与 continuous batching 的错配

Continuous batching（ORCA/vLLM）成立依赖两个前提：

1. **iteration-level scheduling**：每个 decode step 重新组装 batch，请求随到随进随出；
2. **PagedAttention 式统一页池**：所有请求的 KV 在同一个 block table 管理下，GPU 侧按 block table 寻址。

IceCache 与这套模型的**结构性错配**有三处：

- **索引私有**：每请求一棵语义 DCI 树（建树发生在该请求 prefill 时），没有统一页池的概念。vLLM 的 block table 是"位置寻址"，DCI 树是"语义寻址"，两者对"页"的定义都不同。
- **选择同步**：当前实现里 recall 是 decode step 内的同步操作（async ring 只是单请求内的异步）。continuous batching 要求选择可以滞后于调度决策——即"token 的页还没到，请求就先不进这个 step"，这需要调度器理解每个请求的 recall 状态。
- **prefill 竞争**：continuous batching 下新请求的 prefill（含建树：40k token × 8 KV 头的全量插入）与在跑请求的 decode 选择**争同一批 CPU 核**。这是单请求世界里不存在的问题。

对应文献里的现成答案各有局限：selective batching（attention 按请求、其余算子按 token 合并）解决 GPU 侧；prefill/decode disaggregation 解决 prefill 竞争；但**"batch 内 B 棵语义树的联合检索"没有现成答案**——这正是要做的东西。

另外一个现实参照：`~/vllm-continuum`（Hanchenli 的 vLLM fork，Continuum scheduling + KV pin 调优，SWE-bench agent 负载）说明你要打的场景——**agent 多请求、超长公共前缀、迭代式访问**——正是 batch + 长上下文交汇的地方，也是"跨请求共享"最有油水的负载。

---

## 4. 研究定位：为什么这是空档

- **算法侧**：FreeKV（speculative retrieval + 隐藏开销）、OmniKV（跨层复用）、ShadowKV、ArkVale、Quest——查其论文评测，全部是 batch_size=1 的精度/延迟实验，**没有一个给出 multi-request 吞吐数字**。它们的"overhead 占比"叙事（FreeKV 报 ArkVale 94%、ShadowKV 73%、InfiniGen 53%）全在单请求语境下。
- **系统侧**：vLLM/SGLang 的 CPU offload（官方 `cpu_offload_gb`、LMCache 类项目）是**全量换入换出**或前缀缓存，没有语义检索——它们牺牲精度换吞吐，正好与 IceCache 的卖点（近似 top-k 召回）互补而不重叠。
- **负载侧**：agent 工作负载（SWE-bench、多轮工具调用）天然带公共前缀 + 长上下文 + 高并发，是"语义检索 offload 必须 batch 化"的最好论据，也是审稿人最容易接受的 motivation。

所以命题可以写成：**"语义检索式 KV offload 在 continuous batching 下的三堵墙（选择成本、传输带宽、调度耦合）及其解法"**——三个子问题各对应一节贡献，逻辑闭环。

---

## 5. 技术路线：四个子问题

### 5.1 选择成本 10× 压缩（CPU 墙的解，最优先）

exp 06 已经把病灶找到了：**叶子层全扫描退化**——叶子簇平均 78.6 个点 vs 请求 60 个邻居，`num_neighbours >= num_points` 在叶子层大面积触发，98% 的 e2 暴力发生在 level 0；正常检索（e5）占 82% 耗时。这是单请求视角下的诊断，batch 视角下它就是"选择成本墙"的攻关地图：

- **增大叶子簇 / 两级页层次**：让叶子点数 ≫ num_neighbours，最后一级仍走投影检索而不是全排序。直接动 M-DCI 树的构建参数或加一层"超级叶"。
- **置信度驱动的查询频率**：exp 03/04 已经证明相邻 token 页选择强相关（全 top-k 重叠 64.54%）但又不稳定（≥90% 重叠仅 0.79%）——固定跨层复用（reuse 6）在单请求上已被质量门否决，但**per-head 置信度驱动的自适应复用**仍在桌上，且它的收益在 batch 下被放大 B 倍。
- **混合粗评分**：Quest 式的页级元数据（每页 K 的 min/max 或均值，常驻 GPU、体积极小）做 GPU 侧粗筛，DCI 只对"粗筛后仍不确定"的请求/头做精查。这把大部分 DCI 查询从 CPU 挪走，是成本数量级下降的候选路径，但会稀释"语义树"的卖点，需要实验定夺（可做成 ablation 而非主方法）。

### 5.2 跨请求页面共享与去重（带宽墙的解）

KV 页面本身是请求私有的，但 agent/RAG 负载里**同一前缀的页在 batch 内逐字相同**：

- 建树/召回阶段加**前缀感知**：共享前缀段的页在 CPU 池里存一份、H2D 传一份、GPU 侧按引用计数挂到多个请求的 block table（vLLM 的 prefix caching 已有页池 + 引用计数基建可借）。
- 收益上界 = batch 内前缀重叠率 × 召回预算中前缀页占比。SWE-bench 类负载这个比例可以很高（公共 system prompt + repo 上下文几十 k token），值得先写个分析脚本对真实 trace 量一下——**这是一个成本极低、可以先跑的验证实验**。
- 语义层面还有一个更激进的想象：不同请求的**语义等价页**（内容高度相似的文档片段）是否可以合并传输——风险大、精度难保证，不推荐作为主贡献，留作 discussion。

### 5.3 带宽预算下的自适应召回（带宽墙的第二解）

PCIe 带宽是全局资源，continuous batching 下应由调度器按预算分配：

- 每 decode step 给 H2D 定一个预算（如 GPU 步时间 × 目标 overlap 率），在 B 个请求间分配 k_i；
- 分配依据用已有的置信度信号（exp 03/04 的 per-head/相邻 token churn）：不确定的请求多给 k，确定的少给；
- 这把"精度-吞吐"从全局开关变成**每请求每 step 的连续旋钮**，是 FreeKV 全局余弦阈值之外的差异化点（呼应之前竞品分析里定的 per-head confidence 路线）。

### 5.4 批级流水线与调度（把三者组织起来）

- 把单请求 async ring 推广为**多请求召回流水线**：32 核在 B 个请求的选择任务间做 work-stealing，每 step 的 CPU 预算 = GPU 步时间；调度器只放行"页已就绪"的 token（类比 FreeKV 的 speculative 隐藏，但从单请求机制升级为批调度策略）。
- 新请求 prefill 与 decode 的 CPU 竞争：prefill/decode 分离（不同进程/NUMA 域），与 codex 已做的 NUMA node 0 绑定实验是同一条线。
- 并发 insert/query 的树安全性是必须补的工程课：M-DCI 树目前看不出有并发读写保护，continuous batching 下 prefill 建树与 decode 查询会真正并发。

---

## 6. 实验设计草案

**主指标**：throughput (req/s、tok/s)、TPOT 分布、SLO attainment（如 TPOT<200ms 的请求占比）、LongBench/Qasper 精度（质量门不变）。

**基线**：
1. vLLM 全量 GPU KV（短上下文上限参照）；
2. vLLM CPU offload 全量换入换出（naive offload 下限）；
3. IceCache 现状（单请求串行服务 N 个请求 = 吞吐下限）；
4.（可选）FreeKV/OmniKV 若有开源 serving 改造则加，否则引其单请求数字说明空档。

**负载**：LongBench 多请求重放（batch 内混合不同任务/长度）；ShareGPT 长对话轨迹；SWE-bench agent 轨迹（前缀共享最充分，对 5.2 最有利）。

**预期核心曲线**：batch size B 从 1 扫到 32——
- 现状 IceCache：吞吐在 B≈2-3 就平掉（CPU 墙），复现第 2 节推演即可作为 motivation 实验 expenditures；
- 加 5.1：墙推到 B≈8-16（撞带宽墙）；
- 加 5.2+5.3：带宽墙再外推，同时精度不降（质量门）。
这条"墙逐步外推"的曲线就是论文的主图。

**第一步（本周可做，不等任何改造）**：两个零成本验证——
1. `assert batch_size==1` 不动，直接**串行地**跑 B 个请求混跑脚本，测吞吐饱和点，验证 B≈2-3 的推演；
2. 写脚本统计 SWE-bench/LongBench 轨迹在典型 batch 组队下的**前缀页重叠率**，给 5.2 定收益上界。

---

## 7. 落地顺序与并行约定

| 阶段 | 内容 | 依赖 |
|---|---|---|
| 0 | 两个零成本验证（§6 第一步） | 无，纯外层脚本 |
| 1 | 选择成本压缩：叶子簇尺寸 / num_neighbours 失衡修复（接 exp 06 结论） | 需动 M-DCI 内部，**先与 codex 对齐，他在 §06 的分析已到"为什么叶子簇这么小"这一步** |
| 2 | 跨请求前缀页共享原型（共享 CPU 池 + 引用计数 + H2D 去重） | 依赖阶段 1 的成本余量 |
| 3 | 多请求召回流水线 + 简易 continuous batching loop（自研，参考 vllm-continuum 的调度逻辑而非移植进 vLLM） | 依赖 1+2 |
| 4 | 带宽预算自适应 k（接 per-head 置信度线） | 依赖 3 的调度骨架 |
| 5 | （远期）vLLM/SGLang 集成或 disaggregation | 论文故事成立后再谈 |

**并行开发警告**：`infer_state.py` 仍由 codex 在并行修改（本次核对时 1837 行、exp 06 是当天新写的）。动 decode 主路径前先确认他的工作面；阶段 0 的两个验证全部在外层脚本做，不碰 `source/icecache/`。

---

## 附：数字与代码证据索引

- TPOT 105.6 / DCI 34.12 / FP16+GQA merge 组合：`experiment/05_dci_system_optimization.md`
- 叶子层 e2 退化（78.6 点 vs 60 邻居，98% 在 level 0）：`experiment/06_mdci_adaptive_stop_review.md`
- 相邻 token 页重叠 64.54% / ≥90% 仅 0.79%：`experiment/04_qasper_dci_churn.md`
- `assert batch_size == 1`：`source/icecache/infer_state.py` L1571（1837 行版）
- `ThreadPoolExecutor(max_workers=1)`：同文件 L356
- `_dci_copy_to_buffer_batched`（按层批式）：同文件 L26, L1536
- H2D 估算：46MB/tok/req = 11 recall 层 × 1024 token × 4KB（K+V×8头×128维×fp16×2）
