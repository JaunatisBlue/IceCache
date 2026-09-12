# KV Cache 语义索引 / 检索开销 —— 竞品地图与引导式阅读

> 面向 `~/IceCache` 这条研究线：DCI 检索占 TPOT 42%（71.45ms / 169.56ms），每 token 30 次 layer-level DCI query。
> 整理日期：2026-09-11

---

## 0. 先定位你自己的命题

你的问题在文献里已经有名字了，叫 **retrieval-based KV cache 的 selection/recall overhead**。有人做过定量对比，数字比你的更狠：

| 系统 | 选择+召回占延迟比例 | 出处 |
|---|---:|---|
| ArkVale | ~94% | FreeKV §1 |
| ShadowKV | ~73% | FreeKV §1 |
| InfiniGen（有 overlap）| 未隐藏部分 ~53% | FreeKV §1 |
| **IceCache（你的复现）** | **42%**（DCI selection）| 本仓库 `experiment/01` |

两个结论先摆在这：
1. **42% 不算坏**——IceCache 的重叠（pipeline + prefetch）比 ArkVale / ShadowKV 做得更好，所以裸露开销更低。
2. **但这个问题已经是显学**。文献里公认有两条对立路线：**把它做少**（reduce）vs **把它挪出关键路径**（hide）。你 AGENT.md 里那份七条路径的分析，正好复现了这个二分。
3. **注意 ICE 在 LongBench 的对手里就有 OmniKV**——也就是「跨层复用」这条路已经被 IceCache 自己列为基线了。

---

## 1. 竞品地图（三层）

### 第一层：正面竞品（同一个问题，不同解法）—— 必读

| 工作 | 会议/年份 | 核心机制 | 走哪条路 | 对你的威胁 |
|---|---|---|---|---|
| **FreeKV** | ICLR 2026 (SJTU+华为) | speculative retrieval + fine-grained correction + double-buffer streaming recall | **hide** | 🔴 最高——跨 token 复用已被做掉 |
| **OmniKV** | ICLR 2025 (蚂蚁) | filter layer 选重要 token，其他层复用其索引 | **reduce**（跨层） | 🔴 高——正是你的 `n_reuse_layers` |
| **LycheeCluster** | arXiv 2026 (哈工大+电子科大) | 结构感知变长 chunk + 三级树 + 三角不等式剪枝 + lazy update | reduce + 换粒度 | 🟠 高——唯一同样走「树」的 |
| **RetrievalAttention** | arXiv 2409.10516 | ANNS 索引 KV，attention-aware 处理 query/key OOD | 换索引 | 🟠 中高——ANNS 路线 |
| **ClusterKV** | arXiv 2412.03213 | K-means 语义簇召回（cosine 距离） | 换粒度（扁平） | 🟠 中高——最直接的语义分组同类 |
| **Louver** | arXiv 2605.06763 | 把稀疏注意力重述为 range searching，零假阴性 | 换数学框架 | 🟡 中——理论威胁大 |
| **ShadowKV** | arXiv 2410.21465 (字节+CMU) | pre-RoPE key 低秩 + landmark/outlier 选择 + value offload | hide + 换表示 | 🟡 中 |
| **InfiniGen** | 2024 | 异步预取，用上一层 query 索引预取当前层 | hide（跨层） | 🟡 中——你的 prefetch |
| **SqueezedAttention** | 2024 | 离线 K-means + 分层质心索引 | reduce | 🟢 低 |
| **MagicPIG** | 2024 | LSH 哈希表 + CPU 端 attention 估计 | hide（异构） | 🟢 低（已是 ICE 基线） |
| **PQCache** | 2024 | 乘积量化 + MIPS | reduce | 🟢 低（已是 ICE 基线） |
| **Quest / ArkVale** | 2024 | 位置分页 + page 上界估计 | — | 🟢 低（已是 ICE 基线） |

### 第二层：索引 / 算法谱系（你要改内核就得懂）

| 工作 | 关系 |
|---|---|
| **DCI / P-DCI** (Li & Malik 2017, arXiv 1703.00440) | M-DCI 的内核，必读原始定义 |
| **IceFormer** (ICLR 2024, Mao et al.) | 把 P-DCI 引进稀疏注意力的第一步 |
| **HNSW** (arXiv 1603.09320) | 同构参照物：skip list 分层 + NSW 图 |
| **Skip list** (Pugh 1990) | 随机几何分层的祖宗 |
| **RP-tree** (Dasgupta & Freund 2008) / Annoy / cover tree / LSH Forest | 「树」的其它做法，可替换内核 |
| **RAPTOR** (ICLR 2024, arXiv 2401.18059) | 「语义树」术语的正统出处（文档级） |

### 第三层：综述与背景

- **A Survey on LLM Acceleration based on KV Cache Management** — arXiv 2412.19442。当目录用，不要通读。
- H2O / StreamingLLM / SnapKV — eviction 路线，知道结论即可。

---

## 2. ⚠️ 必须先消化的坏消息：FreeKV

**FreeKV 的核心洞察，就是你实验 03 的假设，一字不差：**

> query vectors in adjacent decoding steps are highly similar (cosine > 0.9 for most heads, all heads > 0.84). This implies that selected KV pages remain almost unchanged between steps. Consequently, the current step does not need to compute and fetch immediately; it can "gamble" that its selection matches the previous step and reuse the previously retrieved results.

它比你的做法更聪明的地方在于**它不解题、它绕过**：
- 你的路线（`ICECACHE_CROSS_TOKEN_DCI`）：判断能不能复用 → 能就省掉 DCI 调用。**绑在关键路径上**。
- FreeKV 的路线：**赌**下一步的选页与上一步相同，用上一步的结果先算 attention，同时**后台**做下一步的 selection + recall；query 相似度跌破阈值才触发 fine-grained correction。**selection/recall 整体移出关键路径**。

再加系统层三件：hybrid layout（GPU 用 NHD、CPU 用 HND）、double-buffer streaming recall、近 100% 延迟隐藏。声称对 ArkVale / ShadowKV 有 **up to 13×** 加速。

同时 FreeKV 也承认了两个软肋，**这两条正好是你的机会**：
> - page-wise selection 在预算极紧时效果变差；
> - 对 DeepSeek-R1 这类推理模型，query similarity 波动更大，correction 触发更频繁，加速比下降。

**战略含义**：「跨 token 复用」这个 idea 已经不能当新意了。你的价值必须建立在**你比它做得更准**上。

---

## 3. 你的差异化空间在哪里（基于本仓库已有数据）

你手上有一个 FreeKV 没有的东西——**实验 04 的 churn 数据**：

| 指标 | 你的实测（Qasper，64 页预算） |
|---|---:|
| 相邻 token 全 top-k 页重叠 | **64.54%** |
| 完全相同的有序选择 | **0.00%** |
| 重叠 ≥ 90% | **0.79%** |
| 第一个 25% 重叠 | 52.97%（低于全集）|

这组数据直接**反证 FreeKV 的前提**。它的推理是「query 余弦 > 0.9 ⟹ 选页几乎不变」；你的实测是「query 相似度很高，但选页平均换掉 35%，且头部页比尾部页更不稳定」。

你实验 03 已经给出了根因假设，而且我认为这是**可以发表的核心论点**：

> head-mean signature 太粗。即使均值向量变化 < 5%（cos ≥ 0.95），**单个 Q-head 需要的页可能不同**。高驱逐率下这些 per-head 差异每 token 都翻转最优集，一次漏刷新会污染后续若干 token 的上下文。

于是差异化的位置就清楚了：

1. **per-head 置信度，而不是 head-mean**（你实验 03 列的候选方向 2）——把「所有 head 都同意才复用」做成一个**可证明的复用判据**，而不是拍阈值。这是 FreeKV 与 OmniKV 都没做的细粒度。
2. **per-head 预算下「何时必须刷新」的可判定准则**——FreeKV 用 0.9 余弦这种全局阈值，你用 head 级 margin/overlap 可预测性，理论上更稳。
3. **把「减少」与「隐藏」组合**：FreeKV 只 hide；OmniKV 只 reduce 跨层；你可以在 per-head reduce 的基础上叠加你已经验证过的 level-2 + FP16 + GQA merge，形成「reduce × hide」的组合拳。
4. **长生成场景的退化**——FreeKV 自己承认推理模型上会退化。你实验 03 里 Qasper 崩掉（43.2 → 17.37）正是同一现象的另一面。把它从「失败」重新叙述成「揭示了 head 级不稳定性的上界」，就是一个完整的贡献。

---

## 4. 引导式阅读路线（7 站，按顺序）

### 第 1 站 · FreeKV（ICLR 2026）
- **链接**：arXiv 2505.13109 ／ code: github.com/sjtu-zhao-lab/FreeKV ／ OpenReview: wXAn7orB1H
- **为什么现在读**：它是你这条线最近的正面竞品，先读它才能决定你还要不要做「跨 token 复用」。
- **带着这些问题读**：
  1. 它的 speculative retrieval 在**哪些层/head** 上赌？是全局一个阈值，还是有分层策略？
  2. fine-grained correction 的触发条件和粒度是什么？per-head 还是 per-layer？
  3. §1 那三张延迟拆解（ArkVale 94% / ShadowKV 73% / InfiniGen 53%）是怎么测的？口径和你 `DCI_PROFILE` 一致吗？
  4. 它怎么处理「赌错」的代价？有没有给出赌错率的定量上界？
- **重点章节**：§1（延迟拆解）、§3 算法（speculative + correction）、§4 系统（hybrid layout / double buffer）、Limitation。
- **读完你应该能回答**：如果我把 per-head 判据替掉它的全局余弦阈值，能保证不退化吗？它的系统层（double buffer / hybrid layout）我能不能直接借来用？
- **可执行动作**：把它代码拉下来，只跑它的延迟拆解脚本，和你的 `DCI_PROFILE` 对齐口径。**这一步能直接告诉你 42% 到底算好还是算差。**

### 第 2 站 · OmniKV（ICLR 2025）
- **链接**：OpenReview ulCAPXYXfa ／ code: github.com/antgroup/OmniKV
- **为什么现在读**：它做的正是你的 `n_reuse_layers=3`，而且是**跨层复用**这条路的代表作。IceCache 已经把它当基线，你必须知道它强在哪。
- **带着这些问题读**：
  1. filter layer 是手选的（配置里 `do_select_layers: "2,8,18"`）——**为什么是这三层**？有没有可解释的选择准则？这直接对应你实验 05 里 `reuse=4/6` 质量崩掉的疑问。
  2. 它观察到「即使相隔 16 层，重要 token 仍高度相似」。**这个 observation 的测量方法是什么**（Figure 1a）？你能用同样方法测 IceCache 的 DCI 选页吗？
  3. `num_wait_load_layers` 是干嘛的？它的流水线怎么设计？
- **重点章节**：§3 三个 insight（intra-layer sparsity / inter-layer similarity / variety across iterations）、Figure 1、Figure 2 三层结构图。
- **读完你应该能回答**：跨层复用的**安全层距**怎么定？有没有可能是「某些层对特别敏感、某些层几乎无所谓」？
- **可执行动作**：用 OmniKV Figure 1a 的方法，在 IceCache 上测**逐层**的选页相似度矩阵。你实验 04 只测了**相邻 token 同层**，还没测**同 token 相邻层**——而这正好是 `n_reuse_layers` 的依据。**这块数据你手上有但没有，值得补。**

### 第 3 站 · LycheeCluster（arXiv 2026）
- **链接**：arXiv 2603.08453
- **为什么现在读**：**唯一一个同样走「语义 + 层次树 + 增量更新」的竞品**，方法论上离你最近，思路还比你细一层。
- **带着这些问题读**：
  1. 它的核心批评是「固定 size 分页会切断逻辑单元（函数定义、JSON 结构）」。**IceCache 的 page size=16 固定 + 按语义分簇**，是否也踩了这个坑？它的 pilot study 说换成分块后 StrucText-Eval 准确率 +15.0%——你在 Code / few-shot 类数据集上有没有类似现象？
  2. 三级树（Coarse Units → Fine Clusters → Chunks）用 **spherical k-means**，而 IceCache 用 DCI 树。**换聚类内核的代价/收益是什么**？
  3. 三角不等式剪枝（Eqn. 2）怎么给出安全上界？**这个剪枝能不能拿来剪 DCI 的候选集**？
  4. lazy update 怎么「嫁接」新 token？比 IceCache 的增量插入简单还是复杂？
- **重点章节**：方法（chunking / 3-level index / pruning / lazy update）、StrucText-Eval 实验、vs ClusterKV 对比段。
- **读完你应该能回答**：IceCache 的「固定 page size + 语义分簇」在结构化工件上是不是有系统性缺陷？这是不是一条独立的改进线？
- **可执行动作**：拿 `lcc` / `repobench-p` 两个代码数据集，看你现在已跑出的 prediction 里是否出现「逻辑单元被切碎」导致的失败样例。**这是低成本、高信号的一步。**

### 第 4 站 · ClusterKV（arXiv 2412.03213）
- **链接**：arXiv 2412.03213
- **为什么现在读**：语义分组的**最简形态**（单层 K-means）。理解「最简版」才知道 IceCache 的多层树多付了什么、多得了什么。
- **带着这些问题读**：
  1. 它为什么**不用 L2 或内积**做聚类距离，而选 cosine？原文说法是 key 向量存在 outlier channel 会带偏 L2。**IceCache 的 DCI 用的是什么距离？受了这个问题影响吗？**（注意 IceCache 有 `T_K` 变换把 MIPS 转成 L2，§4.2 式 2/3）
  2. 它的 cluster 大小可变、token 位置不连续，因此**选 k 个簇对应的 token 数不可预测**——这是个真实的系统麻烦。IceCache 怎么规避的？（提示：node → page 的映射）
  3. 它报告 cluster 粒度 cache 命中率 63%–74%。**你的 IceCache 命中率是多少？** 这是可对标的硬指标。
- **重点章节**：§III-B 聚类、§III-C 簇粒度选择、§III-D 效率顾虑、§IV 系统设计。
- **读完你应该能回答**：语义分组的收益主要来自「减少内部碎片」还是「提高索引效率」？这两件事你分别能拿到多少？
- **可执行动作**：测 IceCache 的 **page 命中率 / 页内相关 token 占比**，和 ClusterKV 的 63–74% 对标。

### 第 5 站 · RetrievalAttention（arXiv 2409.10516）
- **链接**：arXiv 2409.10516
- **为什么现在读**：它指出了 ANNS-on-KV 的一个**根本性困难**——query 与 key 的分布不一致（OOD），这解释了很多 ANNS 方法为什么必须召回很大比例的 KV（论文说 ~20%）。
- **带着这些问题读**：
  1. 「attention-aware」的索引是怎么构造的？它怎么缓解 Q→K 检索的 OOD？
  2. 它报告只扫描 1–3% 的 key 就能达到 recall > 0.95。**IceCache 的 DCI recall 是多少？** 你的 `estimate_select_recall` 里有没有暴露这个指标？
  3. 传统 ANNS（含 RobustVamana）在 attention 向量上表现很差。**DCI 有没有同样问题？** 你的实验 04 churn 高，会不会部分就是 OOD 造成的？
- **重点章节**：§4.3–4.4（recall vs 扫描量）、§5 Related Work（对照各家索引）。
- **读完你应该能回答**：DCI 选页的失准，有多少来自「页级近似」、有多少来自「Q/K 分布不匹配」？这两个误差源该分别怎么治？
- **可执行动作**：在你的 passkey 长上下文场景上，测 DCI 的 **page-level recall**（真 top-k 页 vs DCI 返回页的重叠）。这是你目前唯一缺的核心质量指标。

### 第 6 站 · Louver（arXiv 2605.06763）
- **链接**：arXiv 2605.06763
- **为什么现在读**：它换了个数学框架——把稀疏注意力当 **range searching**，做**精确阈值检索、保证零假阴性**。如果成立，它从根上否掉了「近似索引」这条路的必要性。
- **带着这些问题读**：
  1. 「零假阴性」的代价是什么？查询复杂度、内存、增量更新成本各是多少？
  2. 它对比了 ANN/MIPS 各家（MagicPIG LSH、PQCache PQ、RetrievalAttention），**它怎么批评 DCI 这一类？**
  3. 它提到的自适应稀疏预算相关（Twilight / Tactic / SampleAttention / BLASST）里，哪一个思路可以搬到你的「自适应提前停止」上？
- **重点章节**：Introduction（问题重述）、索引设计（range searching）、与 ANN/MIPS 的对比、增量更新。
- **读完你应该能回答**：在「精确 + 零假阴性」这个更强的主张面前，「更快的近似检索」还有没有价值？如果有，价值在哪个工况？
- **可执行动作**：暂时只看，不动手。它更可能影响你的**中期路线选择**，而不是当下的 TPOT 优化。

### 第 7 站 · 索引谱系（按需，不要通读）
- **HNSW**（arXiv 1603.09320）——只读 §1 + 层级构建那一节。目的是确认你脑中那个对应关系：`HNSW = skip list 分层 + NSW 图`，`M-DCI = skip list 分层 + DCI 投影排序`。**分层机制同源。**
- **P-DCI**（arXiv 1703.00440）——M-DCI 的内核定义。对着 `dci.c` 读，重点看 lower bound 聚合与优先队列。
- **Skip list**（Pugh 1990）——半小时。看 `promotion_prob` 的几何分布来源，确认你 dci.c 里那段 `while(drand48() > promotion_prob) i++` 不是随便写的。
- **RAPTOR**（arXiv 2401.18059）——只读方法节。作为「语义树」这个说法的正统出处，写论文时的 Related Work 会用到。
- **Survey**（arXiv 2412.19442）——当检索目录。

---

## 5. 一页速查：读完每站能拿到的「硬指标对标」

| 指标 | 你的现状 | 对标来源 | 用途 |
|---|---|---|---|
| selection+recall 占延迟 | 42% | FreeKV: ArkVale 94% / ShadowKV 73% / InfiniGen 53% | 判断你的 baseline 是否已经够好 |
| page 命中率 | **待测** | ClusterKV: 63–74% | 语义分组的收益量化 |
| page-level recall | **待测** | RetrievalAttention: 扫描 1–3% 达 recall > 0.95 | 判断失准来源 |
| 相邻 token 选页重叠 | 64.54% | FreeKV 假设 >0.9 余弦 ⟹ 近乎不变 | **你的反例，核心贡献点** |
| 同 token 相邻层选页相似度 | **待测** | OmniKV Figure 1a | 给 `n_reuse_layers` 找依据 |
| 跨层复用的安全层距 | reuse=4/6 质量崩 | OmniKV 手选 filter layer | 解释你的失败 |

**两个「待测」是最高优先级**——都不需要跑新实验，只需要在现有 `infer_state.py` 里加计数器就能拿到。

---

## 6. 建议的下一步动作顺序

1. **读 FreeKV**（半天），把它的延迟拆解口径和你 `DCI_PROFILE` 对齐 → 确认 42% 的定位。
2. **补两个指标**（各 1 小时）：同 token 相邻层选页相似度、page-level recall。写进 `infer_state.py` 的 profiling。
3. **读 OmniKV §Figure 1a**，用它的方法测 IceCache 的层间相似度 → 这决定 `n_reuse_layers` 有没有「安全值」。
4. **读 LycheeCluster + ClusterKV**（各半天），确定「换粒度」是不是一条独立可做的线。
5. **重写 Related Work 定位**：把「跨 token 复用」从「我的新点」改成「FreeKV 已做，但它的前提在 per-head 粒度上不成立，这是本文的修正」。

---

## 附：链接清单

| 简称 | 链接 |
|---|---|
| FreeKV | arXiv 2505.13109 · github.com/sjtu-zhao-lab/FreeKV · OpenReview wXAn7orB1H |
| OmniKV | OpenReview ulCAPXYXfa · github.com/antgroup/OmniKV |
| LycheeCluster | arXiv 2603.08453 |
| ClusterKV | arXiv 2412.03213 |
| RetrievalAttention | arXiv 2409.10516 |
| Louver | arXiv 2605.06763 |
| ShadowKV | arXiv 2410.21465 |
| RAPTOR | arXiv 2401.18059 |
| HNSW | arXiv 1603.09320 |
| P-DCI | arXiv 1703.00440 |
| KV Cache Survey | arXiv 2412.19442 |
| IceCache | arXiv 2604.10539 · OpenReview yHxSKM9kdr |
