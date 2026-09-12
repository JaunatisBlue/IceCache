# 接手状态与下一步（2026-09-12）

> codex 已收工，由我全权接手。本文档 = 现状盘点 + 三个突破 + 下一步方向。

---

## 0. 先纠正一个事实：仓库里的数字混用了**两张不同的 GPU**

这是本次盘点最重要的发现，它解释了仓库里长期存在的口径矛盾。

| | GPU 0 `0000:3b:00.0` | GPU 1 `0000:af:00.0` |
|---|---|---|
| PCIe 链路 | **width 1 / max 16**（×1，约 0.82 GB/s） | **width 16 / max 16**（健康，约 12 GB/s） |
| 现在是否可用 | ✅ 可用（唯一） | ❌ `nvidia-smi` 看不到（已故障/掉卡） |
| 谁用过它 | **exp 04–17 全部 qasper 线**（run 脚本一律 `CUDA_VISIBLE_DEVICES=0`） | **exp 05 主表**（passkey 系统优化，文档明写 "GPU 1"） |
| 该卡上的 `recall_wait` | **44–48 ms/token** | **2.17 ms/token** |

- 实测（本次）：H2D = D2H = **0.82 GB/s**，D2D = 786 GB/s → 瓶颈在链路不在 GPU；PCIe 3.0 ×1 理论 0.985 GB/s，吻合。
- ⇒ **"召回占 TPOT 46% / recall_wait 45 ms" 是 ×1 卡的伪影**，不是算法性质。
- ⇒ 健康链路上 **传输几乎免费**（exp 05：recall_wait 2.17 ms → FP16 后 **0.29 ms**/tok），主导项是 **CPU**。

**这直接回答了一个原本悬着的问题**：传输/布局方向（语义簇重排、合并 gather）在健康链路上**天花板≈0**——因为传输本来就不在关键路径上。**该方向可以放弃，不必等修链路。**

---

## 1. 三个突破

### 突破一（系统）· 质量安全的单请求端到端 8.3%：TPOT 115.21 → **105.60 ms**

在**健康卡（GPU 1）**、passkey 40k 上，四项叠加且质量不降（F1 **45.48** vs 基线 45.07）：

| 手段 | 效果 |
|---|---|
| 32 物理核绑定（Q-head 数 = 物理核数） | 115.21 → 111.40 ms |
| **FP16 recall**（AVX/F16C 转换 + 启动能力探测 + 单元测试对拍 NumPy） | `recall_wait` **2.17 → 0.29 ms/tok** |
| **native GQA page merge**（替换每 token 80 次 NumPy/Python 去重） | dedup **4.0 → 0.28 ms/tok**；postprocess 8–9 → **4.22 ms/tok** |
| 组合 | **105.60 ms**，DCI 41.41 → 34.12 ms/tok |

**同时被明确否决**（都有 A/B 与解释）：reuse 6 层（TPOT 98.20 更快但 F1 崩到 41.30）、async recall ring（反而 129.72）、next-layer prefetch（125.88）、纯 NUMA 单节点绑定（168.52）、batched cross-layer gather 在 `page_topks=0` 下反向变慢 1.0%。
> 这一段是**可直接写进论文的系统章节**：每一处都有"更快 ⇔ 不降质"的成对证据，且给出了否定理由。

### 突破二（算法）· promotion 是真杠杆，且**推翻了 IceCache 的实现直觉**

- 微基准：promotion `0.0025 → 0.10`，叶子数 264 → 503，**query 时间 0.72 → 0.36 ms**。
- qasper 双样本：promotion 0.05 使 native query **腰斩**、profiled TPOT −12.4%；0.10 使 native query **−63.3%**。
- **关键认知**：直觉上"把叶子放大以避免叶子全排序"是**错的**——全排序次数不是正确的优化目标；**更细的划分 + 更小的局部排序反而更快**。
- 全局 0.05 掉 **4.25 F1**（且损失集中在一个 yes/no 样本的措辞漂移）→ **层间梯度方案**（早层 0.01 / 晚层 0.05）把损失压到 **−0.79 F1**、native query **−21.3%**。
- 对比：`num_to_visit` 杠杆 **0×**（被 `max()` 吞掉）。**promotion 是唯一被证实有效的树侧算法旋钮。**

### 突破三（测量）· 把"检索侧优化空间"系统证伪，并留下可复用插桩

- 真实杠杆只有两个：**`num_neighbours` 10.6×**、`field_of_view` 5.6×；`num_to_visit` **0×**、`prop_to_visit` 无效、early-stop v1/v2（截断从未触发）无效。
- C 层探针：叶子平均 **78.6 点 vs 请求 60**，e2 全搜索占 18% 且 **98% 在 level 0**。
- **churn**：相邻 token 全 top-k 页重叠 **64.54%**、≥90% 仅 **0.79%**、完全相同 **0.00%** → 直接**反证 FreeKV 的"余弦 >0.9 ⟹ 选页不变"前提**。
- 一套可复用资产：M-DCI C 层插桩（出口计数/计时/分层）、统一锚点 **F1 = 45.48**、以及本次新增的 `ICECACHE_DIAG` 五段计时 + 地址 dump 工具链。
> 这组"我们排除了什么"是**方法论贡献**，也是竞品地图里缺失的一块。

---

## 2. 被证伪 / 需要作废的

| 项 | 状态 |
|---|---|
| 语义簇重排 / 合并 gather（方向①） | **放弃**。健康链路传输近乎免费，天花板≈0 |
| 跨 token 复用（FreeKV 路线） | 已被 FreeKV 占；且 03 实验 Qasper 崩到 17.37 |
| 层跳过 DCI（exp 10） | 架构不可行（三处崩溃 + 竞态） |
| per-head / per-layer 内存预算分配 | 文献红海（Fluxion/HeadWiseKV/BaKlaVa/AdaKV/HeadKV…） |
| qasper 线上的所有绝对 TPOT/recall 数字 | 需在健康链路重测，或明确标注 ×1 环境 |
| **地址"叶子连续"** | 实测推翻：同 head 相邻 leaf 相差 **131072 B**（= `cpu_n_bytes_per_page`），是规则 stride |

---

## 3. 下一步方向

### 阻塞项（我做不了，需要你）
**把可用 GPU 换到 ×16 槽位 / 救活 `0000:af:00.0` / 换机器。** 这是所有传输数字的前提。

### 不依赖链路、现在就能做的（建议主线）

1. **把 promotion 的层间分配做成算法贡献。** 已有证据：native query 有 −63.3% 的潜力，而全局配置要付 4.25 F1。**问题形式化**：在质量约束下，为每个 anchor layer（乃至 head）选 promotion，最大化 native-query 节省。exp 10 的层敏感度 oracle 是这条路的开始（它失败在"跳过 DCI"，不是失败在"分层"）——换到"分层调 promotion"这条路，架构上是安全的。
2. **补齐 qasper 线上的 CPU 侧成本清单。** 现有 profile 里除 native query 外还有 `page_metadata` 6.67 ms/tok、`index_update` 3.56、`query_postprocess` 4.35——**这些都在 CPU 上、且都不受链路影响**。先把它们各自的优化空间量出来。
3. **把 exp 05 的 8.3% 组合固化为默认配置并补一道质量门。** 目前它只在 passkey 40k 上验过；需要在 LongBench 多数据集上确认不降质。

### 修复链路之后
4. **原样重跑 `experiment/run_17_diag.sh`**（工具链已就绪），用健康链路重测成本分解，确认"传输不在关键路径"这一结论，并重新评估 batched gather（它在 `page_topks=32` 下是 +2.2%）。
5. 若健康链路上 CPU 仍主导 → **算法主线定为"降低每步选择/查询成本"**，而非"减少传输字节"。

---

## 附：关键数字速查

| 指标 | 值 | 出处 / 条件 |
|---|---|---|
| TPOT 基线 | 115.21 ms | exp05, passkey 40k, **GPU 1（健康）** |
| TPOT 最优组合 | **105.60 ms**（−8.3%），F1 45.48 | exp05, 同上 |
| `recall_wait`（健康链路） | 2.17 → **0.29** ms/tok（FP16） | exp05, GPU 1 |
| TPOT 基线 | 138.48 ms | exp10, qasper20, **GPU 0（×1）** |
| `recall_wait`（×1） | 44.46–47.90 ms/tok | exp10 / 本次 exp17 |
| `recall_gather` | 18.28–21.2 ms/tok | exp17 / exp08 |
| native DCI query | 15.7–16.9 ms/tok | exp08 / exp10 / exp17 |
| promotion 收益 | native query −21.3%（层梯度）/−63.3%（全局 0.10） | exp08 / exp07 |
| `num_to_visit` 杠杆 | 0× | 微基准 + exp14/15/16 |
| 叶子平均点数 vs 请求 | 78.6 vs 60 | C 层探针 |
| churn 重叠 / ≥90% / 全同 | 64.54% / 0.79% / 0.00% | exp04 |
| 同 head 相邻 leaf 地址间隔 | 131072 B（规则 stride） | 本次 exp17 |
| PCIe | ×1，H2D=D2H=0.82 GB/s，D2D 786 GB/s | 本次 |
