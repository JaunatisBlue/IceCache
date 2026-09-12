# IceCache 51 份实验日志系统性剖析（DCI_PROFILE 对比）

> 数据源：`~/IceCache/experiment/logs/dci_opt/*.log`，共 51 个文件，其中 30 份含有效 `DCI_PROFILE` JSON。
> 全部实测量纲：ms / token（decode 阶段）。关键缩写：TPOT = decode_tpot_ms；select = dci_select_ms/tok；
> nq = native_query_ms/tok；wait = recall_wait_ms/tok；gather = recall_gather_ms/tok；pages = recall_pages/tok。

## 1. 全景对比表（30 份有效 profile）

| log | TPOT | select | nq | wait | gather | pages | idx_upd | call/ tok | dci_share |
|---|---|---|---|---|---|---|---|---|---|
| async_recall_ab_baseline_gpu1_37k | 115.2 | 41.4 | 31.2 | **2.17** | 18.8 | 4667 | 3.57 | 10 | 0.36 |
| async_recall_ring_v2_37k | 199.3 | 64.9 | 31.5 | — | 22.0 | 5034 | 2.81 | 10 | 0.33 |
| async_recall_ring_v2_rep_37k | 129.7 | 48.3 | 37.9 | — | 21.1 | 4677 | 3.62 | 10 | 0.37 |
| async_recall_v1_37k | 208.7 | 75.0 | 42.0 | — | 22.6 | 5026 | 2.98 | 10 | 0.36 |
| baseline.log | 720.1 | 617.0 | — | — | — | — | — | 30 | 0.86 |
| batched_recall_numa0_gpu0_40k | **110.8** | 25.7 | 21.0 | 20.9 | 14.1 | 2306 | 5.94 | 10 | 0.23 |
| batched_recall_numa0_gpu0_40k_rep2 | 110.9 | 25.6 | 21.0 | 21.0 | 14.2 | 2312 | 5.93 | 10 | 0.23 |
| batched_recall_topk0_numa0_gpu0_40k | 155.6 | 41.3 | 35.8 | 44.7 | 17.1 | 4692 | 6.26 | 10 | 0.27 |
| cos005.log | **99.3** | 19.8 | — | — | — | — | — | 9.2 | 0.20 |
| cos010.log | **92.1** | 16.5 | — | — | — | — | — | 7.0 | 0.18 |
| layer_sens_baseline_qasper20 | 138.5 | 22.4 | 16.8 | 44.5 | 17.9 | 5143 | 3.49 | 10 | 0.16 |
| nonbatched_recall_numa0_gpu0_40k | 113.8 | 25.4 | 20.8 | 16.3 | 16.1 | 2315 | 5.93 | 10 | 0.22 |
| nonbatched_recall_numa0_gpu0_40k_rep2 | 112.8 | 25.4 | 20.8 | 16.3 | 15.7 | 2312 | 5.88 | 10 | 0.23 |
| nonbatched_recall_topk0_numa0_gpu0_40k | 154.1 | 40.4 | 35.1 | 39.9 | 17.7 | 4689 | 5.96 | 10 | 0.26 |
| nonbatched_topk0_numa0_cpu0_16c_gpu0_40k | 168.5 | 53.0 | 46.9 | 39.7 | 18.0 | 4674 | 5.34 | 10 | 0.31 |
| no_redundant_d2d_37k | 228.5 | 44.7 | 34.1 | 92.96 | 23.0 | 5024 | 2.92 | 10 | 0.20 |
| parallel_level_0_37k | 775.4 | 598.1 | — | — | — | — | — | 10 | 0.77 |
| parallel_level_1_37k | 226.1 | 55.9 | — | — | — | — | — | 10 | 0.25 |
| parallel_level_2_37k | 215.8 | 44.3 | — | — | — | — | — | 10 | 0.21 |
| promotion_0p01_qasper20 | 138.5 | 21.6 | 16.1 | 44.9 | 17.9 | 5189 | 3.59 | 10 | 0.16 |
| promotion_0p05_qasper20 | **129.3** | **13.8** | **9.1** | 44.8 | 18.0 | 5174 | 2.79 | 10 | 0.11 |
| promotion_hybrid_l17_qasper20 | 136.4 | 17.9 | 12.7 | 45.5 | 18.4 | 5260 | 3.32 | 10 | 0.13 |
| promotion_layer_cost_qasper8 | 143.4 | 21.3 | 15.7 | 47.2 | 18.8 | 5433 | 3.94 | 10 | 0.15 |
| promotion_profile_0p01_qasper2 | 155.2 | 25.0 | 19.0 | 48.3 | 20.1 | 5561 | 6.17 | 10 | 0.16 |
| promotion_profile_0p05_qasper2 | 136.0 | 14.4 | 9.6 | 50.0 | 17.6 | 5696 | 3.82 | 10 | 0.11 |
| promotion_profile_0p10_qasper2 | 132.2 | 11.8 | 7.0 | 47.5 | 17.5 | 5461 | 4.93 | 10 | 0.09 |
| system_breakdown_37k | 211.5 | 42.8 | 32.6 | 93.5 | 19.8 | 5020 | 2.52 | 10 | 0.20 |

## 2. 分组洞察

### 2.1 结构性事实
- **recall_wait 是最大单项**：同上下文（40k）下 recall_wait≈16-21ms（batched 链路优化后）或 40-45ms（topk0：无 topk 剪枝、页面更多），而 37k 早版本更多（93ms）。
- **recall_wait 与召回页数强相关**：pages/tok 2306（NUMA batched）→ 4692（topk0）→ 5020（37k 早版）→ 5433（promotion_layer_cost）。
- **native_query 与 pages 无关**：native_query 只由 promotion 层次结构决定（ratio_1 0.05 → 9.1ms；0.01 → 16.8ms），与 pages/tok 变化不联动——选页开销与搬运开销解耦。
- **native_query/call 全层均匀**（≈1.53-1.73ms），没有任何一层天然便宜；跨层复用 3 让每个 token 只做 10 次 DCI（30 个 recall 提交 = 10 anchor × 3 层组）。

### 2.2 已有优化被验证有效
- **batched recall（exp 05）**：TPOT 138.5 → 110.8ms（-20%），主要来自 recall_wait 44.5 → 20.9ms、gather 17.9 → 14.1ms、pages 5143 → 2306（topk 生效后页数减半）。**当前最优运行配置**。
- **promotion ratio_1=0.05（exp 07/08）**：native_query 16.8 → 9.1ms（-46%），TPOT 138.5 → 129.3ms（F1 不变）。ratio_1=0.10 更低（6.96ms）但 qasper2 小样本、需复核。
- **hybrid l17（exp 08）**：ratio_1=0.01 前 17 层 + 0.05 后 → native_query 12.7ms，F1 -0.79（可接受的轻微损失换 5.7ms）。
- **cos 阈值（cos005/010）**：dci_calls/tok 10 → 9.2/7.0，select 19.8/16.5ms，TPOT 99/92ms —— 但这两个日志 profile 不完整（无 wait/gather），也缺 F1，属**未完成实验**。

### 2.3 未完成/失效实验（低置信，勿当结论）
- **async_recall \***：recall_wait 2.17ms（ab_baseline）为全场最低，但该分支不在当前代码/脚本中，无法复现；且同事甚至把 wait 做没了（疑被记进 gather）→ **结构上限参考**，不可直接当路线。
- **baseline.log / parallel_level_0**：单线程无 batched，select 600ms，只证明"无优化=爆炸"。
- **longbench_baseline_ice64.log**：跑挂了（argparse 参数错误），无 profile。
- **layer_sens_skip\***：skip 机制因架构限制无法工作（decode_phase 条件断言失败），产出 None。

## 3. F1 全景（pred/llama-3.1/<name>/result.json，qasper 数据集）

| 实验 (qasper20) | F1 | TPOT | native_query | 备注 |
|---|---|---|---|---|
| baseline | **45.48** | 138.5 | 16.8 | 锚点 |
| promotion 0.01 | **45.48** | 138.5 | 16.1 | 与基线持平 |
| **promotion 0.05 全层** | **41.23 ⚠️** | 129.3 | 9.1 | **F1 掉 4.25，不可用** |
| hybrid l17 (r=0.05) | 44.69 | 136.4 | 12.7 | F1 -0.79，省 21% nq |
| **hybrid l17 (r=0.10)** | **44.74** | 135.4 | 11.75 | F1 几乎持平，nq 再省 8% |
| churn | 45.07 | — | — | |
| xtok (cross-token) | 17.37 💥 | — | — | 跨 token reuse 彻底失败 |

**关键修正**：此前 handoff 报告"promotion 0.05 全层 = F1 不变"是错误的——实际 **F1 从 45.48 掉到 41.23（-4.25）**。全层高 ratio 加速是亏的；**hybrid（前层保守 + 后层激进）才是唯一既省速度又不伤质量的配置**。

qasper8 小样本上 0.05/0.10 反而高于 0.01（31.23/31.33 vs 26.93），说明小样本方差极大，一切结论必须以 qasper20 为准。

## 4. Oracle 自适应早停的上限量化

`qasper_adaptive_oracle_stats.json`（11680 个查询采样）：

| 阈值 | stop_fraction（平均） | oracle_recall |
|---|---|---|
| 0.8 | 25.2% | 0.99994 |
| 0.9 | 25.9% | 1.00000 |
| 0.95 | 26.8% | 1.00000 |

- **只用 25% 的检索层次就能达到 0.9996 recall**（fixed_fraction 0.25 → recall 0.99961）
- 换算到页面搬运：pages/tok 5143 → ~1290（**-75%**），recall_wait 44.5 → ~11ms（**-33ms**）
- **但当前只有 oracle trace（纯统计），没有实际早停的 C++ 实现**——这是最大的未开发金矿
- 早停实现需改 `select.cu`（M-DCI 查询内核），风险中等

## 5. page-topks 重大发现

- **`--page-topks 0`（所有历史实验的传法）→ layer2topk=0 → num_neighbours = 全部页面 → pages/tok 5143（最大值）**
- **默认值（不传时）是 `b//2 = 32`** → 理论上 pages/tok 直接减半
- 约束检查：`k < b - ns - nw` → 32 < 64-2-2=60 ✓ 安全
- **正在验证：topk ∈ {16, 32, 48} 扫描** —— 若 F1 不掉，这就是零代码改动的最优性价比优化（wait -50%）

## 6. 核心结论

1. **recall_wait 是当前架构下唯一的结构性瓶颈**（~21-45ms/tok，占 TPOT 15-33%），由"同步提交 + 同流隐式等待"（L1719 `c2g_stream.synchronize()`）决定——**任何纯 DCI 选页优化都无法消除它**。
2. **已验证的优化栈**（按性价比排序）：
   - batched_recall（exp 05）：wait 44.5→20.9、pages 5143→2306（已合入，当前最优）
   - hybrid promotion ratio（exp 08）：nq 16.8→11.75，F1 持平（r=0.10 最佳）
   - **page-topks 减半**（验证中）：纯参数，预计 wait 再砍 50%
3. **未开发的最大机会**：oracle 早停 4 倍页面削减（需 C++ 改动）+ 跨 token 预取/双流异步搬运（async 分支曾有 2ms 证据）。
4. **纪律**：一切加速必须 F1≥45.48（qasper20 锚点）才成立；小样本（qasper8/2）结论不可信。

## 4. 待最小验证清单（cheapest first）
- [ ] 跑一次 `promotion_0p10` 全 20 样本（qasper20）+ F1 → 确认 0.10 是否真的省更多且不掉 F1。
- [ ] cos010 正式跑 qasper20 + F1（现在只 qasper2 未测质量）。
- [ ] 验证 wait 是否由 `rids.cpu()`/`nr.cpu()` 隐式同步引起（在 profile 里加计时——纯 Python 侧零风险）。
- [ ] 检查 `_dci_copy_to_buffer_batched` 是否真的能在计算流上双缓冲（C++ 内核支持度）。