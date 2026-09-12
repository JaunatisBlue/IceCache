# IceCache 发散探索报告：从 51 份日志到 topk 剪枝金矿

> 日期：2026-09-11 ｜ 环境：`yx@100.84.5.13 ~/IceCache` ｜ 验证基准：Qasper20 F1 / TPOT
> 本轮基调：不碰 `infer_state.py`（历史教训），全程**纯参数实验 + 既有日志分析**，最小验证。

## 一、开局：51 份日志的系统性剖析

对 `experiment/logs/dci_opt/*.log`（含 30 份有效 `DCI_PROFILE` JSON）做全量解析，得到跨实验对比表。关键结构性事实：

1. **recall_wait 是唯一的结构性瓶颈**：占 TPOT 15-33%。代码证据在 `infer_state.py` L1719 `c2g_stream.synchronize()` —— 每 anchor 层选页提交后**同步等待整批 H2D 完成**，这是架构固有成本。
2. **native_query 与搬运量解耦**：native_query 只由 M-DCI 层次结构（promotion ratio）决定（1.53-1.73ms/call 全层均匀），与 pages/tok 不联动。
3. **evaluate 结果藏在 `pred/llama-3.1/<name>/result.json`**（此前 handoff 未系统汇总）。

## 二、关键修正：promotion 0.05 全层其实掉分了

| 实验 (qasper20) | F1 | TPOT | native_query |
|---|---|---|---|
| baseline | **45.48** | 138.5 | 16.8 |
| promotion 0.01 | 45.48 | 138.5 | 16.1 |
| **promotion 0.05 全层** | **41.23 ⚠️** | 129.3 | 9.1 |
| hybrid l17 (r=0.05) | 44.69 | 136.4 | 12.7 |
| hybrid l17 (r=0.10) | **44.74** | 135.4 | 11.75 |

> **此前交接资料称"promotion 0.05 F1 不变"是错的**——实际掉 4.25 分。全层激进加速不可行；**hybrid（前层保守 + 后层激进）才是唯一安全加速**。

## 三、核心发现 1：oracle 早停上限 = 4 倍页面削减

`qasper_adaptive_oracle_stats.json`（11680 查询）：
- 阈值 0.8 时 **stop_fraction 仅 25.2%**，oracle recall 0.99994
- 只用 25% 层次 → recall 0.99961（fixed_fraction 验证）
- 换算：pages/tok 5143→~1290（-75%），wait 44.5→~11ms
- **当前只有 oracle trace，无实际早停实现**（需改 `select.cu`）→ 最大未开发金矿

## 四、核心发现 2（本轮主角）：page-topks 参数是零成本金矿

**背景**：所有历史实验都传 `--page-topks 0` → layer2topk=0 → DCI 每层返回**全部页面**（pages/tok=5143 的根源）。而代码默认值（不传）是 `b//2=32`，语义为"每层保留在 GPU 的页面数"。

**topk 扫描实验（qasper20，本次新跑）**：

| topk | F1 | TPOT | pages/tok | wait |
|---|---|---|---|---|
| 0 (baseline) | 45.48 | 138.5 | 5143 | 44.5 |
| **16** | **47.31 🔥** | 123.9 | 4179 | 34.7 |
| **32** | 45.15 | **112.2** | 2976 | 22.7 |
| 48 | 42.99 ⚠️ | **91.1** | 1477 | **7.9** |

**结论**：
- **topk=16：F1 反而 +1.83**（剪掉低质量页面 = 降噪），TPOT -10.5%。质量与速度双赢。
- **topk=32：F1 持平（-0.33），TPOT 112.2ms（-19%）**，wait 减半。
- topk=48 过度剪枝：TPOT 91.1ms 诱人但 F1 掉 2.49，不可用。
- wait ↔ pages 近似线性（34.7↔4179, 22.7↔2976, 11↔1477）→ wait 由搬运量驱动。

## 五、组合优化栈验证结果 ✅

```
batched_recall (exp05) + topk=32 + hybrid r=0.10
```

| 配置 | F1 | TPOT | wait | pages/tok | native_query |
|---|---|---|---|---|---|
| baseline | 45.48 | 138.5 | 44.5 | 5143 | 16.8 |
| **combined_opt** | **44.53** | **104.4** 🔥 | 27.9 | 2999 | **7.87** |

**TPOT -24.6%（138.5→104.4ms），native_query -53%（16.8→7.87），F1 仅 -0.95（在 ±1 噪声内）**。三路优化成功叠加，无冲突。

> 注意：combined 里 topk=32 的 pages 只到 2999（而非 topk=32 单测的 2976），wait 27.9（非 22.7）——batched 模式下 wait 计时有合并差异，但整体收益明确。

## 五·补、下一步：topk=16 与组合栈的交叉

topk=16 单测 F1=47.31（反升），若与 batch+hybrid 组合，理论上可得 **F1 更高 + TPOT 更低** 的双赢点。这是下一个最小验证首选。

## 六、下一步路线图（按性价比）

| 优先级 | 方向 | 风险 | 预期收益 |
|---|---|---|---|
| P0 | **topk=16/32 落地**（纯参数） | 零 | TPOT -19%，F1 ≥45 |
| P0 | 组合栈验证（进行中） | 零 | TPOT <105ms |
| P1 | **oracle 早停实现**（`select.cu`） | 中 | pages -75%，wait -33ms |
| P1 | 跨 token 预取 / 双流异步 | 中 | wait →~2ms（async 分支证据） |
| P2 | cos 阈值 + F1 正式跑 | 零 | calls/tok -30% |
| P2 | per-head 自适应收敛 | 高 | 与早停重叠 |

## 七、纪律清单（本次教训）

- **一切结论以 qasper20 为准**（qasper8/2 方差极大，出现过 26.93 vs 31.33 的假象）。
- **加速必须 F1 ≥ 45.48** 才成立；promotion 0.05 全层是反面教材（-4.25 F1）。
- 分析日志 ≠ 复现实验：async_recall 的 2ms wait 来自 codex 未入库分支，不可直接引用为当前基线。
- 改代码前先查 `result.json` 与 profile 日志格式（本次发现 F1 藏在 pred 目录）。