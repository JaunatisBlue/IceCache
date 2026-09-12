# Batch / Serving 下的 DCI 选择成本 —— 与「跨 token / 跨层」优化线的合并

> 交接：03_cross_token_dci + 04_qasper_dci_churn。结论：不发展全局跨 token reuse；保留论文现有跨层 reuse（n_reuse_layers=3）。
> 本文档 = 交接结论复核 + 与 codex 白天进展（exp 05/07/08）合并后的下一落点。

## 0. 结论

**同意交接判断：不做全局跨 token reuse。** 且有两个更强的理由——

1. **04 已证伪默认前提**：Qasper 相邻 token 相邻 head 的 DCI 页面集合，完整 top-k 重叠中位 66.7%、P90 81.7%、≥90% 只有 0.79%，没有任何"相邻 token 页选择稳定"的基础；03 的 Qasper 崩坏（43.2→17.37）是这一点的端到端确认。
2. **08 把「召回传输」推成了主闸门**：Qasper20 各配置 recall_wait ≈ 45ms/tok、占 TPOT 33%，而 native query ≈ 9-16ms/tok。即使跨 token reuse 让选择成本归零，**也动不了 recall_wait 这块最大的饼**——所以"继续做"不应是再做一轮 reuse。

## 1. 现在的真实瓶颈排序（Qasper20 实测，exp 08 + layer_cost 日志）

| 分量 | ms/tok | 占比 | 归属 |
|---|---:|---:|---|
| recall_wait（H2D 等待） | 47.2 | 33% | 传输/调度/流水线 |
| recall_gather（CPU 侧组装） | 18.8 | 13% | 传输 |
| native DCI query（M-DCI 树检索） | 15.7 | 11% | 树结构/并行 |
| query postprocess（diff/dedup/map） | 4.3 | 3% | 后处理 |
| index_update（树更新） | 3.9 | 3% | 树更新 |
| GPU decode | ~53-61 | ~37% | 模型 |

→ 三层诉求相互独立：**DCI 选择（算法，exp 05/07/08 在做）**、**recall 传输（系统，exp 05 batched-recall 已结论：保持 per-layer、默认关）**、**调度（serving 方向，尚未开始）**。

## 2. “继续做”的三个口（按性价比排序）

### 口 A：层间敏感度白名单（算法，成本低，先做）
- exp 08 已证明 layerwise hybrid（早层 0.01 / 晚层 0.05）几乎无损（-0.79 F1、-21.3% native query、-1.5% TPOT），但用**固定的层分组**。
- layer_cost 日志已给出逐层实际成本（native query/call 各层 ≈ 1.53-1.62ms，几乎打平）→ 成本维度对各层无差别，**敏感度才是分层的依据**。
- 落点：把 Qasper**每层去掉该层 DCI 的 oracle 质量测试**做成一次诊断跑（20 样本子集），得到"该层是不是质量贡献层"，然后给一个层分组白名单/黑名单控制，做第二次 20 样本验证。若能维持 45.07（子集）口径，且 native query 再降 ~20%（只查敏感层），就是一条**不撞 recall_wait 墙、纯算法可发表的增益**。

### 口 B：per-head 自适应收敛（算法，中期）
- 05 的 oracle 已表明：per-head 25% 访问上限时 set-recall 99.03%、90% 收敛准则下 100% 命中、平均名义 cap 25.87%。
- 但 06 澄清：名义 num_to_visit 是死参数，**真正要做的是在单次树遍历内部、按 head 的候选取向收敛提前停**——这会动 M-DCI C++ 内核，不是 Python 层能完成的。
- 定位：作为口 A 之后的下一个算法贡献，独立于 serving。

### 口 C：Batch/Serving 方向（系统，新方向，上一步文档已展开）
- 上一步已交付 `docs/batching_serving_direction.md`：batch 下先撞 CPU 选择墙（B≈2-3）、后撞 H2D 带宽墙（B≈8-16）、GPU 计算墙最远。
- 单请求优化（exp 05 的 8.3%）救不了 serving，因为 batch 把成本乘 B；核心叙事 = **每请求选择成本压 ~10×（接 06/07 叶子层退化）+ 跨请求前缀页共享 + 带宽预算自适应 k + 批级召回流水线调度**。
- 零成本第一步：① 串行混跑 B 请求验证吞吐饱和点；② 统计 SWE-bench/LongBench 轨迹前缀页重叠率给共享收益定上界。

## 3. 我建议的下一步顺序（不撞车 codex）

1. **今天**：口 A 的 oracle 敏感度诊断脚本 + 8 样本跑通（~30-40min），产出"逐层敏感度表"；
2. **明天**：口 A 验证 + 层分组 20 样本确认（+/-0.8 F1 内、native query -20%）；
3. 与 codex 确认主工作面后，再开口 C（serving）的零成本验证。

## 4. 交付物（本会话产出）

- 本文档：`docs/batch_serving_dci_convergence.md`
- 上一步已交付：`docs/batching_serving_direction.md`
- 下一步（口 A）：`experiment/run_10_layer_sensitivity.sh` + `experiment/10_layer_sensitivity.md`（视与 codex 协调而定）