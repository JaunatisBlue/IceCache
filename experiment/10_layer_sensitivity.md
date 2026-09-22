# Experiment 10 — Layer-sensitivity: which layers need DCI at all?

> 背景：交接结论「不做全局跨 token reuse，保留跨层 reuse（3）」。03/04 的 churn 数据
> 已证明相邻 token 页选择不稳定（full-k overlap 中位 66.7%，>=90% 仅 0.79%）。
> 但 DCI 开销中 **native query is spread across layers**（layer_cost 日志显示各层
> native_query/call ≈ 1.53-1.62ms 几乎打平）；recall_wait（~45ms/tok）才是主闸门。
> 因此「继续做」落点 = 层敏感度白名单：**哪些层根本不需要 DCI，也仍然可以无损？**

## Step 0 — Baseline 复现（当前代码 + 环境）

- 配置：Qasper20 固定子集（indices 见 SUBSET_INDICES），Llama-3.1-8B-Instruct，
  page 64/16、sink/win 2/2、reuse 3、ratio_1=0.01、ratio_2=0.2、parallel level 2、
  FP16 recall、batch-layer-recall off、cross-token off。与 exp 08 完全一致。
- 结果（2026-09-11 20:38 实测）：
  - **Qasper F1 = 45.48** ✅ 与 exp 08 baseline 45.48 完全一致 → 锚点稳了。
  - TPOT 138.48 ms/tok；native query 16.79 ms/tok；recall_wait 44.46 ms/tok；
    dci_select 22.40 ms/tok。decode total 51.5s / 372 measured steps。
- 结论：当前代码、环境、口径可重复。任何新实验对比 45.48。

## Step 1 — 层敏感度 oracle（计划）

目标：找出「去掉该层 DCI 查询/保留页面集合不变」时，对 Qasper 输出质量影响最小的层。
- 方法：对每个 anchor layer（0, 3, 6, ..., 29，共 10 个）各跑一次 Qasper20，该层
  跳过 DCI（直接复用上一步的 page 集合），其余层正常。比对 F1 掉落。
- 预期：如果某些后层（如 26/29）对页面集合不敏感，则可安全跳过其 DCI → 每 token 少
  1-2 次 native query（当前 10 次/层组 → 8 次），贡献一个纯算法、不撞 recall_wait
  的可发表增益。
- 风险控制：若跳过某层导致 F1 剧烈下降（> 1.0），则该层必须在白名单内。

## Step 2 — 白名单验证（计划）

把 Step 1 中「无损跳过」的层集合做成 env 控制（如 ICECACHE_SKIP_LAYERS），在
Qasper20 全量验证。预期 native query -20%、TPOT -1.3%（参考 exp 08 的 layerwise
收益），F1 保持在 45.48 ± 0.8 内。

## 2026-09-11 更新 — skip 实验暂缓（技术死因）

尝试给 infer_state.py 加 `ICECACHE_SKIP_DCI_LAYERS` env（被列出的 anchor 层强制复用
上一层、跳过 DCI+recall，走后层复用 offset 路径）做层敏感度筛查。**三处崩溃**：

1. `scatter_pages` C++ 需 4 个 tensor，`(None, None)` 直接 TypeError；
2. 用 `prev_eids`（跨 token 语义）作层间 eids 传 C++，形状/offset 不兼容 → segfault；
3. `_DCI_first_call`（prefill 建树）也调 `check_reuse`，skip 强制复用未建树层 →
   `dci_db[reuse_id]` 为 None → AttributeError。

加 `_decode_phase` / `_in_estimate` / 线程 id / `_DCI_first_call` 包装等多重保护均无法
稳定隔离：**`prefill_evict_extra_pages` 是异步 executor，与 decode 的 estimate 共享
状态，时序竞态导致 skip 标志误读**。

**结论**：layered "DCI skip" 在现架构下不安全。prefill 建树与 decode 查询都依赖
`check_reuse` 的稳定语义，跳过会破坏树存在性假设。要做到层敏感度必须改
`_DCI_first_call` 的建树决策（skip 层仍建树但 decode 不用其查询），需更深的架构改动，
暂缓。当前代码已回滚到 skip 默认空集 = base 等价（base 8 样本验证跑通）。
**可行替代**：用 codex 既有的 `ICECACHE_PROMOTION_FAST_START_LAYER`（exp 08 已验证
安全的层间 promotion 梯度）做「粗树/细树」的层间敏感度近似，推荐优先。

## Artifacts

- Runner: run_10_layer_sensitivity_step0_baseline.sh（Step0 复现）
- Runner: run_10_layer_sensitivity.sh（Step1 筛查，已因 skip 机制不可行而暂停使用）
- Log: logs/dci_opt/layer_sens_baseline_qasper20.log
- Predictions: pred/llama-3.1/layer_sens_baseline_qasper20/
- Result（Step0 baseline）: **45.48**