# `docs/` 索引

IceCache `sys-optimize` 分支的文档与复现工具。19 篇分析文档 + 21 个脚本。
本文件说明**先读什么、哪些是现行结论、读数字前必须知道的口径**。

---

## 0. 读数字之前必读（口径）

1. **本机只有一张卡：`00000000:3B:00.0`，PCIe gen3 ×1（width 1 / max 16，实测 0.82 GB/s）。**
   健康卡 `0000:af:00.0`（×16）已掉卡。凡标 **"GPU 1"** 的历史收益（FP16 recall、native GQA merge、8.3% 组合等）**本机无法复现**。
   本机全部数字都在 ×1 语境下 ⇒ `recall_wait`（24%）是**链路伪影**，不是 CPU。
2. **两套量级别混**：
   - decode **增量 DCI 更新**（`index_update`）只值 **~3% TPOT**；
   - **整个 CPU 侧值 ~44% TPOT**（36k：`dci_select` 29.4% + `recall_gather` 7.5% + `page_metadata` 4.6% + `index_update` 2.1%）。
3. **测量协议**（血泪换来的，务必遵守）：
   - 每个 session **首跑比后续慢 6-8%** ⇒ 先跑一次丢弃用的 warm-up；
   - 重复数 **≥3（奇数更稳）**，臂按位置交错；
   - 用 `decode_step_latency` 的 mean/std/p50/p95 **+ 逐样本 `Decode latencies` 交叉核对**，先剔离群再算均值；
   - 始终检查"补丁不碰的阶段"是否同步变化（若是，判定为漂移）。
4. **DCI 选择状态跨进程不可复现**（同配置两次运行 72% 记录不同）。⇒ `ICECACHE_DIAG` 的 leaf/地址 dump **只在同进程内自洽有效**；跨进程的 page-id 对比不能当等价性判据。
5. **`N_GARBAGES` 切的是字符数不是 token 数**（≈3.75 chars/token）：`134775→36000`、`140000→37394`（"37k"）、`150000→40060`（"40k"）。

---

## 1. 现行结论（先读这两篇）

| 文档 | 内容 |
|---|---|
| **`IceCache_sys-optimize_相对main的改进盘点_与CPU侧剩余空间.md`** | 分支相对 `main` 的全部改进（A 默认生效 / B 默认关 / C 测量资产 / D 已证伪 / E 工程卫生）+ **优化效果总表（本机实测 / GPU1 历史 / 无效反向 三档）** + 36k 的 CPU 成本清单 + 剩余空间排序。**想了解"现状与下一步"读这篇。** |
| **`IceCache_最终CPU验证_四阶段_2026-09-15.md`** | CPU 地址向量化的四道门：① 地址等价性 ② 4×36k 阶段级 A/B ③ 全量 Qasper 200 条质量门 ④ 交错端到端 TPOT。含 §2.1"为什么 36k 档 TPOT 无分辨力、不必重测"的量化论证 + 复现命令。**要引用数字读这篇。** |

---

## 2. `sys-optimize` 这条线的过程记录（按时间序）

| 文档 | 内容 | 状态 |
|---|---|---|
| `DeepSeek_增量DCI地址路径优化交接.md` | 任务书：decode 增量 DCI 更新与 CPU 地址准备优化 | 输入（2026-09-14） |
| `DeepSeek_增量DCI地址路径优化_结果.md` | 结果、证据与交接。含 **§9 修订记录**（撤回 `num_to_visit` 归因、把 ×5.35 降级为观察、把 DCI 非确定性改为三假设） | **现行**（详细版） |
| `IceCache_CPU开销归属_2026-09-12.md` | 解码侧成本归属；**"闪 sync 是承重的"**等否定结论；五段计时拆分 | 历史（结论仍被引用） |
| `IceCache_CPU优化与测量功效_实验22-23.md` | CPU 侧优化与**测量功效**（多少重复才测得出来） | 历史 |
| `IceCache_诊断设计_计时拆分与地址可合并性.md` | 联合诊断的**设计**（待批准执行的原始方案） | 历史（已执行） |
| `IceCache_联合诊断结果_2026-09-12.md` | 上述设计的**结果**（计时拆分 + 地址可合并性判定） | 历史 |
| `IceCache_双缓冲尝试_2026-09-12.md` | 双缓冲 + event 定序：槽位竞态 bug 与修复（×1 卡上 −18.7%） | 历史（代码默认关） |
| `IceCache_最终CPU验证_四阶段_2026-09-15.md` | 见 §1 | 现行 |

---

## 3. DCI 算法线

| 文档 | 内容 |
|---|---|
| `IceCache_日志系统剖析_51logs.md` | 51 份实验日志的系统性剖析（`DCI_PROFILE` 横向对比） |
| `IceCache_发散探索报告_51logs与topk金矿.md` | 从 51 份日志里挖出的 `topk` 方向 |
| `earlystop_run15_16_summary.md` | early-stop 实验 15/16（**截断从未触发** ⇒ 该方向无效） |
| `batch_serving_dci_convergence.md` | batch / serving 下的 DCI 选择成本，与"跨 token / 跨层"优化线的合并 |

> `promotion_prob`（唯一被证实有效的树侧旋钮）的语义、标定表、以及**两个实验方向矛盾**这一关键事实，记在 `IceCache_sys-optimize_相对main的改进盘点_与CPU侧剩余空间.md` §3。

---

## 4. 工况 / 竞品 / 大局

| 文档 | 内容 |
|---|---|
| `IceCache_接手状态与下一步.md` | 交接盘点：**两张 GPU 混用的口径纠正**、三个突破、被证伪清单、下一步（后来部分已过期） |
| `IceCache_三条推荐方向_传输侧.md` | 传输侧三条推荐方向（已按代码二次核对；**健链路上天花板≈0**） |
| `IceCache_带宽受限探索_实验18-21.md` | 实验 18–21：prefetch / knob / 地址优化 / budget |
| `IceCache_创新点分析.md` | 哪些方向能撑起论文创新点（已过文献核查） |
| `competitor_reading_guide.md` | KV Cache 语义索引 / 检索开销的**竞品地图与引导阅读** |
| `batching_serving_direction.md` | 吞吐方向思考：batch 之下的语义检索 KV offload |

---

## 5. 工具与复现（这一节的都是**代码**，不是文档）

> 放在 `docs/` 下是因为 `experiment/` 被 `.gitignore` 覆盖（`git add -A` 会静默跳过），
> 而报告要引用这些脚本 —— 必须让新 clone 拿得到。

### `addr_opt/` —— runner（env 驱动）

| 脚本 | 用途 |
|---|---|
| `run_addr_opt_profile.sh` | **单跑入口**：`<run-name> <max-samples>`，env 见文件头。日志落 `experiment/logs/addr_opt/` |
| `run_final_cpu_tests.sh` | **四阶段一次性验收**（≈65 min），本次交付用的就是它 |
| `run_ab_matrix.sh` | 小样本 A/B 矩阵（正确性 diag + 2×2 计时） |
| `run_20sample_ab.sh` | 20 样本端到端 A/B + 自动出报告与 F1 |
| `run_passkey36k_ab.sh` | 36k passkey A/B（含 `N_GARBAGES` 标定表；两臂等长靠 `--num-tests` + `--profile-new-tokens`） |
| `run_control.sh` / `run_leafcontrol.sh` / `run_head_control.sh` | 三种"不可复现性"对照（同配置两次 / 3 次 leaf 数 / **未改动代码**两次） |
| `run_followup.sh` | full-run 状态等价 + per-call 曲线 |
| `run_null_test.sh` | 备用的零语义扰动对照（**未运行**，被 `run_control.sh` 取代） |

### `probe/` —— 分析与判读

| 脚本 | 用途 |
|---|---|
| `probe_addr_formula_equiv.py` | 地址公式**单元**等价测试（CPU-only） |
| `show_addr_equiv.py` | 摘要 `ICECACHE_ADDR_EQUIV_CHECK` 的结果 |
| `compare_addr_diag.py` / `find_divergence.py` | 两份 diag dump 对比 / 定位首次分叉（基址无关） |
| `analyze_ab_paired.py` | **逐样本配对** TPOT（消除样本构成混杂，必用） |
| `summarize_reps.py` | 多重复取均值 + within-arm sd |
| `report_ab_tpot.py` | 两臂并排表（TPOT mean/std/CV/p50/p95 + 各阶段） |
| `compare_call_records.py` / `analyze_tinsert.py` | per-call 记录跨运行对比 / `T_insert` 曲线 |
| `compare_preds.py` | 逐样本预测文本对比 |

### 常用命令

```bash
# 四阶段验收（约 65 分钟）
bash docs/addr_opt/run_final_cpu_tests.sh final

# 阶段级解析（ms/token、ms/boundary、ms/layer-update）
python docs/parse_index_profile.py A1=experiment/logs/addr_opt/ab_A1.log B1=experiment/logs/addr_opt/ab_B1.log

# 逐样本配对（也是判漂移的对照）
python docs/probe/analyze_ab_paired.py A=.../q20_A.log B=.../q20_B.log

# F1（在 IceCache/benchmark 下）
python longbench_eval.py --model llama-3.1 --name <run-name>
```

---

## 6. 命名约定 & 本次整理未做的事

- **三类前缀并存**：`IceCache_*.md`（中文分析，主流）、英文名（早期笔记，如 `earlystop_*`、`competitor_*`）、`DeepSeek_*`（agent 交接）。
  **本次未重命名历史文档**（避免打断既有引用）；如需统一前缀，建议单独提交并同步改引用。
- **本次只修了一处悬空引用**：结果文档里那条指向"脚本迁移前旧位置"的路径，已改为 `docs/parse_index_profile.py`。
- 其余 `docs/**/*.md` 里引用的路径已全部核对存在（审计脚本见提交说明）。
- 运行日志等**产物**仍在被 ignore 的 `experiment/logs/`，不入库；需要时可
  `cp -r experiment/logs/addr_opt docs/addr_opt/logs`。
