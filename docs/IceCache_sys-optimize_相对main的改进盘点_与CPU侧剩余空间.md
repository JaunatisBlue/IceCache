# `sys-optimize` 相对 `main` 的改进盘点 + CPU 侧剩余空间

> 2026-09-15。merge-base = `6a7ecfd`。本分支 18 个提交；`main` 只多 1 个提交（`72afda9 更新.gitignore`），**内容上本分支是超集**。
>
> **前提（决定所有数字怎么读）**：本机现在只剩 `00000000:3B:00.0` 一张卡，`nvidia-smi` 报
> `pcie.link.gen.current=3, width.current=1, width.max=16` ⇒ **PCIe 3.0 ×1（≈0.82 GB/s）**。
> 这正是交接文档点名的瓶颈卡；健康卡 `0000:af:00.0`（×16）已掉卡。
> 因此凡是标"GPU1"的历史收益，现在**无法在本机复现**；本机所有数字都在 ×1 语境下。

---

## 1. 分支结构

| | |
|---|---|
| 提交数 | 18（`main..sys-optimize`） |
| 主要代码改动 | `infer_state.py` **committed +1066 行**，另有**未提交 +408/−3**（本轮 profiling + 地址向量化） |
| 其他代码文件 | `utils.py`(+17/−2)、`adapter/modeling.py`(修正)、`benchmark/longbench_pred.py`(+70)、`benchmark/passkey_pred.py`(+29) |
| C 扩展 | `patches/mdci-fp16-recall.patch`、`patches/mdci-batched-layer-recall.patch`、`scripts/install_mdci_fp16_recall.sh` |
| 文档 | `docs/` 下 20+ 篇实验记录 + `AGENT.md` |
| 开关数量 | merge-base 的 `infer_state.py` **一个 `ICECACHE_*` 开关都没有**；现在 **18 个** |

---

## 2. A 类 · 默认生效的真实改进

| # | 项 | 开关 / 默认 | 改了什么 | 实测 | 条件 |
|---|---|---|---|---|---|
| 1 | **FP16 recall 暂存** | `ICECACHE_FP16_RECALL`（代码默认 0，**所有 run 脚本设 1**） | CPU DCI 索引仍 FP32；选中页在 pinned host memory 里转 FP16 再 H2D，**PCIe 字节减半**。需 M-DCI `dtype=2` 补丁；启动时探测能力、与 NumPy 单元对拍 | `recall_wait` **2.17 → 0.29 ms/tok** | GPU1（×16） |
| 2 | **native GQA page merge** | 同补丁，符号存在即用（否则回退 NumPy） | 把每 token ~80 次 NumPy/Python 去重换成 native 有序合并 | dedup **4.0 → 0.28**；postprocess **8–9 → 4.22 ms/tok** | GPU1 |
| 3 | **`utils.first_k_unique` 重写** | 无开关（默认生效） | 原来 `np.unique` **全排序**再取前 k；改成"遇 k 即停"的线性去重 | 未单独计时；纯 CPU、无质量风险 | 任意 |
| 4 | **`ICECACHE_FAST_ADDR`** | 默认 **1** | recall 侧源地址列表从"逐页 `.item()` 循环"改为 NumPy 视图 | `recall_gather` **约减半** | 任意 |
| 5 | **`ICECACHE_DCI_PARALLEL_LEVEL`** | 默认 **2**（merge-base 里根本没有这个旋钮，是硬编码） | 把 native DCI 并行度变成可调；`level 0 → 2` 实测 DCI 选择时间 **−92.6%**、TPOT **−72.2%** | 见 `experiment/04_dci_parallel_level.md` | 任意 |
| 6 | **`ICECACHE_VEC_ADDR`**（本轮，**未提交**） | 默认 **1**（2026-09-14 翻） | decode 增量地址准备向量化：用 `c2p` 一次取物理页号 + NumPy 整数运算，替掉逐 leaf `data_ptr()` 循环 | `index_address_prepare` **−58.2%（36k）/ −71.4%（20 样本 qasper）**；per-leaf **−77% / −86%**；端到端 ≈ **−0.5%** | 任意 |

> 1 + 2 + OMP 核绑定在 GPU1/passkey 40k 上叠加到 **TPOT 115.21 → 105.60（−8.3%），F1 45.48 vs 45.07**。

### 2.5 优化效果总表（全部有实验支撑的量）

> 证据三档：**① 本机实测**（GPU0 ×1，可复现）/ **② GPU1 历史**（×16，卡已掉，本机不可复现）/ **③ 无效或反向**（同样是实验结论，负结果也要记）。

**① 本机实测（GPU0 ×1）**

| 手段 | 测点 | 效果 | 条件 |
|---|---|---|---|
| `ICECACHE_VEC_ADDR` | `index_address_prepare` | **1.710 → 0.525 ms/tok（−71.5%）** | 3 样本 qasper |
| `ICECACHE_VEC_ADDR` | `index_address_prepare` | **1.979 → 0.566（−71.4%）** | 20 样本 qasper |
| `ICECACHE_VEC_ADDR` | `index_address_prepare` | **1.277 → 0.534（−58.2%）** | 36k passkey |
| `ICECACHE_VEC_ADDR` | per-leaf `data_ptr()` 循环 | **−86.5% / −86.1% / −77.2%** | 上述三种工况 |
| `ICECACHE_VEC_ADDR` | `index_update` 整条 | 4.304→3.224（−25%）/ 4.124→2.652（−35.7%）/ 4.013→3.038（−24.3%） | 同上 |
| `ICECACHE_VEC_ADDR` | 质量 | qasper 20 样本 F1 **45.44 → 45.48**；36k passkey accuracy **1.0 → 1.0**（4 跑） | — |
| `ICECACHE_VEC_ADDR` | 端到端 TPOT | **不可分辨**（预期 −0.5%，低于单跑 ±1.5% 噪声底） | — |
| `ICECACHE_DOUBLE_BUFFER` | TPOT **149.9 → 121.9（−18.7%）**；`recall_wait` **48.65 → 0.0135**；F1 46.43 持平 | 2 样本（默认关；收益全来自隐藏 DMA 等待，健链路应≈0） | ×1 |
| （测量资产）首个 session 首跑 | 比后续慢 **+6~8%** | 36k / 20 样本两轮均观测到 | ×1 |

**② GPU1 历史（A100 ×16，40,060-token passkey，64 gen / 8 warmup）**

| 手段 | 效果 | 质量 |
|---|---|---|
| **组合（32 物理核 + FP16 recall + native GQA merge）** | TPOT **115.21 → 105.60 ms（−8.3%）**，DCI 41.41→34.12 | 20 样本 qasper F1 **45.48 vs 45.07** ✔ |
| 32 物理核绑定（Q-head 数 = 物理核数） | TPOT 115.21 → 111.40（−3.3%） | — |
| **FP16 recall** | `recall_wait` **2.17 → 0.29 ms/tok（−87%）**；TPOT 单独 −1.5% | 启动能力探测 + 与 NumPy 逐位对拍 |
| **native GQA page merge** | dedup **4.0 → 0.28**；postprocess **8–9 → 4.22**；TPOT 109.20→105.60（−3.3%） | 比例 1/2/4/8 随机等价测试通过 |
| `DCI_PARALLEL_LEVEL` 0→2 | DCI 选择 **598.12 → 44.26（−92.6%）**，TPOT **775.37 → 215.75（−72.2%）** | passkey 正确 |
| **promotion 0.01→0.05（全局）** | native query **−43.5%**（exp08 20 样本）/−50%（exp07）；TPOT −6.6% | **F1 −4.25 ✘** |
| promotion 0.01→0.10（全局） | native query **−63.3%**，profiled TPOT −14.8% | 未过质量门 ✘ |
| **promotion 层间梯度（早 0.01 / 晚 0.05）** | native query **−21.3%** | F1 **−0.79**（TPOT 仅 −1.5%，因 recall_wait ~45 ms 主导） |
| batched cross-layer gather | `page_topks=32`：TPOT **113.28 → 110.80（−2.2%）**，提交数 30→10，gather 16.09→14.09 | pass ✔（`topks=0` 反向，见 ③） |
| NUMA：staging/index 内存放 GPU 本地节点 | 早期隔离测试 gather −2.4 ms/tok | 条件不可比，仅记为线索 |

**③ 无效或反向（同样是实验结论）**

| 手段 | 结果 |
|---|---|
| async recall ring | 115.21 → **129.72（+12.6%）** |
| next-layer prefetch | **125.88（+9.3%）** |
| DCI visit ratio 0.50 / 0.25 | 118.22（+2.6%）/ 113.26（−1.7%）；**`num_to_visit` 杠杆 0×** |
| contiguous native scratch | 115.30（±0） |
| AVX2 dual accumulator（改点积） | **118.72（+3.0%）** |
| 16 物理核（而非 32） | **126.68（+10.0%）** |
| reuse 6 层 | TPOT **98.20（−14.8%）** 但 F1 **41.30 vs 45.07 → 不可用** |
| 纯 NUMA node0 + 16 局部核 | **168.52（+46%）**，native query 46.92 |
| batched gather @ `page_topks=0` | 154.13 → **155.62（+1.0%，更慢）**；gather 只降 0.62，recall_wait 涨 4.85 |
| `valid_cache` 去重 | −0.16%（噪声内） |
| 删掉 recall `synchronize()` | **2× 慢 + KV 数据损坏**（sync 是单缓冲下的正确性保证） |
| 语义簇重排 / 合并 gather | 健链路上传输近乎免费 ⇒ **天花板≈0** |
| `num_to_visit` / `prop_to_visit` 调参 | **0×** |

**两个从表里读得出来的结构性事实**：

1. **CPU 侧各项的传导比不一样。** `recall_wait` 之外的段里，地址准备/`recall_gather` 位于 H2D 提交之前，属严格串行 ⇒ 省 1 ms 就少 1 ms TPOT（本轮实测：address prepare −0.74 ms/tok）。而 `native_query` 可与异步 H2D 部分重叠：exp08 里 native query 降 3.43 ms，TPOT 只降 2.08 ms ⇒ **传导比 ≈60%**。**按 21.3% 换算到 36k：native_query 36.27 → −7.73 ms，TPOT 约 −4.6 ms ≈ −3%。**
2. **×1 卡把很多收益压平了。** 同一个 promotion 层间梯度在 GPU1 上只换到 TPOT −1.5%（因为当时 recall_wait 已 45 ms 主导）；本机 36k 的 `recall_wait` 是 35.4 ms（24.1%），结构类似 ⇒ **不要把 native-query 的百分比直接当 TPOT 收益**。

---

## 3. B 类 · 已实现、但默认关闭（有明确理由）

| # | 项 | 开关 / 默认 | 实测与结论 |
|---|---|---|---|
| 7 | **双缓冲 + per-layer 槽位簿** | `ICECACHE_DOUBLE_BUFFER` / **0** | ×1 卡上 TPOT **149.9 → 121.9（−18.7%）**、F1 持平 46.43。原理是把 host 侧 `c2g_stream.synchronize()` 换成 `wait_event`。⚠️ 收益全部来自**隐藏 DMA 等待**，健链路应≈0；且第一版有槽位竞态（已修） |
| 8 | **batched cross-layer recall** | `ICECACHE_BATCH_LAYER_RECALL` / **0** | 3 层 reuse group 的 distinct 页合成一次 native OpenMP + 一次更大 H2D。`page_topks=0` 满载下**反向变慢 ~1%**（丢了逐层 copy/compute 重叠）；`page_topks=32` 下 **+2.2%** |
| 9 | **跨 token DCI 复用** | `ICECACHE_CROSS_TOKEN_DCI` / **0** | 余弦门控复用上一 token 的选择。qasper 上崩到 **17.37**；且已实测相邻 token 全 top-k 重叠仅 64.54%、完全相同 0.00% ⇒ 反证 FreeKV 前提 |
| 10 | **promotion 层间梯度** | `ICECACHE_PROMOTION_FAST_START_LAYER` / `_FAST_RATIO`，默认 `-1`（关） | 全局 0.05 掉 4.25 F1；层间方案 native query **−21.3%**、只掉 0.79 F1。**目前唯一的树侧有效算法旋钮** |

**promotion 这个点的精确语义**（`_prepare_prefill`，`infer_state.py:800-805`）：在 **prefill 阶段为每个 anchor 层**（`check_reuse(i)==0`）构造 DCI 树时决定 `promotion_prob` —— `i < start_layer` 用 `ratio_1`（默认 0.01），`i >= start_layer` 用 `_FAST_RATIO`（如 0.05）；`promotion_prob_subseq`（=`ratio_2`=0.2，控制更深层）不变。**建树时固定，整个请求内不再变化**，decode 期的增量插入不改它。reuse 层共用其 anchor 的树。

它调的是"点向上一级晋升的概率"：概率越高 → 上层节点越多 → **叶子更多、划分更细**。exp07 微基准（4096 点 / dim 128 / 60 neighbours / parallel 2）：

| promotion | 层数 | 叶子 | N/leaves | 中位查询 ms |
|---:|---:|---:|---:|---:|
| 0.0025 | 3 | 264 | 15.52 | 0.72–0.74 |
| 0.01 | 4 | 283 | 14.47 | 0.92–0.94 |
| 0.02 | 4 | 305 | 13.43 | 0.90–0.91 |
| 0.05 | 5 | 388 | 10.56 | 0.52–0.54 |
| 0.10 | 5 | 503 | 8.14 | 0.36–0.37 |

**反直觉点**：promotion 上去并不是"造出更大更少的叶子来避免叶子内全排序"，而是**造出更多更细的叶子、让局部排序变小**，查询反而快一倍。⇒ "全排序次数"不是正确的优化目标。这也是它区别于 `num_to_visit`（0×）的地方。

**⚠️ 但"promotion 影响质量"的方向，仓库内部两个实验是矛盾的** —— 这一点必须知道，否则会被单个实验带偏：

| 实验 | 子集 | promotion 0.01 | promotion 0.05 | 方向 |
|---|---|---:|---:|---|
| exp07 | Qasper **8 样本**（长度分层） | F1 26.93 | F1 **31.23** | **更高更好** |
| exp08 | Qasper **20 样本**（长度分层） | F1 45.48 | F1 **41.23** | 更差（−4.25） |

⇒ **质量轴的方向未确立**：真实效应量小于子集噪声。可确定的只有两件事：
1. **速度轴单调可靠**（0.0025→0.10：叶子 264→503、中位查询 0.73→0.37 ms；GPU1 上 native query −43.5%、TPOT −6.6%）。
2. exp08 的 −4.25 **基本来自一个样本的措辞漂移**：某 yes/no 样本 `Yes.`（F1 100）→ `Yes, ...`（语义正确，F1 11.11），单样本值 88.89/20 ≈ **4.4 分**，比总跌幅 4.25 还大 ⇒ 其余样本合计反而略回升。**不是 KV 损坏，是 metric 敏感**。
3. 旁证：同一批 promotion 设置**在 passkey 上全部 pass**（exp04/05 的 Passkey 列全 pass）——passkey 是 exact-match、答案短，措辞漂移无从发生。⇒ "质量是否受损"**高度依赖指标**。

**因此"全局 0.05 不可用"的正确理由是"风险/收益"，不是"promotion 高必然更差"**：效应方向在 8/20 样本间就翻转（不可控）；promotion 在 **prefill 建树时固定、全程不变**（无自适应补偿，一旦分错页整条请求都错）；而速度收益只有 6.6%（GPU1）/≈3%（本机 36k）。层间梯度把这个风险压低（暴露面减半、只让晚层承担），**但 −0.79 不是 0** ⇒ 仍需过质量门，且样本量要远大于 20。
| 11 | 其他旋钮 | `ICECACHE_SKIP_DCI_LAYERS`、`ICECACHE_DCI_PROP_TO_VISIT` | 层跳过架构不可行（三处崩溃+竞态）；`prop_to_visit` 实测 0×（`max()` 吞掉） |

---

## 4. C 类 · 测量资产（不是"改进"，但是这条线最耐用的产出）

| 项 | 开关 / 默认 | 价值 |
|---|---|---|
| decode 侧成本归属 profile（`DCI_PROFILE`） | `profile_dci` | 给出 `recall_wait / native_query / recall_gather / page_metadata / query_*` 的分段口径 —— **本盘的 CPU 清单就靠它** |
| 五段计时 + 地址 dump | `ICECACHE_DIAG` / 0 | `addr_prep / copy_to_buffer / H2D / cast / wait_residual`，附选页 leaf id + 地址 |
| churn / adaptive 追踪 | `ICECACHE_TRACE_DCI_CHURN`、`_ADAPTIVE` / 0 | 相邻 token 选页重叠、early-stop 触发率 |
| M-DCI C 层插桩 | 独立构建路径 | 出口计数/计时/分层（叶子 78.6 点 vs 请求 60，e2 占 18% 且 98% 在 level 0） |
| 微基准 | `roundtrip_latency.py`、`h2d_calibrate.py` | 往返 13 µs、PCIe 实测 0.82 GB/s |
| 本轮新增 | `ICECACHE_PROFILE_CALL_DUMP`、`ICECACHE_PROFILE_STEP_SAMPLES`、`ICECACHE_ADDR_EQUIV_CHECK` + `docs/{addr_opt,probe}/` 22 个脚本 | per-call 明细、逐步延迟分布、地址逐元素等价护栏 |

---

## 5. D 类 · 被证伪 / 回退（负结果同样是结论）

| 项 | 结论 |
|---|---|
| 直接删 recall sync | **灾难**：2× 慢 + 数据损坏。sync 是单缓冲下的正确性保证，不是冗余等待 |
| `valid_cache` 去重 | −0.16%，无效（成本在目标 stride 写） |
| 语义簇重排 / 合并 gather | 健链路上传输近乎免费 ⇒ 天花板≈0，**方向放弃** |
| `num_to_visit` 作为杠杆 | **0×**（query 侧被 `max()` 吞，construction 侧被 `min(i,…,num_points)` 夹到批量大小） |
| "同 head 相邻 leaf 地址连续" | 实测推翻：规则 stride **131072 B** = `cpu_n_bytes_per_page` |
| qasper 线上的绝对传输数字 | ×1 卡伪影，需重测或明确标注 |

---

## 6. E 类 · 工程卫生

- `adapter/modeling.py`：`enable_icecache` 现在回填 `self._icecache_infer_state`（否则 `get_profile_stats` 拿不到 state）；补文件尾换行。
- `.gitattributes`（LF 归一）、`.gitignore` 扩充（`experiment/`、pred 产物、日志、passkey.jsonl）。
- 清掉误入库的 pred 产物与 stray log；一次性脚本归 `experiment/archive/`。
- `AGENT.md`（338 行）—— 仓库级 agent 约定。

---

## 7. CPU 侧成本清单（36k passkey，GPU0 ×1，vec=1，TPOT = **147.29 ms/token**）

| 阶段 | ms/token | % TPOT | 性质 |
|---|---:|---:|---|
| **dci_select 链（合计）** | **43.364** | **29.4%** | CPU |
| └ `native_query`（DCI C 内核） | 36.274 | 24.6% | CPU，**最大单项** |
| └ `query_postprocess` | 5.398 | 3.7% | CPU |
| └ `query_diff` | 2.270 | 1.5% | CPU |
| └ `query_mapping` | 1.186 | 0.8% | CPU |
| └ `query_d2h` | 0.592 | 0.4% | CPU/小传输 |
| └ `query_dedup` | 0.338 | 0.2% | CPU |
| `recall_gather` | 11.117 | 7.5% | CPU（地址准备 + `copy_to_buffer`） |
| `page_metadata` | 6.797 | 4.6% | CPU |
| `index_update` | 3.062 | 2.1% | CPU（本轮已把地址段从 1.277 压到 0.500） |
| **`recall_wait`** | **35.449** | **24.1%** | **PCIe ×1 等待，不是 CPU** |
| 其余（GPU attention / scatter / 框架胶水） | ≈47 | ≈32% | GPU |

**CPU 合计 ≈ 64 ms/token ≈ 44% 的 TPOT。** `dci_share = 0.2944`（DCI 相关占 29.4%）。

> 注意：`dci_select` 是父口径，其子项之和（46.06）略大于它，说明子项有轻微重叠/嵌套；上表按"子项明细 + 父项合计"并列给出，不要相加。
>
> **这修正了一直被引用的"CPU 只值 3%"**：那个数字只覆盖 **decode 增量更新**那一片（index_update）。真正的 CPU 侧是它的 **20 倍**。

---

## 8. CPU 侧剩余空间（按"证据强度 × 收益"排序）

| # | 候选 | 预期收益（36k 换算） | 证据强度 | 备注 |
|---|---|---|---|---|
| 1 | **把 promotion 层间梯度打开并过质量门** | native query **−21.3%** ⇒ 36k 上 −7.7 ms/tok；按 **60% 传导比** ⇒ **≈ −4.6 ms/tok ≈ −3% TPOT** | **强**：exp08 已实测（native query −21.3%、代价 −0.79 F1）；**代码里旋钮已存在**（`ICECACHE_PROMOTION_FAST_START_LAYER`/`_FAST_RATIO`），默认关 | 零新代码。只需在 qasper 20 样本 + 36k passkey 上各跑一次质量门 |
| 2 | **量 `page_metadata` 6.8 ms/tok（TPOT 4.6%）** | 未知（先把 6.8 拆开） | **弱-中**：只有一条旧结论"成本在目标 stride 写"，**从未尝试优化** | 与 `index_update` 同级，且完全 CPU、不受链路影响 |
| 3 | **`recall_gather` 11.1 ms/tok（7.5%）** | 已减半；剩余主要是真实 memcpy | 中：FASTADDR 已吃掉 Python 部分 | 36k 上约 5500 页/token，要看能否**少传页**（改变 topk 语义，需质量实验） |
| 4 | **`query_postprocess` 5.4 ms/tok（3.7%）** | 已由 native merge 从 8-9 降到 4.2（GPU1 口径） | 中 | 继续压需先区分剩余部分是 reshape 还是 `ascontiguousarray` |
| 5 | **`index_update` 3.1 ms/tok（2.1%）** | 本轮到 2.1 → 已拿 0.74 | 强但收益小 | 剩余主体是 `native insert` 0.35 + `page alloc` 0.30 |
| 6 | **零改码：OMP/NUMA 绑定** | 未知 | **必测**：机器是 2×Xeon 5218 / 2 NUMA 节点，GPU 在 node0，**当前所有运行都没有 `numactl`/`taskset`/`OMP_PROC_BIND`** | README 已给出 `OMP_PROC_BIND=spread OMP_PLACES=cores` 建议，但**从未在本机验证**。旧笔记里"纯 NUMA 绑定 168.52（更差）"是传输受限时代的结论，需在 CPU 主导的现状下重测 |

**阻塞项（不在我们手里）**：把 GPU 换回 ×16 槽位。在 ×1 上，`recall_wait` 24.1% 的链路等待会把 CPU 侧的端到端可见度压制（本轮实测：CPU 省 0.74 ms/token → TPOT 变化 <1%，低于噪声）。

**传导比不是统一的**（见 §2.5 末）：H2D 提交**之前**的段（地址准备、`recall_gather`）严格串行 ⇒ **1:1**；`native_query` 可与异步 H2D 部分重叠 ⇒ 实测约 **60%**（exp08: native query −3.43 ms → TPOT −2.08 ms）。估算 CPU 侧收益时要按段用各自的传导比，不要把 native-query 的百分比直接当 TPOT 收益。

---

## 9. 一句话

**这条分支已经把"传输侧"做透了并证伪了大半（健链路上天花板≈0），把"测量"做成了资产；CPU 侧只动了 `native_query` 的一部分和 `index_update` 的地址段 —— 还剩 ≈64 ms/token 的 CPU 在桌面上，其中最大的一块（`native_query` 24.6%）已经有一个实测有效但默认关闭的旋钮。**
