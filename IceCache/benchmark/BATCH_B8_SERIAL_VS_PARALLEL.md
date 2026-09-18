# B=8 批量(并行) prefill 与串行 prefill 对照报告

提交：`142657a batch v3.0` ｜ 日期：2026-09-17 ｜ 机型：A100 80GB PCIe，Llama-3.1-8B-Instruct，fp16
环境：`OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE`

## 0. 结论摘要

1. **8 个 prompt 的输出是合理的**：8 条序列互不相同、无跨请求污染（同 prompt 在 B=8 与 B=2 下结果一致、`serial` 与 `native` 查询后端 8/8 逐 token 一致）、每请求的树/页/查询计数独立、页归属不重叠断言通过。
2. **但与串行 prefill 不是 bit 一致**：prefill 边界处 8/8 请求的 argmax 相同、logits 只差 **≈1 个 fp16 ULP**；然而在贪心解码下这 1 ULP 会被放大——116 个生成 token 里只有 **36.2%** 位置相同，2/8 槽位从第 1 个 token 就分叉。
3. **性质是数值噪声，不是逻辑缺陷**：两条路径**各自都 bit 级确定**；差异 ≈1 ULP；归因实验表明它住在**注意力/KV 产生的 hidden state**（Δh 0.5–2 ULP），与 lm_head 的 GEMM 形状无关。
4. **性能上没有优势，且更费显存**（本配置）：批量 prefill 一次前向 3.005 s vs 串行 8 次合计 ≈2.881 s；GPU 峰值 **21.79 GB vs 18.33 GB**（批量多 3.46 GB，主因是 `[8,1136,vocab]`≈2.33 GB 的瞬时 logits 与 padding 激活）。
5. **DCI 查询侧没有「一个线程负责一个 prompt」这回事**：native 是「固定 16 线程 × 256 个 (请求,head) 任务、动态领取」，线程数只由 `query_threads` 决定；**建树**才是逐请求的，且每请求各有独立 asyncio 线程（B=8 时 8 套，可并发）。

---

## 1. 被测对象与实验臂

prompt 由 probe 生成：第 i 个请求 `length = 1024 + 16*i`、`seed = 101 + i`（B=8 → 长度 1024…1136，8 个都不同）。**注意这是随机词袋文本**（30 词词表随机取词），因此下一 token 的 top1−top2 裕度极小（最紧 0.0156），是「贪心对数值差异最敏感」的输入。

| 臂 | batch | prefill 模式 | 查询后端 | 步数（main+retire+admit） |
|---|---|---|---|---|
| **A** | 8 | batched | serial | 8+4+4 |
| **B** | 8 | sequential | serial | 8+4+4 |
| **C** | 8 | batched | native（含候选 parity） | 8+4+4 |
| **D** | 2 | batched | serial | 8+4+4 |
| **E** | 1 | batched | serial | 8+2+2 |

A/B 只差 prefill 模式；A/C 只差查询后端；A/D 是「同 prompt、不同邻居数量」的隔离对照（slot 1 的 prompt 在三种设置下完全相同）。

---

## 2. 结果

### 2.1 资源与时间

| 字段 | A (B=8 batched) | B (B=8 seq) | C (B=8 nat) | D (B=2) | E (B=1) |
|---|---|---|---|---|---|
| `workspace_bytes` | 128 MiB | 128 MiB | 128 MiB | 32 MiB | 16 MiB |
| `auto gpu_pages` | 19232 | 19232 | 19232 | 6080 | 2080 |
| prefill 墙钟 | **3.005 s**（1 次前向） | **≈2.881 s**（8 次之和：0.581,0.318,…） | 3.044 s | 0.816 s | 0.518 s |
| `batch_query_seconds` | 4.484 s | 4.460 s | **4.829 s** | 0.972 s | 0.300 s |
| `decode_steps` | 16 | 16 | 16 | 16 | 8 |
| GPU 峰值 | **21.79 GB** | **18.33 GB** | 21.79 GB | 17.63 GB | 16.84 GB |
| `retire_gpu_pages_returned` | 512 | 512 | 512 | 512 | —（B=1 无 retire） |
| 线程数变化 | 18→16 | 18→16 | 18→16 | 6→4 | — |
| `native_scheduler` | — | — | engine{team 16, tasks **256**} | — | — |
| `raw_equal_elements` | — | — | **12288** | — | — |

要点：
- **批量 prefill 在本配置下没有墙钟优势**（3.005 vs 2.881 s）。padding 到 `[8,1136]` = 9088 行 vs 真实 8640 行（仅多 5% 计算），收益被「串行 8 次前向本身也不慢」抵消。
- **批量 prefill 更费显存**（+3.46 GB）。可剥离项：为取 8 个「最后一个真实 token」的 logits，批量路径会临时解除 lm_head 截断 → 实化 `[8,1136,128256]` ≈ **2.33 GB**。只对最后 8 行算 lm_head 即可省掉（未做）。
- `native` 批量查询在 B=8 下 `query_tasks=256`、`omp_team_size=16`；本测 `batch_query_seconds` 反而**高于** serial（4.829 vs 4.484）——候选搜索虽然批量化了，但候选回灌到 Python 侧仍要**逐请求**做 `diff_pages_by_head`，任务粒度小（256 个）导致调度开销占比上升。

### 2.2 正确性

| 判据 | 结果 |
|---|---|
| A vs B 逐槽位首次分叉位置 | `[None, 1, 2, None, 1, 0, None, 0]`（None=16 token 全同） |
| A vs B 相同位置比例 | **42/116 = 36.2%** |
| A vs C（serial vs native 查询后端） | **8/8 逐 token 完全一致** |
| `B[slot1]` vs `D[slot1]`（串行 B=8 vs 批量 B=2，同 prompt） | **完全一致** |
| `A[slot1]` vs `D[slot1]` | 首个分叉 = 位置 1 |
| 8 条序列是否互不相同 | 是（无重复；slot 0 被 admit 后重新开始，长度 4，其余 16） |
| 8 棵树是否都建起 | 是（`native_points_before[0][:8] = [960]×8`） |
| 每请求查询计数形状 | 8 × 32 层 |
| 页归属不重叠 | 断言通过（跨 8 请求 × 32 层无重复物理页） |

**解读**：分叉**不是**「邻居数量」造成的（slot 1 的 prompt 在 A/B/D 完全相同，B[1]==D[1]，只有 A[1] 不同），也**不是**查询后端造成的（A==C）。它只与「prefill 走批量还是逐请求」有关。

### 2.3 数值归因（B=8）

| 量 | 实测 |
|---|---|
| 批量 prefill 跑两次 | `max|Δ| = 0`（bit 确定） |
| 串行 prefill 跑两次 | `max|Δ| = 0`（bit 确定） |
| 批量 vs 串行，最后真实 token 的 **hidden state** | `max|Δ| = 0.016–0.0625`，而 `|h| ≈ 25–46` ⇒ **0.5–2 个 fp16 ULP** |
| 批量 vs 串行，最后真实 token 的 **logits** | `max|Δ| = 0.0156–0.0186`，而 logits 量级 ~17–18（ULP≈0.0166）⇒ **≈1 ULP** |
| prefill 边界 argmax（8 请求） | **8/8 一致** |
| 把 lm_head 的 GEMM 形状统一后重比 logits | 差异**不变**（比值 1.004） |

⇒ 残留差异**住在注意力/KV 路径**（batched ragged prefill 与单请求 prefill 的 fp16 归约顺序不同），**不是** lm_head 形状造成的。因此「只算最后 B 行」能省 2.33 GB 显存，但**消不掉**这 1 ULP。

### 2.4 DCI 树管理的并发模型（回答「一个线程负责八个 prompt」）

| 环节 | 实际执行者 | 合并 or 逐请求 |
|---|---|---|
| prefill 的 q/k/v 投影 | 主线程一次 GEMM 覆盖 `[8,1136]` | **合并** |
| prefill 的 KV 写入 | 主线程逐请求循环 | 逐请求 |
| prefill 的 ragged attention | **一次** `BatchPrefillWithPagedKVCacheWrapper.forward` 覆盖全部真实 token | **合并** |
| **建树/淘汰** | **每请求自己的 asyncio loop 线程 + 单 worker executor**（B=8 → 8 套，可并发；`_dci_future` 逐层 await 同步） | 逐请求（可并发） |
| decode 的查询（serial） | **主线程逐请求顺序** 8 次 `_DCI_query`（其内部还开 OpenMP `parallel_level=2`） | 逐请求 |
| decode 的查询（native） | **1 个固定 OpenMP team（16 线程）× 256 个 (请求,head) 任务**，`schedule(dynamic)` 动态领取 | 批量树搜索，**粒度 = head** |
| decode 的候选回灌（native） | 主线程逐请求顺序 `diff_pages_by_head` | 逐请求 |
| decode 的 recall / scatter / append | 主线程逐请求循环 | 逐请求 |
| decode 的注意力 | **一次** paged decode 覆盖全部 8 行 | **合并** |

**准确表述**：
- 「**一个线程管理一个 prompt 的所有层的对应 head**」——**不成立**。线程数与 B 无关（=`query_threads`）；native 下**同一线程会先后处理不同 prompt 的 head**；serial 下是**一个主线程串行遍历 8 个 prompt**。
- 真正「逐请求并行」的是**建树/淘汰**：每个请求一条独立的 asyncio 线程，B=8 时有 8 条，可同时建 8 棵树（各自只写自己的树）。
- 「decode 合并矩阵运算」——**部分成立**：投影、`build_attention_metadata`、最终 paged attention 是合并的；**DCI 搜索/回灌、recall/scatter、KV 写入仍是逐请求**。

**并发风险**：DCI 树本体没有锁。安全性完全靠「阶段分离（prefill 先全量建树 → 再进 batch decode）+ `decode_attention` 内先查后写 + `_dci_future.result()` 的 await」。若将来把 offload 写树挪到查询之前、或让活跃请求在 decode 期间重新 prefill，就会出现同树读-写竞争——**显式互斥未覆盖**。

---

## 3. 判定

- **功能性/逻辑性：通过。** 8 个 prompt 各自独立、结果自洽、无跨请求污染；`serial` 与 `native` 查询后端逐 token 一致；每请求的树、页、查询计数、页归属全部正确。
- **数值性：等价到 ≈1 fp16 ULP，但不是 bit 等价。** 因此**不能用「逐 token 相等」作为批量 prefill 的验收标准**；应采用：各路径自身确定性 + logit 级 ULP 一致 + prefill 边界 argmax 一致 + 逐请求记账一致。四项都成立。
- **性能性：本配置下无优势，且多占 ~3.5 GB 显存。** 批量 prefill 的价值不在这一组（短 prompt × 8），而在「一次前向省掉 7 次层间 kernel 启动」的更大 B / 更长 prompt 场景——需要用真实长度分布再测。

---

## 4. 建议（按优先级）

1. **修文档口径**：`BATCH_DECODE.md` 中「sequential 与 batched 产出完全相同的 `generated_token_ids`」已更正为「≈1 ULP 等价、token 序列可能分叉」。
2. **省掉 2.33 GB 瞬时张量**：批量 prefill 只对「最后真实 token」的 B 行算 lm_head（顺带去掉 ~0.0076 的额外数值偏移）。不改数值结论，只省显存与时间。
3. **若必须逐 token 复现**：只能让两条路径走同一个 kernel 形状（放弃批量化收益）或用确定性/更高精度路径——不建议。
4. **质量 A/B 的方法学**：fp16 下贪心输出对 1 ULP 敏感，任何「比生成文本」的评测都要按统计口径，或比 logits。
5. **补 DCI 树的读写互斥**（若将来要重叠 prefill/decode）。

---

## 5. 复现命令

```bash
cd /home/yx/IceCache/IceCache
export PYTHONPATH=/home/yx/IceCache/IceCache/source OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE
BIN=/home/yx/miniconda3/envs/icecache/bin/python

# A / B / C / D / E（见 §1 表格）
$BIN -u benchmark/batch_decode_probe.py --batch-size 8 --prefill-mode batched    --query-backend serial  --steps 8 --extra-steps 4 --post-admit-steps 4 --output /tmp/cmp_A.json
$BIN -u benchmark/batch_decode_probe.py --batch-size 8 --prefill-mode sequential --query-backend serial  --steps 8 --extra-steps 4 --post-admit-steps 4 --output /tmp/cmp_B.json
$BIN -u benchmark/batch_decode_probe.py --batch-size 8 --prefill-mode batched    --query-backend native --compare-native-raw --cpu-replay-repeats 0 --steps 8 --extra-steps 4 --post-admit-steps 4 --output /tmp/cmp_C.json
$BIN -u benchmark/batch_decode_probe.py --batch-size 2 --prefill-mode batched    --query-backend serial  --steps 8 --extra-steps 4 --post-admit-steps 4 --output /tmp/cmp_D.json
$BIN -u benchmark/batch_decode_probe.py --batch-size 1 --prefill-mode batched    --query-backend serial  --steps 8 --extra-steps 2 --post-admit-steps 2 --output /tmp/cmp_E.json
```

数值归因脚本：`/tmp/prefill_logit_cmp.py`（批量/串行各自确定性、logits 差）、`/tmp/diag_hs_lmhead.py`（hidden state 差与 lm_head 形状效应）。

---

## 6. 不确定项

- **prompt 是合成词袋**：下一 token 裕度被人为压小，36.2% 的重合率**不能外推**到自然文本；自然文本上重合率应显著更高（但仍非 bit 等价）。
- 只测了 1 组 seed/长度组合、每个请求只取最后真实 token 做数值比较；未做多 seed 重复统计。
- `retire_gpu_pages_returned = 512` 是**物理页池 free-list 的增量**（= 该请求常驻页 16/层 × 32 层），不是该请求的 prompt 页数（≈64/层）；口径需按下标含义读。
- 未测 B>8、未测 prefill/decode 交错（continuous batching）。
