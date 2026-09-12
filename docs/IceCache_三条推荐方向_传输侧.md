# 三条推荐方向（v3 · 已按代码二次核对 + 方法学收紧）

> v1（2026-09-12）被指出"把研究假设当结论"；v2 逐条核对并收紧；v3 又发现 v2 仍有两处硬错误，由 codex 指出、已在代码中复核确认。
> **v3 结论：方向已收敛正确，但实验方法仍需收紧——只做一项 GPU 联合诊断，不三线并跑。**
> 代码核对对象：`.icecache_src/infer_state.py`（本地副本）、`.icecache_src/mdci/src/py_dci.c`。

---

## 0. 关键事实（已代码核实，可直接引用）

### 0.1 地址布局：**不是"叶子连续"，是"规则 stride"**

```python
# infer_state.py
cpu_n_bytes_per_page = 2 * page_size * n_kv_heads * head_dim * itemsize   # L142-143（含 n_kv_heads）
stride = cpu_n_bytes_per_page                                            # L923
offset = page_size * head_dim * itemsize                                 # L957（不含 n_kv_heads）
addr(i, j) = _base + i * offset + j * stride                             # L960-961
```

因为 `stride`（完整多头页）**含 `n_kv_heads`**，而 `offset`（单 head 一页）**不含**，所以：

> **同一个 head 内，leaf `j` 与 `j+1` 相隔一个完整的多头页——它们是规则 stride 地址，不是字节连续，当前无法直接合并成一次大 `memcpy`。**

**v2 的"为每个 head 分配叶子连续的 CPU 地址"是错的**，此处第三次订正。

顺带：`map_list / reverse_map_list`（`py_dci.c` L2637-2644）是 `reuse_update_node` 里的**changed-leaf 重映射**，**不能**用来证明"同一子树的叶子编号连续"。DFS/subtree-interval 性质**必须单独验证**。

### 0.2 召回计时：`recall_gather` 已含 CPU gather，`recall_wait` 只是残差等待

```python
gather_start = perf_counter()                     # L1259
... 填充 _src_address_buffer ...
DCI.copy_to_buffer(...)                           # L1273  CPU 侧同步 gather
profile_recall_gather_seconds += perf_counter() - gather_start   # L1281  已结算
... dst.copy_(src, non_blocking=True)             # L1289  H2D，在 c2g_stream 上
... cuda_cast_buffer.copy_(dst, ...)              # L1291  cast
...
recall_wait_start = ...                           # L1380
c2g_stream.synchronize()                          # L1384
profile_recall_wait_seconds += perf_counter() - recall_wait_start # L1386
```

所以：
- **不存在"从 `recall_wait` 里再拆出等待 CPU gather"**——CPU gather 在 L1281 已经结算完，且在 `synchronize()` 之前同步完成。
- `recall_wait = 47.2 ms` 是**到达同步点时 c2g stream 的剩余等待**，**不等于 H2D 总时间**。
- 正确分解应为：`CPU 地址准备` → `copy_to_buffer` → `H2D` → `CUDA cast/copy` → `synchronize()` 残差。
- **H2D 与 cast 必须用 CUDA events 测量，且不能新增同步点。**
- ⇒ **这项验证必须用 GPU**，不是零 GPU 问题。

### 0.3 传输置信度需要质量 oracle

要预测的是
$$k_{\min}(q) = \text{保持 attention/logits/答案质量所需的最小页面数}$$
仅有 margin / churn / 候选稳定度**没有标签**。至少需要一种 oracle：不同 k 下的 attention 输出误差 / logits KL / 最终答案得分。
⇒ **churn AUC 不能替代**；这也是 GPU 实验，不是零 GPU。

### 0.4 其他已核实
- 所有 head **共用同一条 leaf→slot 顺序**（`page_indices = tile(arange(max_num_leaves), (n_kv_heads,1))`，L965），leaf-major / head-minor。
- `assert kvc.batch_size == 1`（L1311）。
- 语义共享正解**不需要确定性索引**：prefix hash 命中 → 复用同一份只读 KV + 同一棵已建 DCI 树 → 新增 token 走 COW。

---

## 1. 订正表（v1 / v2 的错误 → v3）

| 版本 | 说法 | v3 订正 |
|---|---|---|
| v1 | "IceCache 只用语义提高命中率，没用语义布局数据" | ❌ 删除。已有 `# data rearrangement`(L955) + `address_update`(L966) |
| v1 | "把随机 PCIe 传输变连续" | ❌ 删除。H2D 本已是一次连续拷贝 |
| v2 | "为每个 KV head 分配**叶子连续**地址" | ❌ **仍错**。是**规则 stride**（间隔 = 完整多头页），不可直接合 memcpy |
| v2 | "拆 `recall_wait`：等 CPU gather vs 等 H2D" | ❌ **仍错**。CPU gather 已在 `recall_gather` 结算；`recall_wait` 只是残差 |
| v2 | "三个零 GPU 验证一起跑" | ❌ **仍错**。其中两项需要 GPU + 质量 oracle |
| v1/v2 | "候选饱和 → 可少传页"；"叶子 78.6 vs 60 → 过度召回" | 均为**搜索侧**结论，不能推传输量 |

---

## 2. 收紧后的三条方向（内容不变，口径已修正）

**🥇 主算法 · 置信度感知的动态传输量**
区分两类置信度：**搜索置信度**（early-stop，少遍历树）≠ **传输置信度**（少取页）。主张：按在线可校准置信度决定**本步传输多少语义页面**，保留质量回退。红线：不是 per-head/per-layer **内存预算**（Fluxion/HeadWiseKV/BaKlaVa/AdaKV/HeadKV/PyramidKV 已占满）；架构上 per-head 独立索引已被 arXiv 2502.06766 实现。
**⇒ 需带质量 oracle 单独设计小实验，排在系统诊断之后。**

**🥈 主系统 · DCI 子树感知的**合并 gather`
目标改为：**先测 DCI 子树共现能否转化为实际地址区间合并**；
- 若能 → 子树感知 gather；
- 若不能（被 stride 布局阻碍）→ 研究 **head-major 布局**或新的 **fused strided-gather**；
- **不能**直接说"按父节点组织后就能连续搬运"。
收益口径：打 `recall_gather` 18.8 ms(13%)；能否碰 `recall_wait` 47.2 ms 取决于 §0.2 的拆分测量。

**🥉 远期 · 精确前缀共享只读 DCI 索引 + KV**
不需要确定性索引；区分**精确前缀（可共享 K/V）**与**语义相似文本（不可共享，hidden state + RoPE）**；语义等价去重不进近期主线。

---

## 3. 下一步：**只做一项 GPU 联合诊断**（已按 codex 意见收紧）

**不批准三线并跑。** 一次小运行同时回答两个问题：

### A. 计时拆分（CUDA events，不新增同步点）
在 `c2g_stream` 上依次 record：
- `e0`：H2D 之前
- `e1`：`dst.copy_(src)` 之后（= H2D 耗时 `e1-e0`）
- `e2`：cast `copy_` 之后（= cast 耗时 `e2-e1`）

配合已有的 `recall_gather`（CPU 地址准备 + `copy_to_buffer`）与 `recall_wait`（残差等待），得到完整五段分解。

### B. 实际地址可合并性（一次 dump，离线分析）
用 1–2 个现有样例，保存每个 `(layer, head)` 本次召回的：
- 选中的 **leaf id**（`rids`）
- 对应的 **parent id**（需从 DCI 取 leaf→parent 映射；若接口不便，先只存 leaf id + 地址）
- **实际源地址**（`page_address_buffer[layer, b, i, rids]`，uint64）

离线计算（本机可做）：
- 真实地址的**连续区间数**（受 stride 限制，大概率 > 选中页数）；
- **固定 stride 的 run 数**（同 head 内 leaf id 连续 ⟹ 地址等差 ⟹ 可用 strided copy 合并）；
- 跨 head：同一 leaf id 被多 head 选中 ⟹ 该 leaf block 内 K 区连续，可合并；
- **理论可合并比例** = 可合并传输次数 / 当前 gather 次数。

### C. 判据（一次运行即可定方向）
| 观测 | 结论 |
|---|---|
| gather 占用高 **且** 地址可合并 | 继续**子树感知 gather** |
| 地址有语义聚集但被 stride 布局阻碍 | 研究 **head-major 布局** |
| 地址几乎不可合并 | **放弃布局主线** |
| **H2D 字节量主导** | 转向**动态减少 k**（即主算法方向） |

### D. 与 codex 的并行冲突规避
`infer_state.py` 正被他并行修改。诊断补丁必须：
- **全部附加、env 门控**（如 `ICECACHE_DIAG_DUMP=1`），默认关闭；
- **不改主路径逻辑**（只加 event record 与 dump）；
- 先确认他的工作面再落。

---

## 4. 坑清单（v3）

1. 不要说"IceCache 没用语义布局"——有 `data rearrangement`。
2. 不要说"叶子连续地址"——是**规则 stride**（间隔 = 完整多头页）。
3. 不要说"把随机 PCIe 传输变连续"——H2D 本已一次连续拷贝。
4. 不要把 `recall_wait` 拆出"等 CPU gather"——它在 `recall_gather` 里。
5. 不要把"候选饱和"当"该少传页"；不要把"叶子 78.6 vs 60"当"多传了页"。
6. 不要用 churn AUC 替代**质量 oracle**。
7. 不要把 batch 墙 B≈2–3 / B≈8–16 当实测（是估算，且与 P/D 网络不同层级）。
8. 不要说"确定性索引是共享的必要条件"——正解是复用只读索引 + COW。
9. **per-head/per-layer 内存预算是红海**；**per-head 独立索引架构已被 arXiv 2502.06766 实现**。
10. 口径分开报：单机 passkey 37k"DCI 42%"、qasper budget64"检索 15.6%"、集群"传输 42.2%"是三个工况。
11. qasper20 只有 20 样本，F1 波动 ±1。
12. 单卡限制：不要选"正面击败 FreeKV / SmartGen / Mooncake"作主张。

---

## 5. 版本轨迹

| 版本 | 变更 |
|---|---|
| v1 | 布/选/共享三方向（含"语义未被用于布局"等错误假设） |
| v2 | 按代码核对修正 8 处，但保留了两处硬错误（"叶子连续"、"拆 recall_wait"）与错误的"三个零 GPU"方案 |
| **v3** | 修正**地址 stride** 与**计时分解**两处硬错误；**只保留一项 GPU 联合诊断**；记录 codex 裁定 |
