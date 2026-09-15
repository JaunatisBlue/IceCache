# 增量 DCI 地址路径优化：结果、证据与交接（`sys-optimize`）

> 承接 `docs/DeepSeek_增量DCI地址路径优化交接.md`。
> 本文件只报告**已实测**的东西；未验证的假设都显式标注。
> 未提交、未切分支、未 reset；工作区仅 `infer_state.py` 一个已修改文件（`git diff --check` 通过、`py_compile` 通过）。

---

## 0. 结论速览

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | `index_address_prepare` 的真实大头是**逐 leaf 的 `cpu_cache[b,j].data_ptr()` 循环**，占 address prepare 的 77%、index_update 的 32%；meta/写回/列表转换都不到 5% | 实测拆分，2 次重复 |
| 2 | 该循环每 leaf 约 **16 µs**（=torch 0-dim 索引 + view + `data_ptr()` 的 Python/ATen 开销），**与树规模无关**，只与新增 leaf 数成正比 | 相关系数 −0.10，per-leaf 常数 0.0156 ms |
| 3 | 地址公式 `pool_base + c2p[b,j]*page_stride + head*head_stride` 与 `data_ptr()` **逐元素完全相等** | 单元测试 5 场景 + 在体断言 **12717/12717** 元素、30 层全覆盖 |
| 4 | 直接用逻辑 page id 代替 `c2p`（`base + j*stride`）在真实 decode 中**100% 错误**（6399/6399 元素不符）；同时存活 1409 个物理页，映射高度碎片化 | 在体断言 |
| 5 | 向量化补丁（`ICECACHE_VEC_ADDR=1`）把 address prepare 从 **1.843 → 0.525 ms/token（−71.5%）**，per-leaf 循环 **1.420 → 0.192（−86.5%）**，且两次重复零波动 | 2×2 A/B |
| 6 | 补丁**没有**把开销转移给 native 侧：`native DCI insert` 0.942→0.979、`reuse_update_node` 0.357→0.382、`native address_update` 0.029→0.031 ms/token，全部落在 run-to-run 噪声内 | 2×2 A/B + 3 次同配置对照 |
| 7 | ⚠️ **本 harness 的 DCI 选择状态本身就不可复现**：同一份代码跑两遍，full-run dump 有 **72.2%** 记录不同；基线两次 71.9%；基线 vs 向量化 71.7%。因此交接文档里"DCI 选择 page ids 相等"这一条在该 harness 上**无法作为判据**，必须替换。**根因未定位**（已定位到"插入/建树侧"，但 OpenMP / promotion RNG / 线程交错三个假设一个都没排除） | 4 组对照（含未改动代码） |
| 8 | ⚠️ 同理 F1 在 3 样本子集上的 run-to-run 波动达 **1.23**（未改动代码 63.63 vs 64.86）。A/B 四跑 F1 全为 63.63，看不出系统性质量变化 | 14 次运行 |
| 9 | 收益量级要摆正：整个增量更新路径 4.48 ms/token ≈ **3.2% 的 TPOT**；本次省下 1.28 ms/token ≈ **0.93%**。recall 侧（`recall_wait` 46-48 + `native_query` 22-25 + `page_metadata` 6.6 + `recall_gather` 12）才是主体 | DCI_PROFILE |
| 10 | `T_insert` 在三个上下文桶上从 **0.598 → 3.202 ms/次**，p95/mean ≈1.6-1.9。⚠️ 这是**观察**：三桶来自三个不同文档，长度/内容/树形/新增 leaf 数/调度同时变化，**不能当复杂度结论**；根因**未定位** | 180 条 per-call 记录 |
| 11 | **20 样本 Qasper A/B（每臂 1 次）**：F1 **45.44 → 45.48（Δ+0.04）**；address prepare **1.979 → 0.566（−71.4%）**、per-leaf **−86.1%**，与 3 样本 A/B 重合。TPOT 均值 141.68→133.43、标准差 31.32→27.14，但**这 −5.8% 是运行漂移**（同配置基线两次就差 +8.53 ms）。据此把 `ICECACHE_VEC_ADDR` **默认翻为 1**；端到端真实收益口径 ≈ **1%**，非 5.8% | §5.5，含 4 条反证 |
| 12 | **36k passkey A/B（每臂 2 次）**：address prepare **1.277 → 0.534（−58.2%）**、per-leaf **0.866 → 0.198（−77.2%）**；精度四跑 accuracy=1.0。naive 臂均值 −2.46% **不可用**——该差值全部来自 session 首跑的冷启动（158.0 vs 同臂 145.9）；剔除后两臂在噪声内。**36k 端到端预期仅 −0.5%，低于单跑 ±1.5% 噪声底** | §5.6，含逐跑展开 |


---

## 1. 交付物

### 1.1 代码（工作区已改）

- `IceCache/source/icecache/infer_state.py`：+364 / −2 行。全部**纯附加、env 门控、默认关闭**（除既有 `profile_dci` 路径）。

### 1.2 新增 env 开关

| env | 默认 | 作用 |
|---|---|---|
| `ICECACHE_VEC_ADDR` | `0` | 向量化 CPU 地址准备（本任务的补丁）。默认关闭，见 §5.4 |
| `ICECACHE_ADDR_EQUIV_CHECK` | `0` | 只读在体等价断言：向量化公式必须与 `data_ptr()` 逐元素相等，否则 `AssertionError`。这是**永久性回归护栏**，可随时零成本开启 |
| `ICECACHE_PROFILE_CALL_DUMP=/path.json` | 空 | atexit 落盘每次 layer 级更新的 per-call 记录（layer / anchor-or-reuse / prev_num_points / 插入 token 数 / 新增 leaf 数 / 各阶段耗时） |

### 1.3 脚本与日志 —— ✅ **已补齐（选项 A 已执行，2026-09-14 13:0x）**

原先 `experiment/` 整体在 `.gitignore` 里（`.gitignore:2`），脚本**对 git 不可见**（`git status` 不显示、`git add -A` 静默跳过），新 clone 拿不到报告引用的任何脚本。已把脚本移到 tracked 的 `docs/` 下，现在：

```text
$ git status --short
 M IceCache/source/icecache/infer_state.py
?? docs/DeepSeek_增量DCI地址路径优化_结果.md
?? docs/DeepSeek_增量DCI地址路径优化交接.md
?? docs/addr_opt/
?? docs/parse_index_profile.py
?? docs/probe/

$ git check-ignore -v docs/addr_opt/run_ab_matrix.sh docs/probe/*.py docs/parse_index_profile.py
(无输出 ⇒ 均不在 ignore 范围内)

$ git add -n docs/
add 'docs/DeepSeek_增量DCI地址路径优化_结果.md'
add 'docs/DeepSeek_增量DCI地址路径优化交接.md'
add 'docs/addr_opt/run_ab_matrix.sh'   ...（共 17 个文件，含全部脚本）
```

维护者只需一条 `git add docs/` 即可把"文档 + 脚本"原子纳入（**本报告不代为 commit**，遵守"不 commit"约定）。

| 文件 | 用途 |
|---|---|
| `docs/addr_opt/run_addr_opt_profile.sh` | 单跑入口（参数化 env），复现 §2/§5 的所有运行 |
| `docs/addr_opt/run_ab_matrix.sh` | 主 A/B 矩阵（正确性 diag + 2×2 计时） |
| `docs/addr_opt/run_followup.sh` | full-run 状态等价 + per-call 曲线 |
| `docs/addr_opt/run_control.sh` | 同配置两次运行的对照（§5.3） |
| `docs/addr_opt/run_leafcontrol.sh` | 3 次同配置 baseline 的 leaf 数对照（§5.3） |
| `docs/addr_opt/run_head_control.sh` | **未改动代码**跑两遍的对照；默认从 `HEAD` 取基线，可传入任意旧副本 |
| `docs/addr_opt/run_null_test.sh` | 备用的"零语义扰动"对照（**未运行**，被 `run_control.sh` 取代） |
| `docs/addr_opt/run_20sample_ab.sh` | 20 样本端到端 A/B（每臂 1 次）+ 自动跑报告与 F1（§5.5） |
| `docs/parse_index_profile.py` | 把 `DCI_PROFILE` 解析成 ms/token、ms/boundary、ms/layer-update 表 |
| `docs/probe/report_ab_tpot.py` | 两臂并排表：TPOT mean/std/CV/p50/p95 + 各阶段 ms/token（§5.5） |
| `docs/probe/analyze_ab_paired.py` | **逐样本配对** TPOT 对比（消除样本构成混杂，§5.5） |
| `docs/probe/probe_addr_formula_equiv.py` | 地址公式单元等价测试 |
| `docs/probe/compare_addr_diag.py` | 两份 diag dump 对比（基址无关） |
| `docs/probe/find_divergence.py` | 定位 full-run dump 首次分叉点与幅度 |
| `docs/probe/compare_call_records.py` | per-call 记录跨运行对比 |
| `docs/probe/analyze_tinsert.py` | `T_insert` 曲线与相关性 |
| `docs/probe/compare_preds.py` | 逐样本预测文本对比 |
| `docs/probe/show_addr_equiv.py` | 等价断言结果摘要 |

**已做过的搬迁后冒烟验证（零 GPU）**：`probe_addr_formula_equiv.py` 5 场景 PASS；`parse_index_profile.py` 复现 §5.1 的 A1/B1 数字（address prepare 1.710→0.525）；`compare_call_records.py` 复现 §5.3 的 `base vs vec mean=1.62 / 25-of-60`。所有脚本 `bash -n` / `py_compile` 通过。

**未迁移的部分**：原始运行日志仍在被 ignore 的 `experiment/logs/addr_opt/`（`corr_*`、`ab_A1/A2/B1/B2`、`diag_full_*`、`tinsert_*`、`head_diag*`），因为 `.gitignore` 的既定意图是"产物按需重跑、不入库"。若希望连日志一起可见：

```bash
cp -r experiment/logs/addr_opt docs/addr_opt/logs
```

---

## 2. Step 1 —— 更细粒度 profiling 与归因

### 2.1 插桩

在 `_DCI_add` 的 `index_address_prepare` 区间内再切四段（全部只在 `profile_dci=True` 时计时）：

```
address_prepare
├─ meta/new_indices          new_num_leaves、new_address 列表、new_indices 分配
├─ per-leaf data_ptr loop    ⟵ 目标
├─ list->ndarray             np.array(tmp_addr, dtype=np.uintp)
└─ page_address_buffer write 花式索引写入
```

并把 `index_address_update` 区间内的 `native address_update` / `reuse_update_node` 也分别计时（交接文档里这两个此前没有单独口径）。另外每次 layer 级更新落一条 per-call 记录，支持按 `prev_num_points` 分桶。

### 2.2 实测（Qasper 子集 `[4, 53, 57]` = 4.5k / 2k / 21k；98 个 measured decode token、6 次 page boundary、180 次 layer 级更新、30 个 DCI 层）

基线（`ICECACHE_VEC_ADDR=0`，A1 为主，A2 用于看波动）：

| 阶段 | ms/token | ms/page-boundary | ms/layer-update | 占 index_update |
|---|---:|---:|---:|---:|
| decode TPOT（全量） | 134.47 | 2196 | 73.2 | — |
| **index update total** | **4.304** | **70.3** | **2.343** | 100% |
| window KV pack | 0.141 | 2.31 | 0.0769 | 3.3% |
| Tensor→NumPy | 0.041 | 0.67 | 0.0222 | 1.0% |
| insert preparation | 0.234 | 3.83 | 0.1276 | 5.4% |
| native DCI insert | 0.944 | 15.42 | 0.5141 | 21.9% |
| ccc writeback | 0.050 | 0.82 | 0.0272 | 1.2% |
| CPU page alloc/capacity | 0.469 | 7.66 | 0.2554 | 10.9% |
| **address update overall** | **2.301** | **37.59** | **1.2530** | 53.5% |
| ├─ **address prepare** | **1.710** | **27.92** | **0.9307** | 39.7% |
| │   ├─ meta/new_indices | 0.024 | 0.40 | 0.0132 | 0.6% |
| │   ├─ **per-leaf data_ptr loop** | **1.298** | **21.20** | **0.7067** | **30.2%** |
| │   ├─ list→ndarray | 0.042 | 0.69 | 0.0229 | 1.0% |
| │   └─ buffer write | 0.070 | 1.14 | 0.0381 | 1.6% |
| ├─ native address_update | 0.027 | 0.45 | 0.0149 | 0.6% |
| └─ reuse_update_node | 0.354 | 5.78 | 0.1928 | 8.2% |

A2（同配置重复）对应值：address prepare `1.975`、per-leaf `1.542`、index update `4.651`、native insert `0.939` → **同配置波动约 +15%**，不改结论。

### 2.3 归因结论（这是 Step 1 要回答的问题）

1. **确认归因到逐 leaf 地址解析，而不是别处。** 2.32 ms/token 的老口径里，`per-leaf loop` 占 1.30，`buffer write` 0.07、`list→ndarray` 0.04、meta 0.02；剩余 ~0.30 是循环骨架（`np.arange`、`new_indices[...] =`、以及 8 次/层的 `perf_counter()` 自身）。
2. **它的成本正比于新增 leaf 数，与树规模无关。** 三桶 `addr_prep` 0.879 / 0.971 / 0.942 ms/update 基本持平；`corr(prev_num_points, addr_leaf_ms) = −0.10`；折算 **0.0156 ms / 新增 leaf**（≈16 µs/leaf，含 8 个 head 摊分）。每次更新新增 leaf 合计 48.6 / 58.1 / 55.6。
3. **anchor 与 reuse 的成本基本相同**（1.00 vs 0.99 ms/update）：reuse 层算出的 `new_address` 列表其实**从未被使用**（reuse 分支用的是 `page_address_buffer[cur_id][0]`），但 `page_address_buffer` 写入是必需的。也就是说 20/30 个层里有约 2/3 的地址准备只为了 1/3 的有效产出——向量化后这一点不再重要，但如果将来要"省事"，reuse 层可以只写 buffer 不建列表。
4. **native insert 才是随树增长的那一项**（见 §6），不要把它和地址路径混为一谈。

### 2.4 按树规模分桶（per layer-level update）

| 桶 | prev_pts | calls | anchor/reuse | 新增 leaf | addr_prep ms | per-leaf loop ms | native insert ms |
|---|---|---:|---|---:|---:|---:|---:|
| p1024-2047 | 2000-2016 | 60 | 20/40 | 48.6 | 0.879 | 0.654 | 0.185 |
| p4096-8191 | 4432-4448 | 60 | 20/40 | 58.0 | 0.971 | 0.747 | 0.298 |
| p16384-32767 | 21264-21280 | 60 | 20/40 | 54.9 | 0.942 | 0.719 | 1.059 |

（`native insert` 列被 reuse 层的 0 摊薄了 3 倍；只看 anchor 的真实值见 §6.1。）

---

## 3. Step 2 —— 地址公式等价性

### 3.1 公式复核

CPU KV pool 由 `PagePool.__post_init__` 一次性分配一个连续 pinned tensor，`page_shape = (2, page_size, n_kv_heads, head_dim)`，因此

```
page_stride = cpu_n_bytes_per_page = 2*page_size*n_kv_heads*head_dim*4 = 131072 B（page_size=16, heads=8, d=128）
head_stride = page_size*head_dim*4 = 8192 B
address(logical_j, head_h)
  = pool.buffer.data_ptr() + c2p[b, logical_j]*page_stride + head_h*head_stride
```

`KvCache.__getitem__` 是 `self.pool[self.c2p[idx]]`，所以慢路径 `cpu_cache[b,j].data_ptr()` 恰好等于上式，**`c2p` 必须保留**：decode 阶段 page 从 `_free_ids` 集合里弹、并被原址回收复用（`_decode_alloc_1_page` 的 `self[i, e_gci].copy_(self[i,-1])`），映射必然离散。

### 3.2 单元测试：`docs/probe/probe_addr_formula_equiv.py`

真 `KvPool`/`KvCache` 构造（`page_size=16, n_kv_heads=8, head_dim=128, fp32`），逐 (batch, page, head) 全枚举比较：

```
[prefill contiguous    ] pages=8  checks=64  fast_mismatch=0  logical_shortcut_mismatch=0    c2p=[0..7]
[free/realloc churn    ] pages=12 checks=96  fast_mismatch=0  logical_shortcut_mismatch=32   c2p=[0..7,4,2,0,8]
[synthetic out-of-order] pages=8  checks=64  fast_mismatch=0  logical_shortcut_mismatch=64   c2p=[7,0,5,1,6,2,4,3]
[before c2p growth     ] pages=8  checks=64  fast_mismatch=0  logical_shortcut_mismatch=64
[after  c2p growth     ] pages=12 checks=96  fast_mismatch=0  logical_shortcut_mismatch=96
PASS
```

三点要留意：

- 连续 prefill 分配时 `logical == physical`，**这正是"看起来能省 c2p"的陷阱**；
- 只要发生一次真实的 free/realloc，`c2p` 立刻变成 `[0,1,2,3,4,5,6,7,4,2,0,8]`——**物理页被乱序回收**；
- `c2p` 扩容（page table 增长）不改变已页的映射，公式在扩容前后都成立。

### 3.3 在体等价断言（`ICECACHE_ADDR_EQUIV_CHECK=1`）

同一个断言挂在生产路径里，覆盖**真实运行**的 anchor 与 reuse 层：

| 运行 | 检查元素 | fast 不符 | 逻辑捷径不符 | 覆盖层 | 同时存活物理页 |
|---|---:|---:|---:|---:|---:|
| `vec=0`（慢路径 + 断言） | 6399 | **0** | **6399（100%）** | 30/30 | 1409 |
| `vec=1`（向量化 + 断言） | 6318 | **0** | 6318（100%） | 30/30 | 1403 |

per-layer 明细（层: 元素/fast不符/捷径不符）：`2:213/0/213 3:213/0/213 … 31:229/0/229`——三层一组数字完全相同，正是 anchor+2 reuse 共用一个树的印证。

**最强的一条**：真实 decode 里 6399/6399 = 100% 的元素用逻辑 page id 都是错的。所以交接文档里"不能直接用逻辑 page id"不是保守假设，是可在数据上量化的必然。

### 3.4 额外发现：`ICECACHE_DIAG` 采集口径

`_diag_collect` 只在 `profile_stage`（即 `profile_dci` 且已过 warmup）为真时采集，并且受 `ICECACHE_DIAG_MAX_RECORDS` 限制。**默认 400 条只覆盖不到 2 个 decode token**，做"全量状态一致性"必须显式放大（本次用 60000，实测 3 样本共 23280-23760 条、52 万个 leaf）。

---

## 4. Step 3 —— 向量化补丁

### 4.1 形态

在 `_DCI_add` 里，`self.vec_addr` 为真时：

```python
c2p_np   = cpu_cache.c2p.numpy()               # 每次调用一次的零拷贝视图
pool_base = cpu_cache.pool.buffer.data_ptr()   # pool buffer 生命周期内稳定

# 每个 head：
_ap_arr  = page_address_formula(pool_base, c2p_np[b, tmp_new_indices],
                                cpu_n_bytes_per_page, inst * offset)
tmp_addr = _ap_arr.tolist()                    # 交给 native 的仍是 Python list
page_address_buffer[cur_id][b, inst, tmp_new_indices] = _ap_arr
```

- 每个 head 一次 NumPy 整数向量运算，**没有** per-element `ctypes.cast`、没有 0-dim tensor 索引、没有 view。
- `new_address` 仍然以 **Python list of int** 传给 `_DCI_add` → `dci_address_update`，与旧路径完全同型同值。
- 不缓存 `pool_base`：`PagePool.buffer` 只在 `__post_init__` 分配一次、之后从不重分配（`c2p` 的 `utils.cat` 扩容与 buffer 无关），所以缓存是安全的；但每次调用取一次只要 ~0.5 µs，为降低生命周期风险就没做缓存。

### 4.2 被证伪的变体（重要，别重走）

原计划允许"只在边界处 `.tolist()`，或直接把 NumPy array 交给 binding"。实测 **把 NumPy 数组直接当 `new_address` 元素传给 `dci_address_update` 会让进程崩溃**：

- `corr_vecnp`：`Segmentation fault (core dumped)`，崩在第一个样本生成途中；
- `ab_C2`：不崩但**挂死**在 `0/3`，GPU 占用 0，需手动 kill。

即 M-DCI 的 Python binding 对这一入参要求 **Python list**（它显然做了 `PyList_Check` 之类的假设），NumPy array 不可互换。该变体已从补丁中**删除**（代码里 `grep vec_addr_np` = 0 命中），文档保留结论以免后人重复。

代价：`.tolist()` 约 0.012-0.013 ms/token（转换桶从 0.042 → 0.012 其实是**降了**，因为不再需要 `np.array(list)`），完全可接受。

---

## 5. Step 4 —— A/B 验证

配置：A100 80GB、32 OMP 线程、OpenBLAS 单线程、Llama-3.1-8B-Instruct、page 16、budget 64、reuse 3、ratio_1 0.01、FP16 recall、Qasper `[4,53,57]`。

### 5.1 计时（2 次重复 ×2 配置，`profile_dci` 打开）

| 阶段（ms/token） | A1 | A2 | B1 | B2 | B 均值 − A 均值 |
|---|---:|---:|---:|---:|---:|
| decode TPOT | 134.47 | 142.04 | 138.19 | 140.57 | +1.1（噪声内） |
| index update total | 4.304 | 4.651 | 3.224 | 3.182 | **−1.275（−28.5%）** |
| address update overall | 2.301 | 2.550 | 1.127 | 1.143 | **−1.291（−53.2%）** |
| └ address prepare | 1.710 | 1.975 | 0.525 | 0.525 | **−1.318（−71.5%）** |
| &nbsp;&nbsp;├ per-leaf data_ptr loop | 1.298 | 1.542 | 0.192 | 0.192 | **−1.228（−86.5%）** |
| &nbsp;&nbsp;├ meta/new_indices | 0.024 | 0.026 | 0.038 | 0.038 | +0.013 |
| &nbsp;&nbsp;├ list→ndarray | 0.042 | 0.045 | 0.012 | 0.013 | −0.031 |
| &nbsp;&nbsp;└ buffer write | 0.070 | 0.075 | 0.051 | 0.051 | −0.022 |
| ├ native address_update | 0.027 | 0.030 | 0.031 | 0.030 | +0.002 |
| └ reuse_update_node | 0.354 | 0.359 | 0.377 | 0.387 | +0.026 |
| native DCI insert | 0.944 | 0.939 | 1.015 | 0.942 | +0.037 |
| CPU page alloc | 0.469 | 0.530 | 0.485 | 0.476 | −0.019 |
| window KV pack | 0.141 | — | 0.163 | — | — |

折算（`98 measured token / 180 layer 级调用 ⇒ 0.5444 tok/call`、`tokens/boundary = 16.33`）：

| | 基线 | 向量化 |
|---|---:|---:|
| address prepare / page-boundary | 27.9 / 32.3 ms | **8.58 / 8.57 ms** |
| address prepare / layer-update | 0.931 / 1.075 ms | **0.286 / 0.286 ms** |
| per-leaf loop / layer-update | 0.707 / 0.840 ms | **0.105 / 0.105 ms** |
| index update / page-boundary | 70.3 / 76.0 ms | **52.7 / 52.0 ms** |

- 波动：基线两次相差 15%（1.710 vs 1.975），向量化两次 **0%**（0.525/0.525）；效应量（−71.5%）远在噪声带之外。
- **native 侧无系统性变化**（要求 5）：native insert +3.9%、reuse_update +7%、native address_update +7%，但对照实验（§5.3）显示这三项在**同一配置的两次运行之间**本来就有 10-18% 的差异，所以这些"+7%"不可解释为补丁效应。
- **无新增 CUDA 同步**（要求 6）：补丁纯 NumPy 整数运算，路径上没有任何新增 `.synchronize()` / event；`ICECACHE_DIAG` 的 h2d/cast 计数在 A/B 中同为 800/800。

### 5.2 正确性

**(a) 地址逐元素相等（要求 1）** —— §3.3。单元测试 384 次比较 + 在体 12717 个元素（`vec=0` 6399 + `vec=1` 6318），**0 不符**。
交接文档里 `index_address_prepare = 2.320 ms/token`，本次基线测得 `1.710 / 1.975`（两次重复），差异在 ±15% 同配置波动范围内；结论与拆分比例不受影响。

**(b) 生成输出** —— 2×2 矩阵，逐样本对比 `pred` 文本：

| 对比 | s0 (4.5k) | s1 (2k) | s2 (21k) |
|---|---|---|---|
| A1 vs A2（基线两次） | 相同 | 相同 | 相同 |
| B1 vs B2（向量化两次） | 相同 | 相同 | 相同 |
| A1/A2 vs B1/B2（4 组） | 相同 | 相同 | **不同（相似度 0.921）** |

唯一的差异是 s2、且 4 组 A↔B 对比**完全一致**（同一处、同相似度）：回答里一串语言名的**枚举顺序**从 `…, Spanish, Kiswahili, Welsh, Yue Chinese, Estonian, …` 变成 `…, Kiswahili, Welsh, Yue Chinese, Spanish, Estonian, …`，**元素集合相同**。

**(c) 质量** —— 逐运行 F1（Qasper，同一子集）：

| 运行 | 代码 | F1 |
|---|---|---:|
| ab_A1 / ab_A2 | 插桩基线 | **63.63 / 63.63** |
| ab_B1 / ab_B2 | 向量化 | **63.63 / 63.63** |
| head_diag1 / head_diag2 | **未改动代码** 两次 | **63.63 / 64.86** |

未改动代码两次就跑出 63.63 与 64.86，**本子集 F1 的 run-to-run 波动 = 1.23**。A/B 四跑全为 63.63，看不出系统性质量变化。

**(d) F1 波动不是本子集特有** —— 其余运行：`diag_full_base 56.49` vs `diag_full_base2 64.86`（同配置！）vs `diag_full_vec 63.63` / `diag_full_vec2 63.63`；`tinsert_base 61.25` vs `tinsert_vec 63.63`。

### 5.3 ⚠️ 关键发现：本 harness 的 DCI 状态不可复现，"page ids 相等"判据不成立

交接文档 Step 4 要求"新旧地址逐元素相等"且"**DCI 选择 page ids 相等**"。前者已证明；**后者在本 harness 上无法达成**，而且不是补丁造成的。三组 full-run diag 对照（`ICECACHE_DIAG_MAX_RECORDS=60000`，3 样本全量、30 层 × 8 head × 全部 measured step）：

| 对比 | 记录总数 | 不同记录 | 首个分叉 | 选中 leaf 总数差 |
|---|---:|---:|---:|---:|
| **未改动代码** run1 vs run2（`head_diag1/2`） | 23520 | 16980 / 23520 = **72.2%** | record 0 (layer 2 / head 0) | — |
| 插桩基线 run1 vs run2（`diag_full_base/base2`） | 23280 | 16743 / 23280 = **71.9%** | record 0 | +1.46% |
| 基线 vs 向量化（`diag_full_base/vec`） | 23280 | 16683 / 23280 = **71.7%** | record 0 | +0.60% |
| 向量化 run1 vs run2（`diag_full_vec/vec2`） | 23520 | 16896 / 23520 = **71.8%** | record 0 | — |

即：**同一份未改动代码跑两遍，分叉幅度（72.2%）比"基线 vs 向量化"（71.7%）还大**。三组都在 record 0（layer 2 / head 0）就出现"34 选 34、只有 25 个相同"以及 leaf id ±1 的整体平移。

per-call 记录把分叉**定位**到了插入/建树侧：3 次同配置 baseline + 1 次向量化，共 4 次运行，对比 anchor 调用的 `new_leaves_total`：

| 对比 | 平均 \|Δleaf\| | 60 次调用中完全相同 |
|---|---:|---:|
| base vs base2 | 1.60 | 28/60 |
| base vs base3 | 1.38 | 27/60 |
| base2 vs base3 | 1.32 | 29/60 |
| **base vs vec** | **1.62** | **25/60** |
| base2 vs vec | 1.78 | 22/60 |
| base3 vs vec | 1.57 | 28/60 |

**同配置之间的差异（1.32-1.60）与跨配置差异（1.57-1.78）不可区分**。`native_insert_ms` 的 |Δ| 同样：同配置 10.8-18.3%，跨配置 12.9-14.7%。

**可以从数据里确定的**：`prev_num_points`（prefill 后的 `num_points`）四次运行**完全相同**（4432、2000、21264 等），但插入后新增 leaf 数不同。因此分叉**位于插入/建树侧**，而不是地址准备侧（地址已由 §3.3 证明同运行内逐元素相等）。

**不能从数据里确定的**：插入侧为什么会非确定。至少有三个互斥候选，**本次一个都没有排除**：

| # | 假设 | 判别实验（最小） |
|---|---|---|
| H1 | `parallel_level=2` 的 OpenMP 非确定归约顺序改变了树结构 | `ICECACHE_DCI_PARALLEL_LEVEL=2` 跑两次 vs `=0` 跑两次；若 0 两次一致 → H1 成立 |
| H2 | `promotion_prob=0.01` 的 C 侧 RNG 未固定 seed | 把 seed 固定/暴露后跑两次；若一致 → H2 成立 |
| H3 | 增量插入（主线程，page boundary）与 `estimate_select_recall`（executor 线程）在同一个 `dci_db` 上交错，顺序取决于时序 | 需要先确认调用时序；再让 executor 走同步路径跑两次 |

H1/H2 可以一次矩阵跑完（2×2），H3 需要先做代码时序审查（**纯读代码**）。**在跑这个矩阵之前，任何"来自 OpenMP"或"来自 promotion"的说法都只是猜测，本报告不再这么写。** 唯一可以确定的结论是：跨配置差异与同配置差异同幅度 ⇒ **没有证据表明地址补丁改变了 DCI 行为**。

> ⚠️ 这条结论对项目其他工作也有影响：此前用 `ICECACHE_DIAG` 的 leaf/地址 dump 做"等价性/确定性"对比的做法，**跨进程不成立**。同一进程内的自洽校验（如 §3.3 的 `data_ptr()` 对照、§3.2 的单元测试）仍然有效。另外 2 样本 / 400 条上限下的早期对比曾经**完全一致**，说明分叉是间歇性的——正因为间歇，才更不能用它当闸门。


### 5.4 因此，补丁的验收结论与开关默认值

**在可验证的口径上，补丁成立**：

1. 地址公式与 `data_ptr()` 逐元素相等（单元 + 在体，含 anchor/reuse、含扩容前后、含非单调 `c2p`）；
2. 传给 native 的 `new_address` 与旧路径同型同值（Python list of int）；
3. 质量无系统性变化（A/B 四跑 F1 全 63.63；对照显示本子集 F1 噪声 ±1.23）；
4. `index_address_prepare` 稳定下降 69-72%，重复运行零波动；
5. native insert / reuse_update / native address_update 落在噪声内，无开销转移；
6. 无新增同步点。

**~~但 `ICECACHE_VEC_ADDR` 的默认值仍留 `0`（关闭）。~~** ~~理由：唯一无法用本 harness 证明的是"逐 token 完全一致"，而它恰恰被 §5.3 证明不可判定。建议按本项目既有节奏（如双缓冲那次）先跑 **20 样本 A/B** 看 F1 锚点，再翻默认值；期间可用 `ICECACHE_VEC_ADDR=1` 显式开启做实验。~~

**2026-09-14 更新：20 样本 A/B 已做，默认值翻为 `1`。** 见 §5.5。补丁的"唯一无法证明项"仍是逐 token 一致性（§5.3），但它不再作为默认值的阻塞条件——判据是 20 样本质量锚点，已通过（ΔF1 = +0.04），且地址等价性是逐元素可证的。

**收益量级提醒**：省下 1.28 ms/token ≈ TPOT 的 0.93%，单次重复无法在 TPOT 上分辨。要看到 TPOT 变化需要更多重复或更长的 decode。

---

### 5.5 20 样本端到端 A/B 与默认值升级（2026-09-14 14:24–14:30）

配置同 §1.3/§2.2，`--max-samples 20`（长度分层抽 20 篇 Qasper，context 2054–21317），每臂 **1 次**重复；加 `ICECACHE_PROFILE_STEP_SAMPLES=1` 采集逐步延迟。产物：`experiment/logs/addr_opt/q20_A.log`、`q20_B.log`；脚本 `docs/addr_opt/run_20sample_ab.sh`、`docs/probe/report_ab_tpot.py`、`docs/probe/analyze_ab_paired.py`。

**TPOT（逐步采样，`DCI_PROFILE.decode_step_latency`）**

| | A 基线 (vec=0) | B 向量化 (vec=1) |
|---|---:|---:|
| decode steps measured | 363 | 348 |
| **TPOT 均值** | **141.68 ms/token** | **133.43 ms/token** |
| **TPOT 标准差** | **31.32 ms** | **27.14 ms** |
| CV | 0.221 | 0.203 |
| p50 / p95 | 136.13 / 195.70 | 130.94 / 176.91 |
| min / max | 100.23 / 350.25 | 89.18 / 321.87 |

自洽校验：`decode_step_latency.n == decode_steps_measured`，`mean_ms == decode_tpot_ms`（两臂均成立）。

**F1（Qasper，同一 20 样本子集）**

| arm | F1 |
|---|---:|
| A 基线 | **45.44** |
| B 向量化 | **45.48** |
| Δ | **+0.04** |

两眼都在本仓库既有 20 样本锚点族（~45.4–45.5）内。

**阶段级（可归因的部分）**

| 阶段 (ms/token) | A | B | Δ |
|---|---:|---:|---:|
| index update total | 4.124 | 2.652 | **−35.7%** |
| └ address prepare | 1.979 | 0.566 | **−71.4%** |
| &nbsp;&nbsp;└ per-leaf data_ptr loop | 1.519 | 0.211 | **−86.1%** |
| ├ native DCI insert | 0.563 | 0.558 | −1.0% |
| ├ CPU page alloc | 0.534 | 0.501 | −6.1% |
| └ reuse_update_node | 0.315 | 0.306 | −3.0% |
| recall_wait | 44.69 | 44.90 | +0.5% |
| native_query | 22.33 | 18.95 | −15.2% |
| recall_gather | 11.29 | 9.94 | −11.9% |
| page_metadata | 6.62 | 6.52 | −1.6% |

address prepare 的 **−71.4%** 与 per-leaf 的 **−86.1%**，与 §5.1 三样本 A/B（−71.5% / −86.5%）几乎重合 ⇒ 这一项在两组**不同样本集**上可复现；native insert / reuse_update / page alloc 仍落在噪声内，无开销转移。

**⚠️ TPOT 的 −5.8% 不能归因给补丁。** 四条证据：

1. **样本构成被混杂**：两臂 decode 步数不同（363 vs 348）。改用逐样本配对（同一批 20 篇，逐条 context 长度一致）后 Δ = **−8.68 ± 7.53 ms**（19/19 样本更快，95% CI [−12.07, −5.30]，排除零）——配对只消除了构成混杂，没消除顺序/机器漂移。
2. **补丁不碰的两个阶段同幅度下降**：`native_query` −15.2%、`recall_gather` −11.9%，合计 −4.7 ms，已超过 −8.68 ms 的一半。
3. **同配置对照**：把 §5.1 的交错运行做同样的配对分析 —— 两次**都是基线**的 `ab_A1` vs `ab_A2` 给出 Δ = **+8.53 ms**（3/3 样本一致变慢，95% CI [+1.27, +15.78]，排除零）；`A1 vs B1` = +3.83 ms（CI 含零）、`A2 vs B2` = −2.02 ms（CI 含零）。即**基线自身的漂移（8.5 ms）不小于跨配置差异（2–9 ms）**。
4. **量级不成立**：补丁能拿掉的只有 1.4 ms/token（实测），≈ 1% TPOT，不是 5.8%。

**结论与默认值决定**：本步的判据是"20 样本质量锚点"，**通过**（ΔF1 = +0.04，两臂都在历史锚点族内）。因此按约定把 `ICECACHE_VEC_ADDR` 的默认值由 `0` 翻为 **`1`**（`infer_state.py` + 脚本 fallback 同步；`ICECACHE_VEC_ADDR=0` 成为 opt-out）。

**但要修正标签**：升级的是"一个**可证明等价、质量中性**的局部优化"，**不是**"端到端提速 5.8%"。诚实口径：

- **阶段级**：`index_address_prepare` **−71%**（跨样本集可复现）；
- **端到端**：≈ **1%（约 1.4 ms/token）**，**1 次重复下不可分辨**。

要拿到可信的端到端数字，需要**交错多次重复（≥3 次/臂）+ 逐样本配对**——这是本 harness 的方差所要求的，与 §5.3 的结论一致。1 次重复的 A/B 只能用来判**质量**，不能用来判 TPOT 幅度。

> 此前所有用 `vec=0` 得到的数字（§2、§5.1、§6.1）现在都是"显式 opt-out"的结果；`docs/addr_opt/*.sh` 里的 A/B 驱动仍显式设置 `ICECACHE_VEC_ADDR=0`/`=1` 以选定臂。

---

### 5.6 36k passkey 端到端 A/B（2026-09-15，每臂 2 次重复）

**先说结论**：36k 上**地址路径本身稳定快 58%（per-leaf −77%）**，但它对 TPOT 的贡献（预期 ~0.5%）**在 2 次重复下测不出来**；naive 臂均值给出的 −2.46% 是**首跑冷启动假象**。

**配置**（`docs/addr_opt/run_passkey36k_ab.sh`）

- 用仓库自带的 `IceCache/benchmark/passkey_pred.py`，`N_GARBAGES=134775` → **`Context length: 36000`**（四跑完全一致）
- `--num-tests 1`（确定性 prompt）+ `--profile-new-tokens 96` + `--profile-warmup-tokens 8` → **四跑的 measured decode steps 全为 87**，即两臂解码等长，无样本构成混杂
- 其余固定配置同 §1.3；交错 A,B,A,B
- 精度：`passkey.jsonl` 四跑 **accuracy = 1.0，length = 36000**

**⚠️ `N_GARBAGES` 切的是字符数，不是 token 数**（`garbage_inf[:n_garbage]`，≈3.75 chars/token）。用真 tokenizer（loc=0.0）标定：

| `N_GARBAGES` | context tokens | 备注 |
|---:|---:|---|
| 36000 | 9,660 | 「想当然」的错值（第一次就踩了） |
| **134775** | **36,000** | 本节使用 |
| 140000 | 37,394 | 文档里"passkey 37k"的长度（9 个历史 log 都是这个值） |
| 150000 | 40,060 | 文档里"passkey 40k"的长度（9 个历史 log） |

`N_GARBAGES` 在全仓库**只出现在 `passkey_pred.py` 自身**（默认 `[38000, 76000, 114000]` → 9.9k / 19.6k / 29.4k tokens），**没有任何脚本设置过它**。历史的 36k/37k/40k 都是命令行临时设的，所以本节这个脚本是**第一个可复现的长上下文设置**。

**结果（每臂 2 次重复，交错 A,B,A,B）**

| 阶段 (ms/token) | A 基线 | B 向量化 | Δ |
|---|---:|---:|---:|
| TPOT mean（臂均值） | 151.95 | 148.20 | **−3.74（−2.46%）** |
| TPOT std | 25.38 | 21.21 | −16.4% |
| p50 / p95 | 146.80 / 216.12 | 144.50 / 201.58 | −1.57% / −6.73% |
| index update total | 4.013 | 3.038 | −24.3% |
| **address prepare** | **1.277** | **0.534** | **−58.2%** |
| **per-leaf data_ptr loop** | **0.866** | **0.198** | **−77.2%** |
| native DCI insert | 0.361 | 0.342 | −5.1% |
| CPU page alloc | 0.324 | 0.305 | −5.9% |
| reuse_update_node | 0.286 | 0.233 | −18.5% |
| recall_wait | 35.216 | 35.237 | **+0.06%** |
| recall_gather | 10.928 | 10.935 | **+0.06%** |
| native_query | 34.657 | 34.007 | −1.87% |
| page_metadata | 7.016 | 6.931 | −1.21% |

**⚠️ 上面那 −2.46% 不能用。** 逐跑展开后，整个差值来自 `p36k_A1` —— **整个 session 的第一跑**：

| run | 顺序 | TPOT mean | p50 | p95 | min | 逐样本 print |
|---|---:|---:|---:|---:|---:|---:|
| `p36k_A1` | 1 | **158.00** | 152.77 | 227.23 | 132.54 | **162.9** |
| `p36k_B1` | 2 | 149.12 | 145.30 | 200.07 | 128.30 | 152.9 |
| `p36k_A2` | 3 | **145.90** | 140.82 | 205.02 | 119.06 | 151.6 |
| `p36k_B2` | 4 | 147.29 | 143.69 | 203.09 | 122.67 | 151.5 |

- `A1` 比同臂的 `A2` 慢 **+8.1%**，而且是**全分布平移**（p50 152.8 vs 140.8、min 132.5 vs 119.1）→ 冷启动，不是长尾。
- **剔除 `A1` 后：`A2`=145.90 vs B 臂均值 148.20 → B 反而慢 2.3 ms**，落在噪声内（B 自身两跑差 1.8 ms）。
- 三条独立读数（`DCI_PROFILE` 逐步均值、逐样本 print、p50）都指向同一结论；`p95` 的"−6.7%"同样是 `A1` 的 227 拉出来的。
- 反证漂移：**补丁不碰的阶段全平**（`recall_wait` +0.06%、`recall_gather` +0.06%）——这一轮**没有** §5.5 那种全局漂移，唯一的异常就是首跑。

**36k 的诚实口径**

- **阶段级（可信，跨工况复现）**：`index_address_prepare` **−58.2%**（36k passkey）/ −71.5%（3 样本 qasper）/ −71.4%（20 样本 qasper）；per-leaf **−77.2%** / −86.1% / −86.5%。36k 上**绝对节省 0.74 ms/token**。
- **端到端**：预期 ≈ **−0.5%**（0.74 / 151.9），**低于本 harness 单跑 ±1.5% 的噪声底** ⇒ 2 次重复在原理上测不出来。这不是"重复不够"，是**该设计不可分辨**；再多的重复也只能把置信区间收窄到 0.5% 量级，而不能改变结论。

**测量协议修正（本轮新增的认知）**：**每个 session 的第一跑比后续慢 6-8%**（本例 158.0 vs 145.9；20 样本 qasper 那轮 A1 也是第一跑且偏慢）。配合"臂按位置交替"，若重复数为偶数，**冷跑会固定落在某一臂**上，制造假的 A/B 差异。以后固定：

1. 先跑一次**丢弃用的 warm-up**（不计入统计）；
2. 重复数 **≥3**（奇数更稳，避免首/末位置固定绑定某一臂）；
3. 用 `decode_step_latency` 的 `mean/std/p50/p95` **+ 逐样本 print 交叉核对**，先剔除离群跑再算均值；
4. 始终检查"补丁不碰的阶段"是否同步变化。

---

## 6. Step 5（后续方向）—— native DCI 增量插入

### 6.1 `T_insert` 曲线（`ICECACHE_PROFILE_CALL_DUMP`，只看 anchor 层，固定 16-token 批次）

| prev_num_points | calls | 新增 leaf/次 | `native_insert_ms` 均值 | p95 | `reuse_update_node` 均值 | per-leaf loop 均值 |
|---|---:|---:|---:|---:|---:|---:|
| 2000-2016 | 20 | 48.6 | **0.598** | 0.966 | — | 0.741 |
| 4432-4448 | 20 | 58.1 | **1.227** | 2.450 | — | 0.975 |
| 21264-21280 | 20 | 56.2 | **3.202** | 6.130 | — | 0.785 |
| reuse 三层同上 | 40/层 | — | 0（reuse 不做插入） | — | 0.223 / 0.449 / 0.318 | 0.737 / 0.956 / 0.772 |

- ⚠️ **这是观察，不是因果结论。** 三个桶分别来自三个**不同文档**（4.5k / 2k / 21k），上下文长度、prompt 内容、树形结构、新增 leaf 数（48.6 / 58.1 / 56.2）与 CPU 调度**同时变化**。所以"树规模 ×10.64 → 插入 ×5.35"、"经验指数 ≈0.70"**只能当描述**，不能当复杂度结论，也不能外推。
- `corr(prev_num_points, native_insert_ms) = 0.544`（60 次 anchor 调用）同样只是这三个桶内的相关，混杂因子未分离。
- **p95/mean ≈ 1.6-1.9**（最高桶 p95 = 6.13 ms vs 均值 3.20 ms）→ 长尾是**观察到的事实**；"叶子分裂 / 节点搬迁"是**未验证的候选解释**。
- 对照组：**`per-leaf loop` 与树规模无关**（`corr = −0.10`）；vector 化后 `corr = −0.005`、per-leaf 0.0156 → 0.0022 ms（−86%）。这从另一个方向确认了 §2 的归因。
- 量级：`native insert` 0.94 ms/token = TPOT 的 0.68%；`reuse_update_node` 0.36 ms/token = 0.26%。

### 6.2 建议的研究顺序（**都还没做**，按"先量后改"排）

1. **先把 `T_insert` 曲线做实（这一步在"找根因"之前）。** 现在只有 3 个点且都被文档内容混杂。做法：
   - `--max-samples` 提到 6-10，拿到 6-10 个**对数间隔**的上下文长度（`length_stratified_subset` 正好给长度分位数）；
   - 优先补一个**同文档、不同截断长度**的对照（同一段文本截成 2k/4k/8k/16k），把"文档内容"这个混杂因子压掉；
   - 报告要同时给出 per-inserted-token 与 per-new-leaf 的归一化值，不要只给原始 ms；
   - 只有在点数够、混杂因素被控住之后，才谈经验指数。**这一步之前，不要选根因假设。**
2. **候选预算：`c_num_to_visit` 已被源码级证伪，不要再扫它（本报告前一版在这里写错了，见 §9）。**
   M-DCI 的 C 源在机器上就有：`/tmp/icecache-mdci-source/src/dci.c`（另有 `/tmp/dci_probe/instr/src/dci.c`、`/tmp/m-dci-icecache/src/dci.c` 两份副本；注意同目录还有 `dci.c.bak_oldpatch`，需确认与已安装 `.so` 同源）。只读检查得到两条硬事实：

   - **construction 侧（`dci_add`，`dci.c:3137` 起）**：函数一进来就把预算夹到**本批插入的 token 数**：
     ```c
     construction_query_config.num_to_visit = min_i(construction_query_config.num_to_visit, num_points);
     construction_query_config.num_to_retrieve = min_i(construction_query_config.num_to_retrieve, num_points);
     ```
     而我们传 `num_points = dci_len = 16`、`c_num_to_visit = num_to_visit = prev_num_points`（2000…21280）。
     ⇒ **`c_num_to_visit` 被夹到 16，传 `prev_num_points` 与传 `16`、传 `100000` 在 construction 侧等价。它是 inert 的，不可能是树规模成本的来源。**
   - **query 侧（`dci_query_single_point_single_level`，`dci.c:3642` 起，`dci.c:3921` 是同类第二处）**：预算是
     ```c
     num_projs_to_visit = max_i(query_config.num_to_visit * num_simp_indices,
                                ceil(query_config.prop_to_visit * num_points * num_simp_indices));
     ```
     `prop_to_visit = 1.0` 时第二项在 `num_to_visit < num_points` 时恒占优 ⇒ **这正好解释了仓库既有实验里"`num_to_visit` 从 12.5% 提到 100% 耗时不变"**（`IceCache_CPU开销归属_2026-09-12.md` §3②）。该结论是结构性的，不是测量偶然。

   因此**下一步不是扫 `c_num_to_visit`**，而是：
   1. **继续读 C 源**（零 GPU）：顺着 `dci_add` → `dci_add_one_point` 找插入内部真正用于"定位/挂接"的搜索预算走哪条路径、是否同样被夹或按树规模消耗。这是目前性价比最高的一步；
   2. 若要动 `prop_to_visit` / `field_of_view`：`dci.c:4069-4070`、`dci.c:4241` 显示 `prop_to_visit < 1.0` 会打开一条 `conv_seen` 早停分支 ⇒ **它们是语义旋钮**，必须先做质量实验，且不能和地址补丁混在同一次 A/B；
   3. 只有 (1) 指向某个具体机制之后，再设计微基准去量"实际访问的 projection / node / candidate 数"，而不是先扫参数。

3. **OpenMP 在小批次上的启动/barrier 成本。** exp 04 只扫了 **selection** 路径（结论：level 2 最好）。插入路径每次只有 16 个 token，值得单独用 `ICECACHE_DCI_PARALLEL_LEVEL=0/1/2` 重扫，只看 `index_native_insert_ms_per_token`（现在已有独立口径）。注意 `strings` 显示 binding 里存在 `py_dci_address_update._omp_fn.0`，说明 native 侧确有并行区——但这只说明"存在并行"，不说明"并行是瓶颈"。
4. **promotion 的影响现在可直接读到。** 新增的 `new_leaves_total` / `new_leaves_max` per-call 记录，让 `promotion_prob` 扫描（既有脚本 `experiment/probe/mdci_promotion_sweep.py`，仍留在被 ignore 的目录里）可以同时给出"新增 leaf 数变化"和"插入耗时变化"。注意：若 H2（RNG 未固定）成立，promotion 扫描本身也会受非确定性干扰，**应先把 H1/H2 的判别矩阵跑完再做**。
5. **长尾定位。** per-call 记录已带 `new_leaves_max` 与各阶段耗时；把 p95 那几个调用拎出来，看是否与 `new_leaves_max` 大或与 promotion 事件相关。相关 → 才继续查 leaf split / relocation。
6. **边界（交接文档已明确，这里再确认一次）**：不要在这一阶段把 token 级检索换成 page centroid 检索。那是另一套算法，会改变稀疏注意力的选择，必须独立做质量实验，不能与地址优化或插入参数扫描混在同一次 A/B 里。


### 6.3 一条更宏观的提醒

`index_update` 整条路径 4.48 ms/token（TPOT 的 3.2%）：地址 prepare 1.84 + native insert 0.94 + page alloc 0.50 + reuse_update 0.36 + 其它 0.84。而同一份 `DCI_PROFILE` 里 `recall_wait 46-48 + native_query 22-25 + page_metadata 6.6 + recall_gather 12 ≈ 90 ms/token`（TPOT 的 65%）。**即使把增量更新整条消灭，天花板也就 3%。** 若目标是总量级收益，DCI selection 侧（exp 04 已定位）与 recall 传输侧（双缓冲那条线）的杠杆更大；本条线适合作为"低风险、可验证、无质量代价"的确定性收益收尾。

---

## 7. 复现命令

> 路径即**当前工作树**的真实路径（脚本已从被 ignore 的 `experiment/` 迁到 `docs/`）。

```bash
# 0. 单跑入口（env 驱动）
ICECACHE_VEC_ADDR=1 bash docs/addr_opt/run_addr_opt_profile.sh <run-name> 3

# 1. 地址公式单元等价
PYTHONPATH=/home/yx/IceCache/IceCache/source \
  /home/yx/miniconda3/envs/icecache/bin/python \
  docs/probe/probe_addr_formula_equiv.py

# 2. 在体等价断言（任一次运行加 ICECACHE_ADDR_EQUIV_CHECK=1 即可，会直接 AssertionError）
ICECACHE_ADDR_EQUIV_CHECK=1 ICECACHE_VEC_ADDR=1 \
  bash docs/addr_opt/run_addr_opt_profile.sh equiv_check 2

# 3. 主 A/B 矩阵（正确性 diag + 2 次重复计时）
bash docs/addr_opt/run_ab_matrix.sh                # np-passthrough 两跑已从矩阵移除（会崩/挂）

# 4. 不可复现性对照（同配置两次 / 未改动代码两次）
bash docs/addr_opt/run_control.sh
bash docs/addr_opt/run_leafcontrol.sh
bash docs/addr_opt/run_head_control.sh   # 默认用 HEAD 版本作对照

# 5. 解析
python docs/parse_index_profile.py A1=experiment/logs/addr_opt/ab_A1.log B1=experiment/logs/addr_opt/ab_B1.log
python docs/probe/analyze_tinsert.py /tmp/call_records_base.json /tmp/call_records_vec.json
python docs/probe/compare_call_records.py base=/tmp/call_records_base.json vec=/tmp/call_records_vec.json

# 6. 20 样本端到端 A/B（§5.5）—— 每臂 1 次，自动出报告 + F1
bash docs/addr_opt/run_20sample_ab.sh

# 7. 逐样本配对（消除样本构成混杂；也可用来做同配置漂移对照）
python docs/probe/analyze_ab_paired.py \
  A=experiment/logs/addr_opt/q20_A.log B=experiment/logs/addr_opt/q20_B.log
# 同配置对照（判定漂移，必做）：
python docs/probe/analyze_ab_paired.py \
  A1=experiment/logs/addr_opt/ab_A1.log A2=experiment/logs/addr_opt/ab_A2.log
```

---

## 8. 风险、遗留与建议的下一步

### 8.1 交付补齐（**必做**，建议选项 A）

| 选项 | 做法 | 评价 |
|---|---|---|
| **A（已执行，2026-09-14）** | 把脚本从被 ignore 的 `experiment/` **移到 tracked 目录** `docs/addr_opt/`、`docs/probe/`、`docs/parse_index_profile.py` | ✅ 已完成并验证：`git status` 可见 5 个新条目、`git check-ignore` 无输出、`git add -n docs/` 列出全部 17 个文件；迁移后冒烟测试全部通过。见 §1.3 |
| B（未采用） | `git add -f experiment/addr_opt experiment/probe docs/parse_index_profile.py` | 可见，但把 ignored 树里的文件塞进 index，与 `.gitignore` 的意图冲突；且文件仍在 `experiment/` 下，后人容易再踩 |
| C（未采用，可后续单独做） | 收窄 `.gitignore`：加 `!experiment/addr_opt/`、`!experiment/probe/*.py` 等否定规则 | 最贴合该 ignore 规则的原意（注释写的是 "Experiment **outputs** are regenerated per run"，管的是产物不是脚本），但改的是仓库级约定，需要维护者确认 |

A 已执行；B/C 均未采用。三个选项都**不含 commit**（遵守"不 commit"的约定）——需要维护者做的只剩一条 `git add`（+ 决定是否提交）。

### 8.2 遗留

| 项 | 状态 / 建议 |
|---|---|
| 逐 token 完全一致 | **本 harness 不可判定**（§5.3）。建议按项目惯例先跑 20 样本 A/B 看 F1 锚点，再决定是否把 `ICECACHE_VEC_ADDR` 默认翻成 1 |
| DCI 状态不可复现的根因 | **未定位**。已定位到"插入/建树侧"（`prev_num_points` 相同、插入后 leaf 数不同），但 H1(OpenMP) / H2(promotion RNG) / H3(线程交错) 一个都没排除。判别矩阵见 §5.3；其中 H3 需要先做纯代码的调用时序审查 |
| `num_to_visit` 归因 | **证据不足，已从"首选嫌疑"撤回**（§6.2 第 2 条 + §9）。在 construction 分支拿到"实际访问的 projection/node/candidate 数"之前，不要再把它写进任何计划 |
| `T_insert` 随树增长 | **当前只是观察**（§6.1）。先扩规模点 + 同文档截断对照，再谈复杂度 |
| `ICECACHE_DIAG` 作为等价性工具的适用边界 | 已修订认知：只在"同进程内自洽"有效，跨进程不可用。建议在 `_diag_collect` 附近加一句注释说明（我未擅自加，以免和并行的其他改动冲突） |
| NumPy 直传 binding | 已证伪并删除（segfault/hang）。若将来要试点，务必先单独跑一遍确认不会挂 |
| 环境 | 运行期间 GPU 已清空（`nvidia-smi` 0 MiB），无残留进程；`ab_C2` 挂死进程已手动终止 |

---

## 9. 修订记录（本轮复核后的更正）

复核指出本报告前一版有三处推断不成立、一处交付状态描述不实。逐条更正如下，**旧表述作废**：

| # | 前一版的说法 | 更正 |
|---|---|---|
| 1 | §6.2 第 2 条把"`num_to_visit = prev_num_points` + `c_prop_to_visit = 1.0` ⇒ 整树扫描 ⇒ 随树增长的每点成本"列为**头号嫌疑**，并据此建议扫 `c_num_to_visit` | **撤回**，并进一步**源码级证伪**：`dci.c` 的 `dci_add` 把 `construction_query_config.num_to_visit` 用 `min_i(…, num_points)` 夹到**本批插入的 token 数（=16）**，所以 `c_num_to_visit` 在 construction 侧完全 inert；query 侧的 `max_i(num_to_visit*…, ceil(prop_to_visit*num_points*…))` 才是仓库既有"零杠杆"实验的结构性解释。改为"先继续读 C 源的插入路径"。见 §6.2 第 2 条 |
| 2 | §6.1 / §0 第 10 行把"树规模 ×10.64 → 插入 ×5.35、经验指数 ≈0.70"当作缩放规律叙述 | 降级为**观察**：三个桶来自三个不同文档，长度/内容/树形/新增 leaf 数/调度同时变化，**不是因果，不能外推**。先把规模点扩到 6-10 并补同文档截断对照。见 §6.1 |
| 3 | §5.3 断言分叉"来自 `parallel_level=2` 的 OpenMP 归约顺序，以及 `promotion_prob=0.01`" | 降级为**三个未排除的候选假设 H1/H2/H3**，并给出各自的判别实验。可确定的只有"分叉在插入/建树侧"与"没有证据表明地址补丁改变 DCI 行为" |
| 4 | 前一轮回复里说"同步放在远程仓库 `docs/` 下"，读起来像已交付 | 不准确。实际是：两份文档 untracked、脚本在 `.gitignore` 覆盖的 `experiment/` 下**对 git 不可见**，新 clone 拿不到任何脚本。补齐方案见 §8.1 |
| 5 | —（自查补充） | **已执行 §8.1 选项 A**：脚本迁到 `docs/addr_opt/`、`docs/probe/`、`docs/parse_index_profile.py`，`git check-ignore` 不再命中、`git add -n docs/` 列出 17 个文件；迁移后零 GPU 冒烟测试通过（§1.3）。仍未 commit（遵守约定） |
| 6 | §5.4 结论"默认值仍留 0" | **已被 §5.5 取代**：20 样本 A/B 通过（ΔF1 = +0.04），`ICECACHE_VEC_ADDR` 默认翻为 **1**。同时修正标签：端到端收益口径是 ≈1%，不是单次 A/B 读出的 5.8%（后者被同配置漂移对照证伪） |
| 7 | 搬迁后首次运行直接失败（`set: pipefail: invalid option name`） | 自查 BUG 并已修：Windows 侧 `Path.write_text()` 会写 CRLF，scp 到 Linux 后 bash 拒绝 `set -o pipefail`。全部 `.sh` 已归一为 LF，并加了推送前归一化步骤。**教训**：从此把"推送前检查 CR"固定进流程 |

**同时确认成立、未改动的部分**：§2 的 profiling 归因、§3 的地址公式与其等价性（单元 + 在体）、§4 的向量化实现与"保留 `list[int]`"、§5.1/§5.2 的计时与质量结论、`ICECACHE_VEC_ADDR` 默认关闭。

