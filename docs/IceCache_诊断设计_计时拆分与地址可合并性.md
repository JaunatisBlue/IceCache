# 联合诊断设计：计时拆分 + 实际地址可合并性

> 状态：**待批准执行**（按 codex 裁定，只批准这一项，不三线并跑）
> 目的：一次小运行回答"系统方向（子树感知 gather / head-major 布局）值不值得继续做"
> 前置事实：见 `IceCache_三条推荐方向_传输侧.md` §0（`addr(i,j) = _base + i*offset + j*stride`，`stride` 含 `n_kv_heads`）

---

## 0. 一次运行要拿到的两组数据

| 组 | 内容 | 输出 |
|---|---|---|
| **A 计时** | CPU 地址准备 / `copy_to_buffer` / H2D / cast / `synchronize()` 残差 | 追加进 `DCI_PROFILE` |
| **B 地址** | 每 `(layer, head)` 本次选中的 leaf id、源地址、(可选)parent id | `.npz` 落盘 |

两组在同一次运行里取，用 1–2 个现有样例（qasper 20 样本里的 1–2 条），不新增实验矩阵。

---

## 1. 改动原则（**必须遵守**，因 `infer_state.py` 正被 codex 并行修改）

1. **纯附加**：只新增 event record、计数器、dump 分支；**不改任何既有控制流与数值逻辑**。
2. **env 门控**：全部由 `ICECACHE_DIAG=1` 与 `ICECACHE_DIAG_DUMP=/path/out.npz` 控制，默认关闭 → 关闭时行为与现在**逐字节一致**。
3. **不新增同步点**：只 record event，**不在热路径加 `synchronize()`**；读取 elapsed 只在已有的 `recall_wait` 同步点之后做。
4. **不碰 C 内核**：本诊断只改 Python 层；parent id 若接口不便，第一阶段先不取。

---

## 2. Patch 设计（`infer_state.py`）

### 2.1 计时（A 组）

已有两段可直接复用：
- `recall_gather`（L1259 起 → L1281 结算）= **CPU 地址准备 + `copy_to_buffer`**
- `recall_wait`（L1380 → L1386）= **到同步点时的残差等待**

新增三段（CUDA events，全部在 `c2g_stream` 上）：

```python
# 在 recall() 内，env 门控
if self.diag_enabled:
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)

with torch.cuda.stream(c2g_stream):
    if self.diag_enabled: e0.record(c2g_stream)
    dst = self.cuda_transit_buffer[:, : 2 * n_transit_pages, :]
    src = self.cpu_transit_buffer[:, : 2 * n_transit_pages, :]
    dst.copy_(src, non_blocking=True)
    if self.diag_enabled: e1.record(c2g_stream)          # H2D 段
    self.cuda_cast_buffer[:, : 2 * n_transit_pages, :].copy_(dst, non_blocking=True)
    if self.diag_enabled: e2.record(c2g_stream)          # cast 段

# 在已有的 recall_wait 同步点之后（不新增同步）读取
if self.diag_enabled and recall_wait_start is not None:
    self.diag_h2d_ms   += e0.elapsed_time(e1)
    self.diag_cast_ms  += e1.elapsed_time(e2)
```

**同时把 `recall_gather` 细分为两段**（地址准备 vs `copy_to_buffer`），在现有 `gather_start` 与 L1273 之间加一个 `perf_counter()` 分隔点即可（纯 CPU 计时，无需 GPU）。

最终五段：`addr_prep` / `copy_to_buffer` / `H2D` / `cast` / `wait_residual`，且 `addr_prep + copy_to_buffer ≈ 现有 recall_gather`（可做自洽校验）。

### 2.2 地址 dump（B 组）

在 `recall()` 的地址填充循环（L1267-1269）内，门控收集：

```python
if self.diag_enabled:
    self.diag_records.append(dict(
        layer=layer_idx, head=i,
        leaf_ids=rids_cpu[i, :nr_cpu[i].item()].numpy().copy(),
        addrs=self.page_address_buffer[layer_idx][b, i, rids_cpu[i, :nr_cpu[i]].numpy()].copy(),
        step=self.diag_step,
    ))
```

- 用 `ICECACHE_DIAG_MAX_RECORDS` 限流（例如只记前 200 次召回），避免撑爆内存。
- 运行结束（或收到 SIGINT）时 `np.savez` 到 `ICECACHE_DIAG_DUMP`。
- **建议只跑单层或少数层**（用现有 layer 白名单开关）以进一步缩量。

### 2.3 parent id（可选，第二阶段）
若容易拿到 leaf→parent 映射则一并存；否则**先用 leaf id + 地址**跑完，因为：
- **地址等差 ⟺ leaf id 连续**（§0.1 的 stride 结构），所以**只靠 leaf id 就能算可合并性**，parent id 不是必需的。

---

## 3. 离线分析（本机，无需 GPU）

对每条记录、每个 `(layer, head)`：

| 指标 | 定义 | 含义 |
|---|---|---|
| `n_sel` | 选中叶子数 | 当前 gather 的传输次数 |
| `n_runs` | leaf id **连续 run** 数（排序后 `id[i+1]==id[i]+1` 合并） | **可合并的 strided 段数** |
| `merge_ratio` | `1 - n_runs / n_sel` | 理论可合并比例 |
| `cross_head_hits` | 同一 leaf id 被 ≥2 个 head 选中的次数 | 该 leaf block 内可整段搬运的量 |
| `addr_span` | 地址 max−min | 判断是否被 stride 拉得很散 |

**关键判据**：
- `n_runs << n_sel`（`merge_ratio` 高）→ 子树/gather 方向成立；
- `n_runs ≈ n_sel` 但地址有聚集 → 被 **stride 布局**阻碍 → 研究 head-major；
- `n_runs ≈ n_sel` 且无聚集 → **放弃布局主线**。

同时对照 A 组：
- `H2D` 段占比高 → 布局优化天花板低，**转向动态减少 k**；
- `copy_to_buffer` 段占比高 → 布局优化（减少碎片）有空间。

---

## 4. 产出与工时

| 项 | 产出 |
|---|---|
| patch | `infer_state.py` 的 env 门控补丁（纯附加） |
| 运行 | 1–2 样例、记录限流、单/少数层 |
| 分析 | `analyze_diag.py`（读 `.npz`，输出 §3 表 + 结论判定） |
| 结论 | 二值化：**继续布局主线 / 转 head-major / 放弃布局 / 转动态 k** |

预估：patch 0.5 天，运行 < 1 小时，分析脚本 0.5 天。

---

## 5. 执行前检查清单

- [ ] 与 codex 确认 `infer_state.py` 当前工作面（避免冲突）
- [ ] 确认 `ICECACHE_DIAG=0` 时与 baseline 结果**一致**（无回归，用现有 qasper 锚点 F1=45.48）
- [ ] 确认记录限流与层白名单已设，避免 OOM / 磁盘撑爆
- [ ] 确认不新增同步点（grep 补丁里无新的 `synchronize()`）
