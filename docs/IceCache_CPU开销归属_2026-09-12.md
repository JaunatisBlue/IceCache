# CPU 侧开销归属（profiling，2026-09-12）

> 目的：不再靠猜，也不再换数字试——用 profiler 把 IceCache 解码侧的开销钉死。
> 工具：`experiment/run_cprofile.sh`（cProfile）、`roundtrip_latency.py`（微基准）、`experiment/run_perf_profile.sh`（perf，符号未解析，弃用）。
> 配置：qasper、3 样本、budget 64、topk 0、reuse 3、FP16 recall、promotion 0.01。
> **注意**：cProfile 会给每次函数调用加开销（9.4M 次调用），**绝对值被放大，只能看比例**。

---

## 0. 结论：最大的单项**不是 CPU 计算**，而是暴露的传输等待

```
_icecache_decode（解码侧 Python 总工作 = 13.37 s，占全程序 75.2 s 中的一部分）
├─ estimate_select_recall ................ 10.88 s   (81%)
│  ├─ c2g_stream.synchronize() ...........  5.05 s   (46%)  ← 纯 idle 等待
│  ├─ _DCI_query（含 C 内核） .............  2.44 s   (22%)
│  │   └─ dciknn C 扩展 ..................  1.87 s   （tottime 0.03 s → 98% 在 C 里）
│  ├─ recall()（gather + 发起 H2D） .......  1.67 s   (15%)
│  └─ _estimate_select_recall_impl 自身 ...  0.81 s   (16%，纯 Python 字节码)
└─ attention / scatter / 其余 .............  2.49 s   (19%)
```

**含义**：
1. **`recall`/`select` 这条链占解码侧 81%** —— 这就是要动的全部。
2. 其中**近一半（46%）是 `c2g_stream.synchronize()` 的 idle 等待**，不是 CPU 消耗，而是**传输没被重叠**（与 exp 18 的结论一致：预取也救不了）。
3. **真正的 CPU 计算里，最大的是 DCI 的 C 内核查询**（10 次/token，约 1.6–1.8 ms/次 → 16–18 ms/token，占 TPOT 约 13%）。
4. Python 侧仍有 **0.81 s 的函数自身字节码**（3120 次调用 → 约 0.26 ms/次）——这是 `FASTADDR` 之后剩下的纯 Python 开销。

---

## 1. 被证伪的两个假设（记录下来，避免重复走）

| 假设 | 量测 | 结论 |
|---|---|---|
| **小传输往返延迟大**（"0.217 ms/层 写 2 KB 说明延迟主导"） | 2 KB pinned H2D/D2H = **13 µs**；8 KB = 22 µs；`torch.tensor(np(8,60), cuda)` = 23 µs；30 层 × 1 次 = **0.39 ms/token** | ❌ **证伪**。往返延迟只有 13 µs 量级，不是瓶颈 |
| **perf 能给出符号级热点** | `perf record` 输出全是 `[unknown]` + 裸地址；DSO 只有一个 `[unknown]` | ❌ 该环境 perf 符号不可用（paranoid=2、无 debuginfo）。改用 cProfile |
| **valid_cache：同组 3 层重复算 `get_valid_entries`**（上一轮） | A/B：135.09 vs 135.31 ms，−0.16% | ❌ 证伪（成本在目标 stride 写） |

---

## 2. CPU 侧成本清单（DCI_PROFILE，`fastaddr` 生效后，ms/token）

| 项 | ms/token | 占 TPOT | 可动性 |
|---|---:|---:|---|
| `recall_wait`（H2D 等待） | **45.06** | 34% | 带宽受限；唯一出路是少传（exp 18/19 已穷尽） |
| `native_query`（DCI C 内核） | **16.37** | 12% | **最大 CPU 项**，在 C 里 |
| `recall_gather` | 9.87 | 7% | 已由 FASTADDR 减半；余下大部分是真实 memcpy（42 MB/token） |
| `page_metadata` | 6.52 | 5% | 成本在目标 stride 写（已证） |
| `query_postprocess` | 4.26 | 3% | Python/NumPy reshape + ascontiguousarray |
| `index_update` | 3.79 | 3% | 树插入 |
| `query_diff` | 1.84 | 1% | — |
| `query_mapping` / `d2h` / `dedup` | 0.80 / 0.59 / 0.29 | 1% | — |

---

## 3. 下一步的三个可动项（按"是否真的能减"排序）

### ① 去掉 host 侧同步：让 `scatter_pages` 与 H2D 同流（**收益最大，风险中**）

`recall()` 在发起 H2D 后立刻 `c2g_stream.synchronize()`（30 次/token），把 CPU 卡在 Dma 完成上——这正是 46% 的 idle 来源。既然下游 `scatter_pages` 是 CUDA op，可以把它放进 `c2g_stream`，用**流内顺序**代替 host 同步：CPU 不再等待，等待自然变成 GPU 侧排队。
- **预期**：把那 46% 的 idle 从 host 侧移走（能否转成收益取决于下游是否真的依赖它）。
- **风险**：流语义 + `transit_buffer` 复用需要仔细核对；`cuda_cast_buffer`/`scatter`/`decode_sdpa` 的流依赖必须重排。
- **必须先验证**：`ICECACHE_DIAG` 已能给出五段计时，可先量化 sync 在关键路径上的占比。

### ② DCI C 内核：先量清楚 1.6 ms/次花在哪，再动（**收益确定，代价是动 C**）

现有信息：`num_to_visit` 杠杆 0×（被 `max()` 吞掉，因为 IceCache 传 `num_to_visit=prev_num_points` 且 `prop_to_visit=1.0`）、`field_of_view` 5.6×、`num_neighbours` 10.6×；C 探针显示 e5（候选预算）82%、e2（叶子全扫）18% 且 98% 在 level 0。
**下一步不是调参，而是把 1.6 ms 拆成"投影 / 优先队列 / 排序 / 结果去重"四段**（在 `dci.c` 里加计时，用已验证可用的独立构建路径），找出真正的大头再动手。

### ③ `_estimate_select_recall_impl` 的 0.81 s 纯 Python 字节码（**安全，收益小到中**）

3120 次调用 → 0.26 ms/次的 Python 开销。逐步核对：per-layer 的 tensor→cpu、clones、`torch.sum`、属性查找。这类改动零风险、可 A/B，但单项只值几 ms/token。

---

## 4. 复现命令

```bash
bash experiment/run_cprofile.sh        # cProfile 归属
python roundtrip_latency.py            # 小传输往返延迟微基准
bash experiment/run_perf_profile.sh    # perf（本机符号不可用，留档）
```

## 5. 一句话给决策

**如果只做一件事：做 ①（去 host 同步）。** 它对着的是解码侧最大的单项（46%），而且和 exp 18 的发现自洽——这个系统里传输**从来没被重叠过**，预取失败是因为把 CPU 工作塞进了同期，而不是"重叠"这个方向错了。①是更干净的实现方式。
