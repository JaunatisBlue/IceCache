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

## 5. 实验：直接去掉 host sync 会怎样 → **sync 是承重的**（⚠️ 本节的结论推翻了下面第 6 节的旧建议）

用 `ICECACHE_NO_RECALL_SYNC=1` 做对照（结果不用于质量，只看耗时）：

| | sync 开 | sync 关 |
|---|---|---|
| 单样本耗时 | ~8.4 s | **~17 s（2×）** |
| 进度 | 20/20 正常完成 | **卡在 2/20，14 分钟无进展** |
| CPU 占用 | — | **3030%（30 核满载）** |
| GPU 利用率 | — | **0%（全空转）** |

**机制**：`cpu_transit_buffer` / `cuda_transit_buffer` / `cuda_cast_buffer` 都是**单份共享**的。去掉 sync 后，下一层的 H2D 会在上一层 DMA 仍在读同一段 pinned 内存时覆盖它 → KV 被写坏 → 模型不吐 EOS → 生成到 max length → 2× 慢，CPU 空转把核烧满而 GPU 饿死。

> **结论：`sync` 不是"多余的等待"，而是单缓冲下的正确性保证。** 正确做法是**双缓冲 + event 定序**，不是删 sync。代码已完整还原（与提交版本逐字节一致），补丁留档 `apply_no_sync_patch.py`。

---

## 6. 重新组织的选项与天花板（按可做性排序）

| # | 方案 | 状态 / 天花板 |
|---|---|---|
| 1 | 直接删 sync | ❌ 已测，灾难性（2× + 数据损坏）。见 §5 |
| 2 | **双缓冲（ping-pong）+ event 定序**：`sync` → `wait_event`，只在缓冲被复用时等待 | ✅ **正解**。天花板 = sync 之后本层剩余的 CPU 工作（`page_metadata` 0.22 ms + scatter 发起 + 胶水 ≈ 0.5 ms/层）→ **≈15 ms/token** |
| 3 | **同层内分块流水**：把 173 页分块 gather / 分块 H2D，令 chunk *i+1* 的 gather 与 chunk *i* 的传输交叠 | ✅ 安全。省 min(gather 0.33, 传输 1.6)/层 ≈ 0.33 ms/层 → **≈10 ms/token** |
| 4 | **DCI C 内核**（`native_query` 16.37 ms/token） | 最大的**真实 CPU** 项；需要内核级工作（先把 1.6 ms/次拆成投影/优先队列/排序/去重四段） |

**注意 2 和 3 都要动缓冲结构**（2 需要双份缓冲，3 需要分块缓冲），所以应先做 2，顺带把结构改到位，3 可在同一改动里加。

---

## 7. 一条结构性结论：**"藏"没有空间，只能"缩短"**

层的依赖链是
`hidden(L-1) → q(L) → DCI 选页 → gather → H2D → scatter → attention(L)`，
而 `q(L)` 必须等层 L-1 **完全**算完（含 MLP）。

**所以召回链与模型严格串行，层内不存在可重叠的 GPU 工作。** 这解释了两件事：
- 预取（exp 18）只能靠"用**当前** hidden 经下一层 `q_proj` 投影"的**近似 query**，所以必然付质量代价（topk48 时 −3.65 F1）；
- IceCache 论文里"pipelining hides indexing and retrieval"的叙述，**在这个 workload 上不成立**。

**⇒ 能做的只有让链更短（少传、快查、快搬），没有"把链藏起来"的空间。** 这也是为什么前面所有"隐藏"尝试（async ring、prefetch、batched gather）在这台机器上全部失败——不是实现问题，是结构问题。

---

## 8. 一句话给决策

**优先级：② 双缓冲+event 定序（≈15 ms/token 上限，且是后续一切流水化的前提）→ ③ 分块流水（≈10 ms/token）→ ④ DCI 内核（16.37 ms/token，最大但需动 C）。**
不要单独删 sync（§5 已证伪）；也不要指望"重叠掉传输"（§7 已证无空间）。

