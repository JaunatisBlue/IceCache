# page_scan 的 TTFT 尾巴 — 串行段清理（stash 复用 + H2D 直推）

对应分支 `tail-greedy-cpp`，改动提交 **`d86be89`**，基线 `algorithm@5eaa622`。
`infer_state.py` 的 `_page_scan_stash_put` 文档字符串引用本文 §2。

## 0. 结论

**改动有效，但在合并前必须先声明一项内存代价。**

| 轴 | 结果 | 证据强度 |
|---|---|---|
| **TTFT** | **−200 … −390 ms，中心 ≈ −300**（38/40 行更快） | 实测；此前报的 −152 是低估，不是上限 |
| **输出一致性** | **160/160 逐字段相同**（40 行 × 4 次启动），均值两侧都是 0.4042 | 逐字段 diff，0 处不同 |
| **常驻内存（floor `VmRSS`）** | **+4,251 MB（+4.15 GiB）** — 真实回归 | 必须声明 |
| **峰值内存（`VmHWM`）** | **−500 MB（更低）** | 不是峰值问题 |

**合并条件：把 stash 的内存上界住**（请求结束释放，或给容量封顶）。它把「每问一份工作集」
变成了「永久一份工作集」——在 156 GB 的机器上不是阻塞项，但不能不声不响地合并。

## 1. 改动是什么

三段，都在 `_page_scan_flush` 的路径上，`+88/−5`：

1. **`_page_scan_stash_put`** — 每个 build 层的 K/V 从 `k.clone()/v.clone()`（每层一对新 host 张量，
   16k prompt 下 68 个张量 / 2.2 GB）改为写进**一个跨 prompt 复用的 pageable 缓冲**，返回其视图。
   分配本来就藏在 worker 线程上，但**释放不是**：最后一个引用死在 `_page_scan_flush` 返回时，
   也就是主线程、TTFT 尾巴里。glibc 给每个 32 MB 块 mmap，那次释放是 **68 次 munmap**，
   实测占了一个 **1591 ms 尾巴里的 332 ms**（row 12，hotpotqa）。
2. **`_page_scan_keys_device`** — flush 原本用 `torch.cat([...]).to(device)` 每问现建 390 MB 的
   host cat 再 H2D，结束再释放（56 ms 的分配器开销）。改为复用一块设备缓冲，
   每个 builder 层一次 `copy_(..., non_blocking=True)` 直接推上去。
3. **重置 `_page_scan_stash_used`** 放在写线程池 `join` 之后——所有视图都已随 `pending` 帧消亡。

**为什么这是「尾巴」而不是「body」**：`_finish_prefill` 的尾巴是 100% 串行的，
没有任何东西藏在 prefill worker 后面（§3）。

## 2. 为什么是 pageable 而不是 pinned —— 实测，而且和直觉相反

这一节是代码注释直接引用的那一节，务必保留在这里。

| 方向 | pageable | pinned | 结论 |
|---|---|---|---|
| **DMA 读**（390 MB builder 切片） | 4.7 GB/s | **12.3 GB/s（3.5x）** | pinned 快，值 50 ms |
| **写入**（32.6 MB copy） | **1.5 ms** | 12 ms（**8x 慢**） | pageable 快 |

**在真进程里**，pinned stash 每层在 worker 线程上花 **125 ms**，34 层就是 **4.3 s**，
足以让 worker 变成关键路径，把 row-12 的 TTFT 推到 **8.3 s**。

**stores 才是错的那一边**，所以 stash 是 pageable，DMA 吃下 4.7 GB/s。

> 一个对合并**有承重作用**的细节：`copy_(..., non_blocking=True)` 从 **pageable** 内存出发时
> 是**同步暂存**的（用「调用后立刻覆写源」的方式验证：目的端保留了覆写前的字节）。
> **这个性质是 pageable 专有的——将来若把 stash 改成 pinned，会静默破坏别名安全论证。**

## 3. 两轮验证

### 3.1 先推翻一个把这条线判死的前提：没有衰减

此前有一个说法：尾巴上省的毫秒只有约 35% 能到达 TTFT——依据是一个探针量到 −478 ms 的尾巴缩减，
而 live TTFT 只动了 −152 ms。**这个说法被证伪了。**

恒等式 `ΔTTFT = Δworker_end + Δtail + Δafter_tail` 在 **6/6 行上残差 0.0 ms**，
`tail_start − worker_end` 在 **24 组行测量上是 0.06–0.12 ms**——尾巴完全串行，
没有任何东西藏在 prefill worker 后面。

那个 152 ms 的缺口全部分解为**行选择 + 估计量**：把上一位 agent 的原始数据用
抗争用估计量重新分析给出 **−217 ms**，row 12 处的通过率实测 **1.06**。

> **尾巴里省的每一毫秒，就是一毫秒的 TTFT。这条线是开的，不是枯竭的。**

### 3.2 本次改动本身

- **一致性**：40 行 × 4 次启动 = 160 条记录，prediction / score / generated_tokens / prompt_tokens
  **0 处不同**；两侧均值都是 0.4042。
- **TTFT**：38/40 更快，唯一为负的是最短的那个 prompt。

## 4. 这之后，page_scan 专属的 TTFT 还剩什么

row 12，改动后尾巴 ≈ **1169 ms**：

| 项 | 量级 | 状态 |
|---|---|---|
| greedy | **~638 ms** | 最大项；见下 |
| **CPU 页写 scatter** | **~296 ms** | 另一个 agent 在做（`_page_scan_write`） |
| H2D | ~171 ms | pinned 变体已否 |

**prefill body 不是能追回差距的地方**：它 71% 是两条后端共享的 offload 流水线，
page_scan 自己只占 394 ms（12%）；body 里的大项两边都在付，追不了 page_scan-vs-DCI 的差。

**greedy 的 ~638 ms 是下一个目标**，而且现在有一个明确的算法角度：循环是 ~1005 步**严格串行**的，
但 `row = s · k_j` 本身**不依赖分配状态**，只有 mask 依赖。若种子序列可预知，
所有 row 可以用少数几个大 bmm 一次算完——而种子按范数降序产生，
所以种子集合被限制在范数排序的一个前缀内。这条线已交给一个专门的探索 agent。

## 5. 相关记录

- 台账：`15_work_ledger.md`
- 端到端劣势的分解与盈亏平衡点：`16_merged_head_to_head.md`
- bit-identical 的 greedy 重写：`11_exact_greedy_fast.md`
