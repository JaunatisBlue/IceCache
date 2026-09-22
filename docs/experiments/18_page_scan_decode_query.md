# page_scan 的 decode 查询路径 — 别名化、去 nonzero、query 常驻设备

对应分支 `explore/tpot-decode`，改动提交 **`6a229dc`**，基线 `algorithm@5eaa622`，
合并 `257e0d8`，文档修正 `e376584`。**合并前已由一个独立 agent 逐条反驳式验证。**

## 0. 结论

**有效，可以合并。这是 decode 轴上目前最干净的一次改动。**

| 轴 | 结果 |
|---|---|
| **TPOT** | **−4.251 ms/token（合并 3v3）**：88.910 → 84.659 ms，比值 **0.9522**，95% CI **[−6.430, −2.072]**，40 行里 29 行更快，符号检验 **p = 0.0032** |
| **配对一致性** | **9/9 全负**（−3.308 … −5.242 ms，比值 0.9412–0.9626） |
| **范围** | base **[88.499, 89.126]** vs branch **[83.885, 85.191]** — **不相交** |
| **输出一致** | **每一个配对都是 40/40 逐字段相同**（prediction / generated tokens / score）；6 个臂共 254 token、均值 0.4042 两侧一致 |

**同代码空跑**（同字节、不同运行）：base +0.628 / +0.606 / −0.021；branch +0.288 / −1.018 / −1.306 ms。
最坏空跑 **1.31 ms** ⇒ 效应是最坏空跑的 **3.2x**。

**臂序不是伪影**：9 个配对里包含一对抗衡序（branch 先跑），给出的是**最大**的 −5.221 ms。
**最重的两个负载配对恰好是两个最小效应**——正好是争用会推向的方向——所以 4.25 ms 若是偏，是**偏保守**。
verifier 自己的判断：方向已定，量级可信到 **±1 ms** 左右。

## 1. 改了什么（四处）

1. **query 常驻设备**：`infer_state.py:1449-1452`，只有 `retrieval_backend != "page_scan"` 才 `.cpu()`。
   此前每层都 D2H 下去、下一行又立刻 H2D 回来。
2. **别名层不再 clone**：`nr = self.prev_nr; rids = self.prev_rids`（原来是 `.clone()`）。
3. **`_query_constants` 不再缓存 built/unbuilt mask**，改为每次从 `_bias_t` 现推。
4. **`_scan_ops` 抽出**，把 `out[rows[m], pos[m]] = flat[m]` 的布尔形式
   （会降级成**三次 `torch.nonzero`**，而 CUDA 上每次 nonzero 都要 host 同步来定输出大小）
   换成**写进 `budget+1` 列、再切片**。

## 2. 最危险的一条：别名化（重点验证对象）

34 层里只有 12 层真查，**22 层是别名**，而且 `prev_nr`/`prev_rids`/`prev_eids` 在**相邻层之间链式传递**。
只要有任何下游**原地写**它，错误就会沿链放大。verifier 用它自己的话说「这是我攻得最狠的一条」，它站住了：

- **穷举静态追踪**：`prev_nr|prev_rids|prev_eids` 全仓库只在 `infer_state.py:1454-1456`（仅 anchor 分支）被写；
  `245-247`/`405-407` 初始化为 `None`；只在 `1470-1471`（`prev_eids`，在 `torch.where` 里，非原地）和 `1478-1479` 被读。
  **没有任何 `+=`、`.fill_`、`.copy_`、`scatter_` 或下标赋值落在这些对象上。**
- **`recall()`（`1352-1418`）在 `source != 0` 时取 `_recall_cpu[source]`（`1382`），根本不求值 `rids`/`nr`**
  ——这两个名字只出现在 `source == 0` 分支（`1384-1387`），而那也是**唯一**写 `_recall_cpu[...]` 的地方。
  所以别名只读它 anchor 的条目，**不可能构造出自引用**。
- **`_apply_selected_pages` 只在 `_DCI_query` 的四处被调**，而 `_DCI_query` 只在 `reuse_id == 0` 分支被调 → **别名永远到不了它**。
- `_cpp.scatter_pages` 的 kernel 收的是 `const int32_t *__restrict__`，**只读**。
- **运行时审计**（真实模型，hotpotqa 0-2 行）：在每个 anchor 处快照 `nr`/`rids` 的 clone，在每个消费者处比对内容，
  并持有强引用以防 id 复用：

  ```
  consumer calls: {'recall': 168, 'alias': 308}
  tensor objects snapshotted: 336      objects produced at anchor AND consumed: 336
  layers only anchoring: 12; only aliasing: 22
  ANOMALIES: 0
  ```

  308 次别名 recall = 22 别名层 × 14 decode 步；336 = 2 张量 × 12 anchor × 14 步。**逐字节相同。**

另外：`self.prev_nr = nr` 是**重新绑定**而不是原地改，所以即使下一个 anchor 插进来也不可能回溯污染别名已经持有的张量。
**那两个 `clone()` 在语义上确实是空操作。**

## 3. 等价性：`_scan_ops` vs 原序列

verifier 写了 `old_ops` 作为 base `_query_device` 序列的逐字转写，并**在 base 上跑真实的 `_query_device` 来校验这份转写**。
两个 worktree 给出**相同摘要**，且 `old == new` 在所有用例上成立：

```
direct: all-reject row / all-accept pos>budget / dup ids / W==budget / H=1 / mixed rows   same=True
real:   H=8 r=4 B=16 | n_built=40 | partial n_built | r=1 | H=1 | B=64 | non-finite q     same=True
repeat-call mismatch on a shared `first` buffer: 0/5
ALL SAME: True (14/14)
```

**为什么严格等价**：`keep & pos<budget` → `idx=pos, src=flat`；`keep & pos>=budget` 与 `~keep` 都被指向列 `budget`、
`src` 被掩成 0、再被切片丢掉。**唯一能让 CUDA `scatter_` 未定义的重索引情形（重复下标）只可能落在被丢弃的 `budget` 列上。**
`idx` 不会为负（`keep` 处 `pos >= 0`，否则掩成 `budget`）。

顺带查掉两个「看起来会坏但没坏」的点：`out[:, :budget]` 是 gap-strided 视图（`stride = budget+1`），
但 `.to(torch.int32)` 返回**C 连续**数组（numpy strides `(budget, 1)`，`C_CONTIGUOUS == True`），
非连续性没有泄漏到 `_apply_selected_pages`；`c["rows"]` 删除后**没有其他读者**（grep 为空）。

## 4. ★ 对作者归因的更正

作者说「12 个 anchor 层/token，**所以**收益来自这里」。**这是错的。**
按真实形状实测拆开：

| 分量 | 每 token |
|---|---|
| D2H+H2D 消除（claim 1） | **0.710 ms**（17%） |
| 别名 clone 消除（claim 2） | 0.44 ms |
| `_scan_ops` 去 nonzero（claim 4） | **1.31 ms**（最大单项） |

即隔离探针解释了 **~60%** 的 live 效应（2.46 / 4.25 ms），**最大的一项是去 `nonzero`，不是 D2H。**
剩下的部分是重叠/派发——这是本项目反复出现的模式。

## 5. 顺带确认/遗留

- **claim 3 是加固而非修 bug**：旧代码其实也不易 stale——唯一不 bump `_bias_ver` 的写（`build` 里的 `578-580`）
  在每次查询**之前**发生**且**改变张量身份。这次改动把「只要将来每个写点都记得 bump」变成「构造上正确」。
  而且**decode 期间根本没有东西写 `_bias_t`**（`insert` 只从 `_page_scan_flush` 跑，即 prefill）。
- **读序遗留（未触发）**：`_scan_ops` 在 bmm **之后**读 mask，而旧的缓存 mask 在 bmm **之前**填充；
  `insert` 先发布 `_reps_t` 后发布 `_bias_t`，交错的读者可能把「新鲜的 bias」与「还是占位零的 representative」
  配到一起 → 给一个没有 K/V 的页打出 0.0 分。**当前不可达**：`insert` 只在 `_page_scan_flush` 里跑，
  且需要 `n_prefetch_layers > 0`，两个入口都是 0。已在 `_scan_ops` 里留注释，未改代码。
- **模块/section docstring 的两处过期陈述**已由 `e376584` 修正（纯文档；该提交的 AST 在剥掉 docstring 后
  与修改前**逐字节相同**，sha256 `ca8f7bf3…`，所以合并所依赖的验证不受影响）。
- **CUDA graph 仍未做**：`107 ms` 捕获成本与 `~180` 次调用盈亏平衡点**未复测**（UNVERIFIED），
  但它支撑的是一个「不做」的决定，不阻塞合并。

## 6. 位置

合并后 TPOT 相对 DCI 的比值应随之改善（DCI 臂不受影响），但**这是推算**：
`0.813 × 0.9522 ≈ 0.774`。**要引用必须在同一交错协议下重测**——
与报告 `17_` 的 −0.3 s 一样，都记在台账第四部分的过期警告里。
