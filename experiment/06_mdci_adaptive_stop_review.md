# M-DCI 分 head 自适应停止 —— 对 GPT 实验的复核与下一步设计

> 复核对象：`experiment/05_adaptive_dci_oracle.md`（远程 `yx@100.84.5.13:~/IceCache`）
> 复核方式：读 M-DCI 源码（`/tmp/mdci-patch-final`）+ IceCache 下发参数（`infer_state.py`）+ 独立微基准实测
> 日期：2026-09-11

---

## 0. 一句话结论（含探针实测，2026-09-11 更新）

**GPT 的方向（"进 M-DCI 内部改"）是对的，但它的根因判断和我的第一版根因判断都不对。用 C 层探针实测后，真正的原因浮出水面：**

### 实测三层结论

1. **`num_to_visit` 是死参数**（代码层已证）：`prop_to_visit=1.0` 下它被 `max()` 吞掉，从 12.5% 拉到 100% 耗时平坦。GPT 那条"29.17/29.12/29.11/31.60 ms"曲线就是这个现象。

2. **`prop_to_retrieve` 也不是闸门**（探针实测推翻我的第一版猜测）：把它从 0.2 拉到 1.0，e2/e5 的**调用次数几乎不变**（e2≈21700，e5≈17500），只有 e5 单次耗时从 255ms 涨到 380ms。所以"检索 80% 节点"这个默认值不是决定搜索量的开关。

3. **真正的主因是 e2 全扫描退化**（探针实测，本次新增）：

   | 出口 | 次数（200 次查询） | 时间 | 占比 |
   |---|---|---|---|
   | **e2 全扫描（暴力）** | 21,985 | 75 ms | 18% |
   | e5 候选量预算门（正常 DCI 检索） | 17,566 | 341 ms | **82%** |
   | e4/e6 | 0 | 0 | 0 |

   **分层后 e2 的真相**：`[MDCI-E2LEVEL] level=0 count=21542` —— **e2 全扫描 98% 发生在叶子层（level 0）**。叶子节点平均只有 **78.6 个点**，而查询请求 `num_neighbours=60`，于是 `num_neighbours >= num_points` 的退化条件在叶子层大面积触发，这些节点**完全没走 DCI 的投影检索，直接全排序**。

### 这意味着什么

- GPT 要加的"三个计数器"我已经做掉了，答案是：**e2 暴力退化 + e5 正常检索，各占 18% / 82%，没有别的隐藏刹车**。
- 但"哪个出口耗时"只是表象。**真正该改的是 e2 的触发条件本身**：叶子节点平均 78 个点、请求 60 个邻居，两者的量级太接近，导致 DCI 的投影检索在最后一级基本失效、退化成暴力。
- 所以下一步的落点**既不是 GPT 的"单遍历 active_heads 掩码"，也不是我的"per-head 束宽向量化"**，而是一个更根本的问题：**为什么叶子簇这么小、以及能不能让最后一级仍然走投影检索**。这才是 82% 那部分（e5）能否被压缩的前提。

（§1–§4 保留原分析作为代码证据链和参数敏感性参考；本节是探针实测的增量结论。）

---

## 1. 复核：`num_to_visit` 为什么是惰性参数

### 1.1 代码证据链（五步，每步都可验证）

**① `num_to_visit` 在循环里只出现在一个位置，且被 `max()` 夹住**

`/tmp/mdci-patch-final/src/dci.c:3688`：

```c
int num_points_to_retrieve =
    max_i(query_config.num_to_retrieve,
        (int)ceil(query_config.prop_to_retrieve * num_points));
int num_projs_to_visit = max_i(
    query_config.num_to_visit * num_simp_indices,
    (int)ceil(query_config.prop_to_visit * num_points * num_simp_indices));
```

注意 `num_points` 是**节点局部**的（`dci.c:3657`：`int num_points = point->cell_indices[0].num_data;`），不是全局点数。

**② 终止条件是一个"质量门 + 预算门"的与或结构**

`dci.c:3842`：

```c
if (num_candidates >= num_neighbours &&
    num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
    if (k + visit_proj >= num_projs_to_visit || num_candidates >= num_points_to_retrieve) {
        break;
    }
}
```

也就是说：**必须先把质量门填满**（候选数够、最细层点数够），才轮到预算门说话。而质量门里的 `num_neighbours` 在多层级路径下被传成 `field_of_view`（`dci.c:4099`、`dci.c:4172`），`min_num_finest_level_points` 被设成外层的 `num_neighbours`（`dci.c:4096`）——**两个都和 `num_to_visit` 无关**。

**③ IceCache 主路径又把 `prop_to_visit` 设成 1.0，直接把 `max()` 的第二项顶满**

`~/IceCache/IceCache/source/icecache/infer_state.py:1295`（`_DCI_query`，decode 主路径）：

| 参数 | 下发值 | 在 C 里的含义 |
|---|---|---|
| `num_neighbours` | `n_dci_pages - layer2topk`（budget 64 → **60**） | 要召回多少页 |
| `field_of_view` | `max(seq_len*1e-3, num_neighbours, 30)` → **≈60** | 树搜索束宽 |
| `num_to_visit` | `prev_num_points`（= 全局点数） | 见下 |
| `prop_to_visit` | **`1.0`** | ⇒ `ceil(1.0 × 节点点数 × 1) = 节点点数` |
| `prop_to_retrieve` | **`0.8`** | ⇒ `0.8 × 节点点数` |

代入 ①：`num_projs_to_visit = max(num_to_visit × 1, 节点点数 × 1)`。而 `num_to_visit` 是**全局**点数、按定义 ≥ 节点点数，所以 `max()` 恒取第一项，且恒 ≥ 整个循环的硬上界 `num_points × num_simp × num_comp`。**这个预算门永远不会触发。**

**④ 就算换到 oracle 分支（`prop_to_visit = -1.0`，让 `num_to_visit` 说话），还有第二道刹车**

`infer_state.py:940` 的探针调用保留了 `prop_to_retrieve=0.8`，于是 `num_points_to_retrieve = 0.8 × 节点点数`。要在 OR 的右边生效，得先攒够 80% 节点点数的候选——**这本身就是一次近乎全节点的扫描**。

**⑤ 循环的推进颗粒度远大于任何 `num_to_visit`**

IceCache 构造 DCI 时是 `DCI(head_dim, 1, 1, ...)`（`infer_state.py:638`），即 `num_comp_indices = num_simp_indices = 1`；而 `#define CLOSEST 128`（`dci.c:55`）。循环里 `k += visit_proj`，`visit_proj` 一次最多跳 128，而单个叶子节点的点数通常远小于 128——**一次迭代就把节点扫完了**，`num_to_visit` 没有任何可细分的空间。

### 1.2 微基准实测（独立脚本，不碰 IceCache 仓库）

脚本：`probe/mdci_knob_sensitivity.py`（构建参数与查询参数逐项对齐 `infer_state.py`）
运行：`OMP_NUM_THREADS=32 python mdci_knob_sensitivity.py`，`N=4096` 点/实例，`num_levels=4`，`num_neighbours=60`，`field_of_view=60`，40 次取中位数。

**单位 ms**

| 扫描量 | 取值 → 耗时 | 杠杆 |
|---|---|---|
| **[A] `num_to_visit`**（oracle 分支，`prop_to_visit=-1.0`） | 512→**1.044**, 1024→**1.043**, 2048→0.840, 4096→0.839 | **0（且变小反而变慢）** |
| **[B] `num_to_visit`**（主路径，`prop_to_visit=1.0`） | 512/1024/2048/4096 → 全部 **0.836–0.838** | **0** |
| [C] `prop_to_retrieve` | 0.01→0.760 … 0.80→0.837, 1.0→0.852 | **1.12×** |
| [D] `field_of_view` | 60→**0.837**, 150→1.125, 300→1.426, 600→2.340, 1200→**4.693** | **5.6×** |
| [E] `num_neighbours` | 8→**0.122**, 16→0.214, 32→0.516, 60→0.837, 240→**1.288** | **10.6×** |

**[B] 是决定性的一行**：`prop_to_visit=1.0` 下，`num_to_visit` 从 12.5% 拉到 100%，耗时平坦到小数点后三位——完美复现 GPT 那条"29.17 / 29.12 / 29.11 / 31.60 ms"的曲线。

**[A] 是第二个发现**：在探针真正生效的那个分支里，**把上限压到 12.5% 反而慢 24%**。合理推断（未直接证实）：单节点预算收紧后，上层返回的候选变差，树不得不展开更多节点来补齐质量门，总工作量不降反升。这条推断本身值得单独验一次，因为它意味着"早停"如果做得粗暴，是负收益。

### 1.3 杠杆排序（这就是下一步该动的地方）

```
num_neighbours   10.6×   ← 语义 = 召回多少页（动它等于动预算，需 recall 兜底）
field_of_view     5.6×   ← 语义 = 树搜索束宽（更"纯"的搜索开销旋钮，推荐主攻）
prop_to_retrieve  1.12×  ← 次要
num_to_visit      0×     ← 死参数，实验 05 全押在这上面
```

---

## 2. 对 GPT 实验的四处修正

| # | GPT 的表述 | 复核结论 |
|---|---|---|
| 1 | "`num_to_visit` 是最大上限，搜索很可能已被 `field_of_view` 或其他内部条件终止" | **方向对，机制错**。不是"提前被别的条件终止"，是 `prop_to_visit=1.0` 经 `max()` 把 `num_to_visit` 整个吞掉。区别很重要：前者要改内核，后者改一个 Python 参数就行。 |
| 2 | "12.5%/25%/50% 的页面 recall 是 99.03%/99.96%/100%，说明 head 收敛早" | **结论目前不成立**。三个探针实测耗时相同（微基准 [A]），说明它们并没有真的少搜。测到的是"一次近乎全量搜索"的 recall，不是"提前停止后的 recall"。**"per-head 收敛早"这个假设还没有被数据支持。** |
| 3 | "外部多次调用 58.3 ms > 一次 31.6 ms，所以原型不能直接用" | **同意，但原因要改**。不是因为分段调用有额外开销，而是因为**每一段本身就没有变便宜**（[A]），所以当然叠加两倍。 |
| 4 | "下一步要进 M-DCI 原生查询循环，维护 `active_heads` 掩码" | **可以，但不是唯一路径，且可能是成本最高的一条**。因为 `num_inst = n_kv_heads`（每个 KV head 一个独立 DCI 实例，见 §3.1），per-head 预算是可以在 Python/绑定层做的，不必先动 `dci.c` 的层循环。 |

### 2.1 三个计数器的具体落点（GPT 提的方向，这里给出坐标）

GPT 想加的三个计数，建议直接打在 `dci_query_single_point_single_level`（`dci.c:3642`）里，按**出口分支**统计而不是笼统计时：

该函数有 5 个出口，正好就是"每个 head 实际在哪停"的答案：

| 出口 | 位置 | 含义 |
|---|---|---|
| E1 | `dci.c:3658` `if (num_points == 0) return 0;` | 空节点 |
| E2 | `dci.c:3730` `if (num_neighbours >= num_points)` | 节点点数 ≤ 请求数 → 全排序，退化为暴力 |
| E3 | `dci.c:3756` `if (returned_num[i] == 0) return 0;` | 投影索引一开始就取空 |
| E4 | `dci.c:3844` 左分支 `k + visit_proj >= num_projs_to_visit` | **预算门（访问量）** |
| E5 | `dci.c:3844` 右分支 `num_candidates >= num_points_to_retrieve` | **预算门（候选量）** |
| E6 | `dci.c:3848` 循环自然结束 `k >= num_points*num_simp*num_comp` | 索引耗尽 |

再配三个计数器：`num_candidates`、`num_returned_finest_level_points`、`k` 的终值。**E5 + E6 的占比会告诉你真实刹车在哪**——从 §1.2 的 [C] 看，我押 E5/E6。

---

## 3. 下一步设计：per-head 自适应该怎么落地

### 3.1 好消息：per-head 结构是现成的

`infer_state.py:638`：

```python
self.dci_db[i] = DCI(self.head_dim, 1, 1, ..., num_inst=self.n_kv_heads, ...)
```

**`num_inst = n_kv_heads`——每个 KV head 一个独立 DCI 实例。** 这意味着：

- "不同 head 有不同难度"在**数据结构层面早就被承认了**，实验 05 的观察（少数困难 head 跑到 50%，多数 25% 稳定）与之一致；
- 给不同 head 不同预算，**不需要新增数据结构**，只需要把现在标量下发的 `num_neighbours` / `field_of_view` 变成长度 `n_kv_heads` 的数组，在 `dci_query` 的实例循环里按 `idx` 取值；
- 这是**低风险、可回滚**的改动，不需要动 `dci.c` 的树遍历逻辑。

### 3.2 建议的自适应量：`field_of_view` 优先，`num_neighbours` 次之

- **`field_of_view` 是首选主攻方向**。它是"束宽"，是纯搜索开销；从 [D] 看它 5.6× 杠杆。它的语义风险也最小：**束宽变小只影响候选质量，不改变"要几页"这个预算语义**。
- **`num_neighbours` 杠杆最大（10.6×）但语义最危险**：它就是 `n_dci_pages - layer2topk`，即"召回多少页"。per-head 减小它 = per-head 降低 KV 预算，直接改变模型的可见上下文。只有在能证明"该 head 的 top-60 和 top-20 几乎一致"时才可用——而这恰好是可以用 §3.3 的判据去测的。
- **`prop_to_retrieve` 不建议动**：1.12× 的收益，却要重新标定整个 recall 曲线，性价比最低。

### 3.3 收敛判据：用"边际增益"而不是"两次结果是否相同"

请注意一个实操陷阱：**并行查询结果不保证逐位可复现**（`parallel_level ≥ 1` 时 OpenMP 累加顺序不定；微基准确认过同参数两次调用返回缓冲区尾部不一致）。所以**不能**用"前后两次页集合完全相同"作为收敛判据。

建议的判据（在**同一次遍历内**、跨层检查）：

```
第 l 层新增入选页数 / 当前已入选页数  <  阈值  →  该 head 收敛
```

即每下钻一层，看"新页带来的边际信息"是否已经枯竭。这是单次遍历内的量，天然可复现（层内是确定性的），且与 GPT 想表达的"候选集合是否稳定"等价。

**但要注意层数可能很少**：`promotion_prob = ratio_1 = 0.01`（`infer_state.py:67`、`:638`），4096 点只生成 4 层（微基准实测 `num_levels=[4]`）。**检查点只有 3–4 个，颗粒度太粗**，可能撑不起细粒度的 per-head 差异化。若层数不够，退路是按"每 N 个候选"设检查点，而不是按层。

### 3.4 与"减少 / 隐藏"二分的关系

这条线在竞品地图里的位置需要重新表述：

- FreeKV = **hide**（把 selection 挪出关键路径）；
- OmniKV = **reduce（跨层）**；
- 实验 05 原本想走的"per-head 自适应预算"= **reduce（per-head 粒度）**——**但杠杆押错了参数**。

修正后的表述应该是：**"per-head 束宽自适应"**——同一 token、同一层内，让已收敛的 KV head 主动收窄树搜索束宽，把计算让给困难 head。这是 FreeKV 的全局余弦阈值和 OmniKV 的跨层复用手选 filter layer **都没做**的粒度，依然是一个可站住的差异化点。

---

## 4. 坑与风险（这四条会直接吃掉实验时间）

1. **`field_of_view` 太小会让进程直接死，不是抛异常。**

   `dci.c:4213`：

   ```c
   num_points_to_expand = max_i(min_i(query_config.field_of_view, temp_idx), k);
   if (num_points_to_expand > query_config.field_of_view) {
       perror("Try to increase field_of_view in the query config\n");
       exit(EXIT_FAILURE);          // ← 整个进程退出
   }
   ```

   微基准复现过：`num_neighbours=60` 配 `field_of_view ∈ {2,5,10,20}` 时打印 `Try to increase field_of_view` 后立刻退出。**所以任何"把 `field_of_view` 调小做自适应"的方案，必须对每个 head 保住 `field_of_view ≥ 该 head 实际需要的展开数`**。这也解释了 IceCache 里那句 `max(..., num_neighbours, 30)` 为什么存在。`try/except` 兜不住这个。

2. **返回缓冲区尾部是未初始化哨兵 `0x01010101`（十进制 16843009）。**

   微基准观察：`query()` 返回的 `int32` 数组，`num_returned` 之外的区域填充 `16843009`。Python 侧必须用 `num_returned` 切片。**建议核查 `_raw_dci_to_pages`（`infer_state.py:895`）——它是 `raw_indices.reshape(n_qo_heads, 2, -1)` 直接 reshape，没有先按 `num_returned` 裁剪**，脏值有可能漏进候选集。（未确认，因为需要真实推理路径才能构造出 `num_returned < num_neighbours` 的场景；建议加一条断言即可排除。）

3. **并行结果不可逐位复现**：收敛判据必须避开"比较两次调用的结果"。

4. **`num_inst > 1` 的 Python 侧数据布局与单实例不同**：往 `(n_inst × rows, dim)` 传数据时，只有实例 0 拿到全部点、其余为空（随后查询段错误）。若要先写 per-head 探针脚本，得先解决这个布局问题，或者干脆绕过 Python 绑定、直接写 C 层探针。

---

## 5. 建议的动作顺序（探针实测后更新）

| 优先级 | 动作 | 产出 |
|---|---|---|
| **P0** | ✅ 已完成：6 出口计数器 + 分层计时（见 §0 实测表） | 刹车点已定位：e2 暴力(18%) + e5 正常检索(82%)，e2 集中在叶子层 |
| **P0** | 修正实验 05 表述：删掉"head 收敛早"的结论（证据不支持），保留"分 head 独立判定"作假设 | 避免把未验证假设写进论文 |
| **P0** | **新的主攻方向：诊断"叶子簇为何这么小"**。叶子节点平均 78.6 点、`num_neighbours=60`，量级太近导致最后一级投影检索失效。可调的是 `promotion_prob=0.01`（`ratio_1`）和页大小 16——提高 `promotion_prob` 会减少层数、增大叶子簇，让最后一级重新走投影检索 | 决定 e2→e5 的退化能否被消除 |
| P1 | 复现 [A]"压小 `num_to_visit` 反而变慢 24%" | 确认早停负收益边界 |
| P1 | 若叶子簇问题解决后 e5 仍是主耗，再把 `field_of_view` 向量化 per-head | 第二个加速点 |
| P2 | 核查 `_raw_dci_to_pages` 哨兵值问题（`infer_state.py:895`） | 排除正确性 bug |

---

## 附录 A：复现方式

```bash
# 探针脚本（本地）
probe/mdci_knob_sensitivity.py
probe/results.txt

# 远程运行
scp probe/mdci_knob_sensitivity.py yx@100.84.5.13:/tmp/dci_probe/
ssh yx@100.84.5.13 'cd /tmp/dci_probe && OMP_NUM_THREADS=32 \
  ~/miniconda3/envs/icecache/bin/python mdci_knob_sensitivity.py'
```

## 附录 B：本文引用的代码位置

| 位置 | 内容 |
|---|---|
| `dci.c:3657` | `num_points` 是节点局部点数 |
| `dci.c:3688-3693` | `num_points_to_retrieve` / `num_projs_to_visit` 定义（含 `max()`） |
| `dci.c:3842-3847` | 单层查询的终止条件（质量门 + 预算门） |
| `dci.c:4096` | 多层级路径下 `min_num_finest_level_points = num_neighbours` |
| `dci.c:4097-4099` | 顶层展开调用，`num_neighbours` 位传入 `field_of_view` |
| `dci.c:4168-4172` | 逐节点展开调用，同上 |
| `dci.c:4213-4217` | `field_of_view` 不足 → `exit(EXIT_FAILURE)` |
| `dci.c:55` | `#define CLOSEST 128` |
| `include/dci.h` | `dci_query_config` 结构与字段注释（"terminates whenever ... whichever happens first"） |
| `infer_state.py:638` | DCI 构造（`num_comp=1, num_simp=1, num_inst=n_kv_heads`） |
| `infer_state.py:895` | `_raw_dci_to_pages` |
| `infer_state.py:915-944` | 实验 05 的探针调用（`prop_to_retrieve=0.8` 被保留） |
| `infer_state.py:1295-1306` | decode 主路径的查询配置（`prop_to_visit=1.0`） |
| `dciknn/core.py:399-437` | `query()` 的参数默认值解析 |
