第一次指导：你这个分支用于进行实验，做好记录。我们首先运行第一个实验，参考论文原文中Table 1: Accuracy comparison of our method (ICE) with SnapKV (SKV), SteamingLLM (SLM), OmniKV (OKV), MagicPig (MPG), PQCache (PQC), ArkVale (AKV), Full KV (FULL) and groundtruth top-k (TOP-k) on LongBench for Llama-3.1-8B-Instruct and Mistral-7B-Instruct. IceCache generally outperforms other methods across various KV-cache budgets and LLMs.进行复现。我只要求复现完成Llama-3.1-8B-Instructx下ice的部分，即64 ICE 27.4 43.2 55.7 55.3 44.4 31.2 33.4 23.7 26.2 72.5 90.3 41.9 6.6 99.5 61.7 51.6 47.8 128 ICE 30.0 44.7 56.5 55.0 45.4 30.0 33.5 24.3 26.5 73.0 91.3 42.4 6.5 100.0 61.5 56.7 48.6 256 ICE 30.6 44.7 56.3 55.2 45.9 30.6 34.6 24.4 26.7 73.0 92.0 43.5 6.7 100.0 62.5 56.4 49.0这最后三行。服务器有两张A100,80G显存，模型存放在 /opt/model/LLM-Research/Meta-Llama-3.1-8B-Instruct ，脚本已在文件夹存放，你修改后进行实验，实验记录放在 experiment 文件夹下面，尽量不做smoketest，而是直接实验。第一次你先做一个，即Budget=64，Method=ice,Single-Document QA下面三个数据，做完后给我数据。

第二次指导：
续完成缺失的实验，得到三个结果数据。然后开始把所有的测试完成，即Budget为64 128 256时，Method为ICE时候，在Single-Document QA Multi-Document QA Summarization Few-shot Learning Synthetic Code上的效果。
注意要两个gpu同时使用，结果进行记录。


第三次指导：
开始探索DCI带来的开销。
```
现在的关键事实是：

```text
TPOT：169.56 ms
DCI selection：71.45 ms，占 42.14%
每个 token：30 次 layer-level DCI query
```

所以优化 DCI 有三条根本路径：

\[
T_{\text{DCI}}
=
\text{查询次数}
\times
\text{单次查询成本}
-
\text{被重叠隐藏的时间}
\]

对应：

1. 少查几次；
2. 每次查得白些；
3. 不减少计算，但把它藏在 GPU 计算后面。

我认为优先级最高的是“少查几次”。

## 一、把固定跨层复用改成置信度驱动复用

原方法固定每三层查询一次：

```text
Layer 2：查询
Layer 3：复用
Layer 4：复用
Layer 5：重新查询
```

问题是它完全不看当前层是否真的适合复用。

更合理的是：

```text
当前层与 anchor 足够相似 → 复用
不确定或差异明显 → 重新 DCI query
```

置信度可以综合：

- anchor 已经被复用了多少层；
- 上一次 DCI top-k 的分数间隔是否明显；
- 相邻层过去的 page-set overlap 是否长期较高；
- 当前 hidden state 与 anchor hidden state 的变化；
- 当前层是否属于检索敏感层。

需要注意：不同层的 Q 不一定处于完全相同的表示空间，不能简单直接比较 Layer 2 的 Q 与 Layer 3 的 Q。更稳妥的是比较进入 attention 前的共享 hidden state，或者学习一个非常小的跨层校准映射。

优势是：

```text
容易复用的位置：少查
发生明显表示变化的位置：及时刷新
```

它比固定 `n_reuse_layers=3` 更有可能保持精度。

## 二、跨 token 复用，而不只是跨 layer 复用

当前代码虽然会比较前后两个 token 的 page 集合，从而只搬运新增页，但它仍然每个 token 都先做完整 DCI query。

然而连续生成时，经常出现：

```text
token t：选中 [3, 17, 41, 88]
token t+1：选中 [3, 17, 41, 90]
```

page 集合可能高度稳定。我们可以让系统判断当前 Query 是否发生了足够大的变化：

```text
变化小 → 直接沿用上一个 token 的 page 集合
变化大 → 运行 DCI 刷新
```

这是“事件驱动 DCI”：

```text
正常生成：复用
主题切换、换句、推理转折：触发查询
最长每 N 个 token：强制刷新一次
```

便宜的触发信号可以是：

\[
1-\cos(q_t,q_{t-1})
\]

不过这里应当在同一个 layer、同一个 head 内比较，因为这样 Q 才处于同一空间。

为了防止漏掉突然重要的远程内容，需要保护机制：

- 设置最大连续复用 token 数；
- Query 变化超过阈值时立即刷新；
- 标点、换行、代码块切换等结构边界触发刷新；
- 对重要 anchor layer 始终查询；
- sink/window 永远保留。

我认为这比单纯跨层复用更自然，因为同一层相邻 token 的 Q 可直接比较，数学上更干净。

## 三、按 head 决定是否需要 DCI

不是所有 attention head 都承担远程检索功能。

一些 head 可能长期只关注：

- 最近几个 token；
- attention sink；
- 标点或格式；
- 固定局部模式。

对这些 head 做完整 DCI 搜索可能没有价值。

可以把 KV heads 分成：

```text
Retrieval heads：
需要 DCI，负责远程语义检索

Local heads：
只使用 sink + window

Stable heads：
大部分时候复用，偶尔刷新
```

这样不一定减少代码中显示的“30 次 layer 调用”，因为一次 layer-level DCI 调用会处理多个 head，但可以显著减少每次查询内部需要处理的 head 数量。

难点在于 head 类型可能随任务变化。固定离线分类实现简单；在线动态路由更有研究意义。

## 四、GPU 上做粗筛，DCI 只处理少量候选

当前 DCI 是 CPU 上的树结构检索，包含大量不规则内存访问。可以为每个 CPU semantic page 在 GPU 上保留一个很小的摘要：

```text
完整 K/V：留在 CPU
page summary：留在 GPU
```

摘要可以是：

- page centroid；
- 若干代表 Key；
- 每个维度的 min/max；
- 低维投影；
- page 中高范数 Key 的 sketch。

生成时：

```text
当前 Query
→ GPU 与全部 page summaries 做一次批量矩阵运算
→ 筛出少量候选 page
→ DCI 只在候选区域中做精确检索
```

为什么可能有效？

假设 `page_size=16`，每页只保留一个 summary，那么 GPU 摘要大约只有完整 Key cache 的 `1/16`，而且完全不保存 Value。相比完整 KV，额外显存可能是可接受的。

它把 CPU 上的随机访问，部分变成 GPU 擅长的规则矩阵乘法。

主要风险是 centroid 会稀释 page 内的“单个极重要 Key”。因此可以每页保留 2–4 个代表向量，或者使用上界摘要，而不是只存平均值。

这是一个系统改动更大的方向，但很有潜力。

## 五、低维 DCI 搜索，再用原始维度重排

当前 head dimension 通常是 128。DCI 虽然用了随机投影建立索引，但候选检查仍可能涉及原始向量空间和大量 CPU 内存访问。

可以改成两阶段：

```text
128 维 Key
→ 压缩为 16/32 维 search sketch
→ 低维 DCI 找候选
→ 只对少量候选使用原始 128 维精确打分
```

低维表示可以来自：

- 随机投影；
- PCA；
- 每层离线学习的投影；
- 量化后的低维 Key；
- 轻量 learned projection。

这里应当优化的是 top-page recall，而不是重构 Key 的误差。也就是说，投影只要保持 page 排名即可。

它能降低：

- DCI index 内存；
- CPU cache miss；
- 距离计算量；
- CPU→内存带宽压力。

风险是近似误差与 IceCache 本身的 page-level 近似叠加。

## 六、让 DCI 自适应提前停止

当前 DCI 搜索参数基本是固定的，例如 field of view、retrieve proportion 和 search ratio。

但并非每个 Query 都同样困难：

```text
简单 Query：
top 候选明显领先
很早就可以停止

困难 Query：
多个候选分数相近
需要搜索更多节点
```

可以使用 top-k margin：

\[
\Delta = s_k-s_{k+1}
\]

或者搜索过程中候选集合是否稳定，决定是否继续展开树。

这相当于：

```text
容易的 token 少搜索
困难的 token 保持完整搜索
```

它不减少 DCI 调用次数，但降低平均单次调用成本。相比直接缩小统一的 field of view，它更不容易损害困难样例。

## 七、用流水线隐藏 DCI，而不是减少它

代码已经有 layer prefetch 的雏形：

```text
GPU 计算 Layer l
同时 CPU 查询 Layer l+1
```

如果每层 GPU 计算约 1–2 ms，单层 DCI 也在相近量级，那么充分重叠后，可以隐藏一部分 CPU 时间。

可以继续改进：

- CPU 查询与 GPU attention 使用不同缓冲区；
- 双缓冲避免互相等待；
- 提前一层或两层生成近似 Query；
- 消除不必要的 `synchronize()`；
- 为 DCI 固定 CPU core 和 NUMA 内存；
- 多请求 serving 时批量提交 DCI query。

但它有两个局限：

- 下一层真实 Query 依赖上一层输出，提前查询只能使用预测/近似 Query；
- profiling 中的 71 ms 计算仍然存在，只是部分不再进入关键路径。

所以流水线是重要的系统优化，但不如“少查询”根本。

## 我最看好的组合

我会把研究主线定义为：

> Confidence-Gated Event-Driven DCI：在 layer 和 token 两个维度上，只在 page selection 可能发生显著变化时刷新 DCI。

状态可以表示为：

\[
P_{t,l,h}
\]

即 token \(t\)、layer \(l\)、head \(h\) 的驻留 page 集合。

系统每次先做一个很便宜的判断：

```text
置信度高：
P(t,l,h) ← 复用已有 page 集合

置信度低：
P(t,l,h) ← DCI-QUERY(q)
```

其中置信度来自：

```text
同层相邻 token 的 Query 变化
历史 page-set 的稳定性
上次 DCI 候选分数间隔
当前复用年龄
layer/head 的历史检索特征
```

再加三个安全阀：

```text
最多连续复用 N 个 token
每隔 M 层设置强制 anchor
检测到 Query 突变时立即刷新
```

这个方向同时比固定跨层复用更细粒度，也比重写 GPU DCI 更容易从现有代码演进。

如果它能把平均 DCI layer queries 从每 token 30 次降到约 10 次，即使单次查询完全不加速，理论上 DCI 时间也可能从约 71 ms 降到约 24 ms；整体 TPOT 的理论量级可能从约 170 ms 降到约 123 ms。当然，最终值取决于复用判断、同步和不可重叠开销。

我会把方向按优先级排成：

1. 跨 token 的置信度复用；
2. 自适应跨层复用；
3. head-aware DCI routing；
4. adaptive early stopping；
5. GPU page-summary 粗筛；
6. 低维索引；
7. 更深的异步流水线。

前四个方向都尽量保留现有 M-DCI，只改变“何时查、查多少”；后面三个才涉及索引与系统架构的大改。
```
上述是一个agent给出的，你看一下
