# Unified RAG × IceCache — 实现分工与接入说明

对应设计文档：`UNIFIED_RAG_ICECACHE_IMPLEMENTATION.md`。
本文件只描述已落地的接缝（seam），不改动 codex 正在维护的运行时文件。

## 文件所有权

| 文件 | 维护者 | 状态 |
| --- | --- | --- |
| `source/icecache/materialization.py` | Claude | 已完成，309 项自检 |
| `source/icecache/materialization_provider.py` | Claude | 已完成，111 项自检 |
| `source/icecache/prefill_materialization.py` | Claude | 已完成，108 项自检 |
| `source/icecache/boundary_materialization.py` | Claude | 已完成，83 项自检（deferred 排空 + stage-B append + 提交后失败 doom + 跨会话无锁 encode） |
| `source/icecache/retrieval.py` | codex | 已导出 canonical `CandidatePlan`，并修复 int32 cast 前范围校验 |
| `source/icecache/infer_state.py` | codex | provider 钩子已存在（`_DCI_query`，约 902 行）；待接 deferred 排空与请求边界 |
| `source/icecache/experimental/kv_materialization.py` | Claude | 已完成，57 项自检 |
| `source/icecache/experimental/unified_page_index.py` | codex | 已完成选择器 |

`_DCI_query` 只由 codex 修改。本文中的接线方式不要求改 C++ scatter，也不要求让
native 层接收字符串 page ID。

## 已交付的控制面 API

### 1. Session 物化状态机

```python
from icecache.materialization import (
    GlobalRAGPageStore, RAGMaterializer, chunk_document, assemble_prompt,
    build_candidate_plan,
)

store = GlobalRAGPageStore()
pages = chunk_document(token_ids, document_id=..., chunk_size=...,
                       descriptor_version=..., descriptor_fn=embed)
store.add_many(pages)

mat = RAGMaterializer(store, n_kv_heads=n_kv_heads)
```

- 状态：`descriptor_only → queued → encoding → cpu_kv → gpu_kv`，失败为
  `failed`；rollback 清空 token span 和逐层 leaf map。
- `begin_prompt(session_id, prompt, page_ids=None)`：在请求边界记录每页的绝对
  prompt 区间并进入 `encoding`；已经在 `cpu_kv/gpu_kv` 的页会报错，而不是复用
  跨上下文 KV（设计文档 §7：每个请求独立 session）。失败/中断回滚覆盖两类页：
  本次新 submit 的页，以及**调用前已处于 `queued`、被本次 begin_encoding 移入
  `encoding`** 的页（调用前就在 `encoding`、属于别的请求的页不动）。回滚依据
  调用前的状态快照，**不依赖 `submit` 的返回值**：`KeyboardInterrupt` 落在
  `submit` 内部、返回值丢失时，已入队的新页仍会被回滚。快照 + submit +
  begin_encoding 在同一把可重入锁内完成，并发线程不会在两步之间插入。
- `register_layer_leaves(session_id, page_id, layer_idx, token2node, num_leaves,
  *, token_offset)`：`token2node` 可以是 2-D 数组，也可以是后端返回的
  `(token2node, ...)` tuple；`token_offset` 是 **本次注册** 的 DCI token 0
  对应的绝对 prompt 位置，**没有默认值**（默认 0 会在偏移为正时静默映射
  错误 token）。
- `resolve_head_leaves(session_id, page_id, layer_idx, head_idx)` 返回 1-D
  `int32` native leaf。页面**未物化**时返回 `None`；页面已物化但该层 map
  缺失（例如刚 `invalidate_layer`）时返回**空数组**，表示该层零贡献。这就是
  唯一的交接点。`head_resolver(session_id, layer_idx)` 返回供 plan 使用的
  `(page_id, head) -> leaves` 闭包：首次访问某页时一次性取出整张 map 并缓存
  各头行（一个 plan 内缓存；之后的注册/失效对下一个 resolver 生效）。返回的
  行是只读视图，只能消费不能原地修改。子类覆写 `resolve_head_leaves` /
  `resolve_page_leaves` 时自动退回逐头路径。已注册但为空的 `[n_heads, 0]`
  map（`require_full_span=False` 且整块 chunk 在 span 外）在批量和逐头两条
  路径都返回零长度行，该页仍算 ready、零贡献，不会 deferred。
- `page_is_ready(session_id, page_id)`：判断页面 KV 是否已物化。用于把
  “未物化”和“已物化但本层无 map”区分开；后者绝不能进入 deferred 通道
  （deferred 会让已编码页再次 `begin_prompt` 并失败）。
- `build_candidate_plan(is_page_ready=...)`：已知未物化的页**不再调用
  resolver**，直接 deferred；若所有选中页都未物化，连 merge 也跳过，直接把
  校验过的 fallback 作为 `ready_leaf_ids` 返回（结果等价，见 §3 的耗时）。
  注意：`is_page_ready` 返回 False 时以物化状态为准，resolver 即使返回数组
  也不会被采用（fail closed）。
- `invalidate_layer(session_id, layer_idx)`：DCI add 重组旧叶后，丢掉该层旧
  map，页面 KV 状态与 span 保留；随后用新的 `token2node` 重新注册。
- `retry(session_id, page_id)`：把 `failed` 或 **中断在 `encoding`** 的页原子
  地退回 `queued`（清空 span 与 leaf map）。否则中断的编码页既不能 `submit`
  也不能 `queue`，只能显式 `fail()` 才能恢复。
- `release_session(session_id)`：清理元数据；CPU/GPU KV 和 DCI 资源的实际释放
  仍由运行时负责。
- `GlobalRAGPageStore.search`：精确 cosine 文本召回，分数按 `page_id` 打破平局，
  无描述符的页跳过。每个页的描述符范数在插入时缓存（描述符不可变），查询时
  只做一次 dict 查找；本机 20k 页 × 1024 维实测 **170 → 74 ms/query**
  （2026-10-22，CPU）。`remove` 会同步删除缓存，缺失时查询路径会重算并回填。
  `top_k=0` 在维度校验之前短路返回空列表（此时坏描述符不会报错）。
  当页数 × 维度 ≥ 65,536 时改走堆叠矩阵路径：首次查询把描述符 `np.stack`
  成 `[n, d]` 矩阵（20k × 1024 实测一次性 128 ms、常驻 +160 MB），之后每次
  查询一次 BLAS matvec + `lexsort`，**6.75 ms/query**（同机 `numactl` 绑核，
  6 次中位数；逐页参考实现 179.9 ms，含范数重算）；top-8 与逐页参考完全相同，
  最大分数差 4.2e-17（BLAS 累加顺序）。带 `allowed_page_ids` 子集时 1.23 ms。
  矩阵在**实际插入或删除**时失效，幂等 `add` 保留；小 store 仍走逐页路径，
  分数逐位不变。

### 2. 请求边界编排

```python
from icecache.prefill_materialization import PrefillMaterialization

# 推荐放进 with：异常或忘记 finish 时自动 abort，不会把页留在 encoding。
with PrefillMaterialization(mat, session_id) as request:
    # 方式 A：调用方已经选好页（例如 codex 的统一选择器）。
    prompt = request.prepare(
        pages=selected_pages,
        prefix_token_ids=prefix_ids,
        suffix_token_ids=suffix_ids,
    )
    # 方式 B：请求边界直接做文本级 RAG 召回（stage 2 前半段）。
    # prompt = request.prepare_from_query(
    #     query_descriptor=embed(query_text), top_k=8,
    #     prefix_token_ids=prefix_ids, suffix_token_ids=suffix_ids,
    #     min_score=None, allowed_page_ids=None,
    # )
    # request.retrieval_hits 是本次实际拼进 prompt 的 (page_id, cosine) 列表。
    # 用 prompt.token_ids 正常 prefill；不在 decode 中编码
    for layer_idx, (token2node, num_leaves) in indexed_layers.items():
        request.register_layer(layer_idx, token2node, num_leaves,
                               token_offset=dci_token0_prompt_position)  # 必填
    # 默认 require_full_span=True：跨 sink/window 边界的部分页会 abort 到
    # fallback；仅在确实只映射被 offload 的 token 时传 False。
    request.finish()          # 成功
    # request.abort(reason)   # prefill/注册失败：整请求回滚为 failed
```

`register_layer` 任一层失败会 abort 整个请求并重新抛出；生成侧随后只看到
fallback，不会看到半套 leaf map。

**阶段 B（append）**：agent tool boundary / 轮次边界把新检索到的 chunk
追加到**当前上下文**，而不是重建 prompt：

```python
request = PrefillMaterialization(mat, session_id)
prompt = request.prepare_append(pages=selected_pages,
                                position_base=current_context_len)
# 只 prefill prompt.token_ids（追加块本身）；position ids 从 position_base 起。
# 注册时 token_offset 仍是会话首次 prefill 的 n_sink_pages * page_size：
# DCI token 0 不随 append 移动。
request.register_layer(layer_idx, token2node, num_leaves,
                       token_offset=dci_token0_prompt_position)
request.finish()
```

`prompt.token_ids` 只含追加的 token，但记录下来的 span 是会话绝对位置
`position_base + [start, end)`（`begin_prompt(..., position_base=...)`）。
`prepare_append` 要求 `position_base` 非负整数；同一页在同一会话重复编码仍
按 fail-closed 拒绝。append 后的 position ids / attention mask / KV 页序列
必须与“正常把 chunk 放进上下文”等价，这是运行时职责，仍需 live 实验
（设计文档 §7 阶段 B）。

`prepare`/`register_layer`/`finish`/`abort` 由内部 `RLock` 串行化，允许
prefill 线程与取消线程并发调用：`abort()` 先于 `prepare()` 也不会丢——prepare
会拒绝启动；`abort()` 在 `begin_prompt` 执行期间到达时，prepare 会把刚落地的
页标为 `failed` 并抛出。`abort()` 每次都重扫已开始的页，不在已 aborted 时
提前返回，因此不会留下卡在 `encoding` 的页。

运行时必须 prefill **`prompt.token_ids` 本身**（不要再用自己的模板另拼一份）：
token 区间是在 `prepare` 时按这份 ids 记录的，任何额外前缀/后缀都会让
`token_offset` 与 leaf 映射整体偏移。

`prepare_from_query(query_descriptor=..., top_k=..., prefix_token_ids=...,
suffix_token_ids=..., min_score=None, allowed_page_ids=None)` 把 stage 2 的
文本检索也收进请求边界：它对 `RAGMaterializer.pages`（全局不可变页库）做
精确 cosine `search`，再调用 `prepare`。检索发生在任何状态变更之前，因此
描述符维度错误、未知 `allowed_page_ids` 或版本/语料错误不会留下半开始的
请求，可以直接重试；`top_k=0` 或 `min_score` 过滤掉全部命中都是合法的
“no-RAG” 请求（prompt 只有前后缀，`finish()` 返回空）。实际拼入 prompt 的
`(page_id, cosine)` 对通过 `request.retrieval_hits` 读取。注意这是文本
embedding 空间的分数，不能与 Prompt 页的 K 质心分数直接比较（设计文档 §9：
第一阶段保持双路检索）。

### 3. CandidateProvider 适配器

```python
from icecache.materialization_provider import (
    DeferredPageQueue, MaterializingCandidateProvider,
)

deferred = DeferredPageQueue()
provider = MaterializingCandidateProvider(
    materializer=mat,
    session_id=session_id,
    n_kv_heads=n_kv_heads,
    fallback_provider=native_provider,          # 现有 DCI/centroid provider
    select_rag=lambda layer, query, state: selected_by_head,  # codex 选择器
    num_leaves=lambda layer, state: state.dci_db[layer].num_leaves,
    on_deferred=deferred,                      # 只入队，不编码
)
# InferState(..., dci_candidate_provider=provider)
```

- `query_leaf_ids(...)` 返回 **已经合并、固定宽度、逐头去重的
  `ready_leaf_ids`**。运行时不要再把 fallback 合一次。
- `provider.last_plan` 是完整的 `CandidatePlan`；`deferred_rag_pages` 只通过
  `on_deferred` 进入队列，永远不进入 native leaf 数组。
- 选择结果可以是逐头 list、`{head: [...]}`，或 codex 的 flat
  `SelectionResult`（按 `page.head_idx` 分组）。
- `on_deferred` 的异常被记录在 `provider.last_deferred_error`，不会中断生成。
  该字段只描述最近一次 `query_leaf_ids`：每次调用开始都会清空，成功调用不会
  继续报告上一次的失败。`provider.last_plan` 是 last-writer-wins，不绑定调用者，
  只用于单调用方 decode 循环的检查/日志，不要在并发请求中读取。
- 纯 Prompt 等价：选择为空时返回值与 `fallback_provider` 完全一致（已测）。
- `apply_rag=False` 为 shadow 基线：仍完整执行选择、plan、指标与 deferred 入队，
  但返回校验后的 fallback leaf，logits 不变；用于先量 materialization hit rate
  与选择指标，再开 live A/B。
- `page_is_ready=`：可选地显式传 `page_is_ready(session_id, page_id) -> bool`；
  不传时自动使用 `materializer.page_is_ready`（若存在）。包装器只暴露
  `head_resolver` 时无法区分“未物化”和“已物化但本层无 map”，会把已编码页
  误送 deferred；这种包装器必须显式传入该 hook（已测两种情形）。
- `trusted_resolver=None` 只对 **未覆写** `head_resolver` /
  `resolve_head_leaves` / `resolve_page_leaves` 的内建 `RAGMaterializer`
  自动打开；子类覆写任一 hook、或在**实例上遮蔽**任一 hook 时必须显式传
  `True` 才走快路径（`resolve_page_leaves` 是 `resolve_head_leaves` 的底层
  实现，单独覆写同样会绕过校验）。快路径跳过逐页重复校验，但每个头仍对合并后的 ready leaf 做一次
  范围检查（索引缩小后不会 fail open）；宽度 > int32 的数组**在 cast 之前**
  做范围检查（否则 `2**32+5` 会别名成 leaf 5）。fallback 始终校验（与
  `retrieval.validate_leaf_ids` 等价的失败类别，重复检查改为跨头向量化）。
- 选择结果里暴露了 `layer_idx` / `head_idx` 的候选会与当前查询层/头比对，
  不一致直接 `ValueError`（选择器混层/混头时叶 ID 仍可能在范围内，只能显式
  检查）。`source == "prompt"` 的条目会被忽略，不做此校验。
- 页面已物化但当前层无 map 时，provider 通过 `page_is_ready` 让该页以零 leave
  参与 plan，而不是 deferred（见 §1 的 `resolve_head_leaves`）。
- 热路径优化（CPU 微基准，**不是** live 结果）：plan 内部用
  `CandidatePlan._from_validated` 跳过重复校验、merge 用 `.tolist()` 迭代；
  已知未物化的页跳过 resolver，全未物化时跳过 merge；`head_resolver` 按页
  批量取 map 并缓存，不再每头重复查表/加锁。本机实测（`/tmp/bench_plan4.py`、
  `/tmp/bench_provider.py`，每档 2000–3000 次）：8 heads × width 64 × 8 页 ×
  每页 16 leaves 时，provider 全 ready 约 **655 µs/call**（上一版约 1070），
  全未物化约 **155 µs/call**；plan 本身分别约 524 / 131 µs，纯 fallback 约
  101（trusted）/96 µs。4 heads × width 32 时 plan 约 222 / 88 / 79 µs。
  仍需 live A/B 判定（见 `component-probes-nominate-live-ab-accepts`）；
  全 ready 的 655 µs 仍是每层每 token 的 CPU 成本，8 层时会明显进入 TPOT，
  接入后必须做 live 测量（必要时改成每轮次缓存 page set）。
- 宽 head 的 fallback 填充已向量化：live 的 `num_neighbours =
  n_dci_pages - layer2topk` 可达数百，Python set 循环会成为 plan 主成本。
  当缺口 `need >= 128` 且 `num_leaves <= 2**20` 时，用一张 bool mask 取
  fallback 中未用到的前 `need` 个（fallback 行已校验为无重复，结果与有序
  set 循环逐元素相同）；更窄或索引过大时保留循环。
- ready leaf 的收集同理：ready 总 leaf 数在 `[128, 1.5×width]` 时用
  `unique(return_index)` + 稳定排序恢复“按页、去重、取前 width 个”的顺序，
  超过 1.5×width 时 Python 循环因提前截断反而更快，仍走循环。A/B 与循环
  路径输出逐元素相等；`/tmp/bench_wide_ab3.py`（8 heads × 8 页 × 16 leaves，
  best-of-3）在 width 256/512 分别为 **1.16x / 1.46x**，width 1024 约 1.85x；
  width 64 时该路径不触发（保持中性）。`fallback` 的 `.tolist()` 也改为仅在
  走循环时惰性构建。短微基准噪声可达 ±20%，这是 CPU 组件测量，仍需 live
  A/B。即使向量化后，全 ready provider 在 width 512（8 heads × 8 页 × 16
  leaves）仍约 **1.23 ms/call**，是每层每 token 的 CPU 成本；宽 head 的 live
  可行性必须先实测，必要时由运行时每轮次缓存 page set。
- **生产形状的 CPU 成本（2026-10-22 实测，`/tmp/bench_live_shape.py`，
  机器 load≈1.3/64 核）**：`tt2t_icecache.py`/`longbench_pred.py` 的生产配置是
  `page_size=16, page_budgets=16, n_sink_pages=2, n_win_pages=2`，
  `page_topks` 默认 0，因此 `n_dci_pages ≤ budget - sink - win = 12`，
  **live `num_neighbours` 约 12**（不是数百；数百需要大得多的
  `page_budgets`）。LLaMA-3.1-8B 有 8 个 KV head、32 层，其中 2 层
  unlimited、0–3 层 reuse，provider 每 token 约在 27–30 层被调用。实测
  （8 heads，每页 16 叶，8 页）：全部 deferred（冷路径）**107–130 µs/层**
  ≈ **3.2–3.9 ms/token**；全部 ready（热路径，未启用 plan cache 时）
  **386–553 µs/层** ≈ **11.6–16.6 ms/token**。成本主要由每层固定的 64 次
  `(page, head)` 解析和 merge 的 Python 开销构成，与 width 关系不大
  （width 12 与 64 仅差约 20%）。这是 CPU 组件估计，不是 live TPOT；
  热路径现由下面的 epoch 失效 plan 缓存覆盖。冷/热路径共同的验证开销也做了
  等价快路径：`num_leaves` 的两个 `np.any` 改为一次 `min`/`max`，
  `_check_leaf_id_bounds` 对 ≤4 字节的有符号 dtype 跳过不可能失败的
  `INT32_MAX` 上界检查，并移除了 cast 后重复的负数检查。
- **epoch 失效的全行 plan 前缀缓存（2026-10-22，`use_prefix_cache=True` 默认开）**：
  同一轮次内 decode 页集通常不变，因此 `MaterializingCandidateProvider` 为每个
  `(session, layer)` 取一个 `PlanPrefixCache`，key 为
  `(width, tuple(per_head_ids))`。只有**每个头都被 ready leaf 填满**的 plan 才写
  缓存；依赖 fallback 的 plan 不入缓存（结果会随 fallback 变化）。命中时重放
  ready 行、deferred 列表与 metrics（`n_fallback_leaves_used=0`），仍按**当前**
  `limits` 复核缓存的逐头 `(min, max)`：索引缩小会在命中路径抛 `ValueError`
  而不是 fail open。`RAGMaterializer` 的每个状态/映射变更（`submit` /
  `begin_encoding` / `complete_encoding` / `mark_gpu` / `mark_cpu` / `fail` /
  `retry` / `release_page` / `release_session` / `invalidate_layer` /
  `_commit_leaf_map` 以及 `begin_prompt` 回滚）都会推进 `_epoch`，缓存条目按
  epoch 失效；`touch` 故意不失效（只更新 LRU/时间戳，不改 leaf 语义）。计算
  过程中发生过变更时 `store` 会因 epoch 不符而丢弃，不会写入跨变更的半新条目。
  每层最多 32 条，满时清空。自定义 readiness hook、非内建 resolver 或没有
  `plan_prefix_cache` 的 materializer 自动禁用；`use_prefix_cache=False` 是
  live A/B 的对照组（输出逐元素相同，仅固定分派）。
  生产形状实测（`/tmp/bench_live_shape.py`，8 heads × 8 页 × 16 叶）：热路径
  **550.7 → 107.1 µs/层**（width 12，5.1x），width 64 时 **671 → 111.3 µs**
  （6.0x）；冷路径只多一次缓存查找，实测 **107–130 µs/层**。30 层估算热路径
  **3.2 ms/token**、冷路径 **3.2–3.9 ms/token**（原 11.6–16.6）。差分验证
  `/tmp/verify5/fuzz_prefix_cache.py`：3,000 场景、12,020 次 plan 与不缓存
  重建逐元素相等（rows/deferred/metrics/异常类型与消息），416 命中、
  11,604 未命中、432 次随机状态变更；`touch` 不失效、命中路径缩小 limits 必须
  报错等语义由自检覆盖。独立对抗审查后又修了三处接线缺陷：(1) 缓存开关现在
  要求 `trusted_resolver=True`（显式传 `False` 时缓存自动关闭，而不是每次调用
  在 `build_candidate_plan` 抛错）；(2) 缓存对象按
  `(materializer 身份, session_id, layer_idx)` 索引，池化 provider 重指
  `session_id` 不会回放旧会话的 leaf；(3) 自动信任检查加入 `_head_rows`
  （`head_resolver` 的底层实现），并在构造时冻结 resolver 工厂，防止构造后
  实例遮蔽绕过检查；(4) `_is_builtin_method` 与发现型 `page_is_ready` 现在还
  要求 `__self__` 就是该 materializer——`m1.head_resolver = m2.head_resolver`
  这类外来绑定虽然 `__func__` 相同，却读 M2 的 leaf，而缓存 epoch 记在 M1 上
  永不失效（独立审查复现为 stale rows，已修复并加自检）。审查方的 640+360
  组差分、4000 组 bounds、2000 组合法 limits、4214 次并发竞态与 dtype 差分
  全部通过，缺陷复现脚本已修复。
  这仍是 CPU 组件测量；下面的合成模型 live A/B 给出了在位成本。
- `provider.stats` 累计每次调用的 `n_ready_rag_pages`、`n_deferred_rag_pages`、
  leaf 用量、deferred 事件/错误与 shadow 调用，并给出
  `materialization_hit_rate = ready / (ready + deferred)`（按 layer×token 的
  “页槽位”计）。`reset_stats()` 清零；用于 Milestone 5 的 hit rate 报告。
- `fallback_provider` 必须是已有的 provider。若要把“原生 DCI 查询”作为
  fallback，需要一个 codex 侧的 wrapper 复刻 `_DCI_query` 的 query 参数
  （`field_of_view`、`parallel_level`、`ratio` 等）；适配器不复制这段原生逻辑。
- `InferState` 目前禁止在 `n_prefetch_layers > 0` 时使用 custom provider；
  接入统一选择时保持 pre-fetch 关闭。

### 4. Deferred 队列

`DeferredPageQueue` 线程安全、按首次出现顺序去重。decode 中只调用
`queue(session_id, page_ids)`；下一次请求边界（agent pause / 轮次边界）再
`drain(session_id)`，走 `PrefillMaterialization` 或受控 chunked prefill。
不要在 decode token 中间同步编码。

### 5. Boundary 编排器（排空 → append → 注册 → 就绪）

`RequestBoundaryMaterializer`（`source/icecache/boundary_materialization.py`）
把上面两节接起来：同一个 queue 同时作为 provider 的 `on_deferred` 和
boundary 的输入，一个实例服务所有会话，同一会话的 boundary 串行、不同会话
并行。

```python
from icecache.boundary_materialization import RequestBoundaryMaterializer
from icecache.materialization_provider import DeferredPageQueue

queue = DeferredPageQueue()
boundary = RequestBoundaryMaterializer(mat, deferred=queue)
provider = MaterializingCandidateProvider(..., on_deferred=boundary.submit_deferred)
# ... decode 只入队 ...

report = boundary.materialize(
    session_id,
    encode=runtime.append_and_index,   # 受控 chunked prefill + _DCI_add
    position_base=state.n_prompt_tokens,
    token_offset=n_sink_pages * page_size,
    max_pages=None,                    # 可选：一次边界最多编码多少页
    require_full_span=True,
)
# report.encoded / skipped_ready / skipped_busy / skipped_doomed /
#        unknown / failed / deferred_next
```

`encode(prompt, page_ids)` 收到只含追加 token 的 prompt 和本次实际进入
`encoding` 的页 ID；它负责 append、`_DCI_add`，并返回
`{layer_idx: (token2node, num_leaves)}`（返回 `{}` 合法：页就绪但暂无 leaf
map，等后续 `token2node` 再注册）。

- **失败策略**：encode/注册失败 → 整批回滚、写进 report，不向生成侧抛异常
  （设计文档 §12 “materializer 失败时 fallback 不崩溃”）；`BaseException`
  （取消、`KeyboardInterrupt`）先回滚再原样抛出。是否重新入队取决于失败
  发生在 **append 提交之前还是之后**：
  - `encode` 抛出/中断（提交前）：页回到 `failed` 并**重新入队**，下个
    boundary 重试；
  - `encode` 返回之后（注册/`finish` 失败或中断）：append 已经在上下文里，
    重试会把同一 chunk 追加两次。这些页被标记为**本会话终态失败**
    （`report.failed`，之后的报告里出现在 `skipped_doomed`），**不重新入队**，
    `stats["doomed_pages"]` 计数；生成侧继续走 fallback。`release_session`
    清除该标记（新上下文可以重新编码）。
- **原子性要求**：`encode` 抛异常时必须让会话 KV 保持原状（丢弃半截
  append）；提交之后的任何失败都由上述 doom 策略兜底。
- 未知页（不在全局页库）直接丢弃并报告——它们永远无法物化；已就绪页跳过；
  另一个请求正在 `encoding` 的页跳过并重新入队。`max_pages` 之外的页同样
  留在队列里。
- `release_session` 只清控制面元数据（队列 + `(session_id, page_id)` 记录）；
  CPU/GPU KV、DCI delta 由运行时释放。

## codex 侧待完成的最小接线

1. `retrieval.py`：加入设计文档 §8 的 `CandidatePlan` 数据类（字段
   `ready_leaf_ids`、`deferred_rag_pages`、`fallback_leaf_ids`、
   `selection_metrics`）；可直接复用
   `icecache.materialization.CandidatePlan`，避免两份定义。
2. `_DCI_query` 的 custom-provider 分支已经是正确形态：
   `validate_leaf_ids(provider.query_leaf_ids(...))`。适配器返回的数组可直接
   通过；**不要** 再合并 `last_plan.fallback_leaf_ids`。
3. 请求边界：prefill 前调用 `PrefillMaterialization.prepare`，用
   `prompt.token_ids` 做正常 prefill；prefill/`_DCI_add` 后按层调用
   `register_layer`，最后 `finish()`。若 `_DCI_add` 触发叶重组，先
   `invalidate_layer` 再用新 `token2node` 注册。
4. `token_offset` 必须由运行时给出（DCI token 0 对应的绝对 prompt 位置），
   **不是 RAG 页面的起始位置**。该参数没有默认值；传页面起点不会报错（只要
   落在 `dci_len` 内），但会在页码不是索引区第一个 token 时静默映射到错误的
   prompt 区间。
   控制面无法从 `token2node` 反推它。按当前 `InferState` 的实现，prefill 阶段
   `prefill_backup_pages` 从 `n_sink_pages` 开始把页搬入临时 CPU cache，随后
   `offload_win_page_to_DCI` 调 `_DCI_add`；因此**首次 prefill add** 的
   `token_offset` 是 `state.n_sink_pages * state.page_size`（运行时需用
   `dci_len` 和 prompt 长度核对）。decode 期间的后继 add 只是把窗口页追加到
   DCI 轴，与 Prompt 内 RAG 页的区间无关。
5. RAG chunk 应尽量与 IceCache `page_size` 对齐。若一个 chunk 跨越
   sink/window 边界，默认的 `require_full_span=True` 会让注册失败并 abort 到
   fallback；确实需要部分映射时显式用 `require_full_span=False`（只映射被
   offload 的 token），不要让半页悄悄映射到错误 token。
6. 选择器（`experimental/unified_page_index.py`）的 flat 结果直接交给
   `MaterializingCandidateProvider`；`selected` 中的 Prompt 页会被
   `build_candidate_plan` 忽略，Prompt 下限已经由 fallback 表达。

## codex 接缝复核（2026-10-22，只读）

- **`_dci_future` 生命周期**：赋值不在 `infer_state.py`，而在
  `adapter/modeling.py:145` 的 prefill 分支（`asyncio.run_coroutine_threadsafe(
  prefill_evict_extra_pages_wrapper, state._loop)`）。同一文件每层开始前
  （第 140 行）先 `result()` 上一个 future，最后一层在调用
  `_finish_prefill` 之前（第 176 行）再排空一次。因此 codex 新加在
  `_finish_prefill`（`infer_state.py:468`）里的
  `result()`/置空在 llama prefill 路径上是防御性重复；它真正覆盖的是
  `end_forward()`（`infer_state.py:552`）这条不经过 adapter 排空的入口。
  executor 只有一个 worker，且每层提交前都排空，故不会死锁；worker 异常会
  在 `result()` 处重新抛出（fail-closed），但 future 不会被置空，下一次
  drain 会再次抛出同一异常——异常不会被吞掉。
- **observer `"query"` 载荷**：`_notify_dci("query", selected_leaf_ids=
  self.selected_page_idx[cur_id])` 在 `diff_pages_by_head` 之后触发。首轮
  它是 `nn_idx_0`；后续轮次是 C 侧 `out_idx`，其每个槽位要么保留一个仍在
  当前查询结果里的旧页，要么用新的 `A` 元素替换（`py_dci.c:2710+`），且
  provider 分支保证两侧等宽、无重复，所以它与本次查询结果的**集合**相同
  （顺序不同）。`RuntimeDescriptorTrace` 存的是集合，语义正确；不要把它
  误当成“本 token 新召回的页”（那是 `recall_idx`）。
- **provider 分支与 `diff_pages_by_head` 的槽位约定**：`selected_page_idx`
  是“槽位→leaf”，`nn_idx_0` 是新查询的 leaf 集合，diff 按值重建槽位；
  provider 返回的逐头合法 leaf 与原生分支同构，不需要额外转换。
- **`_counts` 不参与 provider 分支**：`validate_leaf_ids` 要求恰好
  `num_neighbours` 宽且无重复，比原生缓冲区的“前 `_counts` 个有效”更严格；
  `_counts` 未取用不会让 provider 分支读到未初始化槽位。

## codex 侧复核结果

2026-10-25 已修复并回归验证四个问题。`validate_leaf_ids` 现在会在 int32
转换前拒绝 `2**32`、`-2**32` 等别名值，并校验 `num_leaves` 的整数范围；
`retrieval.CandidatePlan` 延迟导出唯一的
`materialization.CandidatePlan`，没有复制类型；统一选择器用
`(source, session, layer, head, page_id)` 标识 page slice，因而同一文档页可在
不同 KV head 独立选择；floor 阶段与最终 `feasible` 都检查 `max_pages`，排序
优先级也改用调用者给定的 confidence。针对性测试 24 项、2,000 组随机不变量
检查以及 materialization 的 309/111/108/83 项自检均通过。

同日继续修复了 catalog 热路径与过滤语义：descriptor/cost 元数据现在复制后
冻结，descriptor norm 在 `add()` 时缓存并随删除同步清理；显式 `page_ids`
包含未知或过期 ID 时会抛 `KeyError`，不再静默缩小候选集。

`SessionMaterialization` 现在也要求显式 `session_id`，catalog 会拒绝把内建的
session A resolver 用于 session B 查询；这把原先的“一会话一实例”命名约定
升级为可执行隔离边界。

native DCI probe 也已确认 `query_page_ids` 的 `raw` 布局：
   两个半区来自 `py_dci.c`/`dci.c` 的固定布局——前半区是
   `token2nodeIndex[top_candidate[i]->id]`（DCI leaf ID），后半区是
   `top_candidate[i]->id`（数据点 ID），每个 query 的 2×k 槽位按
   `[leaf IDs | point IDs]` 排列。所以 `reshape(n, 2, k)[:, 1]` 取 point ID
   是**对的**（不是 infer_state 里用于 prefetch 的“第二组候选”）。backend
   现在会使用 `_counts` 忽略未使用槽位，并分别检查 leaf 与 point 半区的范围；
   若 ABI 布局改变或 native 返回损坏数据，会在 page ID 翻译前 fail closed。
   2026-10-25 的仓库内 native probe（4096 global 页/head、64 private Prompt
   页/head、128 维、8 KV heads、16 Q heads、top-8、overfetch=2、30 queries）
   得到 240 个 head-query 的 mean recall@8=0.9099、P05=0.75、min=0.50、
   exact-top1=0.9208；build 282.5 ms，近似查询均值 31.5 ms，对应 Python exact
   均值 828.3 ms。它只证明合成 descriptor 的 native 候选路径与 GQA 合并，
   不能替代真实 RAG 质量或端到端 TTFT/TPOT。
   但 `index.query` 同时返回的 `_counts` 被丢弃了：`dci_query` 只写前
   `num_returned` 个有效槽位，malloc 出来的其余槽位未初始化。小规模实测
   （1024 点，`counts==k`）中 head 1 的 point 半区仍出现了 `1024`
   （等于 `num_points`，越界），head 0 则全部在界内；其中越界值会被
   `0 <= idx < num_points` 静默丢弃，而落在界内的未初始化值会被映射到
   **错误的页**。建议：按 `_counts` 截断每个 (head, query) 的 point 槽位，
   并用生产形状的 index 确认多实例 point ID 是 per-instance 还是全局
   （若是全局，`page_ids[head][idx]` 需要减去 `head * num_points`）。
   这两点都不影响本侧：我们只在 codex 返回页 ID 之后消费，且在
   `catalog.search` 里对未知 id 会静默跳过（见第 6 条，属同类风险）。

## 验收状态

已完成（纯 CPU、无模型）：

- 纯 Prompt 等价：选择为空时逐元素等于 fallback。
- 文本检索的范数缓存：缓存值与直接 `np.linalg.norm` 逐位相等；删除某个
  缓存条目后重算的分数序列与带缓存时完全相同；`remove` 删除缓存，
  re-add 重新计算。
- 文本检索的堆叠矩阵路径（300 页 × 256 维自测）：排序与独立逐页参考一致，
  `allowed_page_ids` 子集一致，`add`/`remove` 使矩阵失效并在重建后找到新页；
  20k × 1024 实测 6.75 ms/query（见 §1）。
- deferred 页不进入 native 数组，fallback 保持固定宽度。
- 双 session 对同一全局页状态独立；`release_session` 互不影响。
- 失败回滚：`begin_prompt`/`register_layer` 失败后页面为 `failed`，无半套 map。
- DCI 叶重组后的失效/重注册。
- 1000 组映射 + 500 组 plan fuzz、8 会话并发 stress。
- provider 输出通过 codex 真实的 `retrieval.validate_leaf_ids`
  （含 RAG/fallback 重叠、窄宽度、shadow 输出）。
- 与 codex **真实选择器**的端到端互操作（`/tmp/interop_codex_selector.py`）：
  `UnifiedPageCatalog.search`（逐 head）→ `select_pages`
  （`min_prompt_pages=1`）→ `SelectionResult` → `query_leaf_ids` →
  `CandidatePlan`。结果为固定宽度 `[2, 3]`、逐 head 无重复；已物化页给出
  `[[0,1,2],[6,7,8]]`；未物化页进入 `deferred_rag_pages` 且按首次出现去重；
  prompt 页被过滤；catalog 上陈旧的 `materialized=True` 不覆盖物化器的真实
  状态。同一脚本复现上面缺陷 2 的 `ValueError`。
- `select_pages` 的 `max_pages` 上限不可信（缺陷 3）：本侧按选择顺序截断到
  `num_neighbours`，所以即使选择器返回超限集合也不会越界；`SelectionResult`
  的直接消费者必须自己检查长度。
- 多轮双 session 闭环：deferred → 下次请求边界物化 → ready；两个 session 对同一
  全局页状态独立，hit rate 从 0 升到 1。
- `release_session` 发生在 encoding 期间时，`finish()` 失败会 abort 整个请求
  （不会留下半完成状态）。
- 已物化但该层 map 缺失（`invalidate_layer` 之后）的页：plan 记录为 ready、
  零 leaf、不进 deferred，请求边界不会再尝试编码它。
- `abort()` 与 `prepare()` 的并发/重入：abort 先到则 prepare 拒绝；abort 在
  `begin_prompt` 期间到达则新页被标 `failed`，不会卡在 `encoding`。
- 中断在 `encoding` 的页可由 `retry()` 恢复为 `queued`（span/leaf map 清空）。
- `token_offset` 缺失为 `TypeError`（不再有 0 默认值）。
- `prepare_from_query`（请求边界文本检索）：prompt 按 `store.search` 的
  分数顺序拼接，`retrieval_hits` 与实际拼入的页一致；`min_score` 过滤到
  空或 `top_k=0` 时是合法 no-RAG 请求（只有前后缀、`finish()` 返回空）；
  描述符维度错误/未知 `allowed_page_ids`/非法 `min_score` 都在任何状态
  变更之前失败，请求保持未开始且可重试；重复调用报 `RuntimeError`。
  `/tmp/fuzz_prepare_from_query.py` 300 组随机语料/查询/阈值与独立的
  `search → assemble_prompt → begin_prompt` 参考逐元素一致；
  `/tmp/stress_prepare_from_query.py` 20 轮 8 线程竞争（恰一个成功，其余
  `RuntimeError`，无页卡在 `encoding`）+ 100 轮 `prepare`/`abort` 竞争
  （结束后不存在 `encoding`/`queued` 页）。
- provider 的信任快路径只自动用于未覆写 resolver 的 `RAGMaterializer`
  （`head_resolver` / `resolve_head_leaves` / `resolve_page_leaves` 任一覆写，
  或在实例上遮蔽，都会关闭自动信任）；int64 别名 ID 在快路径上也会被拒绝；
  `last_deferred_error` 不跨调用残留（参数校验失败也会先清空）。
- provider 的 `page_is_ready` 显式参数：包装器无该 hook 时已物化但本层无 map
  的页会被 deferred（记录在文档中的已知限制）；显式传入后不再 deferred。
- `KeyboardInterrupt` / `asyncio.CancelledError` 等 `BaseException` 在
  `register_layer` / `finish` 中也会 abort 整个请求，页面落到 `failed`，
  不会停在 `encoding`；`begin_prompt` 在 `begin_encoding` 变异循环中被
  `BaseException` 打断时，也会回滚本次新入队的页（含已进入 `encoding` 的页），
  以及调用前已 `queued`、本次被移入 `encoding` 的页；回滚只看调用前快照，
  `submit` 内部的 `KeyboardInterrupt`（返回值丢失）同样覆盖。调用前已在
  `encoding` 的页不动（`/tmp/stress_begin_prompt_race.py`：8 线程 × 50 轮无死锁）。
- decode-cold 快路径：已知未物化的页不调用 resolver；全未物化时跳过 merge，
  ready/fallback/metrics/deferred 顺序与慢路径一致（自检含 resolver 不得被
  调用的断言）。
- `head_resolver` 批量解析：绑定闭包内的结果与 `resolve_head_leaves` 逐头
  一致（含空 map/越界/未物化）；已注册的 `[n_heads, 0]` 空 map 在批量路径
  返回零长度行且不 deferred；缓存只影响同一 plan 内的重复访问，重新注册
  后新 resolver 立即看到新 map；覆写 `resolve_head_leaves` 的子类仍走逐头
  路径（自检用计数子类验证）。
- 宽 head（`num_neighbours` 数百）的 merge：fallback 填充与 ready leaf 收集
  的向量化分支与有序 Python 参考逐元素相等（`/tmp/fuzz_merge_wide.py` 400
  组随机宽度 64–1024、含跨页重复、部分 ready、trusted/untrusted 交叉校验）。
- 宽 head 的真实 provider 差分（`/tmp/fuzz_provider_wide.py`，60 组，宽度
  128–512，混入空 map 与未物化 ghost 页）：`query_leaf_ids` 与逐头
  `resolve_head_leaves` 参考逐元素相等。
- 第四轮独立对抗审查（2026-10-22，当前 revision 无已验证缺陷）：新增
  `/tmp/verify3/fuzz_merge_bounds.py` 4,000 组，独立参考实现逐元素比对
  rows / deferred 首次出现顺序 / 全部 metrics，覆盖 `vec_ready`、
  `loop_ready`、`vec_fill`、`list_fill`、`loop_fill`、`no_fill` 六个分支，
  width 32–512、`num_leaves` 恰为 `2**20` 与 `2**20+1`、`total_ready` 恰在
  128 与 `(3*width)//2` 两侧；`/tmp/verify3/fuzz_begin_prompt.py` 5,000 组
  随机状态机（七个 pre-state × 显式/prompt 派生 id 列表），与独立模型比对
  包括“中断落在 `submit` 内部”的五种时序；并发测试证明另一个线程的
  `submit` 会阻塞在外层 RLock 上，中断后观察到的已是回滚后的 `failed`。
  回归差分：plan 20,000 组、resolver 2,400 次、provider e2e 4,000 组、
  800 个并发 `query_leaf_ids`（plan_count/stats 均为 800）。
- prefill 偏移 dry run（`/tmp/prefill_dryrun.py`）：5 页 × 4 头 × 3 层，
  `token_offset = sink`，逐叶核对 leaf 的成员 token 落在该页 prompt 区间内；
  跨 sink 边界的页默认 fail closed，`require_full_span=False` 时只映射被
  offload 的部分。
- 跨层/跨头候选被拒绝（仅在候选暴露 `layer_idx`/`head_idx` 时检查）。
- `prepare`/`abort` 与 `register_layer`/`abort` 的并发压力：
  `/tmp/stress_prefill_race.py` 300+300 轮，无页停留在 `encoding`。
- Stage-B append（`position_base`）：`begin_prompt` / `prepare_append` 只平移
  记录的 span，prompt 本身仍是局部坐标；`materialization` 自检 309 项（含
  `position_base` 为 `-1` → `ValueError`、`1.5`/`True` → `TypeError` 且不留
  记录），`prefill_materialization` 自检 108 项（含 `token_offset=96` 的逐头
  leaf map、重复 `prepare` 拒绝、已 ready 页拒绝重编码）。
- 请求边界编排器（`boundary_materialization.py`，自检 83 项）：排空 deferred
  → `prepare_append` → 逐层注册 → `finish` → 页变 `cpu_kv`；已 ready 跳过、
  未知页丢弃并报告、`encoding` 页跳过并重新入队、`max_pages` 溢出回队，
  编码失败时 `abort` 且**不抛异常**；`BaseException` 从进入 `encode` 起视为
  可能已提交，页一律 doom 后原样抛出；
  `require_full_span` 默认 fail closed，`False` 时裁剪到 offload 区间。
  提交前失败重新入队；提交后（注册/`finish`）失败则标记 `skipped_doomed`
  终态、不重试、不回队，`release_session` 清除（避免重复 append）；此时
  `report.n_tokens` 是已提交的块长度，调用方**必须**用它推进上下文长度。
  自检覆盖
  两种阶段、doomed 页被再次 defer 时不再调用 encode、以及提交后
  `KeyboardInterrupt` 的 doom 路径。并发由 `_SessionGate`（条件锁 + busy +
  generation）负责，**不跨用户 `encode` 持锁**：两会话互相 release 不死锁，
  release 不等待在途 boundary，在途 boundary 发现 generation 变化后报告失败、
  不回队、不注册。`release_session` 在会话 gate 内清理队列/doomed 标记并注销。
- `/tmp/interop_boundary.py`：decode 侧 provider 把未物化页写进
  `deferred_rag_pages`，同一共享 queue/boundary 在请求边界处理后，**同一
  session** 的 provider 立即解析出 native leaf；第二个 session 仍视两页为未
  物化，并在 `position_base=40` 处独立编码，s1 的 span (0,8)/(8,16) 不受影响。
- 上述互操作现在有仓库内的可重复入口
  `benchmark/agent_cache/materialization_boundary_probe.py`。2026-10-25 运行时，
  同一 global page 在 session A 的绝对 span `[16,24)` 解析为逐头 leaves
  `[4,5]`，在 session B 的 `[32,40)` 解析为 `[8,9]`；释放 A 后 B 仍返回
  `[8,9]`。两次边界控制面分别约 457/320 µs，但 encode callback 使用合成
  `token2node`，因此这些数字明确不是 TTFT、TPOT 或真实 prefill 时间。
- `/tmp/stress_boundary.py`：2 个 session × 4 线程 × 40 轮，外加 decode 侧
  重复入队 ticker；另用注入失败的编码器让 C/D 收敛。终态统计
  `{'boundaries': 166, 'encoded_pages': 96, 'failed_pages': 24, 'unknown_pages': 0, 'doomed_pages': 0}`，
  全部页 `cpu_kv`、无 `encoding` 残留、四会话队列均为空、`token_start` 均
  ≥ 各自 `position_base`。
- 第五轮独立对抗审查（boundary/append 专项，当前 revision 无已验证缺陷）：
  `/tmp/verify4/fuzz_boundary.py` 3,000 组模型驱动场景，独立参考模型逐次
  核对 requested/ready/busy/unknown/active/overflow 分类、`encoded`/`failed`、
  `layers`/`n_tokens`、绝对 span、逐层 leaf、失败后无残留 map、队列内容
  （含显式 `page_ids` 不排空旧队列的语义）、累计 `stats`，并随机穿插
  `release_session`；覆盖 ok 367 / empty 132 / fail 104 / malformed 121 /
  overflow 164 / busy 131 / ready-skip 2101（malformed = 提交后失败，进入 doom 分支）。`/tmp/verify4/
  fuzz_provider_boundary.py` 1,500 轮真实 provider → deferred → boundary
  → 再查询闭环（3 会话、12 页、3 个 shadow 会话）：deferred 集合逐轮精确，
  物化后同一选择零 deferred 且 native leaf 命中，其它会话保持未物化；
  共物化 374 页。两者均无失败。
- 边界注册热点优化（2026-10-22，`/tmp/verify4/bench_boundary.py`，生产形状
  12 页 × 30 层 × 8 头、span 16 token）：一次边界物化的控制面耗时
  **148.2 → 39.6 ms**（1 页 12.7 → 3.55 ms）。两处改动：
  (1) `map_span_to_leaves` 用“每头一次 `np.sort` + 位移比较去重”替代
  `np.unique`（小数组上调用开销占主导），并对越界检查做整块向量化；
  (2) `register_layer_leaves` 产出的 map 由构造保证合法，经
  `_register_leaf_map_validated` 提交，跳过重复的 `_validate_leaf_map`
  （每次 page×layer 约 0.18 ms）。该快速路径拒绝可写数组，避免注册后的 map
  被原地修改；公共 `register_leaf_map` 仍走完整校验。差分：
  `/tmp/verify4/fuzz_span_map.py` 5,000 组与独立 `np.unique` 参考逐元素相等
  （含负值/重复/标量 limits/int32+int64/只读断言）；全部既有外部回归
  （fuzz 1000+500、resolver 2608、merge 3000+4000、provider 4000+60、压力、
  互操作）重跑通过。
- **`token_offset` 方向陷阱**：语义是“DCI token 0 的绝对 prompt 位置”，
  即 `dci_index = prompt_position - token_offset`（减，不是加）；正向偏移
  不会报错，只会静默映射到错误的 token。本轮参考模型一开始就把它写反了，
  已写进 `register_layer_leaves` 的 docstring。
- plan 前缀缓存差分（`/tmp/verify5/fuzz_prefix_cache.py`）：真实
  `RAGMaterializer` + 2 会话 × 8 页 × 3 层，随机 submit/begin_encoding/
  注册 `complete_encoding`/`fail`/`retry`/`mark_gpu`/`mark_cpu`/
  `invalidate_layer`/`touch`/`release_page`/`release_session` 与随机宽度/重复
  page_id/ghost 选择/两种 limits 交错；3,000 场景、12,020 次 plan 与不缓存
  重建逐元素相等（416 命中、11,604 未命中、192 次预期异常、432 次状态变更）。
  `touch` 不失效、状态变更后未命中、命中路径缩小 limits 报错同时由
  `materialization`/`materialization_provider` 自检覆盖。
- **合成模型 live A/B（2026-10-22，GPU 0，真实 IceCache kernel）**：本机两张
  GPU 各被一个 vLLM 服务占用约 72 GiB，Qwen3-1.7B 又因 GQA group_size=2 不被
  `batch_prefill` 支持，因此用随机权重的 tiny Llama（8 层、32 Q / 8 KV /
  head_dim 128、512-token prompt、page_size 16 / budget 16 / sink 2 / win 2，
  索引层 2–7）跑真实 prefill + decode。真实 `token2node`（[8, 448]，DCI 首
  token 绝对位置 32）逐层注册 28 个 RAG 页，选择器每头返回全部 28 页，
  width=12；native fallback 复刻 `_DCI_query` 的同一次 DCI query（无副作用）。
  每臂独立进程、串行重复（`/tmp/live_synth_ab.py`）：
  * 在位计时：fallback DCI query **0.37–0.48 ms/call**（两臂相同，对照成立）；
    plan-only **1.95–2.73 ms/call（缓存关）→ 0.25–0.31 ms/call（缓存开）**
    （7.6–10.5x）；6 个索引层合计 **14.4–15.0 → 2.5–3.0 ms/token**；缓存
    每层首调 miss、其余 15/16 hit。
  * 生产 width=12（从 28 个 offload 页里选 12 页，`numactl` 绑定、每臂独立
    进程、串行 2 次）更干净：plan-only **1.04–1.05 ms/call（缓存关）→
    0.177–0.180 ms/call（缓存开，字符串直通分支后）**（5.8–5.9x），6 层合计
    **6.28–6.37 → 1.46–1.46 ms/token**。整 token：native **17.45–17.75 ms**、
    shadow 关 **25.00/24.80** → 开 **19.38/19.58**、`apply_rag=True` 关
    **24.19/24.37** → 开 **18.92–19.08**。缓存开/关差 **5.14–5.23
    ms/token**，与 plan 差值 4.7–4.9 ms 基本逐 token 对上；native 与 RAG
    选择会改变 GPU 页行为，仍不能直接互比。
  * 整 token steady median（vLLM 同卡驻留，28 页那轮噪声大）：shadow 关
    **34.4 / 42.0 ms**、shadow 开 **26.7 / 22.6 ms**；`apply_rag=True` 关
    **33.8 / 37.2 ms**、开 **20.5 / 24.1 ms**。符号稳定，绝对值不能当生产
    TPOT；只有同选择集的 cache on/off 是干净对照（native 与 RAG 选择会改变
    GPU 页行为，不能直接比）。
  * 结论：缓存把稳态 plan 成本从“随选择页数近似线性”降到基本恒定；但命中仍
    要解析 28×8 个候选，所以生产选择集应接近 DCI 预算（~12 页），而不是
    prompt 的全部 offload 页。合成模型 GPU 计算远小于 8B，provider 占比是
    上限；生产 TPOT 必须在 GPU 空出后用 Llama-3.1-8B 复测。
- **满命中跳过 fallback（2026-10-22，`skip_fallback_on_full_hit=True` 默认开）**：
  plan 缓存命中要求每个头都由 ready leaf 填满，此时 fallback 的值可证明不会被
  使用；provider 现在先解析 selection 并 `probe` 缓存，命中就直接返回缓存行，
  完全跳过本 token 的原生 DCI query。只在 `apply_rag=True` 且缓存启用时生效
  （shadow 必须返回 fallback，skip 无效；`use_prefix_cache=False` 或非内建
  resolver 同样无效）。命中的 plan 带 `fallback_queried=False`，
  `fallback_leaf_ids` 仅镜像 ready 行以保持固定形状（`n_fallback_leaves_used=0`）；
  未命中时多一次缓存查找；空缓存直接跳过 probe/选择解析（冷窗口），非空缓存
  但选择不同则本层计数会有 2 次 miss（probe + build）。`PlanPrefixCache.probe`
  仍按当前 `limits` 复核缓存 bounds。
  生产 width=12 的合成 live A/B（GPU 0、`numactl` 绑定、独立进程、串行 2 次）：
  * provider 调用 **572–575 → 125–127 µs/call**，每 token（6 层）
    **3.96–3.98 → 1.45–1.46 ms**；fallback 调用 **96 → 6**（每层只在首 token
    一次）。（热路径优化前的同一对照为 587–594 → 137.5–138.4 µs/call、
    19.10/19.11 → 16.07/16.23 ms。）
  * 整 token steady median **18.92/19.08 → 16.16/16.33 ms**，即
    **−2.75 ~ −2.76 ms/token（−14.4%~−14.5%）**。
  * 两臂 16 token × 6 层返回数组的 sha256 **完全相同**
    （`dd390a0f72b9…2805a`，两轮重复一致），所以这是纯成本削减，不是选择变化。
  * 组件基准（`/tmp/bench_live_shape_cache.py`，width=12，8/16/28 页，30 层）：
    热路径（缓存开 + skip）**47.5 / 55.5 / 66.8 µs/层**（skip 初版为
    58.6 / 77.8 / 106.8），30 层估算 **1.43 / 1.66 / 2.00 ms/token**；冷路径
    （全部 deferred）缓存开 **112.4 / 132.7 / 164.8** vs 缓存关
    **102.1 / 123.2 / 154.5 µs/层**——空缓存门控把 probe 开销压到约 10 µs/层
    （未加门控时是 43–92 µs）。选择解析里普通字符串走直通分支，省掉每调用
    224 次 `_plan_page_id` 调用，是热路径进一步下降的主要来源。
  * 合成模型 GPU 计算远小于 8B，绝对占比会不同；但被跳过的是与模型无关的原生
    DCI query，方向确定。生产决定仍应在 8B/LongBench 上复核。
- **`kv_materialization` 独立数值复核（2026-10-22，5 个确认缺陷已修，自检
  49 → 55；2026-10-23 再修 uint 回绕与 fingerprint fail-open，55 → 57）**：(1) “Numerical note” 原称同表同位置可逐位一致，实测不成立——
  `apply_rope` 在 float64 累加后一次舍入，live 模型逐算子按自身 dtype 舍入，
  fp16 下约 1/3 元素差 1 ulp；已把契约改为“float64 旋转恒等精确”，不再声称
  复现模型舍入。(2) `pack_pages` 只检查 keys dtype，values 会被静默转换；
  `MaterializedPage` 与 `pack_pages` 现在都拒绝 keys/values dtype 不一致
  （含 duck-typed 输入）。(3) `context_fingerprints` 映射缺页 id 时原返回
  None，直接跳过跨上下文门；现在与 positions 映射一致地抛 `ValueError`。
  (4) uint8/uint16 位置数组的 `np.diff` 会回绕（`[5,0] → 251 > 0`），非单调
  位置被接受；`PageKV` 与 `_positions_array` 改为相邻比较。
  (5) `rope_cos_sin` 先 `reshape(-1)` 再做 ndim 检查，2-D `position_ids` 被
  静默摊平；现在先检查维度并拒绝。
- **三路独立对抗性复核第二轮（2026-10-22，3 个确认缺陷已修，自检
  296 / 111 / 104 / 66 / 55）**：
  (1) provider 的 auto-trust 原来只对 `head_resolver`/readiness 做
  `__self__` 校验，`plan_prefix_cache` 只查 callable：外部绑定方法或只覆盖
  该方法的子类会让缓存挂在**另一个** materializer 的 epoch 上，本 materializer
  的 leaf-map 变更永不失效、持续回放旧行。现在 `plan_prefix_cache` 也进入
  auto-trust，工厂在构造时冻结，并在首次取缓存时校验 `_owner is materializer`
  （post-construction shadow 被忽略；若工厂返回外来缓存则 `RuntimeError`）。
  (2) `normalise_selected_page_ids` 的普通字符串直通分支接受 `""`，并把它
  作为 deferred 页 ID 交给调用方；而公开的 `CandidatePlan` 构造器与
  `DeferredPageQueue` 都拒绝空 ID。现在规范化阶段直接 `ValueError`。
  (3) `PrefillMaterialization._start` 在 `begin_prompt()` 返回与
  `_begun = list(begun)` 之间被 `KeyboardInterrupt` 打断时，`abort()` 扫描
  空列表，刚启动的页会永久停在 `encoding`（调用方也拿不到可 `retry` 的 ID）。
  现在该窗口内的 `BaseException` 会：标记请求 aborted、逐页 `fail`、再原样
  抛出。三个缺陷的复现脚本（`/tmp/review_state_A/foreign_cache.py`、
  `/tmp/review_state_B/repro_stranded.py`、空 ID repro）现在都 fail closed。
  同时修正 docstring：`on_deferred` 只吞普通 `Exception`，`BaseException`
  照旧传播。
- **三路独立对抗性复核第三轮（2026-10-23，18 个确认缺陷已修；自检
  299 / 111 / 108 / 83 / 57）**：
  * kv：`materialize_page`/`KVPageStore.pack` 对无 fingerprint 的页 fail-open
    → 现在视为不可验证并拒绝；uint 位置数组加 shift 会在模 2**bits 处回绕
    → 先在 int64 中做加法（`uint8 [250..] + 10` 复现）；"Numerical note"
    现明确 rotary 表始终 float64。
  * API：所有 layer/head/step/token_offset/token span 校验统一走
    `_require_index` / `_require_span`，拒绝 bool（`True` 曾等于 1）；
    `n_kv_heads` 拒绝 bool/浮点；`chunk_document`/`search` 拒绝 bool
    `chunk_size`/`overlap`/`top_k` 与字符串 `allowed_page_ids`；
    `normalise_selected_page_ids` 拒绝把整串当每头选择；零宽
    `fallback_leaf_ids` 在 `build_candidate_plan` 就拒绝；
    `map_span_to_leaves` 对两个等长 1-D 数组的二元组 fail closed。
  * 并发/生命周期：(1) `release_page`/`release_session` 先 bump `_epoch`
    再删除（信号落在删除后、bump 前会让外部 plan cache 永久回放已释放
    的页，且重试不再 bump）；(2) `begin_prompt` 新增 `begun_out`，
    `_start` 在调用返回边界被信号打断时能逐页 `fail`（不再永久
    `encoding`）；(3) `finish` 在 `complete_encoding` 已提交但
    `_finished` 未置位的窗口内先观察提交状态，已 ready 的页不再被
    abort 误杀；(4) boundary 不再跨用户 `encode` 持有会话锁，改为
    `_SessionGate`（busy + generation）：两会话互相 `release` 不再 ABBA
    死锁，release 不等待在途 boundary，而在途 boundary 通过 generation
    发现释放、报告失败且不回队；(5) `encode` 一进入就视为可能已提交，
    `BaseException` 一律 doom 而非回队（信号可在 append 后、return 前
    到达）；(6) provider 删除 `_plan_caches` 注册表，每次经冻结工厂取
    materializer 自己的 `(session, layer)` 缓存，`release_session` 真正
    回收（长驻 provider 不再每会话泄漏一个缓存对象）；(7)
    `submit_deferred` 与 `release_session` 共用会话 gate，提交不再穿插
    清理（release 返回后的提交按取消契约视为违规，已在 docstring 写明）。
  17 个回归脚本全部 exit 0；旧复现脚本中仍断言旧行为的（strand/重复
  append/死锁/旧缓存对象）按预期失败，输出已人工核对为修复后的行为。

待运行时验证：

- provider 的 live TPOT 成本。生产形状的 CPU 组件估计为冷路径
  3.2–3.9 ms/token、热路径（plan cache 开启）约 3.2 ms/token（30 层，
  见 §3）；合成模型 live A/B（见上方验收状态）已在真实 decode 中量到
  plan-only **1.45–1.46 ms/token**（6 层、生产 width=12，缓存开；热路径优化前
  为 1.54–1.56）与 **2.5–3.0 ms/token**（6 层、28 页），且满命中跳过 fallback 再省
  **2.87–3.05 ms/token**；但 8B 的 GPU 计算占比和真实 LongBench prompt 仍需
  复测；A/B 一律用
  `use_prefix_cache=False` 做同代码对照（缓存只在每头都被 ready leaf 填满时
  生效与命中）。
- 真实 prefill 的每层 `token2node` 与 DCI `token_offset`：合成 Llama 的
  真实 IceCache prefill 已验证（`token2node=[8,448]`，DCI 首 token 绝对位置
  32，逐层注册）；生产模型/真实 prompt 待复测。
- DCI add 后旧页 `selected_page_idx` / `ccc` / `cc2gp` 一致性。
- 真实 materialization hit rate、TTFT（含召回/拼接/真实 prefill）、TPOT、
  GPU `max_memory_allocated/reserved`、CPU RSS 与页数。
- n=200 上与 DCI 的逐行对比；注意 DCI 自身不可复现（历史上 9% 行漂移），
  正确性门槛用 `page_scan`/`pq` 一类可复现指标。

## Stage-4 运行测量协议（GPU 可用后）

四组对照，同机同数据，**每组单独进程、串行运行**（跨 backend 绝不能同进程，
同一进程里 page_scan 会因 DCI 驻留而慢约 1.5 s）：

1. `no-rag`：纯 Prompt，正常 IceCache page_scan，不装 provider。
2. `rag-shadow`：完整召回/选择/plan/deferred，但 `apply_rag=False`
   （返回已验证 fallback）。用来把“测量与物化开销”从“logits 改变”中分离。
3. `rag-live`：`apply_rag=True`。
4. `plain-rag`：经典文本 RAG（检索片段拼进 prompt，不复用 KV）——设计文档的
   质量基线。

每行记录（n=200，按行配对，保留原始值）：

- TTFT：请求开始到首个生成 token，**包含**召回、选择、拼接、真实 prefill、
  deferred 提交；不要减去其中任何一段。入口计时器不统一：`longbench_pred`
  报 TTFT+1·TPOT 且无 sync，`page_scan_compare` 报 sync 后的 TTFT 并输出
  `retrieval_stats`——固定用一个入口，跨入口数字不可比。
- TPOT：首 token 之后的纯 decode；冷启动（前若干请求物化为 `cpu_kv`）与
  稳态分开统计，并单独报告 materialization hit rate。
- GPU：每轮重置后读 `torch.cuda.max_memory_allocated()` /
  `max_memory_reserved()`。
- CPU：稳态 RSS（预热后）、KV 页数、DCI 页数、deferred 队列深度。
- 每层核对 `num_neighbours ≈ page_budgets - page_topk`、provider 只装在
  启用 RAG 的层、`token_offset = n_sink_pages * page_size`。

测量纪律（都是本项目踩过的坑）：

- 跨 backend 计时统一 `--threads 64`；`--threads` 只节流 DCI、不节流
  numpy，混用会凭空造出 page_scan 加速。
- 用 `numactl` 绑定 NUMA；同一份代码未绑定时曾从 515 ms 漂到 976 ms。
- 跨时间窗的比值必须带一个“同代码对照”窗口；曾有一次 70% 的 margin
  来自窗口漂移。
- 复制实验串行跑，不并发；并发“独立复制”会互相抢占锁，连噪声底都会漂
  （DCI 空对照 0.7% vs 6.7%）。
- 报告与统计量匹配：Doc 19 的头条是均值，拿中位数去比会凭空造出
  ~350 ms 的“回归”。
- 任何新加的 instrumentation 先单独量开销；一次测量里的 `synchronize()`
  曾把差值从 ~0 挪到 −91 ms。
- 正确性门槛用可复现指标（`page_scan`/`pq`/逐行比对）；DCI 对照臂自身
  在同样代码下有约 9% 行漂移，不能拿它判对错。
- Stage-4 通过线：`rag-live` 的生成质量不低于 `plain-rag`；TTFT/TPOT/内存
  全部如实报告；无预算违约；session 结束后 KV/私有索引无泄漏。
