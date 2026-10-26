# Unified RAG × IceCache：实现设计与 Coding 指南

## 1. 目标

本项目希望把传统 RAG 的文档召回与 IceCache 的历史 Prompt KV 召回统一到一个“语义页”控制面中。统一的含义不是把所有对象强行塞进同一棵物理 DCI 树，而是让它们共享候选表达、相关度排序、资源成本模型和最终选择器，并在执行阶段根据物化状态走不同的数据路径。

最终系统需要回答两个问题：当前请求最值得保留或加载哪些页，以及这些页如何以合法的 K/V 进入当前会话的注意力计算。第一个问题是检索与资源优化问题，第二个问题是上下文相关的 KV 物化问题。当前代码已经覆盖第一个问题的大部分控制面，但第二个问题仍是主要开发任务。

## 2. 必须遵守的正确性边界

Transformer 深层 K/V 依赖前缀内容、因果上下文和 token 位置，K 还受到 RoPE 的位置旋转。因此，不能默认把一段文档离线编码成深层 KV，然后注入任意会话。除非未来预训练模型实验能够证明质量等价，否则全局层只能持久保存文档文本、token、稳定描述符和可选的浅层辅助数据；深层 KV 必须由当前会话重新编码产生。

因此，第一版系统采用以下不变量：

1. 全局 RAG 页是不可变元数据，不直接拥有可跨会话复用的深层 KV。
2. Prompt 页及已经编码的 RAG KV 都属于单一会话，其他会话不可见。
3. 不同 transformer 层的向量不能互相比较；每个索引和查询都绑定一个 `layer_idx`。
4. 不同 KV 头拥有独立页空间；GQA 中多个 Q 头只能在其对应 KV 头内部合并候选。
5. 选择器可以使用全局资源预算，但必须满足每个 KV 头的最低页宽，否则结果不能映射到当前 IceCache 定宽张量。
6. 未物化的 RAG 页只能生成物化请求，不能作为 native leaf ID 进入 recall/scatter。

## 3. 推荐架构

```text
                         ┌──────────────────────────────┐
                         │ Global RAG Page Store        │
                         │ text / token_ids / descriptor│
                         └──────────────┬───────────────┘
                                        │ immutable metadata
                                        ▼
┌──────────────────────┐       ┌──────────────────────────────┐
│ Session Prompt DCI   │──────▶│ Unified Candidate Plane      │
│ real leaf centroids  │       │ relevance + error + cost     │
└──────────────────────┘       └──────────────┬───────────────┘
                                              │ chance-constrained selection
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
              native prompt leaf                      descriptor-only RAG page
                         │                                         │
                         │ immediate recall                         │ materialize
                         ▼                                         ▼
                 IceCache diff/recall                 session-local chunked prefill
                                                                   │
                                                                   ▼
                                                        CPU/GPU KV + native leaf
                                                                   │
                                                                   └── next query可召回
```

物理上建议继续使用两个分区：全局、只读的 RAG 描述符目录，以及每个会话独立的 Prompt/KV delta。查询接口在逻辑上合并两者。这样可以避免其他会话的私有页污染有限的 DCI 结果缓冲区，也简化删除、同步和租户隔离。

## 4. 已有核心代码

`source/icecache/retrieval.py` 定义运行时扩展协议。`DCIObserver` 接收 build、add 和 query 事件；`CandidateProvider` 可以替换候选生成，但继续复用 IceCache 原有的页差分、CPU recall 和 GPU scatter。`DCIObserverGroup` 用于按照确定顺序组合多个观察器。

`source/icecache/experimental/unified_page_index.py` 定义统一页元数据、分层分头目录、会话物化成本、经验机会约束选择器、局部搜索和小规模精确 oracle。

`source/icecache/experimental/centroid_provider.py` 从真实 DCI token-to-leaf 拓扑计算页级 K 质心，并作为精确余弦 `CandidateProvider` 工作。它只返回已经存在于 IceCache 中的合法叶。

`source/icecache/experimental/shadow_joint_selector.py` 在真实查询上联合考虑 Prompt 叶和 RAG 描述符，但只记录决策，不改变模型生成。它是实现物化器之前的安全验证路径。

`source/icecache/experimental/dci_page_backend.py` 是多 KV 头 M-DCI 候选后端，包含原生结果容量保护和 GQA 候选合并。

核心召回入口位于 `source/icecache/infer_state.py::_DCI_query`。后续实现应尽量保持 `_DCI_query → diff_pages_by_head → recall → scatter_pages` 这条成熟路径不变，而在候选生成前后增加物化协调。

## 5. 建议的数据模型

### 5.1 全局文档页

```python
@dataclass(frozen=True)
class GlobalRAGPage:
    page_id: str
    document_id: str
    segment_id: str
    token_ids: np.ndarray
    descriptor: np.ndarray
    descriptor_version: str
    text_hash: str
    error_radius: float
```

`page_id` 必须在整个全局目录中稳定且唯一。`token_ids` 是物化时的权威输入；不能仅保存解码后的文本，因为重新 tokenize 可能改变特殊 token 和边界。`descriptor_version` 用于在更换 embedding 或投影后拒绝混合不兼容向量。

### 5.2 会话物化记录

```python
MaterializationState = Literal[
    "descriptor_only",
    "queued",
    "encoding",
    "cpu_kv",
    "gpu_kv",
    "failed",
]

@dataclass
class SessionPageMaterialization:
    session_id: str
    page_id: str
    state: MaterializationState
    token_start: int | None
    token_end: int | None
    layer_leaf_ids: dict[int, np.ndarray]
    last_access_step: int
    error: str | None
```

全局页对象不能直接保存 `materialized=True`，因为同一文档在会话 A 中可能已经位于 GPU，而在会话 B 中仍只有描述符。状态必须以 `(session_id, page_id)` 为键。

### 5.3 联合候选

候选至少需要包含来源、层、头、相关度、误差上界、物化状态、场景对齐的成本向量，以及 native leaf ID 或 materialization request。对于 `descriptor_only` RAG 页，native leaf ID 必须为 `None`。

## 6. 请求生命周期

### 6.1 文档摄取

离线摄取流程读取文档，按稳定 token 边界切片，生成 `GlobalRAGPage`，然后写入全局目录。第一版可以使用普通文本 embedding；不要一开始就训练注意力对齐描述符。需要同时保存 chunk 的 token 数和 hash，以便物化结果可以复现和失效。

### 6.2 请求开始

请求建立独立 `session_id` 和物化表。系统先用用户查询在全局 RAG 目录做一次文本级召回，选出的文档 chunk 在主 Prompt prefill 之前拼接或通过 chunked prefill 编码。这是第一版最稳妥的路径，因为它天然产生位置和上下文正确的 K/V。

初始 prefill 完成后，IceCache 会像处理普通 Prompt 一样对这些 token 建立 DCI 叶。物化表随后把对应 RAG `page_id` 标记为 `cpu_kv` 或 `gpu_kv`，并记录其 token 区间和逐层叶映射。

### 6.3 Decode 查询

每个被索引层收到 GQA query 后，候选平面分别查询当前会话 Prompt 分区和全局 RAG 描述符分区，再合并候选。选择器在总页数、每头页宽、Prompt 页下限和资源成本分位数约束下最大化效用。

如果选择结果全部具有 native leaf ID，可以直接进入现有 diff/recall 路径。如果结果包含 `descriptor_only` RAG 页，本 token 不应等待完整编码；系统记录物化请求，并用下一个最优的已物化候选补位。物化完成后，该 RAG 页才能参与后续 token 的实际召回。

### 6.4 会话结束

会话结束时必须取消未完成的物化任务，释放其 CPU/GPU KV 页，删除 session delta 索引，并清理 `(session_id, page_id)` 状态。全局 RAG 描述符和 token 不受影响。

## 7. Session-local materializer 的实现方案

建议新增 `source/icecache/materialization.py`，并定义以下接口：

```python
class RAGMaterializer(Protocol):
    def submit(self, session_id: str, page_ids: list[str]) -> list[str]: ...
    def poll_ready(self, session_id: str) -> list[str]: ...
    def resolve_native_leaves(
        self, session_id: str, page_id: str, layer_idx: int
    ) -> np.ndarray | None: ...
    def release_session(self, session_id: str) -> None: ...
```

第一版不要在 decode token 中间运行任意长度的模型编码。建议采用两个阶段。

阶段 A 在请求 prefill 前完成文本 RAG：先召回文档，再把 token 插入 Prompt。这条路径应当首先实现，因为它能够建立端到端质量基线。

阶段 B 才尝试异步、阶段性物化：仅在 agent tool boundary、轮次边界或显式 pause 点追加文档 chunk，并重新执行受控的 chunked prefill。需要确认 append 后的位置 ID、attention mask 和 KV 页序列与把文档正常放进上下文时等价。

物化完成后，不应手工猜测叶 ID。应让 `_DCI_add` 或等价的公开包装更新 DCI，随后从 `token2node` 中解析新增 token 对应的实际叶。由于 DCI add 可能重组已有叶，必须把旧的 `selected_page_idx`、`ccc` 和 `cc2gp` 一致性纳入测试。

## 8. CandidateProvider 的闭环升级

当前 `CandidateProvider.query_leaf_ids` 只能返回已有 native leaf。闭环版本建议采用两阶段接口，而不是让一个数组同时表达两种对象：

```python
@dataclass
class CandidatePlan:
    ready_leaf_ids: np.ndarray
    deferred_rag_pages: tuple[str, ...]
    fallback_leaf_ids: np.ndarray
    selection_metrics: dict[str, float]
```

运行时先使用 `ready_leaf_ids`。若某个选中的 RAG 页未物化，则提交 `deferred_rag_pages`，并使用 `fallback_leaf_ids` 保持每个 KV 头的固定宽度。下一次查询时，provider 通过 materializer 将已经完成的 RAG 页解析为 native leaf。

不要直接修改 C++ scatter 接收字符串页 ID；字符串身份只属于控制面，进入 native 层之前必须解析为经过范围验证的 `int32` leaf ID。

## 9. 描述符路线

第一阶段使用普通文本 embedding 作为全局 RAG 描述符，Prompt 页使用真实 K 质心。这两个空间并不天然对齐，因此第一阶段可以保留双路检索并在归一化分数后合并，而不要宣称是同一个向量空间。

第二阶段收集真实 query、叶成员、native DCI 命中和注意力贡献，训练一个小型投影，使文本 chunk embedding 和 Prompt 页表示落入共享空间。训练目标可以从简单的对比学习开始：被原生注意力使用或被 oracle 判为有用的页作为正例，同层同头的其他页作为负例。

第三阶段再比较文本 embedding、K 质心、学习投影和跨层复用四个方案。必须分别报告 retrieval recall 和最终生成质量，因为高页召回不保证 K/V 本身有效。

## 10. 机会约束成本模型

每个候选的成本不是独立随机变量。PCIe 拥塞、批处理和同一 decode step 的多层 recall 会同时影响多个页，因此候选应使用对齐的场景向量，而不是为每页独立采样。

建议分别记录以下成本：描述符查询时间、在线编码时间、CPU→GPU recall 时间、GPU 页占用时间，以及因同步造成的实际阻塞时间。部署保证必须在单一固定配置的 trace 上进行 train/calibration/test 或时间切分；跨配置日志只能用作压力场景。

若校准样本不足以支持目标分位数，系统必须返回 vacuous/unknown，而不是默认为满足 95% 保证。

## 11. 推荐 Coding 顺序

### Milestone 1：文本级 RAG 基线

新增全局文档页存储、文档切片和 embedding 接口。实现 query→top-k chunk→Prompt 拼接，并在不开启统一 KV 逻辑时验证答案质量。验收条件是同一数据集上可以稳定复现 no-RAG 与 text-RAG 两个结果。

### Milestone 2：物化状态机

新增 `materialization.py`，实现按会话存储的状态转换、失败回滚和 session cleanup。暂时只支持 prefill 前物化。验收条件是两个会话引用同一全局页时状态完全独立，结束一个会话不会影响另一个。

### Milestone 3：RAG token 到 DCI 叶的映射

在正常 prefill 中记录每个 RAG chunk 的 token 区间，并结合每层 `token2node` 得到相关叶集合。验收条件是每个记录的叶 ID 都在对应头的 `num_leaves` 范围内，而且其成员至少包含该 chunk 的一个 token。

### Milestone 4：闭环 CandidatePlan

扩展 provider 返回 ready、deferred 和 fallback 三类结果。未物化 RAG 页只能进入 deferred；ready 和 fallback 必须经过 `validate_leaf_ids`。验收条件是在 RAG 物化较慢或失败时，生成仍可使用 Prompt fallback 正常继续。

### Milestone 5：影子模式与闭环模式对照

对同一批请求同时记录原生 DCI、影子联合选择和闭环联合选择。先要求闭环模式在只注册 Prompt 页时与原生路径行为等价，再逐渐加入 RAG 页。验收条件包括 leaf recall、logit 差异、生成答案、TTFT、TPOT 和内存峰值。

### Milestone 6：真实成本校准

使用固定模型和硬件收集 event-level trace，校准选择器的成本场景和 conformal 上界。验收条件是独立测试段的覆盖率达到目标，并报告时间漂移下的覆盖率。

### Milestone 7：学习型共享描述符

只有在前六个阶段稳定后再训练共享描述符。该阶段的验收条件不是 ANN recall 单项提升，而是在相同页预算下提高任务质量或降低物化/recall 成本。

## 12. 必须具备的测试

单元测试需要覆盖跨层和跨头页拒绝、其他会话 Prompt 页不可见、每头页下限、Prompt 页下限、成本场景对齐、provider 结果范围、重复 leaf ID、物化状态转换、失败回滚和 session cleanup。

集成测试需要覆盖纯 Prompt 等价路径、RAG 只进入 shadow 不改变 logits、prefill 前 RAG 物化、同一文档的双会话隔离、DCI add 后旧页映射一致性，以及 materializer 失败时 fallback 不崩溃。

端到端测试至少比较以下系统：无 RAG 的完整上下文、普通文本 RAG、普通 RAG 加原生 IceCache、影子统一选择、闭环统一选择，以及仅作为风险实验的离线深层 KV 注入。

## 13. 指标

质量指标应采用具体任务的 EM、F1、ROUGE 或 judge score，同时测量相对完整上下文的 logit KL 或 token 一致率。检索指标包括 RAG chunk recall、native leaf recall、每层每头候选 recall 和 materialization hit rate。

系统指标包括 TTFT、TPOT、decode P50/P95/P99、编码时间、DCI 查询时间、recall 阻塞时间、PCIe 字节数、CPU/GPU 峰值页数，以及机会约束的实际违约率。

所有结果必须区分随机初始化、合成描述符和预训练模型；前两者只能证明接线或算法行为，不能证明 RAG 质量。

## 14. 已知实现陷阱

当前原生 DCI 的结果缓冲区容量与 `prop_to_retrieve * num_leaves` 有关，超出容量可能造成进程内存破坏，因此不能只依赖 Python 异常。M-DCI 包装必须继续保留动态容量上限。

当前编译的 CUDA prefill 内核没有覆盖所有 GQA group size 和 head dimension。Qwen2.5-0.5B 的 group size 7，以及测试中的 head dimension 64，都会在 kernel dispatch 阶段失败。选择模型前应先验证内核组合，或补充 C++/CUDA dispatch 实例。

短上下文刚超过 GPU 页预算时，`n_dci_pages - page_topk` 可能不为正。运行时已经增加明确错误，但生产实现更理想的行为是延迟启用 DCI，直到存在至少一个合法候选槽。

Observer 回调可能运行在工作线程中，必须自行保证并发安全；重型磁盘写入和模型编码不应直接阻塞 callback。

最后，不能把“统一逻辑选择”误写成“一个模型级 ANN 查询”。IceCache 的正确边界是每个被索引层、每个 KV 头的独立空间，层复用只能作为需要单独消融的近似。

## 15. 完成定义

当且仅当以下链路成立时，可以称为第一版统一 RAG/IceCache 原型：全局文档可以被摄取和检索；选中的文档能够在当前会话上下文中产生合法 KV；物化后的页能够被解析为逐层逐头 native leaf；联合选择器能在成本和每头宽度约束下混选 RAG 与 Prompt 页；生成质量至少不低于普通 RAG 基线；系统能够报告真实 TTFT、TPOT、内存和预算违约率；任意会话结束后都不会泄漏 KV 页或私有索引状态。

在此之前，系统应被描述为“统一候选控制面与运行时接线原型”，而不是已经完成的 KV-native RAG。

## 16. 当前状态与 M5+ 接入边界

截至当前工作树，M1–M4 的控制面主体已经存在：`materialization.py` 提供全局
RAG 页、会话私有物化状态和 `CandidatePlan`；`materialization_provider.py`
负责把已物化页解析成 native leaf，并把未物化页放入 deferred 通道；
`prefill_materialization.py` 与 `boundary_materialization.py` 已经定义初次 prefill
和轮次边界 append 的事务语义。因此，剩余工作的核心并不是再实现一套状态机，
而是把这些协议接入真正拥有模型执行、DCI 更新和请求生命周期的运行时。

目前仍不能称为完成的原因有三点。`retrieval.py` 已经以延迟导出的方式提供唯一
的 `materialization.CandidatePlan` 类型契约，并修复了 leaf ID 在 int32 转换前
可能发生的范围别名；统一选择器的跨 head 身份、页数预算、confidence 排序和
session cost resolver 隔离也已经补齐。尽管如此，`MaterializingCandidateProvider` 在内部
生成完整的 `CandidatePlan`，但 `InferState._DCI_query` 当前只消费最终的
`ready_leaf_ids` 数组；完整 plan 只通过 provider 的 `last_plan` 暴露，deferred
页则通过 `on_deferred` 回调产生副作用。这个接口能运行，却尚未形成由运行时明确
观察、统计并绑定到本次查询的两阶段控制面。其次，现有 prefill helper 只定义了
事务边界，真正的模型 forward、position ids、attention mask、KV append、
`_DCI_add` 和 `token2node` 注册仍需由 adapter/runtime 接入。最后，当前证据主要
是控制面自检、CPU 微基准和合成模型实验，尚缺生产模型上的真实 TTFT、TPOT、
CPU/GPU 内存以及双会话在线隔离数据。

### 16.1 协调前必须固定的所有权

M5+ 开工前应先把以下所有权写死，否则同一个事件可能被两侧处理两次。推荐继续
采用“provider 负责生成 plan 和发布 deferred，`InferState` 只消费 native leaf
并记录 plan”的模式，因为现有 `MaterializingCandidateProvider._record_plan` 已经
保证 deferred callback 的异常不会击穿 decode。采用该模式后，`InferState`
可以读取或接收与本次查询原子关联的 plan、发布 observer 指标，但不得再次提交
`deferred_rag_pages`。如果改成由 `InferState` 提交 deferred，则必须先移除
provider 的 `on_deferred` 副作用，并提供原子返回 `(effective_leaf_ids, plan)` 的
新接口；不能同时保留两条提交路径。

模型 adapter 应当独占 prefill/append 的提交权，因为只有它知道实际送入模型的
token、绝对 position、attention mask、当前上下文长度以及 `_DCI_add` 何时完成。
materializer 只记录这些事实，不能自己猜测或推进上下文长度。边界 append 成功后，
adapter 必须用 `BoundaryReport.n_tokens` 更新会话长度；如果 append 已提交但 leaf
注册失败，则该页进入 doomed/fallback，绝不能自动重试并把相同 chunk 再追加一次。
session 关闭时也必须先停止该 session 的 decode/deferred 生产，再依次释放运行时
KV/DCI 资源与控制面记录，不能只调用 `RAGMaterializer.release_session` 就宣称无泄漏。

### 16.2 推荐的实际接入顺序

第一步只做 plan 可观测性，不改变 logits。`retrieval.py` 的 canonical
`CandidatePlan` 类型契约已经完成；待协调的工作是让 `InferState` 在一次 provider
调用之后拿到与该调用对应的 plan，逐层保存或发送 observer 事件。此时保持
`apply_rag=False`，验证返回
给 native 路径的 leaf 与 fallback 逐元素相同，同时确认每个 deferred 页只入队
一次。若仍通过 `last_plan` 交接，则 provider 必须是单请求、单调用方实例；更长期
的接口应原子返回 effective leaf 与 plan，避免并发时的 last-writer-wins 歧义。

第二步接初次请求的真实 RAG prefill。检索和选择必须发生在模型 prefill 之前，
adapter 使用 `PrefillMaterialization.prepare` 返回的 `prompt.token_ids` 原样执行
forward；不能在外面重新套模板。每个已索引层完成 `_DCI_add` 后，用真实
`token2node` 和 `num_leaves` 调用 `register_layer`，其中首次 prefill 的
`token_offset` 应由运行时核验为 `n_sink_pages * page_size`，而不是任何 RAG 页的
起点。全部层注册成功后才能 `finish()`；任何一层失败都应回到 Prompt fallback。

第三步再接阶段 B 的 deferred 排空。只有 agent/tool/turn boundary 等显式安全点
可以调用 `RequestBoundaryMaterializer.materialize`；decode token 中间不得触发模型
编码。encode callback 需要以事务方式追加 token、运行 chunked prefill、更新 DCI，
并返回逐层 `(token2node, num_leaves)`。先用单会话证明 deferred→append→ready 的
闭环，再开启两个 session 同时引用同一 global page，确认两边的 span、leaf map、
队列、失败和释放完全独立。

仓库内的 `benchmark/agent_cache/materialization_boundary_probe.py` 已经覆盖这一
阶段的 CPU 控制面闭环：相同 global page 在两个 session 的不同绝对位置得到不同
leaf map，释放其中一个 session 后另一个仍然可查询。该 probe 的 encode callback
仍然合成 `token2node`，所以它只证明编排和隔离；把 callback 替换为 adapter 的
真实 chunked prefill + `_DCI_add`，正是 M5+ 尚待协调的运行时工作。

第四步才进行 live A/B。至少保留 `no-rag`、`rag-shadow`、`rag-live` 和
`plain-rag` 四个独立进程的串行对照；TTFT 必须覆盖文本召回、选择、prompt 拼接、
真实 prefill 和 deferred 提交，TPOT 则从首 token 后单独统计，并区分冷物化和稳态。
每次请求同时记录 GPU allocated/reserved 峰值、CPU RSS、KV/DCI 页数、deferred
深度、materialization hit rate 和 plan/fallback 调用次数。双会话实验除性能外还要
在释放其中一个 session 后继续查询另一个 session，以证明不存在共享 deep KV、
leaf map 串扰或错误的缓存失效。

### 16.3 M5+ 验收闸门

只有在 shadow 模式 leaf 与原生 fallback 逐元素一致、未物化页从未进入 native
数组、真实 prefill 的 token span 能逐层映射回正确 DCI leaf、失败路径不会重复
append、双会话互不影响，并且 TTFT/TPOT/内存原始数据能够按请求复现时，才可以
打开 `apply_rag=True` 做质量实验。若生产模型或 CUDA dispatch 暂时不可用，可以
继续完善控制面测试和测量脚本，但结果必须标为 synthetic/wiring evidence，不能把
它写成真实 RAG 性能结论。
