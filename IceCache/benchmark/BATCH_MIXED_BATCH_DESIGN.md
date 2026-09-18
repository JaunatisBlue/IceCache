# 阶段 4：混批（prefill chunk 与 decode token 共处一次 forward）

日期：2026-09-18 ｜ 前置提交：`0e90ac9`（阶段 1/2/2b/3/5）｜ 机型：A100 80GB PCIe，Llama-3.1-8B-Instruct，fp16
相关：`BATCH_PARALLELIZATION_PLAN.md`（§3 阶段 4 的定位）、`BATCH_CODE_FLOW.md`、`BATCH_DECODE.md`

## 实施状态

| 步骤 | 内容 | 状态 | 验证 |
|---|---|---|---|
| A | 页池：双向分配 + 增长语义 + 容量派生 | **已完成** | `test_page_pool_two_ended.py` 17/17 PASS；GPU 复验 uniform/skewed/sequential 全 exit=0，无回归，skewed 默认参数即可跑 |
| C1 | 抽出 `icecache_continuation_layer`（serial/batch 共用） | **已完成** | `tests/test_b1_chunked_continuation.py` + `test_c1_sparse_continuation.py` → 8 passed, 1 xfailed |
| C2 | 受限 row 集 `_decode_slots`（6 处） | **已完成** | decode-only 路径逐 token 相同 |
| C3 | `mixed_attention_forward`（按行分派） | 待实现 | — |
| C4 | `step_mixed` + `begin_chunked_prefill` | 待实现 | — |
| C5 | `batch_mixed_probe.py`（等价性 / 阻塞性） | 待实现 | — |

已提交：`c59ffbd`（A + C1 + C2）。

---

## 0. 一句话

把「新请求的 prefill chunk」与「老请求的 decode token」放进**同一次 model forward**，使长 tool result 到达时不再独自占满一个 forward、从而不阻塞其它请求的 decode。
依赖两件事：**页池的预留契约**（已完成）与 **batch 级的按行分派**（本方案）。

---

## 1. 现状取证（决定「不要从零造 chunk」）

**单请求侧的 chunk 机器早就存在**：

| 组件 | 位置 | 说明 |
|---|---|---|
| `ForwardMode.CONTINUATION_PREFILL` | `infer_state.py:64` | 显式的 continuation 模式 |
| `_prepare_continuation` / `_finish_continuation` | `infer_state.py:570 / 632` | `use_dci == False`（全驻留）的 chunk |
| `_prepare_continuation_sparse` / `_finish_continuation_sparse` | `infer_state.py:786 / 880` | `use_dci == True` 的 chunk；**明确 `raise NotImplementedError("sparse continuation supports batch_size == 1")`** |
| `continuation_sdpa_batched` | `infer_state.py:745` | chunk 的因果 paged prefill |
| `_icecache_continuation` | `adapter/modeling.py:346-475` | 逐层主体：`retrieve_blocks` → `_pack_resident` → `_write_chunk_into_stage` → `continuation_sdpa_batched`，末层 `_finish_continuation_sparse` 写 KV |
| 测试 | `tests/test_b1_chunked_continuation.py`、`tests/test_c1_sparse_continuation.py` | 已有覆盖 |

**batch 侧完全没有**：`grep -n continuation batch.py` → **0 命中**。
`batch.py` 的 attention 入口是 `forward_mode` **整体二选一**（`adapter/modeling.py:492-496`：`BATCH_PREFILL` → `prefill_attention_forward`，否则 → `attention_forward`）。

⇒ 阶段 4 = **把已有的 per-request chunk 机器接进 batch 的按行分派**，而不是新写一套 chunk 逻辑。

---

## 2. 三个必须先解决的问题

### 2.1 页池：碎片与容量（已完成，必须先做）

**实测到的碎片是真的**（新增探针指标 `gpu_pool`）：GPU 池空闲 **544** 页时，**最大连续 run 只有 32** 页
⇒ 任何需要 >32 页连续空间的**一次性整段 prefill** 都会失败，尽管总空闲足够。这就是 `Not enough contiguous free pages in pool` 的来源。

三处修法（见 §4.1）：

1. **双向分配**：decode 的单页取自**低位**、prefill 的 run 取自**高位**，两个前沿相向而行，只有真的没页了才失败。
   （顺带把「取哪个页」从 CPython `set.pop()` 的实现细节变成显式契约。）
2. **`_prefill_alloc_n_pages` 的增长语义**：旧代码每次调用都**替换** `c2p` —— 既泄漏被替换的那一段，又让后续段落落在不相邻的物理页上，序列会被静默切成两块。
   现在预留一次、后续只确认容量，并且「不能再增长」会**显式报错**而不是静默丢掉上一段。
3. **容量派生**：CPU 池是**每请求一个、且被该请求的所有层共享**，所以容量必须是 `n_layers × 每层页数`；
   同时**过大会明显变慢**（池用 `pin_memory=True`）：skewed 下 10240 页 → prefill 3.81 s，62976 页 → **6.00 s**（实测）。
   ⇒ 紧派生 `1.5 × n_layers × ceil(Lmax / page_size)`（实测峰值用量比池大小低约 25%，且比旧码手填的参数更快：3.711 s vs 3.910 s）。

> **一个反而让方案变简单的发现（读码得到，决定了 C3 的写法）**：
> **chunk 路径其实不需要连续 run。** `_prepare_continuation_sparse` 是
> `kvc.decode_alloc_1_token(self.alloc_page)` **逐页**增长（`infer_state.py:838-839`），
> 而页是通过 `c2p` 间接寻址的，所以物理上不连续完全不影响正确性。
> 连续 run 只是**一次性整段 prefill** 的实现选择（`prefill_alloc_n_tokens` → `alloc_contiguous_pages`），不是语义要求。
>
> ⇒ 因此：**chunked prefill 应当走逐页增长（复用 continuation 机制），而不是给 chunk 做预留**；
> `reserve_prefill_pages` 的定位收窄为「整段预分配 + 把不能增长这件事变成显式报错」；
> 双向分配的价值收窄为「保证一次性 prefill 的 run 不被 decode 前沿吃掉」。
> 这条也说明：**阶段 4 不会因为碎片而受阻** —— 前提是 C3 走 continuation 而不是重新引入连续分配。

### 2.2 排布：为什么必须 flatten

一次 forward 的输入是一个矩形张量。若按 `[B, L]` 排布、L 取 chunk 长度，则 **decode 行也被迫算 k 个位置**，
把混批要省的东西原样加回去。因此必须像 vLLM / SGLang 那样**展平**：

```
hidden_states : [1, total, hidden]           total = Σ 段长 = n_decode + Σ k_i
段            : decode 段长 1；prefill 段长 k_i
qo_indptr     : 段边界（FlashInfer 的两个 wrapper 都吃这个）
position_ids  : decode 段 = 该请求的 seq_len；prefill 段 = chunk 起点 + arange(k)
```

代价：`cache_position = arange(total)` 会让 HF 的 causal mask 变成 `total²`（本路径不使用该 mask，只付分配成本）。

### 2.3 按行分派

每层，**先 decode 段、后 prefill 段**：

| 段 | 每层做什么 |
|---|---|
| decode | `batch_query` → `_recall_prepare`/`_recall_commit` → 共享 `_handler` 的稀疏 attention（现有机器，只把「活动行」限制到 decode 行） |
| prefill | `retrieve_blocks` → `scatter_pages` → `_pack_resident` → `_write_chunk_into_stage` → `continuation_sdpa_batched`（现有机器，逐请求调用） |

**顺序是设计的一部分，不是实现细节**：decode 段先执行，老请求的 decode 才不会在同一层内被新请求的建树/搬运拖住；
prefill 段自己的 `_dci_future` barrier 只影响它自己。

---

## 3. 改造点与依赖关系

```
A. 页池（双向分配 + 增长语义 + 容量派生）                 ← 已完成并验证
   │  它保证一次性 prefill 的连续 run 不被 decode 前沿吃掉；
   │  chunk 本身走逐页增长（见 §2.1 的发现），所以不依赖连续 run
   ▼
C1. modeling 抽出每层函数 icecache_continuation_layer(...)  ← 已完成并验证
   │  serial 与 batch 共用一份「chunk 如何进入 KV 并被注意力读取」
   │  验证：现有 tests/test_b1_chunked_continuation.py 与
   │        tests/test_c1_sparse_continuation.py → 8 passed, 1 xfailed
   ▼
C2. batch 的行元数据 + 受限 row 集                          ← 已完成（验证中）
   │  batch_query / decode_attention / build_attention_metadata /
   │  _prepare_decode / _cleanup_decode_handlers / _finish_decode
   │  六处从 self.active_indices 改为 self._decode_slots
   │  不作混批时它是恒等变换 ⇒ decode-only 路径逐位不变
   ▼
C3. mixed_attention_forward（按行分派 + 输出装配）           ← 待实现
   ▼
C4. step_mixed（flatten 输入 / position_ids / lm_head gather / per-row barrier）← 待实现
   │  lm_head 的 gather 位置由 C3 的段表决定，所以必须在 C3 之后
   ▼
C5. batch_mixed_probe.py（同进程等价性 + 阻塞性测量）        ← 待实现
```

**为什么是这个顺序**：A 是初始 prefill 能拿到连续 run 的物理前提，也是容量不再随手填错的保证；
C1 是「单一实现」的前提（不然 serial 与 batch 各有一份 chunk 主体，后续任何修改都会分叉），而且它**必须在 C3 之前做完并跑通现有测试**，
否则 C3 一旦出错就分不清是新代码的问题还是抽取引入的问题；
C2 是 C3 能存在的机械前提（不改受限 row 集，decode 段会把正在 prefilling 的请求当成 decode 行去查它的 DCI 树）；
C4 的 lm_head 位置依赖 C3 的段表。

---

## 4. 关键处理步骤

### 4.1 A —— 页池（已实现，见 `kv_cache.py` / `batch.py`）

```python
# kv_cache.PagePool：两个前沿
def alloc_page(self):                      # decode：最低空闲页
    if self._reclaimed_low: return self._reclaimed_low.pop()
    while self._next_low < self.n_max_pages:
        page_id = self._next_low; self._next_low += 1
        if page_id in self._free_ids:
            self._free_ids.discard(page_id); return page_id
    raise RuntimeError("page pool exhausted (no free page left)")

def alloc_contiguous_pages(self, num):     # prefill：最高的连续 run，升序返回
    ...  sorted_ids = sorted(self._free_ids, reverse=True) ...

def free_page(self, page_id):              # 游标已越过的页走回收表
    self._free_ids.add(page_id)
    if page_id < self._next_low: self._reclaimed_low.append(page_id)

def max_contiguous_free_pages(self): ...   # 报碎片，而不只是报空闲数
```

```python
# kv_cache.KvCache：预留契约
def reserve_prefill_pages(self, n_pages, alloc_page=None): ...   # 第一个 chunk 之前调用一次
def _prefill_alloc_n_pages(self, n, alloc_page=None):
    if self.n_real_pages >= n: return            # 已在预留范围内：只推进 seq_len
    if self.n_real_pages: raise RuntimeError("... reserve ... before the first chunk")
    ...                                          # 否则按老路径分配整段
```

```python
# batch.BatchInferState：进入 forward 之前就把「装不下」说清楚
def require_prefill_capacity(self, real_lens): ...   # 同时检查空闲页数与最大连续 run
```

**验收**：`benchmark/test_page_pool_two_ended.py` 17/17 PASS（取向 / 回收 / 越游标复用 / 预留契约 / 缺预留时报错 / 池空语义）。
**注意**：碎片假设**没有**在合成扰动下复现成瓶颈（已分配的 run 不会被 decode 拿走，因为 decode 只从 free 集合取页）；
真正的失败原因是**容量**，已由派生修好。双向分配保留的理由是确定性与「reversed run 与 decode 前沿分离」的显式契约。

### 4.2 C1 —— 抽出每层函数（`adapter/modeling.py`）

把 `_icecache_continuation` 里 `kvc = state.kv_caches[cur_id]` 到 `attn_output = attn_output.reshape(bsz, q_len, -1)`
之间的主体抽成模块级函数，serial 与 batch 共用：

```python
def icecache_continuation_layer(state, layer_idx, query_states, key_states, value_states,
                                position_embeddings, q_len):
    """一层 continuation chunk 的注意力，输入是已投影、未 RoPE 的 q/k/v。

    query_states/key_states: [1, q_len, n_heads, head_dim]（RoPE 前）
    value_states:            [1, q_len, n_kv_heads, head_dim]
    返回 [1, q_len, n_heads * head_dim]。
    """
    kvc = state.kv_caches[layer_idx]
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    if state.use_dci:
        eids, nr = state.retrieve_blocks(layer_idx)
        if int(nr.sum()) > 0:
            state.scatter_pages(layer_idx, eids, nr)
        stage, indices, indptr, last_len = state._pack_resident(layer_idx, q_len)
        state._cont_pack[layer_idx] = (stage, indices, indptr, last_len)
        counts = state._resident_valid_counts(layer_idx)
        per_head = counts.sum(0)
        ps = state.page_size
        pages = (per_head + q_len + ps - 1) // ps
        state._cont_per_head[layer_idx] = per_head
        state._cont_run_start[layer_idx] = torch.cumsum(pages * ps, 0) - pages * ps
        state._write_chunk_into_stage(layer_idx, key_states, value_states)
        out = state.continuation_sdpa_batched(
            layer_idx, query_states, stage, indices, indptr, last_len)
        state._cont_kv[layer_idx] = (key_states, value_states)
    else:
        state.append_paged_kv_cache(layer_idx, key_states, value_states)
        out = state.prefill_sdpa(layer_idx, query_states)
    return out.reshape(1, q_len, -1)
```

`_icecache_continuation` 改成调用它；**行为必须逐位不变**，由现有的
`tests/test_b1_chunked_continuation.py` / `tests/test_c1_sparse_continuation.py` 守住。

### 4.3 C2 —— 行元数据与受限 row 集（`batch.py`）

段的唯一真相是一次 forward 内冻结的列表，layer 0 建立、层间只读：

```python
@dataclass(frozen=True)
class _Segment:
    kind: str        # "decode" | "prefill"
    slot: int        # 槽位号（不是行号）
    offset: int      # 在 flatten 后的 [1, total] 里的起点
    length: int      # 1 或 chunk 长度

    @property
    def last(self):  # lm_head 取该段的哪个位置
        return self.offset + self.length - 1
```

```python
    @property
    def _decode_slots(self):
        """decode 分支这一次 forward 要服务的槽位。

        没有混批时就是 active_indices；混批时只有 decode 段。像 batch_query /
        decode_attention / build_attention_metadata / _prepare_decode /
        _finish_decode 这些方法原本硬编码 self.active_indices，必须改读这里，
        否则会把正在 prefilling 的请求当成 decode 行去查它的 DCI 树。
        """
        return (self._mix_decode_slots if self._mix_decode_slots is not None
                else self.active_indices)
```

替换点（5 处，逐一）：`batch_query`、`decode_attention`、`build_attention_metadata`、`_prepare_decode`、`_finish_decode`。

### 4.4 C3 —— `mixed_attention_forward`

顶层结构与 `_icecache_continuation` 逐层同构（layer 0 建立段表与 prepare、末层 finish），只是段表由 batch 持有：

```python
    def mixed_attention_forward(self, attn, hidden_states, position_embeddings,
                                output_attentions=False):
        if output_attentions:
            raise ValueError("batch attention does not return attention weights")
        bsz, total, _ = hidden_states.shape          # 必须是 [1, total, hidden]
        if bsz != 1 or total != self._mix_total:
            raise ValueError("mixed batch expects one flattened row")
        layer_idx = attn.layer_idx
        if layer_idx != self._next_layer:
            raise RuntimeError(f"expected attention layer {self._next_layer}, got {layer_idx}")

        if layer_idx == 0:
            self._prepare_decode()                    # 只对 decode 段 begin_forward
            for seg in self._prefill_segments:
                state = self.states[seg.slot]
                if state.use_dci:
                    state._prepare_continuation_sparse(1, seg.length)
                else:
                    state._prepare_continuation(1, seg.length)

        cfg = attn.config
        query, key, value = self._project_qkv(attn, hidden_states, cfg)   # [total, h, d]
        out = torch.empty(1, total, self.n_qo_heads * self.head_dim,
                          dtype=hidden_states.dtype, device=self.device)

        # decode 段先做：老请求的 decode 不被新请求的建树/搬运拖住（同一层内）
        if self._mix_decode_slots:
            rows = self._mix_decode_offsets
            q = query[rows].unsqueeze(1).contiguous()
            k = key[rows].unsqueeze(1).contiguous()
            v = value[rows].unsqueeze(1).contiguous()
            out[0, rows] = self.decode_attention(layer_idx, q, k, v).reshape(len(rows), -1)

        # prefill 段：逐请求调用共享的层函数（与 serial 路径同一份实现）
        for seg in self._prefill_segments:
            sl = slice(seg.offset, seg.offset + seg.length)
            out[0, sl] = icecache_continuation_layer(
                self.states[seg.slot], layer_idx,
                query[sl].unsqueeze(0), key[sl].unsqueeze(0), value[sl].unsqueeze(0),
                position_embeddings, seg.length)[0]

        if layer_idx == self.n_layers - 1:
            for seg in self._prefill_segments:
                state = self.states[seg.slot]
                if state.use_dci:
                    state._finish_continuation_sparse(1, seg.length)
                else:
                    state._finish_continuation(1, seg.length)
            self._finish_decode()                     # 只对 decode 段

        if self._step_active:
            self._next_layer += 1
        return attn.o_proj(out), None
```

**不变量**（写成断言，混批路径全程有效）：

1. 段表在一次 forward 内冻结；`_mix_total == Σ length`。
2. decode 段 `length == 1`，且其 `position_ids` 等于该请求当前 `seq_len`。
3. prefill 段的 `length` 必须等于该请求本次 chunk 的 token 数，且 chunk 起点等于该请求已 prefilled 的 token 数。
4. 同一 slot 不得同时出现在 decode 段与 prefill 段。
5. prefill 段**每个 chunk 都要**在末层调 `_finish_continuation*`（它负责在末层写 KV），
   但 ``state._finish_prefill`` / 建树只在**首个 chunk 的状态建立**时或整段完成时按需处理 ——
   逐页增长与窗口轮转由 `_prepare_continuation_sparse` 负责，与 decode 的窗口生命周期同构。
6. `_dci_future` 只属于 prefill 段；decode 段不得 await 它。
7. prefill 段必须已有 `_prepare_prefill` 建立过 `kv_caches`（见 §4.5 的 `begin_chunked_prefill`）。

### 4.5 C4 —— `step_mixed`

```python
    def step_mixed(self, model, decode_input_ids, prefill_chunks):
        """一次 forward 同时推进 decode 行与 prefill chunk。

        decode_input_ids: [n_decode, 1]，按 active_indices 顺序（混批里通常是全部活跃槽位）
        prefill_chunks:   [(slot, tokens_1d)]
        """
        # 1) 校验：段表、页池预留、chunk 连续性、同一 slot 不重复
        # 2) flatten 输入
        #    input_ids   = cat([decode tokens] + [chunk tokens])[None]
        #    position_ids= cat([seq_len per decode] + [chunk_start + arange(k) per chunk])[None]
        #    cache_position = arange(total)
        # 3) lm_head patch：按段取 last 位置（decode 段 last == offset）
        # 4) model(input_ids, position_ids=..., cache_position=..., use_cache=False)
        # 5) 返回 [n_seg, 1, vocab]
```

`_mix_segments` 由 `decode_slots + prefill_chunks` 生成；`_prepare_prefill` / `reserve_prefill_pages` 由调用方在该请求注册时完成。

### 4.6 验收（C5）

`benchmark/batch_mixed_probe.py`，同进程内两路对照（必要时由一次 forward 与两次 forward 组成）：

1. **等价性**：混批（n 行 decode + 1 个 k-token chunk）与「先单独跑到同一状态再分别 forward」产生**相同的 logits**。
   判据用 logits（噪声基线为 0），**不要**用 `generated_token_ids`——跨进程比对不是有效判据（见阶段 5 的实测）。
2. **不阻塞**：chunk 到达的那一步，decode 行的每步延迟相对纯 decode 的增幅。
3. **页池健康**：`max_contiguous_free_pages` / `n_alloc_run_failures`。

---

## 5. 风险与未解问题

1. **`_prepare_continuation_sparse` 的窗口推进是 Python 逐 token 循环**（q_len × n_layers 次 `decode_alloc_1_token`）⇒ chunk 越大，这段 CPU 越贵；chunk 大小需要实测权衡。
2. **prefill 段仍是逐请求一次 kernel 调用**（`continuation_sdpa_batched` 是 bsz=1 语义）⇒ 混批省的是「不单独占一个 forward」，不是 kernel 数量。
3. **`_pack_resident` 逐层有 host/device 往返**，混批会把这些 CPU 工作塞进 decode 的同一层里 ⇒ 必须测量它是否吃掉了收益。
4. **窗口/树的不变量**：`_prepare_continuation_sparse` 已保证「树与窗口不相交」，但混批下 decode 段与 prefill 段交替调用 `retrieve_blocks`/DCI，需要确认该不变量在**同一 forward 内**仍成立。
5. **CUDA graph 仍不可用**（Python 控制流）。
6. `use_dci == False`（整段驻留）的路径走 `_prepare_continuation`，其 `page budget` 会在 chunk 超出时报错 —— 混批里要给这类请求回落成「独占 forward」。

---

## 6. 与「明确不做」的边界

**不把单请求的跨层复用（`n_reuse_layers` / `check_reuse`）搬进 batch**：它靠 `DCI.reuse_copy_node` / `reuse_update_node` + `c2p` 偏移别名实现「多层共享同一份 KV 页」，
与 batch 的「一个物理页只属于一个 (请求, 层)」不变量（`validate_ready`）冲突，是 KV 布局级改动，
收益上限却只有搜索那 ~60 ms/步。理由与代价见 `BATCH_PARALLELIZATION_PLAN.md` §3 阶段 4。
