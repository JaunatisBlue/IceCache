# IceCache `batch` 分支调用流程走读

> 只读代码走读，未修改任何源码，未跑 GPU/实验。
> 远端分支：`/home/yx/IceCache` 当前 `batch` 分支（未切分支、未 commit）。
> 对应源码树：`/home/yx/IceCache/IceCache/source/icecache/`、`benchmark/`、`3rdparty/flashinfer/`。
> 参考实现：`/home/yx/IceCache/code_ref/{vllm,sglang}`（vllm 为 V1 架构；HF transformers 在远端 conda 环境内）。

---

## 0. 背景与全局不变量

`batch` 分支的目标是：**把若干已经"各自独立 prefill 过"的 IceCache 请求，放进一个固定容量的 slot 集合里，做一次同步的 batched decode**。每个 slot 要么是 *active*（持有参与 decode 的请求），要么是 *free*。`active_indices`（按升序排列的 slot 下标）贯穿整条链路，是 batch 维度的事实来源。

关键设计要点（来自 `batch.py:1-11` 文档字符串与 `__init__`）：

- `capacity = len(states)` 固定；`batch_size = len(active_indices)` 随 retire/admit 变化（`batch.py:50,108-115`）。
- **没有调度器**：`admit` 由调用方显式调用（`batch.py:10,662`），因此不是 continuous batching。
- 支持任意 `B>=1`、变长（每行带自己的 `position_ids`）、请求退出/复用（`batch.py:7-10`）。
- 同一物理 GPU 页同一时刻只能属于一个 `(请求, 层)` —— 由 `validate_ready` 的 `seen` map 与 `retire` 的 `freed` set 双重保证（`batch.py:259-300,640-654`）。
- 树（DCI，存 CPU 页）与 window（GPU 上的滑窗页）**不相交**：prefill 时 `ev_gpi[:, :ns]=-1; ev_gpi[:, -budget+ns:]=-1`，即只把 sink 之后、semantic 之前的页 evict 到树（`infer_state.py:986-993`）。

下面所有"第 k 行"默认指 *active 请求的第 k 行*，由 `active` 顺序（`batch.py:311` 等）确定，**不是 slot 号**（slot 号仅在 `active == 0..B-1` 时重合，一旦 retire 非尾部 slot 就会分叉，`batch.py:412-414` 有显式注释）。

---

## 1. 入口到出口的分层调用链（`文件:行号`）

完整一次 `batch.step` 的调用栈（decode 模式）：

1. **驱动**：`batch_decode_probe.py:281 run_steps` → `batch.step(model, ids)`（`batch.py:185`）。
   - `ids = torch.stack([tokens[i] for i in active])`，形状 `[batch_size, 1]`，按 `active_indices` 顺序取行（`batch_decode_probe.py:288-289`）。
2. **`BatchInferState.step`**（`batch.py:185`）：拿 `_forward_lock`（排它），置 `_step_thread`，转 `_step_impl`。
3. **`BatchInferState._step_impl`**（`batch.py:196`）：
   - 校验 `input_ids` 形状 `[batch_size,1]`、设备（`batch.py:202-205`）。
   - 计算 `expected_position = [states[i].seq_len for i in active_indices]`（每行绝对位置，`batch.py:211-213`）。
   - 用 `icecache_state(model, self)` 上下文管理器把本 batch 绑定到已 patch 的模型（`batch.py:228-229`），见 §adapter。
   - 构造 `cache_position = [max_seq]`（长度 1，标量），`position_ids = expected_position`，调 `model(input_ids, use_cache=False, position_ids=..., cache_position=...)`（`batch.py:240-244`）。
   - 校验 `_next_layer == n_layers`（每层都走过 attention，`batch.py:245-246`）。
4. **HF `LlamaForCausalLM.forward`** → 逐层 `LlamaDecoderLayer` → 调用被 patch 的 attention。
5. **`_icecache_attn_forward`**（`adapter/modeling.py:478`）：因为 `infer_state` 是 `BatchInferState`，分派到 `infer_state.attention_forward(self, hidden_states, position_embeddings, output_attentions)`（`adapter/modeling.py:491-493`）。
6. **`BatchInferState.attention_forward`**（`batch.py:557`）：
   - 每层 `attn.q_proj/k_proj/v_proj` → `apply_rotary_pos_emb`（`batch.py:581-592`，rotary 见 `adapter/modeling.py:42`）。
   - **仅 layer 0**：`self._prepare_decode()`（`batch.py:569-570`）。
   - `output = self.decode_attention(layer_idx, query, key, value)`（`batch.py:594`）。
   - **最后一层**：`self._finish_decode()`（`batch.py:602-603`）；并 `_next_layer += 1`（`batch.py:604-605`）。
7. **`BatchInferState.decode_attention`**（`batch.py:512`）：
   - `results = self.batch_query(layer_idx, query)`（`batch.py:518`）→ 见 §4。
   - 每个 active 请求：`self._recall_one(i, layer_idx, results[pos])`（`batch.py:520`）→ recall + scatter，见 §3。
   - `state.append_paged_kv_cache(layer_idx, key[pos:pos+1], value[pos:pos+1])`（`batch.py:524`）→ 把本步新 token 写入 GPU paged KV。
   - `indices, indptr, last, valid = self.build_attention_metadata(layer_idx)`（`batch.py:527`）→ CSR 组装，见 §5。
   - `self._handler.begin_forward(indptr, last, n_qo_heads, n_kv_heads, head_dim, page_size, data_type)`（`batch.py:532`）。
   - `output = self._handler.forward(query, self._pool.buffer, indices, page_valid_entries=valid, dci=any_dci)`（`batch.py:535`），`self._handler.end_forward()`（`batch.py:539`）。
   - **仅 layer 0 的 offload 分支**：若有任何 active 请求的 `offload_win_flag[-1]`，等待 `decode_backup_stream` 后对每个 layer 调 `offload_win_page_to_DCI(l)` 并清 flag（`batch.py:540-547`）。
8. **回到 `attention_forward`**：`o_proj` → 返回 `(output, None)`（`batch.py:599-606`）。
9. **`BatchInferState._finish_decode`**（`batch.py:550`）：遍历 active 调 `state._finish_decode(1)`，重置 `_decode_handlers_armed`，累加计时（`batch.py:551-555`）。`state._finish_decode` 自身对每个 `decode_handler_tab` 调 `handler.end_forward()`（`infer_state.py:895-897`）。
10. **回到 model forward** → 拿到 `out.logits`，probe 用 `out.logits[row,-1]` 取下一 token（`batch_decode_probe.py:295`）。

> 注：`decode_attention` 里做实际计算的是 `batch._handler`（`batch.py:83` 单例），而 `state.decode_handler_tab` 的 `begin_forward/end_forward` 在 `_prepare_decode/_finish_decode` 里也被调用（`infer_state.py:884-893,896`）——两者并存（详见 §7 P2-3）。

---

## 2. prefill 路径（作为对照，并说明为何现在串行）

### 2.1 单请求 prefill 的调用链（`_icecache_prefill`）

`_icecache_prefill`（`adapter/modeling.py:52`）被 `enable_icecache` 时 patch 进 attention；由 `forward_mode == INITIAL_PREFILL` 分派（`adapter/modeling.py:509-517`）：

1. layer 0：`state._prepare_prefill(bsz, q_len)`（`adapter/modeling.py:72` → `infer_state.py:305`），重置 `kv_caches/_page_log/selected_page_idx/c2p` 等。
2. `q/k/v` 投影 + reshape + `apply_rotary_pos_emb`（`adapter/modeling.py:102-124`）。
3. 稀疏预算下切片 key 做 DCI 投影 `projected`（`adapter/modeling.py:130-134`）。
4. `kvc.prefill_alloc_n_tokens(q_len, state.alloc_page)`（`adapter/modeling.py:137` → `kv_cache.py:234`）：`seq_len += q_len`，按需分配连续 GPU 页（`_prefill_alloc_n_pages`，`kv_cache.py:220`，`alloc_contiguous_pages`，`kv_cache.py:36`）。
5. `state.append_paged_kv_cache(cur_id, key_states, value_states)`（`adapter/modeling.py:139` → `infer_state.py:1352`）。
6. 异步提交 `prefill_evict_extra_pages_wrapper`（`adapter/modeling.py:146-149` → `infer_state.py:1537`）：内部 `prefill_evict_extra_pages`（`:1541`）→ 若超预算则 `prefill_backup_pages`（`:1560`）把 prompt 尾部页拷到 CPU，再 `_DCI_first_call`（`:1580`，建树，`:911`）→ 写 `_page_log`（`:1586`）→ `tmp_cpu_kvc.clear()`（`:1589`）。
7. `state.prefill_sdpa(cur_id, query_states)`（`adapter/modeling.py:151` → `infer_state.py:1615`）。
8. 最后一层：`_dci_future.result()` 同步、`state._finish_prefill(bsz, q_len)`（`adapter/modeling.py:177-181`）。

### 2.2 为什么是串行 + 被排除在吞吐外

- 驱动侧：`batch_decode_probe.py:226 for i, state in enumerate(states):` **逐请求** prefill，每次用 `icecache_state(model, state)` 绑定**单个** `infer_state` 再 `model(...)`（`batch_decode_probe.py:232-243`）。
- 单请求假设贯穿 prefill：
  - `enable_icecache(model, infer_state=state)` 把模型 attention 绑定到**一个** state（`adapter/modeling.py:570,584`）；batch 化需要一次 forward 同时绑定多个 state。
  - `_icecache_prefill` / `_DCI_first_call` / `_DCI_add` 大量硬编码 `b=0`、注释 "!! Here assume batch size = 1"（`infer_state.py:1011,1038,1046,1146,1156,1158`；`page_address_buffer[cur_id][b, i, ...]` 中 `b` 恒为 0）。
  - `_check_prefilled` 要求 `state.batch_size == 1`（`batch.py:162-163`）。
- prefill 的 KV 写入是**逐 token 顺序** `append_paged_kv_cache`，且每个请求独占一段 GPU **连续页**（`alloc_contiguous_pages`，`kv_cache.py:36`）；探测脚本据此给池子按 `n_layers * sum(pages)` 定容（`batch_decode_probe.py:170-185`）。
- 因此 probe 的计时把 prefill 单独记在 `prefill_seconds`，decode 吞吐只统计 `step_ms`（`batch_decode_probe.py:243,293,391`）。**"批量 prefill、再分别建树"的改造点见 §7 P1-2。**

---

## 3. DCI 查询的内部链路

`decode_attention` 里每个 active 请求都走：`batch_query`（收集候选）→ `_recall_one`（回 GPU）→ `append`（写新页）→ attention。

### 3.1 `_DCI_query`：用 query 走树拿候选（`infer_state.py:1169`）

1. `num_neighbours = n_dci_pages - layer2topk[cur_id]`（`infer_state.py:1175`）。
2. 若 `nn_idx_override is None`：调 `self.dci_db[cur_id].query(...)` 走 DCI 树（`infer_state.py:1199-1209`）。
   - **`nn_idx` 形状 `[n_qo_heads, 2, n]` 的由来**：DCI `query` 返回的 1D 数组长度为 `n_qo_heads * 2 * num_neighbours`（`batch.py:451` 校验；`mdci_batch.c:92` 分配 `heads*neighbours*2`）。其中 `*2` 来自 DCI 返回的两段并列数组：**页索引** 与 **页内偏移**（`mdci_batch.c:141-145`：`memcpy(out, nearest[0], copied)` 后 `memcpy(out+neighbours, nearest[0]+neighbours, copied)`）。reshape 成 `[n_qo_heads, 2, neighbours]`（`infer_state.py:1215`）。
   - `nn_idx_0 = nn_idx[:,0,:].reshape(n_kv_heads, ratio, -1)` = 候选**页索引**；`nn_idx_1` = 偏移（仅 prefetch 用，`infer_state.py:1216-1219`）。
3. `ratio>1` 时做跨实例去重 `first_k_unique`（`infer_state.py:1222-1226`）。
4. **增量 diff（`diff_pages_by_head`）**：
   - 首帧：`selected_page_idx[cur_id] = nn_idx_0`，`recall_idx = nn_idx_0`，`evicted_idx = c2p[0, ns:ns+n]`，并把 `cc2gp[b, head, padded] = evicted_idx`（`infer_state.py:1237-1243`）。`cc2gp`（cpu_cache_page→gpu_pool_page）是树页↔GPU 页的映射。
   - 后续帧：`recall_idx, evicted_idx, out_idx = DCI.diff_pages_by_head(nn_idx_0, selected_page_idx[cur_id], ccc[b,head,padded].numpy(), cc2gp[b].numpy())`（`infer_state.py:1245`）。`ccc`（change-count cache，`:1001,1071-1100`）记录每页自上次的改动；`selected_page_idx` 更新为 `out_idx`（`:1246`）。
   - 返回 `(evicted_idx, recall_idx, evict_num)`（`infer_state.py:1253`），其中 `evict_num = (recall_idx>=0).sum(1)`（`infer_state.py:1250`）。

### 3.2 `recall`：CPU→transit 拷贝（`infer_state.py:1364`）

- `recall(layer_idx, 0, rids, nr)`（`batch.py:501` → `infer_state.py:1364`）。
- 按 `(head, rid)` 填 `_src_address_buffer[counter:counter+nr[i]] = page_address_buffer[layer_idx][b, i, rids[i,:nr[i]]]`（`infer_state.py:1380-1382`）。**契约**：`page_address_buffer[layer_idx][b, i, rid]` 是 DCI 叶页 `rid`（头 `i`）在 CPU 上的地址；`_src_address_buffer` 必须按 `(head, rid)` 顺序填充，与 `copy_to_buffer` 的 `offset_s = n_kv_heads*page_size*head_dim` 目标布局一致（`infer_state.py:1386-1390`）。
- 在 `c2g_stream` 上 `DCI.copy_to_buffer` 把 CPU 页拷到 `cpu_transit_buffer`，再 `copy_` 到 `cuda_transit_buffer` / `cuda_cast_buffer`（`infer_state.py:1384-1400`）。

### 3.3 `scatter_pages`：写入 GPU resident 槽位（`infer_state.py:1457`）

- `_cpp.scatter_pages(self.cuda_cast_buffer, self.kv_caches[layer_idx].pool.buffer, eids, nr)`（`infer_state.py:1459`）。
- **重要**：C++ 侧**不尊重 `nr=0`**——它按 `eids` 的完整宽度搬运，若 `nr` 全 0 仍会写入 **stale 的 cast-buffer 内容**，污染 resident 槽（`infer_state.py:286-288` 注释；同 `batch.py:509`）。**因此调用方必须自己判断 `int(nr.sum()) > 0` 才调**（`batch.py:509-510`、`infer_state.py:289-290`、`infer_state.py:1306-1312` 的 `retrieve_blocks` 也遵守该契约）。

### 3.4 `page_valid_entries` 的 semantic 段更新

- `_recall_one` 在 recall 后刷新：`state.page_valid_entries[layer_idx][ns:ns+n] = dci_db[layer_idx].get_valid_entries(selected_page_idx[layer_idx]).T`（`batch.py:504-507`），再 `c2g_stream.synchronize()`。
- 该 tensor 形状 `[n_real_pages, n_kv_heads]`：每个页、每个头记录该页**有效（live）token 数**，供 FlashInfer decode kernel 的 `page_valid_entries` 使用（见 §5 / decode.cuh）。

### 3.5 三元组 `(eids, rids, nr)` 含义

- `eids`：要填充的 GPU semantic 槽位页号，`[n_kv_heads, k]`（`infer_state.py:1304`、`recall` 的 `evicted_idx` 经 `c2p` 得到）。
- `rids`：DCI 候选叶页号（`recall_idx`），`[n_kv_heads, ?]`。
- `nr`：每个头要 scatter 的页数 `[n_kv_heads]`；`nr.sum()` 为本次实际搬运的页数（`infer_state.py:1374`）。

---

## 4. native 批量查询（`BatchInferState._batch_query_impl` + `mdci_batch.c`）

### 4.1 Python 侧组装（`batch.py:378`）

- serial 后端：直接 `for pos,i in active: self._query_one(i, layer_idx, query_states[pos], ...)`（`batch.py:408-415`），每个请求在自己的 `_DCI_query` 里走树+diff。
- native 后端（`batch.py:416-464`）：
  - 收集 `dci_slots` = 所有 active 且 `use_dci`、预算已满（`n_real_pages>=budget`）的请求（`batch.py:417-420`）。
  - 对每个 slot 组装四个列表（`batch.py:421-441`）：
    - `capsules`：`db._dci_inst`（DCI 实例指针，需 `_orig_indices is None`，`:431-433` 拒绝 remap）。
    - `queries`：`query_states[pos].reshape(-1, head_dim).float().cpu().numpy()`（`batch.py:435-436`）。
    - `neighbours`：`n_dci_pages - layer2topk[layer_idx]`（`batch.py:437-438`）。
    - `fields`：`max(int(seq_len*search_ratio), 30)`（`:439-440`）。
  - 调 `self._native.batch_query(capsules, queries, neighbours, fields, ratio, query_threads)`（`batch.py:443-445`）。
  - 回喂：`results[pos] = state._DCI_query(0, layer_idx, query_states[pos].cpu().transpose, nn_idx_override=candidates)`（`batch.py:459-462`），由 `_DCI_query` 做 `diff_pages_by_head` / `selected_page_idx` 更新与 `(eids,rids,nr)` 产出。

### 4.2 C 侧 `batch_query`（`mdci_batch.c:40`）

- 解析 `capsules/queries/neighbours/fields/ratio/threads`（`mdci_batch.c:43-44`）。
- 校验每请求 `db->num_inst*ratio` 与 query 形状 `[heads, dim]`（`mdci_batch.c:78-85`），`heads = num_inst*ratio`（`mdci_batch.c:78`）。
- **关键调度**：`total_heads = Σ_i (req[i].db->num_inst * ratio)`（`mdci_batch.c:98`），即把所有 *(请求×head)* 任务铺平成一条队列；用 `#pragma omp parallel for num_threads(threads) schedule(dynamic)`（`mdci_batch.c:110`）**动态领取**。
  - **为何不按请求静态分线程组**：不同请求的 `num_inst*ratio`（头数）、树大小、以及逐请求的 `field_of_view`/`neighbours` 都不同（`batch.py:437-440` 逐请求算）。静态按请求分组会因各请求工作量不均而严重失衡；`schedule(dynamic)` 让 worker 空闲即认领下一个 `(请求,head)` 任务（`mdci_batch.c:106-118` 注释）。
  - 循环内 `flat → (r, h)` 解码：`h -= req[r].db->num_inst*ratio`（`mdci_batch.c:113-117`）；`h/ratio` 选 DCI 实例（`mdci_batch.c:120`）；`dci_query` 带 `parallel_level=0`（不在 OpenMP 区内再开嵌套并行，`mdci_batch.c:121`）。
- 输出 `results[i]` 为 `heads*neighbours*2` 的 int32 数组（`mdci_batch.c:92-97`），交回 Python 侧 reshape 为 `[n_qo_heads,2,n]`（§3.1）。

### 4.3 native 能否处理"树结构/长度不同的新成员"（admit）

- 能。`_batch_query_impl` native 路径对每个 active slot **独立**构建 `capsules/neighbours/fields`，`neighbours` 与 `fields` 都按 `state` 自身算（`batch.py:437-440`）；OpenMP 队列按 `(请求,head)` 平铺，天然支持不同树规模/长度（§4.2）。`admit` 进来的新请求走的是**自己独立的 prefill**（probe `batch_decode_probe.py:313-326`），新建 DCI 树且 `_orig_indices is None`，满足 `batch.py:431` 的约束。故 `native batch_query` 对异构新成员无硬限制。

---

## 5. 每个环节的数据形状 + 关键不变量

| 环节 | 张量 | 形状 | 说明 / 行号 |
|---|---|---|---|
| 输入 | `input_ids` | `[B, 1]` | 按 active 行（`batch.py:202`） |
| 位置 | `position_ids` | `[B, 1]` | 每行 `states[i].seq_len`（`batch.py:211`） |
| 位置 | `cache_position` | `[1]` | 标量 `max_seq`（`batch.py:243`） |
| 投影后 | `query` | `[B, n_qo_heads, 1, head_dim]` | `attention_forward` 转置后（`batch.py:588-593`） |
| 投影后 | `key/value` | `[B, n_kv_heads, 1, head_dim]` | 按 active 行（`batch.py:589-590`） |
| GPU 池 | `pool.buffer` | `[n_max_pages, 2, n_kv_heads, page_size, head_dim]` | HND 布局（`kv_cache.py:17,75-87`） |
| 页映射 | `c2p` | `[1, n_real_pages]` | cache 页→pool 页（`kv_cache.py:108`） |
| 树↔GPU | `cc2gp` | `[1, n_kv_heads, kvc_capacity]` | cpu 页→gpu pool 页（`infer_state.py:999`） |
| batch meta | `indices` | `[total_pages]` | 各请求 `c2p[0,:n]` 拼接（`batch.py:327`） |
| batch meta | `indptr` | `[B+1]` | `cumsum(n_real_pages)`（`batch.py:323-326`） |
| batch meta | `last` | `[B]` | 各请求 `last_page_len`（`batch.py:328`） |
| batch meta | `valid` | `[total_pages * n_kv_heads]` | `page_valid_entries` 拼接（`batch.py:341`） |
| DCI | `nn_idx` | `[n_qo_heads, 2, n]` | 页索引+偏移（`infer_state.py:1215`） |
| DCI | `selected_page_idx` | `[n_kv_heads, n]` | 当前 resident 树页（`infer_state.py:1240,1246`） |
| DCI | `ccc` | `[B, n_kv_heads, kvc_capacity]` | 改动计数缓存（`infer_state.py:1001,1071`） |
| 回写 | `eids/rids/nr` | `eids:[n_kv_heads,k]`、`rids:[n_kv_heads,?]`、`nr:[n_kv_heads]` | recall 三元组（`infer_state.py:1304,1253`） |
| 输出 | `out.logits` | `[B, 1, vocab]` | 按 active 行取下一 token（`batch_decode_probe.py:295`） |

**不变量（务必遵守）**：

1. **按行张量 = 活跃请求，不是 slot 号**：`input_ids/query/key/value/out.logits/results` 第 k 行 = 第 k 个活跃请求（`batch.py:412-414,522-523`；`active_indices` 驱动，`batch.py:108`）。slot 号仅在 `active==0..B-1` 时重合。
2. **同一物理 GPU 页同一时刻只属一个 `(请求,层)`**：`validate_ready` 用 `seen` map、`retire` 用 `freed` set 双保险（`batch.py:290-299,640-654`）；`admit` 额外检查页不相交（`batch.py:685-702`）。
3. **树与 window 不相交**：prefill 把 `ev_gpi[:, :ns]` 与 `[:, -budget+ns:]` 置 -1 只 evict semantic 段（`infer_state.py:986-996`）。
4. **`cache_position` 保持长度 1**：见 §7 P0-3 / 论证——HF 的 causal mask 在 `use_cache=False` 且 `past_key_values=None` 时 `kv_length = input_embeds.shape[1] = 1`（`masking_utils.py:729`），且该 mask 在 patch 后的 attention 里**被丢弃**（`_icecache_attn_forward` 完全不读 `attention_mask`，`adapter/modeling.py:478-540`）；真实位置由 `position_ids` 经 RoPE 提供（`batch.py:238-243`）。

---

## 6. 与 vLLM / SGLang 的职责对照表

> 证据取自已克隆的 `code_ref` 真实代码（vllm 为 V1 架构；HF 行为取远端 conda 环境内 transformers）。行号为 `code_ref/vllm/...` 相对路径。

### 6.1 请求状态

| 概念 | vLLM（V1） | SGLang | 我们（`batch.py`/`infer_state.py`） |
|---|---|---|---|
| 一个请求 | `vllm/v1/request.py` `Request`；状态机 `RequestStatus`（`FINISHED_STOPPED=3` `:145`、`FINISHED_ABORTED=5` `:147`） | `python/sglang/srt/managers/io_struct.py` `BatchTokenIDOutput`（`:1435`） | 每个活跃请求 = 一个 `InferState` 实例（`batch.py:51,122-123`）；`active_indices` 里的 slot |
| 一组请求 | `Request` 列表 + `vllm/v1/core/scheduler.py`（`:599` 置 `FINISHED_STOPPED`） | scheduler 维护 `Req` 队列 | `BatchInferState.states`（固定容量 slot 数组，`batch.py:51`） |
| 完成/退出 | `RequestStatus.FINISHED_*`（`v1/request.py:145-147`） | `release_kv_cache(req, tree_cache)`（`srt/mem_cache/common.py:276`） | `retire(index)`（`batch.py:611`）——**只释放 GPU 页** |

### 6.2 页池所有权与分配

| 概念 | vLLM（V1） | SGLang | 我们 |
|---|---|---|---|
| 页池 | `vllm/v1/core/kv_cache_manager.py` `KVCacheManager`（`class :19`），`self.block_pool`（`List[KVCacheBlock]` `:53`），`free_block_queue`（`FreeKVCacheBlockQueue` `:59`） | `python/sglang/srt/mem_cache/base_prefix_cache.py` `BasePrefixCache.free_kv_row`（`class :13, :435`）、`evict`（`class :451`） | `PagePool`（`kv_cache.py:10`）：`buffer`（`_free_ids` 集合 `:24`）、`alloc_page` pop（`:33`）、`free_page` add（`:55`）；`KvPool`（`kv_cache.py:64`）；`BatchInferState._pool` 共享（`batch.py:53`） |
| 分配/释放 | `KVCacheManager.allocate_slots`（`kv_cache_manager.py:157`）、`free(request)`（`:270`）、`can_allocate` 由 `free_block_queue.num_free_blocks`（`:99, :207, :234, :381`） | `BasePrefixCache.evict_for_alloc`（`base_prefix_cache.py:454`），`release_kv_cache`→`tree_cache.free_kv_row`（`common.py:276,345`） | `KvCache.decode_alloc_1_token`/`prefill_alloc_n_tokens`（`kv_cache.py:204,234`）、`alloc_contiguous_pages`（`kv_cache.py:36`）、`InferState.alloc_page`（`infer_state.py:1595`，超池时释放 evicted 页 `:1606`） |
| 块表（页→物理） | `vllm/v1/worker/block_table.py` `BlockTable`（`class :13`）：`block_table_np`（`block_ids` 数组 `:52`）、`num_blocks_per_row`（`row 计数 :43`）、`add_row`（`append_row :43-56`） | radix 树 / `ReqToTokenPool`（`common.py:315` `free_kv_row_segments`） | `KvCache.c2p`（cache 页→pool 页，`kv_cache.py:108`）；逐层 `kv_caches[layer].c2p`（`batch.py:272,315`） |

**对照关系**：
- vLLM `block_pool.allocate/free` ↔ 我们 `PagePool.alloc_page/free_page`（`kv_cache.py:33,55`）。
- vLLM `BlockTable.block_table_np`（每请求一组 block id）↔ 我们 `c2p`（每请求 `c2p[0,:n_real_pages]`，`batch.py:315`）。
- vLLM `KVCacheManager.free_block_queue.num_free_blocks` 做 `can_allocate` ↔ 我们 `PagePool.n_free_pages` / `alloc_contiguous_pages`（`kv_cache.py:30,36`）。

### 6.3 batch 元数据（CSR / 块表 / 槽映射）

| 概念 | vLLM | 我们 `build_attention_metadata`（`batch.py:305`） |
|---|---|---|
| query 偏移 | `AttentionMetadata.query_start_loc`（`vllm/attention/backends/placeholder_attn.py:106`，`list(accumulate(query_lens, initial=0))` `:344`） | `indptr` 是**页** CSR（非 token），见下 |
| 序列长度 | `seq_lens`（`placeholder_attn.py` / `gpu_input_batch.py:80 num_tokens`） | `seq_lens`（每 active 请求，`batch.py:117-120`） |
| 块表展开 | `block_tables`（`vllm/attention/ops/paged_attn.py:32` 参数） | `indices` = 所有 active 请求的 `c2p[0,:n]` 拼接（`batch.py:327`） |
| 槽映射 | `slot_mapping`（`vllm/worker/model_runner.py` 等） | `last`（每请求 `last_page_len`，`batch.py:328`）+ `valid`（`page_valid_entries`，`batch.py:341`） |
| 输入批 | `InputBatch`（`vllm/v1/worker/gpu_input_batch.py:48`：`block_table`、`num_tokens`、`add_row` `:242`） | `BatchInferState` 本身即输入批：`active_indices`/`seq_lens`/`build_attention_metadata` |

**对照关系（映射）**：
- `indptr`（`batch.py:323`，`cumsum(n_real_pages)`，`长度=B+1`）↔ vLLM `query_start_loc`/`seq_start_loc`（页级变体；注意我们这里是**页** CSR 不是 token CSR，因为 FlashInfer decode 以页为单元）。
- `indices`（`batch.py:327`，扁平页列表）↔ vLLM `block_tables` 的展平（`paged_attn.py:32`）。
- `last` ↔ vLLM 的 `slot_mapping`/每序列尾部页长度。
- `valid`（`page_valid_entries`）↔ vLLM 没有等价物（这是 IceCache 的 sparse DCI 专属：标记每页每头的有效 token 数，供 `decode.cuh` 掩码）。

### 6.4 请求退出与资源回收

| 概念 | vLLM | SGLang | 我们 |
|---|---|---|---|
| 退出标记 | `RequestStatus.FINISHED_STOPPED/ABORTED`（`v1/request.py:145-147`），`scheduler.py:599` 置位 | `release_kv_cache`（`common.py:276`） | `retire(index)`（`batch.py:611`） |
| 释放块表 | `KVCacheManager.free` → `free_block_queue` 回收（`kv_cache_manager.py:270`） | `tree_cache.free_kv_row` / `evict`（`base_prefix_cache.py:435,451`） | `retire` 遍历 `c2p` 把 GPU 页 `free_page` 回池（`batch.py:640-654`） |
| CPU 池 / 树 | vLLM 无 CPU offload 树（仅 GPU KV） | SGLang radix 树在 CPU（`BasePrefixCache`） | **我们只回收 GPU 页**；CPU `KvCache`（`cpu_kv_caches`）与 DCI 索引**不释放**——只 `del states[index]` 丢引用，等调用方/GC（`batch.py:655-658` 注释明确"CPU/DCI resources become unreferenced and are reclaimed by the caller/GC"） |

> **明确差异**：vLLM/SGLang 的回收在调度器内闭环（free 块表同时回收对应显存/树）。我们 `retire` **只**释放绑定的 GPU `PagePool` 页；CPU 池页与 DCI 树随 `InferState` 对象一起由调用方持有，batch 不主动回收（`batch.py:616-619,655-658`）。这意味着长期 continuous-batching 场景下 CPU 池与 DCI 会有泄漏（见 §7 P1-1）。

---

## 7. 走读发现的疑点 / 待改点清单（P0 / P1 / P2）

> 给后续三个 agent（写码 / review / 实验）的输入。所有结论均带来源行号。

### P0（正确性 / 必须确认）

- **P0-1 `recall`/`scatter_pages` 在 `nr.sum()==0` 时仍可能被误调。**
  `scatter_pages` 的 C++ 实现不尊重 `nr=0`，会把 stale cast-buffer 写入 resident 槽（注释 `infer_state.py:286-288`）。当前**仅** `batch.py:509`、`infer_state.py:289`、`infer_state.py:1306` 三处显式 `if int(nr.sum())>0` 守护。`_recall_one` 之外任何新调用点（尤其未来 prefetch 路径 `estimate_select_recall`，`infer_state.py:1402`）必须同样守护，否则静默污染 KV。建议：把守护下沉进 `scatter_pages` 本身。
- **P0-2 `cache_position` 长度 1 的安全性依赖"HF causal mask 被丢弃"这一事实。**
  论证（已确认）：`use_cache=False` 且未传 `past_key_values` 时，HF `create_causal_mask` 的 `kv_length = input_embeds.shape[1] = 1`（`masking_utils.py:729`）；而 patch 后的 `_icecache_attn_forward` 完全不读 `attention_mask`（`adapter/modeling.py:478-540`），真实注意力由 FlashInfer 用 `page_valid_entries`/`indptr` 完成。所以 `cache_position=[1]` 不参与任何有效计算。风险点：若某次改动让 HF 真实 attention 路径生效（例如 `attn_implementation` 非 patch），长度-1 的 `cache_position` 会对变长请求给出错误因果掩码。建议回归测试锁定"patch 后 causal_mask 不被消费"。
- **P0-3 双重 `begin_forward` / 两套 decode handler。**
  `decode_attention` 用 `batch._handler`（`batch.py:83,532,535`）做实际计算；而 `_prepare_decode/_finish_decode` 又对每个 `state.decode_handler_tab[b]` 调 `begin_forward/end_forward`（`infer_state.py:884-893,896`）。batch 路径里后者的 `forward` 从不被调用。需确认这是遗留/为 prefetch 预留，还是会导致 workspace 双分配或状态错位（特别是 `kv_decode_indptrs_tab`/`kv_last_page_lens` 与 `batch._handler` 的 `indptr/last` 是否一致）。

### P1（泛化 / 吞吐，需要动手）

- **P1-1 `B>2` 从未真正执行过（只跑过 B=2）。**
  数据结构本身是通用的（`active_indices` 驱动、`build_attention_metadata` 的 CSR、`_batch_query_impl` 的 `dci_slots` 都是任意 B）。但仍发现**隐含两请求/串行依赖**：
  - prefill 定容假设"retire 只释放原长连续段"，新 admit 请求不能更长，否则即使总空闲页够也找不到足够长**连续**段（`batch_decode_probe.py:174-184`，`alloc_contiguous_pages` `kv_cache.py:36`）。B 增大后碎片风险上升。
  - probe 默认 `batch-size=2`，且 retire 只测 slot 0（`batch_decode_probe.py:276`）；更一般的 `active != 0..B-1` 仅被这个特例覆盖。
  - `batch.py` 未找到硬 `==2` 断言；但 `_DCI_query`/`_recall_one` 都是逐 active 循环，未发现按 B 分叉的逻辑。**结论**：数据通路支持任意 B，但连续页分配与测试覆盖是 B>2 的真实瓶颈，需补 B∈{1,3,4,8} 的回归。
- **P1-2 prefill 是串行且被排除在吞吐外；"批量 prefill + 分别建树"改造点。**
  阻塞点（§2.2）：
  - `_icecache_prefill` / `_DCI_first_call` / `_DCI_add` 硬编码 `b=0`、注释 "assume batch size = 1"（`infer_state.py:1011,1038,1046,1146,1156,1158`）；`page_address_buffer[cur_id][b, ...]` 的 `b` 维度未泛化。
  - `_check_prefilled` 要求 `state.batch_size == 1`（`batch.py:162`）。
  - `enable_icecache` 一次只绑定一个 `infer_state`（`adapter/modeling.py:570,584`）；要批量 prefill 需一个能同时持有多请求的 "batch prefill state"，且 `attention_forward` 现在只 dispatch `DECODE` 给 `BatchInferState`（`adapter/modeling.py:491-493`），要新增 prefill 分支。
  - 最小改造：让 `KvCache`/`_DCI_*` 支持 `batch_size>1`（把 `b=0` 改为真实 `b`），并新增 `BatchInferState.prefill_step` 在一次 `model(input_ids=[B, q_len])` 内对每行分别 `prefill_alloc_n_tokens` / 建树。
- **P1-3 `retire` 不回收 CPU 池与 DCI 树（§6.4）。**
  连续 batching 长期运行会泄漏。`retire` 应可选地 `state.clear()` CPU 资源（`KvCache.clear` `kv_cache.py:242` 已存在，释放 `c2p` 页），并 `del state.dci_db` / `state._page_log`。当前注释明确这是调用方职责（`batch.py:655-658`），但若要做真 continuous batching，batch 层必须接管。

### P2（效率 / 清洁）

- **P2-1 `_prepare_decode` 里 per-state `decode_handler_tab` 的 `begin_forward` 在 batch 模式下是冗余分配**（P0-3 的伴随项）；统一到 `batch._handler` 可省一份 workspace。
- **P2-2 native `batch_query` 要求 `OPENBLAS_NUM_THREADS=1`**（`batch.py:74-75`）且 `producer_sha256` 必须与编译期一致（`batch.py:78-80`）。多线程 DCI + OpenBLAS 会争核，需在文档/CI 固化该约束。
- **P2-3 `query_states` 在 `_query_one` 里 `.cpu().detach().transpose(0,1)`**（`batch.py:359`）每请求一次 CPU 拷贝；native 路径在 `_batch_query_impl` 里也逐请求 `.cpu()`（`batch.py:435,461`）。对大 B 这是显著 host 往返，可改为一次性 `query_states[active].reshape(...).cpu()` 批拷贝。
- **P2-4 `validate_ready` 每次 step 前全量扫描所有页**（`batch.py:259-300`）O(B×pages×layers)，热路径上偏重；可改为只在 admit/retire/分配变更时增量维护 `seen`。

### 关于 §任务特别评估点的结论

- **B>2 未跑过**：数据通路通用，但连续页分配与测试覆盖是真实瓶颈（P1-1）。
- **prefill 串行**：由 `_icecache_prefill` 的 `b=0` 单请求假设与 `enable_icecache` 单 state 绑定导致（P1-2）。
- **admit 新成员 + native batch_query**：**可行**——native 路径逐请求独立组装 `capsules/neighbours/fields` 且 OpenMP 队列平铺异构任务（§4.2/§4.3）。
- **变长下 `cache_position` 长度 1**：**安全**，条件见 P0-2 论证（HF mask 被丢弃 + `kv_length=1`）。

---

## 8. 一次 `batch.step` 的文字时序图

缩进表示调用层级；`[锁]` = `_forward_lock`；`[GPU页]` = 触碰共享 GPU KvPool；`[DCI]` = 触碰 CPU 树 / CPU transit。

```
driver: run_steps()
└─ batch.step(model, ids)                        [锁] acquire _forward_lock, _step_thread=get_ident()
   └─ _step_impl()
      ├─ validate_ready()                        遍历 active: 校验页所有权 seen map            [GPU页]
      ├─ with icecache_state(model, batch):        绑定 model._icecache_infer_state = batch
      │  └─ model(input_ids, position_ids=per-row, cache_position=[max_seq])
      │     └─ HF LlamaModel.forward → 逐层 LlamaDecoderLayer
      │        └─ patched attn = _icecache_attn_forward
      │           └─ (isinstance BatchInferState) → BatchInferState.attention_forward(layer)
      │              │
      │              ├─ [layer 0 only] _prepare_decode()                  [锁]+[GPU页]
      │              │   └─ for each active i: state._prepare_decode(1)
      │              │      ├─ decode_alloc_1_token() → 分配新 GPU 页      [GPU页] (kv_cache.py:204)
      │              │      ├─ 更新 offload_win_flag / window backup        [GPU页]
      │              │      └─ state.decode_handler_tab[b].begin_forward()  (see P0-3)
      │              │
      │              ├─ q/k/v proj + apply_rotary_pos_emb (per-row position_ids)
      │              │
      │              ├─ decode_attention(layer, query, key, value)
      │              │   ├─ batch_query(layer, query)
      │              │   │   ├─ serial: for each active → _query_one → _DCI_query   [DCI]
      │              │   │   └─ native: 组装 capsules/queries/neighbours/fields
      │              │   │      └─ _native.batch_query()  # pragma omp parallel for schedule(dynamic)
      │              │   │         └─ 平铺 Σ(请求×head) 任务，动态领取，每任务 dci_query
      │              │   │      └─ 每结果回喂 state._DCI_query(nn_idx_override=...)  →(eids,rids,nr)  [DCI]
      │              │   │
      │              │   ├─ for each active i (pos=row):
      │              │   │   ├─ _recall_one(i, layer, results[pos])
      │              │   │   │   ├─ recall(layer,0,rids,nr)  填 _src_address_buffer → copy_to_buffer (CPU→transit→cast)  [DCI]+[GPU页]
      │              │   │   │   ├─ if int(nr.sum())>0: scatter_pages(eids,nr) 写 GPU resident 槽    [GPU页] (P0-1 守护)
      │              │   │   │   └─ 刷新 page_valid_entries[layer][ns:ns+n] = get_valid_entries(selected_page_idx)
      │              │   │   └─ append_paged_kv_cache(layer, key[pos:pos+1], value[pos:pos+1])  写本步新 token  [GPU页]
      │              │   │
      │              │   ├─ build_attention_metadata(layer) → (indices, indptr, last, valid)   [GPU页] CSR
      │              │   ├─ _handler.begin_forward(indptr, last, ...)
      │              │   ├─ _handler.forward(query, pool.buffer, indices, page_valid_entries=valid, dci=any_dci)  ← FlashInfer decode kernel (decode.cuh)
      │              │   └─ _handler.end_forward()
      │              │
      │              ├─ [layer 0 only] offload 分支: 若 offload_win_flag[-1]
      │              │   └─ wait_stream(decode_backup_stream); for l: offload_win_page_to_DCI(l)  (→ _DCI_add)   [DCI]+[GPU页]
      │              │
      │              └─ o_proj → return (output, None)
      │              │
      │              ├─ [last layer] _finish_decode()
      │              │   └─ for each active: state._finish_decode(1) → decode_handler_tab[b].end_forward()
      │              └─ _next_layer += 1
      │
      └─ (异常) _failed=True; _cleanup_decode_handlers() 释放已 begin 的 handler   [锁]
      ─ 返回 logits
   └─ release _forward_lock, _step_thread=None
driver: out.logits[row,-1].argmax() → 下一 token
```

**加锁点**：`_forward_lock` 在 `step` 进入时获取、退出时释放（`batch.py:187-194`）；`_prepare_decode` 内的页分配与 `decode_handler_tab.begin_forward` 发生在该锁保护区内（`batch.py:471-493`）。`retire`/`admit` 会先检查 `_step_active` 与 `_query_active` 拒绝并发（`batch.py:625-628,675-678`）。**碰 GPU 页的位置**：`decode_alloc_1_token`（`_prepare_decode`）、`recall`/`scatter_pages`（`_recall_one`）、`append_paged_kv_cache`、`build_attention_metadata` 后的 `forward`（`decode_attention`）、以及 offload 分支的 `offload_win_page_to_DCI`。**碰 CPU 树/transit**：`batch_query`/`_DCI_query`/`recall` 的 `copy_to_buffer`（在 `c2g_stream` 上异步）。

---

## 附录 A：本次走读未确认项

- 未运行 GPU/benchmark，所有性能数字（吞吐、step_ms）均未测量；本文档只描述代码结构与不变量。
- `code_ref` 仅含 vllm(V1)/sglang，无 transformers 源码；HF `cache_position`/`_update_causal_mask` 行为取自远端 conda 环境内 transformers（`masking_utils.py:745 create_causal_mask`、`modeling_llama.py:382`），已在 P0-2 给出精确行号。
- `_cpp.scatter_pages` / `DCI.copy_to_buffer` / `DCI.diff_pages_by_head` / `dci_db.get_valid_entries` 为编译扩展（`.so`），本文档依据 Python 侧调用约定（`infer_state.py`、`mdci_batch.c`）推断，未读其 C++ 实现。
