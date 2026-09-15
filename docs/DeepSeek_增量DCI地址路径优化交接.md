# DeepSeek 交接：Decode 增量 DCI 更新与 CPU 地址准备优化

## 1. 任务目标

当前分支：`sys-optimize`。

研究目标是降低 IceCache decode 阶段的增量 DCI 更新开销。每当 window page 填满并被驱逐到 CPU，系统会对从第 3 层开始的 30 个 layer 更新 DCI/tree-reuse 状态。当前 page size 为 16，因此该开销表现为每 16 个 decode token 出现一次明显的 latency spike。

当前优先任务不是修改 DCI 检索语义，也不是把 token 检索替换成 page centroid 检索。首先优化已经通过 profiling 定位到的 CPU 地址准备路径，要求保持地址逐元素完全一致，因此不应影响精度。

请勿切换分支、reset、覆盖现有修改或提交代码。工作区现有的 `infer_state.py` 修改是本轮新加的 profiling 插桩，应保留并在此基础上工作。

## 2. Decode 增量更新调用链

```text
page boundary
  -> decode_backup_win_page()
  -> offload_win_page_to_DCI(layer_idx)   # 30 layers
     -> pack/reshape window K/V
     -> _DCI_add(batch, layer, key, value)
        -> Tensor -> NumPy
        -> 准备旧 tree、leaf、ccc 元数据
        -> anchor layer: dci_db.add_query(..., do_query=False)
        -> reuse layer: DCI.reuse_update_node(...)
        -> 分配新增 CPU pages
        -> 构造新 leaf 对应的 CPU 地址
        -> dci_db.address_update(...)
```

主要代码：

- `IceCache/source/icecache/infer_state.py::_DCI_add`
- `IceCache/source/icecache/infer_state.py::offload_win_page_to_DCI`
- `IceCache/source/icecache/kv_cache.py::KvCache.__getitem__`
- `IceCache/source/icecache/kv_cache.py::PagePool.__post_init__`

配置 `n_reuse_layers=3` 时：

- 每 3 层中的 anchor layer 执行真正的 `dci_db.add_query` 增量插入；
- 中间两个 reuse layer 执行 `DCI.reuse_update_node`；
- 因此需要分别测量，不能把二者都称作“树插入”。

## 3. 已完成的 profiling 插桩

`InferState.get_profile_stats()` 已增加以下字段，仅在 `profile_dci=True` 时计时：

- `index_pack_ms_per_token`
- `index_numpy_ms_per_token`
- `index_prepare_ms_per_token`
- `index_native_insert_ms_per_token`
- `index_ccc_writeback_ms_per_token`
- `index_page_alloc_ms_per_token`
- `index_address_update_ms_per_token`
- `index_address_prepare_ms_per_token`
- `index_native_address_update_ms_per_token`
- `index_reuse_update_ms_per_token`

当前插桩已经通过：

```bash
/home/yx/miniconda3/envs/icecache/bin/python -m py_compile \
  IceCache/source/icecache/infer_state.py
git diff --check
```

## 4. 已获得的实测数据

环境：单张 NVIDIA A100 80GB，32 OpenMP threads，OpenBLAS 单线程，Llama-3.1-8B-Instruct，page size 16，page budget 64，reuse layers 3，promotion ratio 0.01。

三个 Qasper 样本的 context 长度约为 2k、4.5k、21k。共测量 98 个 decode token、180 次 layer-level index update，即 6 次 page boundary × 30 layers。

```text
decode TPOT                         145.466 ms/token
index update total                   4.997 ms/token
index address update overall         2.850 ms/token
index address prepare                2.320 ms/token
native DCI incremental insert        0.912 ms/token
CPU page allocation/capacity         0.606 ms/token
reuse_update_node                    0.363 ms/token  # 包含在 address overall 内
insert preparation                   0.196 ms/token
window KV pack                       0.171 ms/token
ccc writeback                        0.071 ms/token
Tensor -> NumPy                      0.048 ms/token
native DCI address_update             0.030 ms/token
```

由此折算：

- 每次 page boundary 跨 30 层的总增量更新约 `4.997 * 98 / 6 = 81.6 ms`；
- 每个 layer update 平均约 `81.6 / 30 = 2.72 ms`；
- 真正的 native `address_update` 只占极小部分；
- 最大区域是调用 native `address_update` 前的 Python 地址准备，约占整个增量更新的 46%。

单个 2k Qasper 样本也得到相同趋势：index update 4.47 ms/token，address overall 2.79 ms/token，说明最大项并非三样本偶发现象。

注意：`index_address_update_ms_per_token` 是一个外层计时，内部包含 `address_prepare`、anchor 的 native `address_update` 或 reuse 的 `reuse_update_node`，不能与这些子项直接相加。

## 5. 地址公式的推导

CPU KV pool 由一个连续的 pinned tensor 一次性分配：

```python
self.buffer = torch.zeros(
    (n_max_pages, *page_shape),
    device="cpu",
    pin_memory=True,
)
```

因此，物理 page `p` 的地址满足：

```text
pool_base + p * page_stride
```

其中现有代码已经定义：

```python
page_stride = self.cpu_n_bytes_per_page
head_stride = self.page_size * self.head_dim * self.cpu_dtype.itemsize
```

prefill 路径已经使用 `base + j * stride + head_offset`，并断言首尾 page 地址符合固定 stride。这证明公式对连续的物理 pool buffer 成立。

但 decode 增量路径中的 `j` 是 cache logical page id，不能直接假设 `physical_page_id == j`。`KvCache.__getitem__` 的真实语义为：

```python
return self.pool[self.c2p[idx]]
```

所以安全公式必须保留 logical-to-physical 映射：

```text
address(logical_j, head_h)
  = pool.buffer.data_ptr()
  + c2p[b, logical_j] * page_stride
  + head_h * head_stride
```

当前慢路径为：

```python
tmp_addr = [
    cast(cpu_cache[b, j].data_ptr() + inst * offset, c_void_p).value
    for j in tmp_new_indices
]
```

其中每个 leaf 都经过 Python loop、`c2p` Tensor indexing、pool Tensor indexing、临时 Tensor view、`data_ptr()`、Python integer 和 `ctypes.cast`。候选优化是一次读取 physical page ids，然后通过 NumPy/整数向量运算生成地址。

## 6. 接下来的工作：必须按顺序完成

### Step 1：继续缩小 profiling 区域

在改变实现前，将 `index_address_prepare` 再拆成：

1. `num_leaves/new_num_leaves/new_indices` 准备；
2. 当前 `cpu_cache[b, j].data_ptr()` 地址循环；
3. `page_address_buffer` 写入。

目的：确认 2.32 ms/token 中有多少真正来自逐 leaf 地址解析，避免错误归因。插桩仍须只在 `profile_dci=True` 时生效。

同时记录每次更新的：

```text
layer id
anchor/reuse
prev_num_points
inserted token count
new leaf count per head / total
每个阶段耗时
```

至少按 tree size 分桶输出统计，确认 native insert 是否随 `prev_num_points` 增长。

### Step 2：做地址等价性微测试

实现一个不改变生产路径的 helper/probe，同时计算：

```python
old = cpu_cache[b, logical_j].data_ptr() + head * head_stride
new = (
    cpu_cache.pool.buffer.data_ptr()
    + int(cpu_cache.c2p[b, logical_j]) * page_stride
    + head * head_stride
)
assert old == new
```

必须覆盖：

- 多个 head；
- 多个新 leaf；
- 人工构造或实际运行中非连续、乱序的 `c2p`；
- cache 扩容前后；
- anchor layer 与 reuse layer。

不允许用 `base + logical_j * stride` 代替 `c2p`，因为 CPU pool 经过分配/回收后可能碎片化。

### Step 3：实现向量化地址生成

建议形态：

```python
pool_base = np.uintp(cpu_cache.pool.buffer.data_ptr())
physical_ids = cpu_cache.c2p[b, tmp_new_indices].numpy().astype(np.uintp)
tmp_addr_np = (
    pool_base
    + physical_ids * np.uintp(self.cpu_n_bytes_per_page)
    + np.uintp(inst * head_stride)
)
```

需要根据 M-DCI Python binding 对 `new_address` 的入参要求，决定保留 NumPy array 还是只在边界处 `.tolist()`。不要在元素级调用 `ctypes.cast`；地址本身是整数，`np.uintp` 与平台指针宽度一致。

可以进一步缓存 `cpu_cache.pool.buffer.data_ptr()`，但首先保证生命周期安全：只要 pool 的底层 tensor 不被重新分配，base pointer 才稳定。cache 的 page table 扩容不等于 pool buffer 重分配。

### Step 4：A/B 验证

必须同时满足：

1. 新旧地址逐元素完全相等；
2. DCI 选择 page ids 相等；
3. 固定 seed 下生成输出一致；
4. Qasper profile 中 `index_address_prepare` 明显下降；
5. `index_native_insert` 不应因该改动发生系统性变化；
6. 无新增 CUDA synchronization。

推荐先跑 1 个样本验证正确性，再跑同一组 3 个 Qasper 样本比较：

```text
context subset indices: [4, 53, 57]
约对应 context: 4.5k, 2k, 21k
```

最终报告必须同时给出：

- ms/token；
- ms/page-boundary；
- ms/layer-update；
- boundary 数、index update calls 数；
- 至少两次重复运行的波动。

## 7. 后续算法研究边界

即使地址优化成功，native M-DCI incremental insert 仍值得研究。当前三样本平均为 0.912 ms/token，且加入 21k context 后明显高于单独 2k 样本的 0.492 ms/token，可能随 tree size 增长。

地址路径优化完成后，应建立：

```text
T_insert(prev_num_points, inserted_tokens, new_leaves, promotion_level)
```

重点判断：

- parent 1-NN search 是否随已有点数增长；
- promotion 是否改变新增 leaf 数及 parent-search 成本；
- leaf split / node relocation 是否造成偶发长尾；
- OpenMP 在一次只插入 16 token/page 的小批量任务上是否线程启动与 barrier 成本过高。

不要在这一阶段改成 page centroid 直接检索。那会改变 token-level 语义召回和稀疏注意力选择，属于不同算法，需要单独的质量实验，不能与地址优化混在一次 A/B 中。

## 8. 交付要求

请交付：

1. 更细粒度 profiling 数据与归因；
2. 地址公式等价性测试；
3. 最小、可开关的向量化补丁；
4. 新旧路径的性能与正确性 A/B；
5. 对 native incremental insert 的下一步算法建议，但不要在无数据时直接改 M-DCI。

