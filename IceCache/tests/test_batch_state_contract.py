"""CPU checks for the fixed B=2 batch boundary and request isolation."""

from types import SimpleNamespace
from threading import Lock

import pytest
import torch

from icecache.adapter.modeling import icecache_state, set_icecache_infer_state
from icecache.batch import BatchInferState
from icecache.infer_state import ForwardMode


def _pool():
    return SimpleNamespace(n_max_pages=16, _free_ids=set())


def _request(pool, pages, *, seq_len=32, valid=None):
    cache = SimpleNamespace(
        c2p=torch.tensor([pages], dtype=torch.int32),
        n_real_pages=len(pages),
        last_page_len=7,
        pool=pool,
        seq_len=seq_len,
    )
    return SimpleNamespace(
        kv_caches=[cache], n_layers=1, seq_len=seq_len,
        use_dci=True, layer2budget=[None], dci_db=[None],
        _pool=pool, batch_size=1, n_qo_heads=4, n_kv_heads=2,
        head_dim=8, page_size=16, dtype=torch.float32,
        device=torch.device("cpu"), layout="HND",
        n_prefetch_layers=0, n_reuse_layers=0,
        ratio=1, search_ratio=1.0,
        n_sink_pages=1, n_win_pages=2, n_groups=1, offload_ratio=2,
        page_valid_entries=[valid],
    )


def _metadata_batch(a, b):
    # Metadata construction has no need to allocate an attention workspace.
    batch = object.__new__(BatchInferState)
    batch.states = (a, b)
    batch.n_layers = 1
    batch.n_kv_heads = 2
    batch.page_size = 16
    batch.device = torch.device("cpu")
    batch._pool = a._pool
    batch._closed = False
    batch._failed = False
    batch._step_active = False
    batch._forward_lock = Lock()
    batch._step_thread = None
    batch._query_active = False
    batch._decode_handlers_armed = False
    batch._next_layer = 0
    batch.forward_mode = ForwardMode.DECODE
    return batch


def test_batch_rejects_unprefilled_request_and_overlapping_pages(monkeypatch):
    pool = _pool()
    a = _request(pool, [1, 3])
    b = _request(pool, [4, 6])
    b.kv_caches = [None]
    with pytest.raises(ValueError, match="not completed prefill"):
        BatchInferState.from_prefilled((a, b))

    b.kv_caches = [_request(pool, [3, 6]).kv_caches[0]]
    monkeypatch.setattr("icecache.batch.kernels.BatchDecodeWithPagedKVCacheWrapper",
                        lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(RuntimeError, match="belongs to both"):
        BatchInferState.from_prefilled((a, b))


def test_attention_metadata_follows_request_page_order_and_head_counts():
    pool = _pool()
    a = _request(pool, [5, 2], valid=torch.tensor([[16, 12], [4, 5]], dtype=torch.int32))
    b = _request(pool, [7, 1, 9], valid=torch.tensor(
        [[2, 16], [10, 11], [3, 6]], dtype=torch.int32))
    a.use_dci = b.use_dci = True
    a.layer2budget = b.layer2budget = [3]
    b.kv_caches[0].last_page_len = 9
    batch = _metadata_batch(a, b)

    indices, indptr, last, valid = batch.build_attention_metadata(0)
    assert indices.tolist() == [5, 2, 7, 1, 9]
    assert indptr.tolist() == [0, 2, 5]
    assert last.tolist() == [7, 9]
    assert valid.reshape(5, 2).tolist() == [
        [16, 12], [7, 7], [2, 16], [10, 11], [9, 9],
    ]
    # A page belonging to B must never silently become an A page.
    b.kv_caches[0].c2p[0, 0] = 5
    with pytest.raises(RuntimeError, match="belongs to both"):
        batch.validate_ready()


def test_state_scope_restores_original_after_error_and_rejects_mid_forward_switch():
    original = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    selected = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    model = SimpleNamespace(_icecache_infer_state=original)

    with pytest.raises(ArithmeticError, match="forward failed"):
        with icecache_state(model, selected):
            assert model._icecache_infer_state is selected
            with pytest.raises(RuntimeError, match="cannot switch"):
                set_icecache_infer_state(model, original)
            with pytest.raises(RuntimeError, match="nested"):
                with icecache_state(model, original):
                    pass
            raise ArithmeticError("forward failed")

    assert model._icecache_infer_state is original
    assert model._icecache_state_scope_active is False
    set_icecache_infer_state(model, selected)
    assert model._icecache_infer_state is selected


def test_step_calls_one_batched_forward_with_absolute_positions_and_restores_state():
    pool = _pool()
    a, b = _request(pool, [1], seq_len=32), _request(pool, [2], seq_len=32)
    batch = _metadata_batch(a, b)
    original = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    class Model:
        _icecache_infer_state = original

        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, *, position_ids, cache_position,
                     use_cache, return_dict):
            assert self._icecache_infer_state is batch
            self.calls.append((input_ids.clone(), position_ids.clone(),
                               cache_position.clone(), use_cache, return_dict))
            for state in batch.states:
                state.seq_len += 1
                state.kv_caches[0].seq_len += 1
            batch._next_layer = batch.n_layers
            return SimpleNamespace(logits=torch.zeros(2, 1, 4))

    model = Model()
    tokens = torch.tensor([[4], [8]])
    result = batch.step(model, tokens, return_dict=True)
    assert result.logits.shape == (2, 1, 4)
    assert len(model.calls) == 1
    observed_ids, positions, cache_position, use_cache, return_dict = model.calls[0]
    assert torch.equal(observed_ids, tokens)
    assert positions.tolist() == [[32], [32]]
    assert cache_position.tolist() == [32]
    assert use_cache is False and return_dict is True
    assert batch.seq_lens == (33, 33)
    assert model._icecache_infer_state is original

    with pytest.raises(ValueError, match="position_ids"):
        batch.step(model, tokens, position_ids=torch.tensor([[33], [32]]))
    assert len(model.calls) == 1
    batch.close()
    with pytest.raises(RuntimeError, match="closed"):
        batch.step(model, tokens)


def test_failed_step_rejects_retry_after_partial_kv_update():
    pool = _pool()
    a, b = _request(pool, [1]), _request(pool, [2])
    batch = _metadata_batch(a, b)
    original = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    class FailingModel:
        _icecache_infer_state = original

        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, **kwargs):
            assert self._icecache_infer_state is batch
            self.calls += 1
            a.seq_len += 1
            a.kv_caches[0].seq_len += 1
            raise RuntimeError("attention failed after first request wrote KV")

    model = FailingModel()
    tokens = torch.tensor([[4], [8]])
    with pytest.raises(RuntimeError, match="attention failed"):
        batch.step(model, tokens)
    assert model._icecache_infer_state is original
    assert model.calls == 1
    with pytest.raises(RuntimeError, match="batch forward failed"):
        batch.step(model, tokens)
    assert model.calls == 1
