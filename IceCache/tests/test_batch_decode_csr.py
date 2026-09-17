"""The B=2 CSR layout must reproduce two isolated B=1 sparse attentions."""

import pytest
import torch

from icecache.kernels import BatchDecodeWithPagedKVCacheWrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [1, 4])
def test_two_request_sparse_decode_matches_isolated_requests(ratio):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(73)
    page_size, n_heads, head_dim = 16, 4, 128
    # HND: [physical_page, key_or_value, head, slot, dimension]
    pages = torch.randn(7, 2, n_heads, page_size, head_dim,
                        device=device, dtype=torch.float16, generator=generator)
    query = torch.randn(2, 1, n_heads * ratio, head_dim, device=device,
                        dtype=torch.float16, generator=generator)
    page_lists = ([5, 2, 0], [6, 1, 4, 3])
    lengths = (7, 11)
    valid_rows = (
        torch.tensor([[16, 10, 14, 8], [5, 16, 9, 13], [7] * n_heads],
                     device=device, dtype=torch.int32),
        torch.tensor([[12, 16, 8, 11], [6, 10, 16, 14],
                      [16, 3, 12, 15], [11] * n_heads],
                     device=device, dtype=torch.int32),
    )

    def run(rows):
        indices = torch.tensor([p for row in rows for p in page_lists[row]],
                               device=device, dtype=torch.int32)
        sizes = [len(page_lists[row]) for row in rows]
        indptr = torch.tensor([0] + [sum(sizes[:i]) for i in range(1, len(rows) + 1)],
                              device=device, dtype=torch.int32)
        last = torch.tensor([lengths[row] for row in rows],
                            device=device, dtype=torch.int32)
        valid = torch.cat([valid_rows[row] for row in rows], dim=0).reshape(-1)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(16 << 20, dtype=torch.uint8, device=device), "HND"
        )
        wrapper.begin_forward(indptr, last, n_heads * ratio, n_heads, head_dim,
                              page_size, data_type=torch.float16)
        try:
            result = wrapper.forward(query[list(rows)], pages, indices,
                                     page_valid_entries=valid, dci=True)
            torch.cuda.synchronize()
            return result
        finally:
            wrapper.end_forward()

    both = run((0, 1))
    alone_a = run((0,))
    alone_b = run((1,))
    torch.testing.assert_close(both[0], alone_a[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(both[1], alone_b[0], atol=1e-2, rtol=1e-2)
