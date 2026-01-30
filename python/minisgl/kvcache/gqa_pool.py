from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import KVCacheLayout
from .mha_pool import MHAKVCache


class GQAKVCache(MHAKVCache):
    """
    Key-Value Cache specifically named for Grouped Query Attention (GQA).
    
    Functionally, this is identical to MHAKVCache because GQA's storage requirement
    is just a standard KV cache with fewer heads than the query heads.
    This class exists for semantic clarity and explicit GQA support.
    
    In GQA:
    - num_kv_heads < num_qo_heads
    - The ratio num_qo_heads // num_kv_heads is the group size.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            dtype=dtype,
            kv_layout=kv_layout,
            device=device,
        )
