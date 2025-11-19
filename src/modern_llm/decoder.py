import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class FlexibleAttentionLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        page_block_size: Optional[int] = None, 
        max_blocks: Optional[int] = None
    ) -> None:
        super().__init__()
        self.d_head: int = d_model // n_head
        self.n_head: int = n_head

        self.W_q: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_k: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_v: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_o: nn.Linear = nn.Linear(d_model, d_model, bias=False)

        if page_block_size is None:
            self.mode: str = "training"
            self.kv_cache: Optional[torch.Tensor] = None
        else:
            self.mode: str = "inference"
            if max_blocks is None:
                raise ValueError("Must specify max_blocks")

            cache_shape = (max_blocks, page_block_size, 2, n_head, self.d_head)
            self.register_buffer("kv_cache", torch.zeros(cache_shape, dtype=torch.float16))

    def forward(
        self, 
        x: torch.Tensor, 
        cu_seqlens: Optional[torch.Tensor] = None, 
        max_seqlen: Optional[int] = None, 
        block_table: Optional[torch.Tensor] = None, 
        cache_seqlens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        if self.mode == "training":
            q = q.view(-1, self.n_head, self.d_head)
            k = k.view(-1, self.n_head, self.d_head)
            v = v.view(-1, self.n_head, self.d_head)

            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, 
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, 
                max_seqlen_k=max_seqlen,
                causal=True
            )
            output = output.view(-1, self.n_head * self.d_head)

        elif self.mode == "inference":
            B, T, _ = x.shape

            q = q.view(B, T, self.n_head, self.d_head)
            k = k.view(B, T, self.n_head, self.d_head)
            v = v.view(B, T, self.n_head, self.d_head)

            output = flash_attn_with_kvcache(
                q,
                k_cache=self.kv_cache[:, :, 0],
                v_cache=self.kv_cache[:, :, 1],
                k=k, 
                v=v,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=True
            )
            output = output.view(B, T, self.n_head * self.d_head)

        return self.W_o(output)
