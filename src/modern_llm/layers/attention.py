from typing import Union, Tuple

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from modern_llm.layers.rotary import RotaryEmbedding, apply_rotary


class FlexibleAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, page_block_size=None, max_blocks=None):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rotary = RotaryEmbedding(self.d_head)

    def forward_dense(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_seqlen: int,
            output_kv: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        # 1. Projections (Shared)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2. Rotary Embeddings (Shared)
        # The logic to reconstruct position_ids from cu_seqlens is now only written once!
        cos, sin = self.rotary(x, seq_len=max_seqlen)

        q = q.view(-1, self.n_head, self.d_head)
        k = k.view(-1, self.n_head, self.d_head)
        v = v.view(-1, self.n_head, self.d_head)

        # Reconstruct pos_ids (Previously duplicated logic)
        device = x.device
        pos_ids = []
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            pos_ids.append(torch.arange(0, end - start, device=device))
        position_ids = torch.cat(pos_ids, dim=0)

        q, k = apply_rotary(q, k, cos, sin, position_ids)

        # 3. Attention (Shared)
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )

        output = self.W_o(out.view(-1, self.n_head * self.d_head))

        if output_kv:
            return output, k, v
        return output
    def forward_paged(
            self,
            x: torch.Tensor,
            kv_cache: torch.Tensor,
            cache_seqlens: torch.Tensor,
            block_table: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        cos, sin = self.rotary(x, seq_len=cache_seqlens.max().item() + T)

        q = q.view(B, T, self.n_head, self.d_head)
        k = k.view(B, T, self.n_head, self.d_head)
        v = v.view(B, T, self.n_head, self.d_head)

        if cache_seqlens is not None:
            position_ids = cache_seqlens.long().unsqueeze(1)
            q, k = apply_rotary(q, k, cos, sin, position_ids)

        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]

        out = flash_attn_with_kvcache(
            q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
            cache_seqlens=cache_seqlens, block_table=block_table,
            causal=True,
        )
        return self.W_o(out.view(B, T, self.n_head * self.d_head))

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_seqlen: int,
            output_kv: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.forward_dense(
            x, cu_seqlens, max_seqlen, output_kv=output_kv)
