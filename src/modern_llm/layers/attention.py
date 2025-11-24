from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from modern_llm import KVCacheBlockManager
from modern_llm.layers.rotary import RotaryEmbedding, apply_rotary


@dataclass
class AttentionContext:
    pass

@dataclass
class TrainingAttentionContext(AttentionContext):
    cu_seqlens: torch.Tensor
    max_seqlen: int


@dataclass
class PrefillAttentionContext(AttentionContext):
    seq_ids: list[Any]
    cu_seqlens: torch.Tensor
    max_seqlen: int
    kv_cache: torch.Tensor
    cache_manager: KVCacheBlockManager


@dataclass
class DecodeAttentionContext(AttentionContext):
    kv_cache: torch.Tensor
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor



def update_cache_vectorized(
        kv_cache: torch.Tensor,  # [num_blocks, 2, n_heads, block_size, d_head]
        k_flat: torch.Tensor,  # [total_tokens, n_heads, d_head]
        v_flat: torch.Tensor,  # [total_tokens, n_heads, d_head]
        seq_ids: List[Any],
        cu_seqlens: torch.Tensor,
        manager: KVCacheBlockManager,
) -> None:
    """
    Writes k_flat and v_flat into the paged cache in a single vectorized operation.
    """
    device = kv_cache.device
    block_size = manager._tokens_per_block

    block_table = manager.get_block_table_for(seq_ids, device=device)

    seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()

    total_lens = [manager._seq_len[sid] for sid in seq_ids]

    start_positions = [total - length for total, length in zip(total_lens, seq_lens)]

    seq_indices_list = []
    global_pos_list = []

    for i, (length, start) in enumerate(zip(seq_lens, start_positions)):
        seq_indices_list.extend([i] * length)
        global_pos_list.extend(range(start, start + length))

    seq_indices = torch.tensor(seq_indices_list, dtype=torch.long, device=device)  # [Total_Tokens]
    global_pos = torch.tensor(global_pos_list, dtype=torch.long, device=device)  # [Total_Tokens]


    logical_block_indices = global_pos // block_size
    block_offsets = global_pos % block_size

    physical_block_ids = block_table[seq_indices, logical_block_indices]

    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]

    k_cache[physical_block_ids, :, block_offsets, :] = k_flat
    v_cache[physical_block_ids, :, block_offsets, :] = v_flat


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

    def _training(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        device = x.device
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        cos, sin = self.rotary(x, seq_len=max_seqlen)

        q = q.view(-1, self.n_head, self.d_head)
        k = k.view(-1, self.n_head, self.d_head)
        v = v.view(-1, self.n_head, self.d_head)

        pos_ids = []
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            pos_ids.append(torch.arange(0, end - start, device=device))
        position_ids = torch.cat(pos_ids, dim=0)

        q, k = apply_rotary(q, k, cos, sin, position_ids)

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )
        return self.W_o(out.view(-1, self.n_head * self.d_head))

    def _prefill(
            self,
            x: torch.Tensor,
            seq_ids: list[Any],
            cu_seqlens: torch.Tensor,
            max_seqlen: int,
            kv_cache: torch.Tensor,
            cache_manager: KVCacheBlockManager,
    ) -> torch.Tensor:
        device = x.device

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        cos, sin = self.rotary(x, seq_len=max_seqlen)

        q = q.view(-1, self.n_head, self.d_head)
        k = k.view(-1, self.n_head, self.d_head)
        v = v.view(-1, self.n_head, self.d_head)

        pos_ids = []
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            pos_ids.append(torch.arange(0, end - start, device=device))
        position_ids = torch.cat(pos_ids, dim=0)

        q, k = apply_rotary(q, k, cos, sin, position_ids)


        B = len(seq_ids)
        for i in range(B):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            T_i = (end - start).item()
            cache_manager.allocate_blocks_for(seq_ids[i], T_i)


        update_cache_vectorized(
            kv_cache=kv_cache,
            k_flat=k,
            v_flat=v,
            seq_ids=seq_ids,
            cu_seqlens=cu_seqlens,
            manager=cache_manager
        )

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )

        return self.W_o(out.view(-1, self.n_head * self.d_head))

    def _decode(
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

    def forward(self, x, attention_context):
        match attention_context:
            case TrainingAttentionContext(cu_seqlens, max_seqlen):
                return self._training(x, cu_seqlens, max_seqlen)
            case PrefillAttentionContext(seq_ids, cu_seqlens, max_seqlen, kv_cache, cache_manager):
                return self._prefill(x, seq_ids, cu_seqlens, max_seqlen, kv_cache, cache_manager)
            case DecodeAttentionContext(kv_cache, cache_seqlens, block_table):
                return self._decode(x, kv_cache, cache_seqlens, block_table)
            case _:
                raise ValueError(f"Unsupported attention context: {attention_context!r}")

