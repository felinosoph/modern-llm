import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)


        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if position_ids is not None:
        if q.dim() == 3:
            position_ids = position_ids.view(-1)

        cos = cos[position_ids]  # [N, Dim]
        sin = sin[position_ids]

        if q.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        elif q.dim() == 4:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)

    else:
        cos = cos[:q.shape[1]].unsqueeze(0).unsqueeze(2)
        sin = sin[:q.shape[1]].unsqueeze(0).unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# --- 1. ATOMS ---

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FlexibleAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, page_block_size=None, max_blocks=None):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Initialize RoPE
        self.rotary = RotaryEmbedding(self.d_head)

        if page_block_size is None:
            self.mode = "training"
            self.kv_cache = None
        else:
            self.mode = "inference"
            if max_blocks is None: raise ValueError("Must specify max_blocks")
            cache_shape = (max_blocks, page_block_size, 2, n_head, self.d_head)
            self.register_buffer("kv_cache", torch.zeros(cache_shape, dtype=torch.float16))

    def forward(self, x, cu_seqlens=None, max_seqlen=None, block_table=None, cache_seqlens=None):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        cos, sin = self.rotary(x, seq_len=4096)

        if self.mode == "training":
            q = q.view(-1, self.n_head, self.d_head)
            k = k.view(-1, self.n_head, self.d_head)
            v = v.view(-1, self.n_head, self.d_head)

            if cu_seqlens is not None:
                pos_ids = []
                for i in range(len(cu_seqlens) - 1):
                    start, end = cu_seqlens[i], cu_seqlens[i + 1]
                    pos_ids.append(torch.arange(0, end - start, device=x.device))
                position_ids = torch.cat(pos_ids)

                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

            # Flash Attention Kernel
            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True
            )
            output = output.view(-1, self.n_head * self.d_head)

        elif self.mode == "inference":
            B, T, _ = x.shape
            q = q.view(B, T, self.n_head, self.d_head)
            k = k.view(B, T, self.n_head, self.d_head)
            v = v.view(B, T, self.n_head, self.d_head)

            if cache_seqlens is not None:
                position_ids = (cache_seqlens - 1).clamp(min=0).long().unsqueeze(1)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

            output = flash_attn_with_kvcache(
                q,
                k_cache=self.kv_cache[:, :, 0],
                v_cache=self.kv_cache[:, :, 1],
                k=k, v=v,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=True
            )
            output = output.view(B, T, self.n_head * self.d_head)

        return self.W_o(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, page_block_size=None, max_blocks=None):
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model)
        self.attention = FlexibleAttentionLayer(d_model, n_head, page_block_size, max_blocks)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, hidden_dim)

    def forward(self, x, **kwargs):
        h = x + self.attention(self.attention_norm(x), **kwargs)
        out = h + self.mlp(self.mlp_norm(h))
        return out


class DecoderOnlyModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            d_model: int,
            n_head: int,
            hidden_dim: int,
            page_block_size: Optional[int] = None,
            max_blocks: Optional[int] = None
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, hidden_dim, page_block_size, max_blocks)
            for _ in range(n_layers)
        ])

        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, **kwargs):
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h, **kwargs)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits