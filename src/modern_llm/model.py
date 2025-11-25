from dataclasses import dataclass

import torch
import torch.nn as nn

from modern_llm.kv_cache import write_to_kv_cache
from modern_llm.layers.attention import FlexibleAttentionLayer
from modern_llm.layers.feedforward import SwiGLUMLP


@dataclass
class ModelSpec:
    vocab_size: int
    n_layers: int
    d_model: int
    n_head: int
    hidden_dim: int
    max_seq_len: int

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, max_seq_len, page_block_size=None, max_blocks=None):
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model)
        self.attention = FlexibleAttentionLayer(d_model, n_head, max_seq_len, page_block_size, max_blocks)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, hidden_dim)

    def forward(self, x: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        block_table: torch.Tensor = None,
        kv_cache: torch.Tensor = None,
        cache_seqlens: torch.Tensor = None,
        output_kv: bool = False):
        if cu_seqlens is not None:
            attn_out = self.attention.forward_dense(
                self.attention_norm(x),
                cu_seqlens,
                max_seqlen,
                output_kv=output_kv
            )
        else:
            attn_out = self.attention.forward_paged(
                self.attention_norm(x),
                kv_cache,
                cache_seqlens,
                block_table
            )
        k_out, v_out = None, None
        if output_kv:
            attn_out, k_out, v_out = attn_out
            write_to_kv_cache(cu_seqlens, block_table, kv_cache, k_out, v_out)
        h = x + attn_out
        out = h + self.mlp(self.mlp_norm(h))

        if output_kv:
            return out, k_out, v_out
        return out


class DecoderOnlyModel(nn.Module):
    def __init__(self, model_spec: ModelSpec):
        super().__init__()
        self.model_spec = model_spec
        self.embed_tokens = nn.Embedding(model_spec.vocab_size, model_spec.d_model)

        self.layers = nn.ModuleList([
            # Pass max_seq_len from spec to block
            TransformerBlock(
                d_model=model_spec.d_model,
                n_head=model_spec.n_head,
                hidden_dim=model_spec.hidden_dim,
                max_seq_len=model_spec.max_seq_len
            )
            for _ in range(model_spec.n_layers)
        ])
        self.norm = nn.RMSNorm(model_spec.d_model)
        self.lm_head = nn.Linear(model_spec.d_model, model_spec.vocab_size, bias=False)

    def forward(self, x: torch.Tensor,  **kwargs):
        """
        Flexible forward pass.

        Args:
            x: Input tokens
            attention_contexts:
                - If a single AttentionContext (e.g., Training), it is applied to all layers.
                - If a List (e.g., Inference with KV Cache), it must match len(layers).
        """
        h = self.embed_tokens(x)

        for layer in self.layers:
            h = layer(h, **kwargs)

            if isinstance(h, tuple):
                h, k, v = h



        h = self.norm(h)
        return self.lm_head(h)

