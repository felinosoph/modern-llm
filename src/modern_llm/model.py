from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

from modern_llm import KVCacheBlockManager
from modern_llm.layers.attention import FlexibleAttentionLayer, AttentionContext, DecodeAttentionContext
from modern_llm.layers.feedforward import SwiGLUMLP

@dataclass
class ModelSpec:
    vocab_size: int
    n_layers: int
    d_model: int
    n_head: int
    hidden_dim: int


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, page_block_size=None, max_blocks=None):
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model)
        self.attention = FlexibleAttentionLayer(d_model, n_head, page_block_size, max_blocks)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, hidden_dim)

    def forward(self, x, attention_context: AttentionContext):
        h = x + self.attention(self.attention_norm(x), attention_context)
        out = h + self.mlp(self.mlp_norm(h))
        return out


class DecoderOnlyModel(nn.Module):
    def __init__(self, model_spec: ModelSpec):
        super().__init__()
        self.model_spec = model_spec
        self.embed_tokens = nn.Embedding(model_spec.vocab_size, model_spec.d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(model_spec.d_model, model_spec.n_head, model_spec.hidden_dim)
            for _ in range(model_spec.n_layers)
        ])

        self.norm = nn.RMSNorm(model_spec.d_model)
        self.lm_head = nn.Linear(model_spec.d_model, model_spec.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, attention_contexts: Union[AttentionContext, List[AttentionContext]]):
        """
        Flexible forward pass.

        Args:
            x: Input tokens
            attention_contexts:
                - If a single AttentionContext (e.g., Training), it is applied to all layers.
                - If a List (e.g., Inference with KV Cache), it must match len(layers).
        """
        h = self.embed_tokens(x)

        # Handle broadcasting for single context (common in training)
        if isinstance(attention_contexts, AttentionContext):
            contexts = [attention_contexts] * len(self.layers)
        else:
            if len(attention_contexts) != len(self.layers):
                raise ValueError(
                    f"Context list length ({len(attention_contexts)}) must match layers ({len(self.layers)})")
            contexts = attention_contexts

        for i, layer in enumerate(self.layers):
            h = layer(h, attention_context=contexts[i])

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    @torch.no_grad()
    def generate(
            self,
            prompt_ids: List[int],
            max_new_tokens: int = 50,
            block_size: int = 16,
            max_blocks: int = 128
    ) -> List[int]:
        """
        Self-contained generation method.
        Encapsulates KV cache allocation and context management internally.
        """
        device = next(self.parameters()).device
        self.eval()

        # --- Internal Setup: The Model manages its own Cache ---
        manager = KVCacheBlockManager(max_blocks=max_blocks, block_size=block_size)

        # Dimensions from spec
        n_head = self.model_spec.n_head
        d_head = self.model_spec.d_model // n_head

        # Allocate VRAM for each layer
        kv_caches = [
            torch.zeros((max_blocks, 2, n_head, block_size, d_head), dtype=torch.float16, device=device)
            for _ in range(len(self.layers))
        ]

        # --- Phase 1: Prefill ---
        seq_id = 0
        prompt_len = len(prompt_ids)
        input_tensor = torch.tensor([prompt_ids], device=device)
        cu_seqlens = torch.tensor([0, prompt_len], device=device, dtype=torch.int32)

        # Construct contexts internally - Caller doesn't need to know!
        prefill_contexts = [
            PrefillAttentionContext(
                seq_ids=[seq_id], cu_seqlens=cu_seqlens, max_seqlen=prompt_len,
                kv_cache=cache, cache_manager=manager
            ) for cache in kv_caches
        ]

        logits = self.forward(input_tensor, attention_contexts=prefill_contexts)
        next_token = torch.argmax(logits[0, -1, :]).item()
        generated_ids = list(prompt_ids) + [next_token]

        # --- Phase 2: Decode ---
        current_len = prompt_len + 1

        for _ in range(max_new_tokens - 1):
            input_tensor = torch.tensor([[next_token]], device=device)

            # Global allocation update (applies to all layers logically)
            manager.allocate_blocks_for(seq_id, 1)
            block_table = manager.get_block_table_for([seq_id], device=device)
            cache_seqlens = torch.tensor([current_len], device=device, dtype=torch.int32)

            # Construct decode contexts internally
            decode_contexts = [
                DecodeAttentionContext(
                    kv_cache=cache, cache_seqlens=cache_seqlens, block_table=block_table
                ) for cache in kv_caches
            ]

            logits = self.forward(input_tensor, attention_contexts=decode_contexts)
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated_ids.append(next_token)
            current_len += 1

        return generated_ids