import torch
import torch.nn as nn
from torch import Tensor
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=typechecker)
def rotate_half(
        x: Float[Tensor, "... dim"],
) -> Float[Tensor, "num_tokens num_heads dim"]:
    dim = x.shape[-1]
    real = x[..., :dim//2]
    imag = x[..., dim//2:]
    return torch.cat([-imag, real], dim=-1)

@jaxtyped(typechecker=typechecker)
def apply_rotary(
        x: Float[Tensor, "num_tokens num_heads dim"],
        cos: Float[Tensor, "num_tokens 1 dim"],
        sin: Float[Tensor, "num_tokens 1 dim"]) -> Float[Tensor, "num_tokens num_heads dim"]:
    """Apply RoPE rotation to a packed [N, H, D] tensor.
    The last dimension encodes (real, imag) pairs for each frequency."""

    num_tokens, num_heads, dim = x.shape

    assert dim % 2 == 0, "Hidden dimension must be even for rope"

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim:int, max_seq_len: int, theta:float =10_000, device:torch.device = torch.device("cpu")):
        super().__init__()
        assert dim % 2 == 0, "Hidden dimension must be even for rope"
        dtype = torch.float32
        self.max_seq_len = max_seq_len
        i = torch.arange(dim//2, dtype=dtype, device=device)
        freqs = 1.0 / (theta ** (2 * i / dim))
        t = torch.arange(max_seq_len, dtype=dtype, device=device)
        phases = torch.outer(t, freqs)
        emb = torch.cat([phases, phases], dim=-1)
        cos_cached = torch.cos(emb).to(dtype)
        sin_cached = torch.sin(emb).to(dtype)
        self.register_buffer( "cos_cached", cos_cached)
        self.register_buffer( "sin_cached", sin_cached)

    def forward(self,
                position_ids: Int[Tensor, "num_tokens"],
                dtype: torch.dtype = torch.float32):
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        if cos.dtype != dtype:
            cos = cos.to(dtype)
            sin = sin.to(dtype)

        return cos, sin

