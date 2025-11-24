import torch
import torch.nn.functional as F
from typing import List, Tuple

from hopper.generate_kernels import batch_hdim


def packed_collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    seqlens = [len(x) for x in batch]
    max_seqlen = max(seqlens)
    packed_inputs = torch.cat(batch, dim=0)
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device=packed_inputs.device).cumsum(dim=0)

    labels = packed_inputs.clone().roll(-1)
    labels[cu_seqlens - 1] = -100

    return packed_inputs, cu_seqlens, max_seqlen, labels


