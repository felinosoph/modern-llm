import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class BlockSpaceManager:
    def __init__(self, max_blocks: int, block_size: int, device: str = "cuda") -> None:
        super().__init__()
        self.block_size: int = block_size
        self.max_blocks: int = max_blocks
        self.device: str = device
        self.free_blocks: List[int] = list(range(max_blocks))
        self.sequences: Dict[str, List[int]] = {}
        self.seq_lens: Dict[str, int] = {}

    def allocate(self, seq_id: str, prompt_len: int) -> None:
        num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(f"OOM: Needed {num_blocks_needed} blocks, {len(self.free_blocks)} free.")
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            allocated_blocks.append(self.free_blocks.pop())
        self.sequences[seq_id] = allocated_blocks
        self.seq_lens[seq_id] = prompt_len

    def step(self, seq_ids: List[str]) -> None:
        for seq_id in seq_ids:
            current_len = self.seq_lens[seq_id]
            if current_len > 0 and (current_len % self.block_size == 0):
                if not self.free_blocks:
                    raise RuntimeError("OOM: No free blocks available during generation")
                new_block = self.free_blocks.pop()
                self.sequences[seq_id].append(new_block)
            self.seq_lens[seq_id] += 1

    def get_metadata_tensors(self, seq_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(seq_ids)
        max_blocks_per_seq = max(len(self.sequences[sid]) for sid in seq_ids)
        
        block_tables = torch.full(
            (batch_size, max_blocks_per_seq), 
            fill_value=-1, 
            dtype=torch.int32, 
            device=self.device
        )
        cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        
        for i, seq_id in enumerate(seq_ids):
            cache_seqlens[i] = self.seq_lens[seq_id]
            blocks = self.sequences[seq_id]
            block_tables[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)
            
        return block_tables, cache_seqlens

    def free(self, seq_id: str) -> None:
        if seq_id in self.sequences:
            blocks_to_free = self.sequences[seq_id]
            self.free_blocks.extend(blocks_to_free)
            del self.sequences[seq_id]
            del self.seq_lens[seq_id]
