from typing import List, Any, Sequence, Tuple

import torch

class KVCacheBlockManager:
    _tokens_per_block: int
    _max_blocks: int
    _free_blocks: List[int]
    _seq_len: dict[Any, int]
    _blocks_per_sequence: dict[Any, List[int]]

    def __init__(self, max_blocks: int, block_size: int) -> None:
        self._tokens_per_block = block_size
        self._max_blocks = max_blocks
        self.reset()

    def allocate_blocks_for(self, sequence_id:Any, num_tokens: int) -> List[int]:
        if sequence_id not in self._seq_len:
            self._seq_len[sequence_id] = 0
            self._blocks_per_sequence[sequence_id] = []

        seq_len = self._seq_len[sequence_id]
        used_in_last_block = seq_len % self._tokens_per_block


        # the % handles the special case of the first allocation
        # if we self.length = 0 , then used_in_last_block = 0
        # block-used_in_last_block = block_size
        # however, this would not create a new block in the first iteration but we want it
        # to be generated. So we % block_size and get a block_size % block_size = 0
        capacity_in_last_block = (self._tokens_per_block - used_in_last_block) % self._tokens_per_block
        tokens_in_last_block = min(capacity_in_last_block, num_tokens)
        tokens_remaining = num_tokens - tokens_in_last_block
        num_blocks = (tokens_remaining + self._tokens_per_block - 1) // self._tokens_per_block
        if len(self._free_blocks) < num_blocks:
            raise RuntimeError("OOM: GPU Cache Full")

        allocated = []
        for _ in range(num_blocks):
            allocated.append(self._free_blocks.pop())
        self._blocks_per_sequence[sequence_id].extend(allocated)

        self._seq_len[sequence_id] += num_tokens
        return allocated

    def reset(self) -> None:
        self._free_blocks = list(range(self._max_blocks))
        self._blocks_per_sequence = {}
        self._seq_len = {}

    def get_sequence_lengths(self, seq_ids: Sequence[Any]) -> torch.Tensor:
        return torch.tensor(
            [self._seq_len.get(sid, 0) for sid in seq_ids],
            dtype=torch.long,
        )

    def get_block_table_for(
            self,
            seq_ids: Sequence[Any],
            device: torch.device,
    ) -> torch.Tensor:
        if not seq_ids:
            return torch.empty(0, 0, dtype=torch.long, device=device)

        max_len = max(len(self._blocks_per_sequence[sid]) for sid in seq_ids)
        block_table = torch.full(
            (len(seq_ids), max_len),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        for i, sid in enumerate(seq_ids):
            blocks = self._blocks_per_sequence.get(sid, [])
            if blocks:
                block_table[i, : len(blocks)] = torch.tensor(
                    blocks,
                    dtype=torch.long,
                    device=device,
                )

        return block_table