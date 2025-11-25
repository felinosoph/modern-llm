import torch


class PretrainCollator:
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, batch_list):
        flat_input = []
        cu_seqlens = [0]
        labels = []

        for item in batch_list:
            # Raw text -> Dense Tokens
            text = item['text']
            tokens = self.tokenizer.encode(text)

            # Truncate
            if len(tokens) > self.block_size:
                tokens = tokens[:self.block_size]

            # Standard Next Token Prediction
            flat_input.extend(tokens[:-1])
            labels.extend(tokens[1:])
            cu_seqlens.append(len(flat_input))

        # To Tensor
        return self._pack_tensors(flat_input, labels, cu_seqlens)

    def _pack_tensors(self, flat_input, labels, cu_seqlens):
        input_tensor = torch.tensor(flat_input, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        cu_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)

        # Max Seq Len for Flash Attention
        seq_lens = cu_tensor[1:] - cu_tensor[:-1]
        max_seq = torch.max(seq_lens).item() if len(seq_lens) > 0 else 0

        return {
            "input_ids": input_tensor,
            "labels": label_tensor,
            "cu_seqlens": cu_tensor,
            "max_seqlen": max_seq
        }


class SFTCollator(PretrainCollator):
    def __init__(self, tokenizer, block_size):
        super().__init__(tokenizer, block_size)

    def format_prompt(self, item):
        # Orca / Generic Chat Format
        sys = item.get('system', '')
        ques = item.get('question', '')
        ans = item.get('chosen', '')

        prompt = f"User: {sys} {ques}\nAssistant: "
        full_text = prompt + ans
        return prompt, full_text

    def __call__(self, batch_list):
        flat_input = []
        cu_seqlens = [0]
        labels = []

        for item in batch_list:
            prompt_text, full_text = self.format_prompt(item)

            prompt_tokens = self.tokenizer.encode(prompt_text)
            full_tokens = self.tokenizer.encode(full_text)

            # Truncate
            if len(full_tokens) > self.block_size:
                full_tokens = full_tokens[:self.block_size]

            flat_input.extend(full_tokens[:-1])
            seq_labels = full_tokens[1:]

            # --- MASKING LOGIC ---
            # Mask out the prompt so we don't train on it
            prompt_len = len(prompt_tokens)
            # We mask indices 0 to (prompt_len - 2) in the shifted labels
            mask_limit = max(0, prompt_len - 1)

            for i in range(len(seq_labels)):
                if i < mask_limit:
                    seq_labels[i] = -100  # PyTorch ignore_index

            labels.extend(seq_labels)
            cu_seqlens.append(len(flat_input))

        return self._pack_tensors(flat_input, labels, cu_seqlens)