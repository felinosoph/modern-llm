import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

from modern_llm.model import DecoderOnlyModel, ModelSpec
from modern_llm.training import Trainer

# --- Config for RTX 4080 ---
BATCH_SIZE = 8  # Reduced for stability with slightly longer texts
ACCUM_STEPS = 8  # Effective batch = 64
BLOCK_SIZE = 512
D_MODEL = 256
N_LAYERS = 8
N_HEAD = 8
MAX_STEPS = 5000

DEVICE = "cuda"
DTYPE = torch.bfloat16


def main():
    # 1. Tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab
    print(f"Vocab Size: {vocab_size}")

    # 2. Dataset: Intel Orca DPO Pairs
    # We use this for BOTH SFT (now) and DPO (later)
    print("Loading Intel/orca_dpo_pairs...")
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")

    # Note: We don't stream here because it's small (~12k rows) and fits in RAM easily.

    # 3. Formatter: Raw Data -> Chat Format
    def format_prompt(row):
        # We ignore 'rejected' for SFT training
        sys = row.get('system', '')
        ques = row.get('question', '')
        ans = row.get('chosen', '')

        # Simple Chat Format
        # User: <question>
        # Assistant: <answer>
        text = f"User: {sys} {ques}\nAssistant: {ans}"
        return text

    # 4. Collate Function
    def collate_fn(batch_list):
        flat_input = []
        cu_seqlens = [0]
        labels = []

        for item in batch_list:
            text = format_prompt(item)
            tokens = enc.encode(text)

            # Truncate to block size
            if len(tokens) > BLOCK_SIZE:
                tokens = tokens[:BLOCK_SIZE]

            # Create Inputs/Labels for Next Token Prediction
            flat_input.extend(tokens[:-1])
            labels.extend(tokens[1:])
            cu_seqlens.append(len(flat_input))

        # Tensorize
        input_tensor = torch.tensor(flat_input, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        cu_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)

        # Calculate max_seqlen for Flash Attention
        seq_lens = cu_tensor[1:] - cu_tensor[:-1]
        max_seq = torch.max(seq_lens).item()

        return {
            "input_ids": input_tensor,
            "labels": label_tensor,
            "cu_seqlens": cu_tensor,
            "max_seqlen": max_seq
        }

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True
    )

    # 5. Model Setup
    spec = ModelSpec(
        vocab_size=vocab_size,
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_head=N_HEAD,
        hidden_dim=D_MODEL * 4
    )

    model = DecoderOnlyModel(spec).to(DEVICE).to(DTYPE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 6. Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        grad_accum_steps=ACCUM_STEPS,
        device=DEVICE,
        dtype=DTYPE,
        log_interval=10
    )

    # 7. Execution
    print("Starting SFT on Orca Pairs (Chosen)...")
    try:
        # Train for a few epochs since dataset is small (12k)
        for epoch in range(3):
            trainer.train_epoch(epoch_idx=epoch)
            trainer.save_checkpoint(f"orca_sft_epoch_{epoch}.pt")

    except KeyboardInterrupt:
        print("Saving emergency checkpoint...")
        trainer.save_checkpoint("interrupted.pt")


if __name__ == "__main__":
    main()