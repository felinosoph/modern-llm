import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

from modern_llm.model import DecoderOnlyModel, ModelSpec
from modern_llm.training import Trainer
from modern_llm.data import PretrainCollator

# --- CONFIG ---
DEVICE = "cuda"
DTYPE = torch.bfloat16
BATCH_SIZE = 32
ACCUM_STEPS = 4
BLOCK_SIZE = 512
D_MODEL = 512
N_LAYERS = 8
N_HEAD = 8


def main():
    print("--- PRETRAINING PHASE (TinyStories) ---")

    # Setup
    enc = tiktoken.get_encoding("cl100k_base")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    collator = PretrainCollator(enc, BLOCK_SIZE)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator)

    # Model
    spec = ModelSpec(enc.n_vocab, N_LAYERS, D_MODEL, N_HEAD, D_MODEL * 4, BLOCK_SIZE)
    model = DecoderOnlyModel(spec).to(DEVICE).to(DTYPE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    trainer = Trainer(
        model, optimizer, train_loader,
        grad_accum_steps=ACCUM_STEPS, device=DEVICE, dtype=DTYPE,
        checkpoint_dir="checkpoints_pretrain"
    )

    # Run
    try:
        trainer.train_epoch(0)  # Infinite streaming epoch
    except KeyboardInterrupt:
        print("Saving Checkpoint...")
        trainer.save_checkpoint("pretrain_final.pt")


if __name__ == "__main__":
    main()