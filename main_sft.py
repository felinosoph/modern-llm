import os
import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

from modern_llm.model import DecoderOnlyModel, ModelSpec
from modern_llm.training import Trainer
from modern_llm.data import SFTCollator

# --- CONFIG ---
DEVICE = "cuda"
DTYPE = torch.bfloat16
PRETRAIN_PATH = "checkpoints_pretrain/pretrain_final.pt"
BATCH_SIZE = 32
ACCUM_STEPS = 4
BLOCK_SIZE = 512
D_MODEL = 512
N_LAYERS = 8
N_HEAD = 8
EPOCHS = 3


def main():
    print("--- SFT PHASE (Orca Instructions) ---")

    # Setup
    enc = tiktoken.get_encoding("cl100k_base")
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    collator = SFTCollator(enc, BLOCK_SIZE)

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collator,
        shuffle=True, pin_memory=True
    )

    # Model
    spec = ModelSpec(enc.n_vocab, N_LAYERS, D_MODEL, N_HEAD, D_MODEL * 4, BLOCK_SIZE)
    model = DecoderOnlyModel(spec).to(DEVICE).to(DTYPE)

    # Load Weights
    if os.path.exists(PRETRAIN_PATH):
        print(f"Loading pretrained weights from {PRETRAIN_PATH}")
        ckpt = torch.load(PRETRAIN_PATH, map_location=DEVICE)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: No pretrain checkpoint found! Training from random init.")

    # Trainer (Lower LR for SFT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    trainer = Trainer(
        model, optimizer, train_loader,
        grad_accum_steps=ACCUM_STEPS, device=DEVICE, dtype=DTYPE,
        checkpoint_dir="checkpoints_sft"
    )

    # Run
    try:
        for epoch in range(EPOCHS):
            print(f"--- Epoch {epoch + 1} ---")
            trainer.train_epoch(epoch)
            trainer.save_checkpoint(f"sft_epoch_{epoch}.pt")
    except KeyboardInterrupt:
        trainer.save_checkpoint("sft_interrupted.pt")


if __name__ == "__main__":
    main()