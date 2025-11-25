import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Callable, Dict, Any


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            train_dataloader,
            val_dataloader=None,
            scheduler=None,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            grad_accum_steps: int = 1,
            checkpoint_dir: str = "checkpoints",
            log_interval: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device
        self.dtype = dtype

        # Training mechanics
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.step = 0
        self.epoch = 0

    def compute_loss(self, batch, return_outputs=False):
        """
        Standard Next-Token-Prediction Loss.
        Override this method later for DPO!
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)  # Usually input_ids shifted by 1

        # Metadata for attention (if your model uses it)
        # We assume the collate_fn handles the creation of these
        cu_seqlens = batch.get("cu_seqlens", None)
        max_seqlen = batch.get("max_seqlen", None)

        # Forward Pass
        logits = self.model(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Flatten for CrossEntropy: (B*T, Vocab)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        if return_outputs:
            return loss, logits
        return loss

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        self.epoch = epoch_idx

        running_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # 1. Forward & Loss
            # We use autocast for mixed precision (bf16 is native on RTX 4080)
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                loss = self.compute_loss(batch)

            # 2. Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()

            running_loss += loss.item() * self.grad_accum_steps

            # 3. Optimizer Step (only every N steps)
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient Clipping (Optional but recommended)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.step += 1

                # Logging
                if self.step % self.log_interval == 0:
                    dt = time.time() - start_time
                    avg_loss = running_loss / self.log_interval
                    print(f"[Epoch {self.epoch} | Step {self.step}] "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                          f"Time: {dt:.2f}s")

                    running_loss = 0.0
                    start_time = time.time()

    def save_checkpoint(self, filename: str = None):
        if filename is None:
            filename = f"ckpt_step_{self.step}.pt"

        path = self.checkpoint_dir / filename
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")