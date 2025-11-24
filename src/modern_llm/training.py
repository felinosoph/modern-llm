import torch
import torch.nn.functional as F

def train_step(model, optimizer, scheduler, batch_data, device="cuda"):
    """
    batch_data: The tuple returned by your packed_collate_fn
    """
    model.train()

    input_ids, cu_seqlens, max_seqlen, labels = batch_data

    input_ids = input_ids.to(device)
    cu_seqlens = cu_seqlens.to(device)
    labels = labels.to(device)


    # (num_total_tokens x vocab_size)
    logits = model(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss.item()