import torch
import torch.nn as nn
import torch.optim as optim
from modern_llm.model import DecoderOnlyModel
from modern_llm.kv_cache import BlockSpaceManager

DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

vocab = {
    "Cats": 0, "are": 1, "the": 2, "cutest": 3,
    "Will": 4, "of": 5, "Many": 6, "EOS": 7
}
idx_to_word = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)

D_MODEL = 128
N_HEAD = 4
HIDDEN_DIM = 512
N_LAYERS = 2
BLOCK_SIZE = 256


def main():
    print(f"--- Starting 'Cat Test' on {DEVICE} ---")
    print(f"--- Constraints: BLOCK_SIZE={BLOCK_SIZE} ---")
    data = [
        [vocab["Cats"], vocab["are"], vocab["the"], vocab["cutest"], vocab["EOS"]],
        [vocab["Will"], vocab["of"], vocab["the"], vocab["Many"], vocab["EOS"]]
    ]

    flat_input = []
    flat_target = []
    cu_seqlens_list = [0]

    for seq in data:
        flat_input.extend(seq[:-1])
        flat_target.extend(seq[1:])
        cu_seqlens_list.append(len(flat_input))

    packed_input = torch.tensor(flat_input, device=DEVICE)
    packed_target = torch.tensor(flat_target, device=DEVICE)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=DEVICE)
    max_seqlen = max(len(s) - 1 for s in data)


    print("\n--> Phase 1: Overfitting (Standard Training Mode)")

    model = DecoderOnlyModel(
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_head=N_HEAD,
        hidden_dim=HIDDEN_DIM,
        page_block_size=None
    ).to(DEVICE).to(DTYPE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-2)  # Higher LR for instant overfitting

    for step in range(60):
        optimizer.zero_grad()
        logits = model(packed_input.unsqueeze(0), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        logits = logits.view(-1, VOCAB_SIZE)
        loss = nn.functional.cross_entropy(logits, packed_target)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:03d}: Loss = {loss.item():.6f}")

    print(f"Final Loss: {loss.item():.6f}")
    weights = model.state_dict()
    del model

    print("\n--> Phase 2: Inference (Paged Attention Mode)")

    infer_model = DecoderOnlyModel(
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_head=N_HEAD,
        hidden_dim=HIDDEN_DIM,
        page_block_size=BLOCK_SIZE,  # Enables Paged Kernel
        max_blocks=100
    ).to(DEVICE).to(DTYPE)

    infer_model.load_state_dict(weights, strict=False)

    kv_manager = BlockSpaceManager(
        max_blocks=100,
        block_size=BLOCK_SIZE,
        device=DEVICE
    )

    run_inference(infer_model, kv_manager, "Cats", vocab["Cats"], ["are", "the", "cutest", "EOS"])
    run_inference(infer_model, kv_manager, "Will", vocab["Will"], ["of", "the", "Many", "EOS"])


def run_inference(model, manager, prompt_text, prompt_id, expected_tokens):
    print(f"\nTesting Prompt: '{prompt_text}' (ID: {prompt_id})")

    user_id = f"user_{prompt_text}"
    manager.allocate(user_id, 1)
    block_table, cache_seqlens = manager.get_metadata_tensors([user_id])
    
    input_t = torch.tensor([[prompt_id]], device=DEVICE)
    with torch.no_grad():
        logits = model(input_t, block_table=block_table, cache_seqlens=cache_seqlens)

    print("Generation: ", end="")
    next_token = torch.argmax(logits[0, -1]).item()
    word = idx_to_word[next_token]
    print(f"{word} ", end="")

    curr_input = torch.tensor([[next_token]], device=DEVICE)

    for _ in range(len(expected_tokens) - 1):
        manager.step([user_id])
        block_table, cache_seqlens = manager.get_metadata_tensors([user_id])

        with torch.no_grad():
            logits = model(curr_input, block_table=block_table, cache_seqlens=cache_seqlens)

        next_token = torch.argmax(logits[0, -1]).item()
        print(f"{idx_to_word[next_token]} ", end="")
        curr_input = torch.tensor([[next_token]], device=DEVICE)

    print("\n(Done)")


if __name__ == "__main__":
    main()
