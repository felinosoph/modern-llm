import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class FlexibleAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, mode="training", max_seq_len=None):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.mode = mode  # Store it explicitly

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)


        if self.mode == "inference":
            if max_seq_len is None:
                raise ValueError("For inference mode, you MUST specify max_seq_len.")

            cache_shape = (1, max_seq_len, 2, n_head, self.d_head)
            self.register_buffer("kv_cache", torch.zeros(cache_shape, dtype=torch.float16))
        else:

            self.kv_cache = None

    def forward(self, x, cu_seqlens=None, max_seqlen=None, cache_seqlens=None):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        if self.mode == "training":
            q = q.view(-1, self.n_head, self.d_head)
            k = k.view(-1, self.n_head, self.d_head)
            v = v.view(-1, self.n_head, self.d_head)

            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True
            )
            output = output.view(-1, self.n_head * self.d_head)

        elif self.mode == "inference":
            # Inference Logic (Linear Kernel)
            B, T, _ = x.shape
            q = q.view(B, T, self.n_head, self.d_head)
            k = k.view(B, T, self.n_head, self.d_head)
            v = v.view(B, T, self.n_head, self.d_head)

            output = flash_attn_with_kvcache(
                q,
                k_cache=self.kv_cache[:, :, 0],
                v_cache=self.kv_cache[:, :, 1],
                k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True
            )
            output = output.view(B, T, self.n_head * self.d_head)

        return self.W_o(output)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 128
N_HEAD = 4
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 1

# Precision Setup
if torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float16
vocab = {
    "Cats": 0, "are": 1, "the": 2, "cutest": 3,
    "Will": 4, "of": 5, "Many": 6, "EOS": 7
}
idx_to_word = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
class TinyLLM(nn.Module):
    def __init__(self, mode="training", max_seq_len=None):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # EXPLICIT PASSING
        self.attn = FlexibleAttentionLayer(
            d_model=D_MODEL,
            n_head=N_HEAD,
            mode=mode,           # Pass the explicit mode
            max_seq_len=max_seq_len
        )
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x, cu_seqlens=None, max_seqlen=None, cache_seqlens=None):
        h = self.embed(x)
        h = self.attn(
            h,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cache_seqlens=cache_seqlens
        )
        logits = self.head(h)
        return logits

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
packed_target = torch.tensor(flat_target,device=DEVICE)
cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=DEVICE)
max_seqlen = max(len(s)-1 for s in data)

print(f"Training Data Packed: {packed_input.tolist()}")
print(f"Training Targets:     {packed_target.tolist()}")
print(f"Cumulative Seqlens:   {cu_seqlens.tolist()}")
print("\n=== STARTING TRAINING ===")
model_train = TinyLLM(mode="training").to(DEVICE).to(DTYPE)
optimizer = optim.AdamW(model_train.parameters(), lr=0.01) # Bumped LR slightly for tiny data
loss_fn = nn.CrossEntropyLoss()

for i in range(50): # 50 steps is plenty for 2 sentences
    optimizer.zero_grad()
    logits = model_train(packed_input, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    loss = loss_fn(logits.float(), packed_target)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"Step {i}: Loss {loss.item():.4f}")

print("Training Complete.")

model_infer = TinyLLM(
    mode="inference",
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE).to(DTYPE)
model_infer.load_state_dict(model_train.state_dict(), strict=False)

def run_inference(prompt_word):
    print(f"\nPrompt: '{prompt_word}'")
    token_id = vocab[prompt_word]

    cache_seqlens = torch.tensor([0], dtype=torch.int32, device=DEVICE)


    x_in = torch.tensor([[token_id]], device=DEVICE)
    output_str = prompt_word


    for _ in range(5):
        with torch.no_grad():
            logits = model_infer(x_in, cache_seqlens=cache_seqlens)

        next_token_id = torch.argmax(logits[0, 0]).item()
        word = idx_to_word[next_token_id]

        if word == "EOS":
            break

        output_str += " " + word

        cache_seqlens += 1
        x_in = torch.tensor([[next_token_id]], device=DEVICE)

    print(f"Generated: '{output_str}'")

run_inference("Cats")
run_inference("Will")
