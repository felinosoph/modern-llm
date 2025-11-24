import torch
from modern_llm import KVCacheBlockManager


@torch.inference_mode()
def generate(model, prompt_tokens, manager:KVCacheBlockManager, max_new_tokens, device="cuda"):
  pass