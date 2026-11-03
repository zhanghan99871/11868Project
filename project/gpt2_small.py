from transformers import GPT2LMHeadModel, AutoTokenizer
from gpt2_loader import * 
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

backend = minitorch.TensorBackend(CudaKernelOps)  # Use your actual backend object

hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
hf.eval()

cfg = hf.config

mt = minitorch.DecoderLM(
    n_vocab=cfg.vocab_size,
    n_embd=cfg.n_embd,
    n_head=cfg.n_head,
    n_positions=cfg.n_positions,
    n_layer=cfg.n_layer,
    p_dropout=0.0,
    ln_eps=cfg.layer_norm_epsilon,
    bias=True,
    backend=backend,
    use_fused_kernel=True,
)

load_gpt2_weights(mt, hf, backend)
mt.eval()

import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2"
)

ids = tokenizer(
    "The capital of France is",
    return_tensors="pt",
)["input_ids"]

with torch.no_grad():
    hf_logits = hf(ids).logits.numpy()

mt_ids = minitorch.tensor_from_numpy(
    ids.numpy().astype(np.float32),
    backend=backend,
)

mt_logits = mt(mt_ids).to_numpy()

print(
    "Max absolute difference:",
    np.max(np.abs(hf_logits - mt_logits))
)

print(
    "PyTorch next token:",
    tokenizer.decode([hf_logits[0, -1].argmax()])
)

print(
    "MiniTorch next token:",
    tokenizer.decode([mt_logits[0, -1].argmax()])
)