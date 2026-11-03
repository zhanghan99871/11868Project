# GPT-2 Medium → MLA → WikiText-2 fine-tuning

## Setup

These scripts sit in the same folder as `decoder_mla.py`.

```bash
python -m pip install 'torch>=2.2' 'transformers>=4.40,<5' 'datasets>=2.18' 'numpy>=1.24' 'pytest>=8'
```

Using `transformers<5` is a conservative choice because the converter uses
`GPT2LMHeadModel`, GPT-2's Conv1D state dict, and the legacy config fields.

## Training (RTX 5080, 16 GB)

```bash
cd path/to/mla_torch
python train.py --output-dir runs/mla-medium --seq-len 128 \
   --batch-size 1 --grad-accum 16 --kv-rank 512 \
   --max-steps 300 --epochs 2 --lr 1e-5 --eval-every 100
```

First launch downloads GPT-2 Medium and WikiText-2 and performs 24 randomized
low-rank K/V initializations. This takes considerably longer than subsequent
checkpoint loads. The pretrained model is loaded only on CPU, and SVD can run
on CUDA. Training uses BF16 autocast and activation checkpointing by default.

If memory is tight, reduce `--seq-len 64` or pass `--svd-device cpu` if
initialization has CUDA allocation issues. To run a 2-step smoke experiment,
use `--max-steps 2 --max-train-examples 32 --max-eval-batches 2`.

`--init-checkpoint runs/mla-medium/last.pt` restarts fine-tuning from saved MLA
weights but **restarts the optimizer and scheduler** (not a bitwise training
resume).

## Evaluation

```bash
python eval.py --checkpoint runs/mla-medium/last.pt \
    --split validation --seq-len 128 --batch-size 1 \
    --prompt 'The capital of France is'
```

Use `--split test` for final reporting after model selection. By default
`eval.py` evaluates the whole split. `train.py` limits validation batches to
30 by default, for faster training feedback.

## Initialization details and limitations

* Embeddings, LayerNorms, MLP weights/biases and attention output projection
  copy exactly from GPT-2 Medium.
* Q_nope copies the GPT-2 Q matrix; scale is adjusted for the larger MLA
  Q/K dimension. Q_rope weights/bias start at zero. K_rope has nonzero random
  initialization so gradient can flow into Q_rope.
* Per-head K/V weights are interleaved and jointly approximated by a
  rank-`kv_rank` truncated randomized SVD. Their biases are copied.
* KV RMSNorm is **disabled** for this conversion (`use_kv_norm=False`);
  otherwise it would destroy the linear low-rank approximation.
* GPT-2's absolute position embeddings are retained. MLA's learned RoPE
  branch is additional and begins inactive. This is not pretrained DeepSeek MLA.
* The converted model **will not match the GPT-2 logits exactly** before
  training; low-rank truncation changes the attention operator.
* This is a correctness/reference MLA implementation: KV cache storage is
  compact, but heads are fully expanded during attention. It is not a
  compute-optimized or device-resident fused MLA kernel.
* WikiText-2 has limited text and is a good sanity-check fine-tuning dataset,
  but can overfit and should not be considered adequate to fully retrain a
  new attention architecture for general language modeling.
* MiniTorch-specific CUDA kernels are not required; this model trains in PyTorch.

## Files

- `decoder_mla.py`: existing decoder with optional KV normalization and
  activation checkpointing.
- `gpt2_mla_init.py`: approximate GPT-2 Medium → MLA weight transfer.
- `dataloader.py`: WikiText-2 tokenizer/cache and block dataloader.
- `train.py`: full-parameter AdamW + BF16 + checkpointing training script.
- `eval.py`: held-out token-weighted perplexity and cached generation.
- `test_training_smoke.py`: CPU-only synthetic training and cache checks.
