"""Evaluate unmodified Hugging Face GPT-2 on the *same* WikiText-2 blocks as MLA.

Place in 11868Project/torch_train/ beside dataloader.py.

Example:
    python torch_train/eval_gpt2_baseline.py --split validation --seq-len 128 --batch-size 1 --max-eval-batches 0

The dataset loader uses the same cached tokenization and [B, T+1] blocks as train.py.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from dataloader import make_dataloader


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='openai-community/gpt2-medium')
    p.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    p.add_argument('--data-cache', default='./data', help='Same cache directory as MLA train.py')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-eval-batches', type=int, default=0,
                   help='0 evaluates the entire split; 30 matches MLA train.py default')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--no-bf16', action='store_true', help='Disable CUDA BF16 autocast')
    p.add_argument('--output', default=None, help='Optional path for JSON metrics')
    return p.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device, *, bf16, max_batches):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = batch.to(device, non_blocking=(device.type == 'cuda'))
        inputs = batch[:, :-1]
        targets = batch[:, 1:].contiguous()

        # Match MLA train.py: teacher-forced next-token logits, FP32 cross-entropy.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bf16):
            logits = model(inputs, use_cache=False).logits
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='sum',
            )

        total_nll += float(loss.item())
        total_tokens += targets.numel()
        if (i + 1) % 50 == 0:
            print(f'evaluated_batches={i+1} evaluated_tokens={total_tokens:,}', flush=True)

    if total_tokens == 0:
        raise RuntimeError('No evaluation tokens: check the dataset and max-eval-batches')
    avg_loss = total_nll / total_tokens
    return {
        'loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 80.0)),
        'tokens': total_tokens,
    }


def main():
    args = parse_args()
    if args.seq_len < 2 or args.batch_size < 1 or args.max_eval_batches < 0:
        raise ValueError('seq-len >= 2, batch-size >= 1, max-eval-batches >= 0 required')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    bf16 = device.type == 'cuda' and not args.no_bf16 and torch.cuda.is_bf16_supported()

    print(f'Model: {args.model}; device: {device}; BF16 autocast: {bf16}', flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    if args.seq_len > model.config.n_positions:
        raise ValueError(f'seq-len {args.seq_len} exceeds GPT-2 context {model.config.n_positions}')

    loader = make_dataloader(
        args.split, seq_len=args.seq_len, batch_size=args.batch_size,
        cache_dir=args.data_cache, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )
    print(f'{args.split}: {len(loader)} total batches; limit: '
          f'{args.max_eval_batches or "all"}', flush=True)
    stats = evaluate(model, loader, device, bf16=bf16,
                     max_batches=args.max_eval_batches)
    results = {
        'model': args.model, 'split': args.split, 'seq_len': args.seq_len,
        'batch_size': args.batch_size, 'max_eval_batches': args.max_eval_batches,
        'bf16': bf16, **stats,
    }
    print(json.dumps(results, indent=2), flush=True)
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(results, indent=2) + '\n')
        print(f'Saved metrics: {dest}', flush=True)


if __name__ == '__main__':
    main()
