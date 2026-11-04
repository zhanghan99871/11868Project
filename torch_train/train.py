"""Fine-tune GPT-2 Medium initialized PyTorch MLA decoder on WikiText-2.

Example:
 python train.py --output-dir runs/mla-gpt2-medium --seq-len 128 \
   --batch-size 1 --grad-accum 16 --epochs 1 --max-steps 300

Pretrained weights are converted by randomized SVD on the first run.
Afterwards supply --init-checkpoint path/to/last.pt to continue from an
MLA checkpoint (model weights only; optimizer is restarted).
"""
import argparse
import gc
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from dataloader import make_dataloader
from decoder_mla import DecoderLM, MLAConfig
from gpt2_mla_init import make_gpt2_mla_config, initialize_from_gpt2

import matplotlib
matplotlib.use("Agg")  # Save figures without opening a GUI
import matplotlib.pyplot as plt


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', default='openai-community/gpt2-medium')
    p.add_argument('--output-dir', default='runs/mla-gpt2-medium')
    p.add_argument('--data-cache', default='./data')
    p.add_argument('--init-checkpoint', default=None)
    p.add_argument('--kv-rank', type=int, default=512)
    p.add_argument('--rope-dim', type=int, default=32)
    p.add_argument('--svd-device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--svd-niter', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=0, help='Optimizer steps; 0 = no limit')
    p.add_argument('--max-train-examples', type=int, default=0, help='Limit dataset blocks; 0 = all')
    p.add_argument('--max-eval-batches', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-steps', type=int, default=20)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--seed', type=int, default=42) 
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    p.add_argument('--no-bf16', action='store_true')
    p.add_argument('--num-workers', type=int, default=0)
    return p.parse_args()


def lm_loss(model, batch, *, bf16=False):
    """Batch [B,T+1] -> causal teacher-forced next-token loss."""
    x = batch[:, :-1]
    targets = batch[:, 1:].contiguous()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                        enabled=bf16):
        logits = model(x)
        # Cross entropy in float32 for stability.
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    return loss


@torch.no_grad()
def evaluate(model, loader, device, bf16=False, max_batches=0):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for n, batch in enumerate(loader):
            if max_batches and n >= max_batches:
                break
            batch = batch.to(device, non_blocking=True)
            loss = lm_loss(model, batch, bf16=bf16)
            tokens = batch[:, 1:].numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    finally:
        model.train(was_training)
    avg = total_loss / max(total_tokens, 1)
    return {'loss': avg, 'perplexity': math.exp(min(avg, 80)), 'tokens': total_tokens}


def save_checkpoint(path, model, step, cfg, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # To avoid GPU checkpoint corruption or collecting unnecessary optimizer states,
    # checkpoint the model weights + architecture only. Optimizer restarts on resume.
    torch.save({
        'model': model.state_dict(),
        'config': asdict(cfg),
        'step': step,
        'metadata': metadata,
    }, path)
    print(f'Saved model checkpoint: {path}', flush=True)

def plot_loss_curve(train_steps, train_losses,
                    val_steps, val_losses, output_dir):
    plt.figure(figsize=(10, 6))

    plt.plot(
        train_steps,
        train_losses,
        label="Training Loss",
        alpha=0.5,
        linewidth=1,
    )

    # Smoothed training loss
    if len(train_losses) >= 10:
        window = min(20, len(train_losses))
        smoothed = np.convolve(
            train_losses,
            np.ones(window) / window,
            mode="valid",
        )

        plt.plot(
            train_steps[window - 1:],
            smoothed,
            label=f"Training Loss (MA={window})",
            linewidth=2,
        )

    if val_losses:
        plt.plot(
            val_steps,
            val_losses,
            "o-",
            label="Validation Loss",
            linewidth=2,
        )

    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("GPT-2 Medium MLA Fine-Tuning Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        Path(output_dir) / "loss_curve.png",
        dpi=200,
    )
    plt.close()


def main():
    args = arguments()
    if args.batch_size < 1 or args.grad_accum < 1 or args.epochs < 1:
        raise ValueError('batch-size, grad-accum and epochs must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for GPT-2 Medium training; use a GPU environment')
    if args.seq_len > 1024:
        raise ValueError('GPT-2 Medium has a 1024-token positional embedding table')
    device = torch.device('cuda')
    bf16 = not args.no_bf16 and torch.cuda.is_bf16_supported()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'arguments.json').open('w') as f:
        json.dump(vars(args), f, indent=2)

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        cfg = MLAConfig(**ckpt['config'])
        model = DecoderLM(cfg)
        model.load_state_dict(ckpt['model'], strict=True)
        del ckpt
        print('Loaded MLA checkpoint; AdamW optimizer will start fresh.', flush=True)
    else:
        # Avoid allocating the full HF model on GPU.
        from transformers import GPT2LMHeadModel
        print(f'Loading teacher initialization: {args.pretrained}', flush=True)
        hf = GPT2LMHeadModel.from_pretrained(args.pretrained).eval().cpu()
        cfg = make_gpt2_mla_config(hf.config, kv_rank=args.kv_rank,
                                  rope_dim=args.rope_dim, dropout=0.0)
        model = DecoderLM(cfg)
        initialize_from_gpt2(model, hf, svd_device=args.svd_device,
                             svd_niter=args.svd_niter)
        del hf
        gc.collect()
        torch.cuda.empty_cache()
    if args.seq_len > cfg.n_positions:
        raise ValueError(f'seq_len exceeds model max positions {cfg.n_positions}')

    model = model.to(device)
    model.gradient_checkpointing = not args.no_gradient_checkpointing
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'BF16 autocast={bf16}; gradient checkpointing={model.gradient_checkpointing}', flush=True)

    train_loader = make_dataloader(
        'train', seq_len=args.seq_len, batch_size=args.batch_size,
        cache_dir=args.data_cache, max_examples=args.max_train_examples,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = make_dataloader(
        'validation', seq_len=args.seq_len, batch_size=args.batch_size,
        cache_dir=args.data_cache, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_updates = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    if args.max_steps:
        total_updates = min(total_updates, args.max_steps)
    total_updates = max(total_updates, 1)

    def lr_at(step):
        if step <= args.warmup_steps and args.warmup_steps > 0:
            return args.lr * step / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_updates - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    train_steps = []
    train_losses = []

    val_steps = []
    val_losses = []
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        accum_count = 0
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            loss = lm_loss(model, batch, bf16=bf16)
            # Partial last group handled by division with accumulated count.
            loss.backward()
            accum_count += 1
            running_loss += loss.detach().item()
            end_of_epoch = (i + 1) == len(train_loader)
            if accum_count < args.grad_accum and not end_of_epoch:
                continue
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(accum_count)
            clip_grad_norm_(model.parameters(), args.clip_grad)
            step += 1
            train_steps.append(step)
            train_losses.append(running_loss / accum_count)
            lr = lr_at(step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % args.log_every == 0:
                print(f'epoch={epoch+1} step={step}/{total_updates} '
                      f'loss={running_loss/accum_count:.4f} lr={lr:.2e} '
                      f'peak_mem={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB',
                      flush=True)
            running_loss = 0.0
            accum_count = 0

            if args.eval_every and step % args.eval_every == 0:
                stats = evaluate(
                    model, val_loader, device, bf16,
                    args.max_eval_batches,
                )

                val_steps.append(step)
                val_losses.append(stats["loss"])

                print(f"validation step={step}: {stats}", flush=True)

                plot_loss_curve(
                    train_steps,
                    train_losses,
                    val_steps,
                    val_losses,
                    output,
                )

                # save_checkpoint(
                #     output / "last.pt",
                #     model,
                #     step,
                #     cfg,
                #     {"source": args.pretrained, "validation": stats},
                # )

                if stats["loss"] < best_val_loss:
                    best_val_loss = stats["loss"]

                    save_checkpoint(
                        output / "best.pt",
                        model,
                        step,
                        cfg,
                        {
                            "source": args.pretrained,
                            "validation": stats,
                        },
                    )

                    print(
                        f"New best validation loss: {best_val_loss:.4f}",
                        flush=True,
                    )
            if args.max_steps and step >= args.max_steps:
                break

        if not (args.max_steps and step >= args.max_steps):
            print(f'Finished epoch {epoch+1}', flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    stats = evaluate(model, val_loader, device, bf16, args.max_eval_batches)
    val_steps.append(step)
    val_losses.append(stats["loss"])

    plot_loss_curve(
        train_steps,
        train_losses,
        val_steps,
        val_losses,
        output,
    )
    print('Final validation:', stats, flush=True)
    with (output / "loss_history.json").open("w") as f:
        json.dump({
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_steps": val_steps,
            "val_losses": val_losses,
        }, f, indent=2)


if __name__ == '__main__':
    main()
