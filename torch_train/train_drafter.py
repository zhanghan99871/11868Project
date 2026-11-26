"""Fine-tune the H-Spec drafter (hspec_drafter.py) against the frozen MLA verifier on WikiText-2.

Example (train.sh in the repo root runs this):
  python torch_train/train_drafter.py --verifier-checkpoint runs/mla-gpt2-medium/best.pt \
      --init distilgpt2 --output-dir runs/hspec-drafter

Initialization: the drafter's trunk (embeddings, LayerNorms, MLPs) comes from distilgpt2, each
layer's kv_up from the verifier layer whose latents it reads, and the output head is the
verifier's frozen LM head.

Each optimizer step: one frozen verifier pass over a batch of WikiText-2 blocks gives the
drafter's context (MLA latents of the read layers, hidden-state taps) and the teacher logits.
--anchors-per-seq random anchors per sequence each define a block [anchor, MASK x k]; the drafter
drafts all blocks in one pass and is trained with H-Spec's loss: 0.1 CE on the data tokens +
0.9 TV to the verifier, with position weights exp(-(j-1)/4). The verifier is never updated.

Validation reports the loss, per-position acceptance (1 - TV: the rejection-sampling acceptance
probability) and top-1 agreement, and tau_greedy = 1 + mean number of leading positions whose
argmax matches the verifier's (teacher-forced on the data, so an estimate of the mean accepted
length per block under greedy decoding). best.pt ({'model', 'config', 'step', 'metadata'}, the
frozen head excluded) is saved whenever validation loss improves, including after the final
evaluation; export it for MiniTorch with torch_train/export_hspec_drafter.py.
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from decoder_mla import DecoderLM, MLAConfig
from hspec_drafter import (DrafterConfig, HSpecDrafter, block_targets, drafter_loss, init_from_gpt2,
                           init_from_verifier, sample_anchors, target_context)
from train import save_if_best

INIT_MODELS = {"distilgpt2": "distilbert/distilgpt2", "gpt2": "openai-community/gpt2"}


def arguments(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune the H-Spec drafter against the MLA verifier on WikiText-2.")
    p.add_argument("--verifier-checkpoint", default="runs/mla-gpt2-medium/best.pt",
                   help="MLA verifier written by torch_train/train.py (frozen)")
    p.add_argument("--output-dir", default="runs/hspec-drafter")
    p.add_argument("--data-cache", default="./data", help="WikiText-2 token cache (torch_train/dataloader.py)")
    p.add_argument("--init", choices=sorted(INIT_MODELS) + ["none"], default="distilgpt2",
                   help="Hugging Face checkpoint for the trunk (embeddings, LayerNorms, MLPs)")
    p.add_argument("--init-checkpoint", default=None,
                   help="continue from a drafter best.pt instead (weights only; optimizer restarts)")
    # drafter shape (attention dims and layer maps follow the verifier: DrafterConfig.for_verifier)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=768, help="drafter width (768 to start from distilgpt2/gpt2)")
    p.add_argument("--block-size", type=int, default=7, help="tokens drafted per pass (k)")
    p.add_argument("--n-taps", type=int, default=5, help="verifier layers fused into the summary")
    p.add_argument("--summary-to-all-positions", action="store_true")
    p.add_argument("--no-position-embeddings", action="store_true")
    # data
    p.add_argument("--seq-len", type=int, default=128, help="tokens per sequence (the verifier was fine-tuned at 128)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--anchors-per-seq", type=int, default=32, help="blocks drafted per sequence per step")
    p.add_argument("--max-train-examples", type=int, default=0, help="limit training sequences; 0 = all")
    p.add_argument("--num-workers", type=int, default=0)
    # optimization (H-Spec: AdamW, wd 0.01, clip 1.0, 1%% linear warmup, half-cosine decay to 0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps; 0 = no limit")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=1.0)
    # loss
    p.add_argument("--ce-weight", type=float, default=0.1)
    p.add_argument("--tv-weight", type=float, default=0.9)
    p.add_argument("--gamma", type=float, default=4.0, help="position weights exp(-(j-1)/gamma)")
    # evaluation / logging
    p.add_argument("--eval-every", type=int, default=500, help="optimizer steps; 0 = only at the end")
    p.add_argument("--max-eval-batches", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast on CUDA")
    args = p.parse_args(argv)
    for name in ("n_layer", "n_embd", "block_size", "n_taps", "seq_len", "batch_size", "anchors_per_seq",
                 "epochs", "grad_accum"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.anchors_per_seq > args.seq_len - 1 - args.block_size:
        raise ValueError(f"--anchors-per-seq must be <= seq_len - 1 - block_size = {args.seq_len - 1 - args.block_size}")
    return args


def load_verifier(path, device):
    """The frozen PyTorch MLA verifier from a torch_train/train.py checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not {"model", "config"} <= set(ckpt):
        raise ValueError(f"{path} is not a torch_train/train.py MLA checkpoint")
    verifier = DecoderLM(MLAConfig(**ckpt["config"]))
    verifier.load_state_dict(ckpt["model"], strict=True)
    verifier.requires_grad_(False)
    return verifier.to(device).eval()


def build_drafter(verifier, args, gpt2_state=None):
    """A drafter for `verifier`: resumed from --init-checkpoint, or new with its trunk from --init
    (gpt2_state overrides the download) and kv_up + head from the verifier. Returns (drafter, info)."""
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        cfg = DrafterConfig(**ckpt["config"])
        cfg.check_verifier(verifier.cfg)
        drafter = HSpecDrafter(cfg)
        drafter.load_state_dict(ckpt["model"], strict=True)
        drafter.set_target_head(verifier.lm_head.weight)  # trained kv_up is kept
        return drafter, {"resumed_from": str(args.init_checkpoint), "resumed_step": ckpt.get("step")}

    cfg = DrafterConfig.for_verifier(verifier.cfg, n_layer=args.n_layer, n_embd=args.n_embd,
                                     n_taps=args.n_taps, block_size=args.block_size,
                                     summary_to_all_positions=args.summary_to_all_positions,
                                     use_position_embeddings=not args.no_position_embeddings)
    drafter = HSpecDrafter(cfg)
    info = {"init": args.init}
    if args.init != "none":
        if gpt2_state is None:
            from transformers import GPT2LMHeadModel
            gpt2_state = GPT2LMHeadModel.from_pretrained(INIT_MODELS[args.init]).state_dict()
        info["trunk_source_layers"] = init_from_gpt2(drafter, gpt2_state)
    init_from_verifier(drafter, verifier)
    return drafter, info


def lr_at(step, total, peak, warmup):
    """Linear warmup to `peak` over `warmup` steps, then half-cosine decay to 0 at `total` (step >= 1)."""
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def param_groups(model, weight_decay):
    """Weight decay on matrices and embeddings only, not on biases, LayerNorms or the MASK vector."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def drafter_step(drafter, verifier, tokens, anchors, args, autocast):
    """Verifier context for `tokens`, the drafter's blocks at `anchors`, and the loss. Returns (loss, stats)."""
    cfg = drafter.cfg
    with autocast():
        latents, hidden, logits = target_context(verifier, tokens, cfg)
    targets, teacher = block_targets(tokens, logits, anchors, cfg.block_size)
    del logits
    latents = [(c.float(), kr.float()) for c, kr in latents]
    with autocast():
        out = drafter(tokens, anchors, latents, hidden.float())
    loss, stats = drafter_loss(out, targets, teacher, gamma=args.gamma, ce_weight=args.ce_weight,
                               tv_weight=args.tv_weight)
    with torch.no_grad():
        agree = (out.argmax(-1) == teacher.argmax(-1)).int()
        stats["greedy_accept_len"] = agree.cumprod(-1).sum(-1).float().mean().item()
    return loss, stats


@torch.no_grad()
def evaluate(drafter, verifier, loader, args, device, autocast):
    """Validation metrics on the same anchors every time (fixed seed)."""
    was_training = drafter.training
    drafter.eval()
    gen = torch.Generator().manual_seed(args.seed + 1)
    sums, per_pos, count, batches = defaultdict(float), defaultdict(lambda: 0.0), 0, 0
    try:
        for i, batch in enumerate(loader):
            if args.max_eval_batches and i >= args.max_eval_batches:
                break
            tokens = batch[:, :args.seq_len].to(device)
            anchors = sample_anchors(tokens.size(0), tokens.size(1), drafter.cfg.block_size,
                                     args.anchors_per_seq, gen, device)
            loss, stats = drafter_step(drafter, verifier, tokens, anchors, args, autocast)
            n = tokens.size(0)
            sums["loss"] += loss.item() * n
            for key in ("ce", "tv", "greedy_accept_len"):
                sums[key] += stats[key] * n
            for key in ("acceptance", "top1_agree"):
                per_pos[key] = per_pos[key] + np.asarray(stats[key]) * n
            count += n
            batches += 1
    finally:
        drafter.train(was_training)
    if not count:
        raise ValueError("empty validation loader")
    result = {key: value / count for key, value in sums.items()}
    result.update({key: (value / count).tolist() for key, value in per_pos.items()})
    result["tau_greedy"] = 1 + result["greedy_accept_len"]
    result["mean_acceptance"] = float(np.mean(result["acceptance"]))
    result["batches"] = batches
    return result


def plot_history(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if history["train"]:
        ax1.plot([h["step"] for h in history["train"]], [h["loss"] for h in history["train"]],
                 alpha=0.5, label="train")
    if history["val"]:
        steps = [h["step"] for h in history["val"]]
        ax1.plot(steps, [h["loss"] for h in history["val"]], "o-", label="validation")
        ax2.plot(steps, [h["tau_greedy"] for h in history["val"]], "o-", label="tau_greedy (est.)")
        ax2.plot(steps, [1 + sum(h["acceptance"]) for h in history["val"]], "s--",
                 label="1 + sum of per-position acceptance")
    ax1.set(xlabel="optimizer step", ylabel="loss (0.1 CE + 0.9 TV)", title="H-Spec drafter loss")
    ax2.set(xlabel="optimizer step", ylabel="tokens per block", title="validation acceptance")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "loss_curve.png", dpi=150)
    plt.close(fig)


def train(drafter, verifier, train_loader, val_loader, args, device, output_dir, bf16=False, info=None):
    """Optimize `drafter` against the frozen `verifier`. Returns (history, best validation loss)."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "arguments.json", "w") as f:
        json.dump({**vars(args), "bf16": bf16, "init_info": info or {}}, f, indent=2)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)

    drafter.to(device).train()
    verifier.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(param_groups(drafter, args.weight_decay), lr=args.lr)
    total = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    total = max(total, 1)
    warmup = int(args.warmup_ratio * total)
    gen = torch.Generator().manual_seed(args.seed)
    source = {"verifier": str(args.verifier_checkpoint), "init": args.init, **(info or {})}
    history, best, step = {"train": [], "val": []}, float("inf"), 0
    k, n_anchors = drafter.cfg.block_size, args.anchors_per_seq
    print(f"{total} optimizer steps ({len(train_loader)} batches/epoch, {args.epochs} epochs), "
          f"warmup {warmup}, bf16={bf16}", flush=True)

    def run_eval():
        nonlocal best
        stats = evaluate(drafter, verifier, val_loader, args, device, autocast)
        history["val"].append({"step": step, **stats})
        print(f"validation step={step}: loss {stats['loss']:.4f} tv {stats['tv']:.4f} "
              f"acceptance {[round(a, 3) for a in stats['acceptance']]} tau_greedy {stats['tau_greedy']:.3f}",
              flush=True)
        best = save_if_best(stats, best, output / "best.pt", drafter, step, drafter.cfg, source)
        with open(output / "loss_history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_history(history, output)

    t0, done = time.perf_counter(), False
    for epoch in range(args.epochs):
        accum, running = 0, defaultdict(float)
        for i, batch in enumerate(train_loader):
            tokens = batch[:, :args.seq_len].to(device, non_blocking=True)
            anchors = sample_anchors(tokens.size(0), tokens.size(1), k, n_anchors, gen, device)
            loss, stats = drafter_step(drafter, verifier, tokens, anchors, args, autocast)
            loss.backward()
            accum += 1
            running["loss"] += loss.item()
            running["tv"] += stats["tv"]
            running["acceptance"] += float(np.mean(stats["acceptance"]))
            running["greedy_accept_len"] += stats["greedy_accept_len"]
            if accum < args.grad_accum and (i + 1) < len(train_loader):
                continue
            for p in drafter.parameters():
                if p.grad is not None:
                    p.grad.div_(accum)
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.clip_grad)
            step += 1
            lr = lr_at(step, total, args.lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, "epoch": epoch + 1, "lr": lr, **{key: v / accum for key, v in running.items()}}
            history["train"].append(record)
            if step == 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                mem = f" peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device.type == "cuda" else ""
                print(f"epoch {epoch + 1} step {step}/{total} loss {record['loss']:.4f} tv {record['tv']:.4f} "
                      f"acceptance {record['acceptance']:.3f} tau_greedy~{1 + record['greedy_accept_len']:.2f} "
                      f"lr {lr:.2e} {elapsed / step:.2f}s/step{mem}", flush=True)
            accum, running = 0, defaultdict(float)
            if args.eval_every and step % args.eval_every == 0:
                run_eval()
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break
    if not history["val"] or history["val"][-1]["step"] != step:
        run_eval()  # final evaluation (also checkpoints if it is the best)
    print(f"Done: {step} steps, best validation loss {best:.4f}, {time.perf_counter() - t0:.0f}s", flush=True)
    return history, best


def main(argv=None):
    args = arguments(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (use --device cpu for a slow CPU run)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    bf16 = not args.no_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

    print(f"Loading verifier {args.verifier_checkpoint}", flush=True)
    verifier = load_verifier(args.verifier_checkpoint, device)
    if args.seq_len > verifier.cfg.n_positions:
        raise ValueError(f"--seq-len exceeds the verifier's {verifier.cfg.n_positions} positions")
    drafter, info = build_drafter(verifier, args)
    n_params = sum(p.numel() for p in drafter.parameters())
    print(f"Drafter: {n_params / 1e6:.1f}M trainable parameters; reads verifier layers "
          f"{drafter.cfg.target_layers}, taps {drafter.cfg.tap_layers}; {info}", flush=True)

    from dataloader import make_dataloader
    pin = device.type == "cuda"
    train_loader = make_dataloader("train", seq_len=args.seq_len, batch_size=args.batch_size,
                                   cache_dir=args.data_cache, max_examples=args.max_train_examples,
                                   num_workers=args.num_workers, pin_memory=pin)
    val_loader = make_dataloader("validation", seq_len=args.seq_len, batch_size=args.batch_size,
                                 cache_dir=args.data_cache, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=pin)
    train(drafter, verifier, train_loader, val_loader, args, device, args.output_dir, bf16=bf16, info=info)


if __name__ == "__main__":
    main()
