"""Fine-tune distilgpt2 as a classic (autoregressive) draft model for the frozen MLA verifier.

The classic-drafter half of eval.sh's ablation. It uses the H-Spec drafter's recipe
(train_drafter.py) so that only the architecture differs: the same loss, 0.1 CE on the data
tokens + 0.9 TV to the verifier, the same data, optimizer and schedule. A classic drafter predicts
one token at a time, so the loss is averaged over next-token positions (no block position weights).

Data:
  default                     raw WikiText-2 blocks (dataloader.py), loss at every position;
  --train-data/--val-data     verifier-generated rows (generate_verifier_data.py), loss only where
                              the drafter predicts generated tokens (logit positions >=
                              prompt_len - 1), which is what it drafts during decoding.

Validation reports the loss, TV, acceptance (1 - TV: the rejection-sampling acceptance probability
at temperature 1), top-1 agreement with the verifier and tau_greedy[k-1] = 1 + mean accepted drafts
per step with k drafts under greedy decoding, k = 1..--eval-k. It is teacher-forced on the
validation tokens, so on verifier-generated rows (the verifier's own greedy continuations) it is
what decoding those rows would give; train_drafter.py's tau_greedy is the same quantity for an
H-Spec block.

best/ (save_pretrained: config.json + model.safetensors with the tied head, plus
training_state.json) is written whenever the validation loss improves. specdec loads it directly:
  python -m specdec.bench --target mla-gpt2-medium --draft-dir runs/distil-drafter-wikitext/best ...

Example (eval.sh runs this):
  python torch_train/finetune_distilgpt2.py --verifier-checkpoint runs/mla-gpt2-medium/best.pt \
      --output-dir runs/distil-drafter-gen-s32 \
      --train-data data/verifier_gen_s32/train.pt --val-data data/verifier_gen_s32/validation.pt
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
import torch.nn.functional as F

from train_drafter import INIT_MODELS, generated_loader, load_verifier, lr_at, param_groups


def arguments(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune distilgpt2 as a draft model for the MLA verifier.")
    p.add_argument("--verifier-checkpoint", default="runs/mla-gpt2-medium/best.pt",
                   help="MLA verifier written by torch_train/train.py (frozen teacher)")
    p.add_argument("--output-dir", default="runs/distil-drafter-wikitext")
    p.add_argument("--data-cache", default="./data", help="WikiText-2 token cache (torch_train/dataloader.py)")
    p.add_argument("--init", default="distilgpt2",
                   help=f"starting checkpoint: {' | '.join(sorted(INIT_MODELS))}, a Hugging Face id, or a "
                        "local save_pretrained directory (e.g. an earlier best/)")
    p.add_argument("--dropout", type=float, default=0.0, help="GPT-2 resid/embd/attn dropout while training")
    # data
    p.add_argument("--train-data", default=None,
                   help="verifier-generated rows (generate_verifier_data.py) instead of raw WikiText-2")
    p.add_argument("--val-data", default=None, help="verifier-generated validation rows (with --train-data)")
    p.add_argument("--loss-start", type=int, default=None,
                   help="first logit position in the loss; default prompt_len - 1 with --train-data "
                        "(only generated tokens), else 0")
    p.add_argument("--seq-len", type=int, default=128, help="input tokens per sequence (generated rows: at most "
                                                            "row length - 1)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-train-examples", type=int, default=0, help="limit training sequences; 0 = all")
    p.add_argument("--num-workers", type=int, default=0)
    # optimization (as train_drafter.py: AdamW, wd 0.01, clip 1.0, linear warmup, half-cosine decay)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps; 0 = no limit")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=1.0)
    # loss
    p.add_argument("--ce-weight", type=float, default=0.1)
    p.add_argument("--tv-weight", type=float, default=0.9)
    # evaluation / logging
    p.add_argument("--eval-k", type=int, default=7, help="largest number of drafts per step in tau_greedy")
    p.add_argument("--eval-every", type=int, default=500, help="optimizer steps; 0 = only at the end")
    p.add_argument("--max-eval-batches", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast on CUDA")
    args = p.parse_args(argv)
    for name in ("seq_len", "batch_size", "epochs", "grad_accum", "eval_k"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (args.train_data is None) != (args.val_data is None):
        raise ValueError("--train-data and --val-data go together")
    if args.loss_start is not None and args.loss_start < 0:
        raise ValueError("--loss-start must be >= 0")
    return args


# --- loss and metrics ------------------------------------------------------------------------------

def distill_loss(logits, teacher_logits, targets, ce_weight=0.1, tv_weight=0.9):
    """ce_weight * CE(targets) + tv_weight * TV(verifier, drafter) per next-token position, averaged.
    logits, teacher_logits [B, L, V]; targets [B, L]. Returns (loss, stats)."""
    logq = F.log_softmax(logits.float(), dim=-1)
    ce = -logq.gather(-1, targets[..., None]).squeeze(-1)
    tv = 0.5 * (F.softmax(teacher_logits.float(), dim=-1) - logq.exp()).abs().sum(-1)
    loss = (ce_weight * ce + tv_weight * tv).mean()
    with torch.no_grad():
        stats = {"ce": ce.mean().item(), "tv": tv.mean().item(),
                 "top1_agree": (logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean().item()}
    stats["acceptance"] = 1 - stats["tv"]
    return loss, stats


def greedy_runs(agree, max_k):
    """Length of the run of consecutive agreements starting at each position, capped at max_k.

    agree [B, L] (drafter argmax == verifier argmax at each next-token position). Only starts with
    max_k positions ahead are scored, so runs are never cut short by the end of the row: returns
    [B, L - max_k + 1]. Starting at position t, greedy speculative decoding with k drafts accepts
    min(run[t], k) of them."""
    n = agree.size(1) - max_k + 1
    if n < 1:
        raise ValueError(f"need at least {max_k} positions to score runs of up to {max_k}")
    alive = torch.ones(agree.size(0), n, dtype=torch.bool, device=agree.device)
    runs = torch.zeros(agree.size(0), n, dtype=torch.long, device=agree.device)
    for j in range(max_k):
        alive = alive & agree[:, j:j + n]
        runs += alive
    return runs


def tau_from_counts(counts):
    """counts[r] = number of starts with run length r (r = 0..max_k) -> [tau(k) for k = 1..max_k],
    tau(k) = 1 + mean(min(run, k)) tokens per verifier step."""
    counts = np.asarray(counts, dtype=np.float64)
    runs, total = np.arange(len(counts)), counts.sum()
    return [float(1 + (np.minimum(runs, k) * counts).sum() / total) for k in range(1, len(counts))]


def distill_step(drafter, verifier, batch, args, autocast, loss_start):
    """Teacher logits, drafter logits and the loss on one batch [B, >= 2]. Returns (loss, stats,
    agree [B, L]) for the logit positions loss_start.. (each predicting the next token)."""
    tokens = batch[:, :args.seq_len + 1]
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    with torch.no_grad(), autocast():
        teacher = verifier(inputs)[:, loss_start:]
    with autocast():
        logits = drafter(input_ids=inputs, use_cache=False).logits[:, loss_start:]
    loss, stats = distill_loss(logits, teacher, targets[:, loss_start:], ce_weight=args.ce_weight,
                               tv_weight=args.tv_weight)
    agree = logits.detach().argmax(-1) == teacher.argmax(-1)
    return loss, stats, agree


@torch.no_grad()
def evaluate(drafter, verifier, loader, args, device, autocast):
    """Validation metrics, weighted by scored positions."""
    was_training = drafter.training
    drafter.eval()
    sums, positions, batches = defaultdict(float), 0, 0
    counts = np.zeros(args.eval_k + 1, dtype=np.int64)
    try:
        for i, batch in enumerate(loader):
            if args.max_eval_batches and i >= args.max_eval_batches:
                break
            loss, stats, agree = distill_step(drafter, verifier, batch.to(device), args, autocast, args.loss_start)
            n = agree.numel()
            sums["loss"] += loss.item() * n
            for key in ("ce", "tv", "top1_agree"):
                sums[key] += stats[key] * n
            counts += torch.bincount(greedy_runs(agree, args.eval_k).flatten(), minlength=args.eval_k + 1).cpu().numpy()
            positions += n
            batches += 1
    finally:
        drafter.train(was_training)
    if not positions:
        raise ValueError("empty validation loader")
    result = {key: value / positions for key, value in sums.items()}
    result["acceptance"] = 1 - result["tv"]
    result["tau_greedy"] = tau_from_counts(counts)
    result["batches"] = batches
    return result


# --- model, data, checkpoints --------------------------------------------------------------------

def load_drafter(init, dropout=0.0):
    """GPT2LMHeadModel from `init` (INIT_MODELS key, Hugging Face id or local directory)."""
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained(INIT_MODELS.get(init, init), resid_pdrop=dropout, embd_pdrop=dropout,
                                           attn_pdrop=dropout)


def check_compatible(drafter, verifier, input_len):
    cfg = drafter.config
    if cfg.vocab_size != verifier.cfg.n_vocab:
        raise ValueError(f"drafter vocabulary {cfg.vocab_size} != verifier vocabulary {verifier.cfg.n_vocab}")
    if input_len > min(cfg.n_positions, verifier.cfg.n_positions):
        raise ValueError(f"{input_len} input tokens exceed the drafter's {cfg.n_positions} / verifier's "
                         f"{verifier.cfg.n_positions} positions")


def build_loaders(args, device):
    """(train, validation, input_len). Resolves args.loss_start and records the data in args."""
    if args.train_data:
        from generate_verifier_data import load_generated
        train_ids, prompt_len = load_generated(args.train_data)
        val_ids, val_prompt_len = load_generated(args.val_data)
        if val_prompt_len != prompt_len or val_ids.size(1) != train_ids.size(1):
            raise ValueError("train and validation data have different prompt/row lengths")
        if args.max_train_examples:
            train_ids = train_ids[:args.max_train_examples]
        if args.loss_start is None:
            args.loss_start = prompt_len - 1  # the logit that predicts the first generated token
        input_len = min(args.seq_len, train_ids.size(1) - 1)
        args.data = {"train_rows": train_ids.size(0), "val_rows": val_ids.size(0), "prompt_len": prompt_len,
                     "row_len": train_ids.size(1)}
        print(f"Generated data: {train_ids.size(0)} train / {val_ids.size(0)} validation rows of "
              f"{train_ids.size(1)} tokens (prompt {prompt_len}); loss from logit position {args.loss_start}",
              flush=True)
        loaders = (generated_loader(train_ids, args.batch_size, shuffle=True, seed=args.seed),
                   generated_loader(val_ids, args.batch_size, shuffle=False))
    else:
        from dataloader import make_dataloader
        if args.loss_start is None:
            args.loss_start = 0
        input_len = args.seq_len
        pin = device.type == "cuda"
        loaders = (make_dataloader("train", seq_len=args.seq_len, batch_size=args.batch_size,
                                   cache_dir=args.data_cache, max_examples=args.max_train_examples,
                                   num_workers=args.num_workers, pin_memory=pin),
                   make_dataloader("validation", seq_len=args.seq_len, batch_size=args.batch_size,
                                   cache_dir=args.data_cache, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=pin))
    if input_len - args.loss_start < args.eval_k:
        raise ValueError(f"--loss-start {args.loss_start} leaves {input_len - args.loss_start} scored positions "
                         f"of {input_len}; tau_greedy needs at least --eval-k {args.eval_k}")
    return (*loaders, input_len)


def save_if_best(stats, best, path, drafter, step, source):
    """save_pretrained to `path` if stats['loss'] beats `best` (never on NaN); returns the new best."""
    if not stats["loss"] < best:
        return best
    path = Path(path)
    drafter.save_pretrained(path, safe_serialization=True)
    with open(path / "training_state.json", "w") as f:
        json.dump({"step": step, "source": source, "validation": stats}, f, indent=2)
    print(f"New best validation loss: {stats['loss']:.4f}", flush=True)
    return stats["loss"]


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
        n = len(history["val"][-1]["tau_greedy"])
        for k in sorted({min(3, n), min(5, n), n}):
            ax2.plot(steps, [h["tau_greedy"][k - 1] for h in history["val"]], "o-", label=f"tau_greedy, k={k}")
    ax1.set(xlabel="optimizer step", ylabel="loss (0.1 CE + 0.9 TV)", title="distilgpt2 drafter loss")
    ax2.set(xlabel="optimizer step", ylabel="tokens per verifier step", title="validation acceptance (greedy)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "loss_curve.png", dpi=150)
    plt.close(fig)


# --- training ------------------------------------------------------------------------------------

def train(drafter, verifier, train_loader, val_loader, args, device, output_dir, bf16=False):
    """Optimize `drafter` against the frozen `verifier`. Returns (history, best validation loss)."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "arguments.json", "w") as f:
        json.dump({**vars(args), "bf16": bf16}, f, indent=2)

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
    source = {"verifier": str(args.verifier_checkpoint), "init": args.init}
    history, best, step = {"train": [], "val": []}, float("inf"), 0
    print(f"{total} optimizer steps ({len(train_loader)} batches/epoch, {args.epochs} epochs), "
          f"warmup {warmup}, bf16={bf16}", flush=True)

    def run_eval():
        nonlocal best
        stats = evaluate(drafter, verifier, val_loader, args, device, autocast)
        history["val"].append({"step": step, **stats})
        print(f"validation step={step}: loss {stats['loss']:.4f} tv {stats['tv']:.4f} "
              f"top1 {stats['top1_agree']:.3f} tau_greedy {[round(t, 3) for t in stats['tau_greedy']]}", flush=True)
        best = save_if_best(stats, best, output / "best", drafter, step, source)
        with open(output / "loss_history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_history(history, output)

    t0, done = time.perf_counter(), False
    for epoch in range(args.epochs):
        accum, running = 0, defaultdict(float)
        for i, batch in enumerate(train_loader):
            loss, stats, _ = distill_step(drafter, verifier, batch.to(device, non_blocking=True), args, autocast,
                                          args.loss_start)
            loss.backward()
            accum += 1
            running["loss"] += loss.item()
            for key in ("tv", "top1_agree"):
                running[key] += stats[key]
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
                      f"top1 {record['top1_agree']:.3f} lr {lr:.2e} {elapsed / step:.2f}s/step{mem}", flush=True)
            accum, running = 0, defaultdict(float)
            if args.eval_every and step % args.eval_every == 0:
                run_eval()
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break
    if not history["val"] or history["val"][-1]["step"] != step:
        run_eval()  # final evaluation (also saves best/ if it is the best)
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
    train_loader, val_loader, input_len = build_loaders(args, device)
    drafter = load_drafter(args.init, args.dropout)
    check_compatible(drafter, verifier, input_len)
    n_params = sum(p.numel() for p in drafter.parameters())
    print(f"Drafter {args.init}: {drafter.config.n_layer} layers, {n_params / 1e6:.1f}M parameters", flush=True)
    return train(drafter, verifier, train_loader, val_loader, args, device, args.output_dir, bf16=bf16)


if __name__ == "__main__":
    main()
