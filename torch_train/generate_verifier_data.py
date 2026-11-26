"""Generate the MLA verifier's own continuations of WikiText prompts: training data for the drafter.

Every --prompt-len-token window of a WikiText split (stride --prompt-stride) is continued by the
frozen verifier for --gen-len tokens: greedy by default (the decoding mode the benchmark uses), or
sampled with --temperature > 0 (--samples-per-prompt continuations each). Rows are
prompt + continuation, saved per split as <out-dir>/<split>.pt:
  {"ids": LongTensor [N, prompt_len + gen_len], "prompt_len", "gen_len", "temperature", "source", ...}
64 + 64 tokens matches specdec.bench (64-token prompts, 64 new tokens).

Why: the drafter runs on the verifier's output. Trained on human text it learns contexts and
targets that decoding never produces (on the benchmark, real tau 1.67 vs 1.78 estimated in
training, and position-1 acceptance 0.45 vs 0.51). train_drafter.py --train-data/--val-data puts
every block inside the generated part, so its prefix, anchor and targets are verifier text.

Usage (train.sh runs this; existing outputs are skipped unless --overwrite):
  python torch_train/generate_verifier_data.py --verifier-checkpoint runs/mla-gpt2-medium/best.pt \
      --out-dir data/verifier_gen
"""
import argparse
import time
from pathlib import Path

import torch

SOURCES = {"wikitext-2": "wikitext-2-raw-v1", "wikitext-103": "wikitext-103-raw-v1"}


def arguments(argv=None):
    p = argparse.ArgumentParser(description="Generate verifier continuations of WikiText prompts.")
    p.add_argument("--verifier-checkpoint", default="runs/mla-gpt2-medium/best.pt")
    p.add_argument("--data-cache", default="./data", help="WikiText token cache (torch_train/dataloader.py)")
    p.add_argument("--source", choices=sorted(SOURCES), default="wikitext-2", help="where the prompts come from")
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--out-dir", default="data/verifier_gen")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--prompt-stride", type=int, default=0, help="window stride; 0 = prompt-len (no overlap)")
    p.add_argument("--max-prompts", type=int, default=0, help="per split, random subset; 0 = all")
    p.add_argument("--max-val-prompts", type=int, default=2000, help="cap for the validation split; 0 = no cap")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--samples-per-prompt", type=int, default=1, help="continuations per prompt (temperature > 0)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (faster; greedy picks may differ at near-ties)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    for name in ("prompt_len", "gen_len", "batch_size", "samples_per_prompt"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.samples_per_prompt > 1 and args.temperature == 0:
        raise ValueError("--samples-per-prompt > 1 needs --temperature > 0 (greedy continuations are identical)")
    return args


def prompts_from_tokens(ids, prompt_len, stride=0, max_prompts=0, seed=0):
    """Windows ids[s : s + prompt_len] for s = 0, stride, 2*stride, ...; a seeded random subset of
    max_prompts of them (in order) if max_prompts > 0. Returns LongTensor [P, prompt_len]."""
    stride = stride or prompt_len
    if ids.numel() < prompt_len:
        raise ValueError(f"source has {ids.numel()} tokens, fewer than prompt_len={prompt_len}")
    starts = torch.arange(0, ids.numel() - prompt_len + 1, stride)
    if max_prompts and max_prompts < len(starts):
        keep = torch.randperm(len(starts), generator=torch.Generator().manual_seed(seed))[:max_prompts]
        starts = starts[keep.sort().values]
    return torch.stack([ids[s:s + prompt_len] for s in starts.tolist()])


@torch.no_grad()
def generate(verifier, prompts, gen_len, batch_size=128, temperature=0.0, seed=0, device=None, bf16=False, log_every=0):
    """prompts [P, L] -> [P, L + gen_len]: each row continued by `verifier` (decoder_mla.DecoderLM)
    with its (c_kv, k_rope) cache; greedy, or sampled at `temperature` (seeded)."""
    device = torch.device(device) if device is not None else next(verifier.parameters()).device
    if prompts.size(1) + gen_len > verifier.cfg.n_positions:
        raise ValueError(f"prompt + gen_len exceeds the verifier's {verifier.cfg.n_positions} positions")
    gen = torch.Generator(device=device).manual_seed(seed)
    verifier.eval()
    rows, t0 = [], time.perf_counter()
    for i, start in enumerate(range(0, prompts.size(0), batch_size)):
        x = prompts[start:start + batch_size].to(device)
        out = [x]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            logits, cache = verifier(x, use_cache=True)
            for step in range(gen_len):
                last = logits[:, -1].float()
                if temperature > 0:
                    nxt = torch.multinomial(torch.softmax(last / temperature, dim=-1), 1, generator=gen)
                else:
                    nxt = last.argmax(dim=-1, keepdim=True)
                out.append(nxt)
                if step + 1 < gen_len:
                    logits, cache = verifier(nxt, past_key_values=cache, use_cache=True)
        rows.append(torch.cat(out, dim=1).cpu())
        if log_every and (i + 1) % log_every == 0:
            done = start + x.size(0)
            print(f"  {done}/{prompts.size(0)} rows, {time.perf_counter() - t0:.0f}s", flush=True)
    return torch.cat(rows)


def load_generated(path):
    """(ids LongTensor [N, T], prompt_len) from a file written by this script."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not {"ids", "prompt_len"} <= set(data):
        raise ValueError(f"{path} was not written by generate_verifier_data.py")
    return data["ids"], int(data["prompt_len"])


def main(argv=None):
    args = arguments(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (use --device cpu for a slow CPU run)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in args.splits if args.overwrite or not (out_dir / f"{s}.pt").exists()]
    for s in set(args.splits) - set(todo):
        print(f"{out_dir / f'{s}.pt'} exists; skipping (--overwrite to regenerate)", flush=True)
    if not todo:
        return

    from dataloader import get_tokens
    from train_drafter import load_verifier
    verifier = load_verifier(args.verifier_checkpoint, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False  # keep greedy picks close to the fp32 MiniTorch verifier
    for split in todo:
        cap = args.max_prompts
        if split == "validation" and args.max_val_prompts:
            cap = min(cap, args.max_val_prompts) if cap else args.max_val_prompts
        tokens = get_tokens(split, args.data_cache, config=SOURCES[args.source])
        prompts = prompts_from_tokens(tokens, args.prompt_len, args.prompt_stride, cap, args.seed)
        prompts = prompts.repeat_interleave(args.samples_per_prompt, dim=0)
        print(f"{split}: {prompts.size(0)} prompts x {args.gen_len} tokens "
              f"({'greedy' if args.temperature == 0 else f'temperature {args.temperature}'})", flush=True)
        t0 = time.perf_counter()
        ids = generate(verifier, prompts, args.gen_len, args.batch_size, args.temperature, args.seed, device,
                       args.bf16, log_every=20)
        path = out_dir / f"{split}.pt"
        torch.save({"ids": ids, "prompt_len": args.prompt_len, "gen_len": args.gen_len,
                    "temperature": args.temperature, "samples_per_prompt": args.samples_per_prompt,
                    "source": args.source, "split": split, "verifier": str(args.verifier_checkpoint)}, path)
        print(f"wrote {path}: {tuple(ids.shape)} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
