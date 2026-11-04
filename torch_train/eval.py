"""Evaluate a trained MLA GPT-2 checkpoint on WikiText-2 and generate text."""
import argparse
import math
import torch
import torch.nn.functional as F

from decoder_mla import DecoderLM, MLAConfig
from dataloader import make_dataloader


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default="./runs/mla-gpt2-medium/best.pt")
    p.add_argument('--data-cache', default='./data')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-eval-batches', type=int, default=0, help='0 = entire split')
    p.add_argument('--prompt', default='The capital of France is')
    p.add_argument('--max-new-tokens', type=int, default=40)
    p.add_argument('--no-bf16', action='store_true')
    return p.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device, bf16=False, max_batches=0):
    model.eval()
    loss_sum, count = 0., 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = batch.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='sum')
        loss_sum += loss.item()
        count += targets.numel()
    mean = loss_sum / max(count, 1)
    return {'loss': mean, 'ppl': math.exp(min(mean, 80)), 'evaluated_tokens': count}


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens, device, bf16=False):
    model.eval()
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    if ids.size(1) > model.cfg.n_positions:
        ids = ids[:, -model.cfg.n_positions:]
    limit = min(max_new_tokens, model.cfg.n_positions - ids.size(1))
    if limit <= 0:
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    history = ids
    cache = None
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        logits, cache = model(history, use_cache=True)
        for i in range(limit):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            history = torch.cat([history, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            if i + 1 < limit:
                logits, cache = model(next_token, past_key_values=cache,
                                      use_cache=True)
    return tokenizer.decode(history[0].tolist(), skip_special_tokens=True)


def main():
    args = arguments()
    from transformers import AutoTokenizer
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = MLAConfig(**ckpt['config'])
    model = DecoderLM(cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bf16 = device.type == 'cuda' and not args.no_bf16 and torch.cuda.is_bf16_supported()
    model.to(device).eval()
    if args.seq_len > cfg.n_positions:
        raise ValueError('seq_len exceeds model context limit')
    loader = make_dataloader(args.split, seq_len=args.seq_len,
                             batch_size=args.batch_size, shuffle=False,
                             cache_dir=args.data_cache,
                             pin_memory=(device.type=='cuda'))
    stats = evaluate(model, loader, device, bf16, args.max_eval_batches)
    print(f'{args.split} metrics:', stats)
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-medium')
    print('Generated text:\n', generate(model, tokenizer, args.prompt,
                                         args.max_new_tokens, device, bf16))


if __name__ == '__main__':
    main()
