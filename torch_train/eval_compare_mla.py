"""Evaluate five GPT-2 Medium variants on IDENTICAL WikiText-2 token blocks.

Models:
  1. Original Hugging Face GPT-2 Medium
  2. Fine-tuned Hugging Face GPT-2 Medium (from --gpt2-checkpoint)
  3. Fine-tuned PyTorch MLA (from --mla-checkpoint)
  4. MiniTorch MLA with the EXACT same MLA checkpoint parameters
  5. Fine-tuned GPT-2 Medium using MiniTorch standard attention (same GPT-2 checkpoint)

Place in <repository>/torch_train/eval_compare_mla.py and invoke from repo root:
  python torch_train/eval_compare_mla.py \
      --checkpoint runs/mla-gpt2-medium/best.pt \
      --data-cache ./data --seq-len 128 --batch-size 1 --max-batches 1

WARNING: MiniTorch's current Linear / Embedding use host-backed, very inefficient
matrix multiplies and dense one-hot embeddings. GPT-2 Medium evaluation may be
extremely slow and memory intensive. Start with --max-batches 1 and small --seq-len.
The dataset is partitioned into blocks based on --seq-len, so compare results
only between runs with IDENTICAL values of that argument.
"""

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mla-checkpoint', '--checkpoint', dest='mla_checkpoint',
                   help='Fine-tuned MLA best.pt')
    p.add_argument('--gpt2-checkpoint',
                   help='Fine-tuned original GPT-2 Medium best.pt')
    p.add_argument('--pretrained', default='openai-community/gpt2-medium')
    p.add_argument('--data-cache', default='./data')
    p.add_argument('--split', default='validation', choices=['validation', 'test', 'train'])
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-batches', type=int, default=1, help='0 = entire split (very slow in MiniTorch)')
    p.add_argument('--attention', choices=['fused', 'unfused'], default='fused',
                   help='MiniTorch MLA attention path')
    p.add_argument('--gpt2-attention', choices=['fused', 'unfused'], default='unfused',
                   help='MiniTorch GPT-2 attention path; fused also enables fused LayerNorm')
    p.add_argument('--fused-layernorm', action='store_true',
                   help='MiniTorch fused LN (only enable after epsilon is verified)')
    p.add_argument('--bf16', action='store_true',
                   help='Enable CUDA BF16 autocast for PyTorch models; default FP32 for parity')
    p.add_argument('--output', default='runs/gpt2-mla-five-way-eval.json')
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--plot-only', default=None, metavar='RESULTS_JSON',
                   help='Plot an existing evaluation JSON without re-running models')
    p.add_argument('--skip-minitorch', action='store_true',
                   help='Skip both MiniTorch evaluations')
    p.add_argument('--skip-minitorch-gpt2', action='store_true',
                   help='Skip the MiniTorch standard GPT-2 evaluation')
    p.add_argument('--verify-weights', action='store_true', help='Check every transferred parameter')
    return p.parse_args()


def minitorch_to_torch_key(name):
    """Return (PyTorch state_dict key, transpose_required) for a MiniTorch parameter."""
    m = re.match(r'^t_layer_(\d+)\.(.+)$', name)
    if m:
        name = f'layers.{int(m.group(1)) - 1}.{m.group(2)}'

    if name.endswith('.weights'):
        stem = name[:-len('.weights')]
        # 1D RMSNorm, ordinary embedding weights are NOT transposed.
        if stem in ('token_embeddings', 'position_embeddings') or stem.endswith(('.q_norm', '.kv_norm')):
            return stem + '.weight', False
        # All other 2D MiniTorch weights represent Linear layers [in,out].
        return stem + '.weight', True
    return name, False


def load_checkpoint_to_minitorch(mt_model, state, backend, *, verify=False):
    """Transfer ALL MiniTorch parameters from PyTorch checkpoint, FP32, no stale random weights."""
    names = mt_model.named_parameters()
    copied, missing, incompatible = [], [], []

    for name, parameter in names:
        source_key, transpose = minitorch_to_torch_key(name)
        # PyTorch GPT-style output head shares weight with token embedding.
        if name == 'lm_head.weights' and source_key not in state:
            source_key = 'token_embeddings.weight'
        if source_key not in state:
            missing.append(f'{name} <- {source_key}')
            continue
        src = state[source_key]
        if transpose:
            src = src.T
        if tuple(src.shape) != tuple(parameter.value.shape):
            incompatible.append(f'{name}: MiniTorch {tuple(parameter.value.shape)} != {source_key} {tuple(src.shape)}')
            continue
        copied.append((name, parameter, source_key, src))

    if missing or incompatible:
        raise ValueError('Checkpoint conversion incomplete:\n' + '\n'.join(missing + incompatible))

    for name, parameter, key, src in copied:
        # Copy into existing Parameter storage; avoids generating ~355M new autograd leaf nodes.
        array = src.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()
        dest = parameter.value._tensor._storage
        np.copyto(dest, array.reshape(-1))
        if verify:
            if not np.array_equal(dest, array.reshape(-1)):
                raise AssertionError(f'Weight copy failed for {name} (source {key})')
    print(f'Loaded {len(copied)} MiniTorch tensors from PyTorch checkpoint', flush=True)


def safe_nll_from_numpy(logits, labels):
    """Sum of teacher-forced next-token negative log-likelihoods in FP64."""
    if logits.shape[:2] != labels.shape:
        raise ValueError(f'Logits/labels mismatch: {logits.shape} vs {labels.shape}')
    logits = logits.astype(np.float64, copy=False)
    max_l = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_l
    log_den = np.log(np.exp(shifted).sum(axis=-1)) + max_l[..., 0]
    chosen = np.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    nll = float(np.sum(log_den - chosen))
    if not np.isfinite(nll):
        raise FloatingPointError('Non-finite MiniTorch NLL')
    return nll


def measure_torch(model, batches, *, device, is_hf, bf16):
    model = model.to(device).eval()
    total_nll = 0.0
    total_tokens = 0
    elapsed = 0.0
    with torch.inference_mode():
        for batch in batches:
            inputs = batch[:, :-1].to(device)
            labels = batch[:, 1:].to(device)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(bf16 and device.type == 'cuda')):
                output = model(inputs, use_cache=False) if is_hf else model(inputs)
                logits = output.logits if is_hf else output
                nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                      labels.reshape(-1), reduction='sum')
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed += time.perf_counter() - start
            total_nll += float(nll.item())
            total_tokens += labels.numel()
    return metrics(total_nll, total_tokens, elapsed)


def measure_minitorch(model, batches):
    import minitorch
    from minitorch.tensor_functions import tensor_from_numpy
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    elapsed = 0.0
    for batch in batches:
        ids = np.ascontiguousarray(batch[:, :-1].numpy(), dtype=np.float32)
        labels = np.ascontiguousarray(batch[:, 1:].numpy(), dtype=np.int64)
        x = tensor_from_numpy(ids, backend=model.backend)
        start = time.perf_counter()
        logits = model(x).to_numpy()
        elapsed += time.perf_counter() - start
        total_nll += safe_nll_from_numpy(logits, labels)
        total_tokens += labels.size
    return metrics(total_nll, total_tokens, elapsed)


def metrics(nll, tokens, seconds):
    loss = nll / tokens
    return {'loss': loss, 'perplexity': math.exp(min(80.0, loss)),
            'total_nll': nll, 'tokens': tokens, 'elapsed_seconds': seconds}



def load_finetuned_hf_gpt2(checkpoint_path, pretrained):
    """Load the model-only checkpoint saved by train_ori.py.

    Accepts {'model': state_dict, 'config': HF config dict}, or a raw
    state_dict. Strict loading prevents accidental partial comparisons.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
        config_data = ckpt.get('config')
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        config_data = ckpt.get('config')
    else:
        state = ckpt
        config_data = None

    if not isinstance(state, dict):
        raise ValueError('Original GPT-2 checkpoint must contain a model state_dict')

    if config_data is None:
        model = GPT2LMHeadModel.from_pretrained(pretrained)
    else:
        config = (GPT2Config.from_dict(config_data)
                  if isinstance(config_data, dict) else config_data)
        model = GPT2LMHeadModel(config)

    model.load_state_dict(state, strict=True)
    del ckpt, state
    return model.eval()


def load_gpt2_state_to_minitorch(mt_model, hf_state, *, verify=False):
    """Map fine-tuned HF GPT-2 weights into MiniTorch's standard-attention DecoderLM.

    GPT-2's Conv1D weights already use [in,out]; unlike MLA weights they
    should NOT be transposed. Copies into existing MiniTorch parameter storage.
    """
    assignments = []

    def add(param, key, value):
        if param is None:
            raise ValueError(f'Missing MiniTorch parameter for {key}')
        expected = tuple(param.value.shape)
        if expected != tuple(value.shape):
            raise ValueError(f'{key}: MiniTorch shape {expected}, checkpoint shape {tuple(value.shape)}')
        assignments.append((param, key, value))

    add(mt_model.token_embeddings.weights, 'transformer.wte.weight',
        hf_state['transformer.wte.weight'])
    add(mt_model.position_embeddings.weights, 'transformer.wpe.weight',
        hf_state['transformer.wpe.weight'])
    for i, layer in enumerate(mt_model.layers):
        prefix = f'transformer.h.{i}'
        qkv_w = hf_state[f'{prefix}.attn.c_attn.weight'].chunk(3, dim=1)
        qkv_b = hf_state[f'{prefix}.attn.c_attn.bias'].chunk(3, dim=0)
        for name, w, b in zip(('q_projection', 'k_projection', 'v_projection'),
                              qkv_w, qkv_b):
            proj = getattr(layer.attention, name)
            add(proj.weights, f'{prefix}.attn.{name}.weight', w)
            add(proj.bias, f'{prefix}.attn.{name}.bias', b)

        proj = layer.attention.out_projection
        add(proj.weights, f'{prefix}.attn.c_proj.weight', hf_state[f'{prefix}.attn.c_proj.weight'])
        add(proj.bias, f'{prefix}.attn.c_proj.bias', hf_state[f'{prefix}.attn.c_proj.bias'])
        for norm_name in ('ln_1', 'ln_2'):
            norm = getattr(layer, norm_name)
            add(norm.weights, f'{prefix}.{norm_name}.weight', hf_state[f'{prefix}.{norm_name}.weight'])
            add(norm.bias, f'{prefix}.{norm_name}.bias', hf_state[f'{prefix}.{norm_name}.bias'])
        for mlp_name, hf_name in [('linear_in', 'c_fc'), ('linear_out', 'c_proj')]:
            proj = getattr(layer.ff, mlp_name)
            add(proj.weights, f'{prefix}.mlp.{hf_name}.weight',
                hf_state[f'{prefix}.mlp.{hf_name}.weight'])
            add(proj.bias, f'{prefix}.mlp.{hf_name}.bias',
                hf_state[f'{prefix}.mlp.{hf_name}.bias'])

    add(mt_model.ln.weights, 'transformer.ln_f.weight', hf_state['transformer.ln_f.weight'])
    add(mt_model.ln.bias, 'transformer.ln_f.bias', hf_state['transformer.ln_f.bias'])
    add(mt_model.lm_head.weights, 'transformer.wte.weight.T',
        hf_state['transformer.wte.weight'].T)

    for param, key, value in assignments:
        array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        dest = param.value._tensor._storage
        np.copyto(dest, array.reshape(-1))
        if verify and not np.array_equal(dest, array.reshape(-1)):
            raise AssertionError(f'Failed transfer: {key}')
    print(f'Loaded {len(assignments)} standard GPT-2 parameter tensors into MiniTorch', flush=True)




def plot_comparison(results, output_json):
    """Save bar plots using measured scores only (never dummy values)."""
    entries = [
        ('GPT-2\nPretrained', 'gpt2_medium'),
        ('GPT-2 FT\nPyTorch', 'gpt2_medium_finetuned'),
        ('GPT-2 FT\nMiniTorch', 'minitorch_gpt2_finetuned'),
        ('MLA FT\nPyTorch', 'torch_mla_finetuned'),
        ('MLA FT\nMiniTorch', 'minitorch_mla_finetuned'),
    ]
    included = [(label, results[key]) for label, key in entries
                if key in results and results[key] is not None]
    if not included:
        raise ValueError('No model results found in evaluation JSON')

    labels = [item[0] for item in included]
    output_json = Path(output_json)
    for metric, ylabel, name in [
        ('loss', 'Cross-entropy (nats/token)', 'loss'),
        ('perplexity', 'Perplexity (lower is better)', 'perplexity'),
        ('elapsed_seconds', 'Elapsed inference time (seconds)', 'time'),
    ]:
        values = [float(item[1][metric]) for item in included]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2.0), 5.5))
        bars = ax.bar(labels, values, width=0.68,
                      color=['#73859b' if 'Pretrained' in label else
                             '#537ca6' if 'GPT-2 FT' in label else '#4c9c87'
                             for label in labels])
        ax.set_ylabel(ylabel)
        ax.set_title('WikiText-2 validation: GPT-2 vs MLA — ' + name)
        ax.grid(axis='y', alpha=0.22)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.4f}' if metric != 'elapsed_seconds' else f'{value:.2f}s',
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
        if metric == 'elapsed_seconds':
            ax.text(0.5, -0.22, 'Inference timing includes implementation overhead; not an optimized throughput benchmark.',
                    ha='center', va='top', transform=ax.transAxes, fontsize=9)
            fig.subplots_adjust(bottom=0.28)
        else:
            fig.tight_layout()
        destination = output_json.with_name(output_json.stem + f'_{name}.png')
        fig.savefig(destination, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print('Saved bar plot:', destination, flush=True)


def main():
    args = parse_args()
    if args.plot_only:
        input_json = Path(args.plot_only)
        plot_comparison(json.loads(input_json.read_text()), input_json)
        return
    if not args.mla_checkpoint or not args.gpt2_checkpoint:
        raise ValueError('--mla-checkpoint and --gpt2-checkpoint are required unless --plot-only is used')
    from dataloader import make_dataloader
    from decoder_mla import DecoderLM as TorchDecoderLM, MLAConfig as TorchMLAConfig
    if args.seq_len < 2 or args.batch_size < 1 or args.max_batches < 0:
        raise ValueError('Bad sequence length, batch size, or max-batches')
    checkpoint_path = Path(args.mla_checkpoint)
    gpt2_checkpoint_path = Path(args.gpt2_checkpoint)
    for path in (checkpoint_path, gpt2_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; PyTorch BF16: {args.bf16}; MiniTorch FP32', flush=True)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = TorchMLAConfig(**checkpoint['config'])
    state = checkpoint['model']
    if args.seq_len > cfg.n_positions:
        raise ValueError(f'seq-len exceeds MLA context size {cfg.n_positions}')

    # Freeze one EXACT subset of batches for all five models, in the same order.
    loader = make_dataloader(args.split, seq_len=args.seq_len,
                             batch_size=args.batch_size, cache_dir=args.data_cache,
                             shuffle=False, num_workers=0, pin_memory=False)
    limit = len(loader) if args.max_batches == 0 else min(args.max_batches, len(loader))
    batches = []
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        batches.append(batch)
    if not batches:
        raise RuntimeError('No validation batches available')
    print(f'Evaluating {len(batches)} identical batches / '
          f'{sum(b[:, 1:].numel() for b in batches):,} tokens', flush=True)

    from transformers import GPT2LMHeadModel
    print('1/5 Loading and evaluating original GPT-2 Medium...', flush=True)
    hf = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
    if args.seq_len > hf.config.n_positions:
        raise ValueError('seq-len exceeds original GPT-2 context')
    base = measure_torch(hf, batches, device=device, is_hf=True, bf16=args.bf16)
    print(f'  GPT-2: loss={base["loss"]:.6f}, ppl={base["perplexity"]:.4f}', flush=True)
    del hf
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print('2/5 Loading and evaluating fine-tuned original GPT-2 Medium...', flush=True)
    tuned_hf = load_finetuned_hf_gpt2(gpt2_checkpoint_path, args.pretrained)
    if args.seq_len > tuned_hf.config.n_positions:
        raise ValueError('seq-len exceeds fine-tuned GPT-2 context')
    tuned = measure_torch(tuned_hf, batches, device=device, is_hf=True, bf16=args.bf16)
    print(f'  Fine-tuned GPT-2: loss={tuned["loss"]:.6f}, ppl={tuned["perplexity"]:.4f}', flush=True)
    # Re-read this checkpoint later for MiniTorch GPT-2, avoiding another HF model.
    del tuned_hf
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print('3/5 Loading and evaluating fine-tuned PyTorch MLA...', flush=True)
    tm = TorchDecoderLM(cfg)
    tm.load_state_dict(state, strict=True)
    tm.eval()
    pyt = measure_torch(tm, batches, device=device, is_hf=False, bf16=args.bf16)
    print(f'  Torch MLA: loss={pyt["loss"]:.6f}, ppl={pyt["perplexity"]:.4f}', flush=True)
    del tm
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    results = {
        'mla_checkpoint': str(checkpoint_path),
        'gpt2_checkpoint': str(gpt2_checkpoint_path),
        'split': args.split,
        'seq_len': args.seq_len, 'batch_size': args.batch_size,
        'max_batches': args.max_batches, 'evaluated_batches': len(batches),
        'torch_bf16': args.bf16,
        'gpt2_medium': base,
        'gpt2_medium_finetuned': tuned,
        'torch_mla_finetuned': pyt,
    }

    if not args.skip_minitorch:
        print('4/5 Constructing and transferring weights into MiniTorch MLA...', flush=True)
        import minitorch
        from minitorch.cuda_kernel_ops import CudaKernelOps
        from minitorch.modules_mla import MLAConfig as MiniConfig, DecoderLM as MiniDecoderLM
        backend = minitorch.TensorBackend(CudaKernelOps)
        mini_cfg = MiniConfig(**checkpoint['config'])
        mt = MiniDecoderLM(mini_cfg, backend=backend,
                           use_fused_kernel=(args.attention == 'fused'),
                           use_fused_layernorm=args.fused_layernorm)
        load_checkpoint_to_minitorch(mt, state, backend, verify=args.verify_weights)
        del checkpoint, state
        gc.collect()
        mini = measure_minitorch(mt, batches)
        print(f'  MiniTorch MLA: loss={mini["loss"]:.6f}, ppl={mini["perplexity"]:.4f}', flush=True)
        print(f'  |MiniTorch - Torch MLA| loss={abs(mini["loss"]-pyt["loss"]):.6f}', flush=True)
        results['minitorch_mla_finetuned'] = mini
        results['minitorch_minus_torch_mla_loss'] = mini['loss'] - pyt['loss']
        results['minitorch_attention'] = args.attention
        results['minitorch_fused_layernorm'] = args.fused_layernorm

    if not args.skip_minitorch and not args.skip_minitorch_gpt2:
        print('5/5 Evaluating fine-tuned GPT-2 Medium in MiniTorch...', flush=True)
        import minitorch
        from minitorch.cuda_kernel_ops import CudaKernelOps
        from minitorch.modules_transfomer import DecoderLM as MiniGPT2DecoderLM
        from transformers import GPT2Config
        backend = minitorch.TensorBackend(CudaKernelOps)
        gpt2_ckpt = torch.load(gpt2_checkpoint_path, map_location='cpu', weights_only=True)
        gpt2_config = GPT2Config.from_dict(gpt2_ckpt['config'])
        if args.seq_len > gpt2_config.n_positions:
            raise ValueError('seq-len exceeds fine-tuned GPT-2 context')
        mgpt2 = MiniGPT2DecoderLM(
            n_vocab=gpt2_config.vocab_size,
            n_embd=gpt2_config.n_embd,
            n_head=gpt2_config.n_head,
            n_positions=gpt2_config.n_positions,
            n_layer=gpt2_config.n_layer,
            p_dropout=0.0,
            ln_eps=gpt2_config.layer_norm_epsilon,
            bias=True,
            backend=backend,
            use_fused_kernel=(args.gpt2_attention == 'fused'),
        )
        load_gpt2_state_to_minitorch(mgpt2, gpt2_ckpt['model'], verify=args.verify_weights)
        del gpt2_ckpt
        gc.collect()
        mtgpt2 = measure_minitorch(mgpt2, batches)
        results['minitorch_gpt2_finetuned'] = mtgpt2
        results['minitorch_minus_torch_gpt2_loss'] = mtgpt2['loss'] - tuned['loss']
        results['minitorch_gpt2_attention'] = args.gpt2_attention
        print(f'  MiniTorch GPT-2: loss={mtgpt2["loss"]:.6f}, ppl={mtgpt2["perplexity"]:.4f}', flush=True)
        print(f'  |MiniTorch - PyTorch GPT-2| loss={abs(mtgpt2["loss"]-tuned["loss"]):.6f}', flush=True)
        del mgpt2
        gc.collect()

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + '\n')
    print('\nSummary (same exact WikiText-2 blocks):')
    for label, value in [('Pretrained GPT-2 Medium', base),
                         ('Fine-tuned GPT-2 Medium', tuned),
                         ('Fine-tuned MLA (PyTorch)', pyt),
                         ('Fine-tuned MLA (MiniTorch)', results.get('minitorch_mla_finetuned')),
                         ('Fine-tuned GPT-2 (MiniTorch)', results.get('minitorch_gpt2_finetuned'))]:
        if value is not None:
            print(f'  {label:27s} loss={value["loss"]:.6f}  PPL={value["perplexity"]:.4f}')
    print(f'Wrote {target}', flush=True)
    plot_comparison(results, target)


if __name__ == '__main__':
    main()
