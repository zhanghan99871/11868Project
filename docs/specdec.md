# Speculative decoding in MiniTorch (GPT-2 small / distilgpt2 → GPT-2 medium)

Preliminary experiments for the 11-868 project. MLA is not implemented yet; this
directory contains the MiniTorch inference path and speculative decoding that the
MLA work will plug into.

The project depends on MiniTorch only. It does not use torch or transformers.
Checkpoints are read with `safetensors.numpy`, text is tokenized with `tokenizers`,
and every entry point asserts that `torch` was never imported.

## Layout

| path | what |
|---|---|
| `minitorch/`, `src/`, `compile_cuda.sh` | copied from `llmsys_hw4`, with the changes listed below |
| `specdec/model.py` | GPT-2 forward in MiniTorch, per-request KV cache with `crop()` rollback |
| `specdec/decoding.py` | autoregressive baseline; speculative decoding (greedy, and rejection sampling per Leviathan et al. 2023) |
| `specdec/reference.py` | independent float64 numpy GPT-2, used only as a parity reference |
| `specdec/test_correctness.py` | correctness gates (run before every benchmark) |
| `specdec/bench.py`, `specdec/summarize.py` | benchmark (JSONL output) and markdown tables |
| `specdec/profile_step.py`, `specdec/profile_spec.py` | cProfile of single forwards / one speculative run |
| `specdec/prepare_prompts.py` | freezes 50 WikiText-2 test prompts (64 tokens each) into `results/prompts_wikitext2.json` |
| `jobs/` | interact-style `salloc` wrappers run under `nohup` |
| `results/*.jsonl`, `logs/` | raw results and job logs |

## Changes to the copied MiniTorch

1. `cuda_kernel_ops.py`: dropped `import torch`. The fused kernels used `torch.cuda.current_stream()`;
   they now use the default stream (0), which is the stream torch returned there.
2. `tensor.py` `contiguous()`: returns `self` when the tensor is already compact row-major.
   Before this change every matmul copied its weight through a GPU `id_map` round trip.
3. `tensor_data.py`: `astype(float32, copy=False)`, so `view()` shares storage (as upstream
   MiniTorch does). Before this change every `view()` of a weight copied it on the host.
4. `src/combine.cu` + `cuda_kernel_ops.make_resident`: device-resident right operand.
   Weights are uploaded once, and `MatrixMultiplyResidentB` runs the same HW1 tiled kernel
   without allocating, copying and freeing B on every call. Inference only.

## Running (Bridges-2)

```bash
cd /jet/home/yni3/11868/project
python -m specdec.prepare_prompts              # login node, CPU; already done
bash jobs/submit.sh 00:40:00 test jobs/stage2.sh   # correctness + micro benchmark
bash jobs/chain3.sh / jobs/stage4.sh           # benchmark sweeps (see the scripts)
python -m specdec.summarize results/sweep_same_node.jsonl
```

Use the venv `/jet/home/yni3/11868/llmsys_hw4/.venv` and `module load cuda/12.4.0`.
Run from the project root, because `ctypes.CDLL` loads kernels by relative path.

## Preliminary results (2026-11-03, V100-32GB, batch 1, fp32)

Setup: 10 WikiText-2 test prompts × 64 prompt tokens, 64 new tokens each. Target is GPT-2 medium
(355M), draft is GPT-2 small (124M). Both run as MiniTorch models with resident weights.
Times are decode wall-clock and exclude prefill.

**Correctness.** MiniTorch logits vs the float64 numpy reference: max |diff| 3.7e-4 (small) and
3.2e-4 (medium), using the same LayerNorm eps. MiniTorch's fused LayerNorm hard-codes eps=1e-8,
while GPT-2 uses 1e-5. That difference alone moves medium logits by 5.5e-2, but argmax still agrees
at every position. Incremental vs full forward and crop+refeed differ by exactly 0. Greedy
speculative output equals target greedy output in every benchmark run (60/60 below).

**Main sweep** (`results/sweep_same_node.jsonl`, one allocation on node v007):

| config | decode tok/s | speedup | acceptance | tokens/iter | draft time share | = target greedy |
|---|---|---|---|---|---|---|
| target AR | 1.66 | 1.00x | - | 1.00 | - | - |
| spec k=1 | 1.66 | 1.00x | 0.736 | 1.73 | 0.32 | 10/10 |
| spec k=2 | 1.73 | **1.04x** | 0.689 | 2.34 | 0.47 | 10/10 |
| spec k=3 | 1.63 | 0.98x | 0.584 | 2.71 | 0.56 | 10/10 |
| spec k=4 | 1.56 | 0.94x | 0.542 | 3.11 | 0.63 | 10/10 |
| spec k=6 | 1.34 | 0.81x | 0.432 | 3.52 | 0.71 | 10/10 |

Per-prompt speedup ranges from 0.85–1.11x (k=1) to 0.79–1.32x (k=3).
Sampling (T=1.0, k=3, `results/main.jsonl`, node v013/v007 mix): acceptance 0.487, 2.41 tokens/iter.

**Per-forward latency** (ms; `results/stage2_micro.jsonl`; cache already holds 64 tokens):

| new tokens T | 1 | 2 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|
| GPT-2 medium | 561 | 603 | 598 | 639 | 643 | 661 |
| GPT-2 small | 276 | 284 | 291 | 297 | 307 | 309 |

**Effect of the MiniTorch fixes** (GPT-2 medium, one decode step): 3574 ms (stock HW4 code) →
1018 ms (fixes 2+3) → 561 ms (fixes 2+3+4, resident weights).

### Why the speedup is small
- Speculative decoding wins when verifying k+1 tokens costs about the same as 1 token *and*
  the draft is much cheaper than the target. The first condition holds: T=9 costs 1.18x T=1.
  The second does not. MiniTorch decode time is dominated by fixed per-op overhead (each op
  does cudaMalloc, H2D, kernel, D2H, cudaFree; elementwise ops average 0.7 ms each). So cost
  scales with layer count, and small/medium = 12/24 layers costs about 0.5x, not 124M/355M ≈ 0.35x.
- Target verify calls inside the loop cost 709–760 ms, vs 623 ms per AR T=1 step. The isolated
  latency grid (`logs/stage3b.log`, `specdec/profile_spec.py`) predicts only about 650–670 ms
  for T=2 at these cache lengths. Draft calls (319–334 ms) show no clear penalty vs draft-only AR
  (320 ms on v013). The extra ~50–90 ms per verify is not explained yet.
- Cost model `speedup ≈ tokens_per_iter · t_AR / (k · t_draft + t_verify)` reproduces the table
  (k=2: 2.34 · 623 / (2 · 325 + 720) = 1.06x).
- Node-to-node variance is about 8% (target AR 1.54 tok/s on v013 vs 1.66 on v007). Speedups must
  use a baseline from the same allocation. The cross-node numbers in `results/main.jsonl` are
  not comparable with each other.

## Smaller draft: distilgpt2 (2026-11-03, `results/sweep_distil.jsonl`, one allocation on v010)

distilgpt2 (82M, 6 layers, same GPT-2 tokenizer) costs 137 ms/step vs 562 ms for the target (0.24x),
compared with 0.49x for GPT-2 small. Correctness gates all pass with it: logits vs numpy ref max |diff| 2.1e-4,
and greedy spec equals target greedy 9/9 in the test and 50/50 in the sweep.

| config | decode tok/s | speedup | acceptance | tokens/iter | draft time share | = target greedy |
|---|---|---|---|---|---|---|
| target AR | 1.84 | 1.00x | - | 1.00 | - | - |
| draft AR (distilgpt2) | 7.51 | 4.09x | - | 1.00 | - | - |
| spec k=1 | 2.32 | 1.26x | 0.685 | 1.67 | 0.19 | 10/10 |
| spec k=2 | 2.53 | **1.38x** | 0.551 | 2.08 | 0.31 | 10/10 |
| spec k=3 | 2.46 | 1.34x | 0.464 | 2.36 | 0.40 | 10/10 |
| spec k=4 | 2.30 | 1.25x | 0.383 | 2.49 | 0.46 | 10/10 |
| spec k=6 | 2.03 | 1.11x | 0.294 | 2.70 | 0.55 | 10/10 |

Per-prompt speedup for k=2 ranges from 1.12x to 1.58x. The cost model predicts 1.30/1.41/1.37/1.28/1.12x
for k=1/2/3/4/6, which matches the measurements.

Compared with GPT-2 small as the draft, acceptance per token drops (k=2: 0.55 vs 0.69), but each draft step
is about 2.5x cheaper, so the best speedup rises from 1.04x to 1.38x. The verify-call penalty seen with the
GPT-2 small draft also mostly disappears: verify costs 567–596 ms vs 562 ms per AR step, compared with
709–760 vs 623 before. That suggests the penalty grows with the size of the model interleaved with the
target, for example host/GPU cache pressure, but this is still unverified.
