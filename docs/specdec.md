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
| `specdec/mla_model.py` | MLA verifier (GPT-2 medium converted to MLA) in MiniTorch, per-request latent cache with `crop()`, hidden-state taps; see "MLA verifier" below |
| `torch_train/hspec_drafter.py`, `project/minitorch_drafter.py` | H-Spec-style parallel drafter: PyTorch (training) and MiniTorch (inference) versions; see "H-Spec drafter" below |
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

## MLA verifier

The target for MLA + H-Spec-style drafting is the PyTorch MLA model trained by
`torch_train/train.py` (GPT-2 medium, `kv_rank` 512, RoPE dim 32). `specdec/mla_model.py`
runs it in MiniTorch with the same interface as `MiniTorchGPT2`, so `decoding.py` and
`bench.py` use it unchanged:

- The cache holds only the MLA latents per layer, `c_kv` [T, 512] and the shared rotary key
  `k_rope` [T, 32] (post-RoPE): 13,056 floats per token vs 49,152 for the MHA GPT-2 medium.
  `cache.latents(layer)` exposes them for a drafter to attend to in place.
- Attention runs through the fused MLA kernel (`CudaKernelOps.mla_fw`), which never
  materializes per-head K/V; matmul weights are device-resident.
- `forward(..., hidden_layers=[...])` also returns the residual stream after those layers at
  every fed position (the drafter's last-token summary input).
- `backend=None` runs the same model with numpy ops on CPU (used by the tests).

```bash
python torch_train/export_mla_verifier.py runs/mla-gpt2-medium/best.pt runs/mla-gpt2-medium/minitorch  # once, needs torch
python -m pytest tests/verifier                     # CPU tests + GPU tests (skipped without CUDA);
                                                    # MLA_CHECKPOINT=... also checks the trained model
python -m specdec.bench --target mla-gpt2-medium --draft distilgpt2 --configs micro --n-prompts 1 --out results/mla_micro.jsonl
python -m specdec.bench --target mla-gpt2-medium --draft distilgpt2 --configs ar_target spec_k2 --out results/mla_sweep.jsonl
```

The export directory defaults to `runs/mla-gpt2-medium/minitorch` (git-ignored); set
`MLA_VERIFIER_DIR` to use another. Results from different `--target`s must be summarized
separately (`summarize.py` refuses to mix them, since speedups use the `ar_target` baseline).

## H-Spec drafter

A block-parallel drafter in the style of H-Spec (arXiv 2609.24197) for the MLA verifier, with a
GPT-2-small-shaped transformer in place of H-Spec's Mamba-attention hybrid. One forward pass
drafts k tokens from `[anchor, MASK x k]` and the drafter keeps **no KV cache**:

- **Latent reuse:** drafter layer i attends to the verifier's cached MLA latents of layer
  `target_layers[i]` (default 12, 14, ..., 22), plus the block's own latents, causally. MLA
  shares a 512-d latent rather than per-head K/V, so the 768-wide drafter reads the 1024-wide
  verifier's cache directly; its `kv_up` starts as a copy of that verifier layer's.
- **Summary:** the verifier's hidden state at the last cached position after `tap_layers`
  (default 1, 6, 11, 16, 22) is fused and added to the anchor embedding.
- **Trunk/head:** 6 pre-LN GPT-2 blocks (768 wide, initialized from distilgpt2), then a learned
  768 -> 1024 projection into the verifier's frozen LM head.

Two implementations:

| | file | used for |
|---|---|---|
| PyTorch | `torch_train/hspec_drafter.py` | training: `forward(tokens, anchors, latents, hidden)` drafts many blocks per sequence in one pass (block mask: a block sees only the prefix before its anchor); `target_context` (frozen verifier pass), `sample_anchors`, `block_targets`, `drafter_loss` (0.1 CE + 0.9 TV with exp(-(j-1)/4) position weights), `init_from_gpt2`, `init_from_verifier`; `draft_block` = one block |
| MiniTorch | `project/minitorch_drafter.py` | inference: port of `draft_block`; attention through the fused MLA kernel (`query_start = P` gives exactly the drafter's mask); `draft_from_cache` / `greedy` read a `specdec` `LatentCache` |

`torch_train/export_hspec_drafter.py` exports a drafter for MiniTorch (the frozen head is not
stored; `MiniTorchHSpecDrafter.from_verifier` takes it from the verifier). Tests:
`python -m pytest tests/model_tests/drafter` (CPU, plus CUDA parity when a GPU is available).

**Training** (`bash train.sh`: data generation, `torch_train/train_drafter.py`, export).
Trunk from distilgpt2, `kv_up` and the frozen head from the verifier
(`runs/mla-gpt2-medium/best.pt`); one frozen verifier pass per batch gives the latents, hidden
taps and teacher logits; loss 0.1 CE + 0.9 TV; AdamW with 1% warmup and cosine decay to 0.

- Default, `DATA=generated`: `torch_train/generate_verifier_data.py` has the verifier continue
  64-token WikiText-2 prompts with 64 greedy tokens (as in the benchmark; `data/verifier_gen/`,
  generated once). The drafter trains on these rows with anchors from the last prompt token on
  (`--min-anchor` = 63), so every block's prefix, anchor and targets are verifier output: 48
  anchors per row, lr 6e-4, 5 epochs, outputs in `runs/hspec-drafter-gen/`. Why: the first
  drafter, trained on human text, reached τ 1.67 in real decoding vs 1.78 estimated in training.
  More prompts: `--source wikitext-103` or `--prompt-stride 32` on the generation script.
- `DATA=wikitext`: the earlier setup (raw WikiText-2 blocks, 32 anchors, lr 3e-4,
  `runs/hspec-drafter/`).

Validation reports the loss, per-position acceptance (1 - TV) and `tau_greedy` (1 + leading top-1
agreements per block, teacher-forced; on generated data it is measured on verifier text, so it
should track the benchmark's tokens/iter more closely). Outputs: `best.pt`, `loss_history.json`,
`loss_curve.png` and the MiniTorch export in `minitorch/`. Extra flags pass through, e.g.
`bash train.sh --max-steps 200 --eval-every 100` for a quick run.

**Decoding** (`specdec/decoding.py: speculative_block`): per iteration, one drafter pass drafts up
to k tokens from the verifier's cache (`draft_from_cache`), one verifier pass scores
`[anchor] + drafts` (appending their latents and returning their hidden taps), and the cache is
cropped to the anchor plus the accepted drafts; the hidden state at the last accepted position is
the next summary. Acceptance is shared with `speculative` (greedy, or Leviathan rejection sampling;
parallel drafts are sampled independently per position, which keeps sampling lossless).

```bash
python -m specdec.bench --target mla-gpt2-medium --configs micro --n-prompts 1 --out results/hspec_micro.jsonl
python -m specdec.bench --target mla-gpt2-medium --draft distilgpt2 \
  --configs ar_target spec_k2 hspec hspec_k3 --out results/hspec_sweep.jsonl
python -m specdec.summarize results/hspec_sweep.jsonl
```

`micro` adds one H-Spec drafter pass ("hspec-drafter", T = block size) to the verifier/draft
latency grid. `hspec[_k{K}][_t{TEMP}]` runs block drafting with K <= block size (default: the
block size); its "tokens/iter" is the real τ and "identical to target greedy" the losslessness
check. The drafter is read from `runs/hspec-drafter/minitorch` (`--hspec-drafter` or
`HSPEC_DRAFTER_DIR` to change).

## Drafter ablation (`eval.sh`)

Drafter architecture x drafter training data, all against the frozen MLA verifier:

| | WikiText-2 text | verifier-generated text (`data/verifier_gen_s32`) |
|---|---|---|
| distilgpt2, classic autoregressive drafter | `runs/distil-drafter-wikitext` | `runs/distil-drafter-gen-s32` |
| H-Spec block drafter | `runs/hspec-drafter-wikitext` | `runs/hspec-drafter-gen-s32` |

All four start from distilgpt2 and use the same verifier-teacher loss (0.1 CE + 0.9 TV),
optimizer, schedule, batch (16 x 128 tokens), 10 epochs and data files, so only the architecture
and the data change. The classic drafters are trained by `torch_train/finetune_distilgpt2.py`
(next-token positions, no block weights; on generated rows only the predictions of generated
tokens are scored; lr 2e-4, `DISTIL_LR`); its validation `tau_greedy[k-1]` is the classic
drafter's mean tokens per verifier step with k drafts, the counterpart of the H-Spec drafter's
(both teacher-forced over every position, so both overestimate decoding; see below).
Its `best/` (`save_pretrained`) loads directly in specdec: `python -m specdec.bench --draft-dir
runs/distil-drafter-gen-s32/best --draft-label distil-gen ...`. The H-Spec cells reuse the two
runs above (trained with the same settings, which `tests/model_tests/drafter/test_eval_sh.py`
checks). The stride-32 generated set has 74,745 rows against WikiText-2's 18.6k blocks of 128, so
the generated-data runs take ~4x as many steps in their 10 epochs.

```bash
bash eval.sh                   # train what is missing, export, benchmark, summarize, plot
STAGES=train bash eval.sh      # or split across allocations: train ...
STAGES=bench bash eval.sh      # ... then benchmark (configs already in results/eval.jsonl are skipped)
N_PROMPTS=20 TEMPS=1.0 bash eval.sh   # more prompts, plus sampling at temperature 1
DRY_RUN=1 bash eval.sh         # print the commands
```

The bench stage runs `spec_k2..k5` for each classic drafter (and pretrained distilgpt2 as a
no-fine-tuning reference, `REFERENCE=0` to skip) and `hspec_k3`, `hspec_k5`, `hspec` (k = 7) for
each H-Spec drafter, on the same prompts, into `results/eval.jsonl` with labels
`distilgpt2-pretrained`, `distil-wikitext`, `distil-gen`, `hspec-wikitext`, `hspec-gen`. Each
bench run also times `ar_target` and stamps its rows with a `run_id`; speedups divide by the
same run's `ar_target` (`specdec.summarize`), so a machine that changes speed between runs (another
job, a different node after `STAGES` splits) cannot skew the comparison. `results/eval.md` has the
per-config table plus the fastest config per drafter; `plot.py` draws `results/figures/`
(speedup and τ vs k, the 2x2 ablation, acceptance by draft position, time per step, training
curves, training estimate vs measured τ, memory) and `ablation.md`.

**Results** (`results/eval_paired.jsonl`, `results/eval_paired.md`, figures in
`results/figures_paired/`; 10 prompts, greedy, every output identical to the verifier's; each
speedup against the verifier alone timed in the same bench run):

| fastest config (speedup) | WikiText-2 | verifier-generated |
|---|---|---|
| distilgpt2 (pretrained: k = 2, 1.16x) | k = 2, 1.34x | k = 2, 1.38x |
| H-Spec | k = 3, 1.34x | k = 7, 1.70x |

| τ (tokens per verifier step) at k = 5 | WikiText-2 | verifier-generated |
|---|---|---|
| distilgpt2 (pretrained: 2.04) | 2.59 | 2.87 |
| H-Spec | 1.83 | 2.37 |

- On WikiText-2 the two architectures tie (1.34x); on verifier-generated text H-Spec is ahead
  (1.70x vs 1.38x). Generated data helps H-Spec far more (+0.54 τ, +0.36x) than distilgpt2
  (+0.28 τ, +0.04x): the WikiText H-Spec drafter collapses after the second draft
  (P(draft j | draft j-1) = 0.51, 0.47, 0.31, 0.15, 0.00), so its τ is flat at 1.83 for k = 3, 5
  and 7, while the generated-data drafter keeps 0.55-0.65 at every position.
- distilgpt2 accepts more at equal k (it conditions on its own drafts) but pays one forward per
  draft (31 ms vs 28 ms for a whole H-Spec block), so its best k is 2 and H-Spec's is 5-7.
- Training-time τ estimates overestimate decoding (distilgpt2, generated, k = 5: 3.82 vs 2.87;
  H-Spec, generated, k = 7: 3.67 vs 2.41): they average over every start position, while decoding
  starts each step right after a rejection, where disagreements cluster.
- Timing: the verifier alone ran at 125 ms per token in the three distilgpt2 runs and 94-95 ms in
  the two H-Spec runs (same split in the first, unpaired run, whose H-Spec speedups were inflated
  to 1.79x and 2.31x). Within each run the ratios are consistent (a verifier call costs 1.03-1.06x
  a one-token call in both), so the paired speedups compare fairly; absolute tok/s across runs do
  not. The H-Spec runs came last both times, so the cause may be the machine over time rather than
  the process; running an H-Spec bench before a distilgpt2 one would tell.

**Memory** (`results/memory.md`, `results/figures_paired/memory.png`, and the measured table in
`results/eval_paired.md`). `python -m specdec.memory` counts parameters and caches exactly from
the models' configs (eval.sh runs it on the exports). Each eval.sh bench run also records a
`memory` row per loaded model: GPU and host bytes measured around its loading, its parameters and
cache. All float32:

| | parameters | weights | own cache per token | reads from the verifier | measured GPU (MiniTorch) |
|---|---|---|---|---|---|
| MLA verifier | 355.6M | 1,357 MB | 13,056 floats (51 KB); its MHA counterpart: 49,152 | - | 1,302 MB |
| distilgpt2 drafter | 81.9M | 312 MB | 9,216 floats (36 KB) | nothing | 340 MB |
| H-Spec drafter | 95.2M | 363 MB | 0 (a pass holds 26,112 floats for its block) | latent cache, hidden taps, LM head (51.5M) | 220 MB |

- H-Spec has more weights (its 38.6M-parameter token embedding table is 40% of them; the summary
  fusion and MLA-shaped attention add 6.0M and 20.6M) but no cache; distilgpt2 adds 36 KB per
  cached token. At batch 1 and 1,024 tokens distilgpt2 adds 348 MB and H-Spec 363 MB; they are
  equal at 1,441 cached tokens (batch x context), and beyond that H-Spec adds less (at
  16 x 1,024 tokens: 888 vs 363 MB).
- On the GPU H-Spec is the smaller one (220 vs 340 MB): MiniTorch keeps embedding tables,
  LayerNorms and caches on the host and uploads only matmul weights, and H-Spec's largest table is
  only gathered from, while distilgpt2 needs its own 154 MB output layer on the GPU. The measured
  values are the uploaded matmul weights plus 30-50 MB of allocation overhead (expected 1,255 /
  309 / 189 MB).
- The MiniTorch H-Spec drafter used to upload its own copy of the verifier's LM head (206 MB on
  the GPU plus a host copy). It now multiplies by the verifier's tensor
  (`MiniTorchHSpecDrafter.from_verifier`, `shares_head`), so the shared head costs nothing extra.
- Not resident: the fused MLA kernel takes `kv_up` from host storage, so every attention call
  copies it to the GPU (about 100 MB per verifier forward, 25 MB per H-Spec pass). Uploading it
  once would cost that much GPU memory and save the transfers.

## Preliminary results (2026-11-03, V100-32GB, batch 1, fp32)

Setup: 10 WikiText-2 test prompts × 64 prompt tokens, 64 new tokens each. Target is GPT-2 medium
(355M), draft is GPT-2 small (124M). Both run as MiniTorch models with resident weights.
Times are decode wall-clock and exclude prefill.

**Correctness.** MiniTorch logits vs the float64 numpy reference: max |diff| 3.7e-4 (small) and
3.2e-4 (medium), using the same LayerNorm eps. MiniTorch's fused LayerNorm hard-codes eps=1e-8,
while GPT-2 uses 1e-5. That difference alone moves medium logits by 5.5e-2, but argmax still agrees
at every position. Incremental vs full forward and crop+refeed differ by exactly 0. Greedy
speculative output equals target greedy output in every benchmark run (60/60 below).

> **Update after the MLA merge:** the fused LayerNorm kernel now uses eps=1e-5 (GPT-2's value),
> recorded as `specdec.common.MINITORCH_LN_EPS`. `specdec.test_correctness` builds its tight reference
> with that eps, so it now compares against the checkpoint eps. All results on this page were
> measured with the old eps=1e-8 kernel.

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
