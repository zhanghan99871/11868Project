# MiniMLA: Multi-head Latent Attention with speculative decoding in MiniTorch

CMU 11-868 (LLM Systems), Group 16.

MiniMLA converts GPT-2 medium to **Multi-head Latent Attention (MLA)** and runs it in **MiniTorch**,
our own CUDA tensor library, with a compressed latent KV cache. It then speeds decoding up with
**speculative decoding**. The main component is a block drafter in the style of H-Spec
(arXiv 2609.24197) that **shares the verifier's latent KV cache** instead of keeping its own. It
drafts a whole block of tokens in one pass by attending directly to the verifier's cached
latents.

| | result |
|---|---|
| MLA verifier | KV cache 3.76× smaller than GPT-2 medium's (13,056 vs 49,152 floats per token); WikiText-2 validation perplexity 21.9 vs 21.2 for GPT-2 medium fine-tuned the same way |
| Speculative decoding | lossless: every output equals the verifier's own greedy decoding |
| Cache-sharing drafter | **1.70×** decoding speedup, against 1.38× for a fine-tuned distilgpt2 drafter; no drafter cache |

## Project formulation

The system has three components: an MLA verifier (the target model), a speculative decoding loop,
and a drafter that shares the verifier's KV cache. Notation: hidden size $D$, $H$ heads, layers
$\ell = 1..L$, a token at position $t$.

### 1. Multi-head Latent Attention (the verifier)

A standard multi-head attention (MHA) layer caches a key and a value for every head:
$2 \cdot H \cdot d_h = 2D$ floats per token per layer. MLA (DeepSeek-V2) instead caches a
low-rank latent that all heads share, plus one small rotary key that all heads also share:

$$[\,c_t;\ \tilde k^R_t\,] = W^{DKV} h_t, \qquad c_t \in \mathbb{R}^{r},\quad k^R_t = \mathrm{RoPE}_t(\tilde k^R_t) \in \mathbb{R}^{d_r}$$

Each head recovers its key and value from the latent, and its query has a content part and a
rotary part:

$$[\,k^C_{t,i};\ v_{t,i}\,] = W^{UKV}_i c_t + b_i, \qquad q_{t,i} = [\,q^C_{t,i};\ \mathrm{RoPE}_t(\tilde q^R_{t,i})\,] = W^Q_i h_t$$

$$o_{t,i} = \sum_{s \le t} \mathrm{softmax}_s\!\left(\frac{q^C_{t,i}\cdot k^C_{s,i} + q^R_{t,i}\cdot k^R_s}{\sqrt{d_n + d_r}}\right) v_{s,i}, \qquad u_t = W^O\,[o_{t,1}; \dots; o_{t,H}]$$

Only $(c_t, k^R_t)$ is cached. Our verifier is GPT-2 medium ($D = 1024$, $H = 16$, $L = 24$)
with rank $r = 512$, rotary dim $d_r = 32$ and head dims $d_n = d_v = 64$:

| per token | per layer | all 24 layers |
|---|---|---|
| MHA (GPT-2 medium) | $2D$ = 2,048 floats | 49,152 floats (192 KB) |
| MLA (ours) | $r + d_r$ = 544 floats | 13,056 floats (51 KB), 3.76× smaller |

Because $k^C_{s,i}$ and $v_{s,i}$ are linear in $c_s$, the up-projection can be folded into the
query and into the attention output. The fused CUDA kernel (`src/mla_kernel.cu`) uses this to
attend directly over the latent cache. It fuses the up-projection algebra, $QK^\top$, the causal
softmax and $AV$ without materializing per-head K/V or the attention matrix.

**From GPT-2 medium to MLA** (`torch_train/gpt2_mla_init.py`, `torch_train/train.py`):
1. Each head's content query copies GPT-2's query weights. Rotary queries start at zero and
   rotary keys at a small random value, so the rotary branch starts inactive.
2. All heads' key and value weights are stacked ($2D \times D$) and replaced by a rank-512
   truncated randomized SVD: $[W^K; W^V] \approx W^{UKV} W^{DKV}$. Biases are copied, and GPT-2's
   absolute position embeddings are kept.
3. The whole model is fine-tuned on WikiText-2 (10 epochs, lr $10^{-5}$, bf16).

The resulting validation perplexity is 21.9, against 21.2 for GPT-2 medium fine-tuned the same
way (`torch_train/train_ori.py`). For inference, the checkpoint is exported torch-free
(`torch_train/export_mla_verifier.py`) and runs in MiniTorch (`specdec/mla_model.py`). The
MiniTorch loss matches PyTorch to within $10^{-6}$.

### 2. Speculative decoding

Each decoding step does three things:
1. A cheap drafter proposes $k$ tokens $\hat x_{t+1..t+k}$, with distributions $q_j$.
2. The verifier scores all of them in **one** forward pass, giving
   $p_j = p(\cdot \mid x_{\le t}, \hat x_{\lt t+j})$ for $j = 1..k{+}1$.
3. The accepted drafts are kept and the rest are rolled back.

- **Greedy:** accept the longest prefix with $\hat x_{t+j} = \arg\max p_j$, then append the
  verifier's own token $\arg\max p_{n+1}$.
- **Sampling** (Leviathan et al., 2023): accept $\hat x_{t+j}$ with probability
  $\min\!\big(1, p_j(\hat x_{t+j}) / q_j(\hat x_{t+j})\big)$. At the first rejection, sample from
  $\mathrm{norm}(\max(p_j - q_j, 0))$; if every draft is accepted, sample from $p_{k+1}$.

Either way the output distribution is exactly the verifier's, so speculation is lossless. A step
emits $\tau = 1 + n_\text{accepted}$ tokens, and the verifier's cache is cropped back to the
accepted prefix. With $t_\text{AR}$ the verifier's time per token when decoding alone:

$$\text{speedup} \approx \frac{\tau \cdot t_\text{AR}}{t_\text{verify}(k{+}1) + t_\text{draft}(k)}$$

Verifying $k{+}1$ tokens costs about as much as decoding one (1.03-1.06× in MiniTorch), so
everything depends on τ and the drafting cost. A classic drafter such as distilgpt2 runs $k$
sequential forward passes, so $t_\text{draft}(k) = k \cdot t_q$, and it also has its own KV cache
to grow and roll back. Implementation: `specdec/decoding.py` (`autoregressive`, `speculative`,
`speculative_block`).

### 3. KV cache sharing: the H-Spec-style block drafter

The drafter keeps **no cache of its own**. At a step, the verifier's cache holds latents for
positions $< P$. The anchor $a$ is the last accepted token (position $P$). The drafter takes the
block $[\,a, \texttt{MASK}, \dots, \texttt{MASK}\,]$ at positions $P..P{+}k$ (with $k \le 7$) and
drafts all $k$ tokens in one pass:

- **Reads the verifier's latent cache in place.** Drafter layer $\ell$ runs MLA attention over the
  verifier's cached latents of layer $m_\ell \in \{12, 14, 16, 18, 20, 22\}$ for $s < P$, plus the
  block's own latents (causal within the block):

  $$\mathcal{K}_\ell = \underbrace{\{(c^{(m_\ell)}_s,\ k^{R,(m_\ell)}_s)\}_{s < P}}_{\text{verifier cache, shared}} \ \cup\ \underbrace{\{(c'_s,\ k'^{R}_s)\}_{P \le s}}_{\text{block, transient}}$$

  This works because an MLA latent is not tied to the verifier's heads or width. The 768-wide
  drafter reads the 1024-wide verifier's 512-d latents through its own $W^{UKV}$, initialized
  from verifier layer $m_\ell$.
- **Summary of the verifier's state.** The verifier's hidden states at position $P{-}1$ after
  layers $\{1, 6, 11, 16, 22\}$ are layer-normed, concatenated, projected down and up, and added
  to the anchor embedding.
- **Verifier's output layer.** Drafts come from $q_j = \mathrm{softmax}(W^\text{vocab}_\text{verifier}\, W^\text{proj}\, \tilde h_{P+j})$,
  using the verifier's frozen LM head (51.5M parameters, read in place) after a learned
  $768 \to 1024$ projection.

The verifier then scores $[a, \hat x_1..\hat x_k]$ in one pass, appending their latents and
returning the hidden taps. Its cache is cropped to the accepted prefix, and the hidden state at
the last accepted position becomes the next summary. The drafter costs one 28 ms pass per step at any
$k$ (distilgpt2: 31 ms per draft), adds **zero** cache floats per token (distilgpt2: 9,216), and
has nothing to roll back.

The trunk is 6 GPT-2 blocks initialized from distilgpt2 (embeddings, LayerNorms, MLPs). Two
implementations exist: PyTorch for training (`torch_train/hspec_drafter.py`) and MiniTorch for
inference (`project/minitorch_drafter.py`).

### 4. Training the drafters

The verifier is frozen. Drafters learn to match it with H-Spec's loss, a cross-entropy term on
the data token plus a total-variation term against the verifier, weighted by block position:

$$\mathcal{L} = \sum_{j=1}^{k} \tilde w_j \Big[\,0.1\,\mathrm{CE}(q_j, x_{P+j}) + 0.9\,\mathrm{TV}(p_j, q_j)\,\Big], \qquad \tilde w_j \propto e^{-(j-1)/4}$$

$1 - \mathrm{TV}(p_j, q_j)$ is exactly the probability that rejection sampling accepts draft $j$.

- **Training data.** Drafters train on **verifier-generated text**: 64-token WikiText-2 prompts
  continued by 64 greedy verifier tokens (`torch_train/generate_verifier_data.py`; 74,745 rows
  at prompt stride 32). Blocks are anchored inside the generated part, so training sees exactly
  what decoding sees.
- **Baseline drafter.** A distilgpt2 drafter is fine-tuned with the same loss at every next-token
  position (`torch_train/finetune_distilgpt2.py`).
- **Ablation.** `eval.sh` crosses drafter type (distilgpt2 vs H-Spec) with training data
  (WikiText-2 vs verifier-generated text).

## Results

Measured in MiniTorch, batch 1, float32, on 10 WikiText-2 prompts with 64 new tokens each,
greedy decoding. Speedups divide by the verifier decoding alone, timed in the same benchmark run.
The numbers come from `results/eval_paired.md`; [docs/figure.md](docs/figure.md) explains each
figure in `results/figures_paired/`.

| drafter | trained on | best k | τ (tokens per step) | speedup |
|---|---|---|---|---|
| distilgpt2 (no fine-tuning) | – | 2 | 1.79 | 1.16× |
| distilgpt2 | WikiText-2 | 2 | 2.07 | 1.34× |
| distilgpt2 | verifier-generated | 2 | 2.14 | 1.38× |
| H-Spec, shared cache | WikiText-2 | 3 | 1.83 | 1.34× |
| **H-Spec, shared cache** | **verifier-generated** | 7 | 2.41 | **1.70×** |

- **Speed.** distilgpt2 accepts more drafts at equal $k$ (it builds on its own drafts), but each
  draft costs another forward pass, so its best $k$ is 2. The H-Spec drafter drafts the whole
  block in one pass, so it profits from $k$ = 5-7.
- **Data.** Verifier-generated training data is what makes H-Spec win. Trained on WikiText-2,
  its acceptance collapses after the second draft and it only ties distilgpt2.
- **Memory** (`results/memory.md`, float32):

| | parameters | own cache per token | measured GPU (MiniTorch) |
|---|---|---|---|
| MLA verifier | 355.6M | 13,056 floats (51 KB) | 1,302 MB |
| distilgpt2 drafter | 81.9M | 9,216 floats (36 KB) | 340 MB |
| H-Spec drafter | 95.2M (+ verifier's LM head, shared) | 0 | 220 MB |

The H-Spec drafter has more weights but no cache. It adds less memory than distilgpt2 beyond
1,441 cached tokens (batch × context), and on the GPU it is already the smaller of the two.

## Repository layout

| path | contents |
|---|---|
| `minitorch/`, `src/`, `compile_cuda.sh` | MiniTorch with CUDA kernels: tiled matmul with device-resident weights, fused softmax and LayerNorm, fused MLA attention (`src/mla_kernel.cu`) |
| `torch_train/` | PyTorch training: MLA model and GPT-2 → MLA conversion (`decoder_mla.py`, `gpt2_mla_init.py`, `train.py`), H-Spec drafter (`hspec_drafter.py`, `train_drafter.py`), distilgpt2 drafter (`finetune_distilgpt2.py`), verifier-generated data, torch-free exports |
| `specdec/` | torch-free inference: GPT-2 and MLA models with KV/latent caches (`model.py`, `mla_model.py`), decoding loops (`decoding.py`), benchmark and tables (`bench.py`, `summarize.py`), memory accounting (`memory.py`) |
| `project/minitorch_drafter.py` | MiniTorch H-Spec drafter (reads the verifier's latent cache and LM head in place) |
| `tests/` | `unit/`, `kernel_tests/`, `verifier/`, `model_tests/` (CUDA tests skip without a GPU) |
| `train.sh`, `eval.sh`, `plot.py` | drafter training, the drafter ablation (train, benchmark, summarize), figures |
| `results/`, `runs/` | benchmark rows, tables and figures; training histories and arguments (checkpoints are git-ignored) |
| `docs/` | `specdec.md` (implementation notes and all experiments), `figure.md` (figure analysis) |
| `README_zh.md` | earlier progress report (speculative decoding with an MHA target, in Chinese) |

## Getting started

Run everything from the repository root: the CUDA kernels are loaded by relative path.

```bash
pip install -r requirements.txt
bash compile_cuda.sh              # builds minitorch/cuda_kernels/*.so (needs nvcc)
python -m pytest tests            # CPU tests run anywhere; CUDA tests skip without a GPU
```

The full pipeline (GPU):

```bash
# 1. GPT-2 medium -> MLA, fine-tuned on WikiText-2 (runs/mla-gpt2-medium), then a torch-free export
python torch_train/train.py
python torch_train/export_mla_verifier.py runs/mla-gpt2-medium/best.pt runs/mla-gpt2-medium/minitorch

# 2. Drafter ablation: generate verifier data, train the missing drafters, benchmark, summarize, plot
RESULTS=results/eval_paired.jsonl SUMMARY=results/eval_paired.md FIGURES=results/figures_paired bash eval.sh

# Or a single benchmark: the verifier alone vs H-Spec drafting with k = 5
python -m specdec.bench --target mla-gpt2-medium --hspec-drafter runs/hspec-drafter-gen-s32/minitorch \
    --configs ar_target hspec_k5 --out results/my_run.jsonl
python -m specdec.summarize results/my_run.jsonl
```

`bash train.sh` trains a single H-Spec drafter (verifier-generated data by default). `DRY_RUN=1
bash eval.sh` prints the ablation's commands without running them. [docs/specdec.md](docs/specdec.md)
covers each component, the MiniTorch changes and every experiment in detail.
