# MiniMLA 项目阶段总结：MiniTorch 投机解码初步结果

2026-11-03 · Group 16

## 摘要

投机解码这一部分可以算初步成功：实现完全基于 MiniTorch，正确性检查全部通过；用 distilgpt2 当 draft、GPT-2 medium 当 target 时，解码最快提速 **1.38 倍**（k=2）。不过，proposal 的核心 MLA 还没开始做。

- **正确性**：贪心投机解码的输出和 target 自己贪心生成的结果逐 token 相同，所有基准运行都是如此。MiniTorch 前向与 float64 numpy 参考实现的 logits 误差不超过 4e-4。
- **加速**：用 GPT-2 small 当 draft 时最多只有 1.04 倍，换成 distilgpt2 后达到 1.38 倍。实测值和成本模型的预测相符。
- **MiniTorch 优化**：修了三个性能问题，GPT-2 medium 生成一个 token 的时间从 3.57 s 降到 0.56 s。
- **主要局限**：每个配置只跑了 10 个 prompt；采样模式（T>0）的数据还不完整；绝对速度很低，只有约 1.8 tok/s。

## 背景与本阶段范围

我们的 proposal（Group 16，MiniMLA）计划在 MiniTorch 里实现 Multi-head Latent Attention 和带 KV cache 的推理。投机解码在 proposal 里是用来验证 KV cache 的应用场景。本阶段先把 MLA 放一放，把推理路径和投机解码框架搭起来，之后 MLA 版的 target 可以直接接进来。

项目只依赖 MiniTorch，不用 torch，也不用 transformers。权重用 `safetensors.numpy` 读取，分词用 `tokenizers`，数据集用 pyarrow 读。没有用 `datasets`，因为它会间接导入 torch。每个入口脚本结束时都会断言 torch 没有被加载。

## 实现

实现包括三部分：用 MiniTorch 写的 GPT-2 推理、投机解码算法，以及对 MiniTorch 的性能修正。

| 文件 | 内容 |
| --- | --- |
| `specdec/model.py` | GPT-2 前向（MiniTorch），每个请求有独立的 KV cache，支持 `crop()` 回滚 |
| `specdec/decoding.py` | 自回归基线；投机解码，包括贪心验证和采样模式下的 rejection sampling（Leviathan et al. 2023） |
| `specdec/reference.py` | 独立的 float64 numpy 版 GPT-2，只用来做对照 |
| `specdec/test_correctness.py` | 正确性检查，每次跑基准前都先跑一遍 |
| `specdec/bench.py`、`specdec/summarize.py` | 基准测试（输出 JSONL）和生成结果表 |
| `minitorch/`、`src/` | 从 HW4 复制过来并做了修改（见下文） |

**投机解码的每一轮：**

1. draft 模型逐个生成 k 个候选 token。
2. target 一次前向同时验证这 k 个 token。
3. 判断从哪里开始不接受：贪心模式下比较 argmax；采样模式下以 min(1, p/q) 的概率接受，一旦拒绝，就从归一化的 max(p−q, 0) 重新采样。
4. 如果 k 个全部接受，再额外得到一个由 target 生成的 token。
5. 两个模型的 KV cache 都回滚到被接受的位置。

**MiniTorch 的修改：**

- 去掉了 `cuda_kernel_ops.py` 里的 `import torch`。fused kernel 改用默认 CUDA stream（0），和原来 torch 在这里返回的 stream 是同一个。
- `contiguous()`：张量已经是紧凑的行主序时直接返回自身。原实现每次矩阵乘都会把权重在 GPU 上完整复制一遍。
- `tensor_data.py`：`view()` 和原张量共享存储，不再在 CPU 上复制。
- 权重常驻 GPU：新增 `DeviceUpload` 和 `MatrixMultiplyResidentB`。权重只上传一次，矩阵乘不再对右乘矩阵反复分配、拷贝和释放。kernel 仍是 HW1 的 tiled matmul。这个改动只用于推理。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 硬件 | Bridges-2，单张 V100-32GB，batch 1，fp32 |
| target | GPT-2 medium（355M 参数，24 层） |
| draft | GPT-2 small（124M，12 层）；distilgpt2（82M，6 层）。三个模型共用 GPT-2 的分词器 |
| 数据 | WikiText-2 test，用固定种子抽取 50 个 64 token 的 prompt；基准测试用前 10 个，正确性测试用最后 3 个 |
| 生成长度 | 每个 prompt 生成 64 个新 token |
| 指标 | 解码阶段的 tok/s（不含 prefill）、接受率、每轮产出的 token 数、draft 耗时占比 |
| 评测方式 | 不同节点之间速度相差约 8%，所以每组对比（包括自回归基线）都在同一次 GPU 分配里跑完 |

## 结果

### 正确性

所有正确性检查都通过了，投机解码不会改变 target 的输出。

- **前向精度**：MiniTorch 的 logits 和 float64 numpy 参考实现相比，最大误差分别为 3.7e-4（GPT-2 small）、3.2e-4（medium）、2.1e-4（distilgpt2）。
- **LayerNorm eps 的差异**：MiniTorch 的 fused LayerNorm 把 eps 写死为 1e-8，而 GPT-2 用的是 1e-5。仅这一点就会让 medium 的 logits 偏移 5.5e-2，但每个位置的 argmax 都不变。所以上一条的误差是在参考实现也用 1e-8 时测的。
- **KV cache**：增量前向和完整前向的结果误差为 0；cache 回滚（crop）后重新输入，误差也为 0。
- **贪心等价**：投机解码的输出与 target 贪心生成的结果完全一致。GPT-2 small 作 draft 时 60/60 个运行一致，distilgpt2 作 draft 时 50/50 一致。
- **采样分布**：在玩具模型上，rejection sampling 的输出分布与目标分布的 TV 距离是 0.012。直接从目标分布采样同样多的样本，TV 距离是 0.011，两者相当。

### 解码加速

加速比 = 投机解码 tok/s ÷ 同一次分配内 target 自回归的 tok/s。贪心解码，10 个 prompt，每个生成 64 个新 token。

| k | GPT-2 small 加速比 | distilgpt2 加速比 | small 接受率 | small 每轮 token 数 | distilgpt2 接受率 | distilgpt2 每轮 token 数 | distilgpt2 draft 耗时占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00x | 1.26x | 0.736 | 1.73 | 0.685 | 1.67 | 0.19 |
| 2 | **1.04x** | **1.38x** | 0.689 | 2.34 | 0.551 | 2.08 | 0.31 |
| 3 | 0.98x | 1.34x | 0.584 | 2.71 | 0.464 | 2.36 | 0.40 |
| 4 | 0.94x | 1.25x | 0.542 | 3.11 | 0.383 | 2.49 | 0.46 |
| 6 | 0.81x | 1.11x | 0.432 | 3.52 | 0.294 | 2.70 | 0.55 |

两组数据各自以同一次分配里的 target 自回归速度为基线：GPT-2 small 那组在节点 v007 上跑，基线 1.66 tok/s（`results/sweep_same_node.jsonl`）；distilgpt2 那组在节点 v010 上跑，基线 1.84 tok/s（`results/sweep_distil.jsonl`）。distilgpt2 作 draft、k=2 时，各 prompt 的加速比在 1.12x 到 1.58x 之间。

distilgpt2 的接受率比 GPT-2 small 低，但它每步只要 137 ms，GPT-2 small 要 276 ms，所以总体反而更快。采样模式目前只有一组数据：GPT-2 small 作 draft，T=1.0，k=3，接受率 0.487，每轮 2.41 个 token。这组数据是跨节点跑的，不能算加速比。

### 单次前向延迟

下表是 cache 里已有 64 个 token 时，一次前向的耗时（权重常驻 GPU）。一次前向处理 9 个新 token，只比处理 1 个慢 12–22%，所以 target 一次验证多个 token 的代价不大。

| 模型 | 层数 | T=1（ms） | T=9（ms） | T=9 / T=1 |
| --- | --- | --- | --- | --- |
| GPT-2 medium | 24 | 561 | 661 | 1.18 |
| GPT-2 small | 12 | 276 | 309 | 1.12 |
| distilgpt2 | 6 | 132 | 161 | 1.22 |

### MiniTorch 优化的效果

GPT-2 medium 单步解码的耗时：原版 HW4 代码要 3574 ms；改掉 `contiguous()` 和 `view()` 的多余复制后降到 1018 ms；再加上权重常驻 GPU，降到 561 ms。

## 分析

加速有限，主要是因为 draft 在 MiniTorch 上不够便宜。用下面的成本模型可以复现实测结果：

```
speedup ≈ tokens_per_iter × t_AR / (k × t_draft + t_verify)
```

以 distilgpt2、k=1/2/3/4/6 为例，模型预测的加速比是 1.30/1.41/1.37/1.28/1.12x，和实测值的差距都在 0.04 以内。

- **验证成本低**：一次处理 9 个 token，只比处理 1 个多花 1.18 倍的时间，投机解码能加速的前提是满足的。
- **draft 不够便宜**：MiniTorch 每个算子都要做一遍 cudaMalloc、拷入、kernel、拷出、cudaFree，仅逐元素算子就平均每个约 0.7 ms。这样一来，一次前向的耗时主要取决于层数，而不是参数量。GPT-2 small 的参数量是 medium 的 0.35 倍，耗时却是 0.49 倍，接近它的层数比例（12/24）。distilgpt2 只有 6 层，耗时是 medium 的 0.24 倍，所以加速明显更好。
- **k 不宜过大**：k 越大，接受率下降得越多，draft 耗时也线性增长，所以两个 draft 都在 k=2 时最快。
- **尚未解释的现象**：用 GPT-2 small 作 draft 时，target 的验证调用要 709–760 ms，而单独测量时约为 650–670 ms，多出 50–90 ms。换成 distilgpt2 后，验证调用为 567–596 ms，单步自回归是 562 ms，这个差距基本消失。怀疑和与 target 交替运行的模型大小有关（例如缓存压力），但还没有验证。

## 局限与下一步

投机解码这部分要达到能写进报告的程度，还差一轮更大规模的实验。之后工作重心转回 MLA。

**局限**

- 样本量小：每个配置只用了 10 个 prompt，也没有计算置信区间。
- 采样模式缺数据：distilgpt2 作 draft 时没有测 T>0。
- 绝对速度低：只有 1.8 tok/s，而且加速比受 MiniTorch 每个算子的固定开销影响很大，不能直接拿来和生产级框架比较。
- 上文提到的验证调用额外开销，原因还不清楚。

**下一步（按优先级）**

- [ ] 用 distilgpt2 作 draft 跑一轮完整实验：同一次分配、全部 50 个 prompt、T=0 和 T=1 都测，并给出置信区间。
- [ ] 实现 MLA：对 GPT-2 medium 的 KV 投影做 SVD 低秩分解，再少量微调，然后作为 target 接进现有的投机解码框架。
- [ ] 降低 MiniTorch 每个算子的固定开销，比如让中间结果留在 GPU 上，或者融合算子。
- [ ] （可选）用 target 的输出对 distilgpt2 做蒸馏，提高接受率。
- [ ] （可选）查明验证调用的额外开销从何而来。

## 复现

```bash
cd /jet/home/yni3/11868/project
module load cuda/12.4.0
source /jet/home/yni3/11868/llmsys_hw4/.venv/bin/activate
python -m specdec.test_correctness --draft distilgpt2
python -m specdec.bench --help
python -m specdec.summarize results/sweep_same_node.jsonl results/sweep_distil.jsonl
```

必须在项目根目录下运行，因为 `ctypes.CDLL` 是按相对路径加载 kernel 的。GPU 作业的提交方式见 `jobs/submit.sh`。
