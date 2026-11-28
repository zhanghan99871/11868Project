| model | parameters | weights (MB, fp32) | shared with verifier | own cache per token | transient per pass |
|---|---|---|---|---|---|
| verifier (MLA) | 355.6M | 1357 | 0.0M | 13,056 floats (51.0 KB) | 0 floats |
| distilgpt2 | 81.9M | 312 | 0.0M | 9,216 floats (36.0 KB) | 0 floats |
| H-Spec | 95.2M | 363 | 51.5M | 0 floats (0.0 KB) | 26,112 floats |

The verifier's MHA counterpart would cache 49,152 floats per token (3.76x its MLA latent cache).

Batch 1:

| setup | total at 128 tokens (MB) | added by drafter at 128 (MB) | total at 1024 tokens (MB) | added by drafter at 1024 (MB) |
|---|---|---|---|---|
| verifier alone | 1363 | - | 1408 | - |
| + distilgpt2 | 1680 | 317 | 1756 | 348 |
| + H-Spec | 1726 | 363 | 1771 | 363 |

distilgpt2 and H-Spec add the same memory at 1,441 cached tokens (batch x context).
