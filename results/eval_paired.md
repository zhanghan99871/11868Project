Memory (bench `memory`: GPU/host bytes measured around loading each model):

| model | role | parameters | measured GPU (MB) | measured host (MB) | cache per token | cache allocated (MB) | reads verifier head |
|---|---|---|---|---|---|---|---|
| mla-gpt2-medium | verifier | 355.6M | 1302 | 1638 | 51.0 KB | 51 | - |
| distilgpt2-pretrained | drafter | 81.9M | 340 | 643 | 36.0 KB | 36 | - |
| distil-wikitext | drafter | 81.9M | 340 | 612 | 36.0 KB | 36 | - |
| distil-gen | drafter | 81.9M | 340 | 612 | 36.0 KB | 36 | - |
| hspec-wikitext | drafter | 95.2M | 220 | 546 | 0.0 KB | 0 | yes |
| hspec-gen | drafter | 95.2M | 220 | 546 | 0.0 KB | 0 | yes |

| config | prompts | decode tok/s | speedup vs target AR | acceptance | tokens/iter | draft time share | identical to target greedy |
|---|---|---|---|---|---|---|---|
| ar_target | 50 | 8.86 | 1.00x | - | 1.00 | - | - |
| spec_k2 [distilgpt2-pretrained] | 10 | 9.27 | 1.16x | 0.402 | 1.79 | 0.32 | 10/10 |
| spec_k3 [distilgpt2-pretrained] | 10 | 8.42 | 1.06x | 0.308 | 1.90 | 0.41 | 10/10 |
| spec_k4 [distilgpt2-pretrained] | 10 | 7.73 | 0.97x | 0.253 | 1.98 | 0.48 | 10/10 |
| spec_k5 [distilgpt2-pretrained] | 10 | 7.10 | 0.89x | 0.217 | 2.04 | 0.53 | 10/10 |
| spec_k2 [distil-wikitext] | 10 | 10.72 | 1.34x | 0.547 | 2.07 | 0.32 | 10/10 |
| spec_k3 [distil-wikitext] | 10 | 10.21 | 1.28x | 0.438 | 2.29 | 0.41 | 10/10 |
| spec_k4 [distil-wikitext] | 10 | 9.68 | 1.21x | 0.376 | 2.46 | 0.48 | 10/10 |
| spec_k5 [distil-wikitext] | 10 | 9.09 | 1.14x | 0.328 | 2.59 | 0.53 | 10/10 |
| spec_k2 [distil-gen] | 10 | 11.06 | 1.38x | 0.583 | 2.14 | 0.32 | 10/10 |
| spec_k3 [distil-gen] | 10 | 10.97 | 1.37x | 0.504 | 2.47 | 0.41 | 10/10 |
| spec_k4 [distil-gen] | 10 | 10.57 | 1.32x | 0.439 | 2.70 | 0.48 | 10/10 |
| spec_k5 [distil-gen] | 10 | 9.95 | 1.24x | 0.386 | 2.87 | 0.53 | 10/10 |
| hspec_k3 [hspec-wikitext] | 10 | 14.14 | 1.34x | 0.283 | 1.83 | 0.22 | 10/10 |
| hspec_k5 [hspec-wikitext] | 10 | 14.00 | 1.32x | 0.173 | 1.83 | 0.21 | 10/10 |
| hspec [hspec-wikitext] | 10 | 13.71 | 1.30x | 0.126 | 1.83 | 0.21 | 10/10 |
| hspec_k3 [hspec-gen] | 10 | 16.93 | 1.60x | 0.406 | 2.19 | 0.22 | 10/10 |
| hspec_k5 [hspec-gen] | 10 | 18.04 | 1.70x | 0.285 | 2.37 | 0.21 | 10/10 |
| hspec [hspec-gen] | 10 | 18.05 | 1.70x | 0.211 | 2.41 | 0.21 | 10/10 |

Fastest config per drafter:

| draft | temperature | best config | decode tok/s | speedup vs target AR | acceptance | tokens/iter | identical to target greedy |
|---|---|---|---|---|---|---|---|
| hspec-gen | 0 | hspec | 18.05 | 1.70x | 0.211 | 2.41 | 10/10 |
| distil-gen | 0 | spec_k2 | 11.06 | 1.38x | 0.583 | 2.14 | 10/10 |
| distil-wikitext | 0 | spec_k2 | 10.72 | 1.34x | 0.547 | 2.07 | 10/10 |
| hspec-wikitext | 0 | hspec_k3 | 14.14 | 1.34x | 0.283 | 1.83 | 10/10 |
| distilgpt2-pretrained | 0 | spec_k2 | 9.27 | 1.16x | 0.402 | 1.79 | 10/10 |
