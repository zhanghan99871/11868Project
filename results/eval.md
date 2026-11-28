| config | prompts | decode tok/s | speedup vs target AR | acceptance | tokens/iter | draft time share | identical to target greedy |
|---|---|---|---|---|---|---|---|
| ar_target | 10 | 7.83 | 1.00x | - | 1.00 | - | - |
| spec_k2 [distilgpt2-pretrained] | 10 | 9.27 | 1.18x | 0.402 | 1.79 | 0.32 | 10/10 |
| spec_k3 [distilgpt2-pretrained] | 10 | 8.47 | 1.08x | 0.308 | 1.90 | 0.41 | 10/10 |
| spec_k4 [distilgpt2-pretrained] | 10 | 7.74 | 0.99x | 0.253 | 1.98 | 0.48 | 10/10 |
| spec_k5 [distilgpt2-pretrained] | 10 | 7.07 | 0.90x | 0.217 | 2.04 | 0.53 | 10/10 |
| spec_k2 [distil-wikitext] | 10 | 10.69 | 1.37x | 0.547 | 2.07 | 0.32 | 10/10 |
| spec_k3 [distil-wikitext] | 10 | 10.13 | 1.29x | 0.438 | 2.29 | 0.41 | 10/10 |
| spec_k4 [distil-wikitext] | 10 | 9.59 | 1.22x | 0.376 | 2.46 | 0.48 | 10/10 |
| spec_k5 [distil-wikitext] | 10 | 9.03 | 1.15x | 0.328 | 2.59 | 0.53 | 10/10 |
| spec_k2 [distil-gen] | 10 | 11.04 | 1.41x | 0.583 | 2.14 | 0.32 | 10/10 |
| spec_k3 [distil-gen] | 10 | 10.94 | 1.40x | 0.504 | 2.47 | 0.41 | 10/10 |
| spec_k4 [distil-gen] | 10 | 10.54 | 1.35x | 0.439 | 2.70 | 0.48 | 10/10 |
| spec_k5 [distil-gen] | 10 | 9.98 | 1.27x | 0.386 | 2.87 | 0.53 | 10/10 |
| hspec_k3 [hspec-wikitext] | 10 | 14.04 | 1.79x | 0.283 | 1.83 | 0.22 | 10/10 |
| hspec_k5 [hspec-wikitext] | 10 | 13.88 | 1.77x | 0.173 | 1.83 | 0.21 | 10/10 |
| hspec [hspec-wikitext] | 10 | 13.68 | 1.75x | 0.126 | 1.83 | 0.21 | 10/10 |
| hspec_k3 [hspec-gen] | 10 | 16.92 | 2.16x | 0.406 | 2.19 | 0.22 | 10/10 |
| hspec_k5 [hspec-gen] | 10 | 18.13 | 2.31x | 0.285 | 2.37 | 0.21 | 10/10 |
| hspec [hspec-gen] | 10 | 18.05 | 2.31x | 0.211 | 2.41 | 0.21 | 10/10 |

Fastest config per drafter:

| draft | temperature | best config | decode tok/s | speedup vs target AR | acceptance | tokens/iter | identical to target greedy |
|---|---|---|---|---|---|---|---|
| hspec-gen | 0 | hspec_k5 | 18.13 | 2.31x | 0.285 | 2.37 | 10/10 |
| hspec-wikitext | 0 | hspec_k3 | 14.04 | 1.79x | 0.283 | 1.83 | 10/10 |
| distil-gen | 0 | spec_k2 | 11.04 | 1.41x | 0.583 | 2.14 | 10/10 |
| distil-wikitext | 0 | spec_k2 | 10.69 | 1.37x | 0.547 | 2.07 | 10/10 |
| distilgpt2-pretrained | 0 | spec_k2 | 9.27 | 1.18x | 0.402 | 1.79 | 10/10 |
