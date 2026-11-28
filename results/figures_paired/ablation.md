| drafter | training data | fastest config | speedup | same-run baseline | tokens/step there | tokens/step at k=5 | training τ estimate (k) |
|---|---|---|---|---|---|---|---|
| distilgpt2 | pretrained | spec_k2 | 1.16x | yes | 1.79 | 2.04 | - |
| distilgpt2 | WikiText-2 | spec_k2 | 1.34x | yes | 2.07 | 2.59 | 2.87 (k=7) |
| distilgpt2 | verifier-generated | spec_k2 | 1.38x | yes | 2.14 | 2.87 | 4.47 (k=7) |
| H-Spec | WikiText-2 | hspec_k3 | 1.34x | yes | 1.83 | 1.83 | 1.89 (k=7) |
| H-Spec | verifier-generated | hspec | 1.70x | yes | 2.41 | 2.37 | 3.67 (k=7) |
