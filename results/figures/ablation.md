| drafter | training data | fastest config | speedup | same-run baseline | tokens/step there | tokens/step at k=5 | training τ estimate (k) |
|---|---|---|---|---|---|---|---|
| distilgpt2 | pretrained | spec_k2 | 1.18x | no | 1.79 | 2.04 | - |
| distilgpt2 | WikiText-2 | spec_k2 | 1.37x | no | 2.07 | 2.59 | 2.87 (k=7) |
| distilgpt2 | verifier-generated | spec_k2 | 1.41x | no | 2.14 | 2.87 | 4.47 (k=7) |
| H-Spec | WikiText-2 | hspec_k3 | 1.79x | no | 1.83 | 1.83 | 1.89 (k=7) |
| H-Spec | verifier-generated | hspec_k5 | 2.31x | no | 2.37 | 2.37 | 3.67 (k=7) |

Warning: verifier ms per call differ across bench runs (ar_target 131, distilgpt2-pretrained 133, distil-wikitext 133, distil-gen 133, hspec-wikitext 104, hspec-gen 103; distil-gen is 1.29x hspec-gen) and some speedups have no same-run ar_target baseline: speedups from different runs are not comparable (re-run them with a paired ar_target; acceptance and tokens/iter are unaffected).
