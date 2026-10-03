# Complete softmax/sparsemax normalizer pair

Both fresh runs retain native AdamW, GPTMini parameters, corpus, seeds and full budgets.
Only attention normalization changes; causal projection follows the existing GPTMini.Convex Lean specification.

| Normalizer | Final train / held-out | Persistent tail | Tail failures | Sampled episodes after long onset | Training / wall seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| adamw-softmax | 23.8095% / 4.7619% | False | 3 | No long onset | 0.17 / 0.32 |
| adamw-sparsemax | 23.8095% / 9.5238% | False | 3 | No long onset | 0.18 / 1.68 |

All budgets, canonical histories, neighboring probes, tensor diagnostics and dense gradients remain archived.
Recovery metrics retain episodes, durations, censored recoveries, fixed tail windows and the whole post-onset grid.
Quality/time rules are prospective descriptive rules for this single pair; the all-six benchmark and independent architecture gates remain outstanding.
Candidate persistent generalization without the memorization plateau retains its separate phase label.
Eligible timing requires complete persistence in both runs and stable grokking in the softmax control.
Descriptive quality rule met: True. Descriptive timing rule met: False.
Scope: CPU_pipeline_fixture; no_learning_result.
No repeatable architecture improvement is certified by this pair.
