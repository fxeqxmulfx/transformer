# Complete softmax/sparsemax normalizer pair

Both fresh runs retain native AdamW, GPTMini parameters, corpus, seeds and full budgets.
Only attention normalization changes; causal projection follows the existing GPTMini.Convex Lean specification.

| Normalizer | Final train / held-out | Persistent tail | Tail failures | Sampled episodes after long onset | Training / wall seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| adamw-softmax | 100.0000% / 100.0000% | False | 2 | 6 | 7466.26 / 9260.63 |
| adamw-sparsemax | 100.0000% / 34.0566% | False | 201 | No long onset | 8771.17 / 12503.70 |

All budgets, canonical histories, neighboring probes, tensor diagnostics and dense gradients remain archived.
Recovery metrics retain episodes, durations, censored recoveries, fixed tail windows and the whole post-onset grid.
Quality/time rules are prospective descriptive rules for this single pair; the all-six benchmark and independent architecture gates remain outstanding.
Candidate persistent generalization without the memorization plateau retains its separate phase label.
Eligible timing requires complete persistence in both runs and stable grokking in the softmax control.
Descriptive quality rule met: False. Descriptive timing rule met: False.
Scope: user_directed_exploratory_pair; no_repeatable_architecture_claim.
No repeatable architecture improvement is certified by this pair.
