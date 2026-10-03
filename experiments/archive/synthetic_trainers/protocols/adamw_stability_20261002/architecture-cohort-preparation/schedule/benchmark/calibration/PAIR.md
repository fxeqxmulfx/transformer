# Complete native AdamW fixed-schedule calibration

Both fresh softmax cases retain the same model, optimizer, corpus, seeds and complete budget.
Only the declared fixed schedule changes. Lower learning rates also reduce per-update decoupled decay; their contributions are not isolated.

| Schedule | Final train / held-out | Persistent tail | Tail failures | Post-onset episodes | Training / wall seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| adamw-constant | 52.3810% / 23.8095% | False | 3 | No long onset | 0.34 / 2.01 |
| adamw-cosine-tail | 38.0952% / 0.0000% | False | 3 | No long onset | 0.38 / 0.67 |

Every failed target, actual learning rate, dense gradient, neighbor probe and native tensor diagnostic remains archived.
Both joint and held-out recovery scores retain sampled durations, censoring, tail windows, the leading partial window and the whole post-onset grid.
Timing ratios require complete stable grokking in both cases. All measured costs remain visible when ratios are ineligible.
A scientific passing calibration still requires six fresh crossed confirmations. CPU fixtures provide no scientific eligibility.
Eligible scientific recipes: None.
Scope: CPU_schedule_pair_fixture; no_learning_result.
No architecture improvement or repeatable benchmark is certified by this calibration.
