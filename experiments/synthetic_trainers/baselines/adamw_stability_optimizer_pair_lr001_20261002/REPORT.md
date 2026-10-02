# Complete AdamW / raw AMSGradW stability calibration

Source setting: *Convexifying Transformers*, arXiv:2211.11052v1, Section 4.
These GPTMini/AdamW / raw AMSGradW experiments and stronger persistence targets are explicit adaptations.

All 2 frozen runs completed 300,000 updates and 1,202 canonical observations. No failed target is omitted.

| Recipe | Final train / held-out | Long confirmation, training s | Tail failures | Persistent tail | Stable grokking |
| --- | ---: | ---: | ---: | --- | --- |
| adamw-short-lr001 | 100.0000% / 100.0000% | 181.81 | 7 | False | False |
| amsgradw-short-lr001 | 100.0000% / 99.9785% | 1595.81 | 10 | False | False |

Each recipe is one initialization on one common calibration split. These are not independent confirmations.
Timing is measured at the first 20-observation joint target confirmation; it does not imply subsequent persistence.
CSV retains actual exposure, losses, optimizer identities, phase flags, costs and memory.
Only recipes meeting the complete phase and final-tail criterion can enter a new independent confirmation plan.
Neighbor/gradient associations do not establish a cause, an internal algorithm, or causality between grokking and double descent.

Eligible calibration recipes: None.

Next action: `preserve_negative_results_and_freeze_one_justified_mechanism_change_without_relaxing_criteria`.
