# Completed midpoint-rate stability calibration

The full `short-lr0002` budget is a negative result: no required low-held-out
memorization plateau precedes generalization, and 150 of the 201 final-tail
canonical observations fail the unchanged joint 99% criterion. All canonical
observations from update 112,750 through 150,000 fail. The final rebound to
88.6598% held-out accuracy does not establish persistence.

This is the mod-97 division adaptation of *Convexifying Transformers*, Section
4, using softmax GPTMini and raw AMSGradW. The frozen recipe retains width
128, two layers, four heads, 423,816 parameters, initialization/data seed 0,
50% exhaustive split, decay 0.1, ten-update warmup, short-final batches, and
150,000 updates. Learning rate 0.0002 is the only declared change from the
original 0.001 control. The score requires both numeric answer and EOS.

| Measurement | Complete result |
| --- | --- |
| Required memorization plateau | Absent |
| First canonical held-out 99% crossing | Update 93,250; 99.0120% |
| First long joint confirmation | Updates 95,750–100,500 |
| Training / wall seconds at long confirmation | 2,875.92 / 3,026.86 |
| Joint failures in final 50,000 updates | 150/201 canonical observations |
| Minimum held-out accuracy in that tail | 0.7947%; first occurs at 112,750 |
| Persistent final performance / stable grokking | False / false |
| Final train / held-out accuracy | 100% / 88.6598% |
| Final incorrect held-out equations | 528 of 4,656 |
| Full-budget training / wall seconds | 4,318.45 / 4,543.32 |
| Separately measured diagnostic seconds | 9.57 |
| Peak allocated / reserved CUDA bytes | 119,166,464 / 146,800,640 |
| Canonical / neighbor / diagnostic observations | 601 / 1,200 / 1,800 |
| Actual examples consumed | 69,840,000 (15,000 corpus passes) |

After legacy confirmation at 94,500, 152 canonical held-out failures occur.
All fail numeric answers; none fails EOS. Of these, 149 remain below target
at both immediate neighbors, none is isolated to its canonical observation,
and one recovers by the next update. One terminal failure lacks an after-budget
probe, so the neighborhood figure contains 151 supported triplets. Categories
can overlap. These observations do not identify the cause.

The [first-collapse inspection](midpoint-first-tail-collapse.md) retains the
six complete tensor samples bracketing the first tail failure. The rise of the
raw second-moment norm between unprobed updates suggests large intervening
gradients under the stated recurrence and numerical assumptions. It is not a
direct measurement of those gradients or a causal explanation. The next paired
optimizer calibration will record the already-computed gradient norm at every
update on both controls; this changes instrumentation and measured host cost,
so it requires a fresh frozen plan and fresh paired timing controls.

The [individual portable archive](../../baselines/amsgradw_stability_short_lr0002_seed0_data0_20261002)
and [complete single-recipe comparison](../../baselines/amsgradw_stability_midpoint_calibration_20261002/REPORT.md)
retain complete histories, diagnostics, neighbor probes, CSV tables, unchanged
plan/source/manuscript fingerprints, checkpoint SHA256, and PNG/PDF figures.
All three distinct PNG figures were visually inspected. Both archives were
verified with system Python without importing PyTorch; all 19 frozen source
hashes and both local manuscript hashes still match. See the
[verification metadata](midpoint-result-validation.json).

All five fresh raw AMSGradW calibrations are negative under the unchanged
criterion. On 2026-10-02 the user explicitly selected AdamW as the primary
optimizer for the continuing stable-grokking benchmark. Preserve these raw
AMSGradW results and use a fresh paired raw control when comparing optimizers.
Historical AdamW has fewer sampled tail failures, but still fails persistence;
stability must be demonstrated prospectively, not assumed. Independent frozen
confirmations, architecture improvement, and complementary mechanics remain
required later stages.
