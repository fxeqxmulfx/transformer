# Completed unchanged-rate stability control

`short-lr001` completed its full 150,000-update budget on the GTX 1050.
It fails the prospectively frozen stable-grokking criterion: no required
memorization plateau precedes first generalization, and the final tail
contains ten joint-target failures. The final rebound does not change this
outcome. This is one exploratory initialization/split, not independent
confirmation or completion of the four-recipe calibration.

The setting is the mod-97 division adaptation of *Convexifying Transformers*,
Section 4: raw AMSGradW, softmax GPTMini, width 128, two layers, four heads,
423,816 parameters, model/data seed 0, learning rate 0.001, decay 0.1, ten-update
warmup, batch 512 with a 48-example epoch tail, and exhaustive 4,656/4,656
train/held-out equations. Both numeric answer and EOS must be correct.
The optimizer recurrence and model match the frozen unchanged-model control.
The phase and persistence definitions remain those in [STABILITY.md](../../STABILITY.md).

| Measurement | Complete result |
| --- | --- |
| Required low-held-out memorization plateau | Absent |
| First long joint 99% confirmation | Onset 52,750; confirmed 57,500 |
| Training / wall time at long confirmation | 1,720.56 / 1,808.46 seconds; support 1/1 |
| Joint failures in final 50,000 updates | 10/201 canonical observations |
| Minimum held-out accuracy in that tail | 58.14% |
| Persistent final performance / stable grokking | False / false |
| Final train / held-out complete RHS accuracy | 100% / 99.9785% |
| Final incorrect held-out equations | 1 of 4,656 |
| Full-budget training / wall time | 4,481.67 / 4,708.09 seconds |
| Separately measured diagnostic time | 10.27 seconds |
| Peak allocated / reserved CUDA bytes | 119,166,464 / 146,800,640 |
| Canonical / neighbor / diagnostic observations | 601 / 1,200 / 1,800 |
| Actual examples consumed | 69,840,000 (15,000 corpus passes) |

Failing tail steps are 102,250, 107,250, 112,250, 117,250, 123,000, 128,000,
132,500, 137,750, 143,750, and 148,500. The tail minimum occurs at 132,500.
Legacy two-observation held-out confirmation is earlier, at 6,000; it does
not certify the stronger persistence target.

After that legacy confirmation, 64 canonical held-out observations fall below
99%. All 64 fail the numeric answer target; EOS remains at target. All failures
have both immediate neighbor probes. In 63 cases both neighbors also fall
below 99%. At 98,500 the preceding score is 99.055%, the canonical score
98.411%, and the following score 97.766%. Zero failures are isolated to the
canonical observation, and zero recover by the next update. These categories
are descriptive and can overlap.

The largest sampled post-confirmation gradient norm is 53.548 at update 85,750,
with a 48-example batch and a joint update/weight-norm ratio of 0.03854.
The inverse temperatures drift from 5.657 at the first sampled update to
1.0001–1.0574 at the final update. These associations do not identify a cause:
the observations sample specified updates, canonical evaluations coincide with
short epoch tails, and lower-rate/full-batch interventions have not completed.
Neighbor failures weaken an explanation restricted to a single canonical
update but do not rule out a longer-lasting sampling-policy effect.

The [portable archive](../../baselines/amsgradw_stability_short_lr001_seed0_data0_20261002)
contains unchanged measurements and diagnostic/probe logs, three CSV tables,
the derived assessment, all supported failure neighborhoods, source/data/plan
fingerprints, artifact checksums, and standalone
[overview](../../baselines/amsgradw_stability_short_lr001_seed0_data0_20261002/plots/stability-overview.png)
and [neighbor curves](../../baselines/amsgradw_stability_short_lr001_seed0_data0_20261002/plots/collapse-neighbors.png)
in PNG/PDF. Checkpoint weights and optimizer buffers remain local; their SHA256
is recorded. The archive was verified with system Python without importing
PyTorch, and both PNG figures were visually inspected. See
[verification metadata](short-lr001-validation.json).

Next, retain the complete independently frozen `wrap-lr001`, `short-lr0003`,
and `short-lr0001` budgets and assemble the all-recipe comparison. Select a
passing calibration only after that comparison, or preserve a negative grid
and freeze a justified controlled follow-up. Independent confirmation and the
architecture/mechanics comparisons remain required and have not begun.
