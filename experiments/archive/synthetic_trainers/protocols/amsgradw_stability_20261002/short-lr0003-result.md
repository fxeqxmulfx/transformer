# Completed lower-rate stability control

`short-lr0003` completed its full 150,000-update budget on the GTX 1050.
It fails the prospectively frozen stable-grokking criterion: the required
low-held-out memorization plateau is absent, and two canonical observations
in the final 50,000 updates fail the joint 99% target. Final train/held-out
accuracy rebounds to 100%/100%. This is one exploratory initialization/split;
independent confirmation and the complete four-recipe comparison remain pending.

The setting is the mod-97 division adaptation of *Convexifying Transformers*,
Section 4: raw AMSGradW, softmax GPTMini, width 128, two layers, four heads,
423,816 parameters, model/data seed 0, decay 0.1, ten-update warmup,
short-final batching, and exhaustive 4,656/4,656 train/held-out equations.
Both numeric answer and EOS must be correct. The sole intervention against
`short-lr001` is lowering the learning rate from 0.001 to 0.0003; model,
optimizer recurrence, split, sampling, observation cadence, full budget,
and the criterion in [STABILITY.md](../../STABILITY.md) remain frozen.
The raw optimizer's decay multiplier is `1 - learning_rate * weight_decay`.
Lowering the rate therefore also reduces shrinkage per update at the same
decay coefficient; this control does not isolate those two contributions.

| Measurement | Complete result |
| --- | --- |
| Required low-held-out memorization plateau | Absent |
| First long joint 99% confirmation | Onset 3,000; confirmed 7,750 |
| Training / wall time at long confirmation | 240.46 / 252.06 seconds; support 1/1 |
| Joint failures in final 50,000 updates | 2/201 canonical observations, at 103,000 and 134,500 |
| Minimum held-out accuracy in that tail | 40.5498%, at update 103,000 |
| Persistent final performance / stable grokking | False / false |
| Final train / held-out complete RHS accuracy | 100% / 100% |
| Final incorrect held-out equations | 0 of 4,656 |
| Full-budget training / wall time | 4,560.76 / 4,785.06 seconds |
| Separately measured diagnostic time | 10.35 seconds |
| Peak allocated / reserved CUDA bytes | 119,166,464 / 146,800,640 |
| Canonical / neighbor / diagnostic observations | 601 / 1,200 / 1,800 |
| Actual examples consumed | 69,840,000 (15,000 corpus passes) |

After legacy held-out confirmation at 3,250, nine canonical observations
fall below 99%: 43,750, 44,000, 44,250, 44,500, 44,750, 45,000, 77,750,
103,000, and 134,500. All nine fail numeric answers; none fails EOS.
Both immediate neighbors remain below the held-out target in every case;
there is no missing neighbor support, isolated canonical failure, or recovery
by the next update. These observations weaken a one-update observation
artifact explanation, without identifying the cause or measuring every update.

At update 103,000, train accuracy is 46.7998%. Held-out accuracy before,
at, and after that update is 40.8935% / 40.5498% / 38.8316%, with 100% EOS
accuracy throughout. The canonical sampled pre-update gradient norm is
17.1718, the actual update/weight-norm ratio is 0.004637, and post-update
inverse temperatures range from 1.0823 to 1.1305. At update 134,500,
held-out accuracy is 97.8737%, with both neighbors also below 99%.
These sampled norms, losses, and temperatures describe associations, not causes.

This lower-rate trajectory has fewer observed target failures than the paired
0.001 short-tail control, and reaches its first long target streak earlier.
Both complete trajectories fail the phase and persistence gate. One paired
calibration does not establish a repeatable speedup, and different recipes
are not averaged into a seed statistic. Finish the 0.0001 control before
selecting any recipe or adaptation; retain all three negative results.

The [portable archive](../../baselines/amsgradw_stability_short_lr0003_seed0_data0_20261002)
contains unchanged measurements and diagnostic/probe logs, three CSV tables,
the derived assessment, source/data/plan fingerprints, artifact checksums,
and standalone [overview](../../baselines/amsgradw_stability_short_lr0003_seed0_data0_20261002/plots/stability-overview.png)
and [neighbor curves](../../baselines/amsgradw_stability_short_lr0003_seed0_data0_20261002/plots/collapse-neighbors.png)
in PNG/PDF. Checkpoint weights and optimizer buffers remain local, with their
SHA256 recorded. The archive passed system-Python verification without importing
PyTorch, and both PNG figures were visually inspected. All 19 frozen calibration
source hashes remain unchanged. See [verification metadata](short-lr0003-validation.json).

Next, finish `short-lr0001` and assemble the complete four-recipe comparison.
Passing calibration remains required before independently frozen confirmation;
architecture and complementary-mechanics comparisons remain required later stages.
