# Completed full-batch stability control

`wrap-lr001` completed its full 150,000-update budget on the GTX 1050.
It fails the prospectively frozen stable-grokking criterion: no required
memorization plateau precedes first generalization, and 26 final-tail
observations fail the joint target. The final 100% rebound does not pass
persistence. This is one exploratory initialization/split, not independent
confirmation or completion of the four-recipe calibration.

The setting is the mod-97 division adaptation of *Convexifying Transformers*,
Section 4: raw AMSGradW, softmax GPTMini, width 128, two layers, four heads,
423,816 parameters, model/data seed 0, learning rate 0.001, decay 0.1,
ten-update warmup, and exhaustive 4,656/4,656 train/held-out equations.
Both numeric answer and EOS must be correct. The sole declared intervention
against `short-lr001` is full 512-example batches across shuffled epochs;
this changes batch exposure while retaining the optimizer recurrence and model.
The criterion remains [STABILITY.md](../../STABILITY.md).

| Measurement | Complete result |
| --- | --- |
| Required low-held-out memorization plateau | Absent |
| First long joint 99% confirmation | Onset 24,250; confirmed 29,000 |
| Training / wall time at long confirmation | 854.82 / 898.16 seconds; support 1/1 |
| Joint failures in final 50,000 updates | 26/201 canonical observations |
| Minimum held-out accuracy in that tail | 1.9759%, at update 110,500 |
| Persistent final performance / stable grokking | False / false |
| Final train / held-out complete RHS accuracy | 100% / 100% |
| Final incorrect held-out equations | 0 of 4,656 |
| Full-budget training / wall time | 4,530.51 / 4,754.69 seconds |
| Separately measured diagnostic time | 10.17 seconds |
| Peak allocated / reserved CUDA bytes | 119,166,464 / 146,800,640 |
| Canonical / neighbor / diagnostic observations | 601 / 1,200 / 1,800 |
| Actual examples consumed | 76,800,000 (16,494.85 corpus passes) |

The original short-tail control consumes 69,840,000 examples (15,000 passes)
in the same number of updates. The wrap intervention consumes 9.9656% more
examples. Its first long target event costs less measured time in this paired
calibration, while both complete trajectories fail phase and persistence.
First-target timing therefore does not establish a stable benchmark or a
repeatable speedup. These two mechanisms are not averaged into a seed statistic.
Both lower-rate controls still need their complete independently frozen budgets.

After legacy held-out confirmation at 24,500, 40 canonical observations fall
below 99%. All 40 fail the numeric answer target; none fails the EOS target.
Every failure has both immediate neighbor probes. Both neighbors stay below
target in 37 cases; two failures are isolated at their canonical observation,
and three recover by the next update. Categories can overlap. All actual
training batches contain 512 examples, including batches ending at an epoch
boundary. Post-confirmation instability therefore occurs under full batches
as well; this intervention does not identify its cause or measure all
between-observation behavior.

At the final-tail minimum, the sampled pre-update gradient norm is 36.838,
and the joint actual update/weight-norm ratio is 0.04046. Learned inverse
temperatures at that post-update sample range from 1.0015 to 1.0169.
Norms, moments, losses, and temperatures are sampled associations, not a
causal account. The report retains each failing neighborhood and all raw
per-tensor observations for further analysis.

The [portable archive](../../baselines/amsgradw_stability_wrap_lr001_seed0_data0_20261002)
contains unchanged measurements and diagnostic/probe logs, three CSV tables,
the derived assessment, source/data/plan fingerprints, artifact checksums,
and standalone [overview](../../baselines/amsgradw_stability_wrap_lr001_seed0_data0_20261002/plots/stability-overview.png)
and [neighbor curves](../../baselines/amsgradw_stability_wrap_lr001_seed0_data0_20261002/plots/collapse-neighbors.png)
in PNG/PDF. Checkpoint weights and optimizer buffers remain local, with their
SHA256 recorded. The archive was verified with system Python without importing
PyTorch, and both PNG figures were visually inspected. See
[verification metadata](wrap-lr001-validation.json).

Next, finish `short-lr0003` and `short-lr0001` and assemble the complete
four-recipe comparison. Passing calibration remains required before freezing
independent confirmation; architecture and complementary mechanics remain
later required stages. Keep all negative results and the existing criterion.
