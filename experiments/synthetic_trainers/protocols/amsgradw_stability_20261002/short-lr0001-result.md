# Completed lowest-rate stability control

`short-lr0001` completed all 150,000 updates and fails the frozen stable-grokking
criterion. It has the required memorization plateau, but never reaches 99%
held-out accuracy. All 201 canonical observations in the final 50,000 updates
fail the joint target. Its final train fit also regresses. This is a complete
negative calibration, not independent confirmation.

The setting is the mod-97 division adaptation of *Convexifying Transformers*,
Section 4: raw AMSGradW, softmax GPTMini, width 128, two layers, four heads,
423,816 parameters, model/data seed 0, learning rate 0.0001, decay 0.1,
ten-update warmup, short-final batching, and exhaustive 4,656/4,656 equations.
Both numeric answer and EOS must be correct. Learning rate is the declared
intervention against the 0.001 control. Its lower value also reduces raw
per-update shrinkage, whose multiplier is `1 - learning_rate * weight_decay`;
this control does not isolate shrinkage from the gradient step.

| Measurement | Complete result |
| --- | --- |
| Required memorization plateau | Updates 1,000–61,250; 242 observations |
| Held-out 99% crossing / long joint confirmation | Absent / absent; timing support 0 |
| Maximum canonical held-out accuracy | 67.1607%, at update 149,750 |
| Joint failures in final 50,000 updates | 201/201 canonical observations |
| Minimum held-out accuracy in that tail | 17.5258%, at update 150,000 |
| Persistent final performance / stable grokking | False / false |
| Final train / held-out complete RHS accuracy | 31.2715% / 17.5258% |
| Final incorrect held-out equations | 3,840 of 4,656 |
| Full-budget training / wall time | 4,523.19 / 4,747.52 seconds |
| Separately measured diagnostic time | 10.24 seconds |
| Peak allocated / reserved CUDA bytes | 119,166,464 / 146,800,640 |
| Canonical / neighbor / diagnostic observations | 601 / 1,200 / 1,800 |
| Actual examples consumed | 69,840,000 (15,000 corpus passes) |

Every final-tail held-out failure includes a numeric-answer target failure.
EOS remains above the 99% target throughout that window; its minimum is
99.9785% (one EOS error). Thus EOS-only scoring does not explain the failure.
Missing long-confirmation timing remains null with support zero.

At update 149,750, train/held-out accuracy is 100%/67.1607%. The immediate
pre-final probe at 149,999 already regresses to 29.9828%/16.6452%, after a full
512-example batch. The final 48-example batch yields 31.2715%/17.5258%.
The regression is therefore not confined to the final canonical observation;
earlier batch-size effects and the cause remain unresolved. No after-budget
probe was measured. At the final sample, the pre-update gradient norm is
48.4098, the actual update/weight-norm ratio is 0.002694, and post-update
inverse temperatures range from 1.5267 to 1.5775. These are sampled associations.

The post-confirmation-neighborhood plot is empty because no held-out target
confirmation occurred. It does not mean the run has no failures. Raw probes
retain the terminal regression and all other scheduled observations.

The [portable archive](../../baselines/amsgradw_stability_short_lr0001_seed0_data0_20261002)
contains unchanged complete measurements and logs, three CSV tables, assessment,
fingerprints, checkpoint SHA256, artifact checksums, and standalone PNG/PDF
[overview](../../baselines/amsgradw_stability_short_lr0001_seed0_data0_20261002/plots/stability-overview.png)
and [post-confirmation neighborhoods](../../baselines/amsgradw_stability_short_lr0001_seed0_data0_20261002/plots/collapse-neighbors.png).
Both figures were visually inspected; the archive was verified without PyTorch.
See [verification metadata](short-lr0001-validation.json).

The original four-recipe calibration is now complete and negative. Preserve
all four trajectories and their comparison before freezing a justified new
control under the unchanged criterion in [STABILITY.md](../../STABILITY.md).
Independent confirmation, architecture improvement, and complementary mechanics
remain required later stages; no scientific confirmation has started.
