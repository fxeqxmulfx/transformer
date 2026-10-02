# Complete mod-193 quarter-split AdamW result

The prospectively frozen `adamw-mod193-fraction25-lr001` calibration finishes
all 150,000 updates. Final exhaustive train / held-out complete RHS accuracy
is 100%/100%. The required delayed-generalization phase is observed, but
three failures in the unchanged final window refute stable grokking. This
recipe cannot open independent confirmation or architecture selection.

The [individual archive](../../baselines/adamw_stability_mod193_fraction25_lr001_seed0_data0_20261002)
and [complete calibration comparison](../../baselines/adamw_stability_mod193_fraction25_calibration_20261002)
retain all 601 canonical observations, 1,200 exhaustive neighbors, 1,800
full-tensor samples and 150,000 pre-update gradient norms, with every raw log,
CSV, source/data fingerprint and PNG/PDF figure. The comparison has no passing
recipe and its independent-confirmation gate is false.

| Frozen gate or observation | Complete measured result |
| --- | --- |
| Pre-target memorization plateau | 4,500–11,500; 29 observations |
| First held-out 99% | 20,000; train / held-out 100%/99.5862% |
| First 20-observation joint confirmation | 20,000–24,750 |
| Confirmation event cost | 678.96 training / 830.95 wall seconds |
| Final-window joint target | 198/201 observations pass |
| Persistent final performance / stable grokking | false / false |
| Final held-out answer CE / EOS CE | 1.02198e-7 / 0 nats |

All failed final-window observations follow full 512-example batches:

| Update | Train accuracy | Held-out accuracy | EOS accuracy, both splits |
| --- | ---: | ---: | ---: |
| 115,000 | 92.7461% | 91.6271% | 100% |
| 119,250 | 98.9421% | 98.7766% | 100% |
| 139,000 | 55.7211% | 48.9206% | 100% |

There are five held-out failures after the legacy two-observation confirmation
at 20,250, all concerning numeric answers. No EOS accuracy falls below 99%;
this threshold count does not assert zero EOS mistakes at every observation.
Four failures remain below target at both immediate neighbors; one recovers
at the next update. None is isolated at the canonical observation and no
neighbor support is missing. Those categories can overlap. The largest dense
gradient norm is 1,786.2872 at 138,783 on a full batch. Accuracy was not
measured at that exact update; its association with later failure does not
identify a causal trigger.

The [overview](../../baselines/adamw_stability_mod193_fraction25_lr001_seed0_data0_20261002/plots/stability-overview.png),
[failure neighbors](../../baselines/adamw_stability_mod193_fraction25_lr001_seed0_data0_20261002/plots/collapse-neighbors.png)
and [complete comparison](../../baselines/adamw_stability_mod193_fraction25_calibration_20261002/plots/calibration-comparison.png)
were visually reviewed; their corresponding PDFs are preserved. No points
or failed budgets are dropped. [Validation](fraction25-result-validation.json)
recomputes both complete archives without PyTorch, checks exact raw bytes,
identical nested copies, all 31 unchanged frozen fingerprints and both earlier
phase/failure prefixes against the final measurements. The final checkpoint
remains in the ignored source directory with its recorded SHA256.

The run costs 4,193.88 training / 5,106.13 wall seconds, including 44.98
diagnostic seconds. Peak CUDA allocation/reservation is 119,654,400 /
157,286,400 bytes. Event timing is observed; successful persistent-target
timing is ineligible. CPU preparations and reporting overlapped portions of
training, so these descriptive costs do not isolate a rate, fraction or
optimizer speedup. One initialization and calibration split provide no
repeatability estimate or mean/sample-SD comparison. This is fixed-length
novel operand-pair generalization, not length transfer or an identified algorithm.

This is an explicit *Convexifying Transformers*, Section 4 adaptation using
the [unchanged prospective plan](fraction25-plan.json). The reduced split
provides the previously missing phase, while persistence remains negative.
Both trainer PID 123201/session 61726 and archive worker PID 123310/session
95343 finish with exit code 0; their terminal handles are consumed. Do not
restart or poll them. The [prepared lower-rate control](lower-rate-preparation.md)
can now be frozen as a fresh calibration, retaining the same task, source,
environment and criterion. Six fresh confirmations, targeted architecture
comparison and scientific complementary tasks remain outstanding.
