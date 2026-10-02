# Complete lower-rate AdamW calibration

Native AdamW at learning rate 0.0003 completes all 150,000 updates with final
train/held-out accuracy 100%/100%. It exhibits the required delayed-generalization
phase, followed by three sampled failure episodes and recovery. Eight of the
201 frozen final-window observations fail 99%, so the original persistence and
stable-grokking gates remain negative. This is one exploratory seed/split.

The [complete individual archive](../../baselines/adamw_stability_mod193_fraction25_lr0003_seed0_data0_20261002/summary.json)
and [whole-stage archive](../../baselines/adamw_stability_mod193_fraction25_lower_rate_calibration_20261002/REPORT.md)
verify without PyTorch. The [audit](lower-rate-result-validation.json) checks
the original measurements and raw logs, exhaustive canonical history, every
nested archive file, local checkpoint SHA256, all 31 frozen core/manuscript
fingerprints and both pinned launchers. Both actual trainer/worker processes
have terminated and their sessions were consumed. No restart is required.

Only rate differs from the preceding complete quarter-split calibration.
Retain mod 193, train fraction 25%, seeds 0/0, 9,264 training / 27,792 held-out
pairs, width 128, two layers, four heads and 436,104 parameters. Native AdamW
uses betas (0.9, 0.98), epsilon 1e-8, bias correction, all-parameter decay 0.1,
ten-update warmup and no clipping. The experiment adapts *Convexifying
Transformers*, Section 4; it measures novel fixed-length operand pairs.

First train 99% is at 1,250. The qualifying low-held-out plateau spans
4,500–27,250 with 92 observations. First held-out 99% is at 95,750, and the
first 20 consecutive joint targets confirm at 100,500. Observed confirmation
costs are 2,765.91 training / 3,375.93 wall seconds. The preserved
[positive phase prefix](first-long-confirmation-lower-rate.md) remains valid.
Eligible persistent-target timing has support zero.

All eight failed canonical points occur at 105,250, 105,500, 109,000, 110,750,
111,000, 111,250, 111,500 and 111,750. Each has numeric-answer failures at both
immediate neighbors and EOS 100%. All eight canonical batches are full 512;
the next update 110,751 uses the 48-example epoch tail, while every other
failure-neighborhood batch is full. This does not identify their cause.
Minimum canonical tail held-out accuracy is 91.5299% at 109,000.

The [complete recovery metrics](lower-rate-recovery-metrics.json) count adjacent
failed canonical points as one episode, independently of the unchanged gate:

| First / last failed update | Failed observations | Minimum joint accuracy | First sampled recovery | Sampled recovery interval |
| --- | ---: | ---: | ---: | ---: |
| 105,250 / 105,500 | 2 | 97.4813% | 105,750 | 500 updates |
| 109,000 / 109,000 | 1 | 91.5299% | 109,250 | 250 updates |
| 110,750 / 111,750 | 5 | 96.9272% | 112,000 | 1,250 updates |

| Fixed final-window updates | Failed / observed targets | Failed fraction | New episode onsets |
| --- | ---: | ---: | ---: |
| [100,000, 110,000) | 3 / 40 | 7.5% | 2 |
| [110,000, 120,000) | 5 / 40 | 12.5% | 1 |
| [120,000, 130,000) | 0 / 40 | 0% | 0 |
| [130,000, 140,000) | 0 / 40 | 0% | 0 |
| [140,000, 150,000] | 0 / 41 | 0% | 0 |

All 153 canonical observations at 112,000–150,000 attain joint 99%, a sampled
span of 38,000 updates; 38,250 updates have elapsed since the last failed
observation. The three latest full windows have no failures. The initial failed
fraction rises before declining, while episode onset counts decline. These
single-history measurements do not establish statistical monotonicity or
permanent recovery. Joint and held-out episode/window counts agree in this case.

Complete instrumentation retains 601 canonical evaluations, 1,200 immediate
neighbors, 1,800 full tensor diagnostic records and all 150,000 pre-update
gradient norms. The largest gradient norm is 1,059.0894 at update 121,171 on
a full batch; no accuracy is measured at that exact update. A large gradient
alone cannot substitute for a failed target. Full costs are 4,109.95 training,
5,020.35 wall and 44.62 diagnostic seconds; peak CUDA allocation/reservation
is 119,654,400 / 157,286,400 bytes on the recorded GTX 1050.

All three original standalone PNG and actual PDF figures were visually reviewed.
The separate [recovery PNG](lower-rate-recovery-curves/recovery.png) /
[PDF](lower-rate-recovery-curves/recovery.pdf) also show all 601 unsmoothed points,
the exact window denominators, episode counts and recovery intervals. Their
source/measurement hashes and CSV are retained without changing original figures.

The user's subsequent cap is a total of 300,000 updates per run. The next
action is a separately frozen continuation of this complete checkpoint, changing
only the total step budget and preserving native optimizer and sampling state.
Keep this original negative 150,000-update result intact. The extended final
window is 250,000–300,000, specified before any additional scientific updates.
The continuation is an exploratory posthoc budget extension; six fresh full-
budget model seeds 4/5/6 crossed with data seeds 2/3 remain necessary after a
passing calibration, followed by architecture controls and complementary tasks.
