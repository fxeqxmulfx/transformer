# Complete 300,000-update native AdamW continuation

Native AdamW completes the user's total cap of 300,000 updates with final
train/held-out accuracy 100%/100%. Delayed generalization remains observed,
but two of the 201 frozen final-window measurements fail joint 99%.
Persistence and stable-grokking eligibility remain false. All six sampled
post-long-onset failure episodes recover before the budget ends. Recovery is
measured separately and does not erase the frozen failures.

The [individual archive](../../baselines/adamw_stability_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002/summary.json)
and [complete comparison](../../baselines/adamw_stability_mod193_fraction25_budget300k_calibration_20261002/REPORT.md)
verify without importing Torch. The [validation](budget300k-result-validation.json)
retains the verification program, exact original log prefixes, all 601 original
canonical records, byte-identical individual/nested archives, every gradient,
all 39 unchanged frozen files and native checkpoint provenance. Trainer PID
135520/session 64768 and archive worker PID 135597/session 37682 both finish
with exit code zero; their terminal sessions are consumed and processes absent.

Only the total budget changed from the original 150,000-update lower-rate run.
The additional updates and 250,000–300,000 final window were frozen before
continuation, with native model/optimizer steps, RNG, permutation and sampling
cursor restored. Preserve the [original negative result](lower-rate-result.md)
and checkpoint SHA256
`875efd8e6c1a07bc250f932391ca8c41ddcf026a2f14eb76093e5ee777a58794`.
This is one exploratory posthoc budget extension, not a fresh independent
confirmation. No run may exceed the total 300,000-update cap.

The unchanged configuration is mod 193, training fraction 25%, seeds 0/0,
9,264 training and 27,792 exhaustive held-out equations, width 128, two layers,
four heads and 436,104 parameters. Native AdamW retains betas (0.9, 0.98),
epsilon 1e-8, bias correction, all-parameter decay 0.1, rate 0.0003, ten-update
warmup, no clipping, batch 512 and the original 48-example epoch tail. The task
adapts *Convexifying Transformers*, Section 4, and measures fixed-length novel
operand pairs. The complete training exposure is 146,273,904 examples.

The qualifying memorization plateau remains 4,500–27,250, first held-out 99%
is at 95,750, and long joint confirmation is 95,750–100,500. The observed
confirmation costs remain 2,765.91 training / 3,375.93 wall seconds. Eligible
persistent timing support is zero; there is no eligible speed ratio.

| First / last failed canonical update | Failed observations | Minimum joint accuracy | First sampled recovery | Sampled recovery interval |
| --- | ---: | ---: | ---: | ---: |
| 105,250 / 105,500 | 2 | 97.4813% | 105,750 | 500 updates |
| 109,000 / 109,000 | 1 | 91.5299% | 109,250 | 250 updates |
| 110,750 / 111,750 | 5 | 96.9272% | 112,000 | 1,250 updates |
| 154,000 / 154,000 | 1 | 96.1464% | 154,250 | 250 updates |
| 274,000 / 274,000 | 1 | 49.1796% | 274,250 | 250 updates |
| 275,500 / 275,500 | 1 | 81.7681% | 275,750 | 250 updates |

The failure at 274,000 has train/held-out 51.1982%/49.1796%, with EOS 100%.
Both immediate neighbors fail numerically: held-out 42.5590% at 273,999 and
60.1684% at 274,001. The canonical batch is full 512 and its preceding batch
is the short 48-example tail. At 275,500 train/held-out is 81.9516%/81.7681%
with EOS 100%, immediately after a short batch. Its preceding observation at
275,499 is 100%/100%; the following held-out score is 81.1996%. The two
neighborhoods differ and do not establish a cause. All measurements and
associated tensor/gradient diagnostics remain available.

| Frozen final-window updates | Failed / observed joint targets | Failed fraction | New episode onsets |
| --- | ---: | ---: | ---: |
| [250,000, 260,000) | 0 / 40 | 0% | 0 |
| [260,000, 270,000) | 0 / 40 | 0% | 0 |
| [270,000, 280,000) | 2 / 40 | 5% | 2 |
| [280,000, 290,000) | 0 / 40 | 0% | 0 |
| [290,000, 300,000] | 0 / 41 | 0% | 0 |

The later tail has fewer failed measurements than the preserved original tail
(2/201 versus 8/201), but a deeper worst canonical failure (49.1796% versus
91.5299%). Whole-history 10,000-update windows show no failures from 160,000
through the end of the 260,000–270,000 window, followed by two new episodes
in 270,000–280,000. Onset frequency therefore does not decrease monotonically
along this history. The final target streak is 275,750–300,000: 98 observations
spanning 24,250 updates. These sampled results establish neither permanent
recovery nor a statistical trend across independent seeds.

Complete instrumentation retains 1,201 canonical evaluations, 2,400 immediate
neighbors, 3,600 full tensor diagnostic records and 300,000 pre-update gradient
norms. The largest gradient is 1,727.9810 at 178,641 on a full batch; accuracy
is not measured at that exact update. Total measured costs are 7,935.15 training,
9,747.62 execution wall and 86.91 diagnostic seconds, excluding intervening
downtime. Peak CUDA allocation/reservation is 122,238,464 / 157,286,400 bytes,
the maximum across the original run and all observed continuation segments.

The original overview, neighbors and comparison figures retain the whole
history. Additional [recovery PNG](budget300k-recovery-curves/recovery.png) /
[PDF](budget300k-recovery-curves/recovery.pdf) and
[whole post-onset frequency PNG](budget300k-recovery-series-curves/series.png) /
[PDF](budget300k-recovery-series-curves/series.pdf) retain exact denominators,
episode durations and all twenty full grid windows. The leading partial window
remains in the CSV and JSON, excluded from nominal-width rates. All five
original PNG and actual PDF figures have been visually reviewed; file/source
hashes are retained. No success criterion or original figure was changed.

The user-directed [sparsemax pair](attention-pair-protocol.md) is next: freeze
both fresh full budgets together, commit the manifest, run sparsemax first and
the fresh softmax control second. It remains exploratory while the all-six
benchmark gate is closed. Independent confirmation, repeatable architecture
improvement and complementary mechanics with separate novel-ID/length transfer
remain outstanding; the negative result does not authorize relaxing those gates.
