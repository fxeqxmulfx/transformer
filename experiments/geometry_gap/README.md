# Early settling and active-answer certificates

The user asks to close the discovery/persistence and certificate gaps after
the eight-seed .01 study. The [frozen protocol](PROTOCOL.md) separates two
changes: reduce the large learning rate before the late measurement window,
and fit the rank-limited readout basis to observed answer classes while still
bounding every rival class. All other training and numerical controls remain.

| Intervention | Changed | Fixed comparison |
| --- | --- | --- |
| Early settling | .01 to .0001 during updates 1000–4000, then hold until 8000 and cool to 1e-8 | Same eight seeds, 10000 updates and all-class certificates against .001 and .01 |
| Active-answer basis | Readout rows used to choose reconstruction directions | Same saved states, training folds, rank budgets and all 97 classes |

Both comparisons are complete and independently verified. Together they
raise mean final rank-32 coverage from 98.015% to 98.504% and the range of
per-seed late coverage minima from 16.07–22.61% to 98.11–98.66%. Correct
predictions left uncertified fall from 608 to 403; incorrect predictions
rise from 139 to 160. Generated evidence is ignored and preserved outside
temporary worktrees. The global training and convergence question remains open.

## Training findings

All eight runs complete 10000 updates in 866.73 seconds. With the original
all-class rank-32 certificate, minimum coverage across the 21 late probes
rises to 97.92–98.55%, versus 16.07–22.61% with the long .01 plateau and
89.95–94.69% with the .001 recipe. All eight now retain at least 99.51%
accuracy at every late probe, versus zero .01 seeds retaining 99%.

The tradeoff is final accuracy: the mean is 99.575%, down from 99.631% for
the long .01 plateau and close to 99.567% at .001. Seven paired final
accuracies fall versus .01 and one is unchanged. Mean final rank-32 coverage
improves from 98.015% to 98.324%; six individual pairs improve and two fall.
No seed reaches 99% coverage with the original certificate. The schedule
substantially improves persistence without eliminating the certificate gap.

All entries below are percentages. Every late probe is retained, including
the .0001 plateau; these are sampled bounds, not claims about unmeasured steps.

| Seed | Final accuracy | Final rank-32 coverage | Minimum late accuracy | Minimum late rank-32 coverage |
| --- | ---: | ---: | ---: | ---: |
| 7 | 99.617 | 98.406 | 99.617 | 98.300 |
| 11 | 99.532 | 98.363 | 99.532 | 98.321 |
| 1 | 99.617 | 98.321 | 99.617 | 98.321 |
| 3 | 99.532 | 98.618 | 99.511 | 98.555 |
| 5 | 99.596 | 98.257 | 99.596 | 98.108 |
| 13 | 99.554 | 98.151 | 99.554 | 98.023 |
| 17 | 99.532 | 98.087 | 99.511 | 97.917 |
| 19 | 99.617 | 98.385 | 99.617 | 98.151 |

All eight first reach 90% sampled accuracy at step 250. First 90% rank-32
coverage and 99% accuracy now occur at 2000–3000, before the late window.
The original .01 arm first reaches 90% coverage at 8500–9250. At the eight
new endpoints, 160 predictions are incorrect and 471 correct predictions
remain uncertified, out of 37640 seed-query evaluations. The .01 counts are
139 and 608; the .001 counts are 163 and 601. These cohorts are separate
from the older 24-state forecast certificate audit.

The independent training verifier passes 80000 updates, 248 probes,
18667456 margins and all eight paired initial states. It finds zero false
certificates or bound violations. All 32 schedule-boundary GPU preflights
and 240 actual optimizer checks match the FP32 reference at zero ulps.
The largest norm discrepancy is `7.08e-8`; feature/logit discrepancies are
at most `4.13e-5`/`2.22e-5`, within unchanged allowances. Maximum scaled
energy/share closure errors are `6.41e-14`/`2.34e-14`.

An additional trace diagnostic confirms nonzero class-contrast motion on
all 32000 updates in the declared hold interval (updates 4001–8000 across
eight seeds). The executed FP32 rate is `0.00009999999747378752` throughout.
The improved sampled persistence therefore occurs while parameters continue
to change at a nonzero learning rate.

Final active energy share is 69.95–74.51%. Mean cumulative share credits
(percentage points) are isotropic readout +44.85, adaptive readout +199.11,
tied encoder -170.72, moment memory -32.58 and total quadratic -1.87;
pure-decay radial credit is zero and rounding is negligible. These remain
opposing finite contributions, not a monotone discovery mechanism.

The schedule was measured from commit `1e9d64f4`, source fingerprint
`647d91fefa737073a92b50735255c6a33cacad1e272a779b8601d762d0c12754`.
Its measured executable is preserved separately from later certificate-only
source changes. The frozen protocol is `7d99841f`.

## Certificate findings

The active-answer basis improves final rank-32 coverage in all 24 paired
endpoint states without changing the trained model or its predictions. Every rival
class remains in the margin bound. The six comparisons below include all
eight seeds and every declared late probe; percentages are coverage of the
4705 validation queries per seed.

| Training schedule | Basis | Mean final coverage | Range of per-seed late minima | Correct but uncertified at endpoints |
| --- | --- | ---: | ---: | ---: |
| .001 original WSD | All classes | 97.970% | 89.95–94.69% | 601 |
| .001 original WSD | Active answers | 98.151% | 96.62–97.66% | 533 |
| .01 original WSD | All classes | 98.015% | 16.07–22.61% | 608 |
| .01 original WSD | Active answers | 98.430% | 40.77–54.73% | 452 |
| .01 early settling | All classes | 98.324% | 97.92–98.55% | 471 |
| .01 early settling | Active answers | 98.504% | 98.11–98.66% | 403 |

The basis change alone improves the long .01 plateau, but its late minima
remain 40.77–54.73%. Earlier LR reduction supplies the larger persistence
improvement. Together, the changes remove 205 of 608 correct-but-uncertified
endpoint predictions (33.7%), while mean accuracy falls by 0.056 percentage
points. The combined endpoints contain 37077 certificates, 403 correct but
uncertified predictions and 160 incorrect predictions out of 37640.

The new basis is not uniformly tighter on individual queries. Across the
eight endpoints, it gains/loses 83/15 certificates at .001, 191/35 at .01,
and 95/27 with early settling. All eight seeds improve within each cohort,
but every early-settling trajectory still has 1–4 sampled coverage decreases.
No rank-32 endpoint reaches 99% coverage under either basis. One original .01 seed
(seed 3) briefly reaches 99% with the active basis at step 9750; it does not
retain that level. The native summary retains all 24 paired endpoints,
rank-33/128 controls, threshold crossings and regression counts.

All 744 state comparisons pass, including 28001184 byte-exact reproductions
of original readout margins and 28001184 new margin bounds. A separate pass
rereads the emitted files and checks all new bounds against independent
FP64 and recorded GPU margins. It passes with zero false certificates;
construction also reports zero bound violations. Construction and independent
verification total 1499.62 seconds, within the frozen 1800-second cap.
All numerical allowances are unchanged.

At requested rank 32, achieved ranks range from 20 to 32 across all folds
and states. The largest coefficient norm is 4932.35 and the largest
orthogonality penalty is `6.76e-11`; both remain included in the bounds.
The smallest retained training singular value is `9.71e-5`. This selected
basis does not establish a full span of the class readouts, nor does its
coverage establish a theorem of reliable discovery or maintenance.

The certificate audit was measured from commit `210efa2a`, source fingerprint
`d37558e4c28a3cde3e7a91902c8dadb15ce7383d2a3b144ec0795720f44075e2`.
Its generated evidence is in `runs/active-01/`; the training evidence is in
`runs/settle-01/`. Native tests pass for the producer (20), training verifier
(3), certificate tools (7), and shared original producer/verifier (16/1).
Formatting and Clippy with warnings denied pass.

## Reproduce

From the repository root, reproduce the schedule arm with native tools and
a CUDA-enabled execution context. The commands use fresh `-02` outputs;
the measured `-01` outputs remain preserved:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --bins
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" preflight
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/class_geometry_lr/runs/lr01-01/report.json \
  experiments/geometry_gap/runs/settle-02
"$CARGO_TARGET_DIR/release/verify" experiments/geometry_gap/runs/settle-02 \
  experiments/class_geometry_lr/runs/lr01-01/report.json
"$CARGO_TARGET_DIR/release/analyze" experiments/geometry_gap/runs/settle-02
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-02
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_wsd/runs/audit-02 \
  experiments/geometry_gap/runs/settle-02
```

Once the new cohort is complete, compare the two certificate bases on all
744 saved states, then independently reread and verify the emitted bounds:

```sh
"$CARGO_TARGET_DIR/release/active-answers" experiments/geometry_gap/runs/active-02 \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-02
"$CARGO_TARGET_DIR/release/active-answers" verify experiments/geometry_gap/runs/active-02
```

The original all-class margins must reproduce byte for byte before each
active-basis measurement. Bounds and fitting metadata are saved before query
outcomes are read for scoring. The second pass checks the emitted files
against direct FP64 and recorded GPU margins, without calling the basis
constructor. Its runtime is charged to the same 1800-second postprocessing
cap. The signed-transport bound and OG3 in
[`review/flat_spectrum.md`](../../review/flat_spectrum.md) already apply to
an arbitrary computed basis with its residual and defect terms retained.

To regenerate the complete paired summary and actual-motion diagnostic
from the preserved measurements:

```sh
jq -f experiments/geometry_gap/summary.jq \
  experiments/geometry_gap/runs/active-01/report.json \
  > experiments/geometry_gap/runs/active-01/summary.json
jq -n -f experiments/geometry_gap/hold_updates.jq \
  experiments/geometry_gap/runs/settle-01/seed*/updates.jsonl
```

The native plot generator exports SVG and PNG. Lines connect the 31 sampled
probes; shaded bands show the range across all eight seeds. The accuracy
panel explicitly uses a 90–100% range, while coverage uses 0–100%.

```sh
jq -sr -f experiments/geometry_gap/trajectories.jq \
  experiments/class_geometry_wsd/runs/audit-02/report.json \
  experiments/class_geometry_lr/runs/lr01-01/report.json \
  experiments/geometry_gap/runs/settle-01/report.json \
  > experiments/geometry_gap/runs/settle-01/trajectories.tsv
gnuplot -c experiments/geometry_gap/trajectories.gp \
  experiments/geometry_gap/runs/settle-01/trajectories.tsv \
  experiments/geometry_gap/runs/settle-01/trajectories
```
