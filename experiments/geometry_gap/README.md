# Early settling and active-answer certificates

The user asks to close the discovery/persistence and certificate gaps after
the eight-seed .01 study. The [frozen protocol](PROTOCOL.md) separates two
changes: reduce the large learning rate before the late measurement window,
and fit the rank-limited readout basis to observed answer classes while still
bounding every rival class. All other training and numerical controls remain.

| Intervention | Changed | Fixed comparison |
| --- | --- | --- |
| Early settling | .01 to .0001 during updates 1000–4000, then hold until 8000 and cool to 1e-8 | Same eight seeds, 10000 updates and all-class certificates against .001 and .01 |
| Active-answer basis | Readout rows used to choose reconstruction directions | Same saved states, training folds, rank budgets and all 97 rival margins |

Earlier LR reduction improves sampled persistence in every paired seed.
The training cohort is complete and independently verified; the separate
active-answer certificate comparison is in progress. Generated evidence is
ignored and preserved outside temporary worktrees. This experiment does
not assert a global training or convergence theorem.

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

## Reproduce

From the repository root, reproduce the schedule arm with native tools and
a CUDA-enabled execution context. Each training run needs a fresh output:

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
  experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/verify" experiments/geometry_gap/runs/settle-01 \
  experiments/class_geometry_lr/runs/lr01-01/report.json
"$CARGO_TARGET_DIR/release/analyze" experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_wsd/runs/audit-02 \
  experiments/geometry_gap/runs/settle-01
```

Once the new cohort is complete, compare the two certificate bases on all
744 saved states, then independently reread and verify the emitted bounds:

```sh
"$CARGO_TARGET_DIR/release/active-answers" experiments/geometry_gap/runs/active-01 \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/active-answers" verify experiments/geometry_gap/runs/active-01
```

The original all-class margins must reproduce byte for byte before each
active-basis measurement. Bounds and fitting metadata are saved before query
outcomes are read for scoring. The second pass checks the emitted files
against direct FP64 and recorded GPU margins, without calling the basis
constructor. Its runtime is charged to the same 1800-second postprocessing
cap. The signed-transport bound and OG3 in
[`review/flat_spectrum.md`](../../review/flat_spectrum.md) already apply to
an arbitrary computed basis with its residual and defect terms retained.

To reproduce the actual-motion diagnostic:

```sh
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
