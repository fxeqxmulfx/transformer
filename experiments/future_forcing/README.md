# Future AdamW forcing from current state

Can the current moments and known schedule certify geometry before later
updates are read? The [frozen protocol](PROTOCOL.md) compares a warm-state
moment cone with the existing uniform bound on 144 windows, including
initialization, early learning, the hold, and late cooldown.

The exact-real future-forcing gap is partly closed: 25 Lean theorems bound
every future AdamW direction and parameter displacement using saved moments
and the known schedule, for arbitrary future gradients. The bounds give
useful late-cooldown certificates. They still fail to explain persistence
through the long hold, and they do not prove reliable discovery.

The separate FP32 replay passed all 144 windows and 444,000 future prefixes,
with zero forcing, rounding, parameter-cap, tube, or energy-share violations
under the frozen allowances. This is a check of preserved trajectories, not
fresh-seed confirmation or a universal floating-point proof.

## What changed

For the main model's beta1=.9 and beta2=.95, a checkpoint cone `m^2<=r*v`
propagates as

```
r_next <= (81/95)*r + 1/5
K_h(r) = 19/14 + (r-19/14)*(81/95)^h.
```

The first inequality has an exact nonnegative remainder and an attaining
scalar gradient. The second holds for any future gradient history, including
zero variance. Absolute-time bias corrections and positive epsilon remain
in the actual direction. Summing coordinate cones gives whole-readout and
active-row direction budgets; decay accumulates separately from forcing.
The resulting radius needs no observed future direction or update norm.
The formal continuation agrees with the original AdamW parameter history.

See [FF1–FF4](../../review/flat_spectrum.md#future-forcing-from-checkpoint-moments)
and the modules
[`FutureMomentCone`](../../src/Transformer/Grokking/GradientEvidence/FutureMomentCone.lean),
[`FutureDirectionNorm`](../../src/Transformer/Grokking/GradientEvidence/FutureDirectionNorm.lean),
and [`FutureAdamTube`](../../src/Transformer/Grokking/GradientEvidence/FutureAdamTube.lean).
The sign-symmetry theorems explain why a magnitude envelope alone cannot
select favorable alignment. They do not disprove typical discovery under CE.

## Fixed comparisons

Every cohort uses the same eight seeds `7,11,1,3,5,13,17,19`, initialization,
data split, training order, .9/.95 AdamW moments, epsilon 1e-8, and decay .1.
No training or optimizer source changed for this study.

| Cohort | Learning-rate schedule | Preserved source |
| --- | --- | --- |
| `lr001` | .001 peak; main-model warmup/stable/cosine decay | `class_geometry_wsd/runs/audit-02` |
| `lr01` | .01 peak; same WSD schedule | `class_geometry_lr/runs/lr01-01` |
| `settle` | .01 peak; reduction from step 1000 to .0001 at 4000; hold to 8000; final cooldown | `geometry_gap/runs/settle-01` |

Each forecast receives one checkpoint, its absolute time, the known schedule,
and the observed-answer partition derived from training labels. All 144
forecasts were frozen before the separate verifier opened any update or
embedding trace. The two tracks use the checkpoint ratios or the uniform
19/14 cone, with identical initial geometry and numerical guards.

## Findings

The table retains initialization and every inconclusive window. Values are
ranges over all eight seeds. A prefix count starts at the first update after
the anchor and stops at the first uncertified step; it never skips an early
failure. The share is active class-contrast energy divided by total centered
readout energy. It is not query-prediction certificate coverage.

| Cohort | Anchor | Updates | Minimum future share floor | Prefix above 50% | Prefix above 60% | Prefix above 2/3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lr001 | 0 | 4000 | 0% | 0 | 0 | 0 |
| lr001 | 1000 | 4000 | 0% | 20–24 | 10–14 | 2–7 |
| lr001 | 4000 | 4000 | 0% | 28–35 | 16–22 | 6–13 |
| lr001 | 6000 | 4000 | 0% | 31–35 | 18–21 | 9–11 |
| lr001 | 8000 | 2000 | 0% | 30–34 | 16–20 | 6–11 |
| lr001 | 9500 | 500 | 52.793–55.592% | 500 | 143–197 | 52–82 |
| lr01 | 0 | 4000 | 0% | 0 | 0 | 0 |
| lr01 | 1000 | 4000 | 0% | 0 | 0 | 0 |
| lr01 | 4000 | 4000 | 0% | 0 | 0 | 0 |
| lr01 | 6000 | 4000 | 0% | 0 | 0 | 0 |
| lr01 | 8000 | 2000 | 0% | 0 | 0 | 0 |
| lr01 | 9500 | 500 | 0–0.000180% | 14–24 | 1–12 | 0–4 |
| settle | 0 | 4000 | 0% | 0 | 0 | 0 |
| settle | 1000 | 4000 | 0% | 0 | 0 | 0 |
| settle | 4000 | 4000 | 0% | 260–321 | 112–179 | 12–84 |
| settle | 6000 | 4000 | 0% | 274–334 | 127–193 | 29–98 |
| settle | 8000 | 2000 | 12.957–14.361% | 293–355 | 143–207 | 44–112 |
| settle | 9500 | 500 | 67.834–72.416% | 500 | 500 | 500 |

At step 9500, all eight settling runs retain a certified share above 2/3
through all remaining 500 updates; the measured minimum shares are
69.939–74.500%. All 4,000 of those seed/update pairs change parameters. The
uniform-cone control already succeeds here, with floors 67.804–72.386%; using
the current moments improves each seed by only 0.0296–0.0305 percentage
points. Thus the late success mainly comes from the schedule and existing
geometry, not a large gain from the new moment cone.

At the start of the settling hold, the prospective 60% certificate lasts
112–179 updates, exactly three more than the uniform control's 109–176.
Its positive floor expires after 2,209–2,301 updates and is zero for the
full 4,000-step window. Yet actual share stays above 67.316% across all
32,000 moving seed/update pairs. The predicted final active radius is
30.162–30.185, compared with the previous observed-forcing radius
0.881–0.987 in [task #204](../geometry_persistence/README.md). Both radii
use the decayed anchor, but only the new one is constructed without future
forcing observations. Its looseness remains a substantive gap.

The cone's dependence on the checkpoint shrinks to 20.31% after ten steps,
0.03452% after fifty, and about 1.19e-7 of its initial value after one hundred.
This explains the small gain from current moments in this bound. It is not
a statement that the trained model forgets its learned geometry.

Reliable discovery and long useful directional control remain open. The
next missing argument must exploit the actual loss and changing features:
why gradients create the useful contrasts and why their cumulative forcing
in damaging directions stays small. A magnitude-only optimizer bound does
not establish either claim. No new prediction certificates were constructed
here, so earlier prediction-coverage figures are unchanged.

## Evidence and checks

`runs/forecast-01/` preserves the frozen forecasts, constructor ledger,
complete verification, and generated summary, including first uncertified
steps and both tracks for every case. Construction took 3.915 seconds and
verification 65.044 seconds, combined 68.958 seconds against the 1800-second
cap. No tolerance was changed after measurement: direction allowance 1e-6,
the declared predicted FP32 rounding cap, and scaled enclosure tolerance
1e-8. These guards were verified on the traces; their universal validity is
not covered by the exact-real Lean proof.

The measured native source is commit `65fefb86`; the protocol was frozen in
`3641a429`. Its source SHA-256 is
`fd6ff2b19db26dc0bbf78dbf5660de2122ae9de3541152d1ca24aab3ff39ba10`.
The preserved executable at `.kanban/artifacts/205/measured-harness` hashes to
`78f170928f3a3130204a9ea13a85075378318e1b652308f1779726e979f4c161`.
The 25 Lean results landed in `958a5f59` and use only the standard axioms.
The full build passed; the global audit remains 36 sorry, zero rests and zero
extra axioms. Generated INDEX reports zero vacuous claims and placeholders.
All eight native unit checks, clippy, formatting, and the forbidden scan pass.

## Reproduce

Run from the repository root with the three preserved source cohorts. Construction
and verification are separate processes; the first freezes every forecast before
the second opens trajectory files. Output directories must be new.

```sh
CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target" cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin future-forcing
.lake/geometry-gap-target/release/future-forcing experiments/future_forcing/runs/forecast-01 experiments/class_geometry_wsd/runs/audit-02 experiments/class_geometry_lr/runs/lr01-01 experiments/geometry_gap/runs/settle-01
.lake/geometry-gap-target/release/future-forcing verify experiments/future_forcing/runs/forecast-01 > experiments/future_forcing/runs/forecast-01/verification.json
jq -f experiments/future_forcing/summary.jq experiments/future_forcing/runs/forecast-01/verification.json > experiments/future_forcing/runs/forecast-01/summary.json
CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target" cargo test --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin future-forcing
CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target" cargo clippy --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin future-forcing --tests -- -D warnings
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/FutureForcingAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
CARGO_TARGET_DIR="$PWD/.lake/native-index-target" LAKE_ARTIFACT_CACHE=false cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml
```

Construction hashes its source, inputs, and each forecast. Verification checks
those hashes again, checks every declared prefix, and retains failures. The
source ledger includes the training support and checkpoint states; neither later
gradients nor observed update magnitudes enter a forecast.
