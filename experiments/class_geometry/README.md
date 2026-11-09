# Discovery and persistence of class-visible geometry

Does the source optimizer build the 32 class contrasts among observed answers,
and does that explain useful compressed geometry across whole trajectories?
The [protocol](PROTOCOL.md), frozen at `9692d9a0`, fixes two original seeds and
six replication seeds. All eight 10000-update runs completed in 868.89 seconds,
within the fixed 1800-second cap. Every update and all 248 geometry probes pass.
The result is a partial mechanism, not an unconditional discovery theorem.

| Comparison | Fixed scope |
| --- | --- |
| Original replay | Seeds 7,11; all 10000 updates; original complete state hashes |
| Replication | Seeds 1,3,5,13,17,19; identical source and budget |
| Class components | Active-answer contrasts, unused-class contrasts, between-group means |
| Update terms | Decay, isotropic readout, adaptive correction, tied encoder, memory, rounding; total squared increment retains all cross terms |
| Geometry | Hidden/readout-aware bases at ranks 0,32,33,128; 31 probes per seed, including 21 late probes |

All updates and every prefix receive finite energy checks. Query labels never
select a basis; current-state geometry retains #198's numerical allowances.
The source files reuse #198 at `8e719f2b`, with rank 33 added as a declared
control. No training equation changes. The independent source evaluator is
the existing FP64 implementation from `flat_approx`.

## What training built

Training labels supply 33 distinct answers and 64 unused classes. Active
contrasts need at most 32 generators; unused contrasts need at most 63, and
between-group means add at most one direction. This dimension count holds
at initialization too. Active contrasts start with 32.78–33.73% of centered
readout energy and finish with 70.16–75.87%; their absolute energy rises
from about 14 to 53–56.

All accuracy and coverage entries below are percentages of 4705 validation
queries. Late minima include every probe from step 5000 through 10000, spaced
250 updates apart. These are **current-state** certificates; #198 instead
tested forecasts of the next update, so the numbers are not interchangeable.

| Seed | Role | Final active energy share | Final accuracy | Final hidden rank-32 coverage | Final readout rank-32 coverage | Minimum late readout rank-32 coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | Original | 71.24 | 99.66 | 96.58 | 99.28 | 95.22 |
| 11 | Original | 72.15 | 99.68 | 97.39 | 99.55 | 95.79 |
| 1 | Replication | 73.77 | 99.62 | 93.07 | 99.15 | 95.94 |
| 3 | Replication | 73.56 | 98.98 | 75.75 | 97.05 | 93.99 |
| 5 | Replication | 70.16 | 99.77 | 76.92 | 98.09 | 95.86 |
| 13 | Replication | 74.42 | 99.62 | 90.50 | 99.13 | 96.26 |
| 17 | Replication | 75.87 | 98.11 | 89.27 | 95.98 | 95.98 |
| 19 | Replication | 71.86 | 99.79 | 98.32 | 99.74 | 97.73 |

All six replications maintain at least 90% accuracy and rank-32 coverage at
every late probe; none maintains 99% of either throughout that window.
Endpoint 99% accuracy succeeds in 4/6, and 99% coverage in 3/6. There is no
claim about certificate values between probes. Active energy share increases
on only 5910–6371 of 10000 updates per seed. Absolute active and unused
energies also fluctuate; no monotone collapse occurred.

All seeds first reach sampled 50% accuracy at step 250 and readout rank-32
majority coverage at 500. First 90% readout coverage is at 750 or 1000;
first 99% at 1500–5500. The analyzer reports every seed's 50/90/99% crossings,
every probe, endpoint and late minimum, including regressions.

Rank 33 adds 0–0.17 percentage points of endpoint readout coverage over rank
32; it does not remove seed 17's gap. Rank 128 reaches 97.96–99.77% endpoint
coverage at a larger budget. Every fold's achieved rank, retained energies,
singular values, coefficient norms and orthogonality penalties are retained
and aggregated by the analyzer. Both rank-zero families agree exactly. A
native rank-33 witness checks that an extra observable direction can matter.

Across all probes and folds, requested rank 32 achieves 20–32 directions;
readout rank 128 achieves 20–96. The smallest retained training singular
value is `1.93e-4`; maximum rank-32 readout coefficient norm is 4372.90,
with maximum orthogonality penalty `1.99e-11`. This conditioning is explicitly
retained in the bounds, not treated as exact or stable feature recovery.

Global active-contrast versus full-readout rank-32 subspace overlap finishes
at 0.832–0.984. The full centered readout still has 20.31–25.96% of its energy
outside its top 32 singular directions. Class concentration does not mean
exact readout rank 32 or a flat complete hidden spectrum. Seed 17 has the
largest active energy share and worst replication endpoint accuracy/coverage:
concentration alone is insufficient.

## Signed mechanism and its limits

For unused classes c,d, the actual partial CE derivative obeys
`<E_c-E_d,g_R,c-g_R,d> >= 0`: logits and softmax probabilities have the same
ordering. Averaging pairs proves the sign for the whole centered unused
group, with changing features and minibatches. All 80000 measured damping
checks satisfy the fixed numerical allowance.

The full tied AdamW update also has adaptive, encoder, memory and rounding
contributions. This study uses **actual minibatch** partial and full gradients;
#198 instead separated full-training readout/encoder credit from sampling.
Each class projection's finite change retains all signed radial terms and
the square of the total increment, including cross terms. The sums close at
every prefix; the maximum scaled energy discrepancy is `8.46e-14`.

For active energy S and competing energy V, exact share change is
`(V*ds-S*dv)/((S+V)*(S+V+ds+dv))`. Applying this identity to the recorded
components gives these mean whole-run contributions across all eight seeds:

| Contribution | Percentage points of active energy share |
| --- | ---: |
| Decay radial term | 0.00 |
| Isotropic readout | +37.53 |
| Adaptive readout correction | +442.21 |
| Tied encoder | -416.20 |
| Moment memory | -21.89 |
| Rounding | +0.000003 |
| Total quadratic term | -1.98 |
| Net change | +39.67 |

These are accumulated signed contributions to the actual trajectory, not
percentages of causal responsibility or retraining ablations. The signs
agree in all seeds. Share attribution telescopes with maximum discrepancy
`1.20e-14`. Common decay preserves relative shares on its own; its mixed-step
interactions remain in the quadratic term. Positive readout contributions
explain the measured direction, but do not bound future encoder/memory opposition.

Twenty new Lean theorems in
[ClassDecomposition](../../src/Transformer/Grokking/FlatSpectrum/FeatureCodes/ClassDecomposition.lean),
[UnusedClassDamping](../../src/Transformer/Grokking/GradientEvidence/UnusedClassDamping.lean)
and [ClassTrajectory](../../src/Transformer/Grokking/GradientEvidence/ClassTrajectory.lean)
prove the actual derivative sign, finite-step condition, class decomposition,
explicit generators, cumulative identities and relative-share law, with
satisfying examples. A separate witness has all centered energy in active
contrasts yet a strictly wrong prediction. The
[derivation](../../review/flat_spectrum.md#class-contrast-discovery-and-persistence)
states the hypotheses and remaining gap: controlling coupled feature learning,
adaptive scales and moments enough to guarantee query alignment and margins
over time. That global theorem remains open.

## Verification and provenance

The standalone verifier reconstructs class energies from all 80008 embedding
states, checks 80000 finite/cumulative update identities, and checks 18667456
margin bounds against an independent FP64 source model and recorded GPU
margins. Zero false certificates and zero bound violations occur. All 26
prior probe states/coverage counts match #198. All six original complete
checkpoint hashes and both histories match; all 240 instrumented/ordinary
step comparisons agree byte for byte.

Maximum parameter-component closure error is zero. Independent feature error
is at most `1.12e-4`, and logit error `9.79e-6`, below the unchanged `1e-3`
allowances. This is audited numerical evidence, not a formal floating-point
proof. Native tests, strict clippy, formatting, full Lean build, axiom audits
and native INDEX generation pass. There remain 36 existing unfinished
statements, with no new sorry, resting theorem, extra axiom or placeholder.

The measured source commit is `832fc5cd`, fingerprint
`4de240fc34ec163ae0eed325eeca2d3364c330465d7e29913568ea855318782a`.
Its executable is preserved. Subsequent changes add verification/analysis,
the rank-33 test and a lint-only iterator change. The pinned trainer hash is
`14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`,
with Kernite revision `a1b6bc2c956e1a5f35960e26b209e1b6c213096c`.

The first startup attempt (`audit-01`) is retained. No update ran: checks
caught a pre-final-RMS versus normalized-feature adapter error in seven seeds,
and an overly strict JSON-number equality check in the eighth. The repair
uses the evaluator's recorded final-RMS denominators and restores #198's
exact-correct-count / `1e-12` loss comparison for serialized history.
A native regression checks the normalized features against independent logits.
Certificate allowances, seeds, ranks, optimizer, probe schedule and cap are
unchanged. The corrected complete attempt uses `audit-02`.

## Reproduce with native tools

From the root, with CUDA available to the execution context:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/class-geometry-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry/Cargo.toml
cargo fmt --manifest-path experiments/class_geometry/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry/Cargo.toml --bins
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/readout_geometry/runs/audit-01/report.json \
  experiments/class_geometry/runs/audit-03
"$CARGO_TARGET_DIR/release/verify" experiments/class_geometry/runs/audit-03 \
  experiments/readout_geometry/runs/audit-01/report.json
"$CARGO_TARGET_DIR/release/analyze" experiments/class_geometry/runs/audit-03
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ClassGeometryAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --locked --release --manifest-path scripts/native_index/Cargo.toml
```

Use a fresh output directory. Generated evidence remains ignored and local;
it is preserved under root `experiments/class_geometry/runs/` and check logs
under root `.kanban/artifacts/199/`.
