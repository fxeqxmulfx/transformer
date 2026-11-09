# AdamW motion and support-complete certificates

The question is whether the main-model AdamW recurrence supplies a useful
finite-horizon persistence guarantee, and whether restoring the direction
between active and unused class means tightens prediction certificates.
The [fixed protocol](PROTOCOL.md) retains every state of all three complete
eight-seed cohorts. No new training or outcome-based selection is planned.

| Part | Changed or derived | Controls |
| --- | --- | --- |
| Optimizer proof | Exact .9/.95 moment invariant and finite learning-rate budget | Arbitrary gradients; explicit floating-point remainder |
| Persistence audit | Analytic bound and actual motion during updates 4001–8000 | All eight seeds in all three cohorts |
| Certificate basis | Active rows centered at the unused-class mean | Same folds, all rivals and numerical allowances; ranks 32 and 33 separate |
| Certificate envelope | Maximum of three valid lower bounds | Extra basis preparation explicitly charged |

Twenty-one Lean theorems now prove the moment invariant, bounds on the
bias-corrected adaptive direction, finite-window parameter motion, conditional
prediction persistence and certificate-envelope soundness. The optimizer
bound allows arbitrary gradient histories and changing features. Preserving
a prediction still requires a positive starting margin and feature-motion
control; discovery and global convergence remain unproved.

The [derivation](../../review/flat_spectrum.md#main-model-adamw-motion)
states every obligation. The full Lean build and axiom audit pass with no
new `sorry` or extra axioms. All numerical audits are complete. The general
motion bound is too loose to certify the observed persistence. Certificates
improve modestly with one extra direction or multiple prepared bases;
the new single rank-32 basis does not improve the early-settling mean.

## Optimizer findings: a valid bound that is still too loose

All 240000 saved embedding updates pass, covering 2979840000 coordinate
checks in 34.20 seconds. Every source trace hash matches its original report.
No coordinate exceeds the moment-derived forcing bound plus its recorded
rounding remainder, even before the fixed relative allowance is needed.
Each cohort has 32000 nonzero updates in the hold window (4001–8000).

The table gives ranges over all eight seeds. Displacements are Frobenius
norms of the complete readout from its step-4000 value; the actual maximum
uses every update in the hold window, not just the 31 geometry probes.

| Schedule | Sum of learning rates during hold | Actual maximum displacement | Analytic displacement budget |
| --- | ---: | ---: | ---: |
| Original .001 WSD | 4.0 | 3.53–4.71 | 633.89–638.93 |
| Original .01 WSD | 40.0 | 13.78–15.40 | 9207.05–9231.85 |
| Early settling at .0001 | 0.4 | 0.76–0.88 | 56.58–56.98 |

Early settling reduces both actual motion and the conservative budget.
However, even its 56.58–56.98 bound exceeds the initial active-contrast norm
17.18–17.90 in every seed. The elementary norm bound therefore cannot even
certify preservation of nonzero active energy over the entire hold. None
of the 24 tested windows passes that sufficient condition. It cannot be
used to claim prediction persistence or discovery on these trajectories.

Observed persistence is much stronger: early-settling readout path lengths
are only 1.24–1.33, and per-seed minimum active-energy shares over all 4000
hold updates are 67.32–72.22%. The original .01 shares fall to 47.75–49.71%.
These are measured quantities, and energy concentration still does not
guarantee alignment or correct predictions. The remaining proof needs
tighter control of accumulated directions and the changing feature map,
plus an argument that creates the initial positive margins.

The motion audit was measured from `cf2483c2`, source fingerprint
`67745a73702e74055bfacd604ed1e048fde7c76eb74d8138e1f3592c5559a286`.
The complete results are preserved in `runs/motion-01.json`.

## Certificate findings

At the early-settling endpoints, mean accuracy stays 99.575% and all 160
incorrect predictions remain uncertified. Every method uses the identical
trained states. Counts below cover 37640 validation evaluations across
the eight seeds; each late-minimum range retains all 21 late probes.

| Method | Basis budget | Mean final coverage | Range of per-seed late minima | Correct but uncertified |
| --- | --- | ---: | ---: | ---: |
| Previous all-class basis | One, rank 32 | 98.324% | 97.92–98.55% | 471 |
| Previous active-centered basis | One, rank 32 | 98.504% | 98.11–98.66% | 403 |
| New basis, centered at unused mean | One, rank 32 | 98.499% | 98.04–98.62% | 405 |
| New basis, centered at unused mean | One, rank 33 | 98.565% | 98.15–98.68% | 380 |
| Envelope of all three | Three, each rank 32 | 98.592% | 98.24–98.72% | 370 |
| Envelope of all three | Three, each rank 33 | 98.626% | 98.26–98.72% | 357 |

At rank 32 the candidate gains 12 endpoint certificates and loses 14 versus
the active-centered basis: three seeds improve, three fall, two are unchanged.
Keeping the additional direction at rank 33 gains 23 and loses zero; seven
seeds improve and one is unchanged. At that rank, no active-basis certificate
is lost in any of the 744 states. This is an observed result under retained
numerical defects, not an assumption that every computed refinement is monotone.

The rank-33 envelope gains 46 endpoint certificates and loses zero, reducing
403 correct-but-uncertified evaluations to 357 (11.4%). All eight seeds
improve. This uses three prepared bases, each allowed rank 33, so it is not
an improvement at the previous single-basis rank-32 budget. The old rank-128
all-class control already reaches 99.455% with 45 correct predictions left
uncertified; that larger-budget result is not a gain from the new method.

The other two cohorts confirm the budget distinction:

| Cohort | Previous active rank 32 | New single rank 32 | New single rank 33 | Three-basis rank-33 envelope |
| --- | ---: | ---: | ---: | ---: |
| Original .001 WSD | 98.151% | 98.130% | 98.225% | 98.257% |
| Original .01 WSD | 98.430% | 98.451% | 98.502% | 98.597% |

The long .01 plateau still has poor persistence: its per-seed late minima
are only 51.18–61.68% with the new rank-33 basis, and 51.65–62.32% with the
rank-33 envelope. One .01 seed reaches 99% endpoint coverage; no early-settling
seed does at rank 32 or 33. No cohort sustains 99% at every late probe at
these ranks. The native summary retains all 24 paired endpoints, all four
rank controls, crossings, regressions and gained/lost counts.

All 744 cases pass with 28001184 byte-exact reproductions of the previous
active-basis margins and 56002368 new margin bounds. The independent artifact
verifier rereads all emitted bounds, validates the envelope maximum and
gained/lost counts, and checks FP64 and recorded GPU margins. It finds zero
false certificates or bound violations. All 204082176 saved moment-coordinate
checks satisfy the raw invariant without needing the numerical allowance;
the largest observed adaptive direction is 0.946162.

Construction takes 1108.78 seconds and independent verification 99.98 seconds,
total 1208.76 within the fixed 1800-second cap. Including the separate
34.20-second trace audit gives 1242.96 seconds. This is incremental
postprocessing: original all-class bounds were already prepared and are
reused from hashed evidence. It is not a measurement of the cost of freshly
preparing all three bases, nor an inference speedup.

Requested rank 32 achieves 20–32 directions across the folds. The maximum
coefficient norm is 4403.78 and maximum orthogonality penalty `6.45e-11`;
both remain in the bound. Requested rank 33 achieves 20–33 directions.
Full fitting metadata, including achieved rank
at every requested budget, is saved before query outcomes are scored.

The certificate audit was measured from `ab98c4ab`, source fingerprint
`1133305de5449d4960f64d07d8135f94fdb1cf04e291e454f4dc09e7cc784e00`.
Results and all emitted bounds are preserved in `runs/support-01/`.
The frozen protocol is `2ebdc43c`; the proof and motion audit are `cf2483c2`.

## Reproduce the certificate audit

Use the fresh `-02` outputs below from the repository root. Existing cohorts
remain immutable. The independent verifier also checks that the envelope
is exactly the maximum of its three constituent lower bounds.

```sh
export CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --bin geometry-dynamics
"$CARGO_TARGET_DIR/release/geometry-dynamics" experiments/geometry_dynamics/runs/support-02 \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01 \
  experiments/geometry_gap/runs/active-01
"$CARGO_TARGET_DIR/release/geometry-dynamics" verify experiments/geometry_dynamics/runs/support-02
```

## Reproduce the update audit and proofs

The update audit streams all 240000 saved embedding steps, checks each
recorded forcing against the moment-derived bound plus the recorded rounding
remainder, and preserves the source trace hashes. Its hold-window motion
budget includes cumulative rounding. The bound on an arbitrary future
FP32 execution is not formally established by these saved-state checks.

```sh
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --bin geometry-motion
"$CARGO_TARGET_DIR/release/geometry-motion" experiments/geometry_dynamics/runs/motion-02.json \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/MainAdamAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml -- --check
```

Regenerate summaries from the preserved measurements without retraining:

```sh
jq -f experiments/geometry_dynamics/summary.jq \
  experiments/geometry_dynamics/runs/support-01/report.json \
  > experiments/geometry_dynamics/runs/support-01/summary.json
jq -f experiments/geometry_dynamics/motion_summary.jq \
  experiments/geometry_dynamics/runs/motion-01.json
```
