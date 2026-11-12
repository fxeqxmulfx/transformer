# Feasible sharpening of frozen grouped endpoints

Can a prediction-preserving endpoint check close a capped training CE gap before
comparison with memorizing candidates? This follows the six-task sharing pilot:
all its selected candidates passed the original 99% held-out accuracy threshold,
but square-times had a nonzero training CE gap in full-batch, minibatch and
forward-only modes. A training-loss hull could prefer an ideal-CE memorizer.
Lean CM11 was proved first and landed at `b3c2e8ae`; it supplies the probability,
spectral-feasibility, strict-decoding and ordinary-CE excess certificates.

| Stage | Coordinates | Decision data |
| --- | --- | --- |
| Parent | Frozen `convex_stress_sharing` endpoint and maps | Original training run |
| Proposal | One-hot parent argmax on active groups, uniform padding, common uniform mixture restoring both moment bounds | Saved probabilities and fixed bounds |
| Selected endpoint | Original if already ideal, any observed argmax is wrong, or CE does not improve by more than 1e-12; otherwise proposal | Every original training observation, including repetitions |
| Budget choice | First ideal endpoint, otherwise lowest training CE | Original training observations |

The unchanged family is convex after fixing the group maps. Rounding is not a
convex map. Actual ordinary CE uses these probability coordinates as logits,
as in the parent. Centered Fourier moments scale with the common mixture;
norms are recomputed after decoding and again on the feasible candidate.
Positive mixing preserves every active argmax, including wrong unseen answers.
This cannot fix missing labels or establish correctness on the whole suite.

The frozen protocol has 36 parent cells: six tasks, two budgets, three original
training modes. No parent is retrained or modified. Save all 36 proposals and
36 selected endpoints, including non-improvements, then all 18 budget choices.
Parents already within 1e-8 of ideal CE retain exactly the old coordinates.
Record parent optimization and grouping costs separately from postprocessing;
the endpoint selection evaluates CE on two tables, with no additional training
gradient. Diagnostic split evaluation and replay have additional cost.

Selection precedes held-out evaluation. Each saved table must preserve every
parent split's correctness counts and class accuracies. The separate raw-table
auditor uses direct Fourier sums and plain-exponential CE. It shares the frozen
data constructor and LAPACK SVD, so its norms are numerical checks, not interval
proofs. `compare-parent` independently checks decoded mixtures, all active
argmax decisions, original multiplicities, CE selection and exact retained
coordinates. Producer replay reconstructs every result from hashed parents.

## Reproduce

Use Rust, FFTW3 and OpenBLAS/LAPACK, with one BLAS thread. The launcher defaults
to two numerical processes with eight frequency workers each. Outputs remain
ignored under root-linked `runs/`. The parent cohort must be the frozen sharing
pilot from producer `ddf9aa4a`, with its exact fingerprint in `protocol.json`.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SchattenMomentsAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_sharpen/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_sharpen/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_sharpen/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_sharpen/Cargo.toml --bins
bash experiments/convex_stress_sharpen/run_pilot.sh experiments/convex_stress_sharing/runs/pilot-v1 experiments/convex_stress_sharpen/runs/pilot-v1 2
OPENBLAS_NUM_THREADS=1 experiments/convex_stress_sharpen/target/release/audit_saved CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT
OPENBLAS_NUM_THREADS=1 experiments/convex_stress_sharpen/target/release/audit_saved CANDIDATE_DIRECTORY/proposal PRODUCER_FINGERPRINT
experiments/convex_stress_sharpen/target/release/audit_saved compare-parent PARENT_DIRECTORY CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT
```

Three native checks exercise infeasible decoding and spectral repair, unchanged
argmax with the CE bound, ideal-CE sharpening, exact already-ideal retention,
repeated observations and rejection of observed decoding errors. Corrupted CE
and altered endpoint selection must be rejected by the saved-table audit.

## Findings

All 36 frozen-parent checks completed under producer `6cb8612f`, fingerprint
`6941b4f1220bf2aaadf4803b1a9bc71193436effcdf938441652b48f1e34906e`.
Exactly three proposals were accepted: square-times with the full-rank budget
in each original training mode. The proposed table reaches the existing
numerical ideal-CE tolerance without changing any active prediction.

| Parent training mode | Original mean CE excess | Selected excess | Original and selected primary accuracy |
| --- | ---: | ---: | ---: |
| Full batch | 0.0009361298 | 2.31e-12 | 100% |
| Minibatch | 0.0001425854 | 2.31e-12 | 100% |
| Forward only | 0.0017483808 | 2.31e-12 | 100% |

All three have mixture coefficient `0.9999999999979994`, maximum operator norm
9.848857801776 and maximum nuclear norm 482.594032287043, below their original
budgets 9.848857801796 and 482.594032288009. The unchanged six-task selection
still passes all 18 original 99% tests, and now all 18 selected endpoints have
the numerical ideal-CE certificate. Addition chooses the rank-one budget;
the other five tasks choose full-rank. This removes the square-times CE
disadvantage against an ideal-CE memorizer; it does not yet measure selection
in an expanded whole-suite hull.

All other 33 selected tables retain exactly their parent coordinates: 15 were
already ideal, 12 proposals did not improve training CE, and six parents had
an observed decoding error. Keeping already-ideal parents matters: decoding
an imperfect unseen completion can force a smaller feasible mixture and worsen
observed CE. Every proposal, including these rejected ones, remains saved.
The small missing-answer errors of the k2/k3 pilot remain unchanged.

All 36 producer replays, 72 independent raw-table audits, 36 parent comparisons
and the 18-choice replay pass. The auditor rejects corrupted CE and an altered
endpoint choice. All three native correctness checks, formatting and clippy
pass. Lean audits report no extra axioms or reliance on unfinished proofs in
this namespace; the repository retains its 36 pre-existing unfinished claims.

Recorded postprocessing totals 44.45 seconds summed across 36 endpoints,
0.84–1.74 seconds each, including parent loading, group reconstruction, norms
and split diagnostics. This excludes saved-file writing, replay and independent
audit, and is not parallel wall time or training time. No additional gradient
was evaluated. Generated evidence remains under root-linked `runs/pilot-v1`
and the local board artifacts for #172. Earlier pilot metrics and the ongoing
35-task hull cohort remain separate; a fresh matched transformer comparison
has not been run.
