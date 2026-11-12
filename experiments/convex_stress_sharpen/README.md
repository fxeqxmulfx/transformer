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

The protocol and source are prepared before this follow-up runs. No measured
sharpening outcome is claimed yet. Earlier pilot metrics and the ongoing
35-task hull cohort remain separate evidence. A fresh matched transformer
comparison has not been run.
