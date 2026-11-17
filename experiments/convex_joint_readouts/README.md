# Joint readout capacity on learned ordered states

Does redistribution of component capacity improve the original recall failure?
CM104 at `33511122` proves a convex complete weighted-readout domain and its
training-only CE gap. It also proves a concrete model with zero old fixed-weight
gap but strictly lower actual CE after reallocating capacity. This Rust experiment
tests that remaining optimizer restriction on immutable learned models.

| Phase | Inputs | Operation |
| --- | --- | --- |
| Diagnostic | All twelve reviewed radial512, radial2048 and Euclidean512 endpoints | Compare complete fixed-weight and joint-capacity gaps; evaluate eight fixed feasible probes; accept no update |
| Mini | Cautious/refined Euclidean512 endpoints, seed1 | 512 additional original minibatch updates using full actual CE coefficients |
| Forward Mini | Corresponding two Euclidean512 endpoints, seed1 | Same schedule, with finite signed CE queries instead of analytic residuals |

Every original class, example, supervised mask, duplicate, short tail and complex
coordinate remains. All 129 components are stored, including learned components
at zero weight. The existing structural zero component holds unused capacity. The
native sequence input and original next-object output are unchanged. All learned
initial states, transitions and frequencies remain intact during this conditional
optimizer phase; this phase is not a proposal to replace the full architecture
with prescribed features. The transformer baseline is unchanged.

Write `U[j,c] = native_weight[j] * readout[j,c]`. Shared capacity is the sum over
components of their maximum class row-l1 size, at most B=16. Actual coefficients
are mean `state * (probability - target_indicator)`, with no extra old-weight
factor. The support minimizer compares every complete component, including one
whose current weight is zero. Convex interpolation acts on these weighted rows;
interpolating weights and normalized rows independently would produce different
logits. Reconstruction normalizes every complete block by its maximum row size.
No nonzero mass or coordinate is thresholded or removed.

The numerical analogue of CM104 is not a rounding certificate. Existing native
capacity tolerances remain explicit: reconstruction rejects excess beyond
2e-12 times B and leaves any smaller negative unused-mass roundoff at zero;
native validation still runs. Every accepted step checks actual reconstructed
logits and CE against its affine prediction and reports residual capacity error.
Zero blocks normalize to zero, and nonfinite or underflowed nonzero norms fail.

Each training step reserves query mass 1e-4 in both modes, then uses minibatch32,
twelve line-search iterations and a fixed maximum interpolation 1/16. The receipt
centre is after that reserve. Forward training uses only the existing finite
feasible signed original-class CE queries. The separate diagnostic uses analytic
residuals for saved endpoints of either mode; it is not forward-only training.
All four 512-update endpoints seal before any held-out scoring. Training uses
the original sampler with seed10001. Original success and separate99% raw/exact
checks are reported without early stopping or held-out model selection.

Native controls reproduce the actual CM104 counterexample, check reallocation
in both modes and complete score reconstruction, and compare every trainable
weighted real/imaginary coordinate with actual CE finite differences. A five-record
fixture retains a duplicate and a short final minibatch. The structurally zero
component has zero coefficients. Invalid capacity, shape and query-reserve inputs
are refused. These controls do not prove unseen generalization.

Use native tooling:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/joint-readouts
export CARGO_BUILD_JOBS=2
cargo test --offline --locked --manifest-path experiments/convex_joint_readouts/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_joint_readouts/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_joint_readouts/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_joint_readouts/Cargo.toml
```

Before execution, archive the producing commit with `git archive`, freeze its
binary, `protocol.json`, the pinned `convex_readout_optimality/inputs.json`, SHA256
identities and `plan` output. Run that frozen binary with new output paths:

```sh
bash experiments/convex_joint_readouts/run.sh \
  /home/misha/lean_projects/transformer /run/media/misha/storage \
  /run/media/misha/storage/transformer-runs/rotary-recurrence \
  /run/media/misha/storage/transformer-runs/freezes/joint-readouts-v1 \
  /run/media/misha/storage/transformer-runs/diagnostics/joint-readouts-v1 \
  /run/media/misha/storage/transformer-runs/joint-readouts/original-v1
```

The wrapper permits one worker with a32GiB address-space cap. Native admission
checks enforce110GB aggregate RAM and both disk thresholds before each minibatch.
A space failure stops execution without cleanup. Generated evidence stays on
storage outside the worktree. Review repeats all twelve diagnostics into another
directory byte-for-byte and independently replays all four original raw reports.
No result or99% success is claimed before those checks.
