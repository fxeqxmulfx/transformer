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

Producer `f5ae34ef` completes all twelve diagnostics and all four fixed training
runs. The joint gaps are1.18036–1.61864, versus the old fixed-weight gaps
0.27326–0.54996. Thus the old conditional lower bounds omitted substantial
capacity-allocation scope. Each endpoint has an actually evaluated full-training
probe lowering CE by0.00566–0.01409, with the best of the eight fixed probes at
eta1/64 in all twelve cases. That lowest-CE probe raises training correct counts
in three cases, lowers them in eight and leaves one unchanged. No diagnostic
probe is accepted or earns new held-out accuracy credit.

The minibatch update rule fails severely. Every one of2048 accepted steps lowers
its own reserved-centre minibatch CE, and every line search selects its maximum
eta1/16. Nevertheless full-training CE rises from2.032–2.086 to4.006–4.063, while
both training and novel accuracy collapse. This is evidence of an unsuitable
stochastic update rule, not a counterexample to the conditional convexity theorem
or proof that these learned states cannot fit the task.

| Source/mode | Train raw before | Train raw after | Train CE after | Novel raw before | Novel raw after | Test raw after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cautious Mini | 67.8711% | 3.8086% | 4.063179 | 14.8438% | 3.5156% | 3.9063% |
| cautious forward | 68.7500% | 4.0527% | 4.037471 | 12.1094% | 2.5391% | 4.2969% |
| refined Mini | 66.7969% | 3.7109% | 4.005676 | 11.5234% | 3.3203% | 5.4688% |
| refined forward | 64.6484% | 4.0527% | 4.008804 | 12.3047% | 1.9531% | 2.5391% |

Raw and exact accuracy agree on these original single-target records. Every
original success flag and every99% flag is false. All four complete raw-logit
replays are identical. Each model retains129 components and all learned content;
component1 is selected in1350 of2048 steps and finishes with13.84–14.76 of the
total native capacity16. This concentration is an observed optimizer outcome,
not component pruning. Full-training diagnostics had selected other components.

All four runs finish the declared512 updates and eight additional epochs. The
two Mini runs each use311,296 CE evaluations; the two forward runs each use
3,588,096, including3,276,800 signed-bias queries. The twelve diagnostics use
294,912 CE evaluations. Maximum accepted-step logit reconstruction error is
4.45e-14, operator-norm excess2.23e-16 and shared-capacity excess0; the largest
observed aggregate RSS before an update is10.981GB. Diagnostic support/change
identity errors stay below4.45e-16/1.58e-13. These are measured floating checks.

The next repair must address the discrepancy between a current minibatch
coefficient and the complete training coefficient before further aggressive
capacity movement. Merely lowering the current batch CE does not suffice.
Both mini modes remain required; full-batch training and unseen-label access
are not proposed as substitutes. Generalization remains unresolved.

Frozen evidence is under `transformer-runs/freezes/joint-readouts-v1`, reports
under `diagnostics/joint-readouts-v1`, and models/receipts under
`joint-readouts/original-v1` on storage. The source fingerprint is
`a4aa537ad5e8bad152e79689913078a0865394c17cd98eb63f7dfbac23168c39`.
