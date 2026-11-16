# Learned bilinear sequence stress probe

Can the CM86 complete complex sequence model learn original GPT stress tasks
with ordinary cross-entropy, using Mini and forward-only Mini updates? This
experiment implements the family proved in
`Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear`, including
CM87 identification and CM88 update identities. The manuscripts are CM86–88
in `review/flat_spectrum.md`; the proof commits are `ed9d359d`, `e9ac5beb` and
`e50d3691`. The older `convex_learned_sequence` experiment uses another family.

For vocabulary size V and context cap T, the complete joint domain has
2 V (V+1) (1+TV+1[T≥2]V²) real coordinates. Immutable object identities,
positions, a bias and adjacent-pair multiplicities provide serialization
addresses. All complex joint entries are trainable. Each original class
recovers two learned, balanced encoders through a complete complex SVD and
a learned output magnitude. Inference actually executes both recovered
encoders and their complex interaction. The joint identity is a diagnostic
and supplies the exact pullback, not a substitute forward implementation.

Absent columns represent exact zeros under CM88. Every nonzero column,
original output class, real/imaginary coordinate and returned singular
direction remains present; no threshold or rank truncation is used. Thin SVD
is padded to the complete query width. Binary64 SVD, cross-entropy and finite
differences are numerical approximations, not machine-checked real arithmetic.
Recovery checks reconstruction and balancedness; each update also compares
actual complex inference with the independent joint expression.

| Mode | Ordinary CE update | Records |
| --- | --- | --- |
| Mini | Complete analytic pullback | Every supervised prefix in each original example batch |
| Forward-only Mini | Two finite CE bias queries per original class and record; CM88 reuses them for every responding real/imaginary coordinate at its response-scaled step | Same indexed examples, duplicates and short tails |

Forward-only training never calls the analytic CE derivative. Its coordinate
steps are response dependent, so it is not a common-step finite difference
of the whole batch. Neither the convexity theorem nor the identification
certificate proves that this numerical optimizer converges on these datasets.

`protocol.json` prospectively fixes a seed-1, five-case diagnostic cohort:
two short Boolean AND cases, modular addition, key-value binding and long
parity. The original native generators, tensor fingerprints, output vocabulary,
example batches, splits and scoring semantics come from `convex_stress` and
`convex_stress_minibatch`. Task names and labels never enter inference. Dataset
length metadata determines the declared context cap. All ten endpoints are
sealed before any held-out score; there is no validation-based selection.

The original success flag and threshold remain reported. Raw token accuracy,
exact sequence accuracy and CE are separate fields; the new target requires
both accuracies at least 99%. Scoring saves every split's metrics and a digest
of every complete raw-logit vector in evaluation order. A separate process
reloads the checkpoint and must reproduce all metrics and raw-logit bits.
This replay checks checkpoint/inference reproducibility; it is not an
independent repeat of training or evidence of three-seed coverage. Existing
transformer results are preserved; there is no matched timing comparison.

CM86 already proves a four-object parity restriction. Long parity is therefore
a capacity negative control. CM87 identifies an unseen answer only when the
observed constraints imply its margin within this family; it does not promise
that every original task supplies such constraints. Failures must remain in
the report. Native controls pass; held-out stress measurements remain pending.
Proof controls supply no stress-case credit.

Run from the repository root with native tooling and OpenBLAS installed:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/bilinear-native-v1
cargo test --offline --manifest-path experiments/convex_bilinear_sequence/Cargo.toml
cargo clippy --offline --manifest-path experiments/convex_bilinear_sequence/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_bilinear_sequence/Cargo.toml
"$CARGO_TARGET_DIR/release/stress" plan
```

Freeze the checked executable and protocol before training. For every declared
case/mode/seed, invoke `stress train WORKSPACE STORAGE COHORT CASE MODE SEED`;
after all endpoints exist, invoke `score` and then `verify` with the same
arguments. A cell directory must not already exist. No overwrite or resume
is implemented. Keep the freeze and generated evidence outside a temporary
worktree, linked through the ignored `runs` directory.

Before each allocation/update and each scoring split, the native guard checks
aggregate process RSS plus a conservative allocation reservation against
110,000,000,000 bytes. It checks both filesystems, requiring 2 GiB free on the
workspace filesystem and 16 GiB on storage. Insufficient RAM/disk stops the
operation without cleanup or model reduction. The model reservation ceiling
is 80 GB. Existing frozen producers must not be changed or restarted.

## Input-frequency update diagnostic

CM89 (`2a9bc07b`, `Updates/Scaling.lean`) proves positive input-only scaling
`w(z)=(N+1)/(1+sum_i response(x_i,z)^2)` on every complete coordinate. Bias
scales stay one. Repeated records and repeated pairs have their original
multiplicities; the latter are squared in the denominator. Scaling is shared
across classes and real/imaginary components only through the proved identical
squared response. It changes neither the model nor ordinary CE. The native
implementation checks integer counts before converting to binary64, retains
positive scales for unobserved coordinates and scales the complete batch mean.

The frozen unscaled control (`36eba937`, `bilinear-stress-v1`) finishes the
two short Boolean training runs at CE0.0260/0.0263 in both modes. Addition's
last training batch is CE4.5727 after200 updates, starting at4.5747. These
training observations motivate `scaled_protocol.json`: only addition, with
the same seed, 200 updates, rate0.01, batches, tails and query step. No
held-out metric was used to choose the change. The rescaling has no proved
convergence or unseen-answer guarantee.

Use `plan-scaled`, `train-scaled`, `score-scaled` and `verify-scaled` with a
fresh cohort directory and newly frozen executable. Both scaled endpoints
must be sealed before scoring. The eighth native control independently
compares all real/imaginary coordinate changes, duplicate responses and a
one-record tail against the complete unscaled update in both modes. This
comparison is separate from the unchanged original five-case producer.
