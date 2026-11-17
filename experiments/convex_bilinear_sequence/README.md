# Learned bilinear sequence stress probe

Can the CM86 complete complex sequence model learn original GPT stress tasks
with ordinary cross-entropy, using Mini and forward-only Mini updates? This
experiment implements the family proved in
`Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear`, including
CM87 identification and CM88 update identities. The manuscripts are CM86–88
in `review/flat_spectrum.md`; the proof commits are `ed9d359d`, `e9ac5beb` and
`e50d3691`. The older `convex_learned_sequence` experiment uses another family.
The seed-1 pilot is complete: both required modes pass two of the five original
primary tests. The additional scaled addition run fits every training answer
but gets no held-out answers correct. All twelve saved models reproduce every
scored raw-logit vector and metric on reload. The overall99% target is unmet.

The next theoretical step is CM91 (`LearnedBilinear/Capacity`): an explicit
squared capacity budget can pay for a nonzero identification residual. A
nonconstant three-record object-pair witness forces the held-out fourth pair
within capacity, while the unrestricted model remains ambiguous. This is a
proved possibility, not a new stress score. The original failed runs are
unchanged; selecting suitable shared structure and a training-only budget for
their unseen answers remains the active generalization problem. A sequence of
immutable objects still predicts one original object, with no numeric meaning
assigned to object IDs and no task-specific answer program in inference.

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
the report. Native controls pass; measured held-out results are recorded below.
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

## Recovery and storage cost control

Recovery now reads each complete class slice directly from the ordered column
map, avoiding a separate tree lookup for every query coordinate. The SVD inputs,
singular directions and reconstruction checks are unchanged. Checkpoint JSON
uses a buffered writer with the same create-new, newline, flush and disk-sync
semantics. Neither change is applied to an already frozen producer.

`recovery_bench WORKSPACE STORAGE` constructs100 original classes,101 query
coordinates and3255 dense stored columns. The complete recovered model is
checked at1200 input-only controls, including the empty prefix and maximum
length, retaining every output class and both complex components. Before and
after direct slice extraction, the output digest and both reconstruction and
balance residuals match exactly. One active-machine timing was19.26s versus
18.71s; this small difference does not establish a stress-training speedup.
The control is independent of stress data and supplies no accuracy credit.

Regenerate with `cargo build --offline --release --manifest-path
experiments/convex_bilinear_sequence/Cargo.toml --bin recovery_bench` and run
the resulting executable with the workspace/storage paths. To reproduce the
lookup reference, replace the direct class slice with the equivalent
`stored.iter().flat_map(|&d| (0..q).map(move |i| joint_ref.entry(class,i,d)))`
using `joint_ref=&joint`, as in commit7031f56b. Generated timing reports remain
local and ignored.

## Verified pilot findings, 2026-11-16

The frozen original producer is commit `36eba937`, stored under
`/run/media/misha/storage/transformer-runs/freezes/bilinear-stress-v1`.
Its complete source fingerprint is
`ff4a22405bd0382eed871dcd0dcbc1bb8588427fcf0e07a36094c53c0b8e6f68`.
The scaled addition producer is commit `7031f56b`, frozen as
`bilinear-scaled-v1`, with source fingerprint
`49342511119dce26c454d049cc1af261aa0aa08ddc3a6644b45e84e0c793041d`.
Generated metadata, training traces, checkpoints, scores and replay receipts
remain in the ignored `runs/original-v1` and `runs/scaled-v1` directories, backed
by that external storage. Both scoring queues finished with exit0.

All ten original endpoints were sealed before any held-out observation.
The scaled comparison also finished both endpoints before scoring. Every one
of the twelve separate checkpoint replays matched all metrics and complete
raw-logit digests. Mini and forward-only Mini have identical reported accuracy
counts in every split; their small CE differences remain in the raw reports.
This is a checkpoint/inference replay, not an independent repeat of training.

The table uses each unchanged original primary split, at seed1. Percentages
and CE are shown for Mini; forward-only Mini gives the same accuracies and CE
to the displayed precision. Historical transformer values are the original
first99 observations in `convex_stress/suite.json`, not new matched-budget runs.

| Original case | Primary examples | Updates | Raw token accuracy | Exact sequence accuracy | CE | Original success | Historical transformer exact |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Boolean AND, shift | 7 | 1000 | 100% | 100% | 0.0223324 | Yes | 100% at250 updates |
| Boolean AND, random | 8 | 1000 | 100% | 100% | 0.0252671 | Yes | 100% at250 updates |
| Modular addition | 4705 | 200 | 0.021254% | 0.021254% | 4.575985 | No | 99.1073% at3250 updates |
| Key-value binding / MQAR | 512 | 200 | 30.2734% | 30.2734% | 4.354632 | No | 99.2188% at500 updates |
| Parity, up to16 bits | 157 | 128 | 77.0701% | 54.1401% | 0.349339 | No | 99.3631% at9750 updates |

The original success flags are retained, so the two Boolean primary passes
count as passes. They do not imply all-split99 performance. The random-AND
ordinary test has95.3125% exact accuracy. On length32/64 and position-shifted
length probes, exact accuracy ranges from28.125% to68.75% across these two
tasks. In parity, every generated EOS is correct; raw token accuracy therefore
overstates answer correctness, which is85/157 on the primary split.

The fixed budgets are diagnostic and substantially shorter than the historical
transformer runs on addition and parity. Parity visits2048 of20000 training
examples, or0.1024 epochs. No conclusion about matched training speed or a
complete convergence comparison follows from this table. The CM86 global
four-object restriction nevertheless remains an independent expressivity
limitation of the family.

Input-only scaling isolates a different failure in addition:

| Addition update geometry | Training raw/exact | Full training CE | Held-out raw/exact | Held-out CE |
| --- | ---: | ---: | ---: | ---: |
| Unscaled | 4.42177% | 4.572614 | 1/4705 =0.021254% | 4.575985 |
| CM89 frequency scaled | 100% | 2.604803 | 0/4705 =0% | 4.654385 |

Both modes give these same counts with unchanged200-update budgets, rate0.01,
seed, samples, original classes and ordinary CE. Scaling improves fitting;
it does not recover missing answers. CM90 (`8a7283d7`, `Updates/Unobserved.lean`)
provides a precise related diagnosis: each unseen ordered two-object input
has a coordinate silent on every other observed pair. A fitting parameter can
change its query margin arbitrarily without changing any training logits.
The theorem is not a proof that every possible algorithm must fail; additional
structure or a different selection bias may favor the correct completion.

The next research problem is to connect unseen interactions through trainable
shared structure, and to supply sufficient context interactions for parity and
length transfer. More capacity or faster fitting alone is not an identification
argument. This pilot does not add35-task,165-task or three-seed coverage, and it
does not inherit the older fixed-family comparison's accuracy claims.
