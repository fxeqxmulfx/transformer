# Learned rotary recurrence for original associative recall

Does a learned rotary extension improve order-sensitive recall over complete
plain learned matrices? The external contract remains an immutable-object
sequence to one next original object. Original object IDs only address learned
parameters. Both variants receive the same original training batches and labels.

CM97/98 (`56ab91ca`, `review/flat_spectrum.md`, and Lean
`ConvexChannel/LearnedRecurrent/Pair` and `Rotary`) establish the complete
complex two-state representation, actual operator/output bounds, the joint
convex score domain, ordinary CE convexity and feasible finite loss queries.
CM98 adds52 proved theorems with no new sorry or extra axioms. Its same-parameter
swapped-prefix witness uses commuting content matrices, so rotary positions
supply the order distinction. These witness parameters appear only in a native
execution control, never in the training initializer or proposal search.

Each component has two complex initial coordinates, a full2x2 complex matrix
for every original object, and two complex readout coefficients for every
original class. The initial maximum norm is at most one. Each matrix/readout
row has absolute coefficient sum at most one. Plain execution applies the
object matrix at every occurrence. Rotary execution applies
`R(t) T[object] R(-t)`, where `R(t)=diag(exp(i*t*theta[0]),exp(i*t*theta[1]))`.
Both real frequencies are learned. The internal counter starts at zero and
increments once per raw input occurrence. There is no maximum-length parameter
or task feature map. The temporary phase cache is recomputed from the current
frequencies and actual minibatch prefix lengths.

The declared state norm is the product maximum norm. The bound does not assert
that arbitrary token-table Gram spectra remain flat under RoPE; the existing
Lean `SequenceModel/RotarySpectrum` counterexample rules out that inference.
Scalar shared conjugation cancels, which motivated the two-state extension.
The model is RoPE-inspired recurrent execution, not transformer attention.

Nonnegative mixture weights sum to fixedcapacity16. The zero component carries
unused mass; every added component remains stored. The convex domain contains
all admissible learned components. Physical matrix/frequency discovery uses an
approximate search and has no global convergence or unseen-answer guarantee.

| Frozen variant | Frequencies | Training |
|---|---|---|
| Plain matrices | No rotary parameters | Original binding, seed1,128 updates, Mini and forward Mini |
| Learned rotary matrices | Two learned real frequencies per component | Same data, seed, updates, capacity and original minibatches |

Both modes reserve the same feasible query centre with raw logit step0.0001.
Mini uses complete ordinary-CE residuals. Forward Mini uses two actual CE
values per original class and record; it uses no CE derivative or recurrent
backward pass. Every loss query is counted. The component search starts from
trainable identity parameters, random complex norm-preserving monomial
matrices/initial phases, and the preceding proposal when available. Rotary
random frequencies start from independent Gaussian(0,1) draws. Plain consumes
the same random draws so content initializations match. Every matrix entry is
subsequently searchable, including the initially zero entries.

Six sweeps use radii2,1,1/2,1/4,1/8,1/16. Each complex initial/matrix coordinate
gets four real/imaginary perturbations from the same coordinate centre. Initial
coordinates project to the unit disk; changed matrix rows project radially to
the unit row-l1 ball. Each real frequency gets both signed perturbations.
All original class readouts are proposed from their complete two-coordinate
CE-signal contractions: choose the negative conjugate direction of the larger
magnitude coefficient under the row-l1 constraint, with first-coordinate ties.
Zero coefficients remain stored numerical values; no class or parameter is
removed from the family. Actual full-class minibatch CE at feasible mixture
step `min(0.1,1/budget)` selects strict improvements. A64-iteration finite CE
line search chooses the final convex update.

Only matrix proposals for objects absent from every raw minibatch prefix can
be skipped, by CM98's exact absent-response theorem. Every object parameter,
class, original supervised target/mask, duplicate and short batch tail remains.
Float checks tolerate2e-12 operator excess and2e-10 times capacity for mixture
mass; these are numerical checks rather than formal real certificates.

The first cohort is a prospective diagnostic:128 original batches of32 examples,
two epochs of the2048-example binding train split, in each of four cells. All
four endpoints seal before any original split is scored. Report original source
success and raw/exact99 separately, including failures. Two epochs alone do not
establish convergence or answer whether more state capacity is necessary.
The scalar family's earlier five-task successes and the transformer baseline
remain separate, unchanged results.

Native controls compare complex execution with independent real4x4 matrix
composition, verify actual feasible signed-query models and long-input bounds,
check object renaming, zero-frequency/plain equivalence, order separation,
complete original-class finite CE signals, dense versus exact-absence search,
retained tiny coordinates, matched initializations, learnable off-diagonal and
frequency coordinates, actual learning of swapped orders in both modes,
checkpoint round-trip logits, original duplicates/tails and cohort seal refusal.
The small learning control is not a stress-test generalization measurement.
All ten controls, Clippy with warnings denied, formatting and shell checks pass.

Reproduce using native tooling:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/rotary-recurrence
export CARGO_BUILD_JOBS=2
cargo test --offline --locked --manifest-path experiments/convex_rotary_recurrence/Cargo.toml
cargo fmt --manifest-path experiments/convex_rotary_recurrence/Cargo.toml --check
cargo clippy --offline --locked --manifest-path experiments/convex_rotary_recurrence/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_rotary_recurrence/Cargo.toml --bin stress
"$CARGO_TARGET_DIR/release/stress" plan
```

Freeze the executable as `stress`, protocol, source commit/hash, binary SHA256
and source archive outside the worktree. Archive this experiment plus
`convex_bilinear_sequence`, `convex_stress_minibatch` and `convex_stress`.
Then run with a fresh cohort directory:

```sh
bash experiments/convex_rotary_recurrence/run.sh WORKSPACE STORAGE FREEZE COHORT
```

Individual commands are `stress train|score|verify WORKSPACE STORAGE COHORT CASE
plain|rotary mini|forward SEED`. The runner uses two mode queues, seals both
families before scoring, then reloads and replays every raw logit and metric.
Original split fingerprints, producer fingerprint, executable hash, checkpoint
hashes and original metric thresholds are checked. Outputs are create-new only;
failed runs stay visible. Generated evidence lives outside the worktree.
Resource guards preserve the110GB aggregate RAM limit,2GiB workspace free and
16GiB storage free minima. Low disk space stops work without cleanup.

## First original binding comparison, 2026-11-17

Producer `15b2bd3e`, freeze `rotary-binding-v1`, source fingerprint
`463197b3624b4b30c70459a6f40bc5f4509463105c398d677c100baeca69d223`, binary SHA256
`5f7e2b1d63cede287ae59ca5f0e96b6b4d635f343b107041608b604d062eb204`.
Generated evidence remains in
`/run/media/misha/storage/transformer-runs/rotary-recurrence/binding-v1`.
All twelve train/score/replay commands exit0. All four complete raw-logit and
metric replays match. Original batch-index traces match across the four cells.
Each model uses4096 original records, two epochs, and retains129 components.

| Family / mode | Full train correct /2048 | Novel validation correct /512 | Test correct /512 | Novel validation CE |
|---|---:|---:|---:|---:|
| Plain / Mini |62 (3.027%)|14 (2.734%)|9 (1.758%)|13.68627|
| Plain / forward |72 (3.516%)|11 (2.148%)|19 (3.711%)|13.83214|
| Learned rotary / Mini |92 (4.492%)|19 (3.711%)|20 (3.906%)|15.93211|
| Learned rotary / forward |86 (4.199%)|15 (2.930%)|15 (2.930%)|10.91809|

There is one supervised answer per example, so raw and exact accuracy coincide.
The original validation and novel-validation split reports coincide here, as in
the original data. Every original success and raw/exact99 flag is false. This
short diagnostic does not support a useful recall advantage for learned rotary
positions. It also does not establish that the family lacks a fitting solution:
full-training accuracy is poor, and no global component oracle is available.

The training trace shows a strong instability. Mean mixture step is0.89951,
0.90227,0.90386 and0.89211 respectively. A typical step therefore replaces
about90% of existing mixture mass with a component selected on32 examples.
Mean current-batch CE falls from14.07/14.12/14.33/14.12 before the update to
1.77/1.76/1.70/1.80 afterward, while full-training CE finishes at
12.96005/13.51785/15.45444/11.54355. All exceed the initial uniform-class
CE `log(100)=4.60517`. This is consistent with destructive interference across
minibatches; it is not a proof that interference alone explains the failure.
A capacity-feasible smaller-step comparison is needed before attributing the
result to positional representation or claiming generalization progress.

Plain runs use71910880 component-loss evaluations; rotary runs use72205024.
Every cell uses278528 line-search loss evaluations. Each forward cell adds
819200 signed class-bias CE queries. Mean/runtime comparisons are descriptive,
not matched to transformer compute: training takes101.59/103.87/106.25/106.93s
in the table order. The rotary learner searches both real frequencies and every
responding complex initial/matrix coordinate; no fixed task weights are used.
All failed checkpoints and raw-logit hashes remain frozen.

The user next requested a fixed-rotation alternative. Compare plain matrices,
fixed positional frequencies and learned frequencies under the same more
conservative update rule. Fixed rotations are an explicit architectural control;
all initial/content/readout parameters remain learned. Preserve these original
unrestricted-step results and prove the feasible fixed-frequency queries before
using that new control. No99% recall claim is made by this experiment.
