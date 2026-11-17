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

## Prospective fixed/plain/learned comparison with capped updates

The CM99 proof extension at `18c7409b` defines the fixed-frequency convex domain,
proves actual same-frequency bias execution and finite-query feasibility, and
provides the concrete CE interference counterexample and step-bound theorem.
All26 new theorems pass full Lean, direct/global axioms, source-warning, native
INDEX and forbidden checks. No convergence result is inferred.

`cautious_protocol.json` freezes three families at the same original128 updates,
seed1, capacity16, original batches and both modes. Plain retains full learned
matrices without rotations. Fixed uses frequencies(1,0.01) in every component
and searches every initial/content/readout coordinate. Learned rotary starts
its identity and random-content proposals at the identical(1,0.01) frequencies
and then searches both of them; previous proposals retain their learned values.
Matched content RNG draws remain identical across all three families. Thus
fixed versus learned begins from the same parameter points. Against the older
learned-rotary producer, both the step policy and frequency initialization change;
only the plain-model comparison isolates the step policy alone.

At zero-based update t, the accepted convex CE line search runs on
`[0,2/(t+2)]`. Its two endpoints and64 inner iterations use actual complete CE.
The component-proposal comparison remains the original fixed feasible step
`min(0.1,1/budget)`; only the accepted mixture update is capped. Every finite
query, original class and record remains present. Fixed-frequency zero and bias
components use the same fixed schedule, as required by CM99. Stored versus
trainable numerical coordinates are reported separately:1206 stored/1204
trainable for fixed,1206/1206 for learned rotary,1204/1204 for plain on this
100-object task. Checkpoint validation rejects a changed fixed schedule.

All six endpoints seal before scoring, followed by complete independent raw
replays. Retain all failed original splits and both original and raw/exact99
success flags. The extra controls cover all three families, matched starts,
fixed-frequency query execution, the actual CE counterexample and capped
line-search/step behavior. A cap is a feasibility and stability experiment;
no extra held-out accuracy is credited before measurements.
All fifteen native controls, Clippy, formatting and shell checks pass.

Use the native build commands above, `stress plan-cautious`, and a fresh frozen
source/executable directory with `cautious_protocol.json` copied to its
`protocol.json`. Then run:

```sh
bash experiments/convex_rotary_recurrence/run.sh WORKSPACE STORAGE FREEZE COHORT -cautious
```

Individual commands gain the same suffix: `train-cautious`, `score-cautious`,
`verify-cautious`; family is `plain`, `fixed` or `rotary`. The original failed
producer remains immutable and is replayed using its original frozen binary.

The cautious comparison is complete at producer `ff00f921`, freeze
`rotary-cautious-v1`, source fingerprint
`f75861a34af01cca344cc7576c77d1f1f0070fa97d127beb8ed249218d581280`, binary SHA256
`a5f24cf06a8a70df48baea016132984594820e1e06b8cbc2577727b66fba2da1`.
Evidence remains in
`/run/media/misha/storage/transformer-runs/rotary-recurrence/cautious-v1`.
All eighteen train/score/replay commands exit0, and all six complete raw-logit
and metric replays are identical. Each cell retains4096 original records, two
epochs,129 components and all original split reports. Every accepted step obeys
the prospective cap; all end at2/129 and the mean step is0.0693890.

| Family / mode | Full train correct /2048 | Novel validation correct /512 | Test correct /512 | Novel validation CE |
|---|---:|---:|---:|---:|
| Plain / Mini |171 (8.350%)|27 (5.273%)|17 (3.320%)|3.77337|
| Plain / forward |166 (8.105%)|15 (2.930%)|19 (3.711%)|3.80737|
| Fixed / Mini |174 (8.496%)|10 (1.953%)|26 (5.078%)|3.76373|
| Fixed / forward |180 (8.789%)|24 (4.688%)|16 (3.125%)|3.69763|
| Learned rotary / Mini |174 (8.496%)|10 (1.953%)|26 (5.078%)|3.76373|
| Learned rotary / forward |180 (8.789%)|24 (4.688%)|16 (3.125%)|3.69763|

All six original success and raw/exact99 flags are false. Raw and exact coincide
because each example has one answer. Capping updates lowers full-training CE to
3.459–3.484 and improves training accuracy relative to the first comparison,
but recall remains poorly fitted. The plain-model result isolates this update
change: its novel CE falls from13.686/13.832 to3.773/3.807, while accuracy remains
5.273%/2.930%. This supports improved stability, not a solution to recall.

Fixed and learned rotations have **exactly equal complete split reports,
including every raw-logit hash**, in each mode. Their mixture weights and all
nonzero-component parameters are also exactly equal. Every nonzero component
in the learned model retains frequencies(1,0.01); the separate zero component
uses(0,0) and has identically zero output. This occurs despite9192 frequency
proposal trials per learned run. There are thousands of nonzero off-diagonal
content entries, so a claim that all matrices became diagonal would be false.
The measurements do not establish that the family or its frequency directions
are globally unhelpful: the numerical proposal search and low full-training
accuracy remain substantial limitations.

Plain/fixed cells each use71910880 component CE evaluations; learned rotary uses
72205024. Every cell uses278528 line-search evaluations and every forward cell
adds819200 class-bias queries. Training takes102.26/102.99s for plain,
107.01/108.20s for fixed and107.38/107.27s for learned, Mini/forward respectively.
These are descriptive concurrent-run timings, not matched transformer budgets.
The next investigation should distinguish insufficient component discovery,
state/readout geometry and training budget before adding another positional
formula. The broader generalization goal and99% recall target remain unresolved.

## Training-only frequency search diagnosis

The `diagnose` binary reads the immutable cautious learned-rotary cell, recreates
all128 original batches and numerical updates, and requires every deterministic
receipt and the final checkpoint bytes to match exactly. It retains the exact
pre-update query centre of the last step. This uses only the original training
split; no held-out target or score selects a perturbation. The source producer
fingerprint is checked explicitly.

On the last selected component and its original final minibatch, the diagnostic
refits the same complete original-class readouts and requires the recorded
component-comparison CE to match exactly. It then measures both frequency
coordinates, both signs, and every dyadic radius from2 through2^-20:88 directions
in total. The first six radii are the original search grid; the rest test finer
resolution. Every direction and loss is retained, including no improvement.
The coarse directions are measured at the final component, whereas the original
search visited some of them earlier with different content parameters.

For each direction, report CE both before and after refitting the readouts,
complete actual state response and numerical capacity excess. A separately
constructed feasible mixture model checks the component-comparison CE to2e-12.
All matrix/readout entries, original classes and records remain present.
CM98's frequency-independent Unit bound justifies these model queries without
a new architecture assumption. The diagnostic accepts no update, measures no
new held-out accuracy and makes no global optimizer claim. Its scope is the
last chosen component, not every component or every point in the full domain.

```sh
cargo build --offline --locked --release --manifest-path experiments/convex_rotary_recurrence/Cargo.toml --bin diagnose
"$CARGO_TARGET_DIR/release/diagnose" WORKSPACE STORAGE CAUTIOUS_COHORT FRESH_OUTPUT mini
"$CARGO_TARGET_DIR/release/diagnose" WORKSPACE STORAGE CAUTIOUS_COHORT OTHER_FRESH_OUTPUT forward
```

Freeze this diagnostic executable, source archive, commit and binary hash before
running it. It writes a byte-replay record and all direction measurements outside
the worktree. The existing110GB RAM and both-disk guards apply throughout replay.

The diagnosis completes at producer `1da2f96c`, freeze
`rotary-frequency-diagnostic-v1`, diagnostic fingerprint
`f80e48c18ca3ba5c18c3e30f930ac81ba39f7787c580a8d1749098b84f170596`, binary SHA256
`d2051377e56e6df63cc2abea72156c49dff5716c11cb4582d6db7eda0328ba2d`.
Evidence is in
`/run/media/misha/storage/transformer-runs/diagnostics/rotary-frequency-v1`.
Both runs exit0. Every one of128 original batch hashes and deterministic
receipts matches in each mode, and both endpoint files reconstruct byte-for-byte.
The final component and its recorded comparison CE also reconstruct exactly.

| Mode | Recorded component CE | Best old-grid CE at final component | Best finer CE | Finer first-frequency change |
|---|---:|---:|---:|---:|
| Mini |2.730541439235010|2.734987597286415|2.730395431702667|+0.0078125|
| Forward Mini |2.547022993748942|2.558957538262184|2.547018666395651|-0.0009765625|

Every old-grid perturbation worsens the criterion at this final component,
whereas a finer feasible perturbation improves it in both modes. The gains
are1.4600753e-4 and4.3273533e-6. Independently executed mixture-model CE confirms
them, with discrepancies at most ordinary floating rounding at the selected
points (4.44e-16 and0), far below either gain. Maximum operator-budget excess
is2.22e-16. The state responses are nonzero. Mini's listed improvement requires
the complete readout refit; holding its readouts fixed at that same perturbation
would worsen CE. All88 directions and both readout variants remain recorded.

This establishes a numerical resolution failure at the inspected final
components; it does not show that resolution is the sole recall bottleneck.
No held-out label entered the diagnosis, no proposed model was accepted, and no
new accuracy result is claimed. The next controlled repair adds finer frequency
proposals within the already proved CM98/99 domain and repeats original recall
with every other protocol choice retained.

## Finer frequency search

Task379 adds a controlled repair of the measured training-only resolution
failure. `refined_protocol.json` retains the cautious learned-rotary data,
seed1, both modes,128 updates, matched initialization, six full component sweeps,
capacity16 and capped accepted steps. After the existing search, each candidate
start gets16 finer frequency-only levels, from1/32 through2^-20. Both signs and
both coordinates are queried. Every proposal refits all original-class readouts
and uses the same complete CE comparison. Initial states and content matrices
still receive their complete original search; no parameter or record is removed.

This changes numerical search resolution inside the CM98/99 proved domain.
It adds no architectural or convergence assumption. Zero extra levels is the
legacy default and must reproduce all128 deterministic receipts and the final
checkpoint bytes of both cautious learned-rotary runs before this repair trains.
A separate numerical fixture has no improving old-grid frequency direction,
but the finer search lowers actual feasible mixture CE in both modes. It also
checks preservation of every initial/content coordinate and finite-query counts.
The aggregate training receipt counts component, class-bias and line-search CE
queries separately, adds the two centre/validation CE calls per training record,
and reports their total. Legacy per-update receipts retain their exact format.

Freeze the native executable, source archive, commit and binary hashes, and copy
`refined_protocol.json` to the freeze's `protocol.json`. Keep the earlier
producers and their failed results intact. After the native checks and the two
legacy `diagnose` replays above, run:

```sh
"$CARGO_TARGET_DIR/release/stress" plan-refined
bash experiments/convex_rotary_recurrence/run.sh WORKSPACE STORAGE FREEZE FRESH_COHORT -refined
```

Both new endpoints seal before any held-out scoring; every original split and
failure is retained, and both complete raw-logit replays must match. This
training-only repair earns no accuracy credit before the measurements. Aggregate
RAM stays below110GB and both disk guards apply throughout training and replay.

All17 native controls, Clippy, formatting and shell checks pass. With extra
refinement disabled, both legacy runs reproduce all128 deterministic receipts
and endpoint bytes exactly, and both complete88-direction probe files match
the earlier diagnostic byte-for-byte. Evidence is stored outside the worktree
at `diagnostics/rotary-refinement-legacy-v1` under the shared runs directory.

The refined comparison completes at producer `534a258f`, freeze
`rotary-frequency-refined-v1`, source fingerprint
`0f0552315bfd778e2f10ac1aac56cd1593ed4b3aa284a85ae46c2e787ebb859f`, binary SHA256
`3a67f9ee693e0fa1248fe6b84d9f5aa7debcb7f5327f82f5eef45dd7f85667b6`.
All six train/score/replay commands exit0. Both complete raw-logit and metric
replays match. Evidence remains in
`/run/media/misha/storage/transformer-runs/rotary-recurrence/refined-v1`.
Both original batch traces equal the cautious producer exactly, and every
accepted step satisfies the same prospective cap. Each run retains4096 records,
two epochs,129 components, all100 original classes and every original split.

| Mode | Full train correct /2048 | Novel validation correct /512 | Test correct /512 | Train CE | Novel CE |
|---|---:|---:|---:|---:|---:|
| Mini |176 (8.594%)|16 (3.125%)|20 (3.906%)|3.51723|3.87758|
| Forward Mini |196 (9.570%)|11 (2.148%)|20 (3.906%)|3.51547|3.77392|

Both original success and raw/exact99 flags remain false. The finer search now
changes the frequencies of all128 nonzero components in each run. First-frequency
ranges are0.977891–1.014997 for Mini and0.984792–1.017882 for forward. Thus the
previous failure to leave the initial schedule is repaired, but original recall
is not. Against the cautious learned/fixed control, both full-training and novel
CE worsen; the small accuracy changes do not establish improved generalization.

Each run uses33704 frequency trials and73001664 component CE calls, compared
with9192 and72205024 before refinement. Each also uses278528 line-search calls
and8192 centre/validation calls; forward adds819200 class-bias queries. Complete
CE totals are73288384 and74107584. Concurrent training times are107.25/107.74s;
these remain descriptive timings, not matched transformer compute budgets.
The training-only diagnosis identified a real local search defect, but fixing
that defect is insufficient. The next investigation should measure state
retention and component fitting before selecting a larger memory or another
operator geometry. No new task reaches99%, and the broader goal remains open.
