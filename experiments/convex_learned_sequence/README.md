# Complete learned-sequence update kernels

Can native updates retain the entire CM75–CM78 complex sequence family while
satisfying the activation domain required by its ordinary cross-entropy proof?
The first component implements the exact projection certificate proved in
CM79, commit `945603c5`, under
`src/Transformer/Grokking/FlatSpectrum/ConvexChannel/LearnedHinge/Projection`.

The native model now executes the full learned complex forward and projects
the whole parameter array with retained component certificates. Complete cone
centres, direction bases and feasible queries now have native generators and
independent exact certificate checks. The Mini proposal rule now uses the full
actual CE direction and certified projection. Forward-only Mini now queries
every certified basis direction and supports complete independent update replay.
Sealed pilot training and independent full replay are complete, as recorded
below. Known feasible points used by controls are not learned outcomes; the
pilot adds no stress-case accuracy credit.

## Complete domains and controls

| Object set / context cap | All prefixes | All scalar patterns | Coordinates per complex vector | Complete real parameter count |
| --- | ---: | ---: | ---: | ---: |
| 2 objects / 2 positions | 7 | 128 | 5 | 5,120 |
| 3 objects / 1 position | 4 | 16 | 4 | 768 |

Object integers are serialization identities, not arithmetic features. Inputs
remain sequences of original immutable objects. The raw row contains the bias
coordinate and one position/object address per visible position. CM79 proves
that its dot product with learned weights is the original raw lookup. The
rows here describe update constraints; they do not supply an inference model.

`DomainPlan` counts every bounded prefix, all scalar masks, both complex
components, every original output object and both readout signs before domain
allocation. An oversized declaration is refused. Inputs exceeding the context
cap, invalid objects and invalid masks are refused without truncation. Empty
prefixes and all original rows remain present.

`ProjectionProblem` generates candidates by enumerating active row subsets
and solving their Gram systems with exact rational Gauss–Jordan elimination.
Singular active systems cannot be inverted; their original inequalities remain
in every verification. A separate verifier checks all four CM79 conditions:
nonnegative multipliers, primal feasibility, complementarity and stationarity.
An accepted certificate proves global nearest-point optimality, so candidate
enumeration can stop once it has one. If no certificate is found, it refuses.
Solver completeness and practical full-suite cost are not inferred from this
finite search or from the controls.

Finite binary64 proposals are imported as their exact dyadic rational values.
Projection outputs stay rational, including denominators such as five. There
is no rounding back to an approximately feasible floating-point model in this
component. Nonfinite input, shape mismatch, failed certificate conditions,
arithmetic overflow, coefficient-bit limits and search/storage limits produce
explicit errors. Rational arithmetic uses the pinned `num-bigint` and
`num-rational` crates. This Rust implementation is tested against the Lean
specification; it is not itself a compiled or formally verified Lean program.

## Actual complex model and atomic updates

`ComplexModel` retains all rational coordinates in pattern/class/readout-sign/
component/coordinate order. Native sign zero denotes Lean's positive (`true`)
readout, and component zero denotes the real (`true`) component. Constructors
check every original prefix inequality before accepting the parameter array.

Each complex vector `u` recovers amplitude `sqrt(norm(u))` and representation
`u/amplitude`. The norm is the maximum coordinate complex magnitude, matching
Lean's function-space instance; the projection metric remains Euclidean in
joint coordinates. The forward reads the recovered learned bias and visible
position/object values, rectifies both components, multiplies by the complex
readout `amplitude*(1+i)` and sums signed neurons. It executes both learned
stages. A gated linear expression is not substituted for inference.

The physical-to-joint constructor follows CM76's exact component-pattern
aggregation. The optional CM78 paired forward calls the same model twice,
renames every input object, aligns every output class and averages complete
complex outputs. The renaming must be an involutive permutation of the entire
original object set. It is an explicit task restriction, not a universal fix.

Real logits enter stable full-class softmax and ordinary cross-entropy. Only
the loss accepts a target. The predictor returns an original object index;
ties select the first index while retaining every output class. Rational
parameters remain authoritative. Factor recovery, complex multiplication and
probabilities use binary64 and are qualified numerical approximations to the
Lean model. Nonfinite conversion/evaluation and detected underflow are refused;
no strict numerical-error enclosure or exact binary64 equivalence is claimed.

Whole-model projection checks every pattern/class/sign/component, retains every
full dual array and verifies that every certificate point matches the new
parameter array. Construction and independent replay must succeed before a
new model is returned. Failure leaves the old immutable model untouched. One
cone workspace is reused sequentially; no model pattern is removed. A
conservative working-memory reservation charges old/proposed/new state,
certificates, rational work and factor caches, with external RSS guards retained.

## Complete geometry for feasible forward queries

`ConeGeometry` implements CM80 and CM81, Lean commits `eb1634bd` and `2d2ca3ae`.
It projects every original normal, retains every full dual certificate and sums
the projected points. Independent replay checks the projections, complete centre
sum, all row slacks and forced-equality flags. Zero and redundant original rows
stay present. The auxiliary equality matrix zeros only non-forced rows; the
original cone remains intact in every feasibility and query check.

Exact Gauss–Jordan elimination retains the entire equality matrix and its full
row-operation matrix. It proposes full-coordinate basis vectors, free-coordinate
indices and recovery coefficients against every equality row. A separate checker
tests all three CM81 identities: kernel membership, identity on free coordinates,
and reconstruction of every original coordinate residual. The Lean theorem
then establishes coverage and uniqueness, rather than trusting a generated rank.
Deleting a feasible basis direction fails the reconstruction check.

Queries accept a complete direction and a positive rational step cap. Every
original row supplies its absolute-response bound; a tight row with nonzero
response refuses the query. The final two points and every bound are independently
replayed. Positive centre mixtures are available to open non-forced boundary
rows. Mixing changes the iterate and can increase loss, as recorded in CM80.
All returned points remain rational. Full models pass the complete constructor;
single-component queries use the equivalent CM83 local replacement described
below before actual recovered complex inference.

`recover_pairings` solves the complete basis Gram system, retaining all
off-diagonal entries, and reconstructs a vector in the full equality kernel.
Independent acceptance rechecks its kernel reconstruction and every supplied
pairing. A certified empty kernel returns a full zero vector; missing pairings
refuse. This implements CM81's exact-pairing premise for supplied rational
values. It does not establish that numerical finite differences equal true
loss derivatives. Merely using the measured pairings as basis coefficients
gives the wrong result on a nonorthogonal basis.

`GeometryPlan` admits the number of normal projections, the worst-case total
active-subset candidates, rational entry count and conservative byte reservation
before further construction. The reservation includes projection workspace,
normal certificates, elimination transforms, basis/recovery arrays and replay
buffers. It describes one complete cone, not an all-pattern training-memory or
runtime guarantee. Full-family caching and optimizer state need their own total
admission; the 110 GB aggregate RSS and both-disk guards remain applicable.

## Complete ordinary CE Mini updates

CM82, Lean commit `ba4b4b93`, derives the actual CE first-order bound and
feasible-query derivative for every raw coordinate, both shared passes and
indexed mean minibatches. `model::learning` implements the complete analytic
proposal. Actual complex inference supplies the full-class softmax residual;
the input-only row, readout sign, component sign and activation branch pull
that residual through every joint coordinate. The target enters only this
ordinary loss calculation. A shared prediction has one CE residual from its
averaged logits; half is pulled through each pass, with inverse class alignment
on the renamed pass. Native renamings are checked involutions.

Each binary64 probability is imported as its exact dyadic value, and the
subtraction of the target, pullback, record accumulation, actual-batch mean and
learning-rate proposal then use checked rational arithmetic. This makes the
residual a numerical approximation to the Lean derivative; it does not establish
an exact floating derivative or a rounding-error bound. Inference continues to
recover and evaluate the entire learned complex computation.

`Record` slices preserve all occurrence indices. Empty batches refuse;
one-record tails normalize by one. `MiniPlan` admits all records, both passes
when selected, every coordinate contribution and the worst-case active-subset
count across every projected cell. Its storage reservation adds the retained
gradient/proposal and residual buffers to the complete model/projection plan.
These accounting limits complement external aggregate RSS and both-disk guards;
they are not a wall-clock or measured allocator bound. No full-family cache is
introduced, and no mask, coordinate, class or record is sampled away.

`mini_step` forms the complete proposal and calls the existing atomic whole-model
projection. The returned result keeps the proposal, complete gradient, mean
training CE, resource plan, new model and every component certificate, allowing
independent replay. Failed admission, invalid records/renaming/rate, coefficient
growth or projection refusal leaves the original model intact. A step does not
by itself guarantee loss reduction at an arbitrary learning rate.

## Forward-only ordinary CE Mini updates

`model::forward_learning` composes CM79--82 without calling the analytic gradient
or pullback. `ForwardConfig` fixes a positive rate, positive centre mixture at
most one, and positive query cap. The update constructs every pattern's checked
geometry and mixes each original class/sign/component vector with its centre.
The whole mixed array passes the original model constructor and factor recovery.
Old and mixed mean CE are both retained: mixing may increase the loss.

Every basis direction of every cell receives two exact feasible queries. Each
query replaces its whole component vector, preserves every original inequality,
and is evaluated by the actual recovered complex forward and full-class CE over
every indexed batch record. Paired predictions use the same two complete shared
passes as ordinary inference. One query replacement is live at a time; no model mask,
class, component or input coordinate is omitted. Every cell has a receipt,
including an empty list only when its certified kernel has no directions.

The two binary64 mean losses are imported as exact dyadics, subtracted, and
divided rationally by twice the exact query step. Complete Gram recovery yields
a direction in the full forced-equality kernel. A proposal from the mixed
iterate, using all recovered coordinates, goes through atomic whole-model
projection. All pairings, steps, query losses, full gradient/proposal, mixed and
new models, projection certificates and final CE are retained in `ForwardUpdate`.
Finite differences remain numerical estimates of CM82's exact derivative;
there is no finite-step, rounding-error or convergence guarantee.

`verify_forward_update` does not call the update generator or an analytic
gradient. It regenerates each checked geometry, checks the entire mixture,
reconstructs every feasible query, re-evaluates every actual batch CE with exact
binary64 receipt comparison, verifies each pairing and the recovered kernel
vector, and checks the full proposal and projection certificates. The final
model is reconstructed under the original model's arithmetic limits. Replacing
it with a model admitted under looser limits cannot bypass replay checks.

`ForwardPlan` admits two complete geometry constructions per pattern, all
normal-projection and final-projection candidates, the full upper bound on basis
directions, all batch evaluations and original forward passes, full-coordinate
forward work, every query-model inequality check, and simultaneous storage for
models, factors, receipts, gradient, proposal and certificates. The per-cone
geometry workspace is reused sequentially. These explicit counts and memory
reservations complement aggregate RAM and both-disk guards; they do not promise
a wall-clock runtime. Generation and independent replay are separate admitted
operations. Records remain indexed and a short batch uses its actual length.

## Local query reconstruction

CM83, Lean commit `ce6970ab`, proves that replacing one complete real or
imaginary vector preserves the full complex domain when the old model is
feasible and the new vector satisfies every original row of its selected cone.
It also proves the full complex forward equals the original per-slot sum with
one whole neuron replaced. No unchanged model coordinate or constraint is removed.

`model::query` implements a private validated replacement tied by an immutable
borrow to its original model. It checks the full vector length, aligned cell
address, coefficient limits and every original oriented row. It then recovers
the entire changed complex neuron using the new component and the old opposite
component. Both the learned representation and readout can change. Every other
neuron uses its existing learned factors. The scorer retains the original
mask/class/sign order and evaluates every actual complex product, with the same
replacement in both paired passes and the same full-class softmax and CE.

For P real parameters, D coordinates per component and K raw prefixes, query
validation changes from checking P*K row terms to D*K; factor recovery changes
from P/(2*D) whole neurons to one. Actual forward evaluation still visits every
neuron. The original conservative `ForwardPlan` bounds remain in force, including
the larger full-reconstruction reservation. The exponential family size is
unchanged. This is reuse of unchanged learned factors, not a fixed task-feature
cache or a new training rule. Binary64 comparison to the previous reconstruction
is separate from the exact Lean proof and from numerical error relative to it.

## Findings and validation

All ten native controls pass. They include an independent 125-point orthant
clipping reference, a slanted projection with exact nonbinary coordinates,
redundant and degenerate constraints, individual corruptions of each certificate
condition, exact binary64 import, complete raw-coordinate lookup, input/budget
refusals and all-pattern coverage. Two distinct proposals are certified for
every one of the 128 two-object patterns; a further proposal is certified for
each of the 16 three-object patterns. These 272 checks retain every row in their
respective domains. They are projection controls, not learning outcomes.

Nine additional model controls pass, for nineteen total. Actual recovered
forwards reproduce CM76's complex outputs `-7+7i` and `4+10i` with differing
component masks, as well as signed class readouts and empty-input behavior.
The CM78 feasible witness returns its two observed objects and the renamed
unseen object, with all three class CEs and whole complex equivariance checked.
This is construction of the proved witness, not a trained model or an accuracy
measurement. A full 768-coordinate projection retains and replays 192 component
certificates. Corrupted or incomplete certificates, infeasible arrays, invalid
objects/renamings and numerical/resource failures are rejected atomically.

Nine further geometry controls pass, for twenty-eight total. They cover the
degenerate diagonal cone, a slanted plane with rational nonorthogonal basis
vectors and recovery, zero and unconstrained cones, lineality, boundary
mixing, malformed queries, resource refusals and individual certificate
corruptions. Every one of the 144 patterns across the two complete domains
gets checked geometry. Across these controls, all 960 normal projections carry
their full certificates, and every certified basis direction is tested with
feasible queries. A full model with
all sixteen three-object patterns executes two actual complex query forwards:
the selected real and imaginary score changes are both `1/16`, with ordinary
full-class CE changing in the expected directions on every original prefix.
This is a known feasible control, not training or a new accuracy measurement.

Two pairing-recovery controls bring the total to thirty. On the plane
`2x+y+z=0`, requested pairings `(2,4)` recover `(-2,1,3)` exactly; the naive
coefficient interpretation produces `(-3,2,4)` and is rejected. Empty-kernel,
missing-pairing and invalid-direction cases are checked. The complete-pattern
control now also recovers all supplied pairings and makes a feasible query
along the recovered vector for every one of the 144 patterns.

Four Mini controls bring the total to thirty-four. The CM82 nonzero witness
reproduces the exact full-coordinate mean bias slope -1/6 for targets
true,false,true; deduplicating gives zero and the one-record tail gives -1/2.
Both complex signs/components and the distinct visible object coordinates are
checked. A nonuniform three-class paired prediction agrees with independent
actual-CE central differences along all 48 coordinates of its all-active cells,
including the inverse output alignment. These finite probes validate the native
implementation locally; they do not replace CM82's general exact derivative.
Base and paired Mini steps both reduce the observed batch CE in their small
controls, retain all 768 parameters and 192 component certificates, and replay
every certificate. Empty/invalid records, a mismatched renaming, nonpositive
rate, overflow and each complete work/storage limit refuse atomically.

Five forward-only controls bring the total to thirty-nine. A complete two-object,
one-position update retains 192 coordinates, 64 cells and all 192 basis probes,
matches the CM82 repeated-record mean slope and the analytic direction within
the stated numerical test tolerance, lowers the observed batch CE, and replays
all queries and projection certificates. The three-object paired control retains
768 coordinates, 192 cells and all 768 probes on a nonuniform prediction and a
one-record tail. Its centre mixture measurably increases CE; its update lowers
CE relative to that mixed point. The complete trace replays independently.

A separate cell control finds a forced-equality cone with a nonorthogonal basis
inside the complete two-position raw domain, queries all of that cell's feasible
directions through the full model, and checks every recovered pairing against
actual CE work. It is a cell control, not a whole two-position training run.
Replay rejects missing cells/directions, wrong coordinate identities, altered
steps/losses/pairings, corrupted gradient/mixture/proposal/certificates and false
final loss or plan. Every total work/storage limit, invalid input/configuration,
and overflow is checked. Analytic gradients appear only in test comparisons,
never in the forward-only implementation. These are local implementation
controls; they provide no new stress-case or sealed-pilot accuracy credit.

Five local-replacement controls bring the library total to forty-four. All
192 component cells of the complete three-object family are checked with both
zero and nonzero replacements: 384 full reconstructions, each evaluated on all
four prefixes, base and paired. Every complex score component, probability,
original object and every class CE agrees bit for bit. A complete two-position
model also checks every basis probe in a forced-equality nonorthogonal cell,
for both signs/components and both classes. A separate test compares all 256
cell/batch/pairing combinations of repeated records and one-record tails.
Shape and index errors, a violation only at the last original row, coefficient
budget, numerical overflow/underflow, invalid inputs and renaming all refuse.
These tests retain the public full constructor as their slow reference.

Development tests, strict all-target Clippy and a release build pass. The Lean
projection/query/learning/replacement components independently pass the full build, 107
direct axiom checks, global audit, native INDEX verification and forbidden-source
scan. Existing 36 `sorry` declarations remain; no new `sorry` or extra axiom was added.
Evidence and failed development logs stay ignored under
`.kanban/artifacts/366/`; generated output is not committed.

From the repository root, with build output on storage:

```sh
export CARGO_BUILD_JOBS=1
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/learned-sequence-v1
cargo fmt --check --manifest-path experiments/convex_learned_sequence/Cargo.toml
prlimit --as=4294967296:4294967296 -- cargo test --offline --locked --manifest-path experiments/convex_learned_sequence/Cargo.toml -- --test-threads=1
prlimit --as=4294967296:4294967296 -- cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_learned_sequence/Cargo.toml -- -D warnings
prlimit --as=4294967296:4294967296 -- cargo build --offline --locked --release --manifest-path experiments/convex_learned_sequence/Cargo.toml
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ProjectionHingeAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/QueryGeometryAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/QueryBasisAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ReplacementHingeAxioms.lean
```

The development/review wrappers check aggregate RSS plus the process allowance
against 110,000,000,000 bytes and require free space on both disks before each
heavy command. Matrix-entry, rational-bit and active-subset limits are internal
representation/work limits, not a claimed end-to-end RAM forecast.

## Prospective sealed pilot protocol

`learned-sequence-pilot` uses the complete three-object, one-position family:
four prefixes, sixteen patterns and 768 real parameters. Both Mini and
forward-only Mini use the shared involution `[0,2,1]`, zero initialization,
eight epochs, ordered batches of size two, rate `1/32`, and ordinary mean CE.
Forward-only Mini uses centre mixture `1/1024` and query cap `1/16384`.
Every update retains and replays the full persisted trace before accepting the
next point. The three indexed training records in `fixtures/pilot_train.json`
have targets `1,0,1`; the repeated record supplies a one-record tail each epoch.
The separate evaluation fixture includes original objects `0,1,2`, with prefix
`[2]` unseen in training. Its interpretation explicitly assumes the task respects
the declared renaming. This pilot is not one of the 165 stress cases.

The `train` command receives no evaluation path. It writes all sixteen updates
per mode, the full coordinates, gradients/proposals, component certificates and
forward query receipts. Its all-training seal hashes every training artifact
and records both endpoints and zero evaluated heldout pools. The `score` command
checks that seal and fully replays both training trajectories before opening
the evaluation file. Raw output retains every original input and target,
predicted object, all complex scores/probabilities and ordinary CE as exact
binary64 bit patterns. A plausible seal/file inventory alone does not pass:
all indexed trajectories must replay with correct repetitions and tail sizes.

The `verify` command independently replays both complete training runs and both
raw score sets, compares every deterministic training artifact and raw output,
and recounts correct/total and mean CE. Source identity and each actual producer
binary hash are retained. Resource checks before each heavy update/replay test
aggregate RSS plus a 4 GiB allowance against 110 GB and require free space on
both disks. Shortage stops without cleanup. Output creation is exclusive;
existing or failed runs are not overwritten.

Commit and freeze the entire crate and fixture inputs before fitting. Hash the
frozen source manifest and producer binary, then use a separate independently
rebuilt binary for the repeat. The CLI is:

```sh
# SOURCE_SHA is the SHA-256 of the frozen relative-path source manifest.
# REVISION is its committed 40-character Git revision. Neither changes mid-run.
bash experiments/convex_learned_sequence/run_pilot.sh \
  "$ORIGINAL_BINARY" "$FROZEN_SOURCE" "$ORIGINAL_OUT" "$REVISION" "$SOURCE_SHA"
bash experiments/convex_learned_sequence/run_pilot.sh \
  "$INDEPENDENT_BINARY" "$FROZEN_SOURCE" "$REPEAT_OUT" "$REVISION" "$SOURCE_SHA"
prlimit --as=4294967296:4294967296 -- "$INDEPENDENT_BINARY" verify \
  "$ORIGINAL_OUT" "$REPEAT_OUT" "$VERIFIED_REPORT"
```

All forty-four model/update controls and five protocol controls pass. The
protocol controls include exact trace roundtrips for both modes, altered CE or
probe rejection, missing-seal refusal before evaluation data is opened, and a
complete hashed inventory whose fake minibatch sequence fails actual replay.
Measured pilot outcomes require the frozen original/repeat execution below;
these protocol checks alone are not accuracy evidence.

## Sealed three-object pilot results

The optimized producer is frozen at `719ba30b`, source-manifest SHA-256
`09828527dc913c26c37f8245ca946ba1c03ef07b9a0886715abffb15def91c14`.
It and a fresh independent build each completed both modes, sealed every
training artifact before reading evaluation inputs, and passed full training
and raw-output replay. All 34 deterministic training artifacts agree between
the original and repeat: configuration, three indexed training records and
sixteen complete step traces for each mode. The final raw outputs retain all
three class probabilities and both complex score components for every record.

| Required mode | Original raw accuracy | Independent repeat | Mean evaluation CE |
| --- | --- | --- | --- |
| Mini | 3/3 (100%) | 3/3 (100%) | 0.2491072676434611 |
| Forward-only Mini | 3/3 (100%) | 3/3 (100%) | 0.1530963328629868 |

Both modes return the original objects `0`, `1`, `2` for prefixes `[0]`, `[1]`,
`[2]`. Only the last prefix is unseen in training. Its transfer relies on the
declared involution and the task's equivariance; this is not arbitrary missing
label identification. The test is a tiny prospective learned-sequence control,
not any of the 165 stress cases. It adds zero stress-case coverage and does
not meet the broader 99% raw/exact requirement.

For the optimized original/repeat, training plus sealing, full replay and scoring
took 28.13/27.98 seconds, with maximum process RSS 6948/7376 KiB. Independent
verification of both complete runs took 17.28 seconds. These are local pilot
observations under concurrent workloads, not a general throughput or RAM bound.
The original frozen producer `7881eb5d` took 808.66/817.93 seconds for its
original/repeat training-and-scoring runs and 540.63 seconds for independent
verification of both. It also passes complete independent training and raw
replay, with the same accuracy and CE. Both producers' original/repeat binary
hashes match despite coming from separate fresh build directories.

After each producer's independent replay succeeded, cross-version comparison
matched all 32 complete step artifacts byte for byte, including every rational
coordinate, projection certificate, query, pairing and CE-bit receipt. Training
and evaluation inputs also match. Configuration comparison accounts explicitly
for the different `revision` and `source_sha`; raw score comparison accounts for
`training_seal_sha`. Every remaining byte agrees, including every original object,
complex score component, probability and CE bit. Both full identities and seals
are retained. This establishes trajectory equivalence on this pilot, not a
universal floating-point refinement theorem.

Evidence remains local under
`/run/media/misha/storage/transformer-runs/learned-sequence-pilot/`;
`verified-v1.json` and `verified-v2.json` record independent recounts and binary
hashes. `compare-v1-v2/` preserves the full source identities, checked hashes,
normalized copies and the list of all 32 matched steps. Frozen sources and
binaries remain in sibling `freezes/learned-sequence-pilot-v1/` and `-v2/`.
The protocol above reproduces this pilot in fresh output directories; no old
output or frozen producer is overwritten.

After the two separate native `verify` commands above have succeeded, the core
cross-version comparisons can be reproduced with Bash (`task_old` and `task_new`
name their original run directories). The `sed` operations remove only the
explicitly retained identity lines of the producer's canonical JSON; all numeric
bit receipts are compared as text, avoiding any JSON-to-float conversion.

```sh
set -euo pipefail
cmp "$task_old/training.json" "$task_new/training.json"
cmp "$task_old/evaluation.json" "$task_new/evaluation.json"
for task_mode in mini forward; do
  for task_index in {0..15}; do
    printf -v task_trace '%s-%04d.json' "$task_mode" "$task_index"
    cmp "$task_old/$task_trace" "$task_new/$task_trace"
  done
done
cmp <(sed '/^  "revision": "/d; /^  "source_sha": "/d' "$task_old/config.json") \
    <(sed '/^  "revision": "/d; /^  "source_sha": "/d' "$task_new/config.json")
cmp <(sed '/^  "training_seal_sha": "/d' "$task_old/score.json") \
    <(sed '/^  "training_seal_sha": "/d' "$task_new/score.json")
```

## Proved quadratic alternative; no new native run

CM85 specifies a separate learned complex quadratic family in
`src/Transformer/Grokking/FlatSpectrum/ConvexChannel/LearnedQuadratic`.
The input remains an immutable object sequence and the output an original
object. Each class retains positive and negative full PSD matrices. Their
matrix square roots recover learned object representations and readouts;
actual inference sums the learned position/object entries, squares complete
complex magnitudes and performs the learned readout. The factors are coupled
through the joint matrices, whose ordinary answer CE and indexed minibatch
means are convex on the full label-independent domain.

Exact physical coverage allows arbitrary finite squared-response width and
nonnegative independent row readouts, split into positive/negative branches.
Recovery has width `d = 1 + T|O|` per branch, or `2|O|d` in total. It does
not preserve a smaller fixed-rank bottleneck or one hidden representation tied
across classes. Dense storage is `4|O|d^2` real coordinates before factors,
optimizer state and solver workspace; no family coordinate is discarded.

The nonzero raw witness changes both learned stages, including representation
direction, and reverses original-object preferences while lowering their CEs.
Its imaginary weights demonstrably change the actual energy. These are feasible
proof comparisons, not training results. R3 bounds approximation error of the
actual complex output through the proved full ordered-address-pair expansion;
flatness supplies no missing labels.

The smaller family loses real expressivity: Lean proves that none of its
points strictly classifies all three-bit parity prefixes. Therefore this
single quadratic stage cannot replace the complete hinge family or be claimed
as a solution of the full stress suite. Richer learned interactions need their
own exact convexity/coverage/cost proof. Unseen-label identification, numerical
matrix-square-root recovery, certified Mini and forward-only Mini updates,
and prospective independent training/replay remain unfinished for this family.
No native producer changed and no new stress or raw-accuracy credit is added.

The formalization can be checked without Python:

```sh
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/LearnedQuadraticAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
```

## Rectangular learned pair/query family: CM86

CM86 proves an exact rectangular alternative with two actual learned complex
encoders and a learned nuclear-norm readout. The last immutable input object
is the query. Earlier objects read learned position weights and a shared table
of adjacent-pair weights; repetitions and overlaps remain. Output classes are
the original objects. Ordinary CE and indexed minibatch means are jointly
convex in the full complex matrices, with both encoders recovered together.

For vocabulary size V and context cap T (total prefix cap T+1), query width is
q=1+V and context dimension d=1+TV+eV², where e=1 exactly when T>=2. Dense
joint storage is 2Vqd real coordinates; recovered L,R add V(q²+dq) complex
entries before normalization copies, gradients, data and spectral workspace.
Recovery uses the q-by-q query Gram matrix. No all-case 110 GB admission or
native solver claim is made. Physical coverage permits arbitrary finite
two-encoder width but does not preserve a smaller fixed rank or encoders tied
across classes.

Lean proves observable geometry changes in both recovered encoders, readout
5 -> 10, and an imaginary-weight contribution to the actual real output.
The raw family represents three-object parity, but cannot strictly classify
four-object parity even with the last object fixed. This explicitly limits
the extension; it is not the full-suite architecture. Flat-spectrum bounds
constrain approximation and exact physical width for the complex coefficient
operator. They do not identify missing original-object answers.

These are proof controls, with no new native producer or accuracy credit.
The formalization is `ConvexChannel/LearnedBilinear`, checked by the direct
Lean build and `scripts/LearnedBilinearAxioms.lean`. Numerical recovery,
unseen-label identification, richer composition, and Mini/forward-only Mini
update certification remain separate obligations.

## Proved identification bridge: task 367

The user requested one explicit Lean `sorry` for the missing connection from
capacity to unseen answers. CM87 and `LearnedBilinear/Identification.lean`
now prove the requested margin-duality theorem for the actual learned raw path. Under observed
unit-margin feasibility, a query class margin is forced for every consistent
joint parameter iff a finite nonnegative combination of observed margin
coefficients certifies it. The certificate reads observed labels and the query
input, with every original class and complex coordinate retained.

The proof includes the exact full coordinate identity, soundness/completeness,
ordinary-CE and weighted-margin-error bridges, an exact rational certificate
checker, a nonconstant genuinely held-out success example and an ambiguous
refusal. The observation operator bounds independent constrained directions
and exposes common-logit and complex null directions. Correctness for a real task
still needs explicit family realizability; neither unknown target parameters
nor held-out labels are fed to the checker. The central statement is unchanged
and proved without `sorry`. No experiment or new accuracy is
claimed. Audit it with `scripts/LearnedIdentificationAxioms.lean` and the global
axiom script; the requested new placeholder is removed, with the older36 unchanged.

The success control observes 0000->false, 1000->true and 0010->true, then
certifies the held-out 1010->true with margin at least3 for every fitting model.
At the other unseen input0100, two fitting models return opposite objects;
neither object admits a positive certificate. Per-record ordinary CE at most
log(1+exp(-1)) supplies unit margins. If each margin has error at most e,
this certificate guarantees query margin at least3-3e. These are proved controls,
not measured training accuracy. Certificate discovery, task-family discovery,
numerical factor recovery and convergence still require their own evidence.

## Remaining work

CM84 (`4b67e11b`) now proves that the existing complete family can interpolate
every full-word logit table. Its actual singleton-neuron response operator has
Gram matrix `10 I`, so a flat spectrum can coexist with `|O|^T` independent
word responses. Exact affine compression of every relative prediction needs
at least `|O|^T` real dimensions. Any fixed finite-code forward preserving all
strict classifications in the family needs at least `|O|^T` stored model bits;
below that budget one feasible model's classification cannot be preserved.
The latter permits nonlinear inference and concerns total stored information,
not peak resident RAM. Neither result lower-bounds one structured task or the
165-case stress suite. Affine logits are not necessary for convex ordinary CE.

The full build, fifteen direct axiom checks, global audit and generated INDEX
pass independently for CM84, with no new unfinished proof. This identifies a
limit on preserving the entire current family, not a new trained architecture.
Removing or sampling masks alone does not prove an equivalent compact model.
Any next restriction must retain the requested learned sequence path, describe
what functions it excludes, and demonstrate the required stress-task behavior.

Extend beyond the small pilot while preserving a declared learned family. The
forward-only rule now queries the actual model and respects its whole domain;
a gated surrogate evaluated outside that domain would not inherit the CE theorem.
CM80 derives a centre by summing certified projections of every original
constraint normal. Its zero-slack rows are exactly the universally forced
equalities; their joint nullspace describes the feasible two-sided query
directions. Degenerate cones retain every coordinate and constraint. A positive
mixture of the current point with this centre admits those directions, with
explicit displacement and convex-loss bounds because mixing can increase loss.
The native centre and complete direction basis now implement these certificates;
no finite-difference gradient identity or convergence follows from domain safety.
CM81 supplies a finite basis-completeness certificate: every proposed direction
annihilates every forced row, its selected free coordinates form an identity,
and every original coordinate residual is an explicit combination of retained
equality rows. The Lean reconstruction theorem proves full kernel coverage and
unique coefficients from those checks. Native elimination generates and
independently verifies them, and full Gram recovery checks every supplied basis
pairing. CM82 derives the actual loss derivative, and both complete native
proposal rules now retain every record and short tail. Both frozen producers
have completed independent training and raw replay, and all cross-version
step artifacts and raw outputs agree. The next architecture must address
complete-family growth with an exact representation or explicit restrictions,
accounting for shared coordinates, dimensions and computation costs. A finite
choice of convenient masks does not establish equivalence to the whole family.
CM79's quantitative descent result retains its actual-loss upper-bound premise.
Each further accuracy claim still requires a separate producer freeze, complete
training seal before heldout scoring, independent repeat and full raw replay.
This small kernel does not replace the 99% raw/exact target on all 165
stress cases or solve the exponential full-family storage and search cost.
