# Revisit complete readouts on learned recurrent states

Can refitting every learned component's class readouts across the training
stream improve original associative recall? The frozen cautious and refined
rotary models fit only 8.5–9.6% of training answers. Their complete states remain
nonconstant with large norms, while each old readout was fitted only on its
component's proposal batch. This experiment tests that optimizer limitation.

CM100 at `16276e1c` proves 22 statements before this implementation: complete
complex row-domain convexity, actual state preservation, linear complete logits,
ordinary full-class and indexed minibatch CE convexity, concrete radial row
feasibility, accepted interpolation and full-family signed CE-query feasibility.
The source is `review/convex/moments.md`, CM100 and Lean
`LearnedRecurrent/Rotary/Readout`. It supplies no fit or generalization guarantee.

| Variant | Starting model | Added training |
|---|---|---|
| Cautious / Mini and forward | Matching mode's immutable cautious seed1 endpoint | 512 original minibatches, 8 epochs |
| Refined / Mini and forward | Matching mode's immutable frequency-refined seed1 endpoint | Identical new batch stream and update settings |

All 129 components, 100 original objects/classes, full content matrices, initial
states and learned frequencies remain present. They were learned in the prior
stage; this controlled conditional step revisits all class readout coefficients.
Every raw object occurrence contributes through actual recurrent execution.
Temporary batch-state values are recomputed from the retained learned content;
there is no supplied feature map, answer program, magnitude pruning or rank cut.
The external contract is still an immutable-object sequence to one next object.

For each original minibatch, reserve the same finite-query centre in both modes.
Mini uses complete ordinary-CE residuals. Forward uses two actual CE evaluations
per original class and record, without a CE derivative or recurrent backward
pass. Contract those scalars with every conjugated complex state and its actual
mixture weight, then average over every original target record. Move every row
by the frozen learning rate 4, apply CM100's complete radial budget rule, and
search the full feasible interpolation interval [0,1] with 64 CE iterations.
Every accepted readout and every original-class logit is checked against actual
model execution. The radial rule is not claimed to be Euclidean projection.

All four endpoints seal before new held-out scoring. Report the original success
criterion and separate raw/exact 99% flags on every original split, including
failures. Independently replay every raw logit. All training CE evaluations are
counted: class-bias queries, line search, centre, scalar validation and actual
accepted-model check. Preserve the original failed producers and transformer
baseline. A gain in training fit alone does not resolve generalization.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/recurrent-readouts
export CARGO_BUILD_JOBS=2
cargo test --offline --locked --manifest-path experiments/convex_recurrent_readouts/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_recurrent_readouts/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_recurrent_readouts/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/convex_recurrent_readouts/Cargo.toml --bin stress
"$CARGO_TARGET_DIR/release/stress" plan
```

Freeze the executable, protocol, source archive, commit and hashes before:

```sh
bash experiments/convex_recurrent_readouts/run.sh WORKSPACE STORAGE SOURCE_RUNS FREEZE FRESH_COHORT
```

`SOURCE_RUNS` is the parent containing the immutable `cautious-v1` and
`refined-v1` rotary cohorts. All generated output belongs outside the worktree.
Two workers each have a 32 GiB address-space cap, with aggregate RAM admission
below 110 GB and continuous native guards. Stop below 2 GiB workspace space or
16 GiB storage space. Every source checkpoint, original dataset identity and
complete learned-content hash is verified.

All four native controls pass, covering both training modes, imaginary order
information, complete class/coordinate updates, unchanged learned content,
actual feasible signed queries and refusal to score an incomplete or modified
four-endpoint cohort. Clippy, formatting and shell checks pass as well.

The first cohort completes at producer `6cc8d452`, freeze
`recurrent-readouts-v1`, source fingerprint
`0b2a966f9f9caea8716d2bf475be283b1d5f3789299a5cbd8f220278b273f7ac`, binary SHA256
`f2b623685e0bed60cb5c96d41d7a6834d4be7e5fe336db0d78cbb2cf757d2174`.
All twelve train/score/replay commands exit0 and all four complete raw-logit and
metric replays match. Evidence is in
`/run/media/misha/storage/transformer-runs/recurrent-readouts/original-v1`.

| Source / mode | Train correct /2048 | Novel validation correct /512 | Test correct /512 | Train CE | Novel CE |
|---|---:|---:|---:|---:|---:|
| Cautious / Mini |1455 (71.045%)|70 (13.672%)|68 (13.281%)|2.06941|3.17215|
| Cautious / forward |1459 (71.240%)|54 (10.547%)|65 (12.695%)|2.05897|3.19051|
| Refined / Mini |1437 (70.166%)|63 (12.305%)|48 (9.375%)|2.09426|3.22125|
| Refined / forward |1393 (68.018%)|66 (12.891%)|75 (14.648%)|2.10767|3.20411|

Both original success and raw/exact99 flags remain false for every cell.
Raw and exact accuracies coincide because each example has one answer. Training
fit improves substantially from the retained sources' 8.5–9.6%, and novel
accuracy improves from their 2.0–4.7%. This demonstrates better fitting with the
same learned state functions; it does not show that those functions identify
the right unseen answers. There is still a large train/held-out gap. Only one
training seed and the original data seed are measured here.

Every complete content hash is unchanged. Each run retains129 components and
processes16384 original records over8 additional epochs; all four batch traces
match. All512 accepted interpolation coefficients are1, and all actual-logit
interpolation discrepancies are0. Maximum operator-budget excess is2.22e-16.
Each Mini run uses1163264 CE evaluations; each forward run4440064, including
3276800 signed class-bias queries. All runs include1114112 line-search and49152
centre/validation/accepted-model calls. Concurrent training times are16.62/18.85s
for cautious and16.27/19.10s for refined, Mini/forward respectively. These are
additional refit costs, not the cost of learning the source content or matched
transformer budgets. The next question is whether further fitting saturates or
whether revisiting shared content is needed;99% remains unresolved.

## Longer fitting with an exact earlier trajectory check

Task383 changes only the prospective update budget, from512 to2048 (32 refit
epochs). `long_protocol.json` keeps the same four starting checkpoints, learning
rate4, query step, line search, row rule, original batch stream and all content
coordinates. Before passing update512, every deterministic receipt, example and
target count, batch hash and complete checkpoint byte must match the immutable
first refit producer. The check reads training traces and model bytes only.
A changed earlier trajectory refuses the run; all four exact-prefix records are
also required by the final scoring seal. No old score selects a new update.

Freeze a new executable/source archive, commit and hashes, copying
`long_protocol.json` as the freeze's `protocol.json`. The original refit cohort
must remain at its declared path below STORAGE. Then run:

```sh
"$CARGO_TARGET_DIR/release/stress" plan-long
bash experiments/convex_recurrent_readouts/run.sh WORKSPACE STORAGE SOURCE_RUNS FREEZE FRESH_COHORT -long
```

Individual train/score/verify commands gain `-long`. Every old result remains
intact. The first cohort's training-stream CE was still falling at epoch8; this
control tests further fitting without claiming that more training identifies
unseen answers. All original metrics, failures, query costs, resource guards
and complete raw replays remain required. No new Lean claim is needed: the
CM100 feasible update and conditional-CE results are unchanged.

The longer cohort completes at producer `b78d4661`, freeze
`recurrent-readouts-long-v1`, source fingerprint
`fba92f6a6c978d8cf62885f971285c2731c63c9d1f743aea0b62f775de8604ed`, binary SHA256
`ca9940ffda45f3f121f20c83e653327d9b9812c079aad348d8247fb8763c3d47`.
All four first512 trajectories and checkpoints match the earlier producer
exactly, all twelve commands exit0, and all four complete raw replays match.
Evidence is in
`/run/media/misha/storage/transformer-runs/recurrent-readouts/long-v1`.

| Source / mode | Train correct /2048 | Novel validation correct /512 | Test correct /512 | Train CE | Novel CE |
|---|---:|---:|---:|---:|---:|
| Cautious / Mini |1598 (78.027%)|79 (15.430%)|68 (13.281%)|1.92094|3.12219|
| Cautious / forward |1629 (79.541%)|52 (10.156%)|74 (14.453%)|1.91165|3.14176|
| Refined / Mini |1595 (77.881%)|69 (13.477%)|65 (12.695%)|1.94823|3.15926|
| Refined / forward |1569 (76.611%)|75 (14.648%)|81 (15.820%)|1.96152|3.14849|

Every original success/raw99 flag remains false. Train fit and CE improve, but
the generalization gap persists and novel accuracy does not improve in every
cell. Each run processes65536 records over32 refit epochs. All learned content
hashes and129 stored components remain unchanged; all four batch streams match.
Every interpolation coefficient is1, actual interpolation errors are0, and
maximum row-budget excess is4.44e-16. Each Mini run uses4653056 CE
evaluations, each forward run17760256, including13107200 signed bias queries.
Concurrent training times are63.93/74.70s for cautious and63.96/73.23s for
refined, Mini/forward respectively; source-content training costs are separate.

Mean pre-update training-stream CE slows substantially. For cautious Mini it is
2.2230/2.0833/2.0547/2.0466 at epochs8/16/24/32; cautious forward gives
2.2127/2.0735/2.0453/2.0376. Refined Mini gives
2.2496/2.1103/2.0846/2.0748, refined forward2.2643/2.1258/2.0978/2.0897.
These changing-minibatch measurements support diminishing gains under this
update rule; they do not certify a convex optimum, insufficient state capacity
or impossibility of recall. The next investigation is whether radial row
normalization can itself stall away from an optimum, before attributing the
remaining failure to architecture or unseen-answer identification alone.

## Exact Euclidean row projection

CM101 at `1cf1d6e8` proves that the earlier radial rule can leave a strictly
worse ordinary-CE point unchanged on two oppositely labeled original prefixes.
CM102 at `9d1ce0fb` proves an explicit complete complex Euclidean projection,
its closest-point property on all four real coordinates, original model
capacity, and a positive CE step escaping that counterexample. All25 new
theorems pass full/source/axiom/INDEX checks and review before this Rust port.

Task386 changes only the row projection in the original512-update refit
protocol. `projected_protocol.json` keeps all four learned starting models,
original minibatches, learning rate4, query step1e-4,64 CE line iterations,
capacity16 and every original coordinate/class. The piecewise magnitude rule
preserves each complex phase; inside rows return unchanged, and zero magnitudes
stay stored as zero values. The old radial update and both earlier protocols
remain available. An old zero value can still receive later gradients.

The new native controls cover all projection branches, complex phases, an
independent variational check and3840 feasible distance competitors, and actual
CM101 CE repair using the existing rate4 and complete update in both modes.
The implementation uses f64; CM102's exact real-arithmetic theorem is not a
roundoff or stochastic-convergence theorem. All existing capacity and actual
logit checks remain in every update.

Build with the same native commands above, then inspect `stress plan-projected`.
Freeze the source commit/archive, executable and hashes, copying
`projected_protocol.json` to `protocol.json` and retaining the old protocol and
plan as `radial_protocol.json` and `plan-radial.json`. Before interpreting new
outcomes, reproduce all four legacy512-update trajectories with the new binary:

```sh
bash experiments/convex_recurrent_readouts/run.sh WORKSPACE STORAGE SOURCE_RUNS FREEZE FRESH_LEGACY_CONTROL
bash experiments/convex_recurrent_readouts/run.sh WORKSPACE STORAGE SOURCE_RUNS FREEZE FRESH_PROJECTED_COHORT -projected
```

Compare every legacy deterministic receipt, batch/count and checkpoint byte
with `recurrent-readouts/original-v1`; compare every original raw split report
and success flag as well. Producer identities differ because the new binary
also contains the projection option. Preserve all old artifacts. Each cohort
seals all four endpoints before scoring, retains all failures and replays all
raw logits. Both disk guards and110GB aggregate RAM limits remain active.
Improved fitting or the proof witness alone earns no stress-test success.

The projected cohort completes at producer `65eb373d`, freeze
`recurrent-readouts-projected-v1`, source fingerprint
`2b40be589c9def1f47fcab9180e28afc2fcd0c88e832fb12e46835a382313617`, binary SHA256
`54662795a59c03e4b5bd38c2d5acdcd49e590d4a25ba2520a32d453292dcca1e`.
All eight native controls and clippy/fmt/shell checks pass. All four legacy
controls match every original512 deterministic receipt, batch/count, checkpoint
byte and original raw split report; only producer metadata differs. All24
train/score/replay commands exit0 across the legacy and new cohorts, and all
eight raw replays match. Evidence is under
`/run/media/misha/storage/transformer-runs/recurrent-readouts/` in
`projected-legacy-control-v1` and `projected-v1`.

| Source / mode | Train correct /2048 | Novel validation correct /512 | Test correct /512 | Train CE | Novel CE |
|---|---:|---:|---:|---:|---:|
| Cautious / Mini |1390 (67.871%)|76 (14.844%)|65 (12.695%)|2.04200|3.15798|
| Cautious / forward |1408 (68.750%)|62 (12.109%)|67 (13.086%)|2.03223|3.16457|
| Refined / Mini |1368 (66.797%)|59 (11.523%)|48 (9.375%)|2.07108|3.21449|
| Refined / forward |1324 (64.648%)|63 (12.305%)|69 (13.477%)|2.08552|3.18236|

Every source-success/raw99 flag remains false. At the matched eight-refit-epoch
budget, training CE is lower in all four cells, while training raw accuracy is
2.49–3.37 percentage points lower. Novel accuracy improves for both cautious
sources and declines for both refined sources. A correct projection and lower
CE therefore do not establish better raw accuracy or solve unseen recall.
The longer radial cohort above used32 epochs and is not the matched comparison.

All original content hashes,129 components,100 classes and batch streams remain
unchanged. Each new run processes16384 records; every interpolation coefficient
is1, actual logit interpolation error is0, and maximum row-budget excess is
2.22e-16. Mini uses1163264 CE evaluations and forward4440064, including3276800
signed bias queries; line and centre/validation/check costs match the legacy
protocol. Concurrent refit times are15.82/18.56s for cautious and15.63/19.00s for
refined, Mini/forward respectively, excluding source-content training.

Training-stream mean CE is still falling at epoch8. Its final values are
2.19114/2.18174 for cautious and2.22372/2.23832 for refined, Mini/forward.
Neither this trajectory nor the failed raw scores certify an optimum of the
conditional readout problem. A training-only optimality-gap diagnostic is the
next way to distinguish remaining optimization error from limits of this
learned content and capacity before revisiting their shared structure.
