# Nonlinear causal paths with a convex output tensor

Can deeper generic causal features repair the remaining depth and recall
failures while ordinary cross-entropy stays convex in the learned coordinates?
Lean comes first: CD5-CD7 preserve complete nonlinear paths; CD8 at `32a9b668`
allows a learned combined path/feature/class tensor under per-class absolute-sum
budgets. The blocks remain fixed. Separate learned routing and decoder factors
would introduce a product and are not covered by that joint-convexity claim.

This directory contains the path bank, a learned output tensor, an allocation
preflight and training-only optimizer checks. No held-out stress accuracy is
claimed from the implementation checks. Independent-routing and additive
baselines and the comparison runner are implemented. `protocol.json` freezes
the full comparison before its first held-out score. The earlier training-only
results below remain tied to their producer.
The interrupted earlier 35-task cohort remains unchanged.

| Component | Fixed construction | Information used |
| --- | --- | --- |
| Input views | All symbols, plus every pair of the four most frequent training symbols | Training input tokens only |
| Previous-distinct block | Move the current complex position code to the latest earlier distinct symbol in its view | Visible prefix and previous path state |
| Completed-successor block | Find the latest completed occurrence of the selected symbol and move to its next visible symbol | Visible prefix and previous path state |
| Stream transport | Identity or fixed phase-conjugated normalized Fourier-4 | Layer index only |
| Terminal features | One-hot final symbol and fixed FNV history bucket | Actual executed path history |

Positions are represented by four complex amplitudes at each prefix location,
including an explicit missing-position slot. Routing reads stream energy from
the actual preceding hidden state. A branch clips that state's norm at one,
moves the selected content, and transports its displacement. Its norm is at
most two. The full block computes the transported skip plus that bounded
displacement; the numerical implementation checks its norm on every step.
No block reads a target, task identity or arithmetic operator.

The current evaluator stores only nonzero complex position rows, retaining even
small roundoff residuals. It still executes the complex skip and branch update;
it does not replace the hidden state with a symbolic pointer. A frozen dense
evaluator from `a440f6a0` checks exact feature-ID agreement across both
transports, depths 0–4 and prefix lengths 1–128. This bounds temporary state
storage by the path's visited positions instead of allocating a dense prefix
matrix at each branch. The norm check applies to the complex position code;
the history is auxiliary metadata and its one-hot terminal features are directly
bounded by one.

The identity/Fourier pair is a coordinate-invariance control. Both the skip
and the branch displacement are transported, and position readout uses energy;
their exact predictions should agree. This is not the same experiment as
changing only a skip while holding a coordinate-sensitive branch fixed, and
does not claim a Fourier accuracy advantage. The unitary skip is the flat
spectral bound. A bounded feature passed to the CD8 head has a logit bound
from its coefficient budget; no whole-network Jacobian flatness is asserted.

There are at most seven views and `2^depth` paths per view. Depth is capped at
four, prefixes at 1024 tokens, vocabulary at 1024, and history slots at 4096.
The preflight uses the original frozen easy/hard depth, long MQAR and short
binding datasets, checks their hashes through the existing native constructor,
and estimates coefficient, gradient, projection scratch, feature-cache and
observation-index allocations for
depths 0, 1, 2 and 4. Explicit buffers above 2 GiB must be rejected before
allocation. An 8 GiB OS address-space limit must also cover data, libraries,
temporary buffers and serialization. Hash collisions remain part of the model.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexDepth
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexDepthAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_depth/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_depth/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_depth/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_depth/Cargo.toml
bash experiments/convex_stress_hull/limited.sh 8 experiments/convex_stress_depth/target/release/convex-stress-depth prepare
```

Twenty-three native checks cover the complex path invariants, early allocation
rejection, independence from labels/task names, CE finite differences, exact
L1 projection, example minibatches with repetitions and tails, empirical
forward frequencies, an exhaustive linear-oracle comparison, and checkpoint
roundtrip/corruption handling. They also check sparse/dense feature equivalence,
additive independence from intermediate states, the independent factor gradients,
budget-preserving embedding, vertex initialization and factor serialization.
These are implementation checks, not benchmark
outcomes or a learning convergence proof.

The full depth cohort contains 288 cells: four original tasks, depths 0/1/2/4,
joint/independent/additive architectures, residual/Fourier transports and
full-batch/minibatch/forward-only modes. Both the original selected split and
every original nontraining split are scored, using their unchanged sequence
metrics and 99% pass threshold. The dense initializer's temporary coexistence
with an independent decoder is included in its allocation estimate. Oversized
cells produce explicit ineligible records and are independently rechecked.

For each task/mode/architecture/transport, choose the first depth in ascending
order whose training CE is at most `1e-8`, otherwise choose the smallest training
CE with shallower ties. Held-out metrics never enter that decision. This is a
numerical small-loss criterion against the nonnegative CE lower bound, not an
assertion of exact minimization. Forward-only creates no loss gradients, even
in replay. Full/mini tensor checkpoints additionally receive an independent
full-data derivative and exhaustive L1-oracle check from saved coefficients;
the factor baseline makes no global gap claim. All original metrics are replayed
and each replay seals the exact result file hash before the cohort summary.

The serial cohort launcher requires a fresh destination. It never overwrites
or silently restarts an interrupted cell. Source, protocol, data and model
fingerprints accompany every result. Training, score construction, checkpoint
writing and separate replay resource logs are retained. The final summary
requires every one of the 288 records and replays; it makes 72 training-only
depth choices. Historical transformer seed-one measurements remain labelled
as historical and do not replace a fresh matched baseline.

```sh
export DEPTH_PRODUCER_COMMIT=$(git rev-parse HEAD)
bash experiments/convex_stress_depth/run_cohort.sh FRESH_COHORT_DIRECTORY
```

The protocol above was frozen before the first held-out score. Partial results
below remain tied to that frozen producer.

The independent baseline uses one Bernoulli probability for each block layer
and a shared learned decoder over input-view/terminal-feature coordinates.
Its full-path weight is the product of layer probabilities. CD5's
`productWeights_mem` and CD8's `liftHead_budget` at `6f71b458` embed it in the
same radius-R combined tensor domain. Optimizing the factors is nonconvex.
It can start from the exact same single-path forward proposal as the joint head.
An update projects the decoder's CE step, then computes the routing gradient
at that decoder and uses at most eight forward backtracking checks, starting
at step 0.1. Every backward pass and forward check must be charged to its run.

The additive baseline evaluates every layer's two blocks on the original state
and learns one combined bounded readout. Its depth-one features equal the
composed depth-one features. Further layers repeat those shallow feature
functions in this energy-invariant library; they do not acquire interactions
between successive states. This is a deliberate capacity control, with its
larger stored parameter count reported rather than interpreted as extra depth.

The initial allocation preflight at `ba53e536` completed in 2.26 seconds
at 244.5 MiB peak RSS under the 8 GiB process limit. Depth-4 MQAR would require
4,714,141,320 bytes with the expanded scratch/index accounting and is rejected
by the 2 GiB budget;
its depths 0, 1 and 2 fit. All four depth choices fit for the two depth tasks
and short binding. These are allocation estimates, not measured training peaks.
The preflight, resource log and component-check logs are retained outside the
worktree in `.kanban/artifacts/175/`.

`train_check.json` fixes a training-only diagnostic with radius `32 * vocab`,
depths 0/1/2/4, and the original data hashes and supervision multiplicities.
Every mode first builds the same training-label frequency proposals, with
log frequencies floored at -24, then selects a feasible proposal by original
training CE. Full-batch and minibatch refine that initializer by projected CE
gradient steps; forward-only constructs no loss gradient. These are not
comparisons of optimizers starting from zero. The initializer's complete data
and time costs are recorded for all three modes.

Full-batch has at most 16 updates and minibatch at most 256, using the original
example batch size, seed-1 shuffle and final short batch. At least one gradient
update precedes the `1e-8` training-CE stopping check. The learning rate is the
reciprocal of the active-feature count. Full gradients produce a numerical
linear-oracle gap; forward-only leaves that field null. This diagnostic does
not claim a proved finite-iteration convergence rate.

The serial launcher enforces an 8 GiB address-space cap on both training and
replay. Checkpoints stream coefficients as binary floats without a large JSON
copy. Each result records the compiled source fingerprint, producer commit,
data fingerprints, model hash, costs and gradient/example/prefix counts.
Replay reloads the saved model, reconstructs training-only feature discovery,
recomputes original training CE and checks the class budgets.

```sh
export DEPTH_PRODUCER_COMMIT=$(git rev-parse HEAD)
# The parent of FRESH_DIRECTORY must exist; existing runs are never overwritten.
bash experiments/convex_stress_depth/run_train_checks.sh FRESH_DIRECTORY gpt-probe-key-value-binding 1
```

No validation or test metric is computed by this command. The larger depth
comparison is running; a fresh matched transformer baseline remains outstanding.

The first check is complete at producer `a440f6a0`, source fingerprint
`21355e87dcc3e1b77e4524e075d4591c9a967a2b3afb478c2348add827147cbf`.
It used all 2,048 original short-binding training examples at depth one.
All three modes selected the same forward initializer with training CE
`3.7373833906779454e-9`; refinement left that reported CE unchanged.

| Mode | Gradient updates / prefixes | Peak training RSS | Total training wall time |
| --- | ---: | ---: | ---: |
| Full batch | 1 / 2,048 | 101,512 KiB | 0.68 s |
| Minibatch | 1 / 32 | 101,552 KiB | 0.66 s |
| Forward only | 0 / 0 | 101,544 KiB | 0.62 s |

Each row additionally pays for the shared initializer's 2,048 examples and
prefixes. Times include data, feature construction, initialization, refinement,
the final diagnostic and checkpoint writing; separate replay took 0.46–0.49 s.
All three binary models replayed with CE difference `3.46e-16` or less and
passed the class-budget check. Their largest class absolute sum is about 24,
below the radius 3,200. The full-gradient linear-oracle gaps remain about
`2.00e-5`; no tighter gradient-gap result is claimed from the small CE itself.
No held-out accuracy was measured, and the one-step gradient refinements did
not discover the good initializer.

The real depth-4 MQAR allocation guard also rejected its 4,714,141,320-byte
buffer request before constructing the cache or head, using 244,012 KiB peak
RSS for data verification. A deliberately corrupted saved CE was rejected on
replay. Models and resource logs live in
`runs/training-check-v1-binding-d1/`, linked to the root checkout; the frozen
producer, checks and negative controls are retained in board artifact #175.

## Frozen cohort: completed short-binding subset

Producer `032762e59acc5876c55f197e127afb4ee411ebc9`, source fingerprint
`d6931b95c2b1d4971c02b718772fd08105c554d2d0bb66b99aedbf1b07b3e56f`,
has completed and replayed all 72 short-binding cells. Each table entry ranges
over full-batch, minibatch and forward-only with both transports. The other
three tasks and the complete 72-choice summary are still pending; these are
candidate accuracies, not selected cohort outcomes.

| Depth | Joint head: original primary accuracy | Independent factors | Additive control |
| --- | ---: | ---: | ---: |
| 0 | 3.52–8.20% | 3.52–8.20% | 3.52–8.20% |
| 1 | 100% | 100% | 100% |
| 2 | 98.63% | 98.63% | 100% |
| 4 | 46.29–50.00% | 46.29–48.05% | 100% |

Every 100% entry also passes every original nontraining split. The joint and
independent depth-two candidates still have training CE `3.7374e-9`, so their
98.63% held-out accuracy demonstrates that a small training loss does not
identify missing answers. Depth-four joint CE is 0.1390–0.1449 and independent
CE is 0.1447–0.1451. This final-depth-only library has no guaranteed identity
path; more steps can discard useful shallower features. The result motivates
retaining them, but does not prove that no head in the depth-four class could
solve the task. The unchanged training-only depth choice remains prospective
until the full cohort and its replays close.

## Frozen cohort: completed easy-depth subset

All 72 `basis-depth-easy` cases from the same frozen producer `032762e5`
have completed and have matching saved-result replay digests. Each entry below
ranges over the three training modes and both transports. These are candidate
outcomes; the full 288-cell inventory and 72 training-only choices remain open.

| Depth | Joint head: original primary accuracy | Independent factors | Additive control |
| --- | ---: | ---: | ---: |
| 0 | 18.55–44.34% | 18.55–44.34% | 18.55–44.34% |
| 1 | 18.55–53.71% | 18.55–44.34% | 18.55–53.71% |
| 2 | 100% | 100% | 18.55–53.71% |
| 4 | 100% | 100% | 18.55–53.71% |

All 24 composed depth-two/four candidates also achieve 100% on every original
nontraining split, including length 128. Their training CE is
`1.3212972929315844e-9`. No depth-zero/one or additive candidate reaches the
99% pass threshold. The shared training-only initializer already supplies these
successful endpoints; the result does not establish that gradient refinement
discovered the representation. The independent factor control succeeds too,
so this subset establishes a benefit from composed depth in this library,
without establishing an advantage from convex joint optimization or Fourier
transport. It does not resolve the hard-depth, MQAR or full-35 result.

## Frozen cohort: first hard-depth successes

The full-batch, minibatch and forward-only joint/residual depth-four endpoints
for `basis-depth-hard` have completed, and all three saved-result digests match
their successful independent replay records. Each reaches **100% on every
original split**, including the primary `validation/length-128/novel` split.
Their training CE is `1.3212972929315849e-9`. These are three completed candidate
endpoints from producer `032762e5`, not a completed hard-task or whole-cohort
aggregate. The other depth-four controls and the final training-only choices
remain in the unchanged running inventory.

All 36 depth-one/two hard-task candidates have also completed. Across all
three modes and both transports, none reaches the original 99% threshold:

| Depth | Joint head: original primary accuracy | Independent factors | Additive control |
| --- | ---: | ---: | ---: |
| 1 | 20.31% | 20.31% | 20.31% |
| 2 | 8.59–42.58% | 24.22–42.58% | 20.31% |

This supplies an initial depth-four success where the shallower candidates
failed. The common training-label initializer already produces the successful
model, so no gradient-discovery advantage follows. The final comparison must
still include the independent/additive and Fourier controls, MQAR, retained
features, the other original tasks and the fresh conventional transformer.
Generated evidence is preserved in `.kanban/artifacts/175/hard-early-v1/` and
the original `runs/cohort-v1/` paths; no training budget or stopping rule changed.

## Retained paths and bounded-memory training components

Lean CD9 at `9136a122` proves that retaining every shallower feature embeds its
head at exactly the same coefficient budget, logits and ordinary CE. The
prospective cumulative bank implements this order: increasing depth, then
input view, then complete path. It keeps the same 4096 history slots, including
their collisions. Its path count per view is `2^(depth+1)-1`; the coefficient
count still grows exponentially. This is a capacity extension, not a guarantee
about a numerical optimizer or an unseen answer.

`stream_cache` writes each original supervised prefix and label once, then
reads fixed-width records with a 64 KiB buffer. It preserves repetitions,
ignored-target positions, whole-example minibatches, shuffles and short tails.
Memory checks include the head, a gradient/count scratch tensor, projection
scratch, I/O and example indexes before allocating a cache. The explicit
2 GiB memory cap and 8 GiB process cap remain. A separate 16 GiB cache-file
cap bounds disk use per case; caches belong to their run directory, and their
construction, I/O and hashes must be charged and reported by a future runner.
Existing cache files are never overwritten or silently resumed.

The streamed objective uses the same ordinary CE and projected head update.
Forward-only evaluation forms no loss derivatives. Its empirical initializer
supports the existing minimum-CE rule and a prospective fixed first-channel
rule: choose the earliest channel reaching a declared training-CE threshold,
otherwise minimum CE. For cumulative banks the order prioritizes shallower
features. A follow-up must freeze that rule before its own training; neither
rule uses held-out scores. The running 288-cell comparison continues to use its
original final-depth bank, initializer and frozen executable.

Twenty-eight native checks now pass. The five additions verify exact retained
features and nonzero logits, bit-identical dense/streamed CE, gradients and
updates across repeated/shuffled batches and tails, identical empirical
initialization under the old rule, threshold ordering, and rejection of
corrupt records, boundaries, lengths and excessive allocations. These are
component tests; no retained-path stress outcome is claimed.

```sh
cargo build --offline --release --manifest-path experiments/convex_stress_depth/Cargo.toml
bash experiments/convex_stress_hull/limited.sh 8 \
  experiments/convex_stress_depth/target/release/convex-stress-depth prepare-retention
```

This command computes only allocation estimates on the four original datasets;
it constructs no training cache or head and scores no held-out metrics.

The preflight at producer `3dac93c3` completed in 2.28 seconds with 252,996 KiB
peak RSS. At cumulative depth four, each depth task needs 524,510,920 bytes of
explicit buffers with streaming, versus 2,189,341,824 / 2,191,138,112 bytes with
the in-memory cache. Its cache file is bounded by about 1.64 GB. Short binding
at depth four needs 1,464,658,120 bytes. MQAR at cumulative depth two needs
1,998,235,656 bytes and fits the 2 GiB explicit limit; depth four still needs
8,846,677,704 bytes and is rejected because of the dense output head itself.
These forecasts do not establish peak training memory or a learning result.
The frozen preflight binary, exact source fingerprint, JSON and resource log
are retained locally in `.kanban/artifacts/175/retention-components/`.

## Retained-path follow-up protocol

`retained_protocol.json` fixes 48 joint-head cases: the same four original tasks,
both transports, all three training modes and both initialization policies.
Each case contains all shorter paths, up to depth four for binding and the two
depth tasks, or depth two for MQAR. These maximum depths come from the preceding
allocation preflight, before any retained-head stress training. Both initializer
policies are reported; the experiment does not choose between them using a
held-out score. The frozen final-only joint, independent and additive candidates
remain the earlier comparison, and this follow-up is explicitly adaptive to it.

The first-channel policy uses `log(1+(vocab-1)*exp(-24)) + 1e-8`. Lean
`fixed_margin_ce` at `79f779ba` proves the reference CE of a pure margin-24
proposal. For MQAR's 548 classes, that value is already about `2.065e-8`;
an unadjusted `1e-8` rule would reject even pure proposals. This correction
calibrates initializer selection only. Full/mini keep the original stopping
threshold `1e-8`, update caps, example batches and learning rate, so a pure
MQAR proposal can still exhaust its refinement cap. The reference value is
not an optimum lower bound for the larger head family.

The same training loop now handles dense and streamed observations. Independent
factors remain restricted to the original composed bank. Replay verifies the
cache hash, byte count, each feature and label against its original prefix, and
each example boundary before replaying all original metrics. Full/mini retain
the independent raw-coefficient CE/linear-oracle audit; forward-only computes
no derivative. All 48 models, caches, costs and failures are retained. A complete
summary requires all 48 sealed replays and reports both policies separately.

The launcher first calls the frozen original executable's complete-summary
check. It refuses to start before all 288 original outcomes and replays close,
then runs serially with the 8 GiB address-space cap. Forecasts conservatively
charge one example index per supervised prefix. Disk estimates include every
cache and model, must remain within 96 GiB, and preserve at least 16 GiB free
space. Each cell checks that reserve again. Existing destinations are rejected.

```sh
export DEPTH_PRODUCER_COMMIT=$(git rev-parse HEAD)
# The reference binary must be built from the frozen 032762e5 producer.
export DEPTH_REFERENCE_BINARY=/path/to/frozen/convex-stress-depth-cohort-v1
bash experiments/convex_stress_depth/run_retained.sh FRESH_RETAINED_COHORT_DIRECTORY
```

Thirty-one native checks pass before this follow-up is frozen, including exact
dense/streamed updates through the shared training loop, vocabulary-dependent
CE calibration and complete inventories for both protocols. A corruption test
also moves an example boundary while keeping all flattened rows unchanged;
the new replay rejects it, protecting minibatch semantics that a full-data CE
check would miss. There are no retained-path stress results yet.

The frozen retained producer is `9da1c677`, with source fingerprint
`5f95365da8434756d09b79ad485aba32e4cd2addcfec7a76f9571a8e5ab92e33`.
Its inventory preflight admits all 48 cases, forecasts at most 2,002,781,192
explicit buffer bytes per case and 67,138,185,792 total disk bytes. The preflight
took 2.39 seconds at 253,256 KiB peak RSS; those are preprocessing measurements,
not training peaks. The launch-gate negative check exited before creating a
training directory when the original cohort still had an unfinished replay.
The frozen follow-up is queued behind that specific live cohort process and
will require its successful exit plus the complete-summary check. The binaries,
launcher, preflight and check evidence stay outside the worktree in
`.kanban/artifacts/175/retained-cohort-v1/`.
