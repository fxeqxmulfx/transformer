# Shared learned complex factors and generalization

Can a capacity constraint with a learned output factor shared across the input
encoders improve unseen-answer accuracy? The previous CM89 addition diagnostic
fits 100% of training answers and gets 0% of held-out answers correct. This is a
new experiment with unchanged original data and the transformer baseline
preserved. The first geometry fails at 133/4704 training and 1/4705 held-out
answers. After dimension-based bias scaling, both modes reach 4704/4704 training
and 4705/4705 held-out answers. This establishes empirical generalization on one
original task and one seed; broader generalization remains unresolved.

The interface is a sequence of immutable objects to one next original object.
IDs are addresses, with no arithmetic or semantic interpretation. Every
component learns complex vectors for all output objects, query addresses and
context addresses. Inference executes those vectors and their complex
interaction, then takes real scores and chooses an original object. All classes
remain present; no label pooling or task-specific answer program enters the
model. The reader retains original position and adjacent-pair multiplicities
from CM86, so its higher-context parity limitation remains.

CM91 (`3cbfec55`, `LearnedBilinear/Capacity`) proves that a declared complete
capacity budget can identify an unseen object via a nonzero residual
certificate. CM92 (`d35d0f11`, `LearnedBilinear/SharedAtoms`) proves the shared
component domain, finite executable representations and feasible forward-only
queries. The complete joint tensor domain is convex. Convexity is **not**
claimed in independently varied factor arrays. Global component discovery and
optimization convergence are not proved.

| Mode | Training signal at the common centre | Complete minibatch |
| --- | --- | --- |
| Mini | Ordinary CE analytic class residuals | Every supervised prefix, original class, duplicate and short tail |
| Forward-only Mini | Two ordinary CE evaluations per class and record | Same records and all real/imaginary coordinate responses |

In the original geometry, each step shrinks the state by `1-h/B`, putting the freed mass on an exact zero
component. Signed class-bias queries of size `h` are actual feasible states,
as proved in CM92. Only the loss receives targets. CM88 supplies all coordinate
responses, including repeated pairs. Real address rows make the complete
complex gradient pair `(g,-g)`, so one dense real array stores both components
exactly under this identity. No coordinate is sampled or discarded.

The native optimizer uses momentum on this complete array, followed by
approximate complex component search. Two random full-vector starts and the
previous candidate undergo eight alternating contractions, normalizing every
coordinate of all three factors. This approximates a linear minimization oracle;
it is not an exact global convex solver. A finite ordinary-CE line search mixes
the proposal into the model. Every attempted step is traced and every proposed
component retained, including zero mass. There is no rank threshold or small
weight deletion. Numerical normalization and mixture-mass residuals are checked;
the real Lean proofs do not certify floating point rounding.

`protocol.json` prospectively fixes original addition, seed 1, both modes and
1000 updates. Its budget `8*sqrt(V*Q*D)` uses only original dimensions. The
sampler retains original example batches and tails, with separate sampler and
component-search RNGs. No held-out answer selects a direction, budget, endpoint
or stopping time. All endpoints must be sealed before any scoring. This uses
more updates than the old 200-update control; it is not a matched-time or
matched-budget comparison and supplies no 35/165 or three-seed coverage claim.

Every original split is scored, retaining source success and separate raw/exact
99% fields. Scoring hashes every full raw-logit vector. A separate process
reloads the endpoint and must reproduce all metrics and raw-logit bits. This
checks saved inference, not an independent training repeat. Data, source,
protocol, binary and checkpoint identities are recorded.

Thirteen native controls cover exhaustive small-prefix complex execution against
independently materialized coefficients; feasible signed model queries; all
real/imaginary coordinate gradients with duplicate records, repeated pairs and
a one-record tail; a planted nonzero component-search control; and both update
modes with exact checkpoint replay; original masks and full-logit digests; and
refusal of unsealed or changed protocols. These are not original-task scores.
The four CM93 controls also check the scaled full tensor, actual feasible query
models, complete scaled coordinate responses, and both modes with checkpoint
geometry retained. They include zero-scale and floating underflow refusals.
The CM94 control compares every complex contraction and objective bit with
dense execution, retains tiny nonzero rows, refreshes changed support and checks
the all-zero case.
The line-search control uses a mixed-label constant-input batch with a known CE
optimum to detect insufficient resolution at large candidate scales.

Use native tooling only:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-atoms-balanced-native-v1
cargo test --offline --locked --manifest-path experiments/convex_shared_atoms/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_shared_atoms/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_shared_atoms/Cargo.toml
"$CARGO_TARGET_DIR/release/stress" plan
```

Freeze the checked executable and sources outside a temporary worktree. Run
`stress train WORKSPACE STORAGE COHORT arithmetic-add MODE 1` for `mini` and
`forward`; then run `score` and `verify` with the same arguments. Cell directories
and evidence files use create-new semantics. No overwrite or resume is provided.

Before training allocations/updates and each scoring split, check aggregate RAM
plus the full reservation against 110,000,000,000 bytes. Keep at least 2 GiB free
on the workspace disk and 16 GiB on storage. Insufficient space or memory stops
the operation without cleanup, reduction or changes to frozen producers.

## First frozen diagnostic, 2026-11-17

Producer commit `3bb8159a`, freeze `shared-atoms-v1`, source fingerprint
`878ccde937e52893fb97eed0f298fc23127cbf93e072fb6ba87d72984f1f428f` and binary
SHA256 `06bcc4a4cf5fc869f2ff12c2e53adbb33412eb7c43c8dfb9d46bb107efd61f17`.
Generated evidence is retained at
`/run/media/misha/storage/transformer-runs/shared-atoms/original-v1`.
All training and scoring queues exit zero. Both checkpoints reproduce complete
metrics and raw-logit bits in a separate process.

| Mode | Training raw/exact | Held-out raw/exact | Training CE | Held-out CE |
| --- | --- | --- | --- | --- |
| Mini | 133/4704 = 2.82738% | 1/4705 = 0.021254% | 4.59864689 | 5.71236592 |
| Forward-only Mini | 133/4704 = 2.82738% | 1/4705 = 0.021254% | 4.59864710 | 5.71236623 |

Each mode uses 1000 updates and 247680 indexed training records, or 52.6531
epochs. The original success flag and raw/exact99 are false. These results do
not improve the original unscaled held-out count, and the loss is worse. Unlike
the earlier frequency-scaled model, this optimizer fails to fit training too.
The result does not establish that every optimizer in this domain must fail.

The Mini endpoint retains 1000 proposed components plus the zero component.
Among the active weights, mean squared query-bias energy is 0.9780685 and mean
squared context-bias energy is 0.9783835 (weighted by component mass, excluding
the zero component). Thus almost all factor energy goes to the always-present
bias addresses. This is a measured optimization-geometry diagnostic, not a
generalization theorem. A positive change of coordinate scale is the next
candidate: preserve every learned coordinate and original object while reducing
the bias addresses' dominance. The next cohort tests this after proving its
geometry and feasible queries. Keep this failed producer intact.

## Dimension-scaled bias geometry

CM93 (`c7118dc1`, `SharedAtoms/Scaling`) proves an invertible positive coordinate
change, convexity of the changed joint capacity region, full executed factor
representations, the complete scaled responses and feasible forward queries.
The new `balanced_protocol.json` differs in geometry: each input bias entry has
scale `1/sqrt(V)`, and every object, position and pair entry has scale one. All
stored complex factors remain learned. This changes the fixed-budget domain;
it is not an assertion that both geometries admit exactly the same parameters.

The actual raw class-bias response is `u=query_scale[0]*context_scale[0]`.
Both modes reserve mass `h/u` and shrink by `1-h/(u*B)`, so actual signed model
queries still change one raw logit by `+/-h`. All complete gradient responses
receive `query_scale[q]*context_scale[d]`. Scoring checks the saved scales against
the frozen protocol. The same 1000 updates, seed, records, factor-search settings
and dimension-only budget are retained. The nonconvex approximate component
search and higher-context parity limitations remain. Scores below were measured
only after both endpoints were sealed.

Use `plan-balanced`, `train-balanced`, `score-balanced` and `verify-balanced`
with the same arguments as above, a fresh frozen executable and a distinct
cohort directory. The original frozen executable is required to replay the first
cohort: its checkpoint schema and source fingerprint remain unchanged there.

## Scaled diagnostic, 2026-11-17

Producer commit `66b08989`, freeze `shared-atoms-balanced-v1`, source fingerprint
`86f9e4d7b960000a34f432ff2b9279ea8d9366c3d2dbd9caf801df649d713a7e` and binary
SHA256 `104bff23210c01ce1fa07d54a86a03c7545127d498120f0e0ec7b199fd9476e2`.
Generated evidence is retained at
`/run/media/misha/storage/transformer-runs/shared-atoms/balanced-v1`.
All training, original-split scoring and separate-process replay commands exit
zero. Every saved raw logit and metric replays exactly in each mode.

| Mode | Training raw/exact | Held-out raw/exact | Training CE | Held-out CE |
| --- | --- | --- | --- | --- |
| Mini | 4704/4704 = 100% | 4705/4705 = 100% | 0.001042251084268 | 0.001125259891768 |
| Forward-only Mini | 4704/4704 = 100% | 4705/4705 = 100% | 0.001042251084259 | 0.001125259891742 |

Both original success and raw/exact99 are true. Each mode retains 1000 updates,
247680 indexed records, 52.6531 epochs and 1001 components, including the zero
component. Mini training takes112.19s and forward-only150.73s on this machine
with the existing unrelated producer running; these are not transformer timing
comparisons. Forward-only uses48049920 finite class-bias loss evaluations.

The Mini endpoint's active mass is7719.7140. Its component-mass-weighted squared
query/context bias energies are0.0000485007/0.0000536857. The matched geometry
change therefore removes the measured bias dominance and enables both fitting
and held-out accuracy in this diagnostic. The learned factors use object IDs
only as addresses; no arithmetic answer rule is supplied. Do not infer from
this one task/seed that MQAR, long parity or all35/165 tasks now pass. Additional
seeds, task families and learned depth remain to be tested. The real proofs
still give capacity and feasible-query identities rather than a theorem that
this approximate optimizer always finds the target family.

## Prospective five-case, three-seed cohort

`three_seed_protocol.json` keeps the CM93 geometry and shared trainable model.
It uses training seeds1,2,3 on the original fixed data splits (data seed1), in
both Mini and forward-only Mini. The budgets are fixed before any new scoring:

| Original case | Updates per mode and training seed |
| --- | --- |
| Boolean AND shift | 1000 |
| Boolean AND random | 1000 |
| Addition | 1000 |
| Short key-value binding / MQAR | 200 |
| Parity16 | 128 |

The binding/parity budgets match the earlier five-case diagnostic. Parity has
only0.1024 epochs at this budget and the proved reader limitation still applies;
this is a diagnostic, not a claim of adequate training or expected success on
every case. Every original split, including longer probes, is retained.

CM94 (`ebe3fd23`, `SharedAtoms/Support`) proves that exactly zero complete-class
gradient rows contribute zero to all complex contractions. The native search
scans the entire stored gradient for these rows at each proposal, preserving
order and every nonzero entry, including tiny ones. It keeps the full gradient
and every parameter vector; there is no rank or magnitude cutoff. This changes
execution cost without changing the mathematical optimizer.

Build with the native commands above in a new target directory, then use
`stress plan-three-seed`. Freeze the checked binary, sources, commit hash and
`three_seed_protocol.json` as `FREEZE/protocol.json`. Start the full cohort with:

```sh
bash experiments/convex_shared_atoms/run_three_seed.sh WORKSPACE STORAGE FREEZE COHORT
```

The runner creates a fresh cohort, caps each of its two workers at32GiB, checks
aggregate RAM and both disks, trains all30 endpoints before scoring, and writes
each command's exit status. It refuses insufficient resources and never removes
old evidence. Each saved endpoint must replay every raw logit independently.
The prior addition seed1 logits provide a separate control for execution
equivalence after the exact-zero optimization. Results remain pending until
the new cohort is complete.

## Finite line-search resolution diagnostic

While the frozen three-seed cohort trains unchanged, its Boolean traces show
many zero component steps and training CE plateaus. A separate independent
two-class control has seven class-zero and three class-one records, all with
the same input, and a candidate logit change of100000000. The optimal class-zero
logit is `ln(7/3)` and optimal CE is the binary entropy of0.7. Twenty-four golden
search iterations return zero and CE`ln(2)`;64 iterations recover the known
optimum using only ordinary CE evaluations. The analytic answer is used only
to check this control, never to propose an optimizer step.

`refined_protocol.json` prospectively tests64 line iterations on original
Boolean AND shift, seed1, both modes, for100 updates. It changes no architecture,
class, coordinate, label, original split, sampler, budget or query geometry.
This is a short diagnostic, not a matched1000-update comparison. Both endpoints
must be sealed before scoring every original split, then every raw logit must
replay. No held-out outcome selected the refinement. The larger cohort and all
earlier frozen producers keep their original protocol and remain intact.

Build and freeze a fresh executable with `refined_protocol.json` saved as the
freeze's `protocol.json`. Use `plan-refined`, then `train-refined` for both modes;
only then run `score-refined` and `verify-refined`, with the same positional
arguments and a fresh cohort directory.

The frozen producer is `41cffd66`, source fingerprint
`0f07c21a8522cb290b06c71f5144b39f666d3564eb594441e02bcb63bbef1929`, binary SHA256
`7fa52c724e036a0faf6a7aac0aa078c966b5c9c9138cc6d6255f4e88260e05c2`, retained in
`/run/media/misha/storage/transformer-runs/freezes/shared-atoms-refined-v1`.
Generated results are in the adjacent `shared-atoms/refined-v1` cohort.
Every command exits zero and both complete raw-logit/metric replays match.

| Split | Mini raw / exact | Forward-only raw / exact | Mini CE | Forward-only CE |
| --- | --- | --- | --- | --- |
| Train | 100% / 100% | 100% / 100% | 0.0002530770 | 0.0002530935 |
| Main validation/novel | 100% / 100% | 100% / 100% | 0.0001680606 | 0.0001680725 |
| Ordinary test | 100% / 100% | 100% / 100% | 0.0000171517 | 0.0000171530 |
| Validation length64 | 64.0625% / 28.125% | 64.0625% / 28.125% | 126.4076 | 128.3244 |
| Test length64 | 72.6563% / 45.3125% | 72.6563% / 45.3125% | 96.4967 | 97.9592 |

Both original success and the primary raw/exact99 flag are true: all14 raw
targets in7 main held-out examples are correct; ordinary test has128/128 raw
targets and64/64 exact examples. Each mode sees3200 indexed examples and6400
supervised records, or25 epochs, in100 updates. Mini takes39.24s and forward-only
40.64s alongside the existing producers, without a transformer timing claim.

Across all length32/64 probes, including position shifts, exact accuracy ranges
28.125%–59.375%. Ordinary position-shift exact accuracy is93.75% on validation
and92.1875% on test. These transfer failures remain failures even though the
source's main success criterion passes. Finer finite search repairs short-task
fitting in this diagnostic; it does not repair length generalization or the
fixed-interaction parity limit. The new CM95 recurrence is proved separately
and has not been trained in this experiment.
