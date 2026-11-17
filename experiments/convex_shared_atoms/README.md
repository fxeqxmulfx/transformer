# Shared learned complex factors and generalization

Can a capacity constraint with a learned output factor shared across the input
encoders improve unseen-answer accuracy? The previous CM89 addition diagnostic
fits 100% of training answers and gets 0% of held-out answers correct. This is a
new experiment with unchanged original data and the transformer baseline
preserved. The first frozen run fails: both modes fit only 133/4704 training
answers and predict 1/4705 held-out answers correctly. Generalization is unresolved.

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

Each step shrinks the state by `1-h/B`, putting the freed mass on an exact zero
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

Seven native controls cover exhaustive small-prefix complex execution against
independently materialized coefficients; feasible signed model queries; all
real/imaginary coordinate gradients with duplicate records, repeated pairs and
a one-record tail; a planted nonzero component-search control; and both update
modes with exact checkpoint replay; original masks and full-logit digests; and
refusal of unsealed or changed protocols. These are not original-task scores.

Use native tooling only:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-atoms-native-v1
cargo test --offline --manifest-path experiments/convex_shared_atoms/Cargo.toml
cargo clippy --offline --manifest-path experiments/convex_shared_atoms/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_shared_atoms/Cargo.toml
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
the bias addresses' dominance. Prove that geometry and feasible loss-query
changes before a separate frozen experiment. Keep this failed producer intact.
