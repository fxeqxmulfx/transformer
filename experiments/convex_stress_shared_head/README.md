# Refit the complete head on shared input features

Can ordinary raw-label cross-entropy improve the remaining original35 miss
when the same shared features are used on observed and unseen inputs? Reviewed
task319 leaves Fibonacci at97.9592% full-batch and98.0230% minibatch/forward raw
accuracy. Its observed CE is exactly unchanged: the old feature maps preserve
every observed source row. The new component will replace those features and
refit the complete head. There are no new model results yet.

The Lean audit precedes native implementation. CM42's
`ContextInterpolation.batchLoss_convex` quantifies over arbitrary fixed real
context weights, source features, input and raw-label maps, original record
IDs and batch. It has no Dirac-row hypothesis. CD8's `headLoss_convex` and
`headBudget_convex` likewise permit a freshly initialized complete head and
arbitrary fixed features. Thus ordinary full-vocabulary CE of the actual new
observed logits is convex in every real and imaginary coefficient, with the
per-class absolute-sum budget. Repeated inputs keep distinct record IDs and
the final minibatch uses its own cardinality.

CM51's `normalized_mass_le_one` and `normalized_features_bound` apply to any
fixed signed feature row divided by the larger of one and its absolute mass.
Bounded complex source coordinates therefore stay bounded. CD8's
`headLogits_bound` charges every coefficient and bounds each actual logit by
the head budget times that coordinate bound. A block selector fixed by an
earlier training-only representation is part of the fixed feature map; it is
not another factor optimized jointly with the head. Neither unitary source
features nor these bounds identify unseen labels.

The audit deliberately does not invoke CM42's `batch_loss_eq`,
`batch_fderiv_eq` or `batch_hasFDerivAt`: their observed Dirac-row premise is
absent. Observed CE, gradients, head coefficients and predictions can change.
CM45 permits a learned mixture only with its template head fixed; it does
not justify multiplying learned mixture weights by this fresh learned head.
No such learned factorization is proposed. These existing statements cover
the required algebra, convex domain and bounds; no new theorem is needed.
`scripts/SharedHeadAxioms.lean` checks their axioms and instantiates genuinely
shared observed rows, repeated records and a short final minibatch.

The prospective native feature family uses ordinal coordinates of the last
two input tokens and the fixed learned source block. Within a block it uses
`[1,u,v,u*u,u*v,v*v,0,0]`, with coordinates centered in[-1,1], absolute-mass
normalization and the complete8-state complex Fourier basis. All raw classes
and both coordinates of every coefficient remain explicit. The coefficient
budget is65536 per class across the entire model. Ordinal, quadratic and
block-sharing biases are declared; no arithmetic answer formula, task name,
generator or held-out score enters construction or selection.

Initialization uses only original training inputs and raw labels. All16 fixed
ordinal directions are checked when a block has two observed classes; the
largest positive interval gap chooses a direction and its midpoint threshold,
with first-index ties and minimum normalized pair margin4. Every candidate,
gap and selected threshold is recorded. Overlapping intervals or more classes
use explicit regularized quadratic class moments (covariance ridge0.01).
Single-class blocks use a constant score, and empty blocks use zero logits.
Absent raw classes remain explicit. A common bias separates observed classes
from absent ones; the entire coefficient tensor is projected afterward and
its actual CE is recorded. This is initialization/search work, not CE descent.

| Mode | Ordinary CE updates | Record handling |
| --- | --- | --- |
| Full | 128 projected complete-head steps, rate16 | Every original record in every step |
| Mini | 8 epochs, rate1, batch127 | Original order, repetitions and actual short tail |
| Forward | Zero gradients and updates | Same declared initialization/search, charged separately |

The final iterate is reported even if CE increases. There is no heldout choice,
restoring of old observed logits or auxiliary loss in the CE updates. Global
per-class L1 projection and stable full-vocabulary CE reuse
`convex_stress_depth/src/head.rs` at `3dac93c32fed41063a8618e4dc83e066da8a8af7`.
The feature map and initializer differ; both actual complex coordinates of
every raw-class coefficient learn. No initializer accuracy, native floating-point
refinement or optimizer convergence is certified by the Lean audit.

Eight synthetic native checks cover independent products after arbitrary head
changes, nonidentity decoders on unseen inputs, quadratic separation of
overlapping linear intervals, every-coordinate finite differences, weighted
minibatches with a real tail, global projection conditions, exact saved-state
replay/mutation rejection, and extraction of only original training records
from the frozen parent. Full/mini change the actual observed CE, while forward
has zero CE derivatives/updates. These fixtures are not original task scores.

Before allocation, the component forecasts every initializer, batch, complete
product, projection and final objective under1T kernel work and56GiB explicit
buffers, with a64GiB process limit. It also bounds record IDs retained in step
histories. The original parent is an externally hashed reference; a future
adapter must separately charge its reconstruction and all new scoring/replay.
The inspection CLI allows at most1GiB of parent JSON and reserves32 times its
bytes plus1GiB for parsing, together with the component buffer forecast. It
then saves input/forecast before training and reconstructs the entire saved
outcome exactly. Hashing, serialization and process resources remain measured.
Before original fitting, a shape-only read finds389/97 source blocks,256 raw
classes and4704 records in the two retained roles. The initial256B allowance
would reject the larger complete tensor's conservative projection forecast.
The prospective limit is1T, retaining the same formula and all coefficients.
A ninth check uses these dimensions and mutates label values to verify that
admission depends on shape; no new model is built or evaluated by that check.

Task329 next freezes the checked component before training-only inspections of
both roles of all three selected319 heads. `inspection-inputs.json` pins every
complete parent artifact; no historical metric is passed to initialization or
training. Task330
separately evaluates all35 tasks, preserving both old selection policies,
all full/mini/forward endpoints and transformer references. Every changed
prediction and observed CE tradeoff must remain visible. The99% target is per
task, not pooled accuracy, and no attainment is inferred before evaluation.

```sh
export LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-head
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_shared_head/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_shared_head/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_shared_head/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_shared_head/Cargo.toml
prlimit --as=68719476736:68719476736 -- lake build
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/SharedHeadAxioms.lean
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
prlimit --as=68719476736:68719476736 -- scripts/native_index/target/release/lean-index --check
experiments/convex_stress_shared_head/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-shared-head"
FRESH_FREEZE/source/experiments/convex_stress_shared_head/run_inspection.sh FRESH_FREEZE/convex-stress-shared-head FROZEN319_FIBONACCI_DIRECTORY FRESH_INSPECTION_ROOT
```

Generated evidence remains on storage under64GiB. Run the same frozen inspector
in a second fresh directory and compare every JSON artifact. Native inputs,
source, executable and all refusals remain visible; no99% result is inferred
from training-only inspection.
