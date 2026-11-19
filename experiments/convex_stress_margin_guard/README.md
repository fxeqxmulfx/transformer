# Convex context refinement with fixed raw margins

Can confidence improve while the stronger reference's raw predictions stay
fixed? Task298 improved validation cross-entropy but regressed raw accuracy
against task296. This component fixes the reviewed block-local CM46 model from
`a8deaa63` as its reference. Every accepted coefficient update must retain at
least half of each actual reference margin and the exact metric argmax on
every eligible input. No unseen target label enters these constraints.

The Lean development comes first: CM47 at `85aa4a2e` proves convexity of the
all-query, all-class margin domain, feasibility of the maximizing reference,
convexity of ordinary CE with a separately fixed diagnostic bank, and preservation
of strict comparisons and first-index argmax at positive retention. Its nonzero
Fourier examples retain wrong labels and expose failure at zero retention.
See `review/convex/moments.md`, CM47, and
`Transformer.Grokking.Convex.ConvexDepth.MarginGuard`. This numerical
implementation does not add an unseen-label identification theorem.

On eligible inputs, the fixed bank contains 93 maps: the exact reference, a
fixed half-scale copy of the reference, then the original 91 context maps.
Every bank uses the exact parent on observed and unsupported inputs.
CM43 `features_bound` and
`logits_bound` bound the scaled reference, and CM46 bounds each reference and
context mixture with the same complete source head. The five source-excluded
count-profile gates from `5e89c616` share coefficients across blocks; the source
block still chooses the complex features and coefficients. Radius and retained
margin are both 0.5. Initially every local simplex row is the reference vertex.

Only these simplex coefficients are optimized. The ordinary full-vocabulary CE
diagnostic retains the original ordered records, repetitions and short tails.
Its reference bank entry reconstructs the original CM46 source-excluded
canonical margin24 diagnostic; the next entry is half that vector. Inactive raw
classes remain in the softmax denominator. Source-state exclusion does not refit
discovery, which already used the observed training records. Flatness supplies
a bound; the context bank and count profiles supply additional structure.

| Mode | Initialization | Subsequent updates |
| --- | --- | --- |
| Full batch | Best capped vertex per gate, then actual verification | 64 complete ordinary CE gradients |
| Minibatch | Same | Eight epochs in original order at the inherited batch size |
| Forward only | Same | Zero gradients and zero optimizer updates |

Initialization and each later proposal use analytic affine segment caps over
every guarded input and every output class. A fixed six-point line search
selects later steps by training diagnostic CE. Cached weighted state/head
products only propose the cap: actual mixed-feature products must pass the
constraints and exact `total_cmp` argmax before acceptance. A failed proposal
is halved at most 16 times, then rejected. Absent minibatch gates stay unchanged.
The lowest complete diagnostic at initialization or an epoch endpoint is
retained. This is a finite feasible search, not an exact constrained linear
oracle or a claimed global optimization result.

Inference uses a fixed arithmetic order. The reference mass is
`0.5 + 0.5 * (a[0] + 0.5 * a[1])`. Remaining context weights form a residual
complex feature row. Scores add the complete original head product of this
row to the scaled exact reference scores. The reference vertex returns the
parent exactly. Pure reference rescaling keeps signed zeros. After optimization,
the public prediction path recomputes every guarded score from its stored rows
and must reproduce the final score hash. Observed and unsupported paths remain
the exact parent, including every original observed-head derivative.

The ten native tests cover dense ordinary CE and finite-difference gradients,
full/mini/forward source training and original derivatives, repeated records,
short tails, two nonzero complex source blocks sharing a gate, actual complex
products and feature/head bounds, adversarial vertex rejection, exact/near/
signed-zero ties, count boundaries and exclusion, preallocation refusal, and
corrupted or rehashed saved state. Reconstruction checks the entire parent,
profiles, gates, coefficients, capped candidates, trace, certificate and costs.

On the 47-record noisy fixture, full batch retains CE0.293904343. Minibatch and
forward reduce diagnostic CE0.325069508 to0.302528484 with zero raw prediction
changes over the whole finite input domain. Minibatch performs28updates;
forward performs none. The separate multiblock fixture guards four unseen
queries in two source blocks sharing one count gate. These are component
fixtures, not a new original-suite result.

The preallocation forecast counts every eligible cell and its actual source
block size before allocating caches. It includes all raw classes, complete
state/head products, context maps, score states, serialization/reconstruction
buffers, and the worst case of all17attempts for initialization and each update.
Memory remains capped at56GiB forecast and64GiB address space. The explicit
finite scalar-work guard is8trillion, increased from the prior32billion to
cover all-query/all-class verification and rejected steps; it is a conservative
upper bound, not measured runtime. The gradient-batch cap remains4096. Runtime
metadata separately records diagnostic visits, cap visits, guard visits,
actual head terms, conservative feature-work charges, and rejected proposals.
Inherited costs remain present in full. Generated evidence is ignored/local.

Positive margins preserve existing mistakes as well as correct answers.
Raw argmax preservation does not certify probability-pooled field decisions,
validation CE improvement, or unseen correctness. A separate complete35-task
comparison follows exact-commit component review, retaining the task296 raw
reference, the task298 CE reference, all modes, refusals and original baselines.
Prior suite results motivated the architecture, so it is not an untouched
architecture-selection holdout.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_margin_guard/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_margin_guard/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_margin_guard/Cargo.toml --all-targets -- -D warnings
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
```
