# Bounded complex refinement of a learned quotient

Can a fixed complex head fit observed raw aliases while preserving the learned
quotient's default answers? CM31 and CM32 in `review/flat_spectrum.md`, proved
at `fbbe4617` and `b52ba949`, establish the representation, convex domain and
feasible updates before this native implementation. Fourteen native checks pass.
The original training-only follow-up now completes all 23 pair cases; held-out
comparison remains separate.

## Model and proof boundary

The input is a validated `convex_stress_quotient::Report`. Discovery is discrete
and uses observed categorical equations. This component fixes its row/column
codes before constructing or optimizing the head; discovery is not jointly
convex. It accepts no task name, target rule, evaluation readout or held-out
labels. Numeric token names are categorical indices, not arithmetic features.

Each discovered latent code has one state for each observed input assigned to
it, plus one default state. Thus the total state count is `N = observations + q`.
The input extractor routes an observed input to its own state and an unobserved
input to its latent code's default. Unknown axes use code zero and remain
explicitly unidentified. An empty latent block has an unsupported class-zero
default; the report counts these defaults and makes no correctness claim for
them. Within a supported block, the most frequent observed raw class is the
representative; ties use the earliest categorical cell of each class, not the
class number. This selection respects permutations of raw class names.

Each block stores its actual normalized complex Fourier matrix `U`. A real
table `T[state,class]` parametrizes the complex head `H = U* T`, where `U*` is
the conjugate transpose. Both real and imaginary coefficient coordinates are
saved. Prediction multiplies the actual saved `U` and `H`; it never substitutes
a direct table lookup for the complex forward pass. The fixed Fourier basis has
unit singular values in exact arithmetic. This is a bound on the chosen feature
basis, not evidence that the target or trained coefficient spectrum is flat.

Observed rows start with margin24 on their raw labels and zero elsewhere.
Default rows have margin24 on their representative. Feasible observed entries
lie in `[-32,32]`, while default rows are fixed affine constraints. CM32 proves
that every such table gives a complex head with real/imaginary L1 column radius
at most `64*N`, a convex feasible set and ordinary full-vocabulary convex CE.
The native update differentiates the actual complex forward pass, pulls the
derivative back through `H(T)`, clips the observed proposal and restores the
fixed defaults. It does not replace the floating-point Gram by an exact identity.

The implementation checks default logits and the column budget after every
update. An independent complex-arithmetic implementation checks gradients and
every reconstructed iterate. Saved steps include batch offsets, loss, both
gradient norms and SHA256 hashes of the complete table and coefficient state.
Endpoint audits separately reconstruct all logits and coefficients, both Gram
products and each actual block's SVD. Their tolerances are numerical checks,
not exact-real certificates. They identify neither arbitrary unseen raw labels
nor the unknown target's compatibility with a declared quotient readout.

## Prospective policy

All three modes initialize from all original observations. Minibatches describe
the subsequent CE updates, not online discovery or initialization from one batch.
The new policy is explicit and does not inherit the older quotient component's
16 full / 256 mini update schedule.

| Mode | CE updates | Batch | Learning rate |
| --- | ---: | --- | ---: |
| Full | 1 | All observed examples | 1 |
| Mini | One epoch | Original order and batch size, including the actual tail | 1 |
| Forward | 0 | None | None |

The initializer already interpolates observed raw labels, so these updates test
preservation and consistency. A separate nontrivial starting table verifies
that full and mini updates genuinely lower ordinary CE and move both complex
coordinates. No optimizer convergence, speedup or additional generalization is
inferred from near-zero initializer loss. Callers may explicitly request up to
4096 updates, subject to the same caps; original comparisons must freeze their
policy before running. Forward mode computes scores but no CE derivatives.

Dimensions are capped at257 input categories,257 latent codes and1024 output
classes. There are at most513 states per block,65536 total states,4 million
stored phases and16 million real head coordinates. No global `N*N` matrix is
allocated. Shape checks precede tensor allocation and reserve
`512 MiB + 128*phases + 128*real_coordinates` below56GiB. Every process also uses
the user's64GiB address-space limit. Audits and each training mode have separate
32-billion scalar-work forecasts, prospectively larger than the old quotient
component's2-billion training allowance. A refused mode retains its reason while
other modes remain available; an unexpected execution failure is returned.

Reports bind the original input, discovery report and three policies by hashes
or exact values. Work counters retain gradient batches/examples, score examples,
head derivatives, table pullbacks, conjugate transforms, clipping entries and
coefficient writes. These are logical operation counts, not measured FLOPs or
total runtime: validation, hashing, SVD and replay require additional work.
Discovery is reused, so new discovery work is zero; the follow-up must still
retain and charge the earlier discovery cost. Construction and checking are
measured separately in the original adapter.

## Component findings and reproduction

The categorical alias fixture has14 raw names sharing seven latent codes. The
old head has positive collision loss; the refined initializer makes zero raw
training mistakes with CE below1e-6 and retains every one of196 pooled answers.
Some unseen raw aliases remain wrong. Full, mini and forward retain the default
logits, and the actual mini tail is replayed. Additional tests cover unsupported
codes, unequal blocks/readout groups, class renaming, reversed observations,
out-of-box proposals, finite differences in both complex coordinates and the
table pullback, weighted unequal batches, serialized float/state/hash replay,
corruption, invalid policies and pre-allocation/work-cap refusals. All14 pass;
formatting and strict Clippy pass. Generated logs remain ignored and local.

```sh
cd experiments/convex_stress_refined_quotient
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- \
  cargo clippy --offline --locked --all-targets -- -D warnings
```

From the repository root, validate the existing proof boundary with
`LAKE_ARTIFACT_CACHE=false lake build` and
`LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean`.
The native index generator's `--check` verifies the committed index. This
component introduces no Lean statement or additional unproved premise.

## Original comparison follow-up

Task243 completed all 1265 frozen stages (23 tasks times55 primes) and their
independent case replays with zero process failures. Six modulus97 stages admit
all three modes: cube-plus, subtraction, addition, square-sum, half-sum and
permuted addition. Their 18 endpoints have zero raw training errors and CE near
3.62e-9. The initializer already fits the observations; this is a consistency
and preservation result, not a convergence or held-out success result.

The complete ledger retains331 Fourier shape refusals,692 source stages without
a code and236 source-stage errors. Fibonacci remains unavailable. Construction
takes185.750755602 recorded seconds and120 gradient batches; forward adds none.
Peak measured run/replay RSS is272068KiB, below the64GiB process cap. Discovery
is reused, so its earlier cost remains charged in the source records. The
training-only summary evaluates no held-out split and selects no parent model.
See `cohort/README.md` for frozen hashes and reproduction.

Land and independently review this component before any original-data run.
A separate adapter and protocol must bind previously saved categorical
discoveries to the original raw observations and original batch order. Preserve
every discovery stage, refusal and cost, both complete105-choice parent arms,
all35 tasks, the three modes, depth outcomes and conventional transformer.
Use observed CE for any candidate selection; keep readout compatibility distinct
from target identification. Evaluate with each task's original declared primary
success rule and retain every additional split as a diagnostic. Construction
may be staged using training data only after its own implementation review;
comparison still waits for the required reviewed original parents.

The running task232 catalogue and completed task237 causal stage remain frozen.
Their unchanged artifacts and executable hashes must survive this addition.
DiffGPT's existing105/105 record across seeds0/1/2 remains a separate reference,
not a replacement for the retained conventional transformer or a freshly
replayed result. Component fixtures do not change the current34/35 full,
34/35 mini and33/35 forward benchmark counts.
