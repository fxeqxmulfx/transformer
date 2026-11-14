# Bounded signed contexts for the complete complex head

Can signed input kernels improve raw-label prediction beyond positive context
mixtures? This Rust component implements the bounded signed combinations in
CM51, proved at `803958cc`, over the CM50 parent reviewed at `a6be8048`. It uses
original ordinal token coordinates and observed source records. It receives
no task name, arithmetic answer formula or held-out answer.

The fixed bank contains the current context rule and 48 kernels: affine and
quadratic polynomial kernels, and constant-plus-Gaussian kernels at six length
scales, with three axis scalings and two positive ridge parameters. The exact
bank and its ordering are in `protocol.json`. Each kernel solves a regularized
system on observed inputs, then normalizes its signed source weights by an
outward bound on their absolute mass. Deterministic Cholesky factorization
must pass a complete inverse-residual check. A refused fit retains the parent;
blocks with more than 256 observations refuse before allocating a system.

Training diagnostics exclude the entire source state, retaining original
record order, repeats and the short tail. The inverse Schur identity computes
the excluded-state weights without refitting the latent discovery. A frozen
canonical conjugate head with margin 24 supplies ordinary cross-entropy over
the complete raw vocabulary, including absent classes. The baseline reproduces
the earlier CM46 diagnostic with its original floating-point operation order.
It is not the in-sample CE of the retained, trained prediction head.

| Mode | Retained source history | New diagnostic selection |
| --- | --- | --- |
| Full batch | Complete original trained head and records | Best bank vertex, then 64 full conditional-gradient steps |
| Minibatch | Original order, repeats, updates and short tail | Best vertex, then eight epochs with the inherited batch size |
| Forward only | Original endpoint with zero training updates | Best bank vertex; no gradients or optimizer updates |

The diagnostic objective is convex in the shared simplex coefficients while
its bank and canonical head are fixed. This does not assert joint convexity
when both features and the complete prediction head vary. Every diagnostic
gradient batch, record visit, line score and actual update is charged separately
from the original training history. An epoch is retained only if its full
diagnostic CE improves. Forward selection evaluates candidates without taking
derivatives.

At prediction time, signed combinations of the actual complex source features
and the parent feature row multiply every original real and imaginary head
coefficient. No canonical label score replaces that complete product. Every
observed row and earlier geometric prediction is unchanged. Other known unseen
inputs in ambiguous blocks can use the new rule, including blocks with
overlapping observed class hulls. The complete feature mixture is normalized
again to cover accumulation roundoff; a pure baseline preserves exact parent
logits. Loading reconstructs and verifies the parent, fits, diagnostics,
selection, rows, bindings and costs.

CM51 proves a feature bound and a conditional raw-margin result. Its query
reference residual and reference realizability remain unproved for these
kernels. Flatness does not supply unseen labels. Negative weights do not
satisfy the earlier positive-mixture Jensen bound. Ordinal distances are an
additional modeling assumption and can change under token renaming. The native
normalization and residual checks are numerical checks, not Lean proofs of
IEEE arithmetic or certificates of unknown target values.

Eight native tests pass. They independently solve every kernel's excluded
system using pivoted Gaussian elimination, compare actual complete complex
products and ordinary CE, check full and tail gradients, and exercise negative
weights, zero rows, failed systems, source mutations and saved-state replay.
Actual full/minibatch/forward histories contain 45 repeated records with a
five-record tail and 1/6/0 source updates. A nonlinear synthetic fixture uses
signed predictions in all modes. Its training-only diagnostic CE improves;
this fixture is not an original benchmark result.

The prospective forecast includes all retained parent costs, kernel matrices,
factorization, inverse residuals, excluded-state diagnostics, full-domain query
solves, repeated-record optimization, source products, serialization and stored
rows. Limits are 256 billion scalar work units per construction, 56 GiB forecast
buffers and 64 GiB process address space. At most 4096 new diagnostic gradient
batches are permitted. Exact models retain the original observations and head;
kernel inverses are reconstructed rather than serialized. The forecast is a
conservative native guard, not an allocator proof.

The `inspect` example reads the six prior/supplement source roles from the
three reviewed task314 model envelopes. It reports construction and
training-only diagnostics without reading evaluation splits. Freeze the
source, executable, protocol and source-envelope identities before inspection.
The separate original comparison must retain all 35 tasks, both policies,
all modes, every baseline and earlier CE/raw-accuracy tradeoff, including
task298. New original raw accuracy has not been measured by this component.

All six training-only inspections complete under the unchanged limits. Every
role fits all 48 kernels without a numerical refusal. The prior changes 7762
domain inputs and the supplement 7999; all new rows contain negative weights.
These counts describe construction over the full input domain, not correct
held-out answers.

| Mode | Role | Parent diagnostic CE | Signed diagnostic CE | New gradient batches / updates |
| --- | --- | ---: | ---: | ---: |
| Full | Prior | 0.040520401 | 0.034273525 | 64 / 4 |
| Full | Supplement | 0.042716681 | 0.036446071 | 64 / 6 |
| Mini | Prior | 0.044387100 | 0.041838623 | 80 / 80 |
| Mini | Supplement | 0.046591783 | 0.044871384 | 80 / 80 |
| Forward | Prior | 0.048352033 | 0.041838623 | 0 / 0 |
| Forward | Supplement | 0.050552015 | 0.044970927 | 0 / 0 |

The minibatch prior retains its initial best vertex after all eight epochs;
its 80 attempted updates remain counted. Maximum forecast buffer is
11,333,537,152 bytes (10.55 GiB); maximum total construction/optimization work
is 237,011,203,952 scalar units. Inspection takes 114.92 seconds with peak RSS
3,529,764 KiB (3.37 GiB). Generated evidence is in `signed-kernel/inspection-v1`
on the storage runs disk. Full Lean, native INDEX and axiom checks pass with
36 unfinished statements, zero dependent rests and zero extra axioms.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_signed_kernel/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_signed_kernel/Cargo.toml -- --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_signed_kernel/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_signed_kernel/Cargo.toml --example inspect
prlimit --as=68719476736:68719476736 -- FROZEN_INSPECT MODEL_0.json MODEL_1.json MODEL_2.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
```

Generated states, reports and logs remain ignored on the storage runs disk.
