# Complete observed-kernel codes with a bounded complex head

Can retaining every compatible coefficient direction remove the older model's
one-direction restriction without losing convex complex cross-entropy training?
CM34 in `review/flat_spectrum.md`, proved at `285dcdb6`, establishes the exact
mathematical construction first. This Rust component implements and checks it.
All 20 component tests pass. Original benchmark evaluation is a separate step;
these fixtures do not change any benchmark score.

## Discovery and identification boundary

The input consists of original categorical observations, the full output
vocabulary, a prime and explicit limits. Rows, columns and observed class names
have separate dictionaries in first-occurrence order. Every supplied equation
is retained: row + column - class = 0. The only additional equations fix the
first row and first class to zero, the two shift gauges justified in CM34.
Missing axes never receive invented zero coordinates. No task name, arithmetic
answer rule, readout or held-out label enters discovery.

The unchanged `convex_stress_quotient` dependency supplies the finite-field
solver and its independent row-operation checker. This component retains the
entire nullspace basis, including a valid zero-dimensional kernel. An additional
checker verifies every direction against every original equation and verifies
full rank on the RREF free coordinates. Together, the original-system replay,
dimension and independence checks establish completeness. Dropped, dependent,
forged and incorrectly gauged directions are rejected.

A query code is the vector of row-plus-column values across all basis
directions. Its matches with observed class vectors and the class alias
partition are invariant under invertible basis changes. Retaining one coordinate
alone can create a false match; a two-direction fixture demonstrates this.
The runtime records full-vector matches, not proof of an unknown target's
realizability or readout compatibility. CM34's conditional decoded-correctness
theorem still requires those premises. A matching unseen code need not identify
a raw alias that has never appeared among the observations.

## Actual complex head and ordinary CE

The state dictionary contains only the vector codes attained by active row and
column pairs, in first-pair order. It allocates no ambient prime-to-the-dimension
grid. Each state has one Fourier coordinate per original observed input in its
fiber and one default coordinate. Thus N = observations + actual range size,
with range size at most active rows times active columns.

The default label is the first matching observed class in the fixed bank.
An unmatched active state uses the first observed raw class and remains marked
unmatched. A query with an unknown axis uses that class's existing default state
and remains marked unknown. These distinct masks survive serialization and
checking; neither fallback is asserted correct.

Within each block, an actual normalized complex Fourier matrix U transforms a
real table T into complex coefficients H = U* T. Prediction multiplies the saved
complex tensors. Both real and imaginary derivatives of ordinary full-vocabulary
softmax cross-entropy are computed, then pulled back through the conjugate
transform. There is no answer-table substitution in the forward calculation.

Observed table rows initialize at margin 24 on the original raw label. Default
rows initialize at margin 24 on the fixed representative. Observed coordinates
are constrained to [-32,32]; defaults are fixed affine constraints. CM32–CM34
give convexity after the extractor is frozen and a real/imaginary L1 column
budget of 64N. Discovery itself is discrete. Unitary feature blocks bound the
chosen feature spectrum; they do not establish flatness of the unknown target
or identify its missing answers.

| Mode | Ordinary CE updates | Batch | Rate |
| --- | ---: | --- | ---: |
| Full | 1 | All observations | 1 |
| Mini | One epoch | Original order and batch size, including the tail | 1 |
| Forward | 0 | None | None |

All modes use all observations during discovery and initialization. The
initializer already fits them, so its near-zero loss establishes neither
optimizer convergence nor unseen generalization. Tests also use a nontrivial
starting table: both full and mini updates lower CE and move both complex
coordinates. Forward computes scores without CE derivatives. Explicit custom
policies allow at most 4096 updates with 0 < rate <= 1, subject to the work cap.

The numerical checker reconstructs every iterate using separate complex
arithmetic, compares actual gradients and hashes of all table/coefficient
coordinates, and audits the final CE, defaults, L1 budget, both Gram products
and each actual block's SVD. Finite differences test both complex coordinates
and the table pullback. These floating-point checks are not Lean execution
certificates. The exact full/minibatch margin-preservation result is CM33.

## Limits and charged work

Inputs allow 2–257 categories per axis and 2–1024 output classes. Discovery
requires all observations to fit the declared equation cap (at most 8190), at
most 1024 coefficient variables and at most eight kernel directions. Its
default elimination limits are 500 million scalar updates and two million
row operations. Refusals preserve their reasons, field certificates and costs;
the equation cap never selects a prefix. Discovery forecasts at most 8 GiB.

Head limits are 66049 actual vector states, 513 Fourier coordinates per block,
131072 total coordinates, four million phases and 128 million real coefficient
coordinates. Before head allocation, the conservative buffer forecast is
1 GiB + 256 bytes per phase + 512 bytes per real coefficient coordinate,
and must stay below 56 GiB. This reserves space for the initializer, three
endpoints, replay models, derivatives, numerical serialization buffers and SVD;
it is not a measured RSS or allocator theorem. Every process also uses the
user's hard 64 GiB address-space limit. Actual head shapes above the older
16-million-coordinate limit are allowed when the forecasts pass.

Auditing and each training mode have separate 32-billion scalar-work forecasts.
The training forecast includes verification of the full vector encoding on
every update. Recorded work includes field elimination, code coordinates,
vector additions, gradient batches/examples, score examples, head derivatives,
table pullbacks, transforms and coefficient writes. These logical counts are
not measured FLOPs or total execution time; validation, hashing and replay add
work. Construction and mode refusals are explicit, and a refused mode preserves
other available modes. Original adapters must retain all earlier costs too.

## Component findings and reproduction

The tests cover genuine two-dimensional kernels, invertible shears, a false
one-coordinate match, forced raw aliases, an entirely unobserved raw class,
unknown axes, unmatched active codes, a zero-dimensional kernel, categorical
renaming, exact serialized replay, parameter/certificate/trajectory mutations,
real/imaginary finite differences, weighted unequal minibatches, clipping,
full/mini/forward preservation and allocation/work refusals. Twenty tests,
formatting and strict Clippy pass under the memory limit. No original task has
been evaluated by this component yet.

```sh
cd experiments/convex_stress_kernel
prlimit --as=68719476736:68719476736 -- cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- \
  cargo clippy --offline --locked --all-targets -- -D warnings
```

From the root, use `LAKE_ARTIFACT_CACHE=false lake build`,
`LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean`, and the native
index generator's `--check` to validate the existing proof boundary. No Lean
statement changes in this component. Generated evidence remains ignored and
local outside the temporary worktree.

The Fourier arithmetic, complex audits and CE derivative paths are attributed
ports of `convex_stress_refined_quotient` at `285dcdb6`. Changes replace scalar
codes with the complete vector range, bind observed raw labels explicitly,
choose the CM34 ordered-bank default, separate unknown and unmatched inputs,
and extend resource forecasts. The frozen 232/237/243/244/248 sources,
executables and protocols remain unchanged. The original adapter and any
training or held-out cohort require separately reviewed prospective protocols.
The retained benchmark counts remain 34/35 full, 34/35 mini and 33/35 forward;
DiffGPT's recorded 35/35 on each of three seeds is a separate reference.

The task251 adapter in `cohort/` now binds all23 original pair tasks and all55
primes while retaining the35-task suite metadata. Seven fixture checks exercise
complete original-kernel/model replay, input and scope binding, corrupted
numerical states after rehashing, explicit refusals and process failures.
Its original training run still requires its own reviewed prospective freeze;
no held-out evaluator or parent selector is part of the adapter.
