# Identification certificates for categorical completions

Can actual observed constraints justify choosing between models with equal
ordinary training cross-entropy? Task223 starts from CM28 and Lean commits
`4fee3d09` and `9cd95708`, before native implementation. They prove that a
cyclic joint module identifies an unknown bijective-code phase table over a
composite domain, including repeated input phases. Missing axes cannot be
silently pinned for this claim. The identification is conditional on the
stated categorical model, not a consequence of flatness or exact observed fit.

The first native component constructs an exact ring row certificate. It uses
actual swaps, unit scalings and row additions. If a pivot column contains
nonunits, bounded coefficient search combines rows to reduce the gcd with the
modulus. Only a checked unit is inverted. The result retains the full matrix,
every operation, completed pivots, stop reason and work. A successful replay
has every coordinate row, proving that the original anchored kernel is zero
by the Lean theorem `kernel_zero_of_rows`.

The independent checker uses i128 row arithmetic; the producer uses bounded
u32 residues. It checks each complete row operation, unit scaling, final matrix,
actual coordinate rows and arithmetic work. Full deterministic replay separately
checks every pivot/search decision and control-work counter. Partial reductions,
nonunit obstructions, missing pivots and caps remain explicit refusals. There
are no CE gradients in certificate construction.

Five tests exhaustively compare all 2274 two-variable square matrices modulo
2 through 6 with their actual finite kernels. They also replay the exact
four-instruction nonunit-pivot example from Lean, exercise larger rectangular
systems through modulus257, every work cap, invalid inputs, JSON round trips,
row/state mutations and changed control-work counters. These are synthetic
checks; no original task has been evaluated by this producer.

The categorical layer now binds the certificate to actual observed cell/label
identities and a proposed code. It checks class-code injectivity and every
original equation before shifting the two gauges. It constructs the exact
matrix from up to4096 observed equations, the two shift pins and one actual
unit class anchor. It never inserts pins for missing axes. The independent
checker rebuilds the partition, signed shifts and every coefficient separately
before replaying the ring certificate. A full replay also binds all decisions
and work. Acceptance means uniqueness within the declared bijective cyclic
model; it does not assert that arbitrary ground truth has that structure.

Five additional tests certify masked domains2–16 with repeated phases and
unknown shifted codes, reproduce the exact six-state Lean witness, and check
categorical renaming. A missing-axis example has two different predictions
with identical observed fit; both are refused. Other checks retain collisions,
conflicts beyond the elimination equation cap, budget stops, original-matrix
and endpoint mutations, and a realizable model with a noncyclic joint module.
The last refusal is a limitation of this sufficient certificate, not a proof
that decoded recovery is impossible. All ten native tests pass.

Matrices have at most1024 variables and8192 rows. The declared default permits
two billion scalar updates, two million row operations, twenty million pivot
tests and combination tests, and one million inverse tests. Storage forecasting
includes matrix, instruction, checker/replay and JSON copies and must fit8GiB.
Numerical processes obey the user's64GiB limit:

```sh
cd experiments/convex_stress_identified
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The adapter in `cohort/` now checks the complete saved modular report, including
every admitted/refused view and all raw spectral/CE audits. Each certificate is
bound to the actual observed class order, phase/code point, model hash, and every
proposed full-table prediction. Lean `6a8fb49e` proves the fixed-view scope: a
strip view also requires its constant-strip structural assumption. An interior
certificate alone does not identify arbitrary strip answers. Unequal padding
and missing axes are not given additional identification pins.

The declared training-only rule compares ordinary CE first. Within a1e-12 tie,
it prefers a verified fixed-view certificate, then preserves earlier order.
This is an explicit structural inductive assumption for selecting a model,
not an unconditional truth guarantee over the larger prediction hull. An
earlier selected model with the same certified model hash keeps its priority.
Five adapter tests cover repeated-phase constant strips, earlier missing-axis
pins, CE ties and improvements, changed held-out scores, empty banks, budget
refusals, and mutations of the actual model, matrix, scope and parent evidence.
All five pass; these are additional to the ten core tests.

```sh
cd experiments/convex_stress_identified/cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Next: connect the adapter to the complete original cohort, freeze a separate
producer, and evaluate the already declared rule. No original certification
or changed original selection has been run by this adapter yet.
The complete task222 parent and its equal-CE failures must remain immutable.
All35 tasks, full batch, minibatch, forward-only, retained depth and the
conventional transformer comparison remain in scope.
