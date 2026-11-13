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

Next: bind this certificate to the actual categorical observation matrix and
model, independently check the class code and all original observations, and
declare a training-only selection rule before a separate original freeze. A
strip view needs its constant-strip structural assumption stated explicitly;
an interior certificate alone does not identify arbitrary strip answers.
The complete task222 parent and its equal-CE failures must remain immutable.
All35 tasks, full batch, minibatch, forward-only, retained depth and the
conventional transformer comparison remain in scope.
