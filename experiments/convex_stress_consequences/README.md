# Composite proposals with nonunique code coordinates

Can bounded exploration of nonunique code spaces recover proposals that the
rank-one component policy refuses, while CM29 certificates still identify
decoded answers? Lean `af4a6d14` precedes native work. Its four-state witness
identifies every answer although the actual joint observed module is
noncyclic. Task225 keeps the task223 frozen original comparison unchanged.

The first component enumerates an explicit finite prefix of an independently
checked prime-field affine space. It retains the original elimination, every
parameter tuple and point, exact work, cardinality/overflow and stop reason.
Every point is checked against every original equation. A separate i128 checker
recomputes tuples, unreduced affine combinations and original row evaluations;
full replay binds all elimination and enumeration decisions. No emitted point
is declared unique, and a bounded prefix is not called a complete space.

The default permits256 points,500 million coordinate products and500 million
original-row products. Field elimination has its own explicit work/nullity
caps. Forecasts include all report/check/replay copies and must fit8GiB; numerical
processes keep the64GiB user limit. Cardinalities beyond u64 remain explicit,
including the257-to-the-eighth synthetic case, without allocating that space.

Five tests cover all793 two-variable affine systems modulo2 and3 against
exhaustive original solutions, nonzero particulars and ordered multidimensional
parameters, every cap, changed matrices/tuples/points/work, and larger domains.
The digit-lift layer now propagates every enumerated digit point through the
exact CM27 residual identity, preserving its order and original residuals.
An independent i128 audit verifies the base, reduced field problem, each
actual next-modulus point and exact work. Four further tests check1,456 small
systems against direct exhaustive representatives, genuinely nonunique digits,
failed lifts, bounded prefixes and altered branches. All nine tests pass.
These are synthetic checks only. Complete multi-stage composite branching,
original decoded query certificates and model/raw-spectrum integration remain required before
any separately frozen original cohort. No new original pass is claimed here.

```sh
cd experiments/convex_stress_consequences
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```
