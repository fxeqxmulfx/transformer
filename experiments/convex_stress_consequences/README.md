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
The component tree now normalizes every emitted prime-field code at its first
nonzero class coordinate, then explores all emitted digit lifts within explicit
budgets. The default permits 16 elimination stages, 512 retained states and two
billion scalar updates. Roots receive turns in order; within each root, the
first surviving digit branch is explored in depth before its siblings. This
avoids spending the entire budget across shallow branches that fail at later
powers. Every invalid multiplicity, duplicate, failed lift and capped state
remains in the report. Equal class multiplicities are a proposal filter, not
an assertion that the observed sample is balanced or a completeness theorem.
The checker independently reconstructs the schedule, all original equations,
unit normalization, digit identities, filters and work. Its forecast, including
retained programs and replay copies, must fit 32 GiB.

Five component tests bring the total to 14. Noncyclic repeated-row fixtures at
moduli 4, 8, 16, 32 and 64 recover valid complete codes under the same default
budget. At each larger modulus, distinct codes decode to the same entire table.
The old breadth-first draft exhausted its budget at modulus 32. A missing-row
fixture deliberately retains multiple incompatible completions: the component
tree does not claim to identify that row. Tests also cover all caps, prime-field
normalization and duplicates, altered branches and malformed parent indices.

The composite layer now forms a declared mixed-radix prefix of those emitted
component banks. It retains each actual component-state tuple, every CM27
Bezout witness, the recombined representative, original equations, class-code
collisions and work. Its default permits 64 tuples and 500 million CRT/final
original-row products, in addition to the component budgets. All retained
programs and replay copies must fit a 48 GiB forecast. The checker uses separate
i128 CRT/original-row arithmetic and independently derives tuple indices,
prefix length and collision outcomes. Coverage refers only to the emitted
component banks; an empty or capped component does not establish that no code
exists in the larger family.

Five more tests bring the native total to 19. Fixtures through modulus 40
combine nonunique prime-power codes across two or three coprime components,
with identical correct full-table decoding. A negative fixture has balanced
component marginals but a colliding joint class code and is refused. With a
missing row, the bank retains 12 different tables fitting all observations,
including every possible class at one missing query. Other tests cover point
and product caps, absent component candidates and 16 report mutations.

These are synthetic checks only. Original decoded query certificates and
model/raw-spectrum integration remain required before any separately frozen
original cohort. No new original pass is claimed here.

```sh
cd experiments/convex_stress_consequences
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```
