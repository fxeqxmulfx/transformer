# Local real polynomial sign-query audit

## Provenance and compatibility

Source: [leanprover/hex-real-roots-mathlib](https://github.com/leanprover/hex-real-roots-mathlib)
revision `53ce31466dc5ff520463249d470119c1e0006e22`.
The copied closure consists only of `Sign.lean`, `SturmChainDefs.lean`, and
`SturmTheorem.lean`, split by proof dependency under `src/Transformer/Sturm`.
The original Apache-2.0 license is retained byte for byte in `LICENSE`.
The upstream toolchain is Lean 4.34.0-rc2; the adapted closure and all new
proofs are built with this project's Lean and Mathlib 4.34.1.
The deprecated sign import is replaced by `Mathlib.Basic.Sign.Basic`.
No Lake dependency, executable, integer certificate machinery, or other
upstream module is added.

## Inspected meanings and hypotheses

- Sign variations remove all zero entries and then count adjacent opposite
  signs. They depend on actual real polynomial evaluations or actual leading
  coefficients, with the degree parity included at negative infinity.
- `IsSturmChain` is a genuine geometric predicate: its head is the input;
  all entries are nonzero; interior zeros have nonzero opposite neighbors;
  the last entry has no real zero; the head pair crosses from negative to
  positive at every head root. The predicate does not assume a root count.
- The copied interval theorem counts the actual Mathlib real-root multiset
  on `(a, b]` and explicitly requires that multiset to be duplicate-free.
  The count at a head root equals its right-hand value. The proof handles
  simultaneous nonadjacent interior zeros and induction over the finite
  union of all actual chain-root sets.
- The real-line theorem uses endpoints beyond every chain zero and proves
  the signs at infinity from polynomial order results. Real coefficients
  are arbitrary; no splitting over the reals or rational coefficients are
  assumed.

## Project extensions

The project proves signed Euclidean-chain construction with decreasing
remainder degrees. Absence of common real zeros propagates to each new
negative remainder; evaluation of the exact division identity proves
opposite neighboring signs and a terminal entry without real roots.
A positive derivative at a zero proves the required Sturm head crossing.

`p / gcd(p, p')` preserves every real root of a nonzero polynomial and
reduces each root multiplicity to exactly one. The proof checks derivative
multiplicity in characteristic zero and exact quotient factorization.
No simplicity assumption remains on the original input.

The signed extension retains both crossing orientations. Its interval
formula requires generic endpoints and sums the signs of `p'(r) q(r)`.
It does not use the unsigned theorem's right-continuity convention at
negative crossings. Bounding all chain zeros gives the signed real-line
formula. A further gcd removes exactly the roots where the query vanishes;
choosing the second polynomial as `p' * q` gives the Sturm--Tarski sum of
`sign(q(r))` over all distinct real roots. This covers arbitrary repeated
roots and a query vanishing at any or all roots.

Quadratic masks in the integer sign are twice the indicators of equality,
strict positivity, and nonnegativity. Their finite recursion gives `2^m`
times the number of roots satisfying all `m` original conditions. The
positive factor prevents cancellation from being mistaken for infeasibility.
Open-interval bounds and a strict objective improvement are additional
actual polynomial signs. A zero query therefore eliminates comparison
with every admissible scalar root exactly.

`Normalization.AnalyticFiberMinimumQuery` combines this with real analytic
preparation and division on one fixed box, for arbitrary base coordinates.
It retains the original equation, every finite analytic sign condition,
and the full universal comparison within the distinguished-variable fiber.
It assumes no analytic curve in the base.

`Normalization.AnalyticBoxMinimumQuery` uses the candidate's actual objective
as a common threshold. Its zero-query conditions compare with every admissible
base coordinate in an arbitrary chosen base slice, including a prescribed
energy coordinate. The scalar competitor is eliminated from a comparison on
the whole fixed box. The universal base quantifier remains explicit.

The finite-atlas extension compares one common threshold on all constrained
energy-graph chart images, with distinct degrees and dimensions allowed.
`Normalization.CompactBallGradientQueries` constructs such comparisons for
the entire original compact ball from analyticity. The nonflat part of the
central fiber is compact. Its prepared charts have actual inverse energy
coordinates, lie in the original ball, and contain all nearby source points.
A finite subcover then covers every sufficiently nearby noncentral energy
fiber. Flat central neighborhoods contain no competitor on those levels.
Both the closed-ball constraint and the actual squared gradient norm retain
their analytic division values, including at boundary centers.

## Limits

The query is a finite arithmetic construction, not yet a uniform finite
analytic-sign description of its parameter set. Proving that description
requires controlling all degree and division branches near singular
parameters. Eliminating the remaining base quantifiers and analytic curve
selection in the projected set remain unproved. The general
`lojasiewicz_modulated` theorem remains sorried.

## Dependency checks

Statements, relevant definitions, typeclass assumptions, and transitive
proof dependencies were inspected, including Mathlib's polynomial field
division, gcd evaluation, multiplicity, derivative, root multiset, real
polynomial order, continuous sign, and finite-sum APIs. The local examples
include a genuine Sturm root crossing, a neutral interior sign collapse,
nonzero endpoint evaluations, repeated roots, vanishing query values,
signed cancellation, mixed sign constraints, and different minima of an
actual polynomial objective.

Run `lake env lean scripts/AnalyticPreparationAxioms.lean` after building.
It checks every noninternal `Transformer.Sturm` declaration and the
external dependency roots of the final preparation, signed-query, and
analytic minimum reductions transitively. Only `propext`,
`Classical.choice`, and `Quot.sound` are accepted. The separate full-tree
check is `lake env lean scripts/Axioms.lean`.
