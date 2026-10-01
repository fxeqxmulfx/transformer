# Real preparation adaptation

The copied existence proof comes from
[Bochao Kong's classical-complex-wpt](https://github.com/BochaoKong/classical-complex-wpt)
at revision `b4a7273fe5c9752753c52e10494097569089642d`. The Apache-2.0 license
is retained in `LICENSE`; the source and modifications are recorded in `NOTICE`.
No package dependency was added.

## Statement and definition inspection

The public result `exists_isWeierstrassPreparation` starts from an actual
`AnalyticAt` germ on `(Fin n → ℝ) × ℝ` and its exact distinguished-variable
derivative order. It constructs analytic lower coefficients that vanish at
zero and an analytic unit with nonzero central value. Their product equals
the original function on a neighborhood. The degree is the specified exact
order; no existence premise, root-selection assumption, or gradient
inequality is included in the preparation predicate.

The proof uses Archimedean `ℓ¹` coefficient spaces, antidiagonal convolution,
summability bounds for translated analytic coefficients, a Neumann inverse
for the quotient operator, and reconstruction by convergent power series.
The real adaptation was checked at these scalar-sensitive steps. It does
not use complex roots, complex differentiation, ultrametric assumptions,
or formal series in place of convergent analytic germs. Degree-zero
preparation and nonconstant linear and quadratic examples are included.

The existence dependency closure and necessary coefficient API were
retained. The finite-variable analytic division adapter and its uniqueness
are adapted from LocalComplexGeometry with its separate source attribution.
Preparation uniqueness and unrelated product-indexed interfaces were
omitted. The files compile with the project's Lean and Mathlib 4.34,
with `autoImplicit` disabled and without warnings or unproved declarations.

## External results

Relevant external statements and definitions were checked for the analytic
inverse theorem, change of origin and coefficient summability, analytic
inversion, and finite polynomial roots. In the subsequent real root-lifting
proof, the finite-order factorization of a scalar analytic germ, binomial
coefficient formulas, polynomial root multiplicity and its nonvanishing
quotient, positive real power inversion, and compact subsequence extraction
were also inspected. Their hypotheses apply to the actual real normed
spaces, positive power exponents, and nonzero monic polynomials used here.
Analytic root division was checked using its finite coefficient formula,
and root-set reconstruction retains every real root. Sign stabilization
uses finite scalar analytic order; positive-side zeros are converted to a
full analytic identity only with the punctured-neighborhood isolated-zero
theorem. The plane coordinate equivalence is evaluation at the unique
coordinate with the actual constant-function inverse.

The finite-minimum extension inspects `Finset.exists_min_image`, the
finite-filter interchange lemmas, analytic subtraction and composition,
the analytic Frechet derivative and Riesz map for the actual gradient,
and monotonicity of squaring on nonnegative norms. Sign stabilization
chooses one branch among all admissible roots, rather than presupposing
an analytic minimizer. Preparation gives a two-way zero equivalence on
a fixed box, so minimum comparisons cover every original root in that
box. The projected global-minimum lifting theorem carries an explicit
analytic base curve and pointwise approaching global-minimum witnesses;
it does not assume or claim that such a projected curve has been selected.
Comparisons in its conclusion range over every point of the original set.

Arbitrary-dividend division reconstructs an actual analytic quotient and
analytic coefficients of a remainder of degree strictly below the divisor.
Uniqueness follows from both convergent one-variable power-series uniqueness
over the real field and the two-sided Neumann inverse on the coefficient
space. Its statement only requires analyticity of the quotient and divisor
coefficients; analyticity of the finite remainder functions is unnecessary.
The scalar-sensitive evaluations and all neighborhood pullbacks remain real.
Joint examples include the exponential dividend over `y²-z₀` and the explicit
division `y³ = y (y²-z₀) + z₀ y`.

The reproducible dependency audit is:

```sh
lake env lean scripts/AnalyticPreparationAxioms.lean
```

It checks every copied preparation and real-germ declaration and the external
dependency roots of preparation, division, general ramified roots, Rückert's
theorem, and analytic fiber-value relations, including transitive axioms.
Only `propext`, `Classical.choice`, and `Quot.sound` are accepted. This
mechanical check supplements statement inspection; it does not replace it.
The repository-wide audit remains required:

```sh
lake build
lake env lean scripts/Axioms.lean
python3 scripts/index.py
```

Preparation and ramified polynomial roots alone do not prove the general
analytic gradient inequality. Curve selection is also proved for nonconstant
analytic plane level germs with finite analytic sign conditions. The
outstanding step is selecting analytic curves in the quantified energy-level
gradient-minimizer sets in arbitrary dimension.
