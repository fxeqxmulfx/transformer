# Finite Shannon entropy support

This directory contains the 27-module import closure of
`PFR.ForMathlib.Entropy.Basic` from [teorth/pfr](https://github.com/teorth/pfr),
commit `3d78981`, originally built with Lean 4.34.0-rc2. The remaining PFR
development and its additional package dependencies are not included.

The original namespaces, copyright notices, and Apache 2.0 license are retained.
These sources supply genuine measure, kernel, entropy, conditional entropy,
mutual information, and conditional independence constructions. They are built
against this repository's Lean and Mathlib 4.36.0, and their complete proof
dependencies are audited with the paper formalization using them.

Local compatibility changes:

- Remove the two scoped `set_option linter.flexible false` directives in
  `Entropy/Measure.lean` and `MeasureTheory/Measure/Prod.lean`.
- Rename the finite support witness `Measure.support` to
  `Measure.entropySupport`, avoiding Mathlib's topological measure support.
- Rename the fiber-based `CondIndepFun` and its unfolding lemma to
  `FiberCondIndepFun` and `fiberCondIndepFun_iff`, avoiding Mathlib's
  sigma-algebra-based conditional independence API.
- Use Mathlib's already upstreamed `IdentDistrib.prodMk` from
  `Mathlib.Probability.IdentDistribIndep`, instead of the duplicate copy, and
  remove the consequently redundant finite-measure assumption on `ν` from
  `IdentDistrib.mul` and its generated additive version.
- For Mathlib 4.36.0, use the successors of deprecated names (`smul_apply`,
  `FunLike.coe_sum`, `FunLike.coe_smul`, `tsum_prod'`, `CompleteLattice.summable`),
  and pass `simp` the fact `(μ Set.univ).toReal ≠ 0` in
  `measureEntropy_univ_smul`, which it no longer finds by itself. Its
  `contrapose` already cancels double negations, so the eight `contrapose!`
  calls whose negation pushing no longer changes anything become `contrapose`,
  and one `try` whose tactic always succeeds is dropped; Mathlib's linters now
  warn about both.

The remaining proof bodies and mathematical content are unchanged.
