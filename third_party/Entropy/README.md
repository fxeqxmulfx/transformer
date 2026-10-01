# Finite Shannon entropy support

This directory contains the 27-module import closure of
`PFR.ForMathlib.Entropy.Basic` from [teorth/pfr](https://github.com/teorth/pfr),
commit `3d78981`, originally built with Lean 4.34.0-rc2. The remaining PFR
development and its additional package dependencies are not included.

The original namespaces, copyright notices, and Apache 2.0 license are retained.
These sources supply genuine measure, kernel, entropy, conditional entropy,
mutual information, and conditional independence constructions. They are built
against this repository's Lean and Mathlib 4.34.0, and their complete proof
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

The remaining proof bodies and mathematical content are unchanged.
