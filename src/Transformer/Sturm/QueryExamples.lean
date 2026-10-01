/-
# Repeated-root and signed-query examples

Cancellation in a signed query is different from absence of a feasible
root. The mask queries retain the actual constrained count and minimum.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.Examples

noncomputable section
open Polynomial Transformer.Normalization
open scoped BigOperators

namespace Transformer.Sturm

/-- A degree-six polynomial with a root of multiplicity four at zero
and a root of multiplicity two at one. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
def repeatedExamplePolynomial : Polynomial ℝ := X ^ 4 * (X - C 1) ^ 2

/-- The repeated-root example is nonzero. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem repeatedExamplePolynomial_ne_zero : repeatedExamplePolynomial ≠ 0 :=
  mul_ne_zero (pow_ne_zero _ X_ne_zero) (pow_ne_zero _ (X_sub_C_ne_zero 1))

/-- Its actual finite real zero set consists exactly of zero and one,
despite the larger multiplicities. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem repeatedExamplePolynomial_roots : repeatedExamplePolynomial.roots.toFinset = {0, 1} := by
  rw [repeatedExamplePolynomial, roots_mul repeatedExamplePolynomial_ne_zero,
    roots_pow, roots_pow, roots_X, roots_X_sub_C, Multiset.toFinset_add,
    Multiset.toFinset_nsmul _ 4 (by norm_num), Multiset.toFinset_nsmul _ 2 (by norm_num)]
  simp

/-- The negative and positive query signs cancel even though two
actual real roots are present. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : realRootQuery repeatedExamplePolynomial (2 * X - 1) = 0 := by
  rw [realRootQuery_eq repeatedExamplePolynomial_ne_zero, repeatedExamplePolynomial_roots]
  norm_num

/-- The positivity mask selects exactly one of those two repeated
roots, giving a strictly positive constrained-root count. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : signConstraintQuery repeatedExamplePolynomial 1 [(2 * X - 1, .positive)] = 2 := by
  rw [signConstraintQuery_eq_count repeatedExamplePolynomial_ne_zero,
    repeatedExamplePolynomial_roots]
  norm_num [AnalyticSignRequirement.Holds, Finset.filter_insert, Finset.filter_singleton]

/-- For the actual objective `(x - 1)²` on both roots in `(-1, 2)`,
zero has a strictly better competitor and one has none. The test
distinguishes them even though both roots are repeated. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let V : Polynomial ℝ := (X - C 1) ^ 2
    signConstraintQuery repeatedExamplePolynomial 1 (betterRootConditions V 0 (-1) 2 []) = 8 ∧
      signConstraintQuery repeatedExamplePolynomial 1 (betterRootConditions V 1 (-1) 2 []) = 0 := by
  intro V
  constructor <;> rw [signConstraintQuery_eq_count repeatedExamplePolynomial_ne_zero,
    repeatedExamplePolynomial_roots] <;>
    norm_num [V, betterRootConditions, intervalSignConditions, AnalyticSignRequirement.Holds,
      Finset.filter_insert, Finset.filter_singleton]

/-- The concrete Euclidean chain has generic endpoints, so the signed
interval theorem applies with an actual zero inside. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (∀ f ∈ signedRemainderChain (X : Polynomial ℝ) 1, f.eval (-1) ≠ 0) ∧
    (∀ f ∈ signedRemainderChain (X : Polynomial ℝ) 1, f.eval 1 ≠ 0) ∧
    (sturmVar (signedRemainderChain (X : Polynomial ℝ) 1) (-1) : ℤ) -
      sturmVar (signedRemainderChain (X : Polynomial ℝ) 1) 1 = 1 := by
  have ha : ∀ f ∈ signedRemainderChain (X : Polynomial ℝ) 1, f.eval (-1) ≠ 0 := by
    simp [signedRemainderChain]
  have hb : ∀ f ∈ signedRemainderChain (X : Polynomial ℝ) 1, f.eval 1 ≠ 0 := by
    simp [signedRemainderChain]
  refine ⟨ha, hb, ?_⟩
  have h := signedRemainderChain_query_Ioc X_ne_zero (fun _ _ => by simp)
    (fun _ _ => by simp) (by norm_num : (-1 : ℝ) ≤ 1) ha hb
  norm_num [Finset.filter_singleton] at h
  exact h

end Transformer.Sturm
