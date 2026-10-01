/-
# Analyticity of every polynomial coefficient

Lower coefficients are the given analytic functions; the leading coefficient
is one and all higher coefficients are zero. This finite coefficient API
allows analytic division by a moving linear root factor.
-/

import Transformer.Normalization.PolynomialTranslation

open Polynomial
open scoped BigOperators

namespace Transformer.Normalization

/-- Every coefficient of an analytic monic family is analytic in its
parameter, including the leading and higher coefficients. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem parameterPolynomial_coeff_analyticAt {d : ℕ} (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (j : ℕ) :
    AnalyticAt ℝ (fun t => (parameterPolynomial d a t).coeff j) 0 := by
  by_cases hj : j < d
  · have heq : (fun t => (parameterPolynomial d a t).coeff j) = a ⟨j, hj⟩ := by
      funext t
      exact parameterPolynomial_coeff a t ⟨j, hj⟩
    rw [heq]
    exact ha ⟨j, hj⟩
  by_cases hjeq : j = d
  · subst j
    have heq : (fun t => (parameterPolynomial d a t).coeff d) = fun _ => 1 := by
      funext t
      simpa only [parameterPolynomial_natDegree] using (parameterPolynomial_monic a t).coeff_natDegree
    rw [heq]
    exact analyticAt_const
  · have heq : (fun t => (parameterPolynomial d a t).coeff j) = fun _ => 0 := by
      funext t
      apply coeff_eq_zero_of_natDegree_lt
      rw [parameterPolynomial_natDegree]
      omega
    rw [heq]
    exact analyticAt_const

/-- Vanishing lower coefficients give the actual central polynomial
`X^d`, rather than only a vanishing value at one point. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem parameterPolynomial_origin {d : ℕ} (a : Fin d → ℝ → ℝ)
    (ha0 : ∀ i, a i 0 = 0) : parameterPolynomial d a 0 = X ^ d := by
  simp [parameterPolynomial, AnalyticPreparation.distinguishedPolynomial, ha0]

/-- A cubic with constant coefficient `t²` and quadratic coefficient `t`
satisfies the analytic and zero-origin hypotheses simultaneously. The
example covers every coefficient index, including those beyond the degree.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then t ^ 2 else
    if i = 2 then t else 0
    (∀ j, AnalyticAt ℝ (fun t => (parameterPolynomial 3 a t).coeff j) 0) ∧
      parameterPolynomial 3 a 0 = X ^ 3 := by
  constructor
  · intro j
    apply parameterPolynomial_coeff_analyticAt
    intro i
    fin_cases i
    · exact analyticAt_id.fun_pow 2
    · exact analyticAt_const
    · exact analyticAt_id
  · apply parameterPolynomial_origin
    intro i
    fin_cases i <;> simp

end Transformer.Normalization
