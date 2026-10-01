/-
# Coefficient formulas for polynomial translation

Finite binomial formulas describe the coefficients after translation in
the polynomial variable. The translation retains the monic degree, so
the coefficients completely reconstruct the translated polynomial.
-/

import Transformer.Normalization.PolynomialFamily
import Mathlib.Algebra.Polynomial.Taylor

open Polynomial Finset
open scoped BigOperators

namespace Transformer.Normalization

/-- A parameter polynomial has exactly the prescribed degree, including
degree zero. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem parameterPolynomial_natDegree {d : ℕ} (a : Fin d → ℝ → ℝ) (t : ℝ) :
    (parameterPolynomial d a t).natDegree = d := by
  apply natDegree_eq_of_degree_eq_some
  change (X ^ d + ∑ i : Fin d, C (a i t) * X ^ i.val : Polynomial ℝ).degree = d
  have hlow := degree_sum_fin_lt (fun i : Fin d => a i t)
  have hlt : (∑ i : Fin d, C (a i t) * X ^ i.val : Polynomial ℝ).degree <
      (X ^ d : Polynomial ℝ).degree := by simpa using hlow
  rw [degree_add_eq_left_of_degree_lt hlt]
  simp

/-- The lower coefficient of a parameter polynomial is its specified
coefficient function. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem parameterPolynomial_coeff {d : ℕ} (a : Fin d → ℝ → ℝ) (t : ℝ) (i : Fin d) :
    (parameterPolynomial d a t).coeff i.val = a i t := by
  simp only [parameterPolynomial, AnalyticPreparation.distinguishedPolynomial,
    coeff_add, coeff_X_pow, ite_eq_right i.isLt.ne, zero_add,
    finsetSum_coeff, coeff_C_mul_X_pow]
  simp [Fin.val_inj]

/-- A monic polynomial of known degree is reconstructed from its lower
coefficients and its leading monomial. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem monic_eq_leading_add_fin {d : ℕ} (P : Polynomial ℝ)
    (hmonic : P.Monic) (hdegree : P.natDegree = d) :
    P = X ^ d + ∑ i : Fin d, C (P.coeff i.val) * X ^ i.val := by
  have htop : P.coeff d = 1 := by simpa only [hdegree] using hmonic.coeff_natDegree
  calc
    P = ∑ k ∈ range (d + 1), C (P.coeff k) * X ^ k := by
      simpa only [hdegree] using P.as_sum_range_C_mul_X_pow
    _ = (∑ k ∈ range d, C (P.coeff k) * X ^ k) + C (P.coeff d) * X ^ d :=
      sum_range_succ _ _
    _ = X ^ d + ∑ i : Fin d, C (P.coeff i.val) * X ^ i.val := by
      rw [htop, C_1, one_mul,
        Fin.sum_univ_eq_sum_range (fun k => C (P.coeff k) * X ^ k) d]
      exact add_comm _ _

/-- The translated coefficient is a finite binomial expression in the
original coefficients and the translation value. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem parameterPolynomial_taylor_coeff {d : ℕ} (a : Fin d → ℝ → ℝ)
    (t c : ℝ) (k : ℕ) :
    (taylor c (parameterPolynomial d a t)).coeff k =
      c ^ (d - k) * (d.choose k : ℝ) +
        ∑ i : Fin d, a i t * (c ^ (i.val - k) * (i.val.choose k : ℝ)) := by
  simp only [parameterPolynomial, AnalyticPreparation.distinguishedPolynomial,
    map_add, map_sum, taylor_mul, taylor_C, taylor_X_pow,
    coeff_add, finsetSum_coeff, coeff_C_mul, coeff_X_add_C_pow]

/-- The binomial coefficients reconstruct the full translated monic
family, without any restriction on the parameter or the polynomial
variable. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem polynomialFamily_translation {d : ℕ} (a : Fin d → ℝ → ℝ) (c : ℝ → ℝ) :
    let b : Fin d → ℝ → ℝ := fun i t =>
      (taylor (c t) (parameterPolynomial d a t)).coeff i.val
    ∀ t y : ℝ, polynomialFamily d a (t, y + c t) = polynomialFamily d b (t, y) := by
  intro b t y
  have hmonic : (taylor (c t) (parameterPolynomial d a t)).Monic := by
    change (taylor (c t) (parameterPolynomial d a t)).leadingCoeff = 1
    rw [leadingCoeff_taylor]
    exact parameterPolynomial_monic a t
  have hdegree : (taylor (c t) (parameterPolynomial d a t)).natDegree = d := by
    rw [natDegree_taylor, parameterPolynomial_natDegree]
  have heq := congrArg (fun P : Polynomial ℝ => P.eval y)
    (monic_eq_leading_add_fin _ hmonic hdegree)
  simpa only [taylor_apply, eval_comp, eval_add, eval_X, eval_C,
    parameterPolynomial_eval, polynomialFamily, eval_pow, eval_finsetSum,
    eval_mul, b] using heq

/-- A translated cubic illustrates reconstruction and all degree and
coefficient hypotheses, with two nonzero lower coefficients. Auxiliary
example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then t else
    if i = 2 then t ^ 2 else 0
    let b : Fin 3 → ℝ → ℝ := fun i t =>
      (taylor t (parameterPolynomial 3 a t)).coeff i.val
    (∀ t, (parameterPolynomial 3 a t).natDegree = 3) ∧
      (∀ t i, (parameterPolynomial 3 a t).coeff i.val = a i t) ∧
      ∀ t y, polynomialFamily 3 a (t, y + t) = polynomialFamily 3 b (t, y) := by
  exact ⟨fun t => parameterPolynomial_natDegree _ t,
    fun t i => parameterPolynomial_coeff _ t i,
    polynomialFamily_translation _ id⟩

end Transformer.Normalization
