/-
# Analytic centering of a monic polynomial family

Translation by minus the next-to-leading coefficient divided by the degree
removes that coefficient identically. All coefficient germs remain analytic
and retain their zero central values. This prevents concentration of the
entire degree at one root after Newton normalization.
-/

import Transformer.Normalization.PolynomialTranslation

open Filter Polynomial Finset
open scoped BigOperators

namespace Transformer.Normalization

/-- Analytic Tschirnhausen translation centers a monic polynomial family of
any positive degree. The coefficient of degree `n` vanishes identically,
and translation gives an exact equality for every parameter and variable.
Auxiliary for the degree induction in singular curve lifting in Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_polynomial_centering {n : ℕ} (a : Fin (n + 1) → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0) :
    ∃ (c : ℝ → ℝ) (b : Fin (n + 1) → ℝ → ℝ), AnalyticAt ℝ c 0 ∧ c 0 = 0 ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      b (Fin.last n) = (fun _ => 0) ∧
      ∀ t y : ℝ, polynomialFamily (n + 1) a (t, y + c t) =
        polynomialFamily (n + 1) b (t, y) := by
  let c : ℝ → ℝ := fun t => -a (Fin.last n) t * ((n + 1 : ℕ) : ℝ)⁻¹
  let b : Fin (n + 1) → ℝ → ℝ := fun i t =>
    (taylor (c t) (parameterPolynomial (n + 1) a t)).coeff i.val
  have hc : AnalyticAt ℝ c 0 :=
    (ha (Fin.last n)).fun_neg.fun_mul analyticAt_const
  have hc0 : c 0 = 0 := by simp only [c, ha0, neg_zero, zero_mul]
  have hb (i : Fin (n + 1)) : AnalyticAt ℝ (b i) 0 := by
    have hform : b i = fun t => c t ^ (n + 1 - i.val) * ((n + 1).choose i.val : ℝ) +
        ∑ j : Fin (n + 1), a j t * (c t ^ (j.val - i.val) * (j.val.choose i.val : ℝ)) := by
      funext t
      exact parameterPolynomial_taylor_coeff a t (c t) i.val
    rw [hform]
    apply ((hc.fun_pow _).fun_mul analyticAt_const).fun_add
    apply Finset.analyticAt_fun_sum
    intro j hj
    exact (ha j).fun_mul ((hc.fun_pow _).fun_mul analyticAt_const)
  have hb0 (i : Fin (n + 1)) : b i 0 = 0 := by
    have hp0 : parameterPolynomial (n + 1) a 0 = X ^ (n + 1) := by
      simp [parameterPolynomial, AnalyticPreparation.distinguishedPolynomial, ha0]
    change (taylor (c 0) (parameterPolynomial (n + 1) a 0)).coeff i.val = 0
    rw [hc0, hp0, taylor_zero, coeff_X_pow]
    simp only [ite_eq_right i.isLt.ne]
  have hcenter : b (Fin.last n) = fun _ => 0 := by
    funext t
    have hsum : (∑ i : Fin (n + 1), a i t * (c t ^ (i.val - n) * (i.val.choose n : ℝ))) =
        a (Fin.last n) t := by
      rw [sum_eq_single (Fin.last n)]
      · simp
      · intro i hi hne
        have hil : i.val < n := by
          have hiv : i.val ≠ n := by
            intro h
            apply hne
            exact Fin.ext (by simpa only [Fin.val_last] using h)
          omega
        rw [Nat.choose_eq_zero_of_lt hil]
        simp
      · intro h
        exact (h (mem_univ _)).elim
    change (taylor (c t) (parameterPolynomial (n + 1) a t)).coeff (Fin.last n).val = 0
    rw [parameterPolynomial_taylor_coeff, Fin.val_last, hsum]
    simp only [Nat.add_sub_cancel_left, pow_one, Nat.choose_succ_self_right]
    dsimp only [c]
    have hd : ((n + 1 : ℕ) : ℝ) ≠ 0 := by positivity
    field_simp
    ring
  exact ⟨c, b, hc, hc0, hb, hb0, hcenter, polynomialFamily_translation a c⟩

/-- The cubic family `y³ + t y² - t²` satisfies all analytic and central
coefficient conditions simultaneously. Centering removes its moving
quadratic coefficient while preserving the actual polynomial identity.
Auxiliary example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then -(t ^ 2) else
    if i = 2 then t else 0
    ∃ (c : ℝ → ℝ) (b : Fin 3 → ℝ → ℝ), AnalyticAt ℝ c 0 ∧ c 0 = 0 ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧ b 2 = (fun _ => 0) ∧
      ∀ t y : ℝ, polynomialFamily 3 a (t, y + c t) = polynomialFamily 3 b (t, y) := by
  apply analytic_polynomial_centering
  · intro i
    fin_cases i
    · exact (analyticAt_id.fun_pow 2).fun_neg
    · exact analyticAt_const
    · exact analyticAt_id
  · intro i
    fin_cases i <;> simp

end Transformer.Normalization
