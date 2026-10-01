/-
# Division by an analytic ramified root

The monic quotient by a moving root has analytic coefficients and degree
one smaller. At the origin its central polynomial is again a pure power.
The actual factorization retains every other root of the original family.
-/

import Transformer.Normalization.PolynomialFamilyCoefficients

open Filter Polynomial Finset
open scoped BigOperators

noncomputable section

namespace Transformer.Normalization

/-- Removing an analytic ramified root from a monic family leaves an
analytic monic family of degree one smaller, with zero central lower
coefficients. The factorization holds for every polynomial variable at
every sufficiently small ramified parameter. Auxiliary for complete
root-branch coverage in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_polynomial_division_by_root {n q : ℕ}
    (a : Fin (n + 1) → ℝ → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (hq : 0 < q) (g : ℝ → ℝ)
    (hg : AnalyticAt ℝ g 0) (hg0 : g 0 = 0)
    (hroot : ∀ᶠ t in nhds (0 : ℝ), polynomialFamily (n + 1) a (t ^ q, g t) = 0) :
    ∃ b : Fin n → ℝ → ℝ, (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      ∀ᶠ t in nhds (0 : ℝ), ∀ y : ℝ,
        polynomialFamily (n + 1) a (t ^ q, y) = (y - g t) * polynomialFamily n b (t, y) := by
  let A : Fin (n + 1) → ℝ → ℝ := fun i t => a i (t ^ q)
  have hpower : AnalyticAt ℝ (fun t : ℝ => t ^ q) 0 := analyticAt_id.fun_pow q
  have hA (i : Fin (n + 1)) : AnalyticAt ℝ (A i) 0 :=
    (by simpa [hq.ne'] using ha i : AnalyticAt ℝ (a i) ((0 : ℝ) ^ q)).comp
      (f := fun t : ℝ => t ^ q) (x := 0) hpower
  have hA0 (i : Fin (n + 1)) : A i 0 = 0 := by simp only [A, zero_pow hq.ne', ha0]
  let Q : ℝ → Polynomial ℝ := fun t => parameterPolynomial (n + 1) A t /ₘ (X - C (g t))
  let b : Fin n → ℝ → ℝ := fun i t => (Q t).coeff i.val
  have hb (i : Fin n) : AnalyticAt ℝ (b i) 0 := by
    have hform : b i = fun t => ∑ j ∈ Icc (i.val + 1) (n + 1),
        g t ^ (j - (i.val + 1)) * (parameterPolynomial (n + 1) A t).coeff j := by
      funext t
      simpa only [parameterPolynomial_natDegree] using
        coeff_divByMonic_X_sub_C (parameterPolynomial (n + 1) A t) (g t) i.val
    rw [hform]
    apply Finset.analyticAt_fun_sum
    intro j hj
    exact (hg.fun_pow _).fun_mul (parameterPolynomial_coeff_analyticAt A hA j)
  have hb0 (i : Fin n) : b i 0 = 0 := by
    change (parameterPolynomial (n + 1) A 0 /ₘ (X - C (g 0))).coeff i.val = 0
    rw [hg0, C_0, sub_zero, parameterPolynomial_origin A hA0, pow_succ',
      mul_divByMonic_cancel_left _ monic_X, coeff_X_pow, ite_eq_right i.isLt.ne]
  have hQdegree (t : ℝ) : (Q t).natDegree = n := by
    change (parameterPolynomial (n + 1) A t /ₘ (X - C (g t))).natDegree = n
    rw [natDegree_divByMonic _ (monic_X_sub_C _), parameterPolynomial_natDegree,
      natDegree_X_sub_C, Nat.add_sub_cancel_right]
  have hQmonic (t : ℝ) : (Q t).Monic := by
    have hpdegree : (parameterPolynomial (n + 1) A t).degree ≠ 0 := by
      rw [degree_eq_natDegree (parameterPolynomial_monic A t).ne_zero,
        parameterPolynomial_natDegree]
      exact_mod_cast Nat.succ_ne_zero n
    change (parameterPolynomial (n + 1) A t /ₘ (X - C (g t))).leadingCoeff = 1
    rw [leadingCoeff_divByMonic_X_sub_C _ hpdegree]
    exact parameterPolynomial_monic A t
  have hQeval (t y : ℝ) : (Q t).eval y = polynomialFamily n b (t, y) := by
    have heq := congrArg (fun P : Polynomial ℝ => P.eval y)
      (monic_eq_leading_add_fin (Q t) (hQmonic t) (hQdegree t))
    simpa only [eval_add, eval_pow, eval_X, eval_finsetSum, eval_mul, eval_C,
      polynomialFamily, b] using heq
  refine ⟨b, hb, hb0, ?_⟩
  filter_upwards [hroot] with t ht
  intro y
  have htroot : (parameterPolynomial (n + 1) A t).IsRoot (g t) := by
    simpa only [IsRoot, parameterPolynomial_eval, polynomialFamily, A] using ht
  have heq := congrArg (fun P : Polynomial ℝ => P.eval y)
    (mul_divByMonic_eq_iff_isRoot.mpr htroot)
  rw [eval_mul, eval_sub, eval_X, eval_C, hQeval, parameterPolynomial_eval] at heq
  simpa only [polynomialFamily, A] using heq.symm

/-- The analytic root `g(t) = t` of `y³ - t³` satisfies every division
hypothesis, including the multiple central root. The quotient has analytic
zero-origin coefficients and retains the other roots in the factorization.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then -(t ^ 3) else 0
    ∃ b : Fin 2 → ℝ → ℝ, (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      ∀ᶠ t in nhds (0 : ℝ), ∀ y : ℝ,
        polynomialFamily 3 a (t, y) = (y - t) * polynomialFamily 2 b (t, y) := by
  let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then -(t ^ 3) else 0
  have ha (i : Fin 3) : AnalyticAt ℝ (a i) 0 := by
    dsimp only [a]
    split_ifs
    · exact (analyticAt_id.fun_pow 3).fun_neg
    · exact analyticAt_const
  have ha0 (i : Fin 3) : a i 0 = 0 := by simp [a]
  have hroot : ∀ᶠ t in nhds (0 : ℝ), polynomialFamily 3 a (t ^ 1, id t) = 0 :=
    Eventually.of_forall (fun t => by simp [polynomialFamily, a])
  have h := analytic_polynomial_division_by_root a ha ha0
    (by omega : 0 < (1 : ℕ)) id analyticAt_id rfl hroot
  simpa only [pow_one, id_eq] using h

end Transformer.Normalization
