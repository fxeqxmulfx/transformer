/-
# Newton normalization and preparation decrease the polynomial degree

Positive-side real root accumulation survives Newton scaling. For a centered
normalized polynomial, preparation at any central root has strictly smaller
degree. Together these give the recursive step in real Newton--Puiseux lifting.
-/

import Transformer.Normalization.NewtonPolynomialScaling
import Transformer.Normalization.PositiveRamification
import Transformer.Normalization.PolynomialCentering

open Filter Set Polynomial
open scoped BigOperators

namespace Transformer.Normalization

/-- The next coefficient of a positive-degree parameter polynomial is
the specified next-to-leading coefficient. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem parameterPolynomial_nextCoeff {n : ℕ} (a : Fin (n + 1) → ℝ → ℝ) (t : ℝ) :
    (parameterPolynomial (n + 1) a t).nextCoeff = a (Fin.last n) t := by
  rw [nextCoeff_of_natDegree_pos (by rw [parameterPolynomial_natDegree]; omega),
    parameterPolynomial_natDegree, Nat.add_sub_cancel_right]
  exact parameterPolynomial_coeff a t (Fin.last n)

/-- Real roots accumulating at positive parameters remain frequent after
Newton scaling. Division by the nonzero power of the positive parameter
lifts each actual root to a root of the normalized polynomial. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem newton_normalized_roots_frequent {d p q : ℕ} (a b : Fin d → ℝ → ℝ)
    (hq : 0 < q)
    (hcoeff : ∀ᶠ s in nhds (0 : ℝ), ∀ i,
      a i (s ^ q) = s ^ (p * (d - i.val)) * b i s)
    (hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, polynomialFamily d a (t, y) = 0) :
    ∃ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ, polynomialFamily d b (s, y) = 0 := by
  have hramified : ∃ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, polynomialFamily d a (s ^ q, y) = 0 :=
    (frequently_positive_power_iff q hq (fun t => ∃ y, polynomialFamily d a (t, y) = 0)).mpr
      hroots
  have hpositive : ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), 0 < s := self_mem_nhdsWithin
  apply (hramified.and_eventually
    (((newton_polynomial_roots_iff a b hcoeff).filter_mono nhdsWithin_le_nhds).and
      hpositive)).mono
  intro s hs
  obtain ⟨y, hy⟩ := hs.1
  have hsne : s ≠ 0 := hs.2.2.ne'
  refine ⟨y / s ^ p, (hs.2.1 hsne _).mp ?_⟩
  have hcancel : s ^ p * (y / s ^ p) = y := mul_div_cancel₀ y (pow_ne_zero _ hsne)
  simpa only [hcancel, polynomialFamily] using hy

/-- At every central root of a centered analytic family with a nonzero
lower central coefficient, real preparation yields a positive degree
strictly smaller than the original degree. No root simplicity or bound
on the original degree is assumed. Auxiliary for the general degree
induction in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem centered_analytic_polynomial_degree_reduction {n : ℕ}
    (a : Fin (n + 1) → ℝ → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (hcenter : a (Fin.last n) 0 = 0) (hnonzero : ∃ i, a i 0 ≠ 0)
    (r : ℝ) (hroot : (parameterPolynomial (n + 1) a 0).IsRoot r) :
    ∃ (m : ℕ) (b : Fin m → ℝ → ℝ) (u : ℝ × ℝ → ℝ), 0 < m ∧ m < n + 1 ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      AnalyticAt ℝ u 0 ∧ u 0 ≠ 0 ∧ ∀ᶠ x in nhds (0 : ℝ × ℝ),
        polynomialFamily (n + 1) a (x.1, r + x.2) = u x * polynomialFamily m b x := by
  let P := parameterPolynomial (n + 1) a 0
  have hPcenter : P.nextCoeff = 0 := by
    rw [parameterPolynomial_nextCoeff, hcenter]
  have hPnontrivial : P ≠ X ^ P.natDegree := by
    intro heq
    obtain ⟨i, hi⟩ := hnonzero
    apply hi
    have hc := congrArg (fun Q : Polynomial ℝ => Q.coeff i.val) heq
    rw [parameterPolynomial_natDegree, parameterPolynomial_coeff, coeff_X_pow,
      ite_eq_right i.isLt.ne] at hc
    exact hc
  have hmult := centered_polynomial_root_multiplicity_lt P
    (parameterPolynomial_monic a 0) hPcenter hPnontrivial r hroot
  obtain ⟨b, u, hm, hb, hb0, hu, hu0, heq⟩ :=
    analytic_polynomial_preparation_at_root a ha r hroot
  refine ⟨P.rootMultiplicity r, b, u, hm, ?_, hb, hb0, hu, hu0, heq⟩
  simpa only [P, parameterPolynomial_natDegree] using hmult.2

/-- The cubic `y³ - t³` has the exact Newton normalization `y = t w`
with normalized polynomial `w³ - 1`. Its real root `y=t` proves the
positive-side accumulation and all scaling hypotheses simultaneously.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
    polynomialFamily 3 (fun i _ => if i = 0 then -1 else 0) (s, y) = 0 := by
  let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then -(t ^ 3) else 0
  let b : Fin 3 → ℝ → ℝ := fun i _ => if i = 0 then -1 else 0
  have hcoeff : ∀ᶠ s in nhds (0 : ℝ), ∀ i : Fin 3,
      a i (s ^ 1) = s ^ (1 * (3 - i.val)) * b i s :=
    Eventually.of_forall (fun s i => by fin_cases i <;> simp [a, b])
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  have hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, polynomialFamily 3 a (t, y) = 0 :=
    (Eventually.of_forall (fun t => ⟨t, by simp [a, polynomialFamily]⟩)).frequently
  exact newton_normalized_roots_frequent a b (by omega) hcoeff hroots

/-- The centered cubic `y³ - 3y + 2 - t` has a double central root at one
and a nonzero lower central coefficient. Its preparation satisfies the
strict-degree-decrease hypotheses simultaneously. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then 2 - t else
    if i = 1 then -3 else 0
    ∃ (m : ℕ) (b : Fin m → ℝ → ℝ) (u : ℝ × ℝ → ℝ), 0 < m ∧ m < 3 ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      AnalyticAt ℝ u 0 ∧ u 0 ≠ 0 ∧ ∀ᶠ x in nhds (0 : ℝ × ℝ),
        polynomialFamily 3 a (x.1, 1 + x.2) = u x * polynomialFamily m b x := by
  apply centered_analytic_polynomial_degree_reduction
  · intro i
    fin_cases i
    · exact analyticAt_const.fun_sub analyticAt_id
    · exact analyticAt_const
    · exact analyticAt_const
  · norm_num
  · exact ⟨0, by norm_num⟩
  · norm_num [IsRoot, parameterPolynomial_eval, polynomialFamily, Fin.sum_univ_succ]

end Transformer.Normalization
