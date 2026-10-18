/-
# Local uniformization: prepared polynomials on `ℝⁿ`

Monic polynomials `y^d + Σ_i a_i(x) y^i` with coefficients on `ℝⁿ`. The
Tschirnhausen translation `y ↦ y - a_{d-1}/d` removes the coefficient of `y^{d-1}`
analytically, and at a point `(x, y₀)` the slice `y ↦ V · P(x, y₀ + y)` of a unit
times the polynomial has analytic order the multiplicity of the root `y₀`, which is
less than `d` when the polynomial is centered and not `y^d`. Auxiliary for local
uniformization for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.Snoc
import Transformer.Normalization.PolynomialTranslation

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base preparedPolynomial distinguishedPolynomial
  distinguishedPolynomial_eval distinguishedPolynomial_monic)
open Transformer.Normalization (parameterPolynomial parameterPolynomial_natDegree
  parameterPolynomial_coeff parameterPolynomial_taylor_coeff polynomialFamily_translation)

variable {n : ℕ}

theorem distinguishedPolynomial_natDegree (d : ℕ) (a : Fin d → Base n → ℝ) (x : Base n) :
    (distinguishedPolynomial d a x).natDegree = d :=
  parameterPolynomial_natDegree (fun i _ => a i x) 0

theorem distinguishedPolynomial_coeff (d : ℕ) (a : Fin d → Base n → ℝ) (x : Base n)
    (i : Fin d) : (distinguishedPolynomial d a x).coeff i = a i x :=
  parameterPolynomial_coeff (fun i _ => a i x) 0 i

theorem distinguishedPolynomial_nextCoeff {k : ℕ} (a : Fin (k + 1) → Base n → ℝ)
    (x : Base n) : (distinguishedPolynomial (k + 1) a x).nextCoeff = a (Fin.last k) x := by
  rw [nextCoeff_of_natDegree_pos (by rw [distinguishedPolynomial_natDegree]; omega),
    distinguishedPolynomial_natDegree, Nat.add_sub_cancel_right]
  exact parameterPolynomial_coeff (fun i _ => a i x) 0 (Fin.last k)

/-- Tschirnhausen translation on `ℝⁿ`: `y ↦ y + s x` with `s = -a_{d-1}/d` turns a monic
polynomial with analytic coefficients vanishing at `0` into one whose coefficient of
`y^{d-1}` vanishes identically. Base-`ℝⁿ` form of `analytic_polynomial_centering`;
auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_centering {d : ℕ} (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    ∃ (s : Base n → ℝ) (b : Fin d → Base n → ℝ), AnalyticAt ℝ s 0 ∧ s 0 = 0 ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      (∀ i : Fin d, (i : ℕ) + 1 = d → ∀ x, b i x = 0) ∧
      ∀ x y, preparedPolynomial d a (x, y + s x) = preparedPolynomial d b (x, y) := by
  cases d with
  | zero =>
    exact ⟨0, a, analyticAt_const, rfl, ha, ha0, fun i => i.elim0,
      fun x y => by simp [preparedPolynomial]⟩
  | succ k =>
    let s : Base n → ℝ := fun x => -a (Fin.last k) x * ((k + 1 : ℕ) : ℝ)⁻¹
    let b : Fin (k + 1) → Base n → ℝ := fun i x =>
      (taylor (s x) (distinguishedPolynomial (k + 1) a x)).coeff i
    have hs : AnalyticAt ℝ s 0 := (ha (Fin.last k)).fun_neg.fun_mul analyticAt_const
    have hs0 : s 0 = 0 := by simp only [s, ha0, neg_zero, zero_mul]
    have hform (i : Fin (k + 1)) : b i = fun x => s x ^ (k + 1 - i) * ((k + 1).choose i : ℝ) +
        ∑ j : Fin (k + 1), a j x * (s x ^ ((j : ℕ) - i) * ((j : ℕ).choose i : ℝ)) :=
      funext fun x => parameterPolynomial_taylor_coeff (fun j _ => a j x) 0 (s x) i
    refine ⟨s, b, hs, hs0, fun i => ?_, fun i => ?_, fun i hi x => ?_,
      fun x y => polynomialFamily_translation (fun j _ => a j x) (fun _ => s x) 0 y⟩
    · rw [hform]
      exact ((hs.fun_pow _).fun_mul analyticAt_const).fun_add (Finset.analyticAt_fun_sum _
        fun j _ => (ha j).fun_mul ((hs.fun_pow _).fun_mul analyticAt_const))
    · have hp0 : distinguishedPolynomial (k + 1) a 0 = X ^ (k + 1) := by
        simp [distinguishedPolynomial, ha0]
      change (taylor (s 0) (distinguishedPolynomial (k + 1) a 0)).coeff i = 0
      rw [hs0, hp0, taylor_zero, coeff_X_pow, ite_eq_right i.isLt.ne]
    · obtain rfl : i = Fin.last k := Fin.ext (by rw [Fin.val_last]; omega)
      have hsum : (∑ j : Fin (k + 1), a j x * (s x ^ ((j : ℕ) - k) * ((j : ℕ).choose k : ℝ))) =
          a (Fin.last k) x := by
        rw [Finset.sum_eq_single (Fin.last k)]
        · simp
        · intro j _ hne
          have hjl : (j : ℕ) < k := by
            have hjk : (j : ℕ) ≠ k := fun h => hne (Fin.ext (by rw [Fin.val_last]; exact h))
            omega
          rw [Nat.choose_eq_zero_of_lt hjl]
          simp
        · intro h
          exact (h (Finset.mem_univ _)).elim
      rw [show b (Fin.last k) x = _ from congrFun (hform _) x, Fin.val_last, hsum]
      simp only [Nat.add_sub_cancel_left, pow_one, Nat.choose_succ_self_right, s]
      have hd : ((k + 1 : ℕ) : ℝ) ≠ 0 := by positivity
      field_simp
      ring

theorem analyticAt_preparedPolynomial {d : ℕ} {c : Fin d → Base n → ℝ} {x₀ : Base (n + 1)}
    (hc : ∀ i, AnalyticAt ℝ (c i) (Fin.init x₀)) :
    AnalyticAt ℝ (fun x : Base (n + 1) => preparedPolynomial d c (Fin.init x, x (Fin.last n)))
      x₀ :=
  ((analyticAt_last x₀).fun_pow d).fun_add (Finset.analyticAt_fun_sum _ fun i _ =>
    ((hc i).comp (analyticAt_init x₀)).fun_mul ((analyticAt_last x₀).fun_pow _))

/-- The slice through `x₀` of a unit times a prepared polynomial has analytic order the
multiplicity of the root `x₀ last` of the polynomial at `init x₀`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticOrderAt_slice {d : ℕ} {V : Base (n + 1) → ℝ} {c : Fin d → Base n → ℝ}
    {x₀ : Base (n + 1)} (hV : AnalyticAt ℝ V x₀) (hV0 : V x₀ ≠ 0) :
    analyticOrderAt (fun y => V (Fin.snoc (Fin.init x₀) (x₀ (Fin.last n) + y)) *
        preparedPolynomial d c (Fin.init x₀, x₀ (Fin.last n) + y)) 0 =
      ((distinguishedPolynomial d c (Fin.init x₀)).rootMultiplicity (x₀ (Fin.last n)) : ℕ) := by
  set P := distinguishedPolynomial d c (Fin.init x₀)
  have hline : AnalyticAt ℝ (fun y : ℝ => (Fin.snoc (Fin.init x₀) (x₀ (Fin.last n) + y) :
      Base (n + 1))) 0 :=
    AnalyticAt.snoc analyticAt_const (analyticAt_const.fun_add analyticAt_id)
  have hx₀ : (Fin.snoc (Fin.init x₀) (x₀ (Fin.last n) + 0) : Base (n + 1)) = x₀ := by
    rw [add_zero, Fin.snoc_init_self]
  have hV' : AnalyticAt ℝ V (Fin.snoc (Fin.init x₀) (x₀ (Fin.last n) + 0)) := by rwa [hx₀]
  have h1 : AnalyticAt ℝ (fun y => V (Fin.snoc (Fin.init x₀) (x₀ (Fin.last n) + y))) 0 :=
    hV'.comp (f := fun y : ℝ => (Fin.snoc (Fin.init x₀) (x₀ (Fin.last n) + y) :
      Base (n + 1))) (x := 0) hline
  have hP : AnalyticAt ℝ (fun y => P.eval (x₀ (Fin.last n) + y)) 0 :=
    (AnalyticOnNhd.eval_polynomial P _ (Set.mem_univ _)).comp
      (f := fun y : ℝ => x₀ (Fin.last n) + y) (x := 0) (analyticAt_const.fun_add analyticAt_id)
  have hord := analyticOrderAt_mul h1 hP
  rw [h1.analyticOrderAt_eq_zero.2 (by simpa only [hx₀] using hV0), zero_add,
    Transformer.Normalization.polynomial_analytic_order_at_root P
      (distinguishedPolynomial_monic d c _).ne_zero] at hord
  simp only [P, distinguishedPolynomial_eval] at hord
  exact hord

/-- A function of `x'` times a unit times a centered prepared polynomial which at
`init x₀` is not `y^d` has monomial charts at `x₀`, given monomial charts for products
whose slices have order less than `d`. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem hasMonomialCharts_of_centered {d : ℕ}
    (ih : ∀ d' < d, ∀ (m : Base n → ℝ) (g : Base (n + 1) → ℝ) (z : Base (n + 1)),
      AnalyticAt ℝ m (Fin.init z) → ¬ m =ᶠ[𝓝 (Fin.init z)] 0 → AnalyticAt ℝ g z →
      analyticOrderAt (fun y => g (Fin.snoc (Fin.init z) (z (Fin.last n) + y))) 0 ≤ d' →
      HasMonomialCharts (fun x => m (Fin.init x) * g x) z)
    {m : Base n → ℝ} {V : Base (n + 1) → ℝ} {c : Fin d → Base n → ℝ} {x₀ : Base (n + 1)}
    (hm : AnalyticAt ℝ m (Fin.init x₀)) (hm0 : ¬ m =ᶠ[𝓝 (Fin.init x₀)] 0)
    (hV : AnalyticAt ℝ V x₀) (hV0 : V x₀ ≠ 0) (hc : ∀ i, AnalyticAt ℝ (c i) (Fin.init x₀))
    (hcenter : ∀ i : Fin d, (i : ℕ) + 1 = d → c i (Fin.init x₀) = 0)
    (hne : distinguishedPolynomial d c (Fin.init x₀) ≠ X ^ d) :
    HasMonomialCharts
      (fun x => m (Fin.init x) * (V x * preparedPolynomial d c (Fin.init x, x (Fin.last n))))
      x₀ := by
  have hlt : (distinguishedPolynomial d c (Fin.init x₀)).rootMultiplicity (x₀ (Fin.last n)) <
      d := by
    cases d with
    | zero => exact absurd (by simp [distinguishedPolynomial]) hne
    | succ k =>
      have hdeg := distinguishedPolynomial_natDegree (k + 1) c (Fin.init x₀)
      by_cases hroot : (distinguishedPolynomial (k + 1) c (Fin.init x₀)).IsRoot
          (x₀ (Fin.last n))
      · have h := (Transformer.Normalization.centered_polynomial_root_multiplicity_lt _
          (distinguishedPolynomial_monic _ c _)
          (by rw [distinguishedPolynomial_nextCoeff]; exact hcenter (Fin.last k) (by simp))
          (by rwa [hdeg]) _ hroot).2
        rwa [hdeg] at h
      · rw [rootMultiplicity_eq_zero hroot]
        omega
  refine ih _ hlt m _ x₀ hm hm0 (hV.fun_mul (analyticAt_preparedPolynomial hc)) ?_
  simp only [Fin.init_snoc, Fin.snoc_last]
  exact (analyticOrderAt_slice hV hV0).le

end Transformer.Uniformization
