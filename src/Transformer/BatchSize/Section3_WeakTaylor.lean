/-
# A uniform Taylor estimate for weak-error analysis

arXiv:2506.12543v1, Section 3.2, equation (1), and Section 4.3, Theorem 1.
This estimate controls the expectation remainder of a single optimizer step.
It is independent of the unproved diffusion approximation.
-/

import Transformer.BatchSize.Section3_Taylor

open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The line Hessian at any parameter, Section 3.2, equation (1),
used in the weak-error argument of Section 4.3, Theorem 1. -/
theorem lossLine_second_derivative {d : ℕ} (φ : EucSpace d → ℝ)
    (x u : EucSpace d) (hf : ContDiff ℝ 2 φ) (t : ℝ) :
    iteratedDeriv 2 (lossLine φ x u) t =
      iteratedFDeriv ℝ 2 φ (x + t • u) (fun _ => u) := by
  let L : ℝ →L[ℝ] EucSpace d := (ContinuousLinearMap.id ℝ ℝ).smulRight u
  have hshift : ContDiff ℝ 2 (fun v => φ (x + v)) :=
    hf.comp (contDiff_const.add contDiff_id)
  have hcomp := L.iteratedFDeriv_comp_right hshift t (i := 2) (by norm_num)
  change iteratedFDeriv ℝ 2 ((fun v => φ (x + v)) ∘ L) t (fun _ => 1) = _
  rw [hcomp, ContinuousMultilinearMap.compContinuousLinearMap_apply]
  simp [L, iteratedFDeriv_comp_add_left]

/-- Nonvacuity of the line Hessian assumptions, Section 3.2. -/
example : ContDiff ℝ 2 (fun x : EucSpace 1 => x 0 ^ 2) := by fun_prop

/-- A bounded Hessian gives a global first-order Taylor error bounded
by M times the squared step norm. This deliberately loose constant suffices
for first-order weak consistency, Section 4.3, Theorem 1. -/
theorem uniform_first_order_taylor {d : ℕ} (φ : EucSpace d → ℝ) (M : ℝ)
    (hf : ContDiff ℝ 2 φ)
    (hH : ∀ y, ‖iteratedFDeriv ℝ 2 φ y‖ ≤ M) (x u : EucSpace d) :
    |φ (x + u) - φ x - fderiv ℝ φ x u| ≤ M * ‖u‖ ^ 2 := by
  have hline : ContDiff ℝ 2 (lossLine φ x u) :=
    hf.comp (contDiff_const.add (contDiff_id.smul_const u))
  have hcoeff (n : ℕ) (hn : n ≤ 2) (y : ℝ) (hy : y ∈ Set.Icc (0 : ℝ) 1) :
      iteratedDerivWithin n (lossLine φ x u) (Set.Icc 0 1) y =
        iteratedDeriv n (lossLine φ x u) y :=
    iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_Icc (by norm_num))
      (hline.contDiffAt.of_le (by exact_mod_cast hn)) hy
  have hb (y : ℝ) (hy : y ∈ Set.Icc (0 : ℝ) 1) :
      ‖iteratedDerivWithin 2 (lossLine φ x u) (Set.Icc 0 1) y‖ ≤ M * ‖u‖ ^ 2 := by
    rw [hcoeff 2 le_rfl y hy, lossLine_second_derivative φ x u hf y]
    have h := (iteratedFDeriv ℝ 2 φ (x + y • u)).le_opNorm (fun _ => u)
    simp only [Fin.prod_univ_two, pow_two] at h ⊢
    exact h.trans (mul_le_mul_of_nonneg_right (hH _) (mul_self_nonneg _))
  have ht : taylorWithinEval (lossLine φ x u) 1 (Set.Icc 0 1) 0 1 =
      φ x + fderiv ℝ φ x u := by
    simp [taylorWithinEval, taylorWithin, taylorCoeffWithin, Finset.sum_range_succ,
      hcoeff 1 (by omega) 0 (by norm_num), lossLine]
    exact gradientCorrelation_eq_fderiv φ x u (hf.differentiable (by norm_num) x)
  have h := taylor_mean_remainder_bound (n := 1) (a := 0) (b := 1) (x := 1)
    (by norm_num) hline.contDiffOn (by norm_num) hb
  rw [ht, Real.norm_eq_abs] at h
  simpa [lossLine, sub_add_eq_sub_sub] using h

/-- Nonvacuity of the uniform remainder estimate, Section 4.3.
A constant loss has zero Hessian and hence satisfies the bound M=1. -/
example : ContDiff ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    ∀ y : EucSpace 1, ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) y‖ ≤ 1 := by
  refine ⟨contDiff_const, ?_⟩
  intro y
  rw [iteratedFDeriv_const_of_ne (by norm_num)]
  simp

end Transformer.BatchSize
