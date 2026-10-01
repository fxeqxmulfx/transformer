/-
# Directional Taylor expansion

arXiv:2506.12543v1, Section 3.2, equation (1).
The cubic remainder requires three continuous derivatives and a bounded
update direction as the learning rate tends to zero. These hypotheses,
implicit in the source, are explicit here. Derivatives are taken along
the actual parameter line, an intrinsic form of the gradient/Hessian terms.
-/

import Transformer.BatchSize.Section4_ErrorFunction

noncomputable section

namespace Transformer.BatchSize

/-- Restriction of the loss to the parameter update line, Section 3.2. -/
def lossLine {d : ℕ} (f : EucSpace d → ℝ) (x u : EucSpace d) (t : ℝ) : ℝ :=
  f (x + t • u)

/-- Gradient correlation per unit step, Section 3.2, equation (1). -/
def gradientCorrelation {d : ℕ} (f : EucSpace d → ℝ) (x u : EucSpace d) : ℝ :=
  deriv (lossLine f x u) 0

/-- Hessian curvature along the update direction, Section 3.2, equation (1). -/
def directionalSharpness {d : ℕ} (f : EucSpace d → ℝ) (x u : EucSpace d) : ℝ :=
  iteratedDeriv 2 (lossLine f x u) 0

/-- The first directional derivative equals the true loss differential,
Section 3.2's gradient correlation in equation (1). -/
theorem gradientCorrelation_eq_fderiv {d : ℕ} (f : EucSpace d → ℝ)
    (x u : EucSpace d) (hf : DifferentiableAt ℝ f x) :
    gradientCorrelation f x u = fderiv ℝ f x u := by
  have hline : HasDerivAt (fun t : ℝ => x + t • u) u 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const u).const_add x
  have hf' : HasFDerivAt f (fderiv ℝ f x) (x + (0 : ℝ) • u) := by
    simpa using hf.hasFDerivAt
  have h := hf'.comp_hasDerivAt 0 hline
  change deriv (fun t : ℝ => f (x + t • u)) 0 = fderiv ℝ f x u
  simpa only [Function.comp_def] using h.deriv

/-- Nonvacuity of the loss differential, Section 3.2. -/
example : DifferentiableAt ℝ (fun x : EucSpace 1 => x 0 ^ 2) 0 := by fun_prop

/-- Gradient correlation is exactly the inner product appearing in
Section 3.2, equation (1), with the genuine Euclidean gradient. -/
theorem gradientCorrelation_eq_inner_gradient {d : ℕ} (f : EucSpace d → ℝ)
    (x u : EucSpace d) (hf : DifferentiableAt ℝ f x) :
    gradientCorrelation f x u = inner (𝕜 := ℝ) (gradient f x) u := by
  rw [gradientCorrelation_eq_fderiv f x u hf]
  have h : (InnerProductSpace.toDual ℝ (EucSpace d)) (gradient f x) = fderiv ℝ f x := by
    simp [gradient]
  rw [← h, InnerProductSpace.toDual_apply_apply]

/-- Nonvacuity of gradient correlation, Section 3.2. -/
example : DifferentiableAt ℝ (fun x : EucSpace 1 => x 0 ^ 2) 0 := by fun_prop

/-- The second derivative along the update line is the actual Hessian
evaluated twice on that direction, Section 3.2, equation (1). -/
theorem directionalSharpness_eq_hessian {d : ℕ} (f : EucSpace d → ℝ)
    (x u : EucSpace d) (hf : ContDiff ℝ 2 f) :
    directionalSharpness f x u = iteratedFDeriv ℝ 2 f x (fun _ => u) := by
  let L : ℝ →L[ℝ] EucSpace d := (ContinuousLinearMap.id ℝ ℝ).smulRight u
  have hshift : ContDiff ℝ 2 (fun v => f (x + v)) :=
    hf.comp (contDiff_const.add contDiff_id)
  have hcomp := L.iteratedFDeriv_comp_right hshift 0 (i := 2) (by norm_num)
  change iteratedFDeriv ℝ 2 ((fun v => f (x + v)) ∘ L) 0 (fun _ => 1) = _
  rw [hcomp, ContinuousMultilinearMap.compContinuousLinearMap_apply]
  simp [L, iteratedFDeriv_comp_add_left]

/-- Nonvacuity of the Hessian identification, Section 3.2. -/
example : ContDiff ℝ 2 (fun x : EucSpace 1 => x 0 ^ 2) := by fun_prop

/-- The second-order loss expansion has a cubic error for a fixed
update direction, Section 3.2, equation (1). This corrects the source's
unstated smoothness and bounded-direction assumptions. -/
theorem directional_taylor_remainder {d : ℕ} (f : EucSpace d → ℝ)
    (x u : EucSpace d) (hf : ContDiff ℝ 3 f) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η ∈ Set.Icc (0 : ℝ) 1,
      |f (x + η • u) - f x - η * gradientCorrelation f x u -
        η ^ 2 / 2 * directionalSharpness f x u| ≤ C * η ^ 3 := by
  have hline : ContDiff ℝ 3 (lossLine f x u) :=
    hf.comp (contDiff_const.add (contDiff_id.smul_const u))
  obtain ⟨C, hC⟩ := exists_taylor_mean_remainder_bound
    (n := 2) (a := 0) (b := 1) (by norm_num) hline.contDiffOn
  refine ⟨max C 0, le_max_right _ _, ?_⟩
  intro η hη
  have hcoeff (n : ℕ) (hn : n ≤ 3) :
      iteratedDerivWithin n (lossLine f x u) (Set.Icc 0 1) 0 =
        iteratedDeriv n (lossLine f x u) 0 := by
    apply iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_Icc (by norm_num))
      (hline.contDiffAt.of_le (by exact_mod_cast hn))
    exact ⟨le_rfl, by norm_num⟩
  have ht : taylorWithinEval (lossLine f x u) 2 (Set.Icc 0 1) 0 η =
      f x + η * gradientCorrelation f x u + η ^ 2 / 2 * directionalSharpness f x u := by
    simp [taylorWithinEval, taylorWithin, taylorCoeffWithin, Finset.sum_range_succ,
      hcoeff 1 (by omega), hcoeff 2 (by omega), gradientCorrelation,
      directionalSharpness, lossLine]
    ring
  have hh := hC η hη
  rw [ht, Real.norm_eq_abs] at hh
  simp only [sub_zero] at hh
  have heq : f (x + η • u) - f x - η * gradientCorrelation f x u -
      η ^ 2 / 2 * directionalSharpness f x u =
      lossLine f x u η - (f x + η * gradientCorrelation f x u +
        η ^ 2 / 2 * directionalSharpness f x u) := by dsimp [lossLine]; ring
  rw [heq]
  exact hh.trans (mul_le_mul_of_nonneg_right (le_max_left C 0) (pow_nonneg hη.1 3))

/-- Nonvacuity of the cubic Taylor hypothesis, Section 3.2. -/
example : ContDiff ℝ 3 (fun x : EucSpace 1 => x 0 ^ 2) := by fun_prop

end Transformer.BatchSize
