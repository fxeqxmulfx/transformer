/-
# The gradient of the partition function

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`Z_{β,μ}(x) = ∫ e^{β ⟨x,y⟩} dμ(y)` is differentiated under the integral sign:

  `∇Z_{β,μ}(x) = β ∫ e^{β ⟨x,y⟩} y dμ(y)`.

This is the one analytic fact behind `eq: logder` and `e:XmuE` of §3.3, which
`Perspective.Section2_GradientFlow` reads off from it.  The domination is
immediate: on a ball of radius `1` around `x` the integrand's derivative is
bounded by the constant `|β| e^{|β|(‖x‖+1)}`, and `μ` is a probability
measure.

Source: arXiv:2312.10794v5, §3.3.
-/

import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap
import Transformer.Perspective.Section2_EnergyKernel

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-- The kernel `z ↦ e^{β ⟨z,y⟩}` is differentiable in its first argument, with
derivative `⟨β e^{β ⟨x,y⟩} y, ·⟩`. -/
theorem hasFDerivAt_expInner (β : ℝ) (y x : EucSpace d) :
    HasFDerivAt (fun z : EucSpace d => Real.exp (β * inner (𝕜 := ℝ) z y))
      (innerSL ℝ ((β * Real.exp (β * inner (𝕜 := ℝ) x y)) • y)) x := by
  have h0 : HasFDerivAt (fun z : EucSpace d => inner (𝕜 := ℝ) z y) (innerSL ℝ y) x := by
    have he : (fun z : EucSpace d => inner (𝕜 := ℝ) z y) = fun z => innerSL ℝ y z := by
      funext z
      simp [real_inner_comm]
    rw [he]
    exact (innerSL ℝ y).hasFDerivAt
  have h1 : HasFDerivAt (fun z : EucSpace d => β * inner (𝕜 := ℝ) z y)
      (β • innerSL ℝ y) x := h0.const_mul β
  refine (HasFDerivAt.exp h1).congr_fderiv ?_
  refine ContinuousLinearMap.ext fun v => ?_
  simp
  ring

/-- `|β ⟨z,y⟩| ≤ |β| (‖x‖+1)` for `y` on the sphere and `z` within `1` of
`x`: the bound that dominates both the kernel and its derivative. -/
theorem abs_inner_le_of_dist_le (β : ℝ) (x z : EucSpace d) (y : SSphere d)
    (hz : dist z x ≤ 1) :
    β * inner (𝕜 := ℝ) z (y : EucSpace d) ≤ |β| * (‖x‖ + 1) := by
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  have h1 : |inner (𝕜 := ℝ) z (y : EucSpace d)| ≤ ‖z‖ := by
    have := abs_real_inner_le_norm z (y : EucSpace d)
    rwa [hy, mul_one] at this
  have h2 : ‖z‖ ≤ ‖x‖ + 1 := by
    have := norm_sub_norm_le z x
    rw [← dist_eq_norm] at this
    linarith
  calc β * inner (𝕜 := ℝ) z (y : EucSpace d)
      ≤ |β * inner (𝕜 := ℝ) z (y : EucSpace d)| := le_abs_self _
    _ = |β| * |inner (𝕜 := ℝ) z (y : EucSpace d)| := abs_mul _ _
    _ ≤ |β| * (‖x‖ + 1) := by
        refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg β)
        linarith

/-- The kernel is integrable against `μ` at any `x` of the ambient space, not
only on the sphere: it is continuous and bounded. -/
theorem integrable_expInner_ambient (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) :
    Integrable (fun y : SSphere d =>
        Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)))
      (μ : Measure (SSphere d)) := by
  refine (integrable_const (Real.exp (|β| * (‖x‖ + 1)))).mono'
    ((continuous_const.mul (continuous_const.inner continuous_subtype_val)).rexp).aestronglyMeasurable
    (Filter.Eventually.of_forall fun y => ?_)
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact Real.exp_le_exp.mpr (abs_inner_le_of_dist_le d β x x y (by simp))

/-- The vector-valued integrand `y ↦ e^{β ⟨x,y⟩} y` is integrable. -/
theorem integrable_expInner_smul (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) :
    Integrable (fun y : SSphere d =>
        Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d))
      (μ : Measure (SSphere d)) := by
  refine (integrable_const (Real.exp (|β| * (‖x‖ + 1)))).mono'
    (((continuous_const.mul
        (continuous_const.inner continuous_subtype_val)).rexp.smul
      continuous_subtype_val).aestronglyMeasurable)
    (Filter.Eventually.of_forall fun y => ?_)
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  rw [norm_smul, hy, mul_one, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact Real.exp_le_exp.mpr (abs_inner_le_of_dist_le d β x x y (by simp))

/-- **Differentiation under the integral sign.**

  `∇Z_{β,μ}(x) = β ∫ e^{β ⟨x,y⟩} y dμ(y)`.

Source: arXiv:2312.10794v5, §3.3, `eq: partition.function`. -/
theorem hasGradientAt_partitionMu (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) :
    HasGradientAt (partitionMu d β μ)
      (β • ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
        ∂(μ : Measure (SSphere d))) x := by
  set m : Measure (SSphere d) := (μ : Measure (SSphere d)) with hm
  -- the derivative of the integrand, as a function of the integration variable
  set F' : EucSpace d → SSphere d → (EucSpace d →L[ℝ] ℝ) :=
    fun z y => innerSL ℝ
      ((β * Real.exp (β * inner (𝕜 := ℝ) z (y : EucSpace d))) • (y : EucSpace d)) with hF'
  have hcont : ∀ z : EucSpace d, Continuous fun y : SSphere d =>
      Real.exp (β * inner (𝕜 := ℝ) z (y : EucSpace d)) :=
    fun z => (continuous_const.mul (continuous_const.inner continuous_subtype_val)).rexp
  have hF'cont : ∀ z : EucSpace d, Continuous (F' z) := by
    intro z
    exact (innerSL ℝ).continuous.comp
      (((continuous_const.mul (hcont z)).smul continuous_subtype_val))
  have hbd : ∀ z ∈ Metric.ball x 1, ∀ y : SSphere d,
      ‖F' z y‖ ≤ |β| * Real.exp (|β| * (‖x‖ + 1)) := by
    intro z hz y
    have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
    have hdist : dist z x ≤ 1 := le_of_lt (Metric.mem_ball.mp hz)
    rw [hF', innerSL_apply_norm, norm_smul, hy, mul_one, Real.norm_eq_abs, abs_mul,
      abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul_of_nonneg_left
      (Real.exp_le_exp.mpr (abs_inner_le_of_dist_le d β x z y hdist)) (abs_nonneg β)
  have hint : HasFDerivAt (fun z : EucSpace d => ∫ y, Real.exp
        (β * inner (𝕜 := ℝ) z (y : EucSpace d)) ∂m) (∫ y, F' x y ∂m) x := by
    refine hasFDerivAt_integral_of_dominated_of_fderiv_le
      (Metric.ball_mem_nhds x one_pos)
      (Filter.Eventually.of_forall fun z => (hcont z).aestronglyMeasurable)
      (integrable_expInner_ambient d β μ x)
      (hF'cont x).aestronglyMeasurable
      (Filter.Eventually.of_forall fun y z hz => hbd z hz y)
      (integrable_const _)
      (Filter.Eventually.of_forall fun y _ _ => hasFDerivAt_expInner d β _ _)
  have hβint : Integrable (fun y : SSphere d =>
      β • (Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d))) m :=
    MeasureTheory.Integrable.fun_smul β (integrable_expInner_smul d β μ x)
  have heq : ∫ y, F' x y ∂m
      = innerSL ℝ (β • ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d))
          • (y : EucSpace d) ∂m) := by
    calc ∫ y, F' x y ∂m
        = ∫ y, innerSL ℝ (β • (Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d))
            • (y : EucSpace d))) ∂m := by
          refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
          simp only [hF', smul_smul]
      _ = innerSL ℝ (∫ y, β • (Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d))
            • (y : EucSpace d)) ∂m) :=
          ContinuousLinearMap.integral_comp_commSL (by simp) (innerSL ℝ) hβint
      _ = innerSL ℝ (β • ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d))
            • (y : EucSpace d) ∂m) := by rw [integral_smul]
  rw [hasGradientAt_iff_hasFDerivAt]
  have hdual : (InnerProductSpace.toDual ℝ (EucSpace d))
      (β • ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d) ∂m)
      = ∫ y, F' x y ∂m := by
    rw [heq]
    exact ContinuousLinearMap.ext fun v => by simp
  rw [hdual]
  exact hint

/-- The partition function is positive: the integrand is bounded below by
`e^{-|β| ‖x‖} > 0` and `μ` is a probability measure. -/
theorem partitionMu_pos (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) :
    0 < partitionMu d β μ x := by
  have hlow : ∀ y : SSphere d, Real.exp (-(|β| * (‖x‖ + 1)))
      ≤ Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) := by
    intro y
    refine Real.exp_le_exp.mpr ?_
    have h := abs_inner_le_of_dist_le d (-β) x x y (by simp)
    rw [abs_neg] at h
    have : -(β * inner (𝕜 := ℝ) x (y : EucSpace d)) ≤ |β| * (‖x‖ + 1) := by
      have e : -β * inner (𝕜 := ℝ) x (y : EucSpace d)
          = -(β * inner (𝕜 := ℝ) x (y : EucSpace d)) := by ring
      rwa [e] at h
    linarith
  have hmono : (∫ _ : SSphere d, Real.exp (-(|β| * (‖x‖ + 1)))
      ∂(μ : Measure (SSphere d))) ≤ partitionMu d β μ x :=
    integral_mono (integrable_const _) (integrable_expInner_ambient d β μ x) hlow
  have hconst : (∫ _ : SSphere d, Real.exp (-(|β| * (‖x‖ + 1)))
      ∂(μ : Measure (SSphere d))) = Real.exp (-(|β| * (‖x‖ + 1))) := by simp
  rw [hconst] at hmono
  exact lt_of_lt_of_le (Real.exp_pos _) hmono

/-- **The first variation of `𝖤_β`.**  `∇(β⁻¹ Z_{β,μ})(x) = ∫ e^{β ⟨x,y⟩} y dμ(y)`
— the factor `β` of `hasGradientAt_partitionMu` is exactly what the `β⁻¹` in
front of the energy cancels.

Source: arXiv:2312.10794v5, §3.3, `e:XmuE`. -/
theorem gradient_partitionMu (β : ℝ) (hβ : β ≠ 0) (μ : ProbSphere d) (x : EucSpace d) :
    gradient (fun z => β⁻¹ * partitionMu d β μ z) x
      = ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
          ∂(μ : Measure (SSphere d)) := by
  refine HasGradientAt.gradient ?_
  have h := hasGradientAt_partitionMu d β μ x
  rw [hasGradientAt_iff_hasFDerivAt] at h ⊢
  refine (h.const_mul β⁻¹).congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  simp [← mul_assoc, inv_mul_cancel₀ hβ]

/-- **The logarithmic derivative.**
`∇(β⁻¹ log Z_{β,μ})(x) = Z_{β,μ}(x)⁻¹ ∫ e^{β ⟨x,y⟩} y dμ(y)`.

Source: arXiv:2312.10794v5, §3.3, `eq: logder`. -/
theorem gradient_log_partitionMu (β : ℝ) (hβ : β ≠ 0) (μ : ProbSphere d)
    (x : EucSpace d) :
    gradient (fun z => β⁻¹ * Real.log (partitionMu d β μ z)) x
      = (partitionMu d β μ x)⁻¹ •
          ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
            ∂(μ : Measure (SSphere d)) := by
  refine HasGradientAt.gradient ?_
  have h := hasGradientAt_partitionMu d β μ x
  rw [hasGradientAt_iff_hasFDerivAt] at h ⊢
  have hlog := (h.log (ne_of_gt (partitionMu_pos d β μ x))).const_mul β⁻¹
  refine hlog.congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  simp [← mul_assoc]
  refine Or.inl ?_
  field_simp

end Perspective
end Transformer
