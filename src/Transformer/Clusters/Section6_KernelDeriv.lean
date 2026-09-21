/-
# The emergence of clusters in self-attention dynamics — the derivative of
  the attention kernel

`e:lipinx` of arXiv:2305.05465v6, §6: `𝒳[μ]` is differentiable in `x`, with
`‖∇_x 𝒳[μ]‖ ≤ 2‖QᵀK‖_op ‖V‖_op R²`.  The derivative of both integrals of
`e:vectorfield` is taken under the integral sign
(`hasFDerivAt_integral_of_dominated_of_fderiv_le`), the weights being bounded
near `x` because `μ` is carried by `B̄(0,R)`.

Source: arXiv:2305.05465v6, `lem: vectorfield.properties`, `e:lipinx`.
-/

import Transformer.Clusters.Section6_Kernel
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv

open Real MeasureTheory

namespace Transformer
namespace Clusters

variable {d : ℕ}

/-- `h ↦ ⟨Qh, Ky⟩`, the derivative in `x` of the exponent of `e:vectorfield`. -/
noncomputable def scoreDual (Q K : ParamMatrix d) (y : EucSpace d) : EucSpace d →L[ℝ] ℝ :=
  (innerSL ℝ (K y)).comp Q

/-- `scoreDual Q K y` is the exponent of `e:vectorfield`, as a function of `x`. -/
theorem scoreDual_apply (Q K : ParamMatrix d) (x y : EucSpace d) :
    scoreDual Q K y x = inner (𝕜 := ℝ) (Q x) (K y) := by
  simp [scoreDual, real_inner_comm]

/-- `y ↦ ⟨Q·, Ky⟩` is continuous. -/
theorem continuous_scoreDual (Q K : ParamMatrix d) : Continuous (scoreDual Q K) := by
  unfold scoreDual; fun_prop

/-- `‖⟨Q·, Ky⟩‖ ≤ ‖QᵀK‖_op ‖y‖`. -/
theorem norm_scoreDual_le (Q K : ParamMatrix d) (y : EucSpace d) :
    ‖scoreDual Q K y‖ ≤ ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖y‖ := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun h => ?_
  rw [scoreDual_apply, ← ContinuousLinearMap.adjoint_inner_right, Real.norm_eq_abs]
  refine (abs_real_inner_le_norm _ _).trans ?_
  have := ((ContinuousLinearMap.adjoint Q).comp K).le_opNorm y
  rw [ContinuousLinearMap.comp_apply] at this
  nlinarith [norm_nonneg h]

/-- **Estimate (e:lipinx).**  `𝒳[μ]` is differentiable in `x` with
`‖∇_x 𝒳[μ]‖_{L^∞} ≤ 2 ‖QᵀK‖_op ‖V‖_op R²`.

As in the source: differentiating under the integral,
`∇_x 𝒳[μ](x) h = ⟨Σ_y⟨Qh, Ky⟩ V y⟩ - ⟨Σ_y⟨Qh, Ky⟩⟩ 𝒳[μ](x)` for the weights
`e^{⟨Qx, Ky⟩}` normalized, and each term is at most `‖QᵀK‖_op ‖V‖_op R² ‖h‖`.

Source: arXiv:2305.05465v6, `e:lipinx`. -/
theorem attentionKernel_hasFDerivAt (Q K V : ParamMatrix d) (R : ℝ) (hR : 0 < R)
    (μ : Measure (EucSpace d)) (hμ : IsProbabilityMeasure μ) (hsupp : IsCarriedBy μ R)
    (x : EucSpace d) :
    ∃ D : EucSpace d →L[ℝ] EucSpace d,
      HasFDerivAt (attentionKernel Q K V μ) D x ∧
        ‖D‖ ≤ 2 * ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖V‖ * R ^ 2 := by
  set M := (ContinuousLinearMap.adjoint Q).comp K
  set L := scoreDual Q K
  have hLc : Continuous L := continuous_scoreDual Q K
  have hK : attentionKernel Q K V μ = fun x =>
      (∫ y, Real.exp (L y x) ∂μ)⁻¹ • ∫ y, Real.exp (L y x) • V y ∂μ := by
    funext x; simp [attentionKernel, L, scoreDual_apply]
  set E0 := Real.exp (‖M‖ * R * (‖x‖ + 1))
  have hb : ∀ᵐ y ∂μ, ∀ x' ∈ Metric.ball x 1,
      Real.exp (L y x') ≤ E0 ∧ ‖L y‖ ≤ ‖M‖ * R ∧ ‖V y‖ ≤ ‖V‖ * R := by
    filter_upwards [ae_mem_closedBall hsupp] with y hy x' hx'
    have hy' : ‖y‖ ≤ R := by simpa using hy
    have hL : ‖L y‖ ≤ ‖M‖ * R :=
      (norm_scoreDual_le Q K y).trans (mul_le_mul_of_nonneg_left hy' (norm_nonneg _))
    have hx'' : ‖x'‖ ≤ ‖x‖ + 1 := by
      have h1 := mem_ball_iff_norm.1 hx'
      have h2 := norm_sub_norm_le x' x
      linarith
    refine ⟨Real.exp_le_exp.2 ?_, hL,
      (V.le_opNorm y).trans (mul_le_mul_of_nonneg_left hy' (norm_nonneg _))⟩
    calc L y x' ≤ ‖L y x'‖ := Real.le_norm_self _
      _ ≤ ‖L y‖ * ‖x'‖ := (L y).le_opNorm x'
      _ ≤ ‖M‖ * R * (‖x‖ + 1) := mul_le_mul hL hx'' (norm_nonneg _) (by positivity)
  have hc1 : ∀ x', Continuous fun y => Real.exp (L y x') := fun x' =>
    Real.continuous_exp.comp (hLc.clm_apply continuous_const)
  have hx1 : x ∈ Metric.ball x 1 := Metric.mem_ball_self one_pos
  have hint1 : Integrable (fun y => Real.exp (L y x)) μ :=
    Integrable.of_bound (hc1 x).aestronglyMeasurable E0 (by
      filter_upwards [hb] with y hy
      rw [Real.norm_of_nonneg (Real.exp_pos _).le]; exact (hy x hx1).1)
  have hint2 : Integrable (fun y => Real.exp (L y x) • V y) μ :=
    Integrable.of_bound ((hc1 x).smul V.continuous).aestronglyMeasurable (E0 * (‖V‖ * R)) (by
      filter_upwards [hb] with y hy
      obtain ⟨h1, -, h3⟩ := hy x hx1
      rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le]
      exact mul_le_mul h1 h3 (norm_nonneg _) (Real.exp_pos _).le |>.trans le_rfl)
  have h₁ := hasFDerivAt_integral_of_dominated_of_fderiv_le (μ := μ)
    (F := fun x y => Real.exp (L y x)) (F' := fun x y => Real.exp (L y x) • L y)
    (bound := fun _ => E0 * (‖M‖ * R)) (Metric.ball_mem_nhds x one_pos)
    (Filter.Eventually.of_forall fun x' => (hc1 x').aestronglyMeasurable) hint1
    ((hc1 x).smul hLc).aestronglyMeasurable
    (by
      filter_upwards [hb] with y hy x' hx'
      obtain ⟨h1, h2, -⟩ := hy x' hx'
      rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le]
      exact mul_le_mul h1 h2 (norm_nonneg _) (Real.exp_pos _).le)
    (integrable_const _)
    (Filter.Eventually.of_forall fun y x' _ => (L y).hasFDerivAt.exp)
  have h₂ := hasFDerivAt_integral_of_dominated_of_fderiv_le (μ := μ)
    (F := fun x y => Real.exp (L y x) • V y)
    (F' := fun x y => (Real.exp (L y x) • L y).smulRight (V y))
    (bound := fun _ => E0 * (‖M‖ * R) * (‖V‖ * R)) (Metric.ball_mem_nhds x one_pos)
    (Filter.Eventually.of_forall fun x' => ((hc1 x').smul V.continuous).aestronglyMeasurable)
    hint2 (by fun_prop)
    (by
      filter_upwards [hb] with y hy x' hx'
      obtain ⟨h1, h2, h3⟩ := hy x' hx'
      rw [ContinuousLinearMap.norm_smulRight_apply, norm_smul,
        Real.norm_of_nonneg (Real.exp_pos _).le]
      exact mul_le_mul (mul_le_mul h1 h2 (norm_nonneg _) (Real.exp_pos _).le) h3
        (norm_nonneg _) (by positivity))
    (integrable_const _)
    (Filter.Eventually.of_forall fun y x' _ => (L y).hasFDerivAt.exp.smul_const (V y))
  set D := ∫ y, Real.exp (L y x) ∂μ
  have hD : 0 < D := integral_exp_pos hint1
  have h := ((hasDerivAt_inv hD.ne').comp_hasFDerivAt x h₁).smul h₂
  rw [hK]
  refine ⟨_, h, ?_⟩
  -- the three integrals, each against the weights `e^{⟨Qx, Ky⟩}`
  have hbx : ∀ᵐ y ∂μ, ‖L y‖ ≤ ‖M‖ * R ∧ ‖V y‖ ≤ ‖V‖ * R := by
    filter_upwards [hb] with y hy
    exact ⟨(hy x hx1).2.1, (hy x hx1).2.2⟩
  have hD' : ‖∫ y, Real.exp (L y x) • L y ∂μ‖ ≤ ‖M‖ * R * D := by
    rw [← integral_const_mul]
    refine norm_integral_le_of_norm_le (hint1.const_mul _) ?_
    filter_upwards [hbx] with y hy
    rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le, mul_comm]
    exact mul_le_mul_of_nonneg_right hy.1 (Real.exp_pos _).le
  have hN : ‖∫ y, Real.exp (L y x) • V y ∂μ‖ ≤ ‖V‖ * R * D := by
    rw [← integral_const_mul]
    refine norm_integral_le_of_norm_le (hint1.const_mul _) ?_
    filter_upwards [hbx] with y hy
    rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le, mul_comm]
    exact mul_le_mul_of_nonneg_right hy.2 (Real.exp_pos _).le
  have hN' : ‖∫ y, (Real.exp (L y x) • L y).smulRight (V y) ∂μ‖ ≤ ‖M‖ * R * (‖V‖ * R) * D := by
    rw [← integral_const_mul]
    refine norm_integral_le_of_norm_le (hint1.const_mul _) ?_
    filter_upwards [hbx] with y hy
    rw [ContinuousLinearMap.norm_smulRight_apply, norm_smul,
      Real.norm_of_nonneg (Real.exp_pos _).le, mul_comm]
    have := mul_le_mul hy.1 hy.2 (norm_nonneg _) (by positivity)
    nlinarith [Real.exp_pos (L y x)]
  refine (norm_add_le _ _).trans ?_
  rw [ContinuousLinearMap.norm_smulRight_apply, norm_smul, norm_smul]
  change ‖D⁻¹‖ * _ + _ ≤ _
  rw [Real.norm_of_nonneg
    (inv_pos.2 hD).le, norm_neg, Real.norm_of_nonneg (inv_pos.2 (pow_pos hD 2)).le]
  have e1 : D⁻¹ * ‖∫ y, (Real.exp (L y x) • L y).smulRight (V y) ∂μ‖ ≤ ‖M‖ * R * (‖V‖ * R) := by
    rw [inv_mul_le_iff₀ hD]; linarith
  have e2 : (D ^ 2)⁻¹ * ‖∫ y, Real.exp (L y x) • L y ∂μ‖ * ‖∫ y, Real.exp (L y x) • V y ∂μ‖
      ≤ ‖M‖ * R * (‖V‖ * R) := by
    rw [mul_assoc, inv_mul_le_iff₀ (pow_pos hD 2)]
    calc _ ≤ ‖M‖ * R * D * (‖V‖ * R * D) :=
          mul_le_mul hD' hN (norm_nonneg _) (by positivity)
      _ = _ := by ring
  nlinarith

/-- The hypotheses of `attentionKernel_hasFDerivAt` are satisfiable: the Dirac
mass at the origin is a probability measure carried by the unit ball. -/
example : (0 : ℝ) < 1 ∧ IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
    IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 :=
  ⟨one_pos, inferInstance, isCarriedBy_dirac 1 zero_le_one⟩

end Clusters
end Transformer
