module

public import Transformer.Clusters.Section6_Kernel
public import Mathlib.Analysis.Calculus.ParametricIntegral
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# The emergence of clusters in self-attention dynamics — the derivative of
  the attention kernel

`e:lipinx` of arXiv:2305.05465v6, §6: `𝒳[μ]` is differentiable in `x`, with
`‖∇_x 𝒳[μ]‖ ≤ 2‖QᵀK‖_op ‖V‖_op R²`. The derivative of both integrals of
`e:vectorfield` is taken under the integral sign
(`hasFDerivAt_integral_of_dominated_of_fderiv_le`), the weights being bounded
near `x` because `μ` is carried by `B̄(0,R)`. The derivative is written out
(`attentionKernelDeriv`), so that its continuity in `(t, x)` along a curve of
measures can be read off it (`Section6_KernelField`).

Source: arXiv:2305.05465v6, `lem: vectorfield.properties`, `e:lipinx`.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer
namespace Clusters

variable {d : ℕ}

/-- **The derivative of the attention kernel** in `x`: with the weights
`w(y) = e^{⟨Qx, Ky⟩}` and `Z = ∫ w dμ`, it is

  `h ↦ Z⁻¹ ∫ w(y) ⟨Qh, Ky⟩ V y dμ(y) - Z⁻² (∫ w(y) ⟨Qh, Ky⟩ dμ(y)) ∫ w(y) V y dμ(y)`

(`hasFDerivAt_attentionKernel`).

Source: arXiv:2305.05465v6, proof of `e:lipinx`. -/
noncomputable def attentionKernelDeriv (Q K V : ParamMatrix d) (μ : Measure (EucSpace d))
    (x : EucSpace d) : EucSpace d →L[ℝ] EucSpace d :=
  (∫ y, Real.exp (scoreDual Q K y x) ∂μ)⁻¹ •
      ∫ y, (Real.exp (scoreDual Q K y x) • scoreDual Q K y).smulRight (V y) ∂μ +
    ((-((∫ y, Real.exp (scoreDual Q K y x) ∂μ) ^ 2)⁻¹) •
      ∫ y, Real.exp (scoreDual Q K y x) • scoreDual Q K y ∂μ).smulRight
        (∫ y, Real.exp (scoreDual Q K y x) • V y ∂μ)

/-- **`𝒳[μ]` is differentiable in `x`**, with derivative
`attentionKernelDeriv`: both integrals of `e:vectorfield` are differentiated
under the integral sign. -/
theorem hasFDerivAt_attentionKernel (Q K V : ParamMatrix d) (R : ℝ) (hR : 0 < R)
    (μ : Measure (EucSpace d)) (hμ : IsProbabilityMeasure μ) (hsupp : IsCarriedBy μ R)
    (x : EucSpace d) :
    HasFDerivAt (attentionKernel Q K V μ) (attentionKernelDeriv Q K V μ x) x := by
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
  have hD : 0 < ∫ y, Real.exp (L y x) ∂μ := integral_exp_pos hint1
  rw [hK]
  exact ((hasDerivAt_inv hD.ne').comp_hasFDerivAt x h₁).smul h₂

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
  refine ⟨_, hasFDerivAt_attentionKernel Q K V R hR μ hμ hsupp x, ?_⟩
  rw [attentionKernelDeriv]
  set M := (ContinuousLinearMap.adjoint Q).comp K
  set L := scoreDual Q K
  have hint1 : Integrable (fun y => Real.exp (L y x)) μ := by
    simpa only [L, scoreDual_apply] using integrable_attentionWeight Q K R μ hsupp x
  set D := ∫ y, Real.exp (L y x) ∂μ
  have hD : 0 < D := integral_exp_pos hint1
  -- the three integrals, each against the weights `e^{⟨Qx, Ky⟩}`
  have hbx : ∀ᵐ y ∂μ, ‖L y‖ ≤ ‖M‖ * R ∧ ‖V y‖ ≤ ‖V‖ * R := by
    filter_upwards [ae_mem_closedBall hsupp] with y hy
    have hy' : ‖y‖ ≤ R := by simpa using hy
    exact ⟨(norm_scoreDual_le Q K y).trans (mul_le_mul_of_nonneg_left hy' (norm_nonneg _)),
      (V.le_opNorm y).trans (mul_le_mul_of_nonneg_left hy' (norm_nonneg _))⟩
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

/-- The hypotheses of `hasFDerivAt_attentionKernel` and
`attentionKernel_hasFDerivAt` are satisfiable: the Dirac
mass at the origin is a probability measure carried by the unit ball. -/
example : (0 : ℝ) < 1 ∧ IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
    IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 :=
  ⟨one_pos, inferInstance, isCarriedBy_dirac 1 zero_le_one⟩

end Clusters
end Transformer
