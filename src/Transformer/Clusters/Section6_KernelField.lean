/-
# The emergence of clusters in self-attention dynamics — the attention
  kernel along a curve of measures

Along a curve `t ↦ μ_t` the velocity field of `e:conteq` is
`w(t, x) = 𝒳[μ_t](x)`.  When the curve is weakly continuous and its measures
are carried near each time by a fixed ball, `w` and its spatial derivative
are continuous in `(t, x)` (`continuous_attentionKernel_curve`,
`continuous_attentionKernelDeriv_curve`): each is assembled from integrals
`∫ f(x, y) dμ_t(y)` of jointly continuous `f`
(`SmoothApprox.continuous_integral_param`), and the normalization
`∫ e^{⟨Qx, Ky⟩} dμ_t` is positive.  When all the measures are carried by one
ball `B̄(0, R)`, `w` is moreover `2 ‖QᵀK‖ ‖V‖ R²`-Lipschitz in `x` by
`e:lipinx` and bounded by `‖V‖ R` by `e:bddinx`: it is a
`GlobalFlow.IsFlowField` (`isFlowField_attentionKernel`), whose flow
transports the solutions of `e:conteq`.  With `e:lipinmu`, `𝒳[μ](x)` is
jointly Lipschitz in `x` and in `μ` for the distance `W₂`
(`exists_norm_attentionKernel_sub_le`), the estimate behind both the
existence and the stability of solutions.

Source: arXiv:2305.05465v6, `lem: vectorfield.properties`, as used in the
proof of `c:wellposedtransformers`.
-/

import Transformer.Clusters.Section6_KernelDeriv
import Transformer.Clusters.Section6_KernelLip
import Transformer.GlobalFlow.Flow
import Transformer.SmoothApprox

open Set Metric Filter Topology MeasureTheory
open scoped NNReal

namespace Transformer
namespace Clusters

open GlobalFlow SmoothApprox

variable {d : ℕ}

/-- **`𝒳[μ]` is Lipschitz** with the constant `2 ‖QᵀK‖ ‖V‖ R²` of
`e:lipinx`, by the mean value theorem. -/
theorem lipschitzWith_attentionKernel (Q K V : ParamMatrix d) {R : ℝ} (hR : 0 < R)
    (μ : Measure (EucSpace d)) [IsProbabilityMeasure μ] (hsupp : IsCarriedBy μ R) :
    LipschitzWith (2 * ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖V‖ * R ^ 2).toNNReal
      (attentionKernel Q K V μ) := by
  refine lipschitzWith_of_nnnorm_fderiv_le
    (fun x => (hasFDerivAt_attentionKernel Q K V R hR μ inferInstance hsupp x).differentiableAt)
    fun x => ?_
  obtain ⟨D, hD, hDle⟩ := attentionKernel_hasFDerivAt Q K V R hR μ inferInstance hsupp x
  rw [hD.fderiv, ← NNReal.coe_le_coe, coe_nnnorm, Real.coe_toNNReal _ (by positivity)]
  exact hDle

/-- **`𝒳` is jointly Lipschitz in the point and the measure**: for probability
measures carried by `B̄(0, R)` and `x'` in that ball,
`‖𝒳[μ](x) - 𝒳[ν](x')‖ ≤ 2 ‖QᵀK‖ ‖V‖ R² ‖x - x'‖ + C W₂(μ, ν)`, with `C` the
constant of `e:lipinmu`. -/
theorem exists_norm_attentionKernel_sub_le (Q K V : ParamMatrix d) {R : ℝ} (hR : 0 < R) :
    ∃ C : ℝ, 0 < C ∧ ∀ μ ν : Measure (EucSpace d), IsProbabilityMeasure μ →
      IsProbabilityMeasure ν → IsCarriedBy μ R → IsCarriedBy ν R →
        ∀ x, ∀ x' ∈ closedBall (0 : EucSpace d) R,
          ‖attentionKernel Q K V μ x - attentionKernel Q K V ν x'‖ ≤
            2 * ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖V‖ * R ^ 2 * ‖x - x'‖ +
              C * Wasserstein.W2 μ ν := by
  obtain ⟨C, hC, hCμ⟩ := attentionKernel_lipschitz_in_measure Q K V R hR
  refine ⟨C, hC, fun μ ν hμ hν hμR hνR x x' hx' => ?_⟩
  have hL := (lipschitzWith_attentionKernel Q K V hR μ hμR).dist_le_mul x x'
  rw [dist_eq_norm, dist_eq_norm, Real.coe_toNNReal _ (by positivity)] at hL
  exact (norm_sub_le_norm_sub_add_norm_sub _ (attentionKernel Q K V μ x') _).trans
    (add_le_add hL (hCμ μ ν hμ hν hμR hνR x' hx'))

/-- The normalization `∫ e^{⟨Qx, Ky⟩} dμ(y)` of `e:vectorfield` is positive. -/
theorem integral_exp_scoreDual_pos (Q K : ParamMatrix d) {R : ℝ} (μ : Measure (EucSpace d))
    [IsProbabilityMeasure μ] (hsupp : IsCarriedBy μ R) (x : EucSpace d) :
    0 < ∫ y, Real.exp (scoreDual Q K y x) ∂μ := by
  simp only [scoreDual_apply]
  exact integral_exp_pos (integrable_attentionWeight Q K R μ hsupp x)

variable {μ : ℝ → Measure (EucSpace d)} [∀ t, IsProbabilityMeasure (μ t)]

/-- `(t, x) ↦ ∫ f(x, y) dμ_t(y)` is continuous for jointly continuous `f`. -/
theorem continuous_integral_curve {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    [FiniteDimensional ℝ F] (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    {f : EucSpace d → EucSpace d → F} (hf : Continuous fun p : EucSpace d × EucSpace d => f p.1 p.2) :
    Continuous fun p : ℝ × EucSpace d => ∫ y, f p.2 y ∂μ p.1 :=
  continuous_integral_param hR hμ continuous_fst (f := fun p y => f p.2 y)
    (hf.comp ((continuous_snd.comp continuous_fst).prodMk continuous_snd))

/-- **The attention kernel is continuous along a curve**: `(t, x) ↦ 𝒳[μ_t](x)`
is continuous when `t ↦ μ_t` is weakly continuous and carried near each time
by a fixed ball. -/
theorem continuous_attentionKernel_curve (Q K V : ParamMatrix d)
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t) :
    Continuous fun p : ℝ × EucSpace d => attentionKernel Q K V (μ p.1) p.2 := by
  have hL := continuous_scoreDual Q K
  have he : Continuous fun p : EucSpace d × EucSpace d => Real.exp (scoreDual Q K p.2 p.1) :=
    Real.continuous_exp.comp ((hL.comp continuous_snd).clm_apply continuous_fst)
  have hZ := continuous_integral_curve (f := fun x y => Real.exp (scoreDual Q K y x)) hR hμ he
  have hN := continuous_integral_curve (f := fun x y => Real.exp (scoreDual Q K y x) • V y) hR hμ
    (he.smul (V.continuous.comp continuous_snd))
  have h0 : ∀ p : ℝ × EucSpace d, ∫ y, Real.exp (scoreDual Q K y p.2) ∂μ p.1 ≠ 0 := fun p => by
    obtain ⟨R, hRt⟩ := hR p.1
    exact (integral_exp_scoreDual_pos Q K (μ p.1) hRt.self_of_nhds p.2).ne'
  have e : (fun p : ℝ × EucSpace d => attentionKernel Q K V (μ p.1) p.2) = fun p =>
      (∫ y, Real.exp (scoreDual Q K y p.2) ∂μ p.1)⁻¹ •
        ∫ y, Real.exp (scoreDual Q K y p.2) • V y ∂μ p.1 := by
    funext p; simp only [attentionKernel, scoreDual_apply]
  rw [e]
  exact (hZ.inv₀ h0).smul hN

/-- **The derivative of the attention kernel is continuous along a curve**:
`(t, x) ↦ ∇_x 𝒳[μ_t](x)` is continuous under the hypotheses of
`continuous_attentionKernel_curve`. -/
theorem continuous_attentionKernelDeriv_curve (Q K V : ParamMatrix d)
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t) :
    Continuous fun p : ℝ × EucSpace d => attentionKernelDeriv Q K V (μ p.1) p.2 := by
  have hL := continuous_scoreDual Q K
  have he : Continuous fun p : EucSpace d × EucSpace d => Real.exp (scoreDual Q K p.2 p.1) :=
    Real.continuous_exp.comp ((hL.comp continuous_snd).clm_apply continuous_fst)
  have hZ := continuous_integral_curve (f := fun x y => Real.exp (scoreDual Q K y x)) hR hμ he
  have hZ' := continuous_integral_curve (f := fun x y => Real.exp (scoreDual Q K y x) • scoreDual Q K y)
    hR hμ (he.smul (hL.comp continuous_snd))
  have hN := continuous_integral_curve (f := fun x y => Real.exp (scoreDual Q K y x) • V y) hR hμ
    (he.smul (V.continuous.comp continuous_snd))
  have hN' := continuous_integral_curve
    (f := fun x y => (Real.exp (scoreDual Q K y x) • scoreDual Q K y).smulRight (V y)) hR hμ
    (isBoundedBilinearMap_smulRight.continuous.comp
      ((he.smul (hL.comp continuous_snd)).prodMk (V.continuous.comp continuous_snd)))
  have h0 : ∀ p : ℝ × EucSpace d, ∫ y, Real.exp (scoreDual Q K y p.2) ∂μ p.1 ≠ 0 := fun p => by
    obtain ⟨R, hRt⟩ := hR p.1
    exact (integral_exp_scoreDual_pos Q K (μ p.1) hRt.self_of_nhds p.2).ne'
  exact ((hZ.inv₀ h0).smul hN').add (isBoundedBilinearMap_smulRight.continuous.comp
    ((((hZ.pow 2).inv₀ fun p => pow_ne_zero 2 (h0 p)).neg.smul hZ').prodMk hN))

/-- **The attention kernel along a curve carried by one ball is a flow
field**: continuous in `(t, x)`, `2 ‖QᵀK‖ ‖V‖ R²`-Lipschitz in `x`
(`e:lipinx`) and bounded by `‖V‖ R` (`e:bddinx`). -/
theorem isFlowField_attentionKernel (Q K V : ParamMatrix d) {R : ℝ} (hR : 0 < R)
    (hsupp : ∀ t, IsCarriedBy (μ t) R)
    (hμ : ∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t) :
    IsFlowField (fun t x => attentionKernel Q K V (μ t) x)
      (2 * ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖V‖ * R ^ 2).toNNReal (‖V‖ * R) where
  continuous := continuous_attentionKernel_curve Q K V (fun _ => ⟨R, Eventually.of_forall hsupp⟩) hμ
  lipschitz t := lipschitzWith_attentionKernel Q K V hR (μ t) (hsupp t)
  bound t x := attentionKernel_norm_le Q K V R hR.le (μ t) inferInstance (hsupp t) x

/-- The hypotheses of the results above are satisfiable: the constant curve
at the Dirac mass at the origin is carried by the unit ball and weakly
continuous, and the first projection of `ℝ^d × ℝ^d` is continuous. -/
example : (0 : ℝ) < 1 ∧ (∀ _ : ℝ, IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1) ∧
    (∀ t₀ : ℝ, ∃ R : ℝ, ∀ᶠ _ in 𝓝 t₀, Measure.dirac (0 : EucSpace d) (closedBall 0 R)ᶜ = 0) ∧
    (∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun _ : ℝ => ∫ x, g x ∂Measure.dirac (0 : EucSpace d)) ∧
    Continuous fun p : EucSpace d × EucSpace d => p.1 :=
  ⟨one_pos, fun _ => isCarriedBy_dirac 1 zero_le_one,
    fun _ => ⟨1, Eventually.of_forall fun _ => isCarriedBy_dirac 1 zero_le_one⟩,
    fun _ _ _ => continuous_const, continuous_fst⟩

end Clusters
end Transformer
