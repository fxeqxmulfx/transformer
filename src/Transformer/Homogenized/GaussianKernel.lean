/-
# Homogenized Transformers — the noise kernel at Gaussian initialization

Formalization of `lem:lemma_app` of arXiv:2604.01978v1, *Homogenized
Transformers*, §5: under `eq: tformers.at.initialization` the drift vanishes
and, at the empirical token measure `μ_X`, the covariance kernel
`eq:covariance_kernel` collapses to a scalar,

  `K[μ](x,y) = (1/d) E_A⟨m_{β,A}[μ](x), m_{β,A}[μ](y)⟩ I_d`.

The computation has two halves.  The first is that the value matrix comes out
of the field: `B_θ[μ](x) = V m_{β,A}[μ](x)`, which is `attnFieldOf_eq_valueMap_softBary`
and is proved here — it is where the softmax barycenter of `Barycenter.lean`
enters the dynamics at all.  The second is `E_V[V a bᵀ Vᵀ] = σ_V²⟨a,b⟩ I_d`
together with `σ_V² = 1/d`, which is the lemma itself.
-/

import Transformer.Homogenized.Barycenter
import Transformer.Homogenized.GaussianDrift
import Transformer.Homogenized.GaussianEnsemble
import Transformer.Homogenized.GaussianCrossMoments
import Transformer.Homogenized.GaussianMatrixKernel
import Transformer.Homogenized.SimplexBary
import Transformer.Homogenized.AttnAverage

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The value matrix comes out of the field -/

/-- The value matrix is linear, so it commutes with a scalar. -/
theorem valueMap_smul {d : ℕ} (θ : HeadParam d) (c : ℝ) (y : EucSpace d) :
    valueMap θ (c • y) = c • valueMap θ y := by
  simp [valueMap]

/-- The value matrix is a continuous linear map on a finite-dimensional space,
so it comes out of a Bochner integral. -/
theorem valueMap_integral {d : ℕ} (θ : HeadParam d) (μ : Measure (EucSpace d))
    (f : EucSpace d → EucSpace d) (hf : Integrable f μ) :
    valueMap θ (∫ y, f y ∂μ) = ∫ y, valueMap θ (f y) ∂μ :=
  ((LinearMap.toContinuousLinearMap
    (Matrix.toEuclideanLin θ.1)).integral_comp_comm hf).symm

/-- **`ξ_θ[μ](x) = V m_{β,A}[μ](x)`**, the identity that lets `lem:lemma_app`
average over `V` and `A` separately: the attention field is the value matrix
applied to the softmax barycenter, the normalizer being a scalar and `V`
linear.

The source writes this for `ξ_θ` rather than `B_θ`, which is the same thing
once `b_{ρ*}[μ] ≡ 0` — the first clause of `lem:lemma_app`.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem attnFieldOf_eq_valueMap_softBary {d : ℕ} (β : ℝ) (θ : HeadParam d)
    (μ : Measure (EucSpace d)) (x : EucSpace d)
    (hint : Integrable (fun y => attnWeight β θ x y • y) μ) :
    attnFieldOf β θ μ x = valueMap θ (softBary β θ.2 μ x) := by
  simp only [attnFieldOf, softBary, softWeight_eq_attnWeight]
  rw [valueMap_smul, valueMap_integral θ μ _ hint]
  congr 1
  exact integral_congr_ae
    (Filter.Eventually.of_forall fun y => (valueMap_smul θ (attnWeight β θ x y) y).symm)

/-- The hypothesis of `attnFieldOf_eq_valueMap_softBary` is satisfiable: at a
Dirac token measure every function is integrable. -/
example {d : ℕ} (β : ℝ) (θ : HeadParam d) (x z : EucSpace d) :
    Integrable (fun y => attnWeight β θ x y • y) (Measure.dirac z) :=
  integrable_dirac enorm_lt_top

/-! ### The kernel -/

/-- The scalar mixed-moment computation in the proof of `lem:lemma_app`, at
the empirical measure of unit tokens.  The two softmax barycenters depend on
the query/key matrix but not on the Gaussian value matrix; their norms are at
most one, so `integral_gaussian_value_cross` applies.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`. -/
theorem gaussian_value_cross_empMeasure {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {x : Idx n → EucSpace d} (hx : ∀ k, ‖x k‖ = 1)
    (i j : Idx n) (u v : EucSpace d) :
    ∫ θ, inner (𝕜 := ℝ) u (valueMap θ (softBary β θ.2 (empMeasure x) (x i))) *
        inner (𝕜 := ℝ) v (valueMap θ (softBary β θ.2 (empMeasure x) (x j))) ∂ρ =
      (σV : ℝ) ^ 2 * inner (𝕜 := ℝ) u v *
        baryCorr β ρ (empMeasure x) (x i) (x j) := by
  simpa only [valueMap, baryCorr, softBary_empMeasure] using
    (integral_gaussian_value_cross hρ
      (measurable_sum_attnProb_smul β x i)
      (measurable_sum_attnProb_smul β x j)
      (fun A => norm_sum_attnProb_smul_le β A hx i)
      (fun A => norm_sum_attnProb_smul_le β A hx j) u v)

/-- The empirical mixed-moment hypotheses are satisfiable: one unit token
and a Gaussian head law at the standard value-matrix scaling. -/
example (d : ℕ) (σA : ℝ≥0) :
    IsGaussianHeadLaw (d + 1) (stdSigmaV (d + 1)) σA
      (gaussHeadLaw (d + 1) (stdSigmaV (d + 1)) σA) ∧
      ∀ _k : Idx 1, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  ⟨isGaussianHeadLaw_gaussHeadLaw _ _ _, fun _ => by simp [basePoint, PiLp.norm_single]⟩

/-- The covariance kernel at an empirical measure of unit tokens is a scalar
multiple of the identity.  This is the unprojected matrix equality underlying
the displayed formula of `lem:lemma_app`; the source sandwiches it between
the two tangent projections.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`. -/
theorem gaussian_covKernel_empMeasure {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {x : Idx n → EucSpace d} (hx : ∀ k, ‖x k‖ = 1)
    (i j : Idx n) (v : EucSpace d) :
    covKernel β ρ (empMeasure x) (x i) (x j) v =
      ((σV : ℝ) ^ 2 * baryCorr β ρ (empMeasure x) (x i) (x j)) • v := by
  simpa only [covKernel, fluctOf_empMeasure, fluct, meanField_gaussian β hρ,
    sub_zero, attnField_self_eq_valueMap, valueMap, baryCorr, softBary_empMeasure] using
    (integral_gaussian_value_matrix_cross hρ
      (measurable_sum_attnProb_smul β x i)
      (measurable_sum_attnProb_smul β x j)
      (fun A => norm_sum_attnProb_smul_le β A hx i)
      (fun A => norm_sum_attnProb_smul_le β A hx j) v)

/-- **Lemma (lem:lemma_app), at the empirical token measure of the source.**
The Gaussian drift vanishes, and the covariance kernel is the scalar
`(1/d) s_{μ_X}(x_i,x_j)` times the identity.  The projections in the source's
display are applied to this matrix when forming the noise cross-variation.

The source explicitly says `μ = μ_X`.  An earlier formalization additionally
asserted this kernel identity for an arbitrary measure, which is stronger than
the cited lemma and is not needed to formalize it.  This theorem states and
proves the source's empirical-measure claim.

Source: arXiv:2604.01978v1, `lem:lemma_app`. -/
theorem gaussian_drift_and_kernel_empMeasure {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {x : Idx n → EucSpace d} (hx : ∀ k, ‖x k‖ = 1)
    (i j : Idx n) :
    (∀ z : EucSpace d, meanFieldOf β ρ (empMeasure x) z = 0) ∧
      ∀ v : EucSpace d,
        covKernel β ρ (empMeasure x) (x i) (x j) v =
          ((1 / (d : ℝ)) * baryCorr β ρ (empMeasure x) (x i) (x j)) • v := by
  refine ⟨meanFieldOf_gaussian β hρ (empMeasure x), ?_⟩
  intro v
  rw [gaussian_covKernel_empMeasure β hρ hx i j v, hσ]

/-- The empirical lemma's hypotheses hold at the standard Gaussian scaling
and one unit token. -/
example (d : ℕ) (σA : ℝ≥0) :
    IsGaussianHeadLaw (d + 1) (stdSigmaV (d + 1)) σA
      (gaussHeadLaw (d + 1) (stdSigmaV (d + 1)) σA) ∧
      (stdSigmaV (d + 1) : ℝ) ^ 2 = 1 / ((d + 1 : ℕ) : ℝ) ∧
      ∀ _k : Idx 1, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  ⟨isGaussianHeadLaw_gaussHeadLaw _ _ _, stdSigmaV_sq _,
    fun _ => by simp [basePoint, PiLp.norm_single]⟩

end Homogenized
end Transformer
