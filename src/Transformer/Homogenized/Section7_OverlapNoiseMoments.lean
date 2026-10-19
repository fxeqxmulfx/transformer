module

public import Transformer.Homogenized.Section7_SimplexDriftBias

/-!
# Actual Gaussian second moments of the overlap noise

ArXiv:2604.01978v1, §7, eq:martingale_Mij_clean and the
quadratic-variation calculation after eq:Gronwall_E_selfcontained.
The original overlap-noise integrand is the genuine derivative of
the overlap observable along the original fluctuation field.

Self-adjoint tangent projections move onto the tested token vectors.
The proved independent Gaussian value-matrix moments then compute
the actual scalar products. Averaging over the original query/key
law retains actual barycenter correlations, with the original σV²
factor before the source's standard-scaling specialization.
All integral expansions are justified by actual integrability.
-/

@[expose] public section

open MeasureTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual original scalar integrand driving the overlap martingale.
Source: arXiv:2604.01978v1, §7, eq:martingale_Mij_clean. -/
noncomputable def overlapNoiseIntegrand {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (i j : Idx n) (θ : HeadParam d) : ℝ :=
  inner (𝕜 := ℝ) (Gfield β ρ X θ i) (X j) + inner (𝕜 := ℝ) (X i) (Gfield β ρ X θ j)

/-- The integrand is the genuine derivative of the original overlap observable.
Source: arXiv:2604.01978v1, §7, eq:martingale_Mij_clean. -/
theorem overlapNoiseIntegrand_eq_fderiv_overlap {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (i j : Idx n) (θ : HeadParam d) :
    overlapNoiseIntegrand β ρ X i j θ =
      fderiv ℝ (fun Y : Idx n → EucSpace d => inner (𝕜 := ℝ) (Y i) (Y j)) X (Gfield β ρ X θ) :=
  (fderiv_overlap i j X (Gfield β ρ X θ)).symm

/-- The original tangent projection can be moved onto the test vector.
Source: arXiv:2604.01978v1, §7, the Gaussian bracket calculation after eq:Gronwall_E_selfcontained. -/
theorem inner_Gfield_eq_inner_proj_value {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (θ : HeadParam d)
    (i : Idx n) (u : EucSpace d) :
    inner (𝕜 := ℝ) u (Gfield β ρ X θ i) =
      inner (𝕜 := ℝ) (proj d (X i) u) (valueMap θ (softBary β θ.2 (empMeasure X) (X i))) := by
  have heq : Gfield β ρ X θ i =
      proj d (X i) (valueMap θ (softBary β θ.2 (empMeasure X) (X i))) := by
    rw [Gfield, fluct, meanField_gaussian β hρ X (X i), sub_zero,
      attnField_self_eq_valueMap, softBary_empMeasure]
  rw [heq]
  exact (real_inner_proj_left (X i) u _).symm

/-- Actual projected scalar noise products are integrable under (G).
Source: arXiv:2604.01978v1, §7, the bracket calculation after eq:Gronwall_E_selfcontained. -/
theorem integrable_inner_Gfield_scalar_cross_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) (u v : EucSpace d) :
    Integrable (fun θ => inner (𝕜 := ℝ) u (Gfield β ρ X θ i) *
      inner (𝕜 := ℝ) v (Gfield β ρ X θ j)) ρ := by
  simp_rw [inner_Gfield_eq_inner_proj_value β hρ]
  simpa only [valueMap, softBary_empMeasure] using
    integrable_gaussian_value_cross hρ
      (measurable_sum_attnProb_smul β X i) (measurable_sum_attnProb_smul β X j)
      (fun A => norm_sum_attnProb_smul_le β A hX i) (fun A => norm_sum_attnProb_smul_le β A hX j)
      (proj d (X i) u) (proj d (X j) v)

/-- Averaging the actual projected Gaussian value products gives the original
barycenter correlation. Source: arXiv:2604.01978v1, §7, the bracket calculation. -/
theorem integral_inner_Gfield_scalar_cross_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) (u v : EucSpace d) :
    (∫ θ, inner (𝕜 := ℝ) u (Gfield β ρ X θ i) * inner (𝕜 := ℝ) v (Gfield β ρ X θ j) ∂ρ) =
      (σV : ℝ) ^ 2 * inner (𝕜 := ℝ) (proj d (X i) u) (proj d (X j) v) *
        baryCorr β ρ (empMeasure X) (X i) (X j) := by
  simp_rw [inner_Gfield_eq_inner_proj_value β hρ]
  exact gaussian_value_cross_empMeasure β hρ hX i j _ _

/-- The actual squared overlap-noise integrand is genuinely integrable.
Source: arXiv:2604.01978v1, §7, the bracket calculation after eq:Gronwall_E_selfcontained. -/
theorem integrable_overlapNoiseIntegrand_sq_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    Integrable (fun θ => overlapNoiseIntegrand β ρ X i j θ ^ 2) ρ := by
  have hi : Integrable (fun θ => inner (𝕜 := ℝ) (X j) (Gfield β ρ X θ i) ^ 2) ρ := by
    simpa only [pow_two] using integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX i i (X j) (X j)
  have hj : Integrable (fun θ => inner (𝕜 := ℝ) (X i) (Gfield β ρ X θ j) ^ 2) ρ := by
    simpa only [pow_two] using integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX j j (X i) (X i)
  have hij := integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX i j (X j) (X i)
  have heq (θ : HeadParam d) : overlapNoiseIntegrand β ρ X i j θ ^ 2 =
      inner (𝕜 := ℝ) (X j) (Gfield β ρ X θ i) ^ 2 +
        (2 * (inner (𝕜 := ℝ) (X j) (Gfield β ρ X θ i) * inner (𝕜 := ℝ) (X i) (Gfield β ρ X θ j))) +
          inner (𝕜 := ℝ) (X i) (Gfield β ρ X θ j) ^ 2 := by
    unfold overlapNoiseIntegrand
    rw [real_inner_comm (X j) (Gfield β ρ X θ i)]
    ring
  simp_rw [heq]
  exact (hi.add (hij.const_mul 2)).add hj

/-- The actual overlap-noise second moment is exactly the original projected
Gaussian covariance formula. Source: arXiv:2604.01978v1, §7, the bracket
calculation after eq:Gronwall_E_selfcontained. The original factor σV²
is retained; the source specializes it to 1/d. -/
theorem integral_overlapNoiseIntegrand_sq_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    (∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) =
      (σV : ℝ) ^ 2 *
        (‖proj d (X i) (X j)‖ ^ 2 * baryCorr β ρ (empMeasure X) (X i) (X i) +
          ‖proj d (X j) (X i)‖ ^ 2 * baryCorr β ρ (empMeasure X) (X j) (X j) +
          2 * inner (𝕜 := ℝ) (proj d (X i) (X j)) (proj d (X j) (X i)) *
            baryCorr β ρ (empMeasure X) (X i) (X j)) := by
  let A (θ : HeadParam d) := inner (𝕜 := ℝ) (X j) (Gfield β ρ X θ i)
  let B (θ : HeadParam d) := inner (𝕜 := ℝ) (X i) (Gfield β ρ X θ j)
  have hi : Integrable (fun θ => A θ * A θ) ρ := integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX i i (X j) (X j)
  have hj : Integrable (fun θ => B θ * B θ) ρ := integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX j j (X i) (X i)
  have hij : Integrable (fun θ => A θ * B θ) ρ := integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX i j (X j) (X i)
  have heq (θ : HeadParam d) : overlapNoiseIntegrand β ρ X i j θ ^ 2 =
      A θ * A θ + 2 * (A θ * B θ) + B θ * B θ := by
    unfold overlapNoiseIntegrand
    rw [real_inner_comm (X j) (Gfield β ρ X θ i)]
    dsimp [A, B]
    ring
  simp_rw [heq]
  have hiSum : Integrable (fun θ => A θ * A θ + 2 * (A θ * B θ)) ρ := hi.add (hij.const_mul 2)
  rw [integral_add hiSum hj, integral_add hi (hij.const_mul 2), integral_const_mul]
  have hAA := integral_inner_Gfield_scalar_cross_gaussian β hρ X hX i i (X j) (X j)
  have hBB := integral_inner_Gfield_scalar_cross_gaussian β hρ X hX j j (X i) (X i)
  have hAB := integral_inner_Gfield_scalar_cross_gaussian β hρ X hX i j (X j) (X i)
  change (∫ θ, A θ * A θ ∂ρ) = _ at hAA
  change (∫ θ, B θ * B θ ∂ρ) = _ at hBB
  change (∫ θ, A θ * B θ ∂ρ) = _ at hAB
  rw [hAA, hBB, hAB, real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq]
  ring

/-- A genuine positive-value-variance head with uniform attention has
actual overlap-noise variance one at an orthonormal pair. -/
example : let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
    (∫ θ, overlapNoiseIntegrand 1 (gaussHeadLaw 2 1 0) X 0 1 θ ^ 2 ∂(gaussHeadLaw 2 1 0)) = 1 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hX : ∀ k, ‖X k‖ = 1 := fun k => by simp [X]
  have hb (k : Idx 2) : baryCorr 1 (gaussHeadLaw 2 1 0) (empMeasure X) (X k) (X k) = 1 / 2 := by
    rw [baryCorr, integral_snd_gaussHeadLaw_zero 2 1
      (fun A => inner (𝕜 := ℝ) (softBary 1 A (empMeasure X) (X k)) (softBary 1 A (empMeasure X) (X k)))]
    norm_num [softBary_empMeasure, attnProb, attnWeight, qkMap, X, Fin.sum_univ_two,
      sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right,
      EuclideanSpace.inner_single_left, PiLp.single_apply]
    rw [EuclideanSpace.real_norm_sq_eq]
    norm_num [Fin.sum_univ_two, PiLp.single_apply]
  change (∫ θ, overlapNoiseIntegrand 1 (gaussHeadLaw 2 1 0) X 0 1 θ ^ 2 ∂(gaussHeadLaw 2 1 0)) = 1
  rw [integral_overlapNoiseIntegrand_sq_gaussian 1 (isGaussianHeadLaw_gaussHeadLaw 2 1 0) X hX, hb 0, hb 1]
  norm_num [X, proj, EuclideanSpace.inner_single_left, PiLp.single_apply]

/-- The actual value samples (I,0) and (0,0) give different original noise
values, exhibiting a nonconstant scalar noise observable. -/
example : let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
    overlapNoiseIntegrand 1 (gaussHeadLaw 2 1 0) X 0 1 (1, 0) = 1 ∧
      overlapNoiseIntegrand 1 (gaussHeadLaw 2 1 0) X 0 1 (0, 0) = 0 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  change overlapNoiseIntegrand 1 (gaussHeadLaw 2 1 0) X 0 1 (1, 0) = 1 ∧
    overlapNoiseIntegrand 1 (gaussHeadLaw 2 1 0) X 0 1 (0, 0) = 0
  constructor
  · unfold overlapNoiseIntegrand
    rw [real_inner_comm (X 1) (Gfield 1 (gaussHeadLaw 2 1 0) X (1, 0) 0)]
    rw [inner_Gfield_eq_inner_proj_value 1 (isGaussianHeadLaw_gaussHeadLaw 2 1 0),
      inner_Gfield_eq_inner_proj_value 1 (isGaussianHeadLaw_gaussHeadLaw 2 1 0)]
    norm_num [softBary_empMeasure, attnProb, attnWeight, qkMap, valueMap, X, proj, Fin.sum_univ_two,
      Matrix.toLpLin_apply, sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right,
      EuclideanSpace.inner_single_left, PiLp.single_apply]
  · unfold overlapNoiseIntegrand
    rw [real_inner_comm (X 1) (Gfield 1 (gaussHeadLaw 2 1 0) X (0, 0) 0)]
    rw [inner_Gfield_eq_inner_proj_value 1 (isGaussianHeadLaw_gaussHeadLaw 2 1 0),
      inner_Gfield_eq_inner_proj_value 1 (isGaussianHeadLaw_gaussHeadLaw 2 1 0)]
    simp [valueMap]
end Transformer.Homogenized
