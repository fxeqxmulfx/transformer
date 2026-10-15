/-
# Actual centroid dependence of the empirical-mean noise

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the original mean-noise bracket calculation. The genuine differential
of the empirical mean is twice the pairing of the two actual centroids.
Thus its actual noise is a single-token average of scalar Gaussian
value terms, rather than an arbitrary sum of unrelated pair noises.

Finite Cauchy–Schwarz and the actual projected Gaussian value moments
give a noise-density bound by the original mean times the average
of the actual squared barycenters. At inverse temperature zero the
barycenters are exactly the centroid, giving

  `E(mean noise)² ≤ 4 σV² m(X)²`.

The bound retains the actual small-mean dependence lost by a uniform
`O(1/d)` estimate. Every integrability and covariance step uses the
original Gaussian head law and actual original query averages.
-/

import Transformer.Homogenized.Section7_MeanOverlapNoise

open Real MeasureTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Pairing with the actual centroid is the original single-token average.
Source: arXiv:2604.01978v1, §7, `eq:def_m_smallbeta` and its mean differential. -/
theorem inner_simplexCentroid_average {d n : ℕ} (u : EucSpace d) (V : Idx n → EucSpace d) :
    inner ℝ u (simplexCentroid V) = (n : ℝ)⁻¹ * ∑ i : Idx n, inner ℝ u (V i) := by
  rw [simplexCentroid, real_inner_smul_right, inner_sum]

/-- The genuine mean differential pairs the two actual centroids.
Source: arXiv:2604.01978v1, §7, the mean observable in `eq:m_decomp_clean`. -/
theorem meanOverlapForm_eq_centroids {d n : ℕ} (X V : Idx n → EucSpace d) :
    meanOverlapForm X V = 2 * inner ℝ (simplexCentroid X) (simplexCentroid V) := by
  have hswap : (∑ i : Idx n, ∑ j : Idx n, inner ℝ (V i) (X j)) =
      ∑ i : Idx n, ∑ j : Idx n, inner ℝ (X i) (V j) := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => real_inner_comm _ _
  simp only [meanOverlapForm_apply, overlapForm_apply, simplexCentroid,
    real_inner_smul_left, real_inner_smul_right, sum_inner, inner_sum,
    Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hswap, ← inv_pow]
  conv_rhs => rw [Finset.sum_comm]
  ring

/-- The original mean derivative is a single-token scalar average.
Source: arXiv:2604.01978v1, §7, the noise integrand after `eq:m_decomp_clean`. -/
theorem meanOverlapForm_eq_centroid_sum {d n : ℕ} (X V : Idx n → EucSpace d) :
    meanOverlapForm X V = 2 * (n : ℝ)⁻¹ * ∑ i : Idx n, inner ℝ (simplexCentroid X) (V i) := by
  rw [meanOverlapForm_eq_centroids, inner_simplexCentroid_average]
  ring

/-- The actual mean-noise scalar is the original single-token projected sum.
Source: arXiv:2604.01978v1, §7, the original Gaussian bracket calculation. -/
theorem meanOverlapNoiseIntegrand_eq_centroid_sum {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (θ : HeadParam d) :
    meanOverlapNoiseIntegrand β ρ X θ = 2 * (n : ℝ)⁻¹ *
      ∑ i : Idx n, inner ℝ (simplexCentroid X) (Gfield β ρ X θ i) := by
  rw [meanOverlapNoiseIntegrand, meanOverlapForm_eq_centroid_sum]
  simp only [noiseField_one_one]

/-- The square of an actual nonempty token average is bounded by its
average square. Source: arXiv:2604.01978v1, §7, the mean-noise bracket estimate. -/
theorem sq_empirical_average_le {n : ℕ} (hn : 0 < n) (F : Idx n → ℝ) :
    ((n : ℝ)⁻¹ * ∑ i : Idx n, F i) ^ 2 ≤ (n : ℝ)⁻¹ * ∑ i : Idx n, F i ^ 2 := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hs : (∑ i : Idx n, F i) ^ 2 ≤ (n : ℝ) * ∑ i : Idx n, F i ^ 2 := by
    simpa only [Finset.card_univ, Fintype.card_fin] using
      (sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := F))
  calc
    _ = ((n : ℝ)⁻¹) ^ 2 * (∑ i : Idx n, F i) ^ 2 := by ring
    _ ≤ ((n : ℝ)⁻¹) ^ 2 * ((n : ℝ) * ∑ i : Idx n, F i ^ 2) :=
      mul_le_mul_of_nonneg_left hs (sq_nonneg _)
    _ = (n : ℝ)⁻¹ * (((n : ℝ)⁻¹ * n) * ∑ i : Idx n, F i ^ 2) := by ring
    _ = _ := by rw [inv_mul_cancel₀ hnR, one_mul]

/-- The actual mean-noise density is bounded by genuine projected
centroid norms and actual squared barycenter correlations.
Source: arXiv:2604.01978v1, §7, the original Gaussian mean bracket calculation. -/
theorem integral_meanOverlapNoise_sq_le_projected_centroid {d n : ℕ} (hn : 0 < n) (β : ℝ)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    (∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ) ≤
      4 * (σV : ℝ) ^ 2 * (n : ℝ)⁻¹ * ∑ i : Idx n,
        ‖proj d (X i) (simplexCentroid X)‖ ^ 2 * baryCorr β ρ (empMeasure X) (X i) (X i) := by
  have hi (i : Idx n) : Integrable (fun θ => inner ℝ (simplexCentroid X) (Gfield β ρ X θ i) ^ 2) ρ := by
    simpa only [pow_two] using
      integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX i i (simplexCentroid X) (simplexCentroid X)
  have he (i : Idx n) : (∫ θ, inner ℝ (simplexCentroid X) (Gfield β ρ X θ i) ^ 2 ∂ρ) =
      (σV : ℝ) ^ 2 * ‖proj d (X i) (simplexCentroid X)‖ ^ 2 *
        baryCorr β ρ (empMeasure X) (X i) (X i) := by
    simpa only [pow_two, real_inner_self_eq_norm_sq] using
      integral_inner_Gfield_scalar_cross_gaussian β hρ X hX i i (simplexCentroid X) (simplexCentroid X)
  calc
    _ ≤ ∫ θ, (4 * (n : ℝ)⁻¹) * ∑ i : Idx n,
        inner ℝ (simplexCentroid X) (Gfield β ρ X θ i) ^ 2 ∂ρ := by
      apply integral_mono (integrable_meanOverlapNoiseIntegrand_sq_gaussian hn β hρ X hX)
        ((integrable_finsetSum _ fun i hi' => hi i).const_mul (4 * (n : ℝ)⁻¹))
      intro θ
      dsimp only
      rw [meanOverlapNoiseIntegrand_eq_centroid_sum]
      have h := sq_empirical_average_le hn (fun i => inner ℝ (simplexCentroid X) (Gfield β ρ X θ i))
      nlinarith
    _ = _ := by
      rw [integral_const_mul, integral_finsetSum _ (fun i hi' => hi i)]
      simp_rw [he, mul_assoc]
      rw [← Finset.mul_sum]
      ring

/-- **The actual Gaussian mean-noise bound retains its original mean factor.**
Source: arXiv:2604.01978v1, §7, the bracket argument in
`thm:clustering_small_beta`, refined using the original centroid differential. -/
theorem integral_meanOverlapNoise_sq_le_mean_barycenters {d n : ℕ} (hn : 0 < n) (β : ℝ)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    (∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ) ≤ 4 * (σV : ℝ) ^ 2 * meanOverlap X *
      ((n : ℝ)⁻¹ * ∑ i : Idx n, baryCorr β ρ (empMeasure X) (X i) (X i)) := by
  have hp (i : Idx n) : ‖proj d (X i) (simplexCentroid X)‖ ^ 2 ≤ meanOverlap X := by
    rw [norm_proj_sq (hX i), ← meanOverlap_eq_norm_simplexCentroid_sq]
    nlinarith [sq_nonneg (inner ℝ (X i) (simplexCentroid X))]
  have hs (i : Idx n) : 0 ≤ baryCorr β ρ (empMeasure X) (X i) (X i) := by
    rw [baryCorr_self]
    exact integral_nonneg fun θ => sq_nonneg _
  calc
    _ ≤ _ := integral_meanOverlapNoise_sq_le_projected_centroid hn β hρ X hX
    _ ≤ 4 * (σV : ℝ) ^ 2 * (n : ℝ)⁻¹ * ∑ i : Idx n,
        meanOverlap X * baryCorr β ρ (empMeasure X) (X i) (X i) :=
      mul_le_mul_of_nonneg_left
        (Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_right (hp i) (hs i)) (by positivity)
    _ = _ := by rw [← Finset.mul_sum]; ring

/-- At actual inverse temperature zero the Gaussian mean-noise density
vanishes quadratically with the actual mean, uniformly in token count.
Source: arXiv:2604.01978v1, §7, part (a) and the mean-noise bracket calculation. -/
theorem integral_meanOverlapNoise_zero_sq_le_mean_sq {d n : ℕ} (hn : 0 < n)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    (∫ θ, meanOverlapNoiseIntegrand 0 ρ X θ ^ 2 ∂ρ) ≤ 4 * (σV : ℝ) ^ 2 * meanOverlap X ^ 2 := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have h := integral_meanOverlapNoise_sq_le_mean_barycenters hn 0 hρ X hX
  simp_rw [baryCorr_zero_temperature_empMeasure] at h
  have hc : (n : ℝ)⁻¹ * ∑ _i : Idx n, meanOverlap X = meanOverlap X := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [← mul_assoc, inv_mul_cancel₀ (by exact_mod_cast hn.ne' : (n : ℝ) ≠ 0), one_mul]
  rw [hc] at h
  convert h using 1
  ring

/-- Nonempty original orthogonal unit queries, actual Gaussian value variance
and a positive actual mean jointly satisfy all refined-noise hypotheses. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℕ) < 2 ∧ (∀ i, ‖X i‖ = 1) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧ meanOverlap X = 1 / 2 := by
  let X : Idx 2 → EucSpace 2 := fun i => EuclideanSpace.single i 1
  refine ⟨X, by decide, fun i => by simp [X], isGaussianHeadLaw_gaussHeadLaw 2 _ _, ?_⟩
  norm_num [meanOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]

end Transformer.Homogenized
