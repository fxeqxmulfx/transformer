/-
# Actual noise second moment for the original empirical mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the bracket calculation following `eq:gronwall_e_clean`.
The mean's noise integrand is its genuine derivative applied to
the original Gaussian diffusion field.

It is exactly the original normalized average of the pair-overlap
noise integrands. Finite Cauchy–Schwarz controls its square by the
same normalized average of their actual squares; no independence
between token pairs is assumed or needed.

All relevant original observables are measurable and their squares
are genuinely integrable under the original head law. The proved
pair-noise second moments therefore give the uniform mean-noise
second-moment bound `3/d` at the source's value variance `1/d`.
The source uses the coarser constant four. No martingale bracket
or tail estimate is introduced as an assumption.
-/

import Transformer.Homogenized.Section7_MeanOverlapGenerator
import Transformer.Homogenized.Section7_GaussianProductQueries
import Mathlib.Algebra.Order.Chebyshev

open Real MeasureTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual noise integrand for the original empirical mean.
Source: arXiv:2604.01978v1, §7, the averaged noise in `eq:m_decomp_clean`. -/
noncomputable def meanOverlapNoiseIntegrand {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (θ : HeadParam d) : ℝ :=
  meanOverlapForm X (noiseField β 1 1 ρ θ X)

/-- The original mean noise is the actual average of the original pair noises.
Source: arXiv:2604.01978v1, §7, the formula for `M̃` after `eq:F_def_clean`. -/
theorem meanOverlapNoiseIntegrand_eq_average {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (θ : HeadParam d) :
    meanOverlapNoiseIntegrand β ρ X θ = ((n : ℝ) ^ 2)⁻¹ *
      ∑ i : Idx n, ∑ j : Idx n, overlapNoiseIntegrand β ρ X i j θ := by
  simp only [meanOverlapNoiseIntegrand, meanOverlapForm_apply, noiseField_one_one,
    overlapForm_apply, overlapNoiseIntegrand]

/-- A finite original pair average has square bounded by its actual average
square. Source: arXiv:2604.01978v1, §7, the mean-noise bracket estimate. -/
theorem sq_empirical_pair_average_le {n : ℕ} (hn : 0 < n) (F : Idx n → Idx n → ℝ) :
    (((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, F i j) ^ 2 ≤
      ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, F i j ^ 2 := by
  have hnR : (n : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 (by exact_mod_cast hn.ne')
  have hs : (∑ i : Idx n, ∑ j : Idx n, F i j) ^ 2 ≤
      (n : ℝ) ^ 2 * ∑ i : Idx n, ∑ j : Idx n, F i j ^ 2 := by
    have h := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun p : Idx n × Idx n => F p.1 p.2)
    simpa only [Fintype.sum_prod_type, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
      Nat.cast_mul, pow_two] using h
  calc
    _ = (((n : ℝ) ^ 2)⁻¹) ^ 2 * (∑ i : Idx n, ∑ j : Idx n, F i j) ^ 2 := by ring
    _ ≤ (((n : ℝ) ^ 2)⁻¹) ^ 2 * ((n : ℝ) ^ 2 * ∑ i : Idx n, ∑ j : Idx n, F i j ^ 2) :=
      mul_le_mul_of_nonneg_left hs (sq_nonneg _)
    _ = ((n : ℝ) ^ 2)⁻¹ * ((((n : ℝ) ^ 2)⁻¹ * (n : ℝ) ^ 2) *
        ∑ i : Idx n, ∑ j : Idx n, F i j ^ 2) := by ring
    _ = _ := by rw [inv_mul_cancel₀ hnR, one_mul]

/-- The actual value matrix applied to the original attention average
is measurable jointly in the original value and query matrices.
Source: arXiv:2604.01978v1, §7, the original Gaussian bracket calculation. -/
theorem measurable_value_attention {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d) (i : Idx n) :
    Measurable (fun θ : HeadParam d =>
      Matrix.toEuclideanLin θ.1 (∑ k, attnProb β θ.2 X i k • X k)) := by
  have hq : Measurable (fun θ : HeadParam d => ∑ k, attnProb β θ.2 X i k • X k) :=
    (measurable_sum_attnProb_smul β X i).comp
      (measurable_snd : Measurable (fun θ : HeadParam d => θ.2))
  exact measurable_gaussianMatrix_query_application
    (fun θ : HeadParam d => θ.1) _ measurable_fst hq

/-- The actual original Gaussian pair-noise integrand is measurable.
Source: arXiv:2604.01978v1, §7, the genuine Gaussian bracket calculation. -/
theorem measurable_overlapNoiseIntegrand_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (i j : Idx n) :
    Measurable (overlapNoiseIntegrand β ρ X i j) := by
  have he (θ : HeadParam d) : overlapNoiseIntegrand β ρ X i j θ =
      inner ℝ (proj d (X i) (X j))
        (Matrix.toEuclideanLin θ.1 (∑ k, attnProb β θ.2 X i k • X k)) +
      inner ℝ (proj d (X j) (X i))
        (Matrix.toEuclideanLin θ.1 (∑ k, attnProb β θ.2 X j k • X k)) := by
    rw [overlapNoiseIntegrand, real_inner_comm (X j) (Gfield β ρ X θ i),
      inner_Gfield_eq_inner_proj_value β hρ X θ i (X j),
      inner_Gfield_eq_inner_proj_value β hρ X θ j (X i)]
    simp only [valueMap, softBary_empMeasure]
  have heq : overlapNoiseIntegrand β ρ X i j = fun θ : HeadParam d =>
      inner ℝ (proj d (X i) (X j))
        (Matrix.toEuclideanLin θ.1 (∑ k, attnProb β θ.2 X i k • X k)) +
      inner ℝ (proj d (X j) (X i))
        (Matrix.toEuclideanLin θ.1 (∑ k, attnProb β θ.2 X j k • X k)) := funext he
  rw [heq]
  exact (measurable_const.inner (measurable_value_attention β X i)).add
    (measurable_const.inner (measurable_value_attention β X j))

/-- The mean's actual noise observable is measurable under the original law.
Source: arXiv:2604.01978v1, §7, the averaged noise in `eq:m_decomp_clean`. -/
theorem measurable_meanOverlapNoiseIntegrand_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) :
    Measurable (meanOverlapNoiseIntegrand β ρ X) := by
  change Measurable (fun θ => meanOverlapNoiseIntegrand β ρ X θ)
  simp only [meanOverlapNoiseIntegrand_eq_average]
  have hpair (i j : Idx n) := measurable_overlapNoiseIntegrand_gaussian β hρ X i j
  fun_prop

/-- The actual average of the original pair-noise squares is integrable.
Source: arXiv:2604.01978v1, §7, the explicit majorant for the mean-noise square. -/
theorem integrable_pairNoise_square_average_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    Integrable (fun θ => ((n : ℝ) ^ 2)⁻¹ *
      ∑ i : Idx n, ∑ j : Idx n, overlapNoiseIntegrand β ρ X i j θ ^ 2) ρ := by
  exact (integrable_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj =>
    integrable_overlapNoiseIntegrand_sq_gaussian β hρ X hX i j).const_mul _

/-- The original mean-noise square is genuinely integrable under (G).
Source: arXiv:2604.01978v1, §7, the original mean bracket calculation. -/
theorem integrable_meanOverlapNoiseIntegrand_sq_gaussian {d n : ℕ} (hn : 0 < n) (β : ℝ)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    Integrable (fun θ => meanOverlapNoiseIntegrand β ρ X θ ^ 2) ρ := by
  have hm : Measurable (fun θ => meanOverlapNoiseIntegrand β ρ X θ ^ 2) := by
    have h := measurable_meanOverlapNoiseIntegrand_gaussian β hρ X
    fun_prop
  apply (integrable_pairNoise_square_average_gaussian β hρ X hX).mono' hm.aestronglyMeasurable
  filter_upwards with θ
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _), meanOverlapNoiseIntegrand_eq_average]
  exact sq_empirical_pair_average_le hn _

/-- **Actual mean-noise density is at most `3/d`, uniformly in token count.**
The original head law and its source-normalized positive value variance
are retained. Source: arXiv:2604.01978v1, §7, the bracket estimate
in the proof of `thm:clustering_small_beta`, improving its constant four. -/
theorem integral_meanOverlapNoiseIntegrand_sq_le_three_div_dim {d n : ℕ} (hn : 0 < n) (β : ℝ)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    (∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ) ≤ 3 / (d : ℝ) := by
  have hi (i j : Idx n) := integrable_overlapNoiseIntegrand_sq_gaussian β hρ X hX i j
  calc
    _ ≤ ∫ θ, ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ :=
      integral_mono (integrable_meanOverlapNoiseIntegrand_sq_gaussian hn β hρ X hX)
        (integrable_pairNoise_square_average_gaussian β hρ X hX) (fun θ => by
          rw [meanOverlapNoiseIntegrand_eq_average]
          exact sq_empirical_pair_average_le hn _)
    _ = ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        ∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ := by
      rw [integral_const_mul, integral_finsetSum _ (fun i hi' => integrable_finsetSum _ fun j hj => hi i j)]
      congr 1
      apply Finset.sum_congr rfl
      intro i hi'
      exact integral_finsetSum _ fun j hj => hi i j
    _ ≤ _ := (le_abs_self _).trans (abs_empirical_pair_average_le hn _ _ fun i j => by
      rw [abs_of_nonneg (integral_nonneg fun θ => sq_nonneg _)]
      exact integral_overlapNoiseIntegrand_sq_le_three_div_dim β hρ hσ X hX i j)

/-- Nonempty original unit tokens and positive source-normalized Gaussian
value variance jointly realize all actual mean-noise hypotheses. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℕ) < 2 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧ (∀ i, ‖X i‖ = 1) := by
  exact ⟨fun i => EuclideanSpace.single i 1, by decide,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, fun i => by simp⟩

end Transformer.Homogenized
