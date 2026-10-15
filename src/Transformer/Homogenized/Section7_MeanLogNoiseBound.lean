/-
# Uniform actual noise bound for a regularized logarithmic mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:s_mu_small_beta_clean` and the original mean bracket calculation.
Write `ε = β² d² σA⁴`. The already proved actual Gaussian mean-noise
second moment is at most `4 m(m+20ε)/d` at the source's adopted
value variance. For positive `a ≥ 20ε`, division by `(m+a)²`
therefore gives an actual logarithmic noise second moment at most
`4/d`, uniformly in the current original mean and in token count.

The fixed choice `a = 20ε + 1/d` is positive in every positive
dimension, even at zero temperature or zero attention variance.
It is at most `21 max(1/d, ε)`, so its deterministic size is within
the source's claimed error scale. No stopping at a positive mean
or singular logarithmic test is used in these estimates.

The accompanying perturbation bound controls the actual algebraic
drift correction `a+a²+80ε` whenever this regularization is at most
one. All probabilistic bounds concern the genuine differential on
the original Gaussian noise field, with no variance identity assumed.
-/

import Transformer.Homogenized.Section7_MeanLogGenerator

open Real MeasureTheory Set
open scoped NNReal

namespace Transformer.Homogenized

/-- A fixed positive regularization at the original deterministic
error scale. Source: arXiv:2604.01978v1, §7, the scales in
`eq:smallbeta_main_bound`, using the proved `20ε` correlation constant. -/
noncomputable def meanLogRegularization (d : ℕ) (β : ℝ) (σA : ℝ≥0) : ℝ :=
  20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 + 1 / (d : ℝ)

/-- The genuine regularization dominates the dimension error scale.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`. -/
theorem one_div_dim_le_meanLogRegularization (d : ℕ) (β : ℝ) (σA : ℝ≥0) :
    1 / (d : ℝ) ≤ meanLogRegularization d β σA := by
  unfold meanLogRegularization
  have h : 0 ≤ 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 := by positivity
  linarith

/-- Positive dimension ensures a globally nonsingular logarithmic
test, including zero temperature. Source: arXiv:2604.01978v1,
§7, the positive-dimensional regime of `thm:clustering_small_beta`. -/
theorem meanLogRegularization_pos {d : ℕ} (hd : 0 < d) (β : ℝ) (σA : ℝ≥0) :
    0 < meanLogRegularization d β σA := by
  exact (one_div_pos.mpr (by exact_mod_cast hd)).trans_le
    (one_div_dim_le_meanLogRegularization d β σA)

/-- The actual regularization dominates the original quadratic
temperature error. Source: arXiv:2604.01978v1, §7,
the proved correlation bound for `eq:s_mu_small_beta_clean`. -/
theorem temperature_error_le_meanLogRegularization (d : ℕ) (β : ℝ) (σA : ℝ≥0) :
    20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 ≤ meanLogRegularization d β σA := by
  unfold meanLogRegularization
  have h : 0 ≤ 1 / (d : ℝ) := by positivity
  linarith

/-- The genuine regularization retains the source's full deterministic
error scale. Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`. -/
theorem meanLogRegularization_le_max (d : ℕ) (β : ℝ) (σA : ℝ≥0) :
    meanLogRegularization d β σA ≤
      21 * max (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by
  unfold meanLogRegularization
  have h1 := le_max_left (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
  have h2 := le_max_right (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
  nlinarith

/-- The original logarithmic drift perturbation is at most six
times the genuine regularization whenever that regularization is small.
Source: arXiv:2604.01978v1, §7, the original drift mismatch in
`eq:drift_mismatch_target_clean`, after shifting its actual mean. -/
theorem meanLogRegularization_perturbation_le_six (d : ℕ) (β : ℝ) (σA : ℝ≥0)
    (ha : meanLogRegularization d β σA ≤ 1) :
    meanLogRegularization d β σA + meanLogRegularization d β σA ^ 2 +
        80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 ≤ 6 * meanLogRegularization d β σA := by
  have h := temperature_error_le_meanLogRegularization d β σA
  have h0 : 0 ≤ meanLogRegularization d β σA :=
    (show 0 ≤ 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 by positivity).trans h
  nlinarith [mul_nonneg h0 (sub_nonneg.mpr ha)]

/-- The true mean factor in the original noise bound is absorbed
by the actual regularized denominator, even when the mean is zero.
Source: arXiv:2604.01978v1, §7, the mean bracket calculation,
refined using the original squared-centroid observable. -/
theorem regularized_mean_noise_ratio_le_one {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (X : Idx n → EucSpace d) (b : ℝ) (hb : b ≤ a) :
    ((meanOverlap X + a) ^ 2)⁻¹ * meanOverlap X * (meanOverlap X + b) ≤ 1 := by
  have hm := meanOverlap_nonneg X
  have hz := meanOverlap_add_pos ha X
  have hp : meanOverlap X * (meanOverlap X + b) ≤ (meanOverlap X + a) ^ 2 := by
    nlinarith [mul_nonneg hm (show 0 ≤ 2 * a - b by linarith), sq_nonneg a]
  calc
    _ = ((meanOverlap X + a) ^ 2)⁻¹ * (meanOverlap X * (meanOverlap X + b)) := by ring
    _ ≤ ((meanOverlap X + a) ^ 2)⁻¹ * (meanOverlap X + a) ^ 2 :=
      mul_le_mul_of_nonneg_left hp (by positivity)
    _ = 1 := inv_mul_cancel₀ (ne_of_gt (sq_pos_of_pos hz))

/-- **The genuine logarithmic mean-noise second moment is at most
`4/d`, with no dependence on the current mean or token count.**
Source: arXiv:2604.01978v1, §7, `eq:s_mu_small_beta_clean` and
the original mean bracket, at §2.3.3's adopted value variance. -/
theorem integral_logMeanOverlapNoise_sq_le_four_div_dim {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hreg : 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 ≤ a)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    (∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ) ≤ 4 / (d : ℝ) := by
  rw [integral_logMeanOverlapNoiseIntegrand_sq ha]
  have h := integral_meanOverlapNoise_sq_le_small_temperature_std hn β hβ hρ hσ X hX
  have hr := regularized_mean_noise_ratio_le_one ha X _ hreg
  calc
    _ ≤ ((meanOverlap X + a) ^ 2)⁻¹ * ((4 / (d : ℝ)) * meanOverlap X *
        (meanOverlap X + 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)) :=
      mul_le_mul_of_nonneg_left h (by positivity)
    _ = (4 / (d : ℝ)) * (((meanOverlap X + a) ^ 2)⁻¹ * meanOverlap X *
        (meanOverlap X + 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)) := by ring
    _ ≤ (4 / (d : ℝ)) * 1 := mul_le_mul_of_nonneg_left hr (by positivity)
    _ = _ := mul_one _

/-- The fixed source-scale regularization realizes the genuine uniform
logarithmic noise bound for every original unit configuration.
Source: arXiv:2604.01978v1, §7, the maximal mean-noise argument. -/
theorem integral_logMeanOverlapNoise_regularization_sq_le {d n : ℕ} (hd : 0 < d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    (∫ θ, logMeanOverlapNoiseIntegrand (meanLogRegularization d β σA) β ρ X θ ^ 2 ∂ρ) ≤
      4 / (d : ℝ) :=
  integral_logMeanOverlapNoise_sq_le_four_div_dim (meanLogRegularization_pos hd β σA) hn β hβ hρ hσ
    (temperature_error_le_meanLogRegularization d β σA) X hX

/-- Positive original attention variance and temperature give the
nonzero small regularization `13/16`, with nonempty original unit
queries and exactly the source's value variance. All noise-bound and
small-perturbation hypotheses therefore hold jointly. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) (1 / 2) (gaussHeadLaw 2 (stdSigmaV 2) (1 / 2)) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ (1 / 4 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    meanLogRegularization 2 (1 / 4) (1 / 2) = 13 / 16 ∧
    0 < meanLogRegularization 2 (1 / 4) (1 / 2) ∧
    20 * (1 / 4 : ℝ) ^ 2 * 2 ^ 2 * (1 / 2 : ℝ) ^ 4 ≤ meanLogRegularization 2 (1 / 4) (1 / 2) ∧
    meanLogRegularization 2 (1 / 4) (1 / 2) ≤ 1 := by
  refine ⟨fun k => EuclideanSpace.single k 1, by decide, fun k => by simp,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, by norm_num, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [meanLogRegularization]

end Transformer.Homogenized
