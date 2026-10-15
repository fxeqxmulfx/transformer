/-
# Uniform small-temperature control of the actual Gaussian correlations

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:s_mu_small_beta_clean`. The actual original pair correlation
differs from the actual squared centroid by at most

  `20 β² d² σA⁴`.

The proof integrates the genuine temperature Taylor remainder.
Its actual first derivative is integrable and centered, and each
original query has exact second moment `d² σA⁴` for unit tokens.
These previously proved identities retain the original Gaussian
initialization, arbitrary separate value variance, both attention
matrix factors, all original queries and arbitrary token count.

The actual diagonal correlations consequently have upper bound
`m + 20 β² d² σA⁴`. The genuine empirical-mean noise density is
then at most `4 σV² m (m + 20 β² d² σA⁴)`. This preserves the
small-mean dependence needed beyond a uniform `O(1/d)` noise bound.
-/

import Transformer.Homogenized.Section7_TemperatureFirstOrder
import Transformer.Homogenized.Section7_MeanOverlapCentroidNoise

open Real MeasureTheory ProbabilityTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The original headwise temperature correlation is measurable
at its genuine query scores. Source: arXiv:2604.01978v1, §7,
the actual head expectation in `eq:s_mu_small_beta_clean`. -/
theorem measurable_headwise_temperature_correlation {d n : ℕ} (β : ℝ)
    (X : Idx n → EucSpace d) (i j : Idx n) :
    Measurable (fun θ : HeadParam d => temperatureBarycenterCorrelation X
      (fun k => inner ℝ (qkMap θ (X i)) (X k)) (fun k => inner ℝ (qkMap θ (X j)) (X k)) β) := by
  simp_rw [qkMap, temperatureBarycenterCorrelation_eq_inner_softBary, softBary_empMeasure]
  exact ((measurable_sum_attnProb_smul β X i).comp measurable_snd).inner (𝕜 := ℝ)
    ((measurable_sum_attnProb_smul β X j).comp measurable_snd)

/-- The actual headwise correlation is genuinely integrable under
every probability head law. Source: arXiv:2604.01978v1, §7,
bounded barycenters in the proof of `eq:s_mu_small_beta_clean`. -/
theorem integrable_headwise_temperature_correlation {d n : ℕ} (hn : 0 < n) (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    Integrable (fun θ : HeadParam d => temperatureBarycenterCorrelation X
      (fun k => inner ℝ (qkMap θ (X i)) (X k)) (fun k => inner ℝ (qkMap θ (X j)) (X k)) β) ρ := by
  apply Integrable.of_bound (measurable_headwise_temperature_correlation β X i j).aestronglyMeasurable 1
  apply ae_of_all
  intro θ
  rw [Real.norm_eq_abs, temperatureBarycenterCorrelation]
  exact (abs_real_inner_le_norm _ _).trans
    ((mul_le_mul (norm_temperatureBarycenter_le_one hn X hX _ β)
      (norm_temperatureBarycenter_le_one hn X hX _ β) (norm_nonneg _) zero_le_one).trans_eq (one_mul _))

/-- **Every original Gaussian empirical barycenter correlation has
the source's uniform quadratic temperature error.**
Source: arXiv:2604.01978v1, §7, `eq:s_mu_small_beta_clean`,
with the explicit constant twenty and the original query second moments
in place of a larger norm of the entire attention matrix. -/
theorem abs_baryCorr_sub_meanOverlap_le_small_temperature {d n : ℕ} (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    |baryCorr β ρ (empMeasure X) (X i) (X j) - meanOverlap X| ≤
      20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  let F := fun t : ℝ => fun θ : HeadParam d => temperatureBarycenterCorrelation X
    (fun k => inner ℝ (qkMap θ (X i)) (X k)) (fun k => inner ℝ (qkMap θ (X j)) (X k)) t
  let D := fun θ : HeadParam d => deriv (fun t => F t θ) 0
  have hF : Integrable (F β) ρ := integrable_headwise_temperature_correlation hn β ρ X hX i j
  have hD : Integrable D ρ ∧ (∫ θ, D θ ∂ρ) = 0 :=
    gaussian_temperatureBarycenterCorrelation_deriv_zero hn hρ X i j
  let R := fun θ : HeadParam d => F β θ - meanOverlap X - β * D θ
  have hR : Integrable R ρ := by convert (hF.sub (integrable_const _)).sub (hD.1.const_mul β) using 1
  have hi := gaussian_head_query_second_moment hρ (X i)
  have hj := gaussian_head_query_second_moment hρ (X j)
  have hB : Integrable (fun θ => 10 * β ^ 2 * (‖qkMap θ (X i)‖ ^ 2 + ‖qkMap θ (X j)‖ ^ 2)) ρ :=
    (hi.1.add hj.1).const_mul (10 * β ^ 2)
  have hpoint (θ : HeadParam d) : |R θ| ≤
      10 * β ^ 2 * (‖qkMap θ (X i)‖ ^ 2 + ‖qkMap θ (X j)‖ ^ 2) := by
    have h := temperatureBarycenterCorrelation_taylor_bound hn X hX
      (fun k => inner ℝ (qkMap θ (X i)) (X k)) (fun k => inner ℝ (qkMap θ (X j)) (X k))
      ‖qkMap θ (X i)‖ ‖qkMap θ (X j)‖
      (abs_query_token_score_le_norm X hX θ.2 i) (abs_query_token_score_le_norm X hX θ.2 j) β hβ
    simpa only [temperatureBarycenterCorrelation_zero] using h
  have he : (∫ θ, R θ ∂ρ) = baryCorr β ρ (empMeasure X) (X i) (X j) - meanOverlap X := by
    have hsub : Integrable (fun θ => F β θ - meanOverlap X) ρ := by
      convert hF.sub (integrable_const _) using 1
    change (∫ θ, F β θ - meanOverlap X - β * D θ ∂ρ) = _
    rw [integral_sub hsub (hD.1.const_mul _),
      integral_sub hF (integrable_const _), integral_const_mul, hD.2]
    have hc : (∫ _θ : HeadParam d, meanOverlap X ∂ρ) = meanOverlap X := by simp
    rw [hc, mul_zero, sub_zero]
    congr 1
    simp only [F, qkMap, temperatureBarycenterCorrelation_eq_inner_softBary, baryCorr]
  calc
    _ = ‖∫ θ, R θ ∂ρ‖ := by rw [he, Real.norm_eq_abs]
    _ ≤ ∫ θ, ‖R θ‖ ∂ρ := norm_integral_le_integral_norm _
    _ ≤ ∫ θ, 10 * β ^ 2 * (‖qkMap θ (X i)‖ ^ 2 + ‖qkMap θ (X j)‖ ^ 2) ∂ρ :=
      integral_mono_ae hR.norm hB (ae_of_all _ fun θ => by simpa only [Real.norm_eq_abs] using hpoint θ)
    _ = _ := by
      rw [integral_const_mul, integral_add hi.1 hj.1, hi.2, hj.2, hX i, hX j]
      ring

/-- The actual squared barycenters have the original mean plus a
uniform quadratic temperature bound. Source: arXiv:2604.01978v1,
§7, the diagonal case in `eq:s_mu_small_beta_clean`. -/
theorem baryCorr_self_le_meanOverlap_add_small_temperature {d n : ℕ} (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) (i : Idx n) :
    baryCorr β ρ (empMeasure X) (X i) (X i) ≤
      meanOverlap X + 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 := by
  have h := (abs_le.mp (abs_baryCorr_sub_meanOverlap_le_small_temperature hn β hβ hρ X hX i i)).2
  linarith

/-- **The genuine mean-noise density retains its actual mean factor
at small temperature.** Source: arXiv:2604.01978v1, §7, the
mean bracket calculation, refined with `eq:s_mu_small_beta_clean`. -/
theorem integral_meanOverlapNoise_sq_le_small_temperature {d n : ℕ} (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) :
    (∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ) ≤ 4 * (σV : ℝ) ^ 2 * meanOverlap X *
      (meanOverlap X + 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by
  apply (integral_meanOverlapNoise_sq_le_mean_barycenters hn β hρ X hX).trans
  apply mul_le_mul_of_nonneg_left _ (by positivity [meanOverlap_nonneg X])
  calc
    _ ≤ (n : ℝ)⁻¹ * ∑ _i : Idx n, (meanOverlap X + 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) :=
      mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i hi =>
        baryCorr_self_le_meanOverlap_add_small_temperature hn β hβ hρ X hX i) (by positivity)
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      rw [← mul_assoc, inv_mul_cancel₀ (by exact_mod_cast hn.ne' : (n : ℝ) ≠ 0), one_mul]

/-- At the source's adopted value normalization the actual small-mean
noise bound has the original dimension factor `1/d`.
Source: arXiv:2604.01978v1, §2.3.3, `σV² = 1/d`, and §7,
the mean bracket calculation and `eq:s_mu_small_beta_clean`. -/
theorem integral_meanOverlapNoise_sq_le_small_temperature_std {d n : ℕ} (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    (∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ) ≤ (4 / (d : ℝ)) * meanOverlap X *
      (meanOverlap X + 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by
  have h := integral_meanOverlapNoise_sq_le_small_temperature hn β hβ hρ X hX
  rw [hσ] at h
  convert h using 1
  ring

/-- Nonempty original orthogonal unit tokens, positive original
Gaussian variances and a positive temperature satisfy all bounds jointly. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
      ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 :=
  ⟨fun k => EuclideanSpace.single k 1, by decide, fun k => by simp,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, by norm_num⟩

end Transformer.Homogenized
