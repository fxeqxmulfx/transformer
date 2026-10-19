module

public import Transformer.Homogenized.Section7_MeanLogNoiseBound

/-!
# Signed actual logarithmic drift bounds for the original empirical mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:drift_mismatch_target_clean` and the mean bracket calculation.
The genuine Gaussian log-mean generator is `g/(m+a)-q_log/2`.
Combining the proved signed original drift bounds with the actual
bound `0 ≤ q_log ≤ 4/d` gives, for `Z=m+a` and
`K=a+a²+80 β² d² σA⁴`,

  `1-Z-4/d-K/Z ≤ L log Z ≤ 1-Z+K/Z`.

These preserve the decreasing logistic part of the actual logarithmic
drift. Its upper comparison field `1-Z+K/Z` is decreasing for positive
`Z` and nonnegative `K`. A shifted deterministic logistic reference
is a supersolution of its corresponding scalar equation.

For the fixed proved regularization, its small regime also gives
the genuine uniform generator bound `|L log Z| ≤ 9`. All stochastic
inputs below are the original Gaussian head and actual generator;
their drift and variance estimates are proved rather than assumed.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped NNReal

namespace Transformer.Homogenized

/-- The genuine Itô correction is half the actual logarithmic noise
second moment. Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`
and Appendix, `def:ito.formula`, for the regularized mean logarithm. -/
theorem sphGenerator_logMeanOverlap_gaussian_eq_div {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X =
      meanOverlapDrift β ρ X / (meanOverlap X + a) -
        (∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ) / 2 := by
  rw [sphGenerator_logMeanOverlap_gaussian ha hn β hρ hσ X hX,
    integral_logMeanOverlapNoiseIntegrand_sq ha]
  ring

/-- The actual shifted logistic algebra preserves its signed drift
and quadratic correction. Source: arXiv:2604.01978v1, §7,
`eq:drift_mismatch_target_clean` and the mean bracket calculation.
This deterministic lemma takes explicit scalar drift and noise bounds;
the application below derives them from the original Gaussian head law. -/
theorem regularized_logistic_drift_scalar_bounds (m a b c g q : ℝ)
    (hm : m ∈ Icc (0 : ℝ) 1) (ha : 0 < a) (hc : 0 ≤ c)
    (hg : g ∈ Icc (m * (1 - m) - c * m - b) (m * (1 - m) + b))
    (hq : q ∈ Icc (0 : ℝ) (2 * c)) :
    g / (m + a) - q / 2 ∈ Icc
      (1 - (m + a) - 2 * c - (a + a ^ 2 + b) / (m + a))
      (1 - (m + a) + (a + a ^ 2 + b) / (m + a)) := by
  have hz : 0 < m + a := by linarith [hm.1]
  have hu : g ≤ (m + a) * (1 - (m + a)) + (a + a ^ 2 + b) := by
    nlinarith [hg.2, mul_nonneg ha.le (sub_nonneg.mpr hm.2)]
  have hl : (m + a) * (1 - (m + a)) - c * (m + a) - (a + a ^ 2 + b) ≤ g := by
    nlinarith [hg.1, mul_nonneg ha.le hm.1, mul_nonneg hc ha.le, sq_nonneg a]
  have hu' : g / (m + a) ≤ 1 - (m + a) + (a + a ^ 2 + b) / (m + a) := by
    apply (div_le_div_of_nonneg_right hu hz.le).trans_eq
    field_simp
  have hl' : 1 - (m + a) - c - (a + a ^ 2 + b) / (m + a) ≤ g / (m + a) := by
    apply le_trans ?_ (div_le_div_of_nonneg_right hl hz.le)
    apply le_of_eq
    field_simp
  constructor <;> linarith [hq.1, hq.2]

/-- The genuine logarithmic upper comparison field is decreasing.
Source: arXiv:2604.01978v1, §7, the logistic comparison in
`eq:logistic_u_def_clean`, after the fixed mean regularization. -/
theorem regularized_logistic_upper_antitone (K x y : ℝ) (hK : 0 ≤ K)
    (hx : 0 < x) (hxy : x ≤ y) : 1 - y + K / y ≤ 1 - x + K / x := by
  have h := div_le_div_of_nonneg_left hK hx hxy
  linarith

/-- Distinct positive values give a strict comparison, as needed
at a deterministic barrier contact point.
Source: arXiv:2604.01978v1, §7, the logarithmic logistic comparison
for the mean in `eq:logistic_u_def_clean`. -/
theorem regularized_logistic_upper_strictAnti (K x y : ℝ) (hK : 0 ≤ K)
    (hx : 0 < x) (hxy : x < y) : 1 - y + K / y < 1 - x + K / x := by
  have h := div_le_div_of_nonneg_left hK hx hxy.le
  linarith

/-- Shifting the actual deterministic logistic reference upward
gives its scalar supersolution inequality.
Source: arXiv:2604.01978v1, §7, the original logistic Cauchy problem,
with the positive deterministic error envelope kept explicit. -/
theorem logistic_shift_supersolution_scalar (u A K : ℝ) (hu : 0 ≤ u) (hA : 0 ≤ A) :
    (u + A) * (1 - (u + A)) + K ≤ u * (1 - u) + A + K := by
  nlinarith [mul_nonneg hu hA, sq_nonneg A]

/-- **The genuine original Gaussian log-mean generator has the
signed logistic bounds, with the original dimension and temperature.**
Source: arXiv:2604.01978v1, §7, the original mean drift mismatch
and bracket calculation, transformed by the genuine logarithmic Itô formula. -/
theorem sphGenerator_logMeanOverlap_mem_Icc {d n : ℕ} (hd : 2 ≤ d) {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hreg : 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 ≤ a)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X ∈ Icc
      (1 - (meanOverlap X + a) - 4 / (d : ℝ) -
        (a + a ^ 2 + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) / (meanOverlap X + a))
      (1 - (meanOverlap X + a) +
        (a + a ^ 2 + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) / (meanOverlap X + a)) := by
  rw [sphGenerator_logMeanOverlap_gaussian_eq_div ha hn β hρ hσ X hX]
  have hg := meanOverlapDrift_small_temperature_mem_Icc hd hn β hβ hρ X hX
  have hq := integral_logMeanOverlapNoise_sq_le_four_div_dim ha hn β hβ hρ hσ hreg X hX
  have hg' : meanOverlapDrift β ρ X ∈ Icc
      (meanOverlap X * (1 - meanOverlap X) - (2 / (d : ℝ)) * meanOverlap X -
        80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
      (meanOverlap X * (1 - meanOverlap X) + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by
    constructor
    · apply le_trans (le_of_eq (by ring)) hg.1
    · exact hg.2
  have hs := regularized_logistic_drift_scalar_bounds (meanOverlap X) a
    (80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) (2 / (d : ℝ)) (meanOverlapDrift β ρ X)
    (∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ) (meanOverlap_mem_Icc X hX) ha
    (by positivity) hg'
    ⟨integral_nonneg fun θ => sq_nonneg _, hq.trans_eq (by ring)⟩
  constructor
  · apply le_trans (le_of_eq (by ring)) hs.1
  · exact hs.2

/-- The small fixed regularization also gives a genuine uniform
bound on the original log-mean generator.
Source: arXiv:2604.01978v1, §7, the bounded generator inputs
to the mean's exponential maximal estimate. -/
theorem abs_sphGenerator_logMeanOverlap_regularization_le_nine {d n : ℕ} (hd : 2 ≤ d)
    (hn : 0 < n) (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap (meanLogRegularization d β σA)) X| ≤ 9 := by
  let a := meanLogRegularization d β σA
  have ha : 0 < a := meanLogRegularization_pos (by omega) β σA
  have h := sphGenerator_logMeanOverlap_mem_Icc hd ha hn β hβ hρ hσ
    (temperature_error_le_meanLogRegularization d β σA) X hX
  have hm := meanOverlap_mem_Icc X hX
  have hK := meanLogRegularization_perturbation_le_six d β σA hsmall
  have hdiv : (a + a ^ 2 + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) / (meanOverlap X + a) ≤ 6 := by
    apply (div_le_iff₀ (meanOverlap_add_pos ha X)).mpr
    change a + a ^ 2 + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 ≤ 6 * (meanOverlap X + a)
    change a + a ^ 2 + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 ≤ 6 * a at hK
    linarith [hm.1]
  have hdR : (2 : ℝ) ≤ d := by exact_mod_cast hd
  have hdim : 4 / (d : ℝ) ≤ 2 := (div_le_iff₀ (by linarith : (0 : ℝ) < d)).mpr (by linarith)
  apply abs_le.mpr
  constructor <;> linarith [h.1, h.2, hm.1, hm.2, hsmall]

/-- Nonzero scalar drift and noise, positive shift and original
mean inside `(0,1)` jointly satisfy every deterministic scalar premise. -/
example : (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) ≤ 1 / 4 ∧
    (1 / 4 : ℝ) ∈ Icc ((1 / 2) * (1 - 1 / 2) - (1 / 4) * (1 / 2) - 1 / 10)
      ((1 / 2) * (1 - 1 / 2) + 1 / 10) ∧ (1 / 4 : ℝ) ∈ Icc (0 : ℝ) (2 * (1 / 4)) ∧
    (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 10 := by norm_num

/-- The positive original Gaussian head, a nonzero small fixed
regularization and original unit queries jointly realize all actual bounds. -/
example : ∃ X : Idx 2 → EucSpace 2, (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) (1 / 2) (gaussHeadLaw 2 (stdSigmaV 2) (1 / 2)) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ (1 / 4 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    meanLogRegularization 2 (1 / 4) (1 / 2) ≤ 1 := by
  refine ⟨fun k => EuclideanSpace.single k 1, le_rfl, by decide, fun k => by simp,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, by norm_num, ?_⟩
  norm_num [meanLogRegularization]

end Transformer.Homogenized
