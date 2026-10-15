/-
# Uniform original logarithmic mean noise with the source confidence scale

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the maximal estimate after the mean bracket calculation.
The actual logarithmic mean noise has variance coefficient `4/d`.
Its two-sided exponential tail, evaluated at four times
`sqrt(T/d log(4/δ))`, leaves failure probability at most `δ/2`.
Measurability and the complement identity give probability at least
`1-δ` for uniform control. The sole stochastic premise is the full
original adapted continuous sphere martingale problem.
-/

import Transformer.Homogenized.Section7_GaussianMeanLogMaximal

open Set MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- A fixed universal multiple of the original source confidence radius.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`, after
the genuine fixed regularized logarithmic test. -/
noncomputable def meanLogNoiseRadius (d : ℕ) (T δ : ℝ) : ℝ :=
  4 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ))

/-- The source confidence scale is positive on its entire parameter range.
Source: arXiv:2604.01978v1, §7, `thm:clustering_small_beta`. -/
theorem meanLogNoiseRadius_pos {d : ℕ} (hd : 0 < d) {T δ : ℝ}
    (hT : 0 < T) (hδ : δ ∈ Ioo (0 : ℝ) 1) : 0 < meanLogNoiseRadius d T δ := by
  have hd0 : (0 : ℝ) < d := by exact_mod_cast hd
  have hl : 0 < Real.log (4 / δ) := Real.log_pos ((lt_div_iff₀ hδ.1).2 (by linarith [hδ.2]))
  exact mul_pos (by norm_num) (Real.sqrt_pos.mpr (mul_pos (div_pos hT hd0) hl))

/-- The square of the actual confidence radius retains the source logarithm.
Source: arXiv:2604.01978v1, §7, the mean maximal inequality. -/
theorem meanLogNoiseRadius_sq {d : ℕ} (hd : 0 < d) {T δ : ℝ}
    (hT : 0 < T) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    meanLogNoiseRadius d T δ ^ 2 = 16 * (T / (d : ℝ) * Real.log (4 / δ)) := by
  have hd0 : (0 : ℝ) < d := by exact_mod_cast hd
  have hl : 0 < Real.log (4 / δ) := Real.log_pos ((lt_div_iff₀ hδ.1).2 (by linarith [hδ.2]))
  rw [meanLogNoiseRadius, mul_pow, Real.sq_sqrt (by positivity)]
  norm_num

/-- The confidence radius gives twice the source logarithm in the actual exponent.
Source: arXiv:2604.01978v1, §7, the mean-noise Chernoff optimization. -/
theorem meanLog_noise_confidence_exponent {d : ℕ} (hd : 0 < d) {T δ : ℝ}
    (hT : 0 < T) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    -(meanLogNoiseRadius d T δ ^ 2 * (d : ℝ) / (8 * T)) = -2 * Real.log (4 / δ) := by
  have hd0 : (0 : ℝ) < d := by exact_mod_cast hd
  rw [meanLogNoiseRadius_sq hd hT hδ]
  field_simp
  ring

/-- The actual confidence radius pays for both signs with a positive reserve.
Source: arXiv:2604.01978v1, §7, optimization of the original maximal estimate. -/
theorem meanLog_noise_exponential_reserve {d : ℕ} (hd : 0 < d) {T δ : ℝ}
    (hT : 0 < T) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    2 * Real.exp (-(meanLogNoiseRadius d T δ ^ 2 * (d : ℝ) / (8 * T))) ≤ δ / 2 := by
  have hq : 0 < 4 / δ := div_pos (by norm_num) hδ.1
  have hl : 0 < Real.log (4 / δ) := Real.log_pos ((lt_div_iff₀ hδ.1).2 (by linarith [hδ.2]))
  rw [meanLog_noise_confidence_exponent hd hT hδ]
  calc
    _ ≤ 2 * Real.exp (-Real.log (4 / δ)) :=
      mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) (by norm_num)
    _ = δ / 2 := by
      rw [Real.exp_neg, Real.exp_log hq]
      field_simp
      ring

/-- The original two-sided maximal failure expression is at most `δ`.
Source: arXiv:2604.01978v1, §7, the confidence choice after the mean bracket estimate. -/
theorem meanLog_noise_exponential_reserve_ennreal {d : ℕ} (hd : 0 < d) {T δ : ℝ}
    (hT : 0 < T) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    2 * ENNReal.ofReal (Real.exp (-(meanLogNoiseRadius d T δ ^ 2 * (d : ℝ) / (8 * T)))) ≤
      ENNReal.ofReal δ := by
  have h := ENNReal.ofReal_le_ofReal ((meanLog_noise_exponential_reserve hd hT hδ).trans
    (by linarith [hδ.1] : δ / 2 ≤ δ))
  rw [ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 2)] at h
  norm_num only [ENNReal.ofReal_ofNat] at h
  exact h

/-- Uniform control of the original logarithmic mean noise holds with probability
at least `1-δ`, with all maximal and event regularity inputs already derived.
Source: arXiv:2604.01978v1, §7, the mean maximal argument in
`thm:clustering_small_beta`, using the actual regularized logarithmic test. -/
theorem gaussian_logMean_noise_probability {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1)
    (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (γ : ℝ) (hinit : ∀ ω, logMeanOverlap (meanLogRegularization d β σA) (X 0 ω) = γ) :
    ENNReal.ofReal (1 - δ) ≤ P {ω | ∀ t ∈ Icc (0 : ℝ) T,
      |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X γ t ω| ≤ meanLogNoiseRadius d T δ} := by
  have hrad := meanLog_noise_exponential_reserve_ennreal (d := d) (by omega) hT hδ
  have hgood := gaussian_logMean_noise_good_measure hd hn β hβ σV σA ρ hρ hσ hsmall P ℱ T hT
    X hMP γ hinit (meanLogNoiseRadius d T δ) (meanLogNoiseRadius_pos (by omega) hT hδ)
  rw [ENNReal.ofReal_sub 1 hδ.1.le]
  norm_num only [ENNReal.ofReal_one]
  exact (tsub_le_tsub_left hrad 1).trans hgood

/-- A deterministic original initial configuration fixes the initial logarithm
in the uniform probability bound. Source: arXiv:2604.01978v1, §7,
`thm:clustering_small_beta`, with the original deterministic initial data. -/
theorem gaussian_logMean_noise_probability_of_initial_configuration {d n : ℕ}
    (hd : 2 ≤ d) (hn : 0 < n) (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1)
    (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hsmall : meanLogRegularization d β σA ≤ 1)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1)
    (X : ℝ → Ω → (Idx n → EucSpace d))
    (hMP : IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X)
    (x₀ : Idx n → EucSpace d) (hinit : ∀ ω, X 0 ω = x₀) :
    ENNReal.ofReal (1 - δ) ≤ P {ω | ∀ t ∈ Icc (0 : ℝ) T,
      |logMeanNoiseResidual (meanLogRegularization d β σA) β ρ X
        (logMeanOverlap (meanLogRegularization d β σA) x₀) t ω| ≤ meanLogNoiseRadius d T δ} := by
  apply gaussian_logMean_noise_probability hd hn β hβ σV σA ρ hρ hσ hsmall P ℱ T hT δ hδ X hMP
  intro ω
  rw [hinit ω]

/-- The original confidence radius is nonnegative also outside the positive
horizon regime. Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`. -/
theorem meanLogNoiseRadius_nonneg (d : ℕ) (T δ : ℝ) : 0 ≤ meanLogNoiseRadius d T δ := by
  exact mul_nonneg (by norm_num) (Real.sqrt_nonneg _)

/-- A genuine positive-variance Gaussian head and its frozen antipodal sphere
solution realize every probabilistic and confidence-scale hypothesis together. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)) (γ : ℝ),
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ meanLogRegularization 2 0 1 ≤ 1 ∧
    (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    IsSphereMartingaleSolution (gaussHeadLaw 2 (stdSigmaV 2) 1)
      (bField 0 (gaussHeadLaw 2 (stdSigmaV 2) 1)) (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1))
      (Measure.dirac ()) ℱ 1 X ∧
    (∀ ω, logMeanOverlap (meanLogRegularization 2 0 1) (X 0 ω) = γ) ∧
    (∃ x₀ : Idx 2 → EucSpace 2, ∀ ω, X 0 ω = x₀) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let K : Idx 2 → EucSpace 2 := ![v, -v]
  refine ⟨ℱ, fun _ _ => K, logMeanOverlap (meanLogRegularization 2 0 1) K,
    le_rfl, by decide, by norm_num, isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2,
    ?_, one_pos, by norm_num, ?_, fun ω => rfl, K, fun ω => rfl⟩
  · norm_num [meanLogRegularization]
  · exact isSphereMartingaleSolution_antipodal_zero_beta _ (Measure.dirac ()) ℱ 1 v (by simp [v])

end Transformer.Homogenized
