module

public import Transformer.Homogenized.Section7_MeanLogScalarConversion

/-!
# The original mean error with deterministic bias and unamplified noise

ArXiv:2604.01978v1, §7, `thm:clustering_small_beta` and
`eq:smallbeta_main_bound`.
The two actual logarithmic comparisons imply an original additive
mean error at most `128 a exp(2T)+16r` for the fixed regularization.
The small comparison regime uses the original positive logistic
reference and its proved relative growth. In the remaining regimes,
the original range controls large noise, large dimension-time bias,
or small initial mean. No lower bound on the initial mean enters
the final universal constants.

The proved canonical regularization is at most `21 max(1/d,ε)`.
Together with the actual confidence radius `4 sqrt(T/d log(4/δ))`,
this fits the source's displayed error with universal constant 4096.
The deterministic scalar inputs are derived from the original Gaussian
solution in the subsequent application.
-/

@[expose] public section

open Set
open scoped NNReal

namespace Transformer.Homogenized

/-- The two logarithmic comparisons give the original mean error uniformly
over all confidence radii and all deterministic comparison regimes.
Source: arXiv:2604.01978v1, §7, the rate in `eq:smallbeta_main_bound`.
This scalar lemma keeps its actual mean, reference, and comparison inputs explicit. -/
theorem mean_log_original_error_envelope (T t r a c K u₀ u m A : ℝ)
    (hT : 0 ≤ T) (ht : t ∈ Icc (0 : ℝ) T) (hr : 0 ≤ r) (ha : 0 < a)
    (hc : 0 ≤ c) (hK : 0 ≤ K) (hca : c ≤ 4 * a) (hKa : K ≤ 6 * a)
    (hu₀ : 0 < u₀) (hu : u ∈ Ioc (0 : ℝ) 1) (hm : m ∈ Icc (0 : ℝ) 1)
    (hrel : u ≤ u₀ * Real.exp T) (hA : 0 ≤ A) (hAup : A ≤ 7 * a * Real.exp T)
    (hupper : Real.log (m + a) ≤ Real.log (u + A) + 2 * r)
    (hlower : 2 * r + (c + 2 * K / u₀) * T < Real.log 2 →
      Real.log u - Real.log (m + a) ≤ 2 * r + (c + 2 * K / u₀) * t) :
    |m - u| ≤ 128 * a * Real.exp (2 * T) + 16 * r := by
  have hz : 0 < m + a := by linarith [hm.1]
  have hEa : 0 ≤ a * Real.exp (2 * T) := by positivity
  have hE : Real.exp T ≤ Real.exp (2 * T) := Real.exp_le_exp.mpr (by linarith)
  have hEa' := mul_le_mul_of_nonneg_left hE ha.le
  have hUpper : m - u ≤ 128 * a * Real.exp (2 * T) + 16 * r := by
    have hb := mean_log_upper_scalar_error_all_radius m u a A r hm.2 ha.le hA hu hz hr hupper
    nlinarith
  have hLower : u - m ≤ 128 * a * Real.exp (2 * T) + 16 * r := by
    by_cases hshort : 2 * r + (c + 2 * K / u₀) * T < Real.log 2
    · have hb := mean_log_lower_scalar_error m u a (2 * r + (c + 2 * K / u₀) * t) hu.1 hz (hlower hshort)
      have hC : 0 ≤ c + 2 * K / u₀ := by positivity
      have htime := mul_le_mul_of_nonneg_left ht.2 hC
      have htime' := mul_le_mul_of_nonneg_left htime hu.1.le
      have hg := mean_log_lower_error_envelope T r a c K u₀ u hT hr ha.le hc hK hca hKa hu₀ hu hrel
      nlinarith
    · by_cases hnoise : 1 / 16 ≤ r
      · nlinarith [hu.2, hm.1]
      · by_cases hdim : 1 / 8 ≤ c * T
        · have hb := mean_log_large_dimension_absorption T a c hT ha.le hca hdim
          nlinarith [hu.2, hm.1]
        · have hinit := mean_log_large_regime_initial_bound T r c K u₀ hu₀
            (lt_of_not_ge hnoise) (lt_of_not_ge hdim) (le_of_not_gt hshort)
          have hi := mul_le_mul_of_nonneg_right hinit (Real.exp_pos T).le
          have hKtime := mul_le_mul_of_nonneg_right hKa (by positivity : 0 ≤ 8 * T * Real.exp T)
          have hTE : T * Real.exp T ≤ Real.exp (2 * T) := by
            have he : T ≤ Real.exp T := by linarith [Real.add_one_le_exp T]
            rw [show 2 * T = T + T by ring, Real.exp_add]
            exact mul_le_mul_of_nonneg_right he (Real.exp_pos T).le
          have hTEa := mul_le_mul_of_nonneg_left hTE ha.le
          nlinarith [hm.1]
  exact abs_le.mpr ⟨by linarith, hUpper⟩

/-- The original mean and logistic reference have the same error envelope
simultaneously on the full source horizon. Source: arXiv:2604.01978v1,
§7, the supremum in `eq:smallbeta_main_bound`. -/
theorem mean_log_original_error_uniform (T r a c K u₀ : ℝ) (hT : 0 ≤ T) (hr : 0 ≤ r)
    (ha : 0 < a) (hc : 0 ≤ c) (hK : 0 ≤ K) (hca : c ≤ 4 * a) (hKa : K ≤ 6 * a)
    (hu₀ : 0 < u₀) (m u A : ℝ → ℝ)
    (hu : ∀ t ∈ Icc (0 : ℝ) T, u t ∈ Ioc (0 : ℝ) 1)
    (hm : ∀ t ∈ Icc (0 : ℝ) T, m t ∈ Icc (0 : ℝ) 1)
    (hrel : ∀ t ∈ Icc (0 : ℝ) T, u t ≤ u₀ * Real.exp T)
    (hA : ∀ t ∈ Icc (0 : ℝ) T, A t ∈ Icc (0 : ℝ) (7 * a * Real.exp T))
    (hupper : ∀ t ∈ Icc (0 : ℝ) T, Real.log (m t + a) ≤ Real.log (u t + A t) + 2 * r)
    (hlower : 2 * r + (c + 2 * K / u₀) * T < Real.log 2 →
      ∀ t ∈ Icc (0 : ℝ) T, Real.log (u t) - Real.log (m t + a) ≤ 2 * r + (c + 2 * K / u₀) * t) :
    ∀ t ∈ Icc (0 : ℝ) T, |m t - u t| ≤ 128 * a * Real.exp (2 * T) + 16 * r := by
  intro t ht
  exact mean_log_original_error_envelope T t r a c K u₀ (u t) (m t) (A t) hT ht hr ha hc hK hca hKa
    hu₀ (hu t ht) (hm t ht) (hrel t ht) (hA t ht).1 (hA t ht).2 (hupper t ht)
    (fun hs => hlower hs t ht)

/-- The original unit-configuration mean and logistic reference have total error at most one.
Source: arXiv:2604.01978v1, §7, the ranges used in `thm:clustering_small_beta`. -/
theorem abs_mean_sub_logistic_le_one {m u : ℝ} (hm : m ∈ Icc (0 : ℝ) 1) (hu : u ∈ Icc (0 : ℝ) 1) :
    |m - u| ≤ 1 := by
  apply abs_le.mpr
  constructor <;> linarith [hm.1, hm.2, hu.1, hu.2]

/-- The actual canonical error and proved confidence radius fit a single
universal constant in the source's displayed deterministic and stochastic scales.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`, with explicit constant 4096. -/
theorem mean_log_error_le_source_scale (d : ℕ) (β : ℝ) (σA : ℝ≥0) (T δ : ℝ) (hT : 0 ≤ T) :
    128 * meanLogRegularization d β σA * Real.exp (2 * T) + 16 * meanLogNoiseRadius d T δ ≤
      4096 * Real.exp (4096 * T) *
        max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2) +
      4096 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ)) := by
  let M := max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2)
  have hM : 0 ≤ M := (by positivity : 0 ≤ 1 / (d : ℝ)).trans (le_max_left _ _)
  have ha : meanLogRegularization d β σA ≤ 21 * M := by
    have h := meanLogRegularization_le_max d β σA
    rw [show β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 = (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2 by ring] at h
    exact h
  have hE : Real.exp (2 * T) ≤ Real.exp (4096 * T) := Real.exp_le_exp.mpr (by linarith)
  have ha' := mul_le_mul_of_nonneg_right ha (Real.exp_pos (2 * T)).le
  have hEM := mul_le_mul_of_nonneg_left hE hM
  have hMp : 0 ≤ M * Real.exp (4096 * T) := mul_nonneg hM (Real.exp_pos _).le
  dsimp only [meanLogNoiseRadius]
  change 128 * meanLogRegularization d β σA * Real.exp (2 * T) +
      16 * (4 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ))) ≤ _
  change _ ≤ 4096 * Real.exp (4096 * T) * M + 4096 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ))
  nlinarith [Real.sqrt_nonneg (T / (d : ℝ) * Real.log (4 / δ))]

/-- Large regularization is already covered by the original deterministic envelope.
Source: arXiv:2604.01978v1, §7, the large-error regime of `eq:smallbeta_main_bound`. -/
theorem mean_log_large_regularization_absorption (a T : ℝ) (ha : 1 ≤ a) (hT : 0 ≤ T) :
    1 ≤ 128 * a * Real.exp (2 * T) := by
  have he := Real.one_le_exp_iff.mpr (show 0 ≤ 2 * T by linarith)
  have hm := mul_le_mul_of_nonneg_left he (by linarith : 0 ≤ a)
  nlinarith

/-- A positive original equilibrium and regularization realize every mean-error
comparison premise together at a positive horizon and a nonnegative interior time. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 4 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 4 * (1 / 4) ∧ (0 : ℝ) ≤ 6 * (1 / 4) ∧
    (0 : ℝ) < 1 ∧ (1 : ℝ) ∈ Ioc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    (1 : ℝ) ≤ 1 * Real.exp 1 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ 7 * (1 / 4) * Real.exp 1 ∧
    Real.log (1 + (1 / 4 : ℝ)) ≤ Real.log (1 + (1 / 4 : ℝ)) + 2 * 0 ∧
    (2 * (0 : ℝ) + (0 + 2 * 0 / 1) * 1 < Real.log 2 →
      Real.log (1 : ℝ) - Real.log (1 + (1 / 4 : ℝ)) ≤ 2 * 0 + (0 + 2 * 0 / 1) * 0) := by
  have he : 1 ≤ Real.exp (1 : ℝ) := Real.one_le_exp_iff.mpr zero_le_one
  have hl : 0 ≤ Real.log (1 + (1 / 4 : ℝ)) := Real.log_nonneg (by norm_num)
  refine ⟨zero_le_one, by norm_num, le_rfl, by norm_num, le_rfl, le_rfl, by norm_num, by norm_num,
    one_pos, by norm_num, by norm_num, by simpa only [one_mul] using he, by norm_num, ?_, by simp, ?_⟩
  · nlinarith
  · intro h
    simpa only [Real.log_one, zero_sub, mul_zero, zero_div, add_zero, neg_nonpos] using hl

/-- The source-scale conversion also has positive dimension, horizon and confidence witnesses. -/
example : (0 : ℝ) ≤ 1 ∧ 1 ≤ meanLogRegularization 2 1 1 := by
  norm_num [meanLogRegularization]

/-- Constant genuine equilibrium paths realize all uniform scalar comparison inputs. -/
example : ∀ t ∈ Icc (0 : ℝ) 1,
    |(1 : ℝ) - 1| ≤ 128 * (1 / 4) * Real.exp (2 * 1) + 16 * 0 := by
  have he : 1 ≤ Real.exp (1 : ℝ) := Real.one_le_exp_iff.mpr zero_le_one
  have hl : 0 ≤ Real.log (1 + (1 / 4 : ℝ)) := Real.log_nonneg (by norm_num)
  apply mean_log_original_error_uniform 1 0 (1 / 4) 0 0 1 zero_le_one le_rfl
    (by norm_num) le_rfl le_rfl (by norm_num) (by norm_num) one_pos
    (fun _ => 1) (fun _ => 1) (fun _ => 1 / 4)
  · exact fun t ht => by norm_num
  · exact fun t ht => by norm_num
  · exact fun t ht => by simpa only [one_mul] using he
  · intro t ht
    constructor
    · norm_num
    · nlinarith
  · exact fun t ht => by simp
  · intro hs t ht
    simpa only [Real.log_one, zero_sub, mul_zero, zero_div, add_zero, neg_nonpos, zero_mul] using hl

end Transformer.Homogenized
