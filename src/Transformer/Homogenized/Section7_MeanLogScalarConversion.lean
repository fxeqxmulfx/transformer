/-
# Scalar conversion of the original logarithmic mean comparison

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the error estimate `eq:smallbeta_main_bound`.
The logarithmic comparisons are converted to differences of the
original mean and the original logistic reference.
The upper estimate uses `exp(2r)-1≤4r` when `r≤1/2`.
The lower estimate uses `1-exp(-s)≤s` and the reference's
relative growth. Its deterministic coefficients satisfy `c≤4a`
and `K≤6a`, as proved for the genuine Gaussian logarithmic drift.

Every scalar drift, range, and reference bound used below is explicit.
The original Gaussian process supplies them in the pathwise application.
In particular the noise radius is never multiplied by the reference's
exponential growth envelope.
-/

import Transformer.Homogenized.Section7_LogMeanLowerBarrier
import Mathlib.Analysis.Complex.ExponentialBounds

open Set

namespace Transformer.Homogenized

/-- Small bounded logarithmic noise has a universal linear exponential increment.
Source: arXiv:2604.01978v1, §7, the logarithmic comparison for `eq:smallbeta_main_bound`. -/
theorem exp_two_noise_le {r : ℝ} (hr : r ∈ Icc (0 : ℝ) (1 / 2)) :
    Real.exp (2 * r) ≤ 1 + 4 * r := by
  have h := Real.abs_exp_sub_one_le (x := 2 * r)
    (by rw [abs_of_nonneg (by linarith [hr.1])]; linarith [hr.2])
  rw [abs_of_nonneg (by linarith [hr.1] : 0 ≤ 2 * r)] at h
  have hh := (le_abs_self (Real.exp (2 * r) - 1)).trans h
  linarith

/-- A logarithmic upper comparison gives an original additive error with
unamplified noise. Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`. -/
theorem mean_log_upper_scalar_error (m u a A r : ℝ) (ha : 0 ≤ a) (hA : 0 ≤ A)
    (hu : u ∈ Ioc (0 : ℝ) 1) (hz : 0 < m + a) (hr : r ∈ Icc (0 : ℝ) (1 / 2))
    (hlog : Real.log (m + a) ≤ Real.log (u + A) + 2 * r) : m - u ≤ 3 * A + 4 * r := by
  have hv : 0 < u + A := by linarith [hu.1]
  have h := Real.exp_le_exp.mpr hlog
  rw [Real.exp_add, Real.exp_log hz, Real.exp_log hv] at h
  have he := mul_le_mul_of_nonneg_left (exp_two_noise_le hr) hv.le
  have hru := mul_le_mul_of_nonneg_left hu.2 (by linarith [hr.1] : 0 ≤ 4 * r)
  have hrA := mul_le_mul_of_nonneg_left hr.2 hA
  nlinarith

/-- The original bounded range controls every larger noise radius as well.
Source: arXiv:2604.01978v1, §7, `eq:smallbeta_main_bound`,
with no smallness restriction on the confidence radius. -/
theorem mean_log_upper_scalar_error_all_radius (m u a A r : ℝ) (hm : m ≤ 1)
    (ha : 0 ≤ a) (hA : 0 ≤ A) (hu : u ∈ Ioc (0 : ℝ) 1) (hz : 0 < m + a) (hr : 0 ≤ r)
    (hlog : Real.log (m + a) ≤ Real.log (u + A) + 2 * r) : m - u ≤ 3 * A + 4 * r := by
  by_cases hsmall : r ≤ 1 / 2
  · exact mean_log_upper_scalar_error m u a A r ha hA hu hz ⟨hr, hsmall⟩ hlog
  · have hbounded : m - u ≤ 1 := by linarith [hu.1]
    linarith

/-- The logarithmic lower comparison gives the original additive mean error.
Source: arXiv:2604.01978v1, §7, the original mean comparison in `eq:smallbeta_main_bound`. -/
theorem mean_log_lower_scalar_error (m u a s : ℝ) (hu : 0 < u) (hz : 0 < m + a)
    (hlog : Real.log u - Real.log (m + a) ≤ s) : u - m ≤ u * s + a := by
  have h := Real.exp_le_exp.mpr (show Real.log u + -s ≤ Real.log (m + a) by linarith)
  rw [Real.exp_add, Real.exp_log hu, Real.exp_log hz] at h
  have he := mul_le_mul_of_nonneg_left (Real.add_one_le_exp (-s)) hu.le
  nlinarith

/-- Polynomial time factors fit within a fixed deterministic exponential envelope.
Source: arXiv:2604.01978v1, §7, the universal deterministic coefficient in `eq:smallbeta_main_bound`. -/
theorem mean_log_time_envelope {T : ℝ} (hT : 0 ≤ T) :
    1 + 4 * T + 12 * T * Real.exp T ≤ 17 * Real.exp (2 * T) := by
  have he := Real.one_le_exp_iff.mpr hT
  have ht := Real.add_one_le_exp T
  have hte := mul_le_mul_of_nonneg_right (show T ≤ Real.exp T by linarith) (Real.exp_pos T).le
  rw [show 2 * T = T + T by ring, Real.exp_add]
  nlinarith [sq_nonneg (Real.exp T - 1)]

/-- The genuine relative-growth bound absorbs the lower comparison's
small-mean denominator into the deterministic error term.
Source: arXiv:2604.01978v1, §7, `eq:logistic_u_smallbeta` and `eq:smallbeta_main_bound`. -/
theorem mean_log_lower_error_envelope (T r a c K u₀ u : ℝ) (hT : 0 ≤ T) (hr : 0 ≤ r)
    (ha : 0 ≤ a) (hc : 0 ≤ c) (hK : 0 ≤ K) (hca : c ≤ 4 * a) (hKa : K ≤ 6 * a)
    (hu₀ : 0 < u₀) (hu : u ∈ Ioc (0 : ℝ) 1) (hrel : u ≤ u₀ * Real.exp T) :
    u * (2 * r + (c + 2 * K / u₀) * T) + a ≤ 2 * r + 17 * a * Real.exp (2 * T) := by
  have hnoise := mul_le_mul_of_nonneg_right hu.2 (by positivity : 0 ≤ 2 * r)
  have hct := mul_le_mul_of_nonneg_right hu.2 (by positivity : 0 ≤ c * T)
  have hcaT := mul_le_mul_of_nonneg_right hca hT
  have hratio : u / u₀ ≤ Real.exp T := (div_le_iff₀ hu₀).2 (hrel.trans_eq (mul_comm _ _))
  have hkr := mul_le_mul_of_nonneg_right hratio (by positivity : 0 ≤ 2 * K * T)
  have hkaT := mul_le_mul_of_nonneg_right hKa (by positivity : 0 ≤ 2 * T * Real.exp T)
  have hg := mul_le_mul_of_nonneg_left (mean_log_time_envelope hT) ha
  calc
    _ ≤ 2 * r + a * (1 + 4 * T + 12 * T * Real.exp T) := by
      simp only [div_eq_mul_inv] at hkr ⊢
      nlinarith
    _ ≤ _ := by nlinarith

/-- Outside the short logarithmic comparison regime, small noise and
small dimension bias force the original initial mean to be small.
Source: arXiv:2604.01978v1, §7, the regimes in `eq:smallbeta_main_bound`. -/
theorem mean_log_large_regime_initial_bound (T r c K u₀ : ℝ) (hu₀ : 0 < u₀)
    (hr : r < 1 / 16) (hcT : c * T < 1 / 8)
    (hlarge : Real.log 2 ≤ 2 * r + (c + 2 * K / u₀) * T) : u₀ ≤ 8 * K * T := by
  have hl : (1 / 2 : ℝ) < Real.log 2 := by linarith [Real.log_two_gt_d9]
  have he : (c + 2 * K / u₀) * T = c * T + (2 * K * T) / u₀ := by ring
  rw [he] at hlarge
  have hratio : (1 / 4 : ℝ) ≤ (2 * K * T) / u₀ := by linarith
  have hm := (le_div_iff₀ hu₀).mp hratio
  nlinarith

/-- Nonnegative time is covered by the fixed original deterministic exponential envelope.
Source: arXiv:2604.01978v1, §7, absorption of time factors in `eq:smallbeta_main_bound`. -/
theorem mean_log_time_le_exp {T : ℝ} (hT : 0 ≤ T) : T ≤ Real.exp (2 * T) := by
  nlinarith [Real.add_one_le_exp (2 * T)]

/-- A large dimension-time bias is already covered by the deterministic envelope.
Source: arXiv:2604.01978v1, §7, the deterministic term in `eq:smallbeta_main_bound`. -/
theorem mean_log_large_dimension_absorption (T a c : ℝ) (hT : 0 ≤ T) (ha : 0 ≤ a)
    (hca : c ≤ 4 * a) (hcT : 1 / 8 ≤ c * T) : 1 ≤ 32 * a * Real.exp (2 * T) := by
  have hc := mul_le_mul_of_nonneg_right hca hT
  have ht := mean_log_time_le_exp hT
  have he := mul_le_mul_of_nonneg_left ht (by positivity : 0 ≤ 32 * a)
  nlinarith

/-- Positive regularization, a genuine logistic equilibrium, and a positive
time jointly realize the upper and lower scalar comparison premises. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (1 : ℝ) ∈ Ioc (0 : ℝ) 1 ∧
    0 < 1 + (1 / 4 : ℝ) ∧ (0 : ℝ) ∈ Icc (0 : ℝ) (1 / 2) ∧
    Real.log (1 + (1 / 4 : ℝ)) ≤ Real.log (1 + (1 / 4 : ℝ)) + 2 * 0 ∧
    Real.log (1 : ℝ) - Real.log (1 + (1 / 4 : ℝ)) ≤ 0 := by
  have hl : 0 ≤ Real.log (1 + (1 / 4 : ℝ)) := Real.log_nonneg (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by simp, ?_⟩
  simpa only [Real.log_one, zero_sub, neg_nonpos] using hl

/-- The deterministic envelope and relative-growth inputs are jointly realized
at the positive equilibrium with nonzero dimension and temperature error constants. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 4 ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 / 8 ∧ (1 / 4 : ℝ) ≤ 4 * (1 / 4) ∧
    (1 / 8 : ℝ) ≤ 6 * (1 / 4) ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ∈ Ioc (0 : ℝ) 1 ∧
    (1 : ℝ) ≤ 1 * Real.exp 1 := by
  refine ⟨zero_le_one, le_rfl, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, one_pos, by norm_num, ?_⟩
  simpa only [one_mul] using Real.one_le_exp_iff.mpr zero_le_one

/-- Nonzero regularization and dimension bias also realize both complementary regimes. -/
example : (0 : ℝ) < 1 / 64 ∧ (0 : ℝ) < 1 / 16 ∧ (0 : ℝ) * 1 < 1 / 8 ∧
    Real.log 2 ≤ 2 * (0 : ℝ) + (0 + 2 * (1 / 8) / (1 / 64)) * 1 ∧
    (1 / 4 : ℝ) ≤ 4 * (1 / 4) ∧ (1 / 8 : ℝ) ≤ (1 / 4) * 1 := by
  have hl := Real.log_two_lt_d9
  refine ⟨by norm_num, by norm_num, by norm_num, ?_, by norm_num, by norm_num⟩
  linarith

end Transformer.Homogenized
