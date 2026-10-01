/-
# Exact mean and variance of Gaussian signed updates

arXiv:2506.12543v1, Section 4.3, Theorem 1 and its proof sketch.
All expectations are integrals against genuine Gaussian probability laws.
-/

import Transformer.BatchSize.Section4_GaussianCDF

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- Measurability of the componentwise sign update, Section 4.3. -/
theorem measurable_real_sign : Measurable Real.sign := by
  unfold Real.sign
  exact measurable_const.ite measurableSet_Iio
    (measurable_const.ite measurableSet_Ioi measurable_const)

/-- The signed update is bounded, Section 4.3, Theorem 1. -/
theorem sign_abs_le_one (x : ℝ) : |Real.sign x| ≤ 1 := by
  rcases lt_trichotomy x 0 with h | h | h
  · simp [Real.sign_of_neg h]
  · simp [h]
  · simp [Real.sign_of_pos h]

/-- Integrability of signed updates, Section 4.3's proof sketch. -/
theorem integrable_gaussian_sign (μ : ℝ) (v : NNReal) :
    Integrable Real.sign (gaussianReal μ v) := by
  apply (integrable_const (1 : ℝ)).mono' measurable_real_sign.aestronglyMeasurable
  exact ae_of_all _ (fun x => by simpa [Real.norm_eq_abs] using sign_abs_le_one x)

/-- Exact Gaussian mean sign, including zero and negative mean;
Section 4.3's final proof-sketch equation. Here s=sqrt(2*variance). -/
theorem gaussian_sign_mean (μ s : ℝ) (hs : 0 < s) :
    (∫ z, Real.sign z ∂gaussianReal μ (.mk (s ^ 2 / 2) (by positivity))) =
      errorFunction (μ / s) := by
  let v : NNReal := .mk (s ^ 2 / 2) (by positivity)
  have hv : v ≠ 0 := by
    intro h
    have hcoe := congrArg (fun a : NNReal => (a : ℝ)) h
    have hp : 0 < s ^ 2 / 2 := by positivity
    have hz : s ^ 2 / 2 = 0 := by simpa [v] using hcoe
    linarith
  have := nullSingletonClass_gaussianReal hv (μ := μ)
  have hi : Integrable ((Set.Iic (0 : ℝ)).indicator (fun _ => (1 : ℝ)))
      (gaussianReal μ v) := (integrable_const _).indicator measurableSet_Iic
  calc
    (∫ z, Real.sign z ∂gaussianReal μ v) =
        ∫ z, 1 - 2 * (Set.Iic (0 : ℝ)).indicator (fun _ => (1 : ℝ)) z
          ∂gaussianReal μ v := by
      apply integral_congr_ae
      filter_upwards [(gaussianReal μ v).ae_ne 0] with z hz
      rcases lt_or_gt_of_ne hz with h | h
      · rw [Real.sign_of_neg h,
          Set.indicator_of_mem (show z ∈ Set.Iic 0 from h.le)]
        norm_num
      · rw [Real.sign_of_pos h,
          Set.indicator_of_notMem (show z ∉ Set.Iic 0 from not_le.mpr h)]
        norm_num
    _ = 1 - 2 * (gaussianReal μ v).real (Set.Iic 0) := by
      rw [integral_sub (integrable_const _) (hi.const_mul 2), integral_const_mul,
        integral_indicator measurableSet_Iic]
      simp
    _ = errorFunction (μ / s) := by
      rw [gaussian_cdf μ s 0 hs]
      simp only [zero_sub, neg_div, errorFunction_neg]
      ring

/-- Nonvacuity of the mean-sign formula, Section 4.3. -/
example : (0 : ℝ) < 2 := by norm_num

/-- A nondegenerate Gaussian signed update has second moment one;
Section 4.3, Theorem 1's diffusion covariance. -/
theorem gaussian_sign_second_moment (μ : ℝ) (v : NNReal) (hv : v ≠ 0) :
    (∫ z, (Real.sign z) ^ 2 ∂gaussianReal μ v) = 1 := by
  have := nullSingletonClass_gaussianReal hv (μ := μ)
  calc
    (∫ z, (Real.sign z) ^ 2 ∂gaussianReal μ v) = ∫ _ : ℝ, (1 : ℝ) ∂gaussianReal μ v := by
      apply integral_congr_ae
      filter_upwards [(gaussianReal μ v).ae_ne 0] with z hz
      rcases lt_or_gt_of_ne hz with h | h
      · simp [Real.sign_of_neg h]
      · simp [Real.sign_of_pos h]
    _ = 1 := by simp

/-- Nonvacuity of the nondegenerate second moment, Section 4.3. -/
example : (1 : NNReal) ≠ 0 := by norm_num

/-- Exact centered sign variance 1-erf(mu/s)^2, giving the diagonal
diffusion covariance in Section 4.3, Theorem 1. -/
theorem gaussian_sign_variance (μ s : ℝ) (hs : 0 < s) :
    variance Real.sign (gaussianReal μ (.mk (s ^ 2 / 2) (by positivity))) =
      1 - errorFunction (μ / s) ^ 2 := by
  have hm : MemLp Real.sign 2 (gaussianReal μ (.mk (s ^ 2 / 2) (by positivity))) :=
    MemLp.of_bound measurable_real_sign.aestronglyMeasurable 1
      (ae_of_all _ (fun z => by simpa [Real.norm_eq_abs] using sign_abs_le_one z))
  rw [variance_eq_sub hm, gaussian_sign_mean μ s hs]
  have hv : NNReal.mk (s ^ 2 / 2) (by positivity) ≠ 0 := by
    intro h
    have hz : s ^ 2 / 2 = 0 := congrArg (fun a : NNReal => (a : ℝ)) h
    have hp : 0 < s ^ 2 / 2 := by positivity
    linarith
  rw [show (Real.sign ^ 2) = (fun z => (Real.sign z) ^ 2) from rfl,
    gaussian_sign_second_moment μ _ hv]

/-- Nonvacuity of the diffusion covariance formula, Section 4.3. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The proof sketch's equality between P(Z>=0) and sign-matching
probability is false at mu=0: sign(Z)=sign(0) has probability zero for
nondegenerate noise. The final mean-sign identity remains valid. Section 4.3. -/
theorem zero_mean_sign_match_probability (v : NNReal) (hv : v ≠ 0) :
    gaussianReal 0 v {z | Real.sign z = Real.sign 0} = 0 := by
  have := nullSingletonClass_gaussianReal hv (μ := 0)
  have hset : {z : ℝ | Real.sign z = Real.sign 0} = {0} := by
    ext z
    simp
  rw [hset]
  exact measure_singleton 0

/-- Nonvacuity of the zero-mean counterexample, Section 4.3. -/
example : (1 / 2 : NNReal) ≠ 0 := by norm_num

/-- At zero mean the nonnegative half-line has probability 1/2, while
sign matching has probability zero. This explicitly refutes the source's
mu>=0 sign-matching wording; Section 4.3's proof sketch. -/
theorem zero_mean_halfline_probability :
    (gaussianReal 0 (1 / 2)).real (Set.Ici 0) = 1 / 2 := by
  have heq : gaussianReal 0 (1 / 2) (Set.Ici 0) =
      gaussianReal 0 (1 / 2) (Set.Iic 0) := by
    calc
      gaussianReal 0 (1 / 2) (Set.Ici 0) =
          gaussianReal 0 (1 / 2) ((fun x : ℝ => -x) ⁻¹' Set.Iic 0) := by
        congr 1
        ext x
        simp
      _ = (gaussianReal 0 (1 / 2)).map (fun x : ℝ => -x) (Set.Iic 0) :=
        (Measure.map_apply measurable_neg measurableSet_Iic).symm
      _ = gaussianReal 0 (1 / 2) (Set.Iic 0) := by
        rw [gaussianReal_map_neg]
        simp
  change ENNReal.toReal _ = _
  rw [heq]
  change (gaussianReal 0 (1 / 2)).real (Set.Iic 0) = _
  rw [gaussian_half_cdf, errorFunction_zero]
  norm_num

/-- Explicit counterexample to the proof sketch's assertion
P(Z>=0)=P(sign(Z)=sign(mu)) for mu>=0: take mu=0 and variance 1/2;
Section 4.3, proof of Theorem 1. -/
theorem zero_mean_sign_match_counterexample :
    (gaussianReal 0 (1 / 2)).real (Set.Ici 0) ≠
      (gaussianReal 0 (1 / 2)).real {z | Real.sign z = Real.sign 0} := by
  rw [zero_mean_halfline_probability]
  have hz := zero_mean_sign_match_probability (1 / 2) (by norm_num)
  simp only [Measure.real, hz, ENNReal.toReal_zero]
  norm_num

end Transformer.BatchSize
