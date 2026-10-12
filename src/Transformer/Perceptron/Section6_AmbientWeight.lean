/-
# Ambient analyticity of the original interaction weight

ArXiv:2601.21366v2, §5.3, the exponential expansion and Weierstrass
M-test for the first variation, and §6, `prop: unified.log` and
`rem:unifiedlog-B`. The actual weight `∫ exp(xᵀBy) dμ(y)` is
real-analytic on the entire ambient Euclidean space in every dimension.

At an arbitrary center its actual Taylor coefficients are the exponential
series composed with the actual inner-product linear functional, multiplied
by the kernel value at that center. Compactness of the original sphere
uniformly bounds both that value and the functional's operator norm.
The preceding series-integration theorem consequently applies to the
original probability integral. No density or atomicity input is used.
-/

import Transformer.Perceptron.Section6_SeriesIntegral
import Transformer.Perceptron.Section5_DominatedVariation
import Transformer.Perceptron.NormalizedMap
import Mathlib.Analysis.Analytic.Constructions
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic

open Real MeasureTheory Set
open scoped BigOperators NNReal ENNReal

namespace Transformer.Perceptron

/-- The actual multilinear exponential coefficients at the actual
ambient center. Source: arXiv:2601.21366v2, §5.3, the M-test expansion,
and §6, the analytic interaction in `rem:unifiedlog-B`. -/
noncomputable def attentionKernelSeries {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x : EucSpace d) (y : SSphere d) : FormalMultilinearSeries ℝ (EucSpace d) ℝ :=
  Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
    (NormedSpace.expSeries ℝ ℝ).compContinuousLinearMap (innerSL ℝ (B (y : EucSpace d)))

/-- These are the actual Taylor coefficients of the original kernel at
every center, with infinite radius. Source: arXiv:2601.21366v2, §5.3,
the exponential series, and §6, analyticity of the first variation. -/
theorem hasFPowerSeriesOnBall_attentionKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x : EucSpace d) (y : SSphere d) :
    HasFPowerSeriesOnBall (fun z => Real.exp (inner (𝕜 := ℝ) z (B (y : EucSpace d))))
      (attentionKernelSeries B x y) x ⊤ := by
  let L : EucSpace d →L[ℝ] ℝ := innerSL ℝ (B (y : EucSpace d))
  have he : HasFPowerSeriesOnBall Real.exp (NormedSpace.expSeries ℝ ℝ) (L 0) ⊤ := by
    rw [map_zero, Real.exp_eq_exp_ℝ]
    exact NormedSpace.exp_hasFPowerSeriesOnBall
  have hc : HasFPowerSeriesOnBall (fun z => Real.exp (L z))
      ((NormedSpace.expSeries ℝ ℝ).compContinuousLinearMap L) 0 ⊤ := by
    simpa only [Function.comp_def, ENNReal.top_div, enorm_ne_top, ite_false] using
      he.compContinuousLinearMap (u := L) (x := 0)
  have ht := (hc.const_smul (c := Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))))).comp_sub x
  simp only [zero_add] at ht
  apply ht.congr
  intro z _
  change Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) *
    Real.exp (inner (𝕜 := ℝ) (B (y : EucSpace d)) (z - x)) = _
  rw [← Real.exp_add, real_inner_comm (z - x) (B (y : EucSpace d)), inner_sub_left]
  congr 1
  ring

/-- Every actual coefficient varies continuously with the original
integration point. Source: arXiv:2601.21366v2, §5.3, integrating the
uniformly bounded exponential series. The operator-valued coefficient,
not just its diagonal evaluation, is continuous. -/
theorem continuous_attentionKernelSeries {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x : EucSpace d) (n : ℕ) : Continuous (fun y : SSphere d => attentionKernelSeries B x y n) := by
  have hB : Continuous (fun y : SSphere d => B (y : EucSpace d)) :=
    B.continuous_of_finiteDimensional.comp continuous_subtype_val
  have hL : Continuous (fun y : SSphere d => innerSL ℝ (B (y : EucSpace d))) :=
    (innerSL ℝ).continuous.comp hB
  have hcomp : Continuous (fun y : SSphere d =>
      (NormedSpace.expSeries ℝ ℝ n).compContinuousLinearMap
        (fun _ : Fin n => innerSL ℝ (B (y : EucSpace d)))) := by
    exact ((NormedSpace.expSeries ℝ ℝ n).compContinuousLinearMapLRight
      (E := fun _ : Fin n => EucSpace d)).cont.comp (continuous_pi (fun _ => hL))
  have hval : Continuous (fun y : SSphere d => Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d)))) :=
    Real.continuous_exp.comp (continuous_const.inner hB)
  exact hval.smul hcomp

/-- The actual coefficients admit a summable uniform norm bound at
radius one. Source: arXiv:2601.21366v2, §5.3, the M-test.
The constants bound the original kernel value and `By` on the original
compact sphere; factorial decay comes from the true exponential series. -/
theorem exists_summable_attentionKernelSeries_bound {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x : EucSpace d) : ∃ b : ℕ → ℝ,
      (∀ n y, ‖attentionKernelSeries B x y n‖ ≤ b n) ∧ Summable b := by
  have hB : Continuous (fun y : SSphere d => B (y : EucSpace d)) :=
    B.continuous_of_finiteDimensional.comp continuous_subtype_val
  have hval : Continuous (fun y : SSphere d => Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d)))) :=
    Real.continuous_exp.comp (continuous_const.inner hB)
  obtain ⟨C, hC, hc⟩ := exists_pos_bound_continuous_compact _ hval
  obtain ⟨M, hM, hm⟩ := exists_pos_bound_continuous_compact _ hB.norm
  refine ⟨fun n => C * ‖NormedSpace.expSeries ℝ ℝ n‖ * M ^ n, ?_, ?_⟩
  · intro n y
    have hyC : Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) ≤ C :=
      by simpa only [abs_of_pos (Real.exp_pos _)] using hc y
    have hyM : ‖B (y : EucSpace d)‖ ≤ M := by simpa only [abs_norm] using hm y
    change ‖Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
      ((NormedSpace.expSeries ℝ ℝ).compContinuousLinearMap
        (innerSL ℝ (B (y : EucSpace d))) n)‖ ≤ C * ‖NormedSpace.expSeries ℝ ℝ n‖ * M ^ n
    rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le]
    calc
      _ ≤ Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) *
          (‖NormedSpace.expSeries ℝ ℝ n‖ * ‖innerSL ℝ (B (y : EucSpace d))‖ ^ n) :=
        mul_le_mul_of_nonneg_left
          ((NormedSpace.expSeries ℝ ℝ).norm_compContinuousLinearMap_le _ n) (Real.exp_pos _).le
      _ ≤ C * (‖NormedSpace.expSeries ℝ ℝ n‖ * M ^ n) := by
        rw [innerSL_apply_norm]
        exact mul_le_mul hyC
          (mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _) hyM n) (norm_nonneg _))
          (by positivity) hC.le
      _ = _ := by ring
  · let q : ℝ≥0 := ⟨M, hM.le⟩
    have hs := (NormedSpace.expSeries ℝ ℝ).summable_norm_mul_pow
      (r := q) (by rw [NormedSpace.expSeries_radius_eq_top]; simp)
    have hq : (q : ℝ) = M := rfl
    rw [hq] at hs
    simpa only [mul_assoc] using hs.mul_left C

/-- The original weight has the actual integrated kernel coefficients
as its Taylor series on a unit ball at every ambient center.
Source: arXiv:2601.21366v2, §5.3 and §6, the analytic first variation.
The complete original probability measure is integrated. -/
theorem hasFPowerSeriesOnBall_attentionWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    HasFPowerSeriesOnBall (attentionWeightMap B μ)
      (integralFormalMultilinearSeries (μ : Measure (SSphere d)) (attentionKernelSeries B x)) x 1 := by
  obtain ⟨b, hb, hs⟩ := exists_summable_attentionKernelSeries_bound B x
  exact hasFPowerSeriesOnBall_integral_of_uniform_bound (μ : Measure (SSphere d))
    (fun z y => Real.exp (inner (𝕜 := ℝ) z (B (y : EucSpace d))))
    (attentionKernelSeries B x) x 1 one_pos
    (fun n => (continuous_attentionKernelSeries B x n).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)) b hb
    (by simpa using hs)
    (fun y => (hasFPowerSeriesOnBall_attentionKernel B x y).mono (by norm_num) le_top)

/-- The original interaction weight is analytic in the entire actual
ambient space. Source: arXiv:2601.21366v2, §5.3 and §6,
`prop: unified.log` and `rem:unifiedlog-B`. The actual Taylor series
has been proved at every ambient center, without a density input. -/
theorem analyticOnNhd_attentionWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) : AnalyticOnNhd ℝ (attentionWeightMap B μ) univ :=
  fun x _ => (hasFPowerSeriesOnBall_attentionWeightMap B μ x).hasFPowerSeriesAt.analyticAt

/-- The original weight is strictly positive at every ambient point.
Source: arXiv:2601.21366v2, §6, the positive first variation in `H_log`.
This uses the entire actual probability measure, before any support conclusion. -/
theorem attentionWeightMap_pos {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : 0 < attentionWeightMap B μ x := by
  have hB : Continuous (fun y : SSphere d => B (y : EucSpace d)) :=
    B.continuous_of_finiteDimensional.comp continuous_subtype_val
  have hval : Continuous (fun y : SSphere d => Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d)))) :=
    Real.continuous_exp.comp (continuous_const.inner hB)
  apply (integral_pos_iff_support_of_nonneg (fun y => (Real.exp_pos _).le)
    (hval.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _))).mpr
  have hs : Function.support (fun y : SSphere d => Real.exp
      (inner (𝕜 := ℝ) x (B (y : EucSpace d)))) = univ := by
    ext y
    simp only [Function.mem_support, mem_univ, iff_true]
    exact (Real.exp_pos _).ne'
  change 0 < (μ : Measure (SSphere d)) _
  rw [hs]
  simp

/-- The logarithm of the actual positive weight is analytic throughout
the ambient space. Source: arXiv:2601.21366v2, §6, the logarithmic
first variation in `H_log` in the proof of `prop: unified.log`.
Positivity is proved for the original integral at every point. -/
theorem analyticOnNhd_log_attentionWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) :
    AnalyticOnNhd ℝ (fun x => Real.log (attentionWeightMap B μ x)) univ := by
  intro x hx
  exact (analyticAt_log (attentionWeightMap_pos B μ x)).comp
    (analyticOnNhd_attentionWeightMap B μ x hx)

end Transformer.Perceptron
