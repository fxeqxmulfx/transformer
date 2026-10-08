import Mathlib.Analysis.SpecialFunctions.Gaussian.FourierTransform
import Mathlib.LinearAlgebra.Complex.FiniteDimensional
import Mathlib.MeasureTheory.Integral.Prod

/-!
# Gaussian regularization before density inversion

Section 3 of arXiv:2412.09080v3 uses continuous densities of normalized
sums, and §5.4 invokes Fourier inversion. The original fixed-five-summands
argument does not establish such a density for every positive bandwidth.
The corrected integrable-power bound has already been proved separately.
To turn that bound into an actual density, this module constructs Gaussian
regularizations of a probability law without assuming absolute continuity.

The kernel is `π c exp(-π² c |z|²)` for `c > 0`. Its integral is one in two
real dimensions. Integrating its translates against the actual law gives
a nonnegative continuous integrable function, also with integral one.
The joint integrability needed for Fubini is proved from translation
invariance and the exact kernel mass, including for singular input laws.

We use `ℂ` as a two-dimensional real inner product space so that the
library's Fourier inversion theorem applies. The volume-preserving
coordinate equivalence with `ℝ × ℝ` is used when recovering the density
of the paper's actual normalized sum. No density is a hypothesis here.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4, Fourier inversion.
-/

open Real MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The two-dimensional Gaussian kernel `π c exp(-π² c |z|²)`.
The coordinate space `ℂ` carries the Euclidean norm and the same Lebesgue
measure as `ℝ × ℝ`. Source: arXiv:2412.09080v3, §3 and §5.4,
Fourier inversion; the Gaussian regularization is supplied explicitly. -/
noncomputable def gaussianHeatKernel2 (c : ℝ) (z : ℂ) : ℝ :=
  Real.pi * c * Real.exp (-(Real.pi ^ 2 * c) * ‖z‖ ^ 2)

/-- The Gaussian regularizer is continuous at every point.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem continuous_gaussianHeatKernel2 (c : ℝ) :
    Continuous (gaussianHeatKernel2 c) := by
  unfold gaussianHeatKernel2
  fun_prop

/-- The regularizing kernel is nonnegative for a positive scale.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem gaussianHeatKernel2_nonneg {c : ℝ} (hc : 0 < c) (z : ℂ) :
    0 ≤ gaussianHeatKernel2 c z := by
  unfold gaussianHeatKernel2
  positivity

example : (0 : ℝ) < 1 := by norm_num

/-- The Gaussian kernel is bounded by its value at the origin.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem gaussianHeatKernel2_le {c : ℝ} (hc : 0 < c) (z : ℂ) :
    gaussianHeatKernel2 c z ≤ Real.pi * c := by
  have h : -(Real.pi ^ 2 * c) * ‖z‖ ^ 2 ≤ 0 := by
    exact mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (by positivity)) (sq_nonneg _)
  unfold gaussianHeatKernel2
  exact mul_le_of_le_one_right (by positivity) (Real.exp_le_one_iff.mpr h)

example : (0 : ℝ) < 1 := by norm_num

/-- The Gaussian regularizer is integrable for every positive scale.
Source: arXiv:2412.09080v3, §5.4, Fourier inversion. -/
theorem integrable_gaussianHeatKernel2 {c : ℝ} (hc : 0 < c) :
    Integrable (gaussianHeatKernel2 c) := by
  have h := GaussianFourier.integrable_cexp_neg_mul_sq_norm_add
    (V := ℂ) (b := ((Real.pi ^ 2 * c : ℝ) : ℂ))
    (by change 0 < Real.pi ^ 2 * c; positivity) 0 0
  have hr : Integrable (fun z : ℂ => Real.exp (-(Real.pi ^ 2 * c) * ‖z‖ ^ 2)) := by
    have hr := h.re
    simp only [zero_mul, add_zero] at hr
    change Integrable (fun z : ℂ =>
      (Complex.exp (-((Real.pi ^ 2 * c : ℝ) : ℂ) * (‖z‖ : ℂ) ^ 2)).re) at hr
    simpa only [add_zero, ← Complex.ofReal_pow, ← Complex.ofReal_mul,
      ← Complex.ofReal_neg, Complex.exp_ofReal_re] using hr
  exact hr.const_mul _

example : (0 : ℝ) < 1 := by norm_num

/-- The regularizer has mass one, with the Fourier normalization
`exp(-2π i ⟨ξ,x⟩)`. Source: arXiv:2412.09080v3, §5.4. -/
theorem integral_gaussianHeatKernel2 {c : ℝ} (hc : 0 < c) :
    ∫ z, gaussianHeatKernel2 c z = 1 := by
  unfold gaussianHeatKernel2
  rw [integral_const_mul, GaussianFourier.integral_rexp_neg_mul_sq_norm (by positivity)]
  simp only [Complex.finrank_real_complex, Nat.cast_ofNat, div_self (by norm_num : (2 : ℝ) ≠ 0),
    Real.rpow_one]
  field_simp

example : (0 : ℝ) < 1 := by norm_num

/-- Gaussian smoothing of the actual law, defined by integrating the
translated kernel against that law. Source: arXiv:2412.09080v3, §3
and §5.4, the continuous-density and Fourier-inversion arguments. -/
noncomputable def gaussianRegularization2 (μ : Measure ℂ) (c : ℝ) (w : ℂ) : ℝ :=
  ∫ z, gaussianHeatKernel2 c (w - z) ∂μ

/-- A Gaussian smoothing of a measure is nonnegative.
Source: arXiv:2412.09080v3, §3 and §5.4, Fourier inversion. -/
theorem gaussianRegularization2_nonneg (μ : Measure ℂ) {c : ℝ} (hc : 0 < c) (w : ℂ) :
    0 ≤ gaussianRegularization2 μ c w := by
  exact integral_nonneg (fun z => gaussianHeatKernel2_nonneg hc (w - z))

example : IsProbabilityMeasure (Measure.dirac (0 : ℂ)) ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, by norm_num⟩

/-- The smoothing is continuous by domination by `π c`, integrable
under a probability law. Source: arXiv:2412.09080v3, §3 and §5.4. -/
theorem continuous_gaussianRegularization2 (μ : Measure ℂ) [IsProbabilityMeasure μ]
    {c : ℝ} (hc : 0 < c) : Continuous (gaussianRegularization2 μ c) := by
  unfold gaussianRegularization2
  apply continuous_of_dominated (bound := fun _ => Real.pi * c)
  · intro w
    exact ((continuous_gaussianHeatKernel2 c).comp (continuous_const.sub continuous_id)).aestronglyMeasurable
  · exact fun w => ae_of_all _ fun z => by
      rw [Real.norm_eq_abs, abs_of_nonneg (gaussianHeatKernel2_nonneg hc _)]
      exact gaussianHeatKernel2_le hc _
  · exact integrable_const _
  · exact ae_of_all _ fun z =>
      (continuous_gaussianHeatKernel2 c).comp (continuous_id.sub continuous_const)

example : IsProbabilityMeasure (Measure.dirac (0 : ℂ)) ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, by norm_num⟩

/-- Translation preserves the kernel mass and integrability. Fubini
therefore applies to the actual law and Lebesgue measure.
Source: arXiv:2412.09080v3, §3 and §5.4, Fourier inversion. -/
theorem integrable_gaussianHeatKernel2_prod (μ : Measure ℂ) [IsProbabilityMeasure μ]
    {c : ℝ} (hc : 0 < c) :
    Integrable (fun p : ℂ × ℂ => gaussianHeatKernel2 c (p.2 - p.1)) (μ.prod volume) := by
  have hcont : Continuous (fun p : ℂ × ℂ => gaussianHeatKernel2 c (p.2 - p.1)) :=
    (continuous_gaussianHeatKernel2 c).comp (continuous_snd.sub continuous_fst)
  apply (integrable_prod_iff hcont.aestronglyMeasurable).mpr
  refine ⟨ae_of_all _ fun z => (integrable_gaussianHeatKernel2 hc).comp_sub_right z, ?_⟩
  have he (z : ℂ) : (∫ w, ‖gaussianHeatKernel2 c (w - z)‖) = 1 := by
    simp_rw [Real.norm_eq_abs, abs_of_nonneg (gaussianHeatKernel2_nonneg hc _)]
    rw [integral_sub_right_eq_self, integral_gaussianHeatKernel2 hc]
  simpa only [he] using (integrable_const (1 : ℝ) : Integrable (fun _ : ℂ => (1 : ℝ)) μ)

example : IsProbabilityMeasure (Measure.dirac (0 : ℂ)) ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, by norm_num⟩

/-- The actual Gaussian smoothing is integrable, including when
the original law is singular. Source: arXiv:2412.09080v3, §3 and §5.4. -/
theorem integrable_gaussianRegularization2 (μ : Measure ℂ) [IsProbabilityMeasure μ]
    {c : ℝ} (hc : 0 < c) : Integrable (gaussianRegularization2 μ c) := by
  exact (integrable_gaussianHeatKernel2_prod μ hc).integral_prod_right

example : IsProbabilityMeasure (Measure.dirac (0 : ℂ)) ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, by norm_num⟩

/-- Gaussian smoothing retains total mass one.
Source: arXiv:2412.09080v3, §3 and §5.4, Fourier inversion. -/
theorem integral_gaussianRegularization2 (μ : Measure ℂ) [IsProbabilityMeasure μ]
    {c : ℝ} (hc : 0 < c) : ∫ w, gaussianRegularization2 μ c w = 1 := by
  unfold gaussianRegularization2
  rw [← integral_integral_swap (integrable_gaussianHeatKernel2_prod μ hc)]
  simp_rw [integral_sub_right_eq_self, integral_gaussianHeatKernel2 hc]
  simp

example : IsProbabilityMeasure (Measure.dirac (0 : ℂ)) ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, by norm_num⟩

end Transformer.Modes
