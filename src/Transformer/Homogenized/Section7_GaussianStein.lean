/-
# Actual one-dimensional Gaussian integration by parts

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`, recalling Vershynin, Lemma 7.2.7.
Gaussian interpolation differentiates the original Gaussian expectation
and integrates the resulting original coordinate factors by parts.
This module proves that scalar identity from the actual Gaussian density.

The density derivative, its decay at both ends of the real line,
and integrability of its actual first moment supply the improper
integration-by-parts hypotheses. Bounded test functions and bounded
continuous derivatives are the regularity already documented in the
Gaussian interpolation statement. No integration-by-parts estimate is
assumed. The resulting identity also includes the degenerate Gaussian
law at variance zero, where the original law is the actual Dirac mass.
-/

import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Gaussian.PoissonSummation
import Mathlib.MeasureTheory.Integral.IntegralEqImproper

open Real MeasureTheory ProbabilityTheory Filter
open scoped Topology NNReal

namespace Transformer.Homogenized

/-- The actual centered Gaussian density has its actual linear score
derivative, including its zero density at variance zero.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`, the scalar
integration-by-parts step behind `eq:gaussian_interp_0`. -/
theorem hasDerivAt_centered_gaussianPDFReal (v : ℝ≥0) (x : ℝ) :
    HasDerivAt (gaussianPDFReal 0 v) (-(x / (v : ℝ)) * gaussianPDFReal 0 v x) x := by
  by_cases hv : v = 0
  · subst v
    simp
  have hvreal : (v : ℝ) ≠ 0 := by exact_mod_cast hv
  have hquad : HasDerivAt (fun z : ℝ => -z ^ 2 / (2 * (v : ℝ))) (-(x / (v : ℝ))) x := by
    have h := (((hasDerivAt_id x).pow 2).neg.div_const (2 * (v : ℝ)))
    simp only [Pi.pow_apply, Pi.neg_apply, id_eq] at h
    convert h using 1
    norm_num
    field_simp
  rw [gaussianPDFReal_def]
  simp only [sub_zero]
  convert hquad.exp.const_mul ((Real.sqrt (2 * π * (v : ℝ)))⁻¹) using 1
  ring

/-- The original centered Gaussian density vanishes at both ends of
the real line. Source: arXiv:2604.01978v1, §7, the boundary term in
the original Gaussian integration-by-parts argument. -/
theorem tendsto_centered_gaussianPDFReal_cocompact (v : ℝ≥0) :
    Tendsto (gaussianPDFReal 0 v) (cocompact ℝ) (𝓝 0) := by
  by_cases hv : v = 0
  · subst v
    simp only [gaussianPDFReal_zero_var]
    exact tendsto_const_nhds
  have hvpos : (0 : ℝ) < v := by exact_mod_cast (pos_iff_ne_zero.mpr hv)
  have h := tendsto_rpow_abs_mul_exp_neg_mul_sq_cocompact
    (by positivity : (0 : ℝ) < (2 * (v : ℝ))⁻¹) (0 : ℝ)
  simp only [Real.rpow_zero, one_mul] at h
  have hscale := h.const_mul ((Real.sqrt (2 * π * (v : ℝ)))⁻¹)
  convert hscale using 1
  · ext x
    rw [gaussianPDFReal_def]
    simp only [sub_zero]
    congr 2
    ring
  · simp

/-- The original first density moment is absolutely integrable.
Source: arXiv:2604.01978v1, §7, the integrable original coordinate
factor in the Gaussian integration-by-parts argument. -/
theorem integrable_mul_centered_gaussianPDFReal (v : ℝ≥0) :
    Integrable (fun x : ℝ => x * gaussianPDFReal 0 v x) := by
  by_cases hv : v = 0
  · subst v; simp
  have hvpos : (0 : ℝ) < v := by exact_mod_cast (pos_iff_ne_zero.mpr hv)
  have h := (integrable_mul_exp_neg_mul_sq
    (by positivity : (0 : ℝ) < (2 * (v : ℝ))⁻¹)).const_mul
    ((Real.sqrt (2 * π * (v : ℝ)))⁻¹)
  convert h using 1
  ext x
  rw [gaussianPDFReal_def]
  simp only [sub_zero]
  have he : -x ^ 2 / (2 * (v : ℝ)) = -(2 * (v : ℝ))⁻¹ * x ^ 2 := by ring
  rw [he]
  ring

/-- A bounded original test function leaves the Gaussian boundary term
zero. Source: arXiv:2604.01978v1, §7, the actual improper
integration-by-parts boundary term in `lem:gaussian_interp`. -/
theorem tendsto_bounded_mul_centered_gaussianPDFReal (v : ℝ≥0)
    (g : ℝ → ℝ) (C : ℝ) (hbound : ∀ x, ‖g x‖ ≤ C) :
    Tendsto (fun x => g x * gaussianPDFReal 0 v x) (cocompact ℝ) (𝓝 0) := by
  apply squeeze_zero_norm (a := fun x => C * gaussianPDFReal 0 v x)
  · intro x
    rw [norm_mul, Real.norm_of_nonneg (gaussianPDFReal_nonneg 0 v x)]
    exact mul_le_mul_of_nonneg_right (hbound x) (gaussianPDFReal_nonneg 0 v x)
  · simpa only [mul_zero] using (tendsto_centered_gaussianPDFReal_cocompact v).const_mul C

/-- The actual scalar Gaussian integration-by-parts identity at positive
variance. Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
The original density, derivative, finite integrals and vanishing boundary
terms are all used; the test derivative is tied to the test function. -/
theorem centered_gaussian_stein_density (v : ℝ≥0) (hv : v ≠ 0)
    (g g' : ℝ → ℝ) (hg : Continuous g) (hg' : Continuous g')
    (hderiv : ∀ x, HasDerivAt g (g' x) x) (C C' : ℝ)
    (hbound : ∀ x, ‖g x‖ ≤ C) (hbound' : ∀ x, ‖g' x‖ ≤ C') :
    (∫ x, gaussianPDFReal 0 v x * (x * g x)) =
      (v : ℝ) * ∫ x, gaussianPDFReal 0 v x * g' x := by
  have hvreal : (v : ℝ) ≠ 0 := by exact_mod_cast hv
  have hfirst := (integrable_gaussianPDFReal 0 v).bdd_mul hg'.aestronglyMeasurable
    (ae_of_all volume hbound')
  have hbase : Integrable (fun x => -(x / (v : ℝ)) * gaussianPDFReal 0 v x) := by
    convert (integrable_mul_centered_gaussianPDFReal v).const_mul (-(v : ℝ)⁻¹) using 1
    ext x; ring
  have hsecond := hbase.bdd_mul hg.aestronglyMeasurable (ae_of_all volume hbound)
  have htail := tendsto_bounded_mul_centered_gaussianPDFReal v g C hbound
  have hparts := integral_deriv_mul_eq_sub
    (u := g) (u' := g') (v := gaussianPDFReal 0 v)
    (v' := fun x => -(x / (v : ℝ)) * gaussianPDFReal 0 v x)
    (fun x hx => hderiv x) (fun x hx => hasDerivAt_centered_gaussianPDFReal v x)
    (hfirst.add hsecond) (htail.mono_left atBot_le_cocompact) (htail.mono_left atTop_le_cocompact)
  have hsecondInt : (∫ x, g x * (-(x / (v : ℝ)) * gaussianPDFReal 0 v x)) =
      -(v : ℝ)⁻¹ * ∫ x, gaussianPDFReal 0 v x * (x * g x) := by
    rw [← integral_const_mul]
    apply integral_congr_ae
    filter_upwards with x
    ring
  rw [integral_add hfirst hsecond, hsecondInt] at hparts
  have hfirstInt : (∫ x, g' x * gaussianPDFReal 0 v x) =
      ∫ x, gaussianPDFReal 0 v x * g' x := by
    apply integral_congr_ae
    filter_upwards with x
    ring
  rw [hfirstInt, sub_self] at hparts
  have hscaled := congrArg (fun r : ℝ => (v : ℝ) * r) hparts
  have hcancel : (v : ℝ) * (-(v : ℝ)⁻¹) = -1 := by
    rw [mul_neg, mul_inv_cancel₀ hvreal]
  rw [mul_add, ← mul_assoc, hcancel, neg_one_mul, mul_zero] at hscaled
  linarith

/-- The original scalar Gaussian expectation obeys integration by parts,
including variance zero and its actual degenerate original law.
Source: arXiv:2604.01978v1, §7, the Gaussian identity used in
`lem:gaussian_interp`, with bounded test function and derivative. -/
theorem centered_gaussian_stein (v : ℝ≥0) (g g' : ℝ → ℝ)
    (hg : Continuous g) (hg' : Continuous g') (hderiv : ∀ x, HasDerivAt g (g' x) x)
    (C C' : ℝ) (hbound : ∀ x, ‖g x‖ ≤ C) (hbound' : ∀ x, ‖g' x‖ ≤ C') :
    (∫ x, x * g x ∂gaussianReal 0 v) = (v : ℝ) * ∫ x, g' x ∂gaussianReal 0 v := by
  by_cases hv : v = 0
  · subst v; simp
  rw [integral_gaussianReal_eq_integral_smul hv, integral_gaussianReal_eq_integral_smul hv]
  exact centered_gaussian_stein_density v hv g g' hg hg' hderiv C C' hbound hbound'

/-- The original centered Gaussian random variable satisfies the same
integration-by-parts identity on its original probability space.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`, its original
coordinate covariance factor. The actual Gaussian pushforward law
supplies the density identity, including degenerate coordinates. -/
theorem gaussian_stein_random_variable {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (hX : HasGaussianLaw X P)
    (hcenter : ∫ ω, X ω ∂P = 0) (g g' : ℝ → ℝ)
    (hg : Continuous g) (hg' : Continuous g') (hderiv : ∀ x, HasDerivAt g (g' x) x)
    (C C' : ℝ) (hbound : ∀ x, ‖g x‖ ≤ C) (hbound' : ∀ x, ‖g' x‖ ≤ C') :
    (∫ ω, X ω * g (X ω) ∂P) = variance X P * ∫ ω, g' (X ω) ∂P := by
  have hmap := hX.map_eq_gaussianReal
  rw [hcenter] at hmap
  calc
    _ = ∫ x, x * g x ∂P.map X :=
      (integral_map hX.aemeasurable (continuous_id.mul hg).aestronglyMeasurable).symm
    _ = (variance X P).toNNReal * ∫ x, g' x ∂P.map X := by
      rw [hmap]
      exact centered_gaussian_stein _ g g' hg hg' hderiv C C' hbound hbound'
    _ = _ := by
      rw [Real.coe_toNNReal (variance X P) (variance_nonneg X P),
        integral_map hX.aemeasurable hg'.aestronglyMeasurable]

/-- Positive variance and a nonconstant original bounded test function
jointly witness every scalar integration-by-parts hypothesis. -/
example : (1 : ℝ≥0) ≠ 0 ∧ HasGaussianLaw (fun x : ℝ => x) (gaussianReal 0 1) ∧
    (∫ x, x ∂gaussianReal 0 1) = 0 ∧ Continuous Real.sin ∧ Continuous Real.cos ∧
    (∀ x, HasDerivAt Real.sin (Real.cos x) x) ∧
    (∀ x : ℝ, ‖Real.sin x‖ ≤ 1) ∧ (∀ x : ℝ, ‖Real.cos x‖ ≤ 1) := by
  exact ⟨one_ne_zero, IsGaussian.hasGaussianLaw_id, integral_id_gaussianReal,
    Real.continuous_sin, Real.continuous_cos, Real.hasDerivAt_sin,
    fun x => by simpa only [Real.norm_eq_abs] using Real.abs_sin_le_one x,
    fun x => by simpa only [Real.norm_eq_abs] using Real.abs_cos_le_one x⟩

end Transformer.Homogenized
