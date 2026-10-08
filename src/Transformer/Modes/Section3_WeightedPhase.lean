import Transformer.Modes.Section3_PhaseTaylor

/-!
# Weighted phase remainders for characteristic derivatives

The derivatives in arXiv:2412.09080v3, §5.4 `eq:br-9.10`, are phase
integrals with linear or quadratic polynomial weights. Integrating the
phase expansion with those weights supplies the finite-third-moment
remainders needed for the first two derivative comparisons.

The linear expansion has a quadratic phase remainder, and the constant
expansion has a linear phase remainder. Their products with the first
and second derivative weights respectively require third moments.
No exponential moment is needed: `exp(iu)` has modulus one.

The statements allow an arbitrary real integrable weight and measure.
Integrability of the weighted polynomial justifies subtraction of its
actual integral; integrability of the weighted absolute remainder
controls the resulting error. Later applications provide these inputs
from the actual law's finite moments and use its standardization to
compute the polynomial integrals.
-/

open Real MeasureTheory Filter
open scoped ENNReal
namespace Transformer.Modes

/-- The constant phase expansion has a linear remainder for every
real frequency. Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
the unscaled second-derivative Taylor step. -/
theorem norm_complexExp_imaginary_sub_one (u : ℝ) :
    ‖Complex.exp ((u : ℂ) * Complex.I) - 1‖ ≤ |u| := by
  have h := norm_complexExp_imaginary_taylor u 0
  norm_num [Finset.sum_range_succ] at h
  exact h

/-- The linear phase expansion has a quadratic remainder for every
real frequency. The constant is nonsharp. Source: arXiv:2412.09080v3,
§5.4 `eq:br-9.10`, the unscaled first-derivative Taylor step. -/
theorem norm_complexExp_imaginary_sub_linear (u : ℝ) :
    ‖Complex.exp ((u : ℂ) * Complex.I) - (1 + (u : ℂ) * Complex.I)‖ ≤ |u| ^ 2 := by
  have h := norm_complexExp_imaginary_taylor u 1
  norm_num [Finset.sum_range_succ] at h
  simpa only [sq_abs] using h

/-- An integrable real weight remains integrable after multiplying
by a continuous unit phase. Source: arXiv:2412.09080v3, §5.4,
the weighted phase integrals for characteristic derivatives. -/
theorem integrable_real_weighted_phase (μ : Measure (ℝ × ℝ)) (w f : ℝ × ℝ → ℝ)
    (hf : Continuous f) (hw : Integrable w μ) :
    Integrable (fun z => (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I)) μ := by
  have hphase : Continuous (fun z => Complex.exp ((f z : ℂ) * Complex.I)) := by fun_prop
  exact hw.ofReal.mul_bdd hphase.aestronglyMeasurable
    (ae_of_all _ fun z => by rw [Complex.norm_exp_ofReal_mul_I])

example : Continuous (fun _ : ℝ × ℝ => (1 : ℝ)) ∧
    Integrable (fun _ : ℝ × ℝ => (1 : ℝ)) stdGauss2 :=
  ⟨continuous_const, integrable_const _⟩

/-- A real weight times the constant phase remainder is controlled
by its weighted absolute first phase moment. Source:
arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the second-derivative
remainder before normalization and Gaussian damping. -/
theorem norm_integral_real_weighted_phase_sub_one
    (μ : Measure (ℝ × ℝ)) (w f : ℝ × ℝ → ℝ) (hf : Continuous f)
    (hw : Integrable w μ) (hmom : Integrable (fun z => |w z| * |f z|) μ) :
    ‖(∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
      ((∫ z, w z ∂μ : ℝ) : ℂ)‖ ≤ ∫ z, |w z| * |f z| ∂μ := by
  have hE := integrable_real_weighted_phase μ w f hf hw
  have hs : (∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
      ((∫ z, w z ∂μ : ℝ) : ℂ) =
      ∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) - (w z : ℂ) ∂μ := by
    rw [← integral_complex_ofReal]
    exact (integral_sub hE hw.ofReal).symm
  rw [hs]
  have he : (fun z => (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) - (w z : ℂ)) =
      fun z => (w z : ℂ) * (Complex.exp ((f z : ℂ) * Complex.I) - 1) := by
    funext z
    ring
  rw [he]
  apply (norm_integral_le_integral_norm _).trans
  apply integral_mono_of_nonneg (ae_of_all _ fun z => norm_nonneg _) hmom
  exact ae_of_all _ fun z => by
    dsimp only
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_left (norm_complexExp_imaginary_sub_one (f z)) (abs_nonneg _)

example : Continuous (fun _ : ℝ × ℝ => (1 : ℝ)) ∧
    Integrable (fun _ : ℝ × ℝ => (1 : ℝ)) stdGauss2 ∧
    Integrable (fun _ : ℝ × ℝ => |(1 : ℝ)| * |(1 : ℝ)|) stdGauss2 :=
  ⟨continuous_const, integrable_const _, integrable_const _⟩

/-- A real weight times the linear phase remainder is controlled
by its weighted absolute second phase moment. Both weighted polynomial
terms are integrable, so their integrals are the actual Taylor terms.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the first-derivative
remainder before normalization and Gaussian damping. -/
theorem norm_integral_real_weighted_phase_sub_linear
    (μ : Measure (ℝ × ℝ)) (w f : ℝ × ℝ → ℝ) (hf : Continuous f)
    (hw : Integrable w μ) (hwf : Integrable (fun z => w z * f z) μ)
    (hmom : Integrable (fun z => |w z| * |f z| ^ 2) μ) :
    ‖(∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
      (((∫ z, w z ∂μ : ℝ) : ℂ) + ((∫ z, w z * f z ∂μ : ℝ) : ℂ) * Complex.I)‖ ≤
        ∫ z, |w z| * |f z| ^ 2 ∂μ := by
  have hE := integrable_real_weighted_phase μ w f hf hw
  have hP : Integrable (fun z => (w z : ℂ) + ((w z * f z : ℝ) : ℂ) * Complex.I) μ :=
    hw.ofReal.add (hwf.ofReal.mul_const _)
  have hpoly : (∫ z, (w z : ℂ) + ((w z * f z : ℝ) : ℂ) * Complex.I ∂μ) =
      (((∫ z, w z ∂μ : ℝ) : ℂ) + ((∫ z, w z * f z ∂μ : ℝ) : ℂ) * Complex.I) := by
    rw [integral_add hw.ofReal (hwf.ofReal.mul_const _), integral_mul_const,
      integral_complex_ofReal, integral_complex_ofReal]
  rw [← hpoly, ← integral_sub hE hP]
  have he : (fun z => (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) -
      ((w z : ℂ) + ((w z * f z : ℝ) : ℂ) * Complex.I)) =
      fun z => (w z : ℂ) * (Complex.exp ((f z : ℂ) * Complex.I) -
        (1 + (f z : ℂ) * Complex.I)) := by
    funext z
    push_cast
    ring
  rw [he]
  apply (norm_integral_le_integral_norm _).trans
  apply integral_mono_of_nonneg (ae_of_all _ fun z => norm_nonneg _) hmom
  exact ae_of_all _ fun z => by
    dsimp only
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_left (norm_complexExp_imaginary_sub_linear (f z)) (abs_nonneg _)

example : Continuous (fun _ : ℝ × ℝ => (1 : ℝ)) ∧
    Integrable (fun _ : ℝ × ℝ => (1 : ℝ)) stdGauss2 ∧
    Integrable (fun _ : ℝ × ℝ => (1 : ℝ) * 1) stdGauss2 ∧
    Integrable (fun _ : ℝ × ℝ => |(1 : ℝ)| * |(1 : ℝ)| ^ 2) stdGauss2 :=
  ⟨continuous_const, integrable_const _, integrable_const _, integrable_const _⟩

/-- Unit phase bounds any weighted characteristic integral by the
absolute weight integral. Source: arXiv:2412.09080v3, §5.4
`eq:big-z-exp`, the derivative moment factors. The inequality also
holds under Bochner's convention for nonintegrable functions; the
quantitative remainder applications above have proved integrability. -/
theorem norm_integral_real_weighted_phase_le (μ : Measure (ℝ × ℝ)) (w f : ℝ × ℝ → ℝ) :
    ‖∫ z, (w z : ℂ) * Complex.exp ((f z : ℂ) * Complex.I) ∂μ‖ ≤ ∫ z, |w z| ∂μ := by
  apply (norm_integral_le_integral_norm _).trans_eq
  apply integral_congr_ae
  exact ae_of_all _ fun z => by
    dsimp only
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, Complex.norm_exp_ofReal_mul_I, mul_one]

example := norm_integral_real_weighted_phase_le stdGauss2
  (fun z => z.1) (fun z => z.2)

end Transformer.Modes
