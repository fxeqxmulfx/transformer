module

public import Transformer.Homogenized.Section7_GaussianStein

/-!
# Gaussian integration by parts with polynomial growth

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and the paragraph after
`eq:interpolation_interm`. The bilinear scores used there give polynomially
growing derivatives, even though the softmax products themselves are bounded.
This module proves the scalar integration-by-parts identity with polynomial
growth of both the original test and its actual derivative.

The source's unrestricted `C²` interpolation statement needs growth control.
Here the actual Gaussian density dominates every fixed polynomial: its
weighted moments are integrable and its weighted boundary terms vanish.
These facts justify improper integration by parts directly, including the
original Dirac law when the variance is zero. The bounds refer to the test
and its actual derivative, rather than an assumed integration identity.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped Topology NNReal

namespace Transformer.Homogenized

/-- Every actual absolute polynomial density moment is integrable.
Source: arXiv:2604.01978v1, §7, polynomial domination following
`eq:interpolation_interm`. -/
theorem integrable_abs_pow_centered_gaussianPDFReal (v : ℝ≥0) (p : ℕ) :
    Integrable (fun x : ℝ => |x| ^ p * gaussianPDFReal 0 v x) := by
  by_cases hv : v = 0
  · subst v; simp
  have hvpos : (0 : ℝ) < v := by exact_mod_cast (pos_iff_ne_zero.mpr hv)
  have h := (integrable_rpow_mul_exp_neg_mul_sq
    (by positivity : (0 : ℝ) < (2 * (v : ℝ))⁻¹)
    (show -(1 : ℝ) < (p : ℝ) from lt_of_lt_of_le (by norm_num) (Nat.cast_nonneg p))).norm.const_mul
      ((Real.sqrt (2 * π * (v : ℝ)))⁻¹)
  convert h using 1
  ext x
  rw [gaussianPDFReal_def]
  simp only [sub_zero, Real.rpow_natCast, norm_mul, Real.norm_eq_abs,
    abs_pow, abs_of_pos (Real.exp_pos _)]
  have he : -x ^ 2 / (2 * (v : ℝ)) = -(2 * (v : ℝ))⁻¹ * x ^ 2 := by ring
  rw [he]
  ring

/-- Multiplying any fixed polynomial growth by the actual Gaussian
density gives an integrable function. Source: arXiv:2604.01978v1,
§7, the finite expectations used after `eq:interpolation_interm`. -/
theorem integrable_polynomial_mul_centered_gaussianPDFReal (v : ℝ≥0)
    (g : ℝ → ℝ) (hg : Continuous g) (C : ℝ) (p : ℕ)
    (hbound : ∀ x, ‖g x‖ ≤ C * (1 + |x| ^ p)) :
    Integrable (fun x => g x * gaussianPDFReal 0 v x) := by
  have h := ((integrable_gaussianPDFReal 0 v).add
    (integrable_abs_pow_centered_gaussianPDFReal v p)).const_mul C
  have hd : Continuous (gaussianPDFReal 0 v) := continuous_iff_continuousAt.mpr
    (fun x => (hasDerivAt_centered_gaussianPDFReal v x).continuousAt)
  apply h.mono' (hg.mul hd).aestronglyMeasurable
  apply ae_of_all
  intro x
  change ‖g x * gaussianPDFReal 0 v x‖ ≤ C *
    (gaussianPDFReal 0 v x + |x| ^ p * gaussianPDFReal 0 v x)
  rw [norm_mul, Real.norm_of_nonneg (gaussianPDFReal_nonneg 0 v x)]
  convert mul_le_mul_of_nonneg_right (hbound x) (gaussianPDFReal_nonneg 0 v x) using 1; ring

/-- Polynomial weighting leaves the actual Gaussian density zero at
both ends of the real line. Source: arXiv:2604.01978v1, §7, the
boundary term for the polynomially growing derivatives of softmax scores. -/
theorem tendsto_abs_pow_centered_gaussianPDFReal_cocompact (v : ℝ≥0) (p : ℕ) :
    Tendsto (fun x : ℝ => |x| ^ p * gaussianPDFReal 0 v x)
      (cocompact ℝ) (𝓝 0) := by
  by_cases hv : v = 0
  · subst v; simp
  have hvpos : (0 : ℝ) < v := by exact_mod_cast (pos_iff_ne_zero.mpr hv)
  have h := (tendsto_rpow_abs_mul_exp_neg_mul_sq_cocompact
    (by positivity : (0 : ℝ) < (2 * (v : ℝ))⁻¹) (p : ℝ)).const_mul
      ((Real.sqrt (2 * π * (v : ℝ)))⁻¹)
  convert h using 1
  · ext x
    rw [gaussianPDFReal_def]
    simp only [sub_zero, Real.rpow_natCast]
    have he : -x ^ 2 / (2 * (v : ℝ)) = -(2 * (v : ℝ))⁻¹ * x ^ 2 := by ring
    rw [he]
    ring
  · simp

/-- The boundary term of the original polynomially growing test vanishes.
Source: arXiv:2604.01978v1, §7, integration by parts for
`eq:gaussian_interp_0`, with the polynomial growth used in its application. -/
theorem tendsto_polynomial_mul_centered_gaussianPDFReal (v : ℝ≥0)
    (g : ℝ → ℝ) (C : ℝ) (p : ℕ) (hbound : ∀ x, ‖g x‖ ≤ C * (1 + |x| ^ p)) :
    Tendsto (fun x => g x * gaussianPDFReal 0 v x) (cocompact ℝ) (𝓝 0) := by
  apply squeeze_zero_norm (a := fun x =>
    C * (gaussianPDFReal 0 v x + |x| ^ p * gaussianPDFReal 0 v x))
  · intro x
    rw [norm_mul, Real.norm_of_nonneg (gaussianPDFReal_nonneg 0 v x)]
    convert mul_le_mul_of_nonneg_right (hbound x) (gaussianPDFReal_nonneg 0 v x) using 1; ring
  · simpa only [zero_add, mul_zero] using ((tendsto_centered_gaussianPDFReal_cocompact v).add
      (tendsto_abs_pow_centered_gaussianPDFReal_cocompact v p)).const_mul C

/-- Scalar Stein identity with polynomial growth of the actual test
and derivative. Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`;
the growth condition justifies the density argument used in its softmax
application. The original variance-zero law is included. -/
theorem centered_gaussian_stein_polynomial (v : ℝ≥0) (g g' : ℝ → ℝ)
    (hg : Continuous g) (hg' : Continuous g') (hderiv : ∀ x, HasDerivAt g (g' x) x)
    (C C' : ℝ) (p : ℕ) (hbound : ∀ x, ‖g x‖ ≤ C * (1 + |x| ^ p))
    (hbound' : ∀ x, ‖g' x‖ ≤ C' * (1 + |x| ^ p)) :
    (∫ x, x * g x ∂gaussianReal 0 v) = (v : ℝ) * ∫ x, g' x ∂gaussianReal 0 v := by
  by_cases hv : v = 0
  · subst v; simp
  have hvreal : (v : ℝ) ≠ 0 := by exact_mod_cast hv
  have hfirst := integrable_polynomial_mul_centered_gaussianPDFReal v g' hg' C' p hbound'
  have hsecond : Integrable (fun x => g x * (-(x / (v : ℝ)) * gaussianPDFReal 0 v x)) := by
    have hweight : ∀ x : ℝ, ‖x * g x‖ ≤ (2 * |C|) * (1 + |x| ^ (p + 1)) := by
      intro x
      have hC : C ≤ |C| := le_abs_self C
      have hx : 0 ≤ |x| := abs_nonneg x
      have ha : |x| ≤ 1 + |x| ^ (p + 1) := by
        by_cases h : |x| ≤ 1
        · exact h.trans (le_add_of_nonneg_right (pow_nonneg hx _))
        · exact (le_self_pow₀ (le_of_not_ge h) (by omega : p + 1 ≠ 0)).trans
            (le_add_of_nonneg_left zero_le_one)
      rw [norm_mul, Real.norm_eq_abs]
      calc
        _ ≤ |x| * (|C| * (1 + |x| ^ p)) := mul_le_mul_of_nonneg_left ((hbound x).trans
          (mul_le_mul_of_nonneg_right hC (by positivity))) hx
        _ = |C| * (|x| + |x| ^ (p + 1)) := by rw [pow_succ]; ring
        _ ≤ |C| * (2 * (1 + |x| ^ (p + 1))) := mul_le_mul_of_nonneg_left
          (by linarith [pow_nonneg hx (p + 1)]) (abs_nonneg C)
        _ = _ := by ring
    have h := (integrable_polynomial_mul_centered_gaussianPDFReal v
      (fun x => x * g x) (continuous_id.mul hg) (2 * |C|) (p + 1) hweight).const_mul (-(v : ℝ)⁻¹)
    convert h using 1
    ext x; ring
  have htail := tendsto_polynomial_mul_centered_gaussianPDFReal v g C p hbound
  have hparts := integral_deriv_mul_eq_sub (u := g) (u' := g') (v := gaussianPDFReal 0 v)
    (v' := fun x => -(x / (v : ℝ)) * gaussianPDFReal 0 v x)
    (fun x hx => hderiv x) (fun x hx => hasDerivAt_centered_gaussianPDFReal v x)
    (hfirst.add hsecond) (htail.mono_left atBot_le_cocompact) (htail.mono_left atTop_le_cocompact)
  have hs : (∫ x, g x * (-(x / (v : ℝ)) * gaussianPDFReal 0 v x)) =
      -(v : ℝ)⁻¹ * ∫ x, gaussianPDFReal 0 v x * (x * g x) := by
    rw [← integral_const_mul]
    apply integral_congr_ae
    filter_upwards with x
    ring
  rw [integral_add hfirst hsecond, hs, sub_self] at hparts
  rw [integral_gaussianReal_eq_integral_smul hv, integral_gaussianReal_eq_integral_smul hv]
  simp only [smul_eq_mul]
  have hc := congrArg (fun r : ℝ => (v : ℝ) * r) hparts
  simp only [mul_add, ← mul_assoc, mul_neg, mul_inv_cancel₀ hvreal, neg_one_mul, mul_zero] at hc
  have hf : (∫ x, g' x * gaussianPDFReal 0 v x) = ∫ x, gaussianPDFReal 0 v x * g' x := by
    apply integral_congr_ae
    filter_upwards with x
    ring
  rw [hf] at hc
  simp only [mul_assoc] at hc
  linarith

/-- An unbounded linear test and its actual constant derivative witness
the polynomial hypotheses, under a genuine positive-variance Gaussian. -/
example : (1 : ℝ≥0) ≠ 0 ∧ Continuous (fun x : ℝ => x) ∧ Continuous (fun _ : ℝ => (1 : ℝ)) ∧
    (∀ x : ℝ, HasDerivAt (fun y : ℝ => y) 1 x) ∧
    (∀ x : ℝ, ‖x‖ ≤ 1 * (1 + |x| ^ 1)) ∧
    (∀ x : ℝ, ‖(1 : ℝ)‖ ≤ 1 * (1 + |x| ^ 1)) := by
  exact ⟨one_ne_zero, continuous_id, continuous_const, hasDerivAt_id,
    fun x => by simp [Real.norm_eq_abs], fun x => by simp [abs_nonneg x]⟩

end Transformer.Homogenized
