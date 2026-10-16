/-
# Small signed oscillations for the printed Kac–Rice hypotheses

Source: arXiv:2412.09080v3, §2.1, `thm:kac-rice`.
The source's fourth hypothesis bounds the probability of a large
derivative modulus by `O(η)`. To check that condition, consider
`r * (2 + re (z * exp (i t/r)))`, with a signed scale `r`
and a point `z` in the unit disk. The scale and disk coordinate
are retained in the actual random field.

Each path is smooth. For every nonzero scale and every disk
coordinate of norm at most one, its sign is fixed and its value
never vanishes. Its derivative can nevertheless have a magnitude
independent of the small scale. The normalization `r * r⁻¹` in
the derivative also makes the formula valid at scale zero, where
the path is constant zero. That scale will have probability zero.

These are deterministic statements about this explicit field.
The probability law, joint density, and derivative-modulus
probability estimate are separate ingredients of the check of
the full printed Kac–Rice statement.
-/

import Transformer.Modes.Section2_KacRice
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.MeanValue

open Real Set Filter MeasureTheory

namespace Transformer.Modes

/-- The actual oscillation phase at signed scale `r`.
Source: arXiv:2412.09080v3, §2.1, the explicit field testing the modulus hypothesis of `thm:kac-rice`. -/
noncomputable def riceCounterPhase (r t : ℝ) : ℂ :=
  Complex.exp (((t / r : ℝ) : ℂ) * Complex.I)

/-- The signed small-amplitude oscillating field, with both its scale and disk parameter explicit.
Source: arXiv:2412.09080v3, §2.1, the field used to test the four hypotheses of `thm:kac-rice`. -/
noncomputable def riceCounterField (ω : ℝ × ℂ) (t : ℝ) : ℝ :=
  ω.1 * (2 + (ω.2 * riceCounterPhase ω.1 t).re)

/-- The actual derivative, including the scale-zero value.
Source: arXiv:2412.09080v3, §2.1, the derivative whose modulus appears in hypothesis four of `thm:kac-rice`. -/
noncomputable def riceCounterDerivative (ω : ℝ × ℂ) (t : ℝ) : ℝ :=
  -(ω.1 * ω.1⁻¹) * (ω.2 * riceCounterPhase ω.1 t).im

/-- The actual phase has unit norm at every time and every real scale.
Source: arXiv:2412.09080v3, §2.1, the explicit oscillating-field construction for `thm:kac-rice`. -/
theorem norm_riceCounterPhase (r t : ℝ) : ‖riceCounterPhase r t‖ = 1 :=
  Complex.norm_exp_ofReal_mul_I _

/-- The actual rotated disk coordinate keeps its original norm.
Source: arXiv:2412.09080v3, §2.1, the disk geometry of the field testing `thm:kac-rice`. -/
theorem norm_mul_riceCounterPhase (z : ℂ) (r t : ℝ) : ‖z * riceCounterPhase r t‖ = ‖z‖ := by
  rw [Complex.norm_mul, norm_riceCounterPhase, mul_one]

/-- Every actual path is smooth, including the constant path at scale zero.
Source: arXiv:2412.09080v3, §2.1, the path regularity in hypothesis one of `thm:kac-rice`. -/
theorem contDiff_riceCounterField (ω : ℝ × ℂ) : ContDiff ℝ ⊤ (riceCounterField ω) := by
  have h : ContDiff ℝ ⊤ (fun t : ℝ => ω.2 * riceCounterPhase ω.1 t) := by
    unfold riceCounterPhase
    have hreal : ContDiff ℝ ⊤ (fun t : ℝ => t / ω.1) := contDiff_id.div_const ω.1
    have hc := Complex.ofRealCLM.contDiff.comp hreal
    exact contDiff_const.mul ((hc.mul contDiff_const).cexp)
  have hre : ContDiff ℝ ⊤ (fun t : ℝ => (ω.2 * riceCounterPhase ω.1 t).re) :=
    Complex.reCLM.contDiff.comp h
  exact contDiff_const.mul (contDiff_const.add hre)

/-- The actual complex oscillation has its displayed derivative for every scale, including zero.
Source: arXiv:2412.09080v3, §2.1, differentiation of the explicit field used for `thm:kac-rice`. -/
theorem hasDerivAt_mul_riceCounterPhase (z : ℂ) (r t : ℝ) :
    HasDerivAt (fun s => z * riceCounterPhase r s)
      (z * (riceCounterPhase r t * ((r⁻¹ : ℂ) * Complex.I))) t := by
  have h := (((hasDerivAt_id t).div_const r).ofReal_comp.mul_const Complex.I).cexp
  simpa only [riceCounterPhase, one_div, id_eq, Complex.ofReal_inv] using h.const_mul z

/-- The actual real oscillating field has exactly the displayed derivative at every time and scale.
Source: arXiv:2412.09080v3, §2.1, the derivative used in hypotheses one, three, and four of `thm:kac-rice`. -/
theorem hasDerivAt_riceCounterField (ω : ℝ × ℂ) (t : ℝ) :
    HasDerivAt (riceCounterField ω) (riceCounterDerivative ω t) t := by
  have h := Complex.reCLM.hasFDerivAt.comp_hasDerivAt t (hasDerivAt_mul_riceCounterPhase ω.2 ω.1 t)
  have hr := (h.const_add 2).const_mul ω.1
  apply hr.congr_deriv
  dsimp only [riceCounterDerivative, Complex.reCLM_apply]
  simp only [← Complex.ofReal_inv, Complex.mul_re, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.I_re, Complex.I_im]
  ring

/-- The ordinary derivative is the actual explicit derivative function everywhere.
Source: arXiv:2412.09080v3, §2.1, the derivative in `thm:kac-rice`. -/
theorem deriv_riceCounterField (ω : ℝ × ℂ) : deriv (riceCounterField ω) = riceCounterDerivative ω :=
  funext (fun t => (hasDerivAt_riceCounterField ω t).deriv)

/-- A nonzero actual scale cancels from the derivative amplitude.
Source: arXiv:2412.09080v3, §2.1, the small-scale derivative behavior testing hypothesis four of `thm:kac-rice`. -/
theorem riceCounterDerivative_of_ne_zero (ω : ℝ × ℂ) (hr : ω.1 ≠ 0) (t : ℝ) :
    riceCounterDerivative ω t = -(ω.2 * riceCounterPhase ω.1 t).im := by
  rw [riceCounterDerivative, mul_inv_cancel₀ hr, neg_one_mul]

/-- The derivative amplitude is bounded by the actual disk radius, independently of the scale.
Source: arXiv:2412.09080v3, §2.1, the actual finite-variance derivative used for `thm:kac-rice`. -/
theorem abs_riceCounterDerivative_le (ω : ℝ × ℂ) (t : ℝ) : |riceCounterDerivative ω t| ≤ ‖ω.2‖ := by
  by_cases hr : ω.1 = 0
  · simp only [riceCounterDerivative, hr, zero_mul, neg_zero, abs_zero]
    exact norm_nonneg _
  · rw [riceCounterDerivative_of_ne_zero ω hr, abs_neg]
    exact (Complex.abs_im_le_norm _).trans_eq (norm_mul_riceCounterPhase _ _ _)

/-- The actual value is bounded by its signed scale times the disk-radius bound.
Source: arXiv:2412.09080v3, §2.1, the actual finite-variance field used for `thm:kac-rice`. -/
theorem abs_riceCounterField_le (ω : ℝ × ℂ) (t : ℝ) : |riceCounterField ω t| ≤ |ω.1| * (2 + ‖ω.2‖) := by
  rw [riceCounterField, abs_mul]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
  have h := Complex.abs_re_le_norm (ω.2 * riceCounterPhase ω.1 t)
  rw [norm_mul_riceCounterPhase] at h
  exact (abs_add_le _ _).trans (by rw [abs_of_pos (by norm_num : (0 : ℝ) < 2)]; linarith)

/-- A disk coordinate of radius at most one leaves a strictly positive offset in the actual oscillating field.
Source: arXiv:2412.09080v3, §2.1, the fixed-sign paths testing `thm:kac-rice`. -/
theorem riceCounterField_offset_pos (ω : ℝ × ℂ) (hz : ‖ω.2‖ ≤ 1) (t : ℝ) :
    0 < 2 + (ω.2 * riceCounterPhase ω.1 t).re := by
  have h := Complex.abs_re_le_norm (ω.2 * riceCounterPhase ω.1 t)
  rw [norm_mul_riceCounterPhase] at h
  have hlo := neg_abs_le (ω.2 * riceCounterPhase ω.1 t).re
  linarith

/-- Every actual nonzero-scale path in the unit disk avoids level zero at every real time.
Source: arXiv:2412.09080v3, §2.1, the actual level-zero behavior of the field testing `thm:kac-rice`. -/
theorem riceCounterField_ne_zero (ω : ℝ × ℂ) (hr : ω.1 ≠ 0) (hz : ‖ω.2‖ ≤ 1) (t : ℝ) :
    riceCounterField ω t ≠ 0 :=
  mul_ne_zero hr (riceCounterField_offset_pos ω hz t).ne'

/-- Such an actual path has no original upcrossings of zero on any time set.
Source: arXiv:2412.09080v3, §2.1, the original upcrossing definition in `thm:kac-rice`. -/
theorem riceCounterField_upcrossingCount_eq_zero (ω : ℝ × ℂ) (hr : ω.1 ≠ 0) (hz : ‖ω.2‖ ≤ 1) (T : Set ℝ) :
    upcrossingCount (riceCounterField ω) 0 T = 0 := by
  have he : upcrossingSet (riceCounterField ω) 0 T = ∅ := by
    ext t
    simp only [upcrossingSet, mem_ofPred_eq, IsUpcrossing, mem_empty_iff_false, iff_false]
    exact fun ht => riceCounterField_ne_zero ω hr hz t ht.2.1
  rw [upcrossingCount, he]
  simp

/-- A nonzero scale and a nonzero disk coordinate jointly realize the smooth fixed-sign path and an actual nonzero derivative.
Source: arXiv:2412.09080v3, §2.1, a genuine moving-path witness for the oscillating construction testing `thm:kac-rice`. -/
example : (1 / 2 : ℝ) ≠ 0 ∧ ‖(Complex.I / 2 : ℂ)‖ ≤ 1 ∧
    HasDerivAt (riceCounterField ((1 / 2 : ℝ), Complex.I / 2)) (-1 / 2) 0 ∧
    upcrossingCount (riceCounterField ((1 / 2 : ℝ), Complex.I / 2)) 0 (Icc 0 1) = 0 := by
  have hr : (1 / 2 : ℝ) ≠ 0 := by norm_num
  have hz : ‖(Complex.I / 2 : ℂ)‖ ≤ 1 := by norm_num [norm_div]
  refine ⟨hr, hz, ?_, riceCounterField_upcrossingCount_eq_zero _ hr hz _⟩
  have h := hasDerivAt_riceCounterField ((1 / 2 : ℝ), Complex.I / 2) 0
  norm_num [riceCounterDerivative, riceCounterPhase] at h
  simpa only [neg_div] using h

end Transformer.Modes
