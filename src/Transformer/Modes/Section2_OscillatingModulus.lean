/-
# The actual modulus of the signed oscillating field

Source: arXiv:2412.09080v3, §2.1, hypothesis four of
`thm:kac-rice`. The explicit field has derivative amplitude
bounded independently of its signed scale, while its second
derivative is bounded by the inverse absolute scale.

The mean value theorem bounds its derivative modulus by
`η / |r|` on the unit disk. Thus a modulus exceeding a fixed
positive tolerance forces the scale into an interval of length
proportional to `η`. This is the deterministic input for the
printed probability bound `O(η)` under a uniform signed scale.
The event is nonempty for a genuine smooth moving path.

The actual field and derivative observations are measurable
in both random parameters at every fixed time. These facts
supply the measurability required for their finite moments
under the probability law, which is defined separately.
-/

import Transformer.Modes.Section2_OscillatingField

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The exact second time derivative of the signed oscillating field.
Source: arXiv:2412.09080v3, §2.1, the explicit derivative-modulus construction testing `thm:kac-rice`. -/
noncomputable def riceCounterAcceleration (ω : ℝ × ℂ) (t : ℝ) : ℝ :=
  -(ω.1 * ω.1⁻¹) * ω.1⁻¹ * (ω.2 * riceCounterPhase ω.1 t).re

/-- The actual derivative has the displayed second derivative at every real time and signed scale.
Source: arXiv:2412.09080v3, §2.1, the actual derivative used in hypothesis four of `thm:kac-rice`. -/
theorem hasDerivAt_riceCounterDerivative (ω : ℝ × ℂ) (t : ℝ) :
    HasDerivAt (riceCounterDerivative ω) (riceCounterAcceleration ω t) t := by
  have h := Complex.imCLM.hasFDerivAt.comp_hasDerivAt t (hasDerivAt_mul_riceCounterPhase ω.2 ω.1 t)
  have hD := h.const_mul (-(ω.1 * ω.1⁻¹))
  apply hD.congr_deriv
  dsimp only [riceCounterAcceleration, Complex.imCLM_apply]
  simp only [← Complex.ofReal_inv, Complex.mul_re, Complex.mul_im, Complex.ofReal_re,
    Complex.ofReal_im, Complex.I_re, Complex.I_im]
  ring

/-- The actual second derivative is bounded by the disk radius divided by the absolute scale.
Source: arXiv:2412.09080v3, §2.1, the bound controlling the actual derivative modulus in `thm:kac-rice`. -/
theorem abs_riceCounterAcceleration_le (ω : ℝ × ℂ) (t : ℝ) :
    |riceCounterAcceleration ω t| ≤ ‖ω.2‖ * |ω.1|⁻¹ := by
  by_cases hr : ω.1 = 0
  · simp [riceCounterAcceleration, hr]
  · rw [riceCounterAcceleration, mul_inv_cancel₀ hr]
    simp only [neg_one_mul, abs_mul, abs_neg, abs_inv]
    calc
      _ ≤ |ω.1|⁻¹ * ‖ω.2‖ := mul_le_mul_of_nonneg_left
        ((Complex.abs_re_le_norm _).trans_eq (norm_mul_riceCounterPhase _ _ _)) (by positivity)
      _ = _ := mul_comm _ _

/-- The actual derivative difference obeys the global time-distance bound from the actual second derivative.
Source: arXiv:2412.09080v3, §2.1, the mean-value estimate for hypothesis four of `thm:kac-rice`. -/
theorem riceCounterDerivative_sub_bound (ω : ℝ × ℂ) (s t : ℝ) :
    |riceCounterDerivative ω t - riceCounterDerivative ω s| ≤ (‖ω.2‖ * |ω.1|⁻¹) * |t - s| := by
  have h := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (s := (Set.univ : Set ℝ)) (f := riceCounterDerivative ω) (f' := riceCounterAcceleration ω)
    (fun u hu => (hasDerivAt_riceCounterDerivative ω u).hasDerivWithinAt)
    (fun u hu => by simpa only [Real.norm_eq_abs] using abs_riceCounterAcceleration_le ω u)
    convex_univ (mem_univ s) (mem_univ t)
  simpa only [Real.norm_eq_abs] using h

/-- The original global derivative modulus is bounded by the mesh size divided by the absolute actual scale on the unit disk.
Source: arXiv:2412.09080v3, §2.1, the exact modulus definition in hypothesis four of `thm:kac-rice`. -/
theorem modulus_riceCounterDerivative_le (ω : ℝ × ℂ) (hz : ‖ω.2‖ ≤ 1) (η : ℝ) (hη : 0 ≤ η) :
    modulusOfContinuity (deriv (riceCounterField ω)) η ≤ ENNReal.ofReal (η / |ω.1|) := by
  rw [deriv_riceCounterField, modulusOfContinuity]
  refine iSup_le (fun t => iSup_le (fun s => iSup_le (fun hts => ?_)))
  apply ENNReal.ofReal_le_ofReal
  calc
    _ ≤ (‖ω.2‖ * |ω.1|⁻¹) * |t - s| := riceCounterDerivative_sub_bound ω s t
    _ ≤ (‖ω.2‖ * |ω.1|⁻¹) * η := mul_le_mul_of_nonneg_left hts (by positivity)
    _ ≤ |ω.1|⁻¹ * η := mul_le_mul_of_nonneg_right
      (by simpa only [one_mul] using mul_le_mul_of_nonneg_right hz (inv_nonneg.mpr (abs_nonneg _))) hη
    _ = _ := by rw [div_eq_mul_inv]; ring

/-- A large actual derivative modulus forces a small actual scale, with the precise ratio needed for the original `O(η)` probability bound.
Source: arXiv:2412.09080v3, §2.1, the quantitative event implication for hypothesis four of `thm:kac-rice`. -/
theorem riceCounter_large_modulus_imp_small_scale (ω : ℝ × ℂ) (hz : ‖ω.2‖ ≤ 1)
    (ε η : ℝ) (hε : 0 < ε) (hη : 0 < η)
    (hlarge : ENNReal.ofReal ε < modulusOfContinuity (deriv (riceCounterField ω)) η) :
    |ω.1| < η / ε := by
  by_cases hr : ω.1 = 0
  · simpa only [hr, abs_zero] using div_pos hη hε
  have habs : 0 < |ω.1| := abs_pos.mpr hr
  have h := (ENNReal.ofReal_lt_ofReal_iff (div_pos hη habs)).mp
    (hlarge.trans_le (modulus_riceCounterDerivative_le ω hz η hη.le))
  rw [lt_div_iff₀ habs] at h
  rw [lt_div_iff₀ hε]
  nlinarith

/-- The actual derivative modulus is finite at every scale and every mesh size, with its global amplitude bound.
Source: arXiv:2412.09080v3, §2.1, the global modulus in hypothesis four of `thm:kac-rice`. -/
theorem modulus_riceCounterDerivative_le_two_norm (ω : ℝ × ℂ) (η : ℝ) :
    modulusOfContinuity (deriv (riceCounterField ω)) η ≤ ENNReal.ofReal (2 * ‖ω.2‖) := by
  rw [deriv_riceCounterField, modulusOfContinuity]
  refine iSup_le (fun t => iSup_le (fun s => iSup_le (fun hts => ?_)))
  apply ENNReal.ofReal_le_ofReal
  have h := abs_add_le (riceCounterDerivative ω t) (-riceCounterDerivative ω s)
  rw [← sub_eq_add_neg, abs_neg] at h
  linarith [abs_riceCounterDerivative_le ω t, abs_riceCounterDerivative_le ω s]

/-- At each fixed real time, the actual phase is measurable in its signed scale.
Source: arXiv:2412.09080v3, §2.1, measurability of the explicit random field testing `thm:kac-rice`. -/
theorem measurable_riceCounterPhase (t : ℝ) : Measurable (fun r => riceCounterPhase r t) := by
  unfold riceCounterPhase
  exact Complex.continuous_exp.measurable.comp
    (((Complex.ofRealCLM.continuous.measurable).comp (measurable_const.div measurable_id)).mul measurable_const)

/-- Every actual field observation is measurable in the original scale and disk parameter.
Source: arXiv:2412.09080v3, §2.1, measurability of the values in hypothesis one of `thm:kac-rice`. -/
theorem measurable_riceCounterField (t : ℝ) : Measurable (fun ω : ℝ × ℂ => riceCounterField ω t) := by
  have hz := measurable_snd.mul ((measurable_riceCounterPhase t).comp measurable_fst)
  exact measurable_fst.mul (measurable_const.add (Complex.reCLM.continuous.measurable.comp hz))

/-- Every actual derivative observation is measurable in the original scale and disk parameter, including scale zero.
Source: arXiv:2412.09080v3, §2.1, measurability of the derivatives in hypothesis one of `thm:kac-rice`. -/
theorem measurable_deriv_riceCounterField (t : ℝ) : Measurable (fun ω : ℝ × ℂ => deriv (riceCounterField ω) t) := by
  simp only [deriv_riceCounterField]
  have hz := measurable_snd.mul ((measurable_riceCounterPhase t).comp measurable_fst)
  exact (measurable_fst.mul measurable_fst.inv).neg.mul (Complex.imCLM.continuous.measurable.comp hz)

/-- The large-modulus event contains a genuine smooth moving path with nonzero scale and nonzero disk coordinate.
Source: arXiv:2412.09080v3, §2.1, a simultaneous witness for the actual event implication used in hypothesis four of `thm:kac-rice`. -/
example : ‖(Complex.I / 2 : ℂ)‖ ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < Real.pi ∧
    ENNReal.ofReal (1 / 2 : ℝ) < modulusOfContinuity
      (deriv (riceCounterField ((1 / 2 : ℝ), Complex.I / 2))) Real.pi := by
  have hphase : riceCounterPhase (1 / 2) (Real.pi / 2) = -1 := by
    unfold riceCounterPhase
    have he : (Real.pi / 2) / (1 / 2 : ℝ) = Real.pi := by ring
    rw [he, Complex.exp_pi_mul_I]
  have hzero : riceCounterPhase (1 / 2) 0 = 1 := by simp [riceCounterPhase]
  have hlow : ENNReal.ofReal (1 : ℝ) ≤ modulusOfContinuity
      (deriv (riceCounterField ((1 / 2 : ℝ), Complex.I / 2))) Real.pi := by
    rw [deriv_riceCounterField, modulusOfContinuity]
    refine le_iSup_of_le 0 (le_iSup_of_le (Real.pi / 2) (le_iSup_of_le ?_ ?_))
    · rw [zero_sub, abs_neg, abs_of_pos (half_pos Real.pi_pos)]
      linarith [Real.pi_pos]
    · norm_num [riceCounterDerivative, hphase, hzero]
  refine ⟨by norm_num [norm_div], by norm_num, Real.pi_pos, ?_⟩
  exact ((ENNReal.ofReal_lt_ofReal_iff zero_lt_one).mpr (by norm_num : (1 / 2 : ℝ) < 1)).trans_le hlow

end Transformer.Modes
