module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Basic

/-!
# Complete complex coordinate resizing and Euclidean row distance

Source: `review/flat_spectrum.md`, CM102. Both real and imaginary coordinates
retain their phase. The Euclidean metric here is distinct from the maximum
norm on recurrent states; the original row capacity remains the absolute sum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection

open Pair

/-- CM102: resize one full complex coordinate, with total zero division. -/
noncomputable def resize (z : ℂ) (s : ℝ) : ℂ := (s / ‖z‖) • z

/-- CM102: resizing to the original magnitude preserves the complete coordinate. -/
theorem resize_self (z : ℂ) : resize z ‖z‖ = z := by
  by_cases hz : z = 0
  · simp [resize, hz]
  · rw [resize, div_self (ne_of_gt (norm_pos_iff.mpr hz)), one_smul]

/-- CM102: a zero proposed magnitude remains a stored zero coordinate. -/
theorem resize_zero (z : ℂ) : resize z 0 = 0 := by simp [resize]

/-- CM102: a valid smaller magnitude is realized exactly, including a zero input. -/
theorem resize_norm (z : ℂ) (s : ℝ) (hs : 0 ≤ s) (hr : s ≤ ‖z‖) :
    ‖resize z s‖ = s := by
  by_cases hz : z = 0
  · subst z
    have hr' : s ≤ 0 := by simpa using hr
    have he : s = 0 := le_antisymm hr' hs
    simp [he, resize]
  · rw [resize, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (div_nonneg hs (norm_nonneg z)),
      div_mul_cancel₀ _ (ne_of_gt (norm_pos_iff.mpr hz))]

/-- CM102: phase preservation attains the reverse-triangle distance bound. -/
theorem resize_distance (z : ℂ) (s : ℝ) (hs : 0 ≤ s) (hr : s ≤ ‖z‖) :
    ‖z - resize z s‖ = ‖z‖ - s := by
  by_cases hz : z = 0
  · subst z
    have hr' : s ≤ 0 := by simpa using hr
    have he : s = 0 := le_antisymm hr' hs
    simp [he, resize]
  · have hp := norm_pos_iff.mpr hz
    have ht : 0 ≤ 1 - s / ‖z‖ := sub_nonneg.mpr ((div_le_one hp).mpr hr)
    have he : z - resize z s = (1 - s / ‖z‖) • z := by
      rw [resize, sub_smul, one_smul]
    rw [he, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht, sub_mul, one_mul,
      div_mul_cancel₀ _ (ne_of_gt hp)]

/-- CM102: arbitrary complex phases cannot improve the magnitude-distance bound. -/
theorem magnitude_distance_le (z v : ℂ) : (‖z‖ - ‖v‖) ^ 2 ≤ ‖z - v‖ ^ 2 := by
  apply sq_le_sq.mpr
  simpa only [abs_of_nonneg (norm_nonneg (z - v))] using abs_norm_sub_norm_le z v

/-- CM102: squared Euclidean distance on all four real row coordinates. -/
noncomputable def squaredDistance (u v : State) : ℝ :=
  ‖u.1 - v.1‖ ^ 2 + ‖u.2 - v.2‖ ^ 2

/-- CM102: this metric is exactly squared Euclidean distance on all four real coordinates. -/
theorem distance_coordinates (u v : State) :
    squaredDistance u v =
      (u.1.re - v.1.re) ^ 2 + (u.1.im - v.1.im) ^ 2 +
      (u.2.re - v.2.re) ^ 2 + (u.2.im - v.2.im) ^ 2 := by
  simp only [squaredDistance, Complex.sq_norm, Complex.normSq_apply,
    Complex.sub_re, Complex.sub_im]
  ring

/-- CM102: every full row distance is nonnegative. -/
theorem squaredDistance_nonneg (u v : State) : 0 ≤ squaredDistance u v :=
  add_nonneg (sq_nonneg _) (sq_nonneg _)

example : ‖resize Complex.I (1 / 2)‖ = 1 / 2 :=
  resize_norm _ _ (by norm_num) (by norm_num)

example : ‖Complex.I - resize Complex.I (1 / 2)‖ = 1 - 1 / 2 := by
  simpa using resize_distance Complex.I (1 / 2) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection
