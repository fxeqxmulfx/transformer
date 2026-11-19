module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound.Scalar

/-!
# Complete complex column Gram energy and its explicit upper bound

Source: `review/convex/moments.md`, CM108. Every matrix entry and both state
coordinates contribute to the actual complex product. No eigenvector is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound

/-- CM108: complete first column energy. -/
noncomputable def left (m : Pair.Matrix) : ℝ := ‖m.1.1‖ ^ 2 + ‖m.2.1‖ ^ 2

/-- CM108: complete second column energy. -/
noncomputable def right (m : Pair.Matrix) : ℝ := ‖m.1.2‖ ^ 2 + ‖m.2.2‖ ^ 2

/-- CM108: retain the complete complex cross-column coefficient. -/
noncomputable def cross (m : Pair.Matrix) : ℂ :=
  (starRingEnd ℂ) m.1.1 * m.1.2 + (starRingEnd ℂ) m.2.1 * m.2.2

/-- CM108: the explicit squared operator bound, computed from all Gram entries. -/
noncomputable def energyBound (m : Pair.Matrix) : ℝ := top (left m) (right m) ‖cross m‖

/-- CM108: the proposed full-matrix operator radius. -/
noncomputable def radius (m : Pair.Matrix) : ℝ := Real.sqrt (energyBound m)

/-- CM108: column energies are nonnegative, including zero columns. -/
theorem left_nonneg (m : Pair.Matrix) : 0 ≤ left m := add_nonneg (sq_nonneg _) (sq_nonneg _)

/-- CM108: the second full column is retained even when it vanishes. -/
theorem right_nonneg (m : Pair.Matrix) : 0 ≤ right m := add_nonneg (sq_nonneg _) (sq_nonneg _)

/-- CM108: the numerical energy bound is nonnegative for every full matrix. -/
theorem energyBound_nonneg (m : Pair.Matrix) : 0 ≤ energyBound m :=
  top_nonneg _ _ _ (left_nonneg m)

/-- CM108: the proposed radius is nonnegative. -/
theorem radius_nonneg (m : Pair.Matrix) : 0 ≤ radius m := Real.sqrt_nonneg _

/-- CM108: the radius realizes the exact numerical squared bound. -/
theorem radius_squared (m : Pair.Matrix) : radius m ^ 2 = energyBound m :=
  Real.sq_sqrt (energyBound_nonneg m)

/-- CM108: actual execution has the full complex Gram energy, not a real-only surrogate. -/
theorem energy_identity (m : Pair.Matrix) (v : State) :
    ‖operator m v‖ ^ 2 = left m * ‖v.ofLp.1‖ ^ 2 + right m * ‖v.ofLp.2‖ ^ 2 +
      2 * (cross m * (starRingEnd ℂ) v.ofLp.1 * v.ofLp.2).re := by
  rw [norm_squared, operator_apply]
  simp only [left, right, cross, Complex.sq_norm, Complex.normSq_apply,
    Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im,
    Complex.conj_re, Complex.conj_im]
  ring

/-- CM108: the entire complex cross term obeys the ordinary norm inequality. -/
theorem cross_bound (m : Pair.Matrix) (v : State) :
    (cross m * (starRingEnd ℂ) v.ofLp.1 * v.ofLp.2).re ≤
      ‖cross m‖ * ‖v.ofLp.1‖ * ‖v.ofLp.2‖ := by
  calc
    _ ≤ ‖cross m * (starRingEnd ℂ) v.ofLp.1 * v.ofLp.2‖ :=
      (le_abs_self _).trans (Complex.abs_re_le_norm _)
    _ = _ := by simp only [norm_mul, Complex.norm_conj]

/-- CM108: the closed numerical bound controls every actual complex-state energy. -/
theorem energy_le (m : Pair.Matrix) (v : State) :
    ‖operator m v‖ ^ 2 ≤ energyBound m * ‖v‖ ^ 2 := by
  have hc := cross_bound m v
  have hp : 0 ≤ energyBound m - left m := sub_nonneg.mpr (top_ge_left _ _ _)
  have hr : 0 ≤ energyBound m - right m := sub_nonneg.mpr (top_ge_right _ _ _)
  have he : ‖cross m‖ ^ 2 = (energyBound m - left m) * (energyBound m - right m) :=
    (gap_product _ _ _).symm
  have hw := weighted_cross _ _ _ ‖v.ofLp.1‖ ‖v.ofLp.2‖ hp hr he
  rw [energy_identity, norm_squared v]
  nlinarith

/-- CM108: convert the proved energy inequality to an actual execution norm bound. -/
theorem output_le (m : Pair.Matrix) (v : State) : ‖operator m v‖ ≤ radius m * ‖v‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (radius_nonneg m) (norm_nonneg v))).mp
  rw [mul_pow, radius_squared]
  exact energy_le m v

/-- CM108: the formula bounds the actual continuous operator on the Hilbert state. -/
theorem operator_le (m : Pair.Matrix) : ‖operator m‖ ≤ radius m :=
  ContinuousLinearMap.opNorm_le_bound _ (radius_nonneg m) (output_le m)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound
