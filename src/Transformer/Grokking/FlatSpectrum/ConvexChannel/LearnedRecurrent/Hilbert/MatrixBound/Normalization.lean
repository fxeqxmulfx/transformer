module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound.Attainment
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.Mixing

/-!
# A feasible full-matrix normalization retaining every unit point

Source: `review/convex/moments.md`, CM108. Scale every real and imaginary entry
by one common positive denominator. This is radial feasibility, not a claim
of closest-point projection in the Frobenius metric or optimizer convergence.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound

/-- CM108: the complete numerical normalization denominator. -/
noncomputable def denominator (m : Pair.Matrix) : ℝ := max 1 (radius m)

/-- CM108: uniformly normalize the complete matrix without discarding coordinates. -/
noncomputable def normalize (m : Pair.Matrix) : Pair.Matrix := (1 / denominator m) • m

/-- CM108: the denominator is positive even for an all-zero matrix. -/
theorem denominator_pos (m : Pair.Matrix) : 0 < denominator m :=
  lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1) (le_max_left _ _)

/-- CM108: scaling the complete stored matrix scales its actual operator. -/
theorem operator_smul (t : ℝ) (m : Pair.Matrix) : operator (t • m) = t • operator m := by
  apply ContinuousLinearMap.ext
  intro v
  apply (WithLp.ext_iff 2).mpr
  simp only [smul_apply, WithLp.ofLp_smul, operator_apply]
  ext <;> simp [Complex.real_smul] <;> ring

/-- CM108: exact norm of the actual normalized complex matrix. -/
theorem normalized_norm (m : Pair.Matrix) :
    ‖operator (normalize m)‖ = radius m / denominator m := by
  rw [normalize, operator_smul, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (div_nonneg (by norm_num) (denominator_pos m).le), operator_norm]
  ring

/-- CM108: the numerical full-matrix rule always produces a feasible Hilbert transition. -/
theorem normalized_bound (m : Pair.Matrix) : ‖operator (normalize m)‖ ≤ 1 := by
  rw [normalized_norm]
  exact (div_le_one (denominator_pos m)).mpr (le_max_right _ _)

/-- CM108: every already feasible full complex matrix remains exactly unchanged. -/
theorem normalize_eq (m : Pair.Matrix) (h : ‖operator m‖ ≤ 1) : normalize m = m := by
  have hr : radius m ≤ 1 := by rwa [operator_norm] at h
  simp [normalize, denominator, max_eq_left hr]

/-- CM108: a matrix on or outside the unit ball normalizes to its exact boundary. -/
theorem normalized_boundary (m : Pair.Matrix) (h : 1 ≤ ‖operator m‖) :
    ‖operator (normalize m)‖ = 1 := by
  have hr : 1 ≤ radius m := by rwa [operator_norm] at h
  rw [normalized_norm, denominator, max_eq_right hr, div_self]
  linarith

/-- CM108: zero entries stay stored, and the whole zero matrix remains zero. -/
theorem normalize_zero : normalize ((0, 0), (0, 0)) = ((0, 0), (0, 0)) := by
  apply normalize_eq
  rw [operator_norm]
  norm_num [radius, energyBound, top, left, right, cross]

/-- CM108: the entire admissible mixing subfamily survives this matrix rule. -/
theorem normalize_mixing (a b : ℝ) (h : a ^ 2 + b ^ 2 ≤ 1) :
    normalize (mixing a b) = mixing a b :=
  normalize_eq _ (mixing_bound a b h)

example : normalize ((Complex.I, 0), (0, Complex.I)) = ((Complex.I, 0), (0, Complex.I)) :=
  normalize_eq _ (by rw [operator_norm]; norm_num [radius, energyBound, top, left, right, cross])

example : ‖operator (normalize ((2 * Complex.I, 0), (0, 2 * Complex.I)))‖ = 1 :=
  normalized_boundary _ (by
    rw [operator_norm]
    norm_num [radius, energyBound, top, left, right, cross, norm_mul])

example : normalize (mixing (3 / 5) (4 / 5)) = mixing (3 / 5) (4 / 5) :=
  normalize_mixing _ _ (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound
