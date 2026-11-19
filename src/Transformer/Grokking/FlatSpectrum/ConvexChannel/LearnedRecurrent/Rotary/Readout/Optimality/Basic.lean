module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Minibatch

/-!
# Exact linear minimization over a complete complex readout row

Source: `review/convex/moments.md`, CM103. Both complex coordinates enter the
coefficient and its norm. The explicit minimizing row handles ties and zeros;
a zero value does not remove that coordinate from the feasible domain.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality

open Pair

/-- Complete unconjugated row contraction, real linear in its coefficient, CM103. -/
noncomputable def pairing (u : State) : State →ₗ[ℝ] ℝ where
  toFun a := (Pair.row u a).re
  map_add' a b := by simp [Pair.row_apply, mul_add]; ring
  map_smul' r a := by
    change (u.1 * (r • a.1) + u.2 * (r • a.2)).re = r * _
    rw [mul_smul_comm, mul_smul_comm, ← smul_add, Complex.smul_re, smul_eq_mul,
      Pair.row_apply]

/-- The bilinear contraction is symmetric, including imaginary terms, CM103. -/
theorem pairing_comm (u a : State) : pairing u a = pairing a u := by
  simp only [pairing, LinearMap.coe_mk, AddHom.coe_mk, Pair.row_apply]
  congr 1
  ring

/-- Readout displacements contract to the difference of full rows, CM103. -/
theorem pairing_sub (v u a : State) : pairing (v - u) a = pairing v a - pairing u a := by
  rw [pairing_comm, map_sub, pairing_comm a v, pairing_comm a u]

/-- Every feasible complete row has contraction at least minus the maximum
complex coefficient magnitude, CM103. -/
theorem pairing_lower (u a : State) (hu : rowSize u ≤ 1) : -‖a‖ ≤ pairing u a := by
  have hr := Complex.re_le_norm (-(Pair.row u a))
  have ho := (Pair.row u).le_opNorm a
  have hm := mul_le_mul_of_nonneg_right (Pair.row_norm_bound u hu) (norm_nonneg a)
  simp only [Complex.neg_re, norm_neg] at hr
  change -‖a‖ ≤ (Pair.row u a).re
  linarith

/-- Negative conjugate phase; total inverse makes the all-zero case zero, CM103. -/
noncomputable def phase (z : ℂ) : ℂ := -(‖z‖⁻¹ : ℝ) • (starRingEnd ℂ) z

/-- The explicit phase always has magnitude at most one, CM103. -/
theorem phase_bound (z : ℂ) : ‖phase z‖ ≤ 1 := by
  by_cases hz : z = 0
  · simp [phase, hz]
  · have hn : ‖z‖ ≠ 0 := norm_ne_zero_iff.mpr hz
    simp [phase, hn]

/-- The explicit phase attains the negative coefficient magnitude, CM103. -/
theorem phase_cost (z : ℂ) : (phase z * z).re = -‖z‖ := by
  by_cases hz : z = 0
  · simp [phase, hz]
  · have hn : ‖z‖ ≠ 0 := norm_ne_zero_iff.mpr hz
    rw [phase, smul_mul_assoc, Complex.conj_mul']
    simp only [Complex.smul_re, ← Complex.ofReal_pow, Complex.ofReal_re, smul_eq_mul]
    field_simp

/-- Explicit minimizing row, choosing the first coordinate at ties, CM103. -/
noncomputable def bestRow (a : State) : State :=
  if ‖a.2‖ ≤ ‖a.1‖ then (phase a.1, 0) else (0, phase a.2)

/-- Both original coordinates of the minimizer obey the row capacity, CM103. -/
theorem bestRow_bound (a : State) : rowSize (bestRow a) ≤ 1 := by
  unfold bestRow
  split <;> simpa [rowSize] using phase_bound _

/-- The explicit feasible row attains the complete row support minimum, CM103. -/
theorem bestRow_cost (a : State) : pairing (bestRow a) a = -‖a‖ := by
  unfold bestRow
  split_ifs with h
  · simpa [pairing, Pair.row_apply, Prod.norm_def, max_eq_left h] using phase_cost a.1
  · simpa [pairing, Pair.row_apply, Prod.norm_def, max_eq_right (le_of_not_ge h)]
      using phase_cost a.2

/-- Exact optimality against every feasible complex row, CM103. -/
theorem bestRow_minimizes (a u : State) (hu : rowSize u ≤ 1) :
    pairing (bestRow a) a ≤ pairing u a := by
  rw [bestRow_cost]
  exact pairing_lower u a hu

example : -‖(Complex.I, (2 : ℂ))‖ ≤ pairing (Complex.I, 0) (Complex.I, 2) :=
  pairing_lower _ _ (by simp [rowSize])

example : pairing (bestRow (Complex.I, (2 : ℂ))) (Complex.I, 2) ≤
    pairing (Complex.I, 0) (Complex.I, 2) :=
  bestRow_minimizes _ _ (by simp [rowSize])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality
