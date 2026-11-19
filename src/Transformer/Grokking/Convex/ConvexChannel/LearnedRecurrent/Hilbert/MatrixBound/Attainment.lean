module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound.Gram

/-!
# The explicit complete matrix bound is sharp

Source: `review/convex/moments.md`, CM108. Construct the nonzero attaining state
from the actual Gram entries, covering the zero-gap case separately. No
external eigenvector, singular-value routine or optimizer hypothesis is used.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound

/-- CM108: the first basis state executes with the first complete column energy. -/
theorem first_energy (m : Pair.Matrix) :
    ‖operator m (WithLp.toLp 2 (1, 0))‖ ^ 2 = left m := by
  rw [energy_identity]
  simp

/-- CM108: a zero first diagonal gap forces the complete complex cross term to zero. -/
theorem cross_zero_of_equal_left (m : Pair.Matrix) (h : energyBound m = left m) :
    cross m = 0 := by
  have he := gap_product (left m) (right m) ‖cross m‖
  change (energyBound m - left m) * (energyBound m - right m) = ‖cross m‖ ^ 2 at he
  rw [h, sub_self, zero_mul] at he
  apply norm_eq_zero.mp
  nlinarith [norm_nonneg (cross m)]

/-- CM108: construct an actual nonzero complex state attaining the proposed bound. -/
theorem energy_attained (m : Pair.Matrix) :
    ∃ v : State, v ≠ 0 ∧ ‖operator m v‖ ^ 2 = energyBound m * ‖v‖ ^ 2 := by
  by_cases h : energyBound m = left m
  · refine ⟨WithLp.toLp 2 (1, 0), ?_, ?_⟩
    · intro hz
      have he := congrArg (fun z : State => z.ofLp.1) hz
      norm_num at he
    · rw [first_energy, h]
      simp
  · have hp : 0 < energyBound m - left m := by
      have hl := top_ge_left (left m) (right m) ‖cross m‖
      change left m ≤ energyBound m at hl
      rcases eq_or_lt_of_le hl with he | he
      · exact False.elim (h he.symm)
      · linarith
    refine ⟨WithLp.toLp 2 (cross m, ((energyBound m - left m : ℝ) : ℂ)), ?_, ?_⟩
    · intro hz
      have he := congrArg (fun z : State => z.ofLp.2) hz
      have he' : energyBound m - left m = 0 := by simpa using congrArg Complex.re he
      linarith
    · have hc : (cross m * (starRingEnd ℂ) (cross m) *
          ((energyBound m - left m : ℝ) : ℂ)).re =
          ‖cross m‖ ^ 2 * (energyBound m - left m) := by
        rw [Complex.mul_conj', ← Complex.ofReal_pow, ← Complex.ofReal_mul,
          Complex.ofReal_re]
      rw [energy_identity, norm_squared]
      change left m * ‖cross m‖ ^ 2 + right m * ‖((energyBound m - left m : ℝ) : ℂ)‖ ^ 2 +
          2 * (cross m * (starRingEnd ℂ) (cross m) *
            ((energyBound m - left m : ℝ) : ℂ)).re =
        energyBound m * (‖cross m‖ ^ 2 + ‖((energyBound m - left m : ℝ) : ℂ)‖ ^ 2)
      rw [hc, Complex.norm_real, Real.norm_eq_abs, sq_abs]
      have he := gap_product (left m) (right m) ‖cross m‖
      change (energyBound m - left m) * (energyBound m - right m) = ‖cross m‖ ^ 2 at he
      have hh := congrArg (fun x : ℝ => (energyBound m - left m) * x) he
      nlinarith

/-- CM108: the closed formula is the actual full complex Hilbert operator norm. -/
theorem operator_norm (m : Pair.Matrix) : ‖operator m‖ = radius m := by
  apply le_antisymm (operator_le m)
  by_contra h
  have hl : ‖operator m‖ < radius m := lt_of_not_ge h
  obtain ⟨v, hv, he⟩ := energy_attained m
  have hp := norm_pos_iff.mpr hv
  have hs := (sq_lt_sq₀ (norm_nonneg (operator m)) (radius_nonneg m)).mpr hl
  have ht := mul_lt_mul_of_pos_right hs (sq_pos_of_pos hp)
  rw [radius_squared] at ht
  have hb := (operator m).le_opNorm v
  have hsq := (sq_le_sq₀ (norm_nonneg (operator m v))
    (mul_nonneg (norm_nonneg (operator m)) (norm_nonneg v))).mpr hb
  rw [mul_pow] at hsq
  nlinarith

example : cross ((Complex.I, 0), (0, Complex.I)) = 0 :=
  cross_zero_of_equal_left _ (by norm_num [energyBound, top, left, right, cross])

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound
