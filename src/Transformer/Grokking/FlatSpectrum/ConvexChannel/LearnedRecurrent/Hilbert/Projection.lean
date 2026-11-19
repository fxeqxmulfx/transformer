module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.Mixing
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection.Metric

/-!
# Exact contraction induced by the old row geometry

Source: `review/convex/moments.md`, CM107. The correct CM102 projection of a
Hilbert isometry need not remain an isometry. This is a geometry witness,
not a diagnosis of the native task's failure or a claim that CM102 is wrong.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert

/-- CM107: apply the existing complete-row projection to both transition rows. -/
noncomputable def projectMatrix (m : Pair.Matrix) : Pair.Matrix :=
  (Rotary.Readout.Projection.project m.1, Rotary.Readout.Projection.project m.2)

/-- CM107: the actual unitary witness violates both old absolute row budgets. -/
theorem mixing_old_rows :
    ‖(mixing (3 / 5) (4 / 5)).1.1‖ + ‖(mixing (3 / 5) (4 / 5)).1.2‖ = 7 / 5 ∧
    ‖(mixing (3 / 5) (4 / 5)).2.1‖ + ‖(mixing (3 / 5) (4 / 5)).2.2‖ = 7 / 5 := by
  norm_num [mixing, Complex.norm_real, Real.norm_eq_abs]

/-- CM107: imaginary state coordinates expose the old maximum-norm expansion. -/
theorem mixing_old_image :
    Pair.operator (mixing (3 / 5) (4 / 5)) (Complex.I, Complex.I) =
      (-Complex.I / 5, 7 * Complex.I / 5) := by
  ext <;> norm_num [Pair.operator_apply, mixing] <;> ring

/-- CM107: this matrix really exceeds the old operator budget, independently
of whether the old sufficient row condition was used to test it. -/
theorem mixing_old_unbounded : 1 < ‖Pair.operator (mixing (3 / 5) (4 / 5))‖ := by
  have h := (Pair.operator (mixing (3 / 5) (4 / 5))).le_opNorm (Complex.I, Complex.I)
  rw [mixing_old_image] at h
  norm_num [Prod.norm_def, norm_div, norm_mul] at h
  linarith

/-- CM107: the actual CM102 formula changes the rational unitary to Q. -/
theorem projected_mixing : projectMatrix (mixing (3 / 5) (4 / 5)) =
    mixing (2 / 5) (3 / 5) := by
  ext <;> norm_num [projectMatrix, mixing,
    Rotary.Readout.Projection.project, Rotary.Readout.Projection.resize,
    Rotary.Readout.Projection.magnitudes, Complex.norm_real, Real.norm_eq_abs,
    Complex.real_smul]

/-- CM107: the old geometry also admits transitions outside the new unit ball. -/
theorem old_broadcast_admissible : ‖Pair.operator ((1, 0), (1, 0))‖ ≤ 1 :=
  Pair.operator_norm_bound _ (by simp) (by simp)

/-- CM107: preserving the old family requires retaining its geometry as a branch. -/
theorem broadcast_hilbert_unbounded : 1 < ‖operator ((1, 0), (1, 0))‖ := by
  have h := (operator ((1, 0), (1, 0))).le_opNorm (WithLp.toLp 2 (Complex.I, 0))
  have he := norm_squared (operator ((1, 0), (1, 0)) (WithLp.toLp 2 (Complex.I, 0)))
  rw [operator_apply] at he
  norm_num at h he
  nlinarith [norm_nonneg (operator ((1, 0), (1, 0)) (WithLp.toLp 2 (Complex.I, 0)))]

/-- CM107: row projection is still closest in its declared complete metric. -/
theorem projected_closest (m v : Pair.Matrix)
    (h₁ : Rotary.Readout.rowSize v.1 ≤ 1) (h₂ : Rotary.Readout.rowSize v.2 ≤ 1) :
    Rotary.Readout.Projection.squaredDistance m.1 (projectMatrix m).1 +
      Rotary.Readout.Projection.squaredDistance m.2 (projectMatrix m).2 ≤
    Rotary.Readout.Projection.squaredDistance m.1 v.1 +
      Rotary.Readout.Projection.squaredDistance m.2 v.2 := by
  exact add_le_add (Rotary.Readout.Projection.project_closest m.1 v.1 h₁)
    (Rotary.Readout.Projection.project_closest m.2 v.2 h₂)

/-- CM107: the projected full complex transition contracts squared energy by13/25. -/
theorem projected_energy (v : State) :
    ‖operator (projectMatrix (mixing (3 / 5) (4 / 5))) v‖ ^ 2 =
      (13 / 25 : ℝ) * ‖v‖ ^ 2 := by
  rw [projected_mixing, mixing_energy]
  norm_num

/-- CM107: repeated actual Q execution multiplies the same energy loss at each depth. -/
theorem projected_power (v : State) (xs : List Bool) :
    ‖runFrom (witness (2 / 5) (3 / 5)) v xs‖ ^ 2 =
      (13 / 25 : ℝ) ^ xs.length * ‖v‖ ^ 2 := by
  apply run_squared_scale
  intro o z
  have h := projected_energy z
  rw [projected_mixing] at h
  exact h

/-- CM107: every nonzero complex state strictly loses norm after positive depth. -/
theorem projected_strict (v : State) (hv : v ≠ 0) (xs : List Bool) (hx : xs ≠ []) :
    ‖runFrom (witness (2 / 5) (3 / 5)) v xs‖ < ‖v‖ := by
  have hp : (13 / 25 : ℝ) ^ xs.length < 1 :=
    pow_lt_one₀ (by norm_num) (by norm_num) (Nat.ne_of_gt (List.length_pos_iff.mpr hx))
  have hs := mul_lt_mul_of_pos_right hp (sq_pos_of_pos (norm_pos_iff.mpr hv))
  have he := projected_power v xs
  nlinarith [norm_nonneg (runFrom (witness (2 / 5) (3 / 5)) v xs), norm_nonneg v]

example : Rotary.Readout.Projection.squaredDistance (mixing (3 / 5) (4 / 5)).1
      (projectMatrix (mixing (3 / 5) (4 / 5))).1 +
    Rotary.Readout.Projection.squaredDistance (mixing (3 / 5) (4 / 5)).2
      (projectMatrix (mixing (3 / 5) (4 / 5))).2 ≤
    Rotary.Readout.Projection.squaredDistance (mixing (3 / 5) (4 / 5)).1 (0, 0) +
      Rotary.Readout.Projection.squaredDistance (mixing (3 / 5) (4 / 5)).2 (0, 0) :=
  projected_closest _ ((0, 0), (0, 0)) (by simp [Rotary.Readout.rowSize])
    (by simp [Rotary.Readout.rowSize])

example : ‖runFrom (witness (2 / 5) (3 / 5)) (WithLp.toLp 2 (Complex.I, 0))
    [true, false]‖ < ‖(WithLp.toLp 2 (Complex.I, (0 : ℂ)) : State)‖ :=
  projected_strict _ (by
    intro h
    have h' := congrArg (fun z : State => z.ofLp.1) h
    simp at h') _ (by simp)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert
