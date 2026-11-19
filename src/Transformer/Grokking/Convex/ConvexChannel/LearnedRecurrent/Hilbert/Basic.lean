module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Pair.Basic
public import Mathlib.Analysis.InnerProductSpace.ProdL2

/-!
# Complete learned complex coordinates with a Hilbert state norm

Source: `review/convex/moments.md`, CM107. The numerical coordinates remain
complete; changing the norm changes their capacity geometry, not their meaning.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert

/-- CM107: both complex channels with their Euclidean norm. -/
abbrev State := WithLp 2 (ℂ × ℂ)

/-- CM107: retain every numerical coordinate when changing the state norm. -/
noncomputable def coordinates : State ≃L[ℂ] Pair.State :=
  WithLp.prodContinuousLinearEquiv 2 ℂ ℂ ℂ

/-- CM107: all four complex entries execute in the new state geometry. -/
noncomputable def operator (m : Pair.Matrix) : State →L[ℂ] State :=
  coordinates.symm.toContinuousLinearMap.comp
    ((Pair.operator m).comp coordinates.toContinuousLinearMap)

/-- CM107: both complete complex coefficients remain in every class readout. -/
noncomputable def row (u : Pair.State) : State →L[ℂ] ℂ :=
  (Pair.row u).comp coordinates.toContinuousLinearMap

/-- CM107: the changed norm leaves the actual matrix product unchanged. -/
theorem operator_apply (m : Pair.Matrix) (v : State) :
    (operator m v).ofLp =
      (m.1.1 * v.ofLp.1 + m.1.2 * v.ofLp.2,
       m.2.1 * v.ofLp.1 + m.2.2 * v.ofLp.2) := by
  simp [operator, coordinates, Pair.operator_apply]

/-- CM107: actual full complex readout, without deleting imaginary terms. -/
theorem row_apply (u : Pair.State) (v : State) :
    row u v = u.1 * v.ofLp.1 + u.2 * v.ofLp.2 := by
  simp [row, coordinates, Pair.row_apply]

/-- CM107: the squared state norm retains all four real coordinates. -/
theorem norm_squared (v : State) :
    ‖v‖ ^ 2 = ‖v.ofLp.1‖ ^ 2 + ‖v.ofLp.2‖ ^ 2 :=
  WithLp.prod_norm_sq_eq_of_L2 v

/-- CM107: every continuous complex linear map still uses both basis states. -/
theorem basis_expansion {F : Type*} [NormedAddCommGroup F] [NormedSpace ℂ F]
    (L : State →L[ℂ] F) (v : State) :
    L v = v.ofLp.1 • L (WithLp.toLp 2 (1, 0)) +
      v.ofLp.2 • L (WithLp.toLp 2 (0, 1)) := by
  have he : v = v.ofLp.1 • WithLp.toLp 2 (1, 0) +
      v.ofLp.2 • WithLp.toLp 2 (0, 1) := by
    apply (WithLp.ext_iff 2).mpr
    ext <;> simp
  calc
    L v = L (v.ofLp.1 • WithLp.toLp 2 (1, 0) +
      v.ofLp.2 • WithLp.toLp 2 (0, 1)) := congrArg L he
    _ = _ := by rw [map_add, map_smul, map_smul]

/-- CM107: no complex linear readout is lost by the coordinate presentation. -/
theorem row_representation (L : State →L[ℂ] ℂ) :
    row (L (WithLp.toLp 2 (1, 0)), L (WithLp.toLp 2 (0, 1))) = L := by
  apply ContinuousLinearMap.ext
  intro v
  rw [row_apply]
  simpa [smul_eq_mul, mul_comm] using (basis_expansion L v).symm

/-- CM107: every full complex linear transition has this matrix representation. -/
theorem operator_representation (L : State →L[ℂ] State) :
    operator (((L (WithLp.toLp 2 (1, 0))).ofLp.1,
        (L (WithLp.toLp 2 (0, 1))).ofLp.1),
      ((L (WithLp.toLp 2 (1, 0))).ofLp.2,
        (L (WithLp.toLp 2 (0, 1))).ofLp.2)) = L := by
  apply ContinuousLinearMap.ext
  intro v
  apply (WithLp.ext_iff 2).mpr
  rw [operator_apply]
  have h := congrArg WithLp.ofLp (basis_expansion L v)
  apply Prod.ext
  · simpa [smul_eq_mul, mul_comm] using (congrArg Prod.fst h).symm
  · simpa [smul_eq_mul, mul_comm] using (congrArg Prod.snd h).symm

/-- CM107: the complete squared coefficient budget bounds the Hilbert readout. -/
theorem row_norm_bound (u : Pair.State) (hu : ‖u.1‖ ^ 2 + ‖u.2‖ ^ 2 ≤ 1) :
    ‖row u‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by norm_num)
  intro v
  have ht : ‖row u v‖ ≤ ‖u.1‖ * ‖v.ofLp.1‖ + ‖u.2‖ * ‖v.ofLp.2‖ := by
    rw [row_apply]
    simpa only [norm_mul] using norm_add_le (u.1 * v.ofLp.1) (u.2 * v.ofLp.2)
  have hc : (‖u.1‖ * ‖v.ofLp.1‖ + ‖u.2‖ * ‖v.ofLp.2‖) ^ 2 ≤
      (‖u.1‖ ^ 2 + ‖u.2‖ ^ 2) * (‖v.ofLp.1‖ ^ 2 + ‖v.ofLp.2‖ ^ 2) := by
    nlinarith [sq_nonneg (‖u.1‖ * ‖v.ofLp.2‖ - ‖u.2‖ * ‖v.ofLp.1‖)]
  have hb := mul_le_mul_of_nonneg_right hu (sq_nonneg ‖v‖)
  rw [← norm_squared v] at hc
  have hs := mul_nonneg (norm_nonneg u.1) (norm_nonneg v.ofLp.1)
  have hs' := mul_nonneg (norm_nonneg u.2) (norm_nonneg v.ofLp.2)
  nlinarith [norm_nonneg (row u v), norm_nonneg v]

example : ‖row (Complex.I / 2, Complex.I / 2)‖ ≤ 1 :=
  row_norm_bound _ (by norm_num [norm_div])

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert
