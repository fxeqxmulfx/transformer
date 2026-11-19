module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.Queries

/-!
# Reuse the complete softmax partition for finite single-class output queries

Source: `review/convex/moments.md`, CM109. Changing one complete output row
changes just its class logit. The original full-vocabulary cross-entropy has
an exact cached evaluation; no derivative or reduced label set is substituted.
The two actual changes may be asymmetric after feasible parameter projection.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.OutputQueries

open scoped BigOperators

variable {O : Type*} [Fintype O] [DecidableEq O]

/-- CM109: the full original-class exponential partition. -/
noncomputable def partition (z : O → ℝ) : ℝ := ∑ c, Real.exp (z c)

/-- CM109: the changed class's actual share of the complete partition. -/
noncomputable def share (z : O → ℝ) (c : O) : ℝ := Real.exp (z c) / partition z

/-- CM109: an actual one-class logit change, retaining every other logit. -/
def shift (z : O → ℝ) (c : O) (delta : ℝ) (d : O) : ℝ :=
  z d + if d = c then delta else 0

omit [DecidableEq O] in
/-- CM109: an existing original class witnesses positive full mass. -/
theorem partition_pos (z : O → ℝ) (c : O) : 0 < partition z :=
  Landscape.Neural.sum_exp_pos z c

omit [Fintype O] in
/-- CM109: the shifted logits are the usual complete function update. -/
theorem shift_eq_update (z : O → ℝ) (c : O) (delta : ℝ) :
    shift z c delta = Function.update z c (z c + delta) := by
  funext d
  by_cases h : d = c
  · subst d
    simp [shift]
  · simp [shift, h]

/-- CM109: expand the full partition, with exactly one changed summand. -/
theorem partition_shift (z : O → ℝ) (c : O) (delta : ℝ) :
    partition (shift z c delta) =
      partition z + Real.exp (z c) * (Real.exp delta - 1) := by
  have he (d : O) : Real.exp (shift z c delta d) =
      Real.exp (z d) + if d = c then Real.exp (z c) * (Real.exp delta - 1) else 0 := by
    by_cases h : d = c
    · subst d
      simp [shift, Real.exp_add]
      ring
    · simp [shift, h]
  unfold partition
  simp_rw [he]
  rw [Finset.sum_add_distrib]
  simp only [Fintype.sum_ite_eq']

/-- CM109: the new full partition factors through its cached original value. -/
theorem partition_factor (z : O → ℝ) (c : O) (delta : ℝ) :
    partition (shift z c delta) =
      partition z * (1 + share z c * (Real.exp delta - 1)) := by
  rw [partition_shift, share]
  have hne := ne_of_gt (partition_pos z c)
  field_simp [hne]

/-- CM109: the logarithm's cached factor is positive for every real change. -/
theorem factor_pos (z : O → ℝ) (c : O) (delta : ℝ) :
    0 < 1 + share z c * (Real.exp delta - 1) := by
  have h := partition_pos (shift z c delta) c
  rw [partition_factor] at h
  exact (mul_pos_iff_of_pos_left (partition_pos z c)).mp h

/-- CM109: exact ordinary complete-class CE at a finite single-logit query. -/
theorem crossEntropy_shift (z : O → ℝ) (target c : O) (delta : ℝ) :
    Landscape.Neural.crossEntropy (shift z c delta) target =
      Landscape.Neural.crossEntropy z target +
        Real.log (1 + share z c * (Real.exp delta - 1)) -
        if target = c then delta else 0 := by
  change Real.log (partition (shift z c delta)) - shift z c delta target = _
  rw [partition_factor, Real.log_mul (ne_of_gt (partition_pos z c))
    (ne_of_gt (factor_pos z c delta))]
  unfold Landscape.Neural.crossEntropy shift partition
  split_ifs <;> ring

/-- CM109: reuse both finite CE queries even when projection makes them asymmetric. -/
theorem two_query_difference (z : O → ℝ) (target c : O) (plus minus : ℝ) :
    Landscape.Neural.crossEntropy (shift z c plus) target -
        Landscape.Neural.crossEntropy (shift z c minus) target =
      Real.log (1 + share z c * (Real.exp plus - 1)) -
        Real.log (1 + share z c * (Real.exp minus - 1)) -
        if target = c then plus - minus else 0 := by
  rw [crossEntropy_shift, crossEntropy_shift]
  split_ifs <;> ring

example : 0 < 1 + share (fun b : Bool => if b then 1 else -1) true *
    (Real.exp (-2 / 3) - 1) := factor_pos _ _ _

example : Landscape.Neural.crossEntropy
    (shift (fun b : Bool => if b then 1 else -1) true (1 / 3)) false =
    Landscape.Neural.crossEntropy (fun b : Bool => if b then 1 else -1) false +
      Real.log (1 + share (fun b : Bool => if b then 1 else -1) true *
        (Real.exp (1 / 3) - 1)) := by
  simpa using crossEntropy_shift (fun b : Bool => if b then 1 else -1) false true (1 / 3)

example : Landscape.Neural.crossEntropy
    (shift (fun c : Fin 3 => if c = 0 then 2 else if c = 1 then -1 else 1 / 2)
      1 (1 / 7)) 1 - Landscape.Neural.crossEntropy
    (shift (fun c : Fin 3 => if c = 0 then 2 else if c = 1 then -1 else 1 / 2)
      1 (-2 / 9)) 1 =
    Real.log (1 + share (fun c : Fin 3 => if c = 0 then 2 else if c = 1 then -1 else 1 / 2)
      1 * (Real.exp (1 / 7) - 1)) -
    Real.log (1 + share (fun c : Fin 3 => if c = 0 then 2 else if c = 1 then -1 else 1 / 2)
      1 * (Real.exp (-2 / 9) - 1)) - (1 / 7 - -2 / 9) := by
  simpa using two_query_difference
    (fun c : Fin 3 => if c = 0 then 2 else if c = 1 then -1 else 1 / 2) 1 1 (1 / 7) (-2 / 9)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.OutputQueries
