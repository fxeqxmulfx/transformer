module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial.Gradient

/-!
# Strictly better feasible CE at a fixed radial update

Source: `review/convex/moments.md`, CM101. All original logits and labels enter
ordinary CE. The example disproves an optimizer implication; it does not claim
that the native recall endpoints equal this witness or identify unseen answers.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial

open Pair
open Landscape.Neural (crossEntropy)

/-- CM101: ordinary two-class CE, with both original logits present. -/
theorem binary_loss (m : ℝ) (b : Bool) :
    crossEntropy (fun c : Bool => if c = b then m else -m) b =
      Real.log (1 + Real.exp (-2 * m)) := by
  have hs : (∑ c : Bool, Real.exp (if c = b then m else -m)) =
      Real.exp m + Real.exp (-m) := by cases b <;> simp [add_comm]
  have he : Real.exp m + Real.exp (-m) =
      Real.exp m * (1 + Real.exp (-2 * m)) := by
    rw [mul_add, mul_one, ← Real.exp_add, show m + -2 * m = -m by ring]
  rw [crossEntropy, hs, ite_eq_left rfl, he,
    Real.log_mul (ne_of_gt (Real.exp_pos _)) (by positivity), Real.log_exp]
  ring

/-- CM101: full-class actual CE at the stalled parameters. -/
theorem stalled_loss (b : Bool) :
    crossEntropy (score (replace witness stalledRows) [b]) b =
      Real.log (1 + Real.exp (-5 / 3)) := by
  have he : score (replace witness stalledRows) [b] =
      (fun c => if c = b then (5 / 6 : ℝ) else -(5 / 6)) := by
    funext c
    rw [stalled_scores]
    split_ifs <;> ring
  rw [he, binary_loss]
  norm_num

/-- CM101: full-class actual CE of the better admissible readouts. -/
theorem better_loss (b : Bool) :
    crossEntropy (score (replace witness betterRows) [b]) b =
      Real.log (1 + Real.exp (-2)) := by
  rw [show score (replace witness betterRows) [b] =
    (fun c => if c = b then 1 else -1) from funext (better_scores b), binary_loss]
  norm_num

/-- CM101: both oppositely labeled records have strictly better feasible CE. -/
theorem strictly_better (b : Bool) :
    crossEntropy (score (replace witness betterRows) [b]) b <
      crossEntropy (score (replace witness stalledRows) [b]) b := by
  rw [better_loss, stalled_loss]
  apply Real.log_lt_log (by positivity)
  simpa only [add_comm] using
    add_lt_add_left (Real.exp_lt_exp.mpr (by norm_num : (-2 : ℝ) < -5 / 3)) 1

/-- CM101: the actual conditional CE family is convex at this nonconstant content. -/
theorem witness_convex (b : Bool) :
    ConvexOn ℝ (rowDomain (Fin 1) Bool)
      (fun u => crossEntropy
        (scoreMap 1 (fun _ : Fin 1 => 1) (fun _ => witness) u [b]) b) :=
  loss_convex _ _ _ _ _

/-- CM101: nonoptimal fixed radial rows inside a complete conditional CE domain.
This is not a fixed point claim about the native mixture-weight query shrink. -/
theorem nonoptimal_radial_point :
    Unit (replace witness stalledRows) ∧ Unit (replace witness betterRows) ∧
    (∀ b, crossEntropy (score (replace witness betterRows) [b]) b <
      crossEntropy (score (replace witness stalledRows) [b]) b) ∧
    (∀ eta : ℝ, 0 ≤ eta → ∀ b c,
      projectRow (stalledRows c - eta • gradient stalledRows b c) = stalledRows c) :=
  ⟨replace_unit _ witness_unit _ (fun c => (stalled_size c).le), better_unit,
    strictly_better, radial_stalls⟩

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial
