module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection.Metric
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial.Failure

/-!
# Exact projection escapes the actual radial CE failure

Source: `review/flat_spectrum.md`, CM102. One positive step computed from the
training CE probability reaches strictly better feasible readouts on both
oppositely labeled original prefixes of CM101. This is not benchmark credit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection

open Pair Radial
open Landscape.Neural (crossEntropy)

/-- CM102: both original records share the same positive wrong-class probability. -/
theorem wrong_shared (b : Bool) : wrongProbability b = wrongProbability true := by
  cases b <;> simp [wrongProbability, Geometry.probability, stalled_scores, add_comm]

/-- CM102: a concrete positive rate determined by the actual training probabilities. -/
noncomputable def escapeRate : ℝ := 4 / (3 * wrongProbability true)

/-- CM102: the displayed actual-CE step size is strictly positive. -/
theorem escapeRate_pos : 0 < escapeRate :=
  div_pos (by norm_num) (mul_pos (by norm_num) (wrong_positive true))

/-- CM102: exact scale of this step against the actual row gradient. -/
theorem escapeRate_product : escapeRate * (3 / 2 * wrongProbability true) = 2 := by
  unfold escapeRate
  field_simp [ne_of_gt (wrong_positive true)]
  ring

/-- CM102: every real/imaginary coordinate of the displaced rows is computed. -/
theorem displaced_row (b c : Bool) :
    stalledRows c - escapeRate • gradient stalledRows b c =
      if c then (2, 1) else (-2, -1) := by
  rw [gradient_alignment, wrong_shared]
  have he : stalledRows c - escapeRate •
      (-(3 / 2 * wrongProbability true) • stalledRows c) =
        (1 + escapeRate * (3 / 2 * wrongProbability true)) • stalledRows c := by module
  rw [he, escapeRate_product]
  cases c <;> norm_num [stalledRows, boundaryRow]

/-- CM102: the new explicit projection reaches CM101's better feasible rows. -/
theorem projected_escape (b c : Bool) :
    project (stalledRows c - escapeRate • gradient stalledRows b c) = betterRows c := by
  rw [displaced_row]
  cases c <;> norm_num [project, magnitudes, resize, betterRows]

/-- CM102: the complete projected model remains inside the original capacity. -/
theorem escape_unit (b : Bool) :
    Unit (replace witness
      (fun c => project (stalledRows c - escapeRate • gradient stalledRows b c))) :=
  replacement_unit _ witness_unit _

/-- CM102: actual ordinary CE strictly improves on both original labels. -/
theorem escape_loss (b : Bool) :
    crossEntropy (score (replace witness
      (fun c => project (stalledRows c - escapeRate • gradient stalledRows b c))) [b]) b <
        crossEntropy (score (replace witness stalledRows) [b]) b := by
  have he : (fun c => project (stalledRows c - escapeRate • gradient stalledRows b c)) =
      betterRows := funext (projected_escape b)
  rw [he]
  exact strictly_better b

/-- CM102: one positive constrained update repairs the proved radial-CE counterexample. -/
theorem repairs_radial_failure :
    0 < escapeRate ∧
    (∀ b, Unit (replace witness
      (fun c => project (stalledRows c - escapeRate • gradient stalledRows b c)))) ∧
    (∀ b, crossEntropy (score (replace witness
      (fun c => project (stalledRows c - escapeRate • gradient stalledRows b c))) [b]) b <
        crossEntropy (score (replace witness stalledRows) [b]) b) :=
  ⟨escapeRate_pos, escape_unit, escape_loss⟩

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection
