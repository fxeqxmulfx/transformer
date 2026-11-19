module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial.Basic

/-!
# Actual CE residuals and a nonoptimal fixed radial row update

Source: `review/convex/moments.md`, CM101. The explicit complete row contraction
is verified against ordinary CE's first-order inequality. Both oppositely
labeled original records give outward radial displacements at the witness.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial

open Pair FlatSpectrum
open Landscape.Neural (crossEntropy)
open Geometry (probability probability_pos sum_probability)
open scoped BigOperators

/-- CM101: conjugate every complex coordinate, retaining the whole row. -/
noncomputable def conjugate (z : State) : State :=
  ((starRingEnd ℂ) z.1, (starRingEnd ℂ) z.2)

/-- CM101: the actual ordinary-CE readout gradient on an original record. -/
noncomputable def gradient (u : Bool → State) (b c : Bool) : State :=
  (probability (score (replace witness u) [b]) c - if c = b then 1 else 0) •
    conjugate (hidden b)

/-- CM101: the full real/imaginary row pairing equals the actual logit change. -/
theorem gradient_pairing (u v : Bool → State) (b c : Bool) :
    (Pair.row (conjugate (gradient u b c)) (v c - u c)).re =
      (probability (score (replace witness u) [b]) c - if c = b then 1 else 0) *
        (score (replace witness v) [b] c - score (replace witness u) [b] c) := by
  rw [actual_score v, actual_score u]
  cases b <;>
    simp [gradient, conjugate, hidden, Pair.row_apply, Complex.mul_re] <;> ring

/-- CM101: verify the explicit row gradient with actual ordinary CE. -/
theorem gradient_first_order (u v : Bool → State) (b : Bool) :
    crossEntropy (score (replace witness u) [b]) b +
      ∑ c, (Pair.row (conjugate (gradient u b c)) (v c - u c)).re ≤
        crossEntropy (score (replace witness v) [b]) b := by
  have h := Training.crossEntropy_first_order
    (score (replace witness u) [b]) (score (replace witness v) [b]) b
  simp_rw [gradient_pairing]
  have he : (∑ c : Bool,
      (probability (score (replace witness u) [b]) c - if c = b then 1 else 0) *
        (score (replace witness v) [b] c - score (replace witness u) [b] c)) =
      (∑ c : Bool, probability (score (replace witness u) [b]) c *
        (score (replace witness v) [b] c - score (replace witness u) [b] c)) -
        (score (replace witness v) [b] b - score (replace witness u) [b] b) := by
    cases b <;> simp <;> ring
  rwa [he]

/-- CM101: positive wrong-class probability for either original record. -/
noncomputable def wrongProbability (b : Bool) : ℝ :=
  probability (score (replace witness stalledRows) [b]) (!b)

/-- CM101: both actual finite-logit wrong-class probabilities are positive. -/
theorem wrong_positive (b : Bool) : 0 < wrongProbability b := probability_pos _ _

/-- CM101: both original classes contribute their actual softmax residual. -/
theorem residual (b c : Bool) :
    probability (score (replace witness stalledRows) [b]) c -
      (if c = b then 1 else 0) =
        if c = b then -wrongProbability b else wrongProbability b := by
  have h := sum_probability (score (replace witness stalledRows) [b]) b
  cases b <;> cases c <;> simp_all [wrongProbability] <;> linarith

/-- CM101: the actual CE gradient points along each existing boundary row. -/
theorem gradient_alignment (b c : Bool) :
    gradient stalledRows b c = -(3 / 2 * wrongProbability b) • stalledRows c := by
  unfold gradient
  rw [residual]
  cases b <;> cases c <;> ext <;>
    norm_num [conjugate, hidden, stalledRows, boundaryRow, map_ofNat] <;>
    ring

/-- CM101: every nonnegative ordinary gradient step returns the same row. -/
theorem radial_stalls (eta : ℝ) (heta : 0 ≤ eta) (b c : Bool) :
    projectRow (stalledRows c - eta • gradient stalledRows b c) = stalledRows c := by
  rw [gradient_alignment]
  have he : stalledRows c - eta • (-(3 / 2 * wrongProbability b) • stalledRows c) =
      (1 + eta * (3 / 2 * wrongProbability b)) • stalledRows c := by module
  rw [he]
  apply project_ray _ (stalled_size c)
  have hq := wrong_positive b
  have hp : 0 ≤ eta * (3 / 2 * wrongProbability b) := mul_nonneg heta (by positivity)
  linarith

/-- CM101: line search cannot improve an unchanged radial proposal. -/
theorem interpolation_stalls (eta : ℝ) (heta : 0 ≤ eta) (a : ℝ) (b c : Bool) :
    (1 - a) • stalledRows c +
      a • projectRow (stalledRows c - eta • gradient stalledRows b c) = stalledRows c := by
  rw [radial_stalls eta heta]
  module

example : projectRow (stalledRows true - (4 : ℝ) • gradient stalledRows false true) =
    stalledRows true := radial_stalls 4 (by norm_num) false true

example : (1 - (1 / 2 : ℝ)) • stalledRows false + (1 / 2 : ℝ) •
    projectRow (stalledRows false - (4 : ℝ) • gradient stalledRows true false) =
      stalledRows false := interpolation_stalls 4 (by norm_num) (1 / 2) true false

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial
