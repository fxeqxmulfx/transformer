module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Oracle
public import Transformer.Grokking.FlatSpectrum.Training.Logits
public import Transformer.Grokking.Geometry.ProbabilityLoss

/-!
# A singleton CE oracle cannot distinguish keys

Source: `review/flat_spectrum.md`, *Repairing minibatch and full-batch failures*,
MF1. The Boolean input with label `b` has a constant first slot and a separating
second slot. The tensor below is the actual ordinary-CE gradient on that input:
the residual `softmax - oneHot` is placed at each slot's observed value.
Its first-order inequality is checked against the existing CE theorem.

Every key has the same optimal singleton cost. Thus always taking the first
key is an exact oracle, even though that key cannot separate the two inputs.
Unobserved decoder rows can retain arbitrary earlier choices.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch

open scoped BigOperators
open Geometry (probability probability_pos probability_le_one)

/-- MF1: slot `false` is constant, while slot `true` carries the Boolean input. -/
def input (b k : Bool) : Bool := if k then b else false

/-- The ordinary-CE gradient of one input, MF1. Its values depend on the current
channel; no optimality, feasibility or constant-logit assumption is imposed. -/
noncomputable def sampleGradient (W : Channel Unit Bool Bool Bool) (b : Bool) :
    Channel Unit Bool Bool Bool := fun q k v c =>
  if v = input b k then probability (logits W q (input b)) c - (if c = b then 1 else 0)
  else 0

/-- At a singleton's observed value the true class minimizes the CE residual. -/
theorem residual_label_le (z : Bool → ℝ) (b c : Bool) :
    probability z b - 1 ≤ probability z c - (if c = b then 1 else 0) := by
  by_cases h : c = b
  · simp [h]
  · simp only [h, ite_false, sub_zero]
    exact (sub_nonpos.mpr (probability_le_one z b)).trans (probability_pos z c).le

/-- The tensor gradient pairs with a channel as the softmax residual pairs with
that channel's logits, MF1. -/
theorem linearCost_sampleGradient (W A : Channel Unit Bool Bool Bool) (b : Bool) :
    linearCost (sampleGradient W b) A =
      ∑ c, (probability (logits W () (input b)) c - (if c = b then 1 else 0)) *
        logits A () (input b) c := by
  cases b <;>
    simp [linearCost, sampleGradient, logits, input] <;> ring

/-- **The actual CE first-order inequality**, MF1; this verifies the relationship
between the explicit tensor and ordinary CE, instead of postulating a gradient. -/
theorem sampleGradient_first_order (W A : Channel Unit Bool Bool Bool) (b : Bool) :
    Landscape.Neural.crossEntropy (logits W () (input b)) b +
        linearCost (sampleGradient W b) (A - W) ≤
      Landscape.Neural.crossEntropy (logits A () (input b)) b := by
  have h := Training.crossEntropy_first_order
    (logits W () (input b)) (logits A () (input b)) b
  rw [linearCost_sampleGradient]
  have heq : (∑ c, (probability (logits W () (input b)) c -
      (if c = b then 1 else 0)) * logits (A - W) () (input b) c) =
      (∑ c, probability (logits W () (input b)) c *
        (logits A () (input b) c - logits W () (input b) c)) -
        (logits A () (input b) b - logits W () (input b) b) := by
    cases b <;> simp [logits, input] <;> ring
  rwa [heq]

/-- Set the observed row to the label; all other rows retain supplied history. -/
def observedDecoder (b : Bool) (history : Bool → Bool → Bool) (k v : Bool) : Bool :=
  if v = input b k then b else history k v

/-- **Every key ties on a singleton**, MF1, regardless of the current model or
the retained choices on unobserved values. -/
theorem singleton_key_cost (W : Channel Unit Bool Bool Bool) (b k : Bool)
    (history : Bool → Bool → Bool) :
    (∑ v, sampleGradient W b () k v (observedDecoder b history k v)) =
      probability (logits W () (input b)) b - 1 := by
  cases b <;> cases k <;>
    simp [sampleGradient, observedDecoder, input]

/-- **Any key, including the constant first key, is an exact singleton oracle**,
MF1. Retaining old unobserved decoder rows does not repair this tie. -/
theorem singleton_oracle (W : Channel Unit Bool Bool Bool) (b key : Bool)
    (history : Bool → Bool → Bool) {A : Channel Unit Bool Bool Bool} (hA : Feasible A) :
    linearCost (sampleGradient W b)
        (vertex (fun _ : Unit => key) (fun _ => observedDecoder b history)) ≤
      linearCost (sampleGradient W b) A := by
  apply linear_oracle _ _ _ ?_ ?_ hA
  · intro q k v c
    by_cases hv : v = input b k
    · simp only [sampleGradient, observedDecoder, hv, ite_true]
      exact residual_label_le _ b c
    · simp [sampleGradient, hv]
  · intro q k
    cases q
    rw [singleton_key_cost, singleton_key_cost]

/-- The first-order and oracle hypotheses have feasible nonzero witnesses. -/
example := sampleGradient_first_order
  (vertex (fun _ : Unit => false) (fun _ _ => id))
  (vertex (fun _ : Unit => true) (fun _ _ => id)) false

example := singleton_oracle (vertex (fun _ : Unit => true) (fun _ _ => id)) true false
  (fun _ => id) (feasible_vertex (fun _ : Unit => true) (fun _ _ => id))

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch
