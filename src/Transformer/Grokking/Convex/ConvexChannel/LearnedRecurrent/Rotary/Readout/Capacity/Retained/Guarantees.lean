module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Current
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Gap

/-!
# Current-content CE and direction bounds for historical caches

Source: `review/convex/moments.md`, CM106. The CM105 correction receives both
signal error and actual learned-content drift. Conclusions concern actual CE
at the current content, not finite content-step descent or unseen identification.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

open Pair Optimality Estimate
variable {J O : Type*} [Fintype J] [Nonempty J] [Fintype O] [Nonempty O] [DecidableEq O]

/-- A historical cache needs both errors in its current-content actual CE
lower bound, CM106. Every feasible complete competitor is covered. -/
theorem cached_loss_lower (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O)
    (target : Fin (n + 1) → O) (u v : J → O → State) (s : Fin (n + 1) → O → ℝ)
    (hf : ∀ j, Unit (f j)) (hg : ∀ i j, Unit (g i j))
    (hu : u ∈ jointDomain B) (hv : v ∈ jointDomain B) :
    jointLoss n f xs target u - estimatedGap B u (cached n g xs s) -
      2 * B * currentError n f g xs target u s ≤ jointLoss n f xs target v := by
  have hB : 0 ≤ B := (size_nonneg u).trans hu
  have he := mul_le_mul_of_nonneg_left
    (current_coefficient_error n f g xs target u s hf hg) (by positivity : 0 ≤ 2 * B)
  have hl := estimated_loss_lower n B f xs target u (cached n g xs s) v hu hv
  linarith

variable [DecidableEq J]

/-- Explicit signal/content error controls the actual direction selected
from a historical complete estimate, CM106. -/
theorem cached_direction_upper (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O)
    (target : Fin (n + 1) → O) (u : J → O → State) (s : Fin (n + 1) → O → ℝ)
    (hf : ∀ j, Unit (f j)) (hg : ∀ i j, Unit (g i j)) (hu : u ∈ jointDomain B) :
    linearCost (proposal B (cached n g xs s) (selected (cached n g xs s)) - u)
      (jointCoefficient n f xs target u) ≤ -estimatedGap B u (cached n g xs s) +
        2 * B * currentError n f g xs target u s := by
  have hB : 0 ≤ B := (size_nonneg u).trans hu
  have he := mul_le_mul_of_nonneg_left
    (current_coefficient_error n f g xs target u s hf hg) (by positivity : 0 ≤ 2 * B)
  have hd := actual_direction_upper n B f xs target u (cached n g xs s) hu
  linarith

/-- Only a gap exceeding both complete errors guarantees a negative actual
first-order direction, CM106; this is not a finite-step or generalization claim. -/
theorem cached_negative (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (g : Fin (n + 1) → J → Parameters O) (xs : Fin (n + 1) → List O)
    (target : Fin (n + 1) → O) (u : J → O → State) (s : Fin (n + 1) → O → ℝ)
    (hf : ∀ j, Unit (f j)) (hg : ∀ i j, Unit (g i j)) (hu : u ∈ jointDomain B)
    (hgap : 2 * B * currentError n f g xs target u s < estimatedGap B u (cached n g xs s)) :
    ∃ v ∈ jointDomain B, linearCost (v - u) (jointCoefficient n f xs target u) < 0 := by
  have hB : 0 ≤ B := (size_nonneg u).trans hu
  apply negative_if_error_small n B f xs target u (cached n g xs s) hu
  exact (mul_le_mul_of_nonneg_left (current_coefficient_error n f g xs target u s hf hg)
    (by positivity : 0 ≤ 2 * B)).trans_lt hgap

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained
