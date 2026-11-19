module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Loss

/-!
# A complete training CE gap allowing mixture-capacity redistribution

Source: `review/convex/moments.md`, CM104. The gap bounds all feasible weighted
readout changes on current learned content. Neither current mixture weights
nor their zero pattern restrict competitors. Exact-real conditional statements
do not supply a floating certificate, raw accuracy or unseen answers.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair Optimality
variable {J O : Type*} [Fintype J] [Nonempty J] [Fintype O] [DecidableEq O]

/-- Full shared-capacity gap using actual indexed CE coefficients, CM104. -/
noncomputable def jointGap (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) : ℝ :=
  linearCost u (jointCoefficient n f xs target u) + B * peak (jointCoefficient n f xs target u)

/-- The complete joint-capacity CE gap is nonnegative at every feasible table, CM104. -/
theorem joint_gap_nonneg [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (hu : u ∈ jointDomain B) : 0 ≤ jointGap n B f xs target u := by
  have h := full_cost_lower B u (jointCoefficient n f xs target u) hu
  unfold jointGap
  linarith

/-- Actual mean CE minus the joint gap bounds every feasible complete
readout/capacity reallocation on the retained learned states, CM104. -/
theorem joint_gap_lower [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u v : J → O → State) (hv : v ∈ jointDomain B) :
    jointLoss n f xs target u - jointGap n B f xs target u ≤ jointLoss n f xs target v := by
  have hf := joint_first_order n f xs target u v
  have hl := full_cost_lower B v (jointCoefficient n f xs target u) hv
  rw [linearCost_sub] at hf
  unfold jointGap
  linarith

/-- All conditional CE improvement, including mixture-weight redistribution,
is bounded by the complete gap, CM104. -/
theorem joint_improvement_le [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u v : J → O → State) (hv : v ∈ jointDomain B) :
    jointLoss n f xs target u - jointLoss n f xs target v ≤ jointGap n B f xs target u := by
  have h := joint_gap_lower n B f xs target u v hv
  linarith

/-- Zero computed gap gives the enlarged conditional optimum, without an
assumed endpoint or optimizer oracle, CM104. -/
theorem joint_optimal_of_zero [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (hg : jointGap n B f xs target u = 0)
    (v : J → O → State) (hv : v ∈ jointDomain B) :
    jointLoss n f xs target u ≤ jointLoss n f xs target v := by
  have h := joint_gap_lower n B f xs target u v hv
  rwa [hg, sub_zero] at h

variable [DecidableEq J]

/-- The proved finite support minimizer has complete linear change minus the
joint gap, CM104; no current-weight factor suppresses a component. -/
theorem joint_best_change (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) :
    linearCost (proposal B (jointCoefficient n f xs target u)
      (selected (jointCoefficient n f xs target u)) - u) (jointCoefficient n f xs target u) =
        -jointGap n B f xs target u := by
  rw [linearCost_sub, selected_cost]
  unfold jointGap
  ring

/-- A positive gap at a feasible point gives a complete feasible direction
with negative actual first-order contraction, CM104. -/
theorem joint_negative_direction [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (hu : u ∈ jointDomain B) (hg : 0 < jointGap n B f xs target u) :
    ∃ v ∈ jointDomain B, linearCost (v - u) (jointCoefficient n f xs target u) < 0 := by
  have hB : 0 ≤ B := (size_nonneg u).trans hu
  refine ⟨proposal B (jointCoefficient n f xs target u)
    (selected (jointCoefficient n f xs target u)), proposal_mem B hB _ _, ?_⟩
  rw [joint_best_change]
  linarith

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
