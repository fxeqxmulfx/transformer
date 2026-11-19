module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Basic

/-!
# Actual training guarantees with a complete estimated coefficient

Source: `review/convex/moments.md`, CM105. The estimated gap receives an explicit
correction for its actual full-table discrepancy. No minibatch, retained or
finite-query estimate is silently identified with the complete training gradient.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate

open Pair Optimality
variable {J O : Type*} [Fintype J] [Nonempty J] [Fintype O]

/-- Gap computed from any complete coefficient estimate, CM105. -/
noncomputable def estimatedGap (B : ℝ) (u h : J → O → State) : ℝ :=
  linearCost u h + B * peak h

/-- Every estimated gap is nonnegative at a feasible table, CM105; this alone
does not identify an actual complete-training descent direction. -/
theorem estimated_nonneg [Nonempty O] (B : ℝ) (u h : J → O → State)
    (hu : u ∈ jointDomain B) : 0 ≤ estimatedGap B u h := by
  have h := full_cost_lower B u h hu
  unfold estimatedGap
  linarith

/-- Complete estimated and actual gaps differ by at most the explicit
coefficient-error correction, CM105, for every feasible current table. -/
theorem gap_error [Nonempty O] (B : ℝ) (u a h : J → O → State)
    (hu : u ∈ jointDomain B) :
    |estimatedGap B u a - estimatedGap B u h| ≤ 2 * B * peak (a - h) := by
  have hB : 0 ≤ B := (size_nonneg u).trans hu
  calc
    _ = |(linearCost u a - linearCost u h) + B * (peak a - peak h)| := by
      unfold estimatedGap
      congr 1
      ring
    _ ≤ |linearCost u a - linearCost u h| + |B * (peak a - peak h)| := abs_add_le _ _
    _ = |linearCost u a - linearCost u h| + B * |peak a - peak h| := by
      rw [abs_mul, abs_of_nonneg hB]
    _ ≤ B * peak (a - h) + B * peak (a - h) :=
      add_le_add (coefficient_error B u a h hu)
        (mul_le_mul_of_nonneg_left (peak_difference a h) hB)
    _ = 2 * B * peak (a - h) := by ring

variable [DecidableEq O]

/-- Corrected estimated gap bounds the actual complete indexed training CE
against every feasible competitor, CM105. Estimator error is not assumed zero. -/
theorem estimated_loss_lower [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u h v : J → O → State) (hu : u ∈ jointDomain B) (hv : v ∈ jointDomain B) :
    jointLoss n f xs target u - estimatedGap B u h -
      2 * B * peak (jointCoefficient n f xs target u - h) ≤ jointLoss n f xs target v := by
  have he := (abs_le.mp (gap_error B u (jointCoefficient n f xs target u) h hu)).2
  have hl := joint_gap_lower n B f xs target u v hv
  change jointGap n B f xs target u - estimatedGap B u h ≤ _ at he
  linarith

/-- All conditional actual mean-CE improvement is bounded by estimated gap
plus full coefficient error, CM105. This is not an unseen-label bound. -/
theorem estimated_improvement_le [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u h v : J → O → State) (hu : u ∈ jointDomain B) (hv : v ∈ jointDomain B) :
    jointLoss n f xs target u - jointLoss n f xs target v ≤
      estimatedGap B u h + 2 * B * peak (jointCoefficient n f xs target u - h) := by
  have hl := estimated_loss_lower n B f xs target u h v hu hv
  linarith

variable [DecidableEq J]

omit [DecidableEq O] in
/-- The explicit finite minimizer has directional contraction minus the
estimated gap for that estimate, CM105; no actual-gradient equality is assumed. -/
theorem estimated_best_change (B : ℝ) (u h : J → O → State) :
    linearCost (proposal B h (selected h) - u) h = -estimatedGap B u h := by
  rw [linearCost_sub, selected_cost]
  unfold estimatedGap
  ring

/-- The estimated support direction has a controlled actual complete-training
contraction, CM105. Feasibility supplies the shared radius in its error term. -/
theorem actual_direction_upper [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u h : J → O → State) (hu : u ∈ jointDomain B) :
    linearCost (proposal B h (selected h) - u) (jointCoefficient n f xs target u) ≤
      -estimatedGap B u h + 2 * B * peak (jointCoefficient n f xs target u - h) := by
  have hB : 0 ≤ B := (size_nonneg u).trans hu
  have he := (abs_le.mp (direction_error B u (proposal B h (selected h))
    (jointCoefficient n f xs target u) h hu (proposal_mem B hB h (selected h)))).2
  rw [estimated_best_change] at he
  linarith

/-- A gap exceeding its actual complete coefficient-error correction gives
a feasible negative actual first-order direction, CM105. This does not assert
finite-step descent or provide an uncomputed stochastic-error certificate. -/
theorem negative_if_error_small [Nonempty O] (n : ℕ) (B : ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u h : J → O → State) (hu : u ∈ jointDomain B)
    (hg : 2 * B * peak (jointCoefficient n f xs target u - h) < estimatedGap B u h) :
    ∃ v ∈ jointDomain B, linearCost (v - u) (jointCoefficient n f xs target u) < 0 := by
  refine ⟨proposal B h (selected h),
    proposal_mem B ((size_nonneg u).trans hu) h (selected h), ?_⟩
  have hd := actual_direction_upper n B f xs target u h hu
  linarith

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate
