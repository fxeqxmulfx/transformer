module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Coefficient

/-!
# A training-only conditional ordinary-CE optimality gap

Source: `review/flat_spectrum.md`, CM103. The gap bounds possible mean CE
improvement over every complete readout table on current learned states. This
exact real-arithmetic result neither certifies floating-point diagnostics nor
global factor optimization, raw accuracy or unseen-answer identification.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality

open Pair
open scoped BigOperators
variable {J O : Type*} [Fintype J] [Fintype O] [DecidableEq O]

/-- Complete computed training CE gap on the actual learned states, CM103. -/
noncomputable def gap (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) : ℝ :=
  linearCost u (coefficient n B w f xs target u) +
    supportSize (coefficient n B w f xs target u)

/-- Every feasible current readout table has nonnegative gap, CM103. -/
theorem gap_nonneg (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (hu : u ∈ rowDomain J O) : 0 ≤ gap n B w f xs target u := by
  have h := table_lower u (coefficient n B w f xs target u) hu
  unfold gap
  linarith

/-- The actual mean CE minus the explicit gap is a lower bound for every
feasible complete replacement, CM103. -/
theorem loss_lower_bound (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u v : J → O → State) (hv : v ∈ rowDomain J O) :
    meanLoss n B w f xs target u - gap n B w f xs target u ≤
      meanLoss n B w f xs target v := by
  have hfirst := mean_first_order n B w f xs target u v
  have hlower := table_lower v (coefficient n B w f xs target u) hv
  rw [linearCost_sub] at hfirst
  unfold gap
  linarith

/-- No feasible conditional readout improvement exceeds the computed gap, CM103. -/
theorem improvement_le_gap (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u v : J → O → State) (hv : v ∈ rowDomain J O) :
    meanLoss n B w f xs target u - meanLoss n B w f xs target v ≤
      gap n B w f xs target u := by
  have h := loss_lower_bound n B w f xs target u v hv
  linarith

/-- A zero gap certifies the conditional minimum against every feasible table,
without assuming an optimal endpoint, CM103. -/
theorem optimal_of_gap_zero (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (hg : gap n B w f xs target u = 0)
    (v : J → O → State) (hv : v ∈ rowDomain J O) :
    meanLoss n B w f xs target u ≤ meanLoss n B w f xs target v := by
  have h := loss_lower_bound n B w f xs target u v hv
  rwa [hg, sub_zero] at h

/-- The explicit feasible minimizing table has linear change minus the gap,
including every original class and full complex coordinate, CM103. -/
theorem best_change (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) :
    linearCost ((fun j c => bestRow (coefficient n B w f xs target u j c)) - u)
      (coefficient n B w f xs target u) = -gap n B w f xs target u := by
  rw [linearCost_sub, bestTable_cost]
  unfold gap
  ring

/-- A positive gap supplies a feasible direction with strictly negative actual
first-order contraction. No arbitrary finite-step decrease is claimed, CM103. -/
theorem negative_direction (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O)
    (u : J → O → State) (hg : 0 < gap n B w f xs target u) :
    ∃ v ∈ rowDomain J O, linearCost (v - u) (coefficient n B w f xs target u) < 0 := by
  refine ⟨fun j c => bestRow (coefficient n B w f xs target u j c), bestTable_mem _, ?_⟩
  rw [best_change]
  linarith

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality
