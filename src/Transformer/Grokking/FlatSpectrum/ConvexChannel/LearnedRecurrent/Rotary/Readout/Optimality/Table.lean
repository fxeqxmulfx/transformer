module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Basic

/-!
# Exact support minimum over every component and original class

Source: `review/convex/moments.md`, CM103. The complete product of row budgets
admits an explicit minimizer. No class, component or complex coordinate is
discarded in the support sum or in the competitor domain.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality

open Pair
open scoped BigOperators
variable {J O : Type*} [Fintype J] [Fintype O]

/-- Linear cost over every full complex row, CM103. -/
noncomputable def linearCost (u a : J → O → State) : ℝ :=
  ∑ j, ∑ c, pairing (u j c) (a j c)

/-- Complete support magnitude, with no small-coefficient truncation, CM103. -/
noncomputable def supportSize (a : J → O → State) : ℝ := ∑ j, ∑ c, ‖a j c‖

omit [Fintype J] [Fintype O] in
/-- The explicit row minimizers form an admissible complete table, CM103. -/
theorem bestTable_mem (a : J → O → State) :
    (fun j c => bestRow (a j c)) ∈ rowDomain J O :=
  fun j c => bestRow_bound (a j c)

/-- The complete explicit table attains the exact support minimum, CM103. -/
theorem bestTable_cost (a : J → O → State) :
    linearCost (fun j c => bestRow (a j c)) a = -supportSize a := by
  unfold linearCost supportSize
  simp_rw [bestRow_cost]
  simp only [Finset.sum_neg_distrib]

/-- A lower bound against all original rows jointly, CM103. -/
theorem table_lower (u a : J → O → State) (hu : u ∈ rowDomain J O) :
    -supportSize a ≤ linearCost u a := by
  have h := Finset.sum_le_sum (s := Finset.univ) (fun (j : J) _ =>
    Finset.sum_le_sum (s := Finset.univ) (fun (c : O) _ =>
      pairing_lower (u j c) (a j c) (hu j c)))
  simpa only [Finset.sum_neg_distrib, linearCost, supportSize] using h

/-- Exact joint linear minimization over every feasible table, CM103. -/
theorem bestTable_minimizes (a u : J → O → State) (hu : u ∈ rowDomain J O) :
    linearCost (fun j c => bestRow (a j c)) a ≤ linearCost u a := by
  rw [bestTable_cost]
  exact table_lower u a hu

/-- Complete table displacements preserve the full contraction, CM103. -/
theorem linearCost_sub (v u a : J → O → State) :
    linearCost (v - u) a = linearCost v a - linearCost u a := by
  simp only [linearCost, Pi.sub_apply, pairing_sub, Finset.sum_sub_distrib]

example : -supportSize (fun (_ : Fin 2) (_ : Bool) => (Complex.I, (2 : ℂ))) ≤
    linearCost (fun (_ : Fin 2) (_ : Bool) => (Complex.I, 0))
      (fun _ _ => (Complex.I, 2)) :=
  table_lower _ _ (by intro j c; simp [rowSize])

example : linearCost (fun (_ : Fin 2) (_ : Bool) => bestRow (Complex.I, (2 : ℂ)))
    (fun _ _ => (Complex.I, 2)) ≤
      linearCost (fun (_ : Fin 2) (_ : Bool) => (Complex.I, 0))
        (fun _ _ => (Complex.I, 2)) :=
  bestTable_minimizes _ _ (by intro j c; simp [rowSize])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality
