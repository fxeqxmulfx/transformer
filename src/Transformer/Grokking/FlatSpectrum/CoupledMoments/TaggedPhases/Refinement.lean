module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases.Feasibility

/-!
# Observed-CE updates without permutation restrictions

Source: `review/flat_spectrum.md`, CM25, categorical capacity extension. Input
categories may share phases, and the input and latent cardinalities may differ.
Actual grouped mismatch counts certify each finite independent axis update.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases

open Finset Schatten ObservedRefinement

variable {S I J C : Type*} [Fintype S] [Fintype I] [DecidableEq I] [DecidableEq C]

/-- Actual observed errors for assigning latent choice `j` to input category
`i`, CM25. The phase type need not equal the input type. -/
noncomputable def axisCost (axis : S → I) (prediction : S → J → C)
    (target : S → C) (i : I) (j : J) : ℝ :=
  ∑ t, if axis t = i then mistake (prediction t j) (target t) else 0

/-- Grouped costs are exactly the original observed mismatch count, CM25,
including unused input categories and repeated observations. -/
theorem grouped_errors (axis : S → I) (prediction : S → J → C)
    (target : S → C) (choice : I → J) :
    (∑ i, axisCost axis prediction target i (choice i)) =
      errors (fun t => prediction t (choice (axis t))) target := by
  unfold axisCost errors
  rw [sum_comm]
  apply sum_congr rfl
  intro t ht
  simp

/-- Independently checked nonincreasing costs give actual observed descent
without a one-to-one assignment constraint, CM25. -/
theorem choice_descent (axis : S → I) (prediction : S → J → C)
    (target : S → C) (a b : I → J)
    (hcost : ∀ i, axisCost axis prediction target i (a i) ≤
      axisCost axis prediction target i (b i)) :
    errors (fun t => prediction t (a (axis t))) target ≤
      errors (fun t => prediction t (b (axis t))) target := by
  rw [← grouped_errors, ← grouped_errors]
  exact sum_le_sum fun i _ => hcost i

/-- Checking every choice in each row certifies the best possible update of
that axis, CM25. It does not certify joint global phase/decoder optimization. -/
theorem choice_minimum (axis : S → I) (prediction : S → J → C)
    (target : S → C) (a : I → J)
    (hmin : ∀ i j, axisCost axis prediction target i (a i) ≤
      axisCost axis prediction target i j) (b : I → J) :
    errors (fun t => prediction t (a (axis t))) target ≤
      errors (fun t => prediction t (b (axis t))) target := by
  exact choice_descent axis prediction target a b (fun i => hmin i (b i))

variable {q : ℕ} [NeZero q] [Fintype C]

/-- Free row-phase assignments preserve the spectral hull and decrease
ordinary CE whenever their actual row costs are no greater, CM25. -/
theorem free_row_step (d : ZMod q → C) (s a b : I → ZMod q)
    (cell : S → I × I) (target : S → C)
    (hcost : ∀ i, axisCost (fun t => (cell t).1)
      (fun t z => d (z + s (cell t).2)) target i (a i) ≤
        axisCost (fun t => (cell t).1) (fun t z => d (z + s (cell t).2)) target i (b i)) :
    pointMass (fun i j => d (a i + s j)) ∈
        DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card I) (fun _ => Fintype.card I) ∧
      loss cell target (pointMass (fun i j => d (a i + s j))) ≤
        loss cell target (pointMass (fun i j => d (b i + s j))) := by
  refine ⟨phase_member d a s, (loss_le_iff cell target _ _).mpr ?_⟩
  exact choice_descent _ _ target a b hcost

/-- The column counterpart permits repeated phases and a different latent
cardinality, with the same ordinary-CE and feasibility guarantee, CM25. -/
theorem free_column_step (d : ZMod q → C) (r a b : I → ZMod q)
    (cell : S → I × I) (target : S → C)
    (hcost : ∀ j, axisCost (fun t => (cell t).2)
      (fun t z => d (r (cell t).1 + z)) target j (a j) ≤
        axisCost (fun t => (cell t).2) (fun t z => d (r (cell t).1 + z)) target j (b j)) :
    pointMass (fun i j => d (r i + a j)) ∈
        DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card I) (fun _ => Fintype.card I) ∧
      loss cell target (pointMass (fun i j => d (r i + a j))) ≤
        loss cell target (pointMass (fun i j => d (r i + b j))) := by
  refine ⟨phase_member d r a, (loss_le_iff cell target _ _).mpr ?_⟩
  exact choice_descent _ _ target a b hcost

example := grouped_errors (id : Bool → Bool) (fun _ j : Bool => j) (fun _ => true) not
example := choice_descent (id : Bool → Bool) (fun _ j : Bool => j)
  (fun _ => true) (fun _ => true) id
  (by intro i; cases i <;> norm_num [axisCost, mistake])
example := choice_minimum (id : Bool → Bool) (fun _ j : Bool => j)
  (fun _ => true) (fun _ => true)
  (by intro i j; cases i <;> cases j <;> norm_num [axisCost, mistake]) id

example := free_row_step (id : ZMod 3 → ZMod 3) (fun _ : Bool => 0)
  (fun _ => 1) (fun i => if i then 1 else 0)
  (fun t : Bool => (t, false)) (fun _ => 1)
  (by intro i; cases i <;> norm_num [axisCost, mistake])
example := free_column_step (id : ZMod 3 → ZMod 3) (fun _ : Bool => 0)
  (fun _ => 1) (fun i => if i then 1 else 0)
  (fun t : Bool => (false, t)) (fun _ => 1)
  (by intro i; cases i <;> norm_num [axisCost, mistake])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases
