module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement.Basic

/-!
# A checkable certificate for one assignment update

Source: `review/flat_spectrum.md`, CM23. Feasible dual row/column potentials
and equality on a permutation certify its minimum assignment cost. This is
one coordinate update, not a global oracle for phase and decoder discovery.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement

open Finset

variable {I S C : Type*} [Fintype I]

/-- Summing feasible dual inequalities bounds every permutation cost (CM23). -/
theorem assignment_lower (cost : I → I → ℝ) (u v : I → ℝ)
    (hdual : ∀ i j, u i + v j ≤ cost i j) (σ : Equiv.Perm I) :
    (∑ i, u i) + (∑ j, v j) ≤ ∑ i, cost i (σ i) := by
  calc
    _ = ∑ i, (u i + v (σ i)) := by rw [sum_add_distrib, Equiv.sum_comp σ v]
    _ ≤ _ := sum_le_sum fun i _ => hdual i (σ i)

/-- Tightness on a permutation proves exact coordinate optimality from an
independently checkable dual certificate (CM23). -/
theorem assignment_minimum (cost : I → I → ℝ) (u v : I → ℝ)
    (hdual : ∀ i j, u i + v j ≤ cost i j) (π : Equiv.Perm I)
    (htight : ∀ i, cost i (π i) = u i + v (π i)) (σ : Equiv.Perm I) :
    (∑ i, cost i (π i)) ≤ ∑ i, cost i (σ i) := by
  have he : (∑ i, cost i (π i)) = (∑ i, u i) + (∑ j, v j) := by
    simp_rw [htight]
    rw [sum_add_distrib, Equiv.sum_comp π v]
  rw [he]
  exact assignment_lower cost u v hdual σ

variable [DecidableEq I] [Fintype S] [DecidableEq C]

/-- The actual observed mismatch matrix for an axis update (CM23). -/
noncomputable def assignmentCost (axis : S → I) (prediction : S → I → C)
    (labels : S → C) (i j : I) : ℝ :=
  ∑ t, if axis t = i then mistake (prediction t j) (labels t) else 0

/-- The dual certificate applies to the actual observation count, not a
surrogate matrix detached from the prediction rule (CM23). -/
theorem assignment_errors (axis : S → I) (prediction : S → I → C)
    (labels : S → C) (u v : I → ℝ)
    (hdual : ∀ i j, u i + v j ≤ assignmentCost axis prediction labels i j)
    (π : Equiv.Perm I)
    (htight : ∀ i, assignmentCost axis prediction labels i (π i) = u i + v (π i))
    (σ : Equiv.Perm I) :
    errors (fun t => prediction t (π (axis t))) labels ≤
      errors (fun t => prediction t (σ (axis t))) labels := by
  have h := assignment_minimum (assignmentCost axis prediction labels) u v hdual π htight σ
  unfold assignmentCost at h
  rw [group_sum axis (fun t j => mistake (prediction t j) (labels t)) π,
    group_sum axis (fun t j => mistake (prediction t j) (labels t)) σ] at h
  exact h

/-- Two observations favor exchanging the old axis assignment (CM23). -/
def swapCost (i j : Bool) : ℝ := if i = j then 1 else 0
private theorem swap_dual (i j : Bool) : (0 : ℝ) + 0 ≤ swapCost i j := by
  cases i <;> cases j <;> norm_num [swapCost]
private theorem swap_tight (i : Bool) : swapCost i (Equiv.swap false true i) = 0 + 0 := by
  cases i <;> norm_num [swapCost]

example := assignment_lower swapCost (fun _ => 0) (fun _ => 0) swap_dual (Equiv.refl _)
example := assignment_minimum swapCost (fun _ => 0) (fun _ => 0) swap_dual
  (Equiv.swap false true) swap_tight (Equiv.refl _)
example := swap_dual false true
example := swap_tight false
example := assignment_errors (id : Bool → Bool) (fun _ j => j) not
  (fun _ => 0) (fun _ => 0)
  (by intro i j; cases i <;> cases j <;> norm_num [assignmentCost, mistake])
  (Equiv.swap false true)
  (by intro i; cases i <;> norm_num [assignmentCost, mistake]) (Equiv.refl _)

/-- A closed certificate achieves a strict improvement over the old assignment (CM23). -/
theorem swap_strict :
    (∑ i : Bool, swapCost i (Equiv.swap false true i)) < ∑ i : Bool, swapCost i i := by
  simp only [Fintype.sum_bool]
  norm_num [swapCost]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement
