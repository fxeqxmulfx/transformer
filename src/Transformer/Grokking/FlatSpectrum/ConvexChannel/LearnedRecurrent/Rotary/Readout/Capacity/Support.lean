module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Normalization

/-!
# Exact full linear minimization with shared component capacity

Source: `review/convex/moments.md`, CM104. Finite maximum attainment supplies
the selected component. All original classes and both complex coordinates
contribute; zeros in a minimizing value never shrink the feasible family.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair Optimality
open scoped BigOperators
variable {O : Type*} [Fintype O]

/-- Complete original-class coefficient support of one component, CM104. -/
noncomputable def blockSupport (a : O → State) : ℝ := ∑ c, ‖a c‖

/-- Every full block support is nonnegative, CM104. -/
theorem support_nonneg (a : O → State) : 0 ≤ blockSupport a :=
  Finset.sum_nonneg (fun c _ => norm_nonneg (a c))

/-- Complete contraction bound for an arbitrary weighted readout block, CM104. -/
theorem block_cost_lower [Nonempty O] (u a : O → State) :
    -(blockSize u * blockSupport a) ≤ ∑ c, pairing (u c) (a c) := by
  have h := Finset.sum_le_sum (s := Finset.univ) (fun (c : O) _ =>
    block_pairing_lower u c (a c))
  simpa only [Finset.sum_neg_distrib, ← Finset.mul_sum, blockSupport] using h

variable {J : Type*} [Fintype J] [Nonempty J]

/-- Maximum complete component support, including components at zero weight, CM104. -/
noncomputable def peak (a : J → O → State) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun j => blockSupport (a j))

/-- Every component enters the complete support comparison, CM104. -/
theorem support_le_peak (a : J → O → State) (j : J) : blockSupport (a j) ≤ peak a :=
  Finset.le_sup' (f := fun j => blockSupport (a j)) (Finset.mem_univ j)

/-- The full shared support maximum is nonnegative, CM104. -/
theorem peak_nonneg (a : J → O → State) : 0 ≤ peak a := by
  let j : J := Classical.choice inferInstance
  exact (support_nonneg (a j)).trans (support_le_peak a j)

/-- Finite comparison proves attainment; no optimizer oracle is a hypothesis, CM104. -/
theorem peak_attained (a : J → O → State) : ∃ j, blockSupport (a j) = peak a := by
  obtain ⟨j, hj, he⟩ := Finset.exists_mem_eq_sup' (s := Finset.univ)
    Finset.univ_nonempty (fun j => blockSupport (a j))
  exact ⟨j, he.symm⟩

/-- A component selected from the proved finite maximum; any tie is valid, CM104. -/
noncomputable def selected (a : J → O → State) : J := (peak_attained a).choose

/-- The selected component attains the full maximum over original components, CM104. -/
theorem selected_peak (a : J → O → State) : blockSupport (a (selected a)) = peak a :=
  (peak_attained a).choose_spec

/-- Global linear bound under the complete shared capacity, CM104. -/
theorem full_cost_lower [Nonempty O] (B : ℝ) (u a : J → O → State)
    (hu : u ∈ jointDomain B) :
    -(B * peak a) ≤ linearCost u a := by
  have hl := Finset.sum_le_sum (s := Finset.univ) (fun (j : J) _ =>
    block_cost_lower (u j) (a j))
  simp only [Finset.sum_neg_distrib] at hl
  have hp : (∑ j, blockSize (u j) * blockSupport (a j)) ≤ B * peak a := by
    calc
      _ ≤ ∑ j, blockSize (u j) * peak a := Finset.sum_le_sum (fun j _ =>
        mul_le_mul_of_nonneg_left (support_le_peak a j) (block_nonneg (u j)))
      _ = size u * peak a := by rw [← Finset.sum_mul]; rfl
      _ ≤ B * peak a := mul_le_mul_of_nonneg_right hu (peak_nonneg a)
  change -(B * peak a) ≤ ∑ j, ∑ c, pairing (u j c) (a j c)
  linarith

variable [DecidableEq J]

/-- One full feasible value concentrating capacity at a selected component,
without deleting any original coordinate, CM104. -/
noncomputable def proposal (B : ℝ) (a : J → O → State) (k : J) (j : J) (c : O) : State :=
  if j = k then B • bestRow (a k c) else 0

omit [Nonempty J] in
/-- The explicit complete proposal obeys the shared capacity, CM104. -/
theorem proposal_mem [Nonempty O] (B : ℝ) (hB : 0 ≤ B) (a : J → O → State) (k : J) :
    proposal B a k ∈ jointDomain B := by
  have hb : blockSize (fun c => B • bestRow (a k c)) ≤ B := by
    apply block_le
    intro c
    rw [rowSize_smul, abs_of_nonneg hB]
    simpa only [mul_one] using mul_le_mul_of_nonneg_left (bestRow_bound (a k c)) hB
  have h := Finset.sum_le_sum (s := Finset.univ) (fun (j : J) _ =>
    show blockSize (proposal B a k j) ≤ (if j = k then B else 0) from by
      change blockSize (fun c => if j = k then B • bestRow (a k c) else 0) ≤
        (if j = k then B else 0)
      by_cases hj : j = k
      · simpa only [ite_eq_left hj] using hb
      · simp only [ite_eq_right hj]
        exact le_of_eq block_zero)
  change size (proposal B a k) ≤ B
  simpa only [size, Finset.sum_ite_eq', Finset.mem_univ, ite_true] using h

omit [Nonempty J] in
/-- The proposal's exact complete contraction, including all original classes, CM104. -/
theorem proposal_cost (B : ℝ) (a : J → O → State) (k : J) :
    linearCost (proposal B a k) a = -(B * blockSupport (a k)) := by
  have h (j : J) : (∑ c, pairing (proposal B a k j c) (a j c)) =
      if j = k then -(B * blockSupport (a k)) else 0 := by
    by_cases hj : j = k
    · subst j
      simp [proposal, pairing_scale, bestRow_cost, blockSupport, ← Finset.mul_sum]
    · simp [proposal, hj, pairing, Pair.row_apply]
  simp only [linearCost, h]
  simp

/-- Finite selection and complete complex phase rows attain the support bound, CM104. -/
theorem selected_cost (B : ℝ) (a : J → O → State) :
    linearCost (proposal B a (selected a)) a = -(B * peak a) := by
  rw [proposal_cost, selected_peak]

/-- The explicit selected proposal minimizes against every shared-capacity
weighted readout table, CM104. -/
theorem selected_minimizes [Nonempty O] (B : ℝ) (a u : J → O → State)
    (hu : u ∈ jointDomain B) :
    linearCost (proposal B a (selected a)) a ≤ linearCost u a := by
  rw [selected_cost]
  exact full_cost_lower B u a hu

example : -((16 : ℝ) * peak (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 2))) ≤
    linearCost (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0))
      (fun _ _ => (Complex.I, 2)) :=
  full_cost_lower _ _ _ (by change size _ ≤ 16; simp [size, block_const, rowSize])

example : proposal 16 (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 2)) 0 ∈
    jointDomain 16 := proposal_mem _ (by norm_num) _ _

example : linearCost (proposal 16 (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 2))
    (selected (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 2))))
      (fun _ _ => (Complex.I, 2)) ≤
    linearCost (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0))
      (fun _ _ => (Complex.I, 2)) :=
  selected_minimizes _ _ _ (by change size _ ≤ 16; simp [size, block_const, rowSize])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
