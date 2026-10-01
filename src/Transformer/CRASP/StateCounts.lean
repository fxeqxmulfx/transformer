/-
# Counting a finite partition of the activation states

arXiv:2506.16055v3, Appendix B.2, construction of the attention sums in
`thm:rtfr_to_TLCl`. A formula for each state turns every integer-valued
function of that state into a weighted past count.
-/

import Transformer.CRASP.LinearCounts

namespace Transformer.CRASP.LinearCount

universe u v
variable {σ : Type u} {α : Type v} [Fintype α]

/-- A constant BOS contribution and the weighted counts of ordinary states.
Source: arXiv:2506.16055v3, Appendix B.2, the terms `A` and `B`. -/
noncomputable def states (ψ : α → Form σ) (b : α) (f : α → ℤ) : LinearCount σ :=
  ⟨f b, Finset.univ.toList.map fun a => (f a, ψ a)⟩

/-- The state sum uses the bound of its state predicates (Appendix B.2). -/
theorem good_states {ψ : α → Form σ} {k : ℕ} (hψ : ∀ a, ψ a ∈ TLCl σ k)
    (b : α) (f : α → ℤ) : (states ψ b f).Good k := by
  intro x hx
  obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hx
  exact hψ a

variable [DecidableEq σ] [DecidableEq α]

/-- A past count is the sum of the indicators on `1,...,i` (Appendix B.2). -/
theorem val_countL_sum (φ : Form σ) (w : List σ) (i : ℕ) :
    (Term.countL φ |>.val w i : ℤ) =
      ∑ j ∈ Finset.Icc 1 i, if φ.sat w j then (1 : ℤ) else 0 := by
  rw [Term.val, Nat.Icc_eq_range', Nat.add_sub_cancel]
  rw [Finset.sum_eq_multiset_sum]
  change ((List.range' 1 i).filter fun j => φ.sat w j).length =
    ((List.range' 1 i).map fun j => if φ.sat w j then (1 : ℤ) else 0).sum
  induction List.range' 1 i with
  | nil => rfl
  | cons j L ih => cases hj : φ.sat w j <;> simp [hj, ih, add_comm]

/-- Weighted state counts recover the sum of any function of the states.
Source: arXiv:2506.16055v3, Appendix B.2, the attention numerator and denominator. -/
theorem val_states {ψ : α → Form σ} (b : α) (f : α → ℤ)
    (w : List σ) (i : ℕ) (q : ℕ → α)
    (hψ : ∀ a j, j ∈ Finset.Icc 1 i → (ψ a).sat w j = decide (q j = a)) :
    (states ψ b f).val w i = f b + ∑ j ∈ Finset.Icc 1 i, f (q j) := by
  have hlist (g : α → ℤ) : (Finset.univ.toList.map g).sum = ∑ a, g a := by
    rw [← List.sum_toFinset _ (Finset.nodup_toList _), Finset.toList_toFinset]
  simp only [states, val, List.map_map, Function.comp_def]
  rw [hlist]
  congr 1
  simp_rw [val_countL_sum, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  simp_rw [hψ _ j hj]
  simp

/-- State predicates and their agreement hypothesis have witnesses (B.2). -/
example : (∀ _ : Unit, (Form.sym true : Form Bool) ∈ TLCl Bool 0) ∧
    (∀ a : Unit, ∀ j ∈ Finset.Icc 1 1,
      (Form.sym true).sat [true] j = decide ((() : Unit) = a)) := by
  constructor
  · intro a; exact ⟨rfl, rfl, le_rfl⟩
  · intro a j hj
    have hj1 : j = 1 := by have := Finset.mem_Icc.mp hj; omega
    subst j
    cases a
    decide

omit [DecidableEq α] in
/-- Nonnegative state contributions give a nonnegative sum (Appendix B.2). -/
theorem val_states_nonneg (ψ : α → Form σ) (b : α) (f : α → ℤ)
    (hf : ∀ a, 0 ≤ f a) (w : List σ) (i : ℕ) : 0 ≤ (states ψ b f).val w i := by
  apply add_nonneg (hf b)
  apply List.sum_nonneg
  intro x hx
  obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
  obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hy
  exact mul_nonneg (hf a) (Int.natCast_nonneg _)

/-- The nonnegative-contribution hypothesis has a witness (Appendix B.2). -/
example : ∀ _ : Unit, (0 : ℤ) ≤ 1 := fun _ => zero_le_one

end Transformer.CRASP.LinearCount
