module

public import Transformer.Grokking.FiniteRecovery.Examples

/-! The sign polynomial for binary modular sums up to 32 positions. The
32 half-integer factors are the leaves of a five-level multiplication tree.
The count is never a root, so its parity gives a strict decision. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Parity

open Finset

/-- Node i after r levels of pairwise multiplication, with leaves numbered 32 to 63. -/
noncomputable def value (m : ℕ) : ℕ → ℕ → ℝ
  | 0, i => (i : ℝ) - 32 + 1 / 2 - m
  | r + 1, i => value m r (2*i) * value m r (2*i+1)

private theorem value_product (m r i : ℕ) :
    value m r i = ∏ j ∈ range (2^r), (((2^r*i+j : ℕ) : ℝ) - 32 + 1/2 - m) := by
  induction r generalizing i with
  | zero => simp [value]
  | succ r ih =>
    rw [value, ih, ih, pow_succ, Nat.mul_two, prod_range_add]
    congr 1 <;> apply prod_congr rfl <;> intro j _ <;> push_cast <;> ring

/-- The absolute size of the root may vary, but its sign is exactly (-1)^m. -/
theorem root_signed (m : ℕ) (hm : m ≤ 32) :
    ∃ a : ℝ, 0 < a ∧ value m 5 1 = (-1 : ℝ)^m * a := by
  let g (j : ℕ) := (j : ℝ) + 1/2 - m
  have hv : value m 5 1 = ∏ j ∈ range 32, g j := by
    rw [value_product]
    norm_num [g, Nat.cast_add, add_sub_right_comm]
  let left := ∏ j ∈ range m, -g j
  let right := ∏ j ∈ range (32-m), g (m+j)
  have hl : 0 < left := prod_pos fun j hj => by
    have hj' : (j : ℝ) + 1 ≤ m := by exact_mod_cast mem_range.mp hj
    dsimp [g]
    linarith
  have hr : 0 < right := prod_pos fun j _ => by
    dsimp [g]
    push_cast
    linarith [Nat.cast_nonneg (α := ℝ) j]
  refine ⟨left * right, mul_pos hl hr, ?_⟩
  rw [hv, ← Nat.add_sub_of_le hm, prod_range_add, ← mul_assoc]
  change _ * right = _ * right
  congr 1
  rw [← card_range m, ← prod_const, ← prod_mul_distrib]
  simp [g]

/-- Number of one-tokens in the whole word. -/
def ones {n : ℕ} (x : Word (ZMod 2) n) : ℕ := ∑ j, (x j).val

theorem ones_le {n : ℕ} (x : Word (ZMod 2) n) : ones x ≤ n := by
  calc
    _ ≤ ∑ _j : Fin n, 1 := sum_le_sum fun j _ => by have := (x j).val_lt; omega
    _ = n := by simp

/-- The enumeration of the half words changes only the indexing of the phase table. -/
theorem modularSum_eq_ones {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) k)
    (x : Word (ZMod 2) (k+k)) : modularSum e x = (ones x : ZMod 2) := by
  obtain ⟨⟨u,v⟩, rfl⟩ := (enumeratedCut e).surjective x
  simp only [modularSum, Equiv.symm_apply_apply, ones, Nat.cast_sum, ZMod.natCast_zmod_val]
  rw [← finSumFinEquiv.sum_comp, Fintype.sum_sum_type]
  change _ = (∑ j, Sum.elim (e u) (e v) (finSumFinEquiv.symm (finSumFinEquiv (Sum.inl j)))) +
    ∑ j, Sum.elim (e u) (e v) (finSumFinEquiv.symm (finSumFinEquiv (Sum.inr j)))
  simp only [Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

example : ∃ a : ℝ, 0 < a ∧ value 17 5 1 = -a := by
  obtain ⟨a, ha, he⟩ := root_signed 17 (by decide)
  exact ⟨a, ha, by norm_num at he; exact he⟩

end Transformer.Grokking.FiniteRecovery.Argand.Parity
