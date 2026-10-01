/-
# Semantics of the rectangle counts

arXiv:2506.16055v3, Appendix E, proof of `thm:majtwo_to_tlc`:
`C_y` counts each order region with a strict counting operator, and counts
equality with the current-position indicator.
-/

import Transformer.CRASP.MajTwoRectCount

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]

/-- A Boolean filter counts the sum of its integer indicators (Appendix E). -/
theorem sum_indicator_filter (L : List ℕ) (f : ℕ → Bool) :
    (L.map fun j => if f j then (1 : ℤ) else 0).sum = ((L.filter f).length : ℤ) := by
  induction L with
  | nil => rfl
  | cons j L ih => cases h : f j <;> simp [h, ih, add_comm]

/-- The paper's position sum is the sum over the one-based range (Appendix E). -/
theorem sum_Icc_eq_range (n : ℕ) (f : ℕ → ℤ) :
    (∑ j ∈ Finset.Icc 1 n, f j) = ((List.range' 1 n).map f).sum := by
  rw [Nat.Icc_eq_range', Finset.sum_eq_multiset_sum]
  simp only [Nat.add_sub_cancel]
  rfl

/-- Split the valid positions into those before, at, and after `i` (Appendix E). -/
theorem range'_split_position {i n : ℕ} (hi : 1 ≤ i) (hn : i ≤ n) :
    List.range' 1 n = List.range' 1 (i - 1) ++ i :: List.range' (i + 1) (n - i) := by
  calc
    List.range' 1 n = List.range' 1 ((i - 1) + (n - i + 1)) := by congr 1; omega
    _ = List.range' 1 (i - 1) ++ List.range' (1 + (i - 1)) (n - i + 1) :=
      List.range'_append_1.symm
    _ = _ := by rw [show 1 + (i - 1) = i by omega, List.range'_succ]

/-- A region which is constant on a range leaves only its unary count. -/
theorem MajRect.sum_of_region (r : MajRect σ) (w : List σ) (i : ℕ) (L : List ℕ)
    (b : Bool) (h : ∀ j ∈ L, r.region i j = b) :
    (L.map fun j => r.value w i j).sum =
      (if r.left.sat w i && b then (1 : ℤ) else 0) *
        ((L.filter fun j => r.right.sat w j).length : ℤ) := by
  rw [← sum_indicator_filter, ← List.sum_map_mul_left]
  apply congrArg List.sum
  apply List.map_congr_left
  intro j hj
  rw [MajRect.value, h j hj]
  cases r.left.sat w i <;> cases r.right.sat w j <;> cases b <;> rfl

/-- `C_y` agrees with the actual number of satisfying positions (Appendix E). -/
theorem MajRect.sum_value_countY (r : MajRect σ) (w : List σ) (i : ℕ)
    (hi : 1 ≤ i) (hn : i ≤ w.length) :
    ((List.range' 1 w.length).map fun j => r.value w i j).sum =
      (r.countY.val w i : ℤ) := by
  have hL : ∀ j ∈ List.range' 1 (i - 1), r.region i j = r.after := by
    intro j hj
    rw [List.mem_range'_1] at hj
    simp [MajRect.region, show ¬ i < j by omega, show ¬ i = j by omega]
  have hR : ∀ j ∈ List.range' (i + 1) (w.length - i), r.region i j = r.before := by
    intro j hj
    rw [List.mem_range'_1] at hj
    simp [MajRect.region, show i < j by omega]
  rw [range'_split_position hi hn, List.map_append, List.sum_append,
    List.map_cons, List.sum_cons, r.sum_of_region w i _ r.after hL,
    r.sum_of_region w i _ r.before hR]
  have heq : r.value w i i =
      if r.left.sat w i && r.right.sat w i && r.equal then (1 : ℤ) else 0 := by
    simp [MajRect.value, MajRect.region]
  rw [heq]
  simp only [MajRect.countY, TermX.val, Form.sat_toX, TermX.val_zeroCount,
    TermX.val_sum, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    TermX.val_choose, TermX.val]
  cases r.left.sat w i <;> cases r.right.sat w i <;> cases r.before <;>
    cases r.equal <;> cases r.after <;> simp <;> omega

/-- Counting either variable uses the position named by the other (Appendix E). -/
theorem MajRect.sum_value_count (r : MajRect σ) (w : List σ) (ξ : Var → ℕ) (v : Var)
    (hi : 1 ≤ ξ v.other) (hn : ξ v.other ≤ w.length) :
    (∑ j ∈ Finset.Icc 1 w.length,
      r.value w ((Function.update ξ v j) .x) ((Function.update ξ v j) .y)) =
        (r.count v |>.val w (ξ v.other) : ℤ) := by
  rw [sum_Icc_eq_range]
  cases v
  · simp only [Var.other, MajRect.count, Function.update_self, Function.update_of_ne
      (show Var.y ≠ Var.x by decide)] at hi hn ⊢
    simpa only [MajRect.value_transpose] using r.transpose.sum_value_countY w (ξ .y) hi hn
  · simp only [Var.other, MajRect.count, Function.update_self, Function.update_of_ne
      (show Var.x ≠ Var.y by decide)] at hi hn ⊢
    exact r.sum_value_countY w (ξ .x) hi hn

/-- The position and region hypotheses above are satisfiable (Appendix E). -/
example : 1 ≤ 1 ∧ 1 ≤ [true].length ∧
    (∀ j ∈ [1], (MajRect.one : MajRect Bool).region 1 j = true) := by
  simp [MajRect.region, MajRect.one]

end Transformer.CRASP
