/-
# Counting rectangles

arXiv:2506.16055v3, Appendix E, the transformation `C_y` in the proof
of `thm:majtwo_to_tlc`. Its three order regions become strict left counts,
the current-position indicator, and strict right counts.
-/

import Transformer.CRASP.MajTwoRectBounds

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- Zero, as the count of an always-false formula (Appendix A.3). -/
def TermX.zeroCount : TermX σ := .countAll (.lt .one .one)

/-- The sum of extended terms, with the empty sum represented by a count. -/
def TermX.sum (L : List (TermX σ)) : TermX σ := L.foldr .add TermX.zeroCount

/-- Keeping or dropping an order region in `C_y` (Appendix E). -/
def TermX.choose (b : Bool) (t : TermX σ) : TermX σ := if b then t else TermX.zeroCount

/-- Interchanging the two variables interchanges left and right order regions. -/
def MajRect.transpose (r : MajRect σ) : MajRect σ :=
  ⟨r.right, r.left, r.after, r.equal, r.before⟩

/-- The number of positions `y` in one rectangle, at a fixed `x` (Appendix E). -/
def MajRect.countY (r : MajRect σ) : TermX σ :=
  .cond r.left.toX
    (TermX.sum [TermX.choose r.before (.countRStrict r.right.toX),
      TermX.choose r.equal (.cond r.right.toX .one TermX.zeroCount),
      TermX.choose r.after (.countLStrict r.right.toX)]) TermX.zeroCount

/-- Counting the variable named by the majority quantifier (Appendix E). -/
def MajRect.count (r : MajRect σ) (v : Var) : TermX σ :=
  match v with
  | .x => r.transpose.countY
  | .y => r.countY

/-- Counting adds at most one level (Appendix E). -/
theorem TermX.depth_sum_le {L : List (TermX σ)} {d : ℕ} (hd : 1 ≤ d)
    (hL : ∀ t ∈ L, t.depth ≤ d) : (TermX.sum L).depth ≤ d := by
  induction L with
  | nil => exact hd
  | cons t L ih =>
      simp only [TermX.sum, List.foldr_cons, TermX.depth, max_le_iff]
      exact ⟨hL t (List.mem_cons_self ..), ih fun u hu => hL u (List.mem_cons_of_mem _ hu)⟩

/-- Summation preserves the plain-logic fragment (Appendix E). -/
theorem TermX.pnpFree_sum {L : List (TermX σ)} (hL : ∀ t ∈ L, t.pnpFree = true) :
    (TermX.sum L).pnpFree = true := by
  induction L with
  | nil => rfl
  | cons t L ih =>
      simp only [TermX.sum, List.foldr_cons, TermX.pnpFree, Bool.and_eq_true]
      exact ⟨hL t (List.mem_cons_self ..), ih fun u hu => hL u (List.mem_cons_of_mem _ hu)⟩

/-- Transposition preserves the unary conditions (Appendix E). -/
theorem MajRect.good_transpose {r : MajRect σ} {d : ℕ} (hr : r.Good d) :
    r.transpose.Good d := ⟨hr.2.1, hr.1, hr.2.2.2, hr.2.2.1⟩

/-- The rectangle count has depth at most one more than its unary tests. -/
theorem MajRect.depth_countY_le {r : MajRect σ} {d : ℕ} (hr : r.Good d) :
    r.countY.depth ≤ d + 1 := by
  have hrd := hr.2.2.2
  have hchoose : ∀ (b : Bool) (t : TermX σ), t.depth ≤ d + 1 →
      (TermX.choose b t).depth ≤ d + 1 := by
    intro b t ht
    cases b <;> simp [TermX.choose, TermX.zeroCount, FormX.depth, TermX.depth, ht]
  have hs : (TermX.sum [TermX.choose r.before (.countRStrict r.right.toX),
      TermX.choose r.equal (.cond r.right.toX .one TermX.zeroCount),
      TermX.choose r.after (.countLStrict r.right.toX)]).depth ≤ d + 1 := by
    apply TermX.depth_sum_le (by omega)
    simp only [List.mem_cons, List.not_mem_nil, or_false]
    rintro t (rfl | rfl | rfl)
    · apply hchoose
      simpa only [TermX.depth, Form.depth_toX] using Nat.add_le_add_right hrd 1
    · apply hchoose
      simp only [TermX.depth, Form.depth_toX, max_le_iff]
      exact ⟨hrd.trans (Nat.le_succ d), Nat.zero_le _, by simp [TermX.zeroCount,
        TermX.depth, FormX.depth]⟩
    · apply hchoose
      simpa only [TermX.depth, Form.depth_toX] using Nat.add_le_add_right hrd 1
  simp only [MajRect.countY, TermX.depth, Form.depth_toX, max_le_iff]
  exact ⟨hr.2.2.1.trans (Nat.le_succ d), hs,
    by simp [TermX.zeroCount, TermX.depth, FormX.depth]⟩

/-- Counting a rectangle introduces no Parikh predicate (Appendix E). -/
theorem MajRect.pnpFree_countY {r : MajRect σ} {d : ℕ} (hr : r.Good d) :
    r.countY.pnpFree = true := by
  have hchoose : ∀ (b : Bool) (t : TermX σ), t.pnpFree = true →
      (TermX.choose b t).pnpFree = true := by
    intro b t ht
    cases b <;> simp [TermX.choose, ht, TermX.zeroCount, TermX.pnpFree, FormX.pnpFree]
  have hs : (TermX.sum [TermX.choose r.before (.countRStrict r.right.toX),
      TermX.choose r.equal (.cond r.right.toX .one TermX.zeroCount),
      TermX.choose r.after (.countLStrict r.right.toX)]).pnpFree = true := by
    apply TermX.pnpFree_sum
    simp only [List.mem_cons, List.not_mem_nil, or_false]
    rintro t (rfl | rfl | rfl) <;> apply hchoose <;>
      simp [TermX.pnpFree, TermX.zeroCount, FormX.pnpFree, hr.2.1]
  simp only [MajRect.countY, TermX.pnpFree, Form.pnpFree_toX, hr.1, hs, Bool.true_and]
  rfl

/-- The bounds and fragment hypotheses have witnesses (Appendix E). -/
example : 1 ≤ 1 ∧ (∀ t ∈ [TermX.one], (t : TermX Bool).depth ≤ 1) ∧
    (∀ t ∈ [TermX.one], (t : TermX Bool).pnpFree = true) ∧
    (MajRect.one : MajRect Bool).Good 0 := by
  simp [TermX.depth, TermX.pnpFree, MajRect.good_one]

variable [DecidableEq σ]

/-- Counting a false predicate gives the additive identity (Appendix A.3). -/
@[simp] theorem TermX.val_zeroCount (w : List σ) (i : ℕ) :
    (TermX.zeroCount : TermX σ).val w i = 0 := by
  simp [TermX.zeroCount, TermX.val, FormX.sat]

/-- Evaluation of a sum is the sum of evaluations (Appendix E). -/
theorem TermX.val_sum (L : List (TermX σ)) (w : List σ) (i : ℕ) :
    (TermX.sum L).val w i = (L.map fun t => t.val w i).sum := by
  induction L with
  | nil => exact TermX.val_zeroCount w i
  | cons t L ih =>
      change t.val w i + (TermX.sum L).val w i = _
      rw [ih, List.map_cons, List.sum_cons]

/-- Evaluation of a static order-region choice (Appendix E). -/
theorem TermX.val_choose (b : Bool) (t : TermX σ) (w : List σ) (i : ℕ) :
    (TermX.choose b t).val w i = if b then t.val w i else 0 := by
  cases b <;> simp [TermX.choose]

/-- Transposition interchanges the two positions (Appendix E). -/
theorem MajRect.value_transpose (r : MajRect σ) (w : List σ) (i j : ℕ) :
    r.transpose.value w i j = r.value w j i := by
  have hreg : r.transpose.region i j = r.region j i := by
    by_cases h : i < j
    · simp [MajRect.region, MajRect.transpose, h, show ¬ j < i by omega,
        show ¬ j = i by omega]
    · by_cases he : i = j
      · subst j; simp [MajRect.region, MajRect.transpose]
      · simp [MajRect.region, MajRect.transpose, h, he, show j < i by omega]
  simp only [MajRect.value, hreg]
  simp only [MajRect.transpose, Bool.and_comm (r.right.sat w i) (r.left.sat w j)]

end Transformer.CRASP
