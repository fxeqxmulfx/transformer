/-
# Turning a majority into a temporal comparison

arXiv:2506.16055v3, Appendix E, `thm:majtwo_to_tlc`: twice the mass of
the positive rectangles exceeds the number of pairs plus twice the mass
of the negative rectangles.
-/

import Transformer.CRASP.MajTwoRectCountSemantics

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- Sum of the rectangle counts for a list (Appendix E, `C_y`). -/
def MajRects.countList (v : Var) (L : List (MajRect σ)) : TermX σ :=
  TermX.sum (L.map fun r => r.count v)

/-- Positive votes of a finite family of binary formulas (Appendix E). -/
def MajRects.positiveCount {m : ℕ} (v : Var) (R : Fin (m + 1) → MajRects σ) : TermX σ :=
  TermX.sum (List.ofFn fun t => countList v (R t).positive)

/-- Negative votes of the family (Appendix E). -/
def MajRects.negativeCount {m : ℕ} (v : Var) (R : Fin (m + 1) → MajRects σ) : TermX σ :=
  TermX.sum (List.ofFn fun t => countList v (R t).negative)

/-- The total number `|w| (m + 1)` of quantified pairs (Appendix E). -/
def MajRects.pairCount (m : ℕ) : TermX σ :=
  TermX.sum (List.replicate (m + 1) (.countAll (Form.topAt 0).toX))

/-- The majority comparison before sugar elimination (Appendix E). -/
def MajRects.majorityX {m : ℕ} (v : Var) (R : Fin (m + 1) → MajRects σ) : FormX σ :=
  .lt (.add (pairCount m) (.add (negativeCount v R) (negativeCount v R)))
    (.add (positiveCount v R) (positiveCount v R))

/-- The temporal formula for the majority quantifier (Appendix E). -/
def MajRects.majority {m : ℕ} (v : Var) (R : Fin (m + 1) → MajRects σ) : Form σ :=
  (majorityX v R).elim

variable [DecidableEq σ]

/-- Counting a list agrees with the sum of its rectangle indicators. -/
theorem MajRects.val_countList (v : Var) (L : List (MajRect σ)) (w : List σ)
    (ξ : Var → ℕ) (hi : 1 ≤ ξ v.other) (hn : ξ v.other ≤ w.length) :
    ((countList v L).val w (ξ v.other) : ℤ) =
      ∑ j ∈ Finset.Icc 1 w.length,
        (L.map fun r => r.value w ((Function.update ξ v j) .x)
          ((Function.update ξ v j) .y)).sum := by
  rw [countList, TermX.val_sum, Nat.cast_list_sum]
  simp only [List.map_map, Function.comp_def]
  simp_rw [← MajRect.sum_value_count _ w ξ v hi hn]
  induction L with
  | nil => simp
  | cons r L ih => simp [List.map_cons, List.sum_cons, Finset.sum_add_distrib, ih]

/-- The total pair count is `|w| (m + 1)` (Appendix E). -/
@[simp] theorem MajRects.val_pairCount (m : ℕ) (w : List σ) (i : ℕ) :
    (pairCount m : TermX σ).val w i = w.length * (m + 1) := by
  simp [pairCount, TermX.val_sum, TermX.val, Form.sat_toX, Nat.mul_comm]

/-- The majority formula compares the original signed vote sum (Appendix E). -/
theorem MajRects.sat_majority {m : ℕ} (v : Var) (R : Fin (m + 1) → MajRects σ)
    (w : List σ) (ξ : Var → ℕ) (hi : 1 ≤ ξ v.other) (hn : ξ v.other ≤ w.length) :
    (majority v R).sat w (ξ v.other) =
      decide ((w.length * (m + 1) : ℤ) <
        2 * ∑ j ∈ Finset.Icc 1 w.length, ∑ t : Fin (m + 1),
          (R t).value w ((Function.update ξ v j) .x) ((Function.update ξ v j) .y)) := by
  have hp : ((positiveCount v R).val w (ξ v.other) : ℤ) =
      ∑ t : Fin (m + 1), ∑ j ∈ Finset.Icc 1 w.length,
        ((R t).positive.map fun r => r.value w ((Function.update ξ v j) .x)
          ((Function.update ξ v j) .y)).sum := by
    rw [positiveCount, TermX.val_sum, Nat.cast_list_sum]
    simp only [List.map_ofFn, List.sum_ofFn]
    apply Finset.sum_congr rfl
    intro t ht
    exact val_countList v (R t).positive w ξ hi hn
  have hm : ((negativeCount v R).val w (ξ v.other) : ℤ) =
      ∑ t : Fin (m + 1), ∑ j ∈ Finset.Icc 1 w.length,
        ((R t).negative.map fun r => r.value w ((Function.update ξ v j) .x)
          ((Function.update ξ v j) .y)).sum := by
    rw [negativeCount, TermX.val_sum, Nat.cast_list_sum]
    simp only [List.map_ofFn, List.sum_ofFn]
    apply Finset.sum_congr rfl
    intro t ht
    exact val_countList v (R t).negative w ξ hi hn
  rw [majority, FormX.sat_elim w _ _ hn]
  simp only [majorityX, FormX.sat, TermX.val, val_pairCount]
  apply decide_eq_decide.mpr
  have hcast (a b : ℕ) : a < b ↔ (a : ℤ) < (b : ℤ) := by omega
  rw [hcast]
  push_cast
  rw [hp, hm]
  simp only [MajRects.value, Finset.sum_sub_distrib]
  rw [Finset.sum_comm (s := Finset.Icc 1 w.length),
    Finset.sum_comm (s := Finset.Icc 1 w.length)]
  omega

/-- All position hypotheses used above have a witness (Appendix E). -/
example : 1 ≤ (fun _ : Var => 1) Var.x ∧ (fun _ : Var => 1) Var.x ≤ [true].length :=
  ⟨le_rfl, le_rfl⟩

end Transformer.CRASP
