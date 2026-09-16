/-
# Testing a fixed subsequence from its middle symbol

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix A, the second half of the proof of
`lem:piecewise_testable_depth`.

With `▷#` available a pattern of `2k + 1` symbols costs only `k + 1` levels:
guess the position `j` of the middle symbol `σ_{k+1}`, check that
`σ₁ ⋯ σ_k` fits into `w[1:j-1]` with the past-only formula of
`Transformer.CRASP.Subsequence`, and that `σ_{k+2} ⋯ σ_{2k+1}` fits into
`w[j+1:n]` with its mirror image, which consumes the pattern from the *left*
through `Form.exAfter`.  The guess itself is one more `◁#`.

As in the one-sided formula, both halves must quantify strictly — the paper's
`φ_L` and `φ_R` use `◁#[⋯] ≥ 1` and `▷#[⋯] ≥ 1`, which let a symbol of a half
reuse the position of the middle symbol.  Patterns shorter than `k + 1`
symbols have no middle to guess and are tested one-sidedly.
-/

import Transformer.CRASP.Subsequence

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-! ## The combinatorics of a pattern split at a position -/

/-- A pattern starting with `a` fits into the suffix `w[i+1:n]` exactly when
some later position `j` carries `a` and the rest of the pattern fits after
it. -/
theorem cons_sublist_drop (w t : List σ) (a : σ) (i : ℕ) :
    (a :: t).Sublist (w.drop i) ↔
      ∃ j, i + 1 ≤ j ∧ j ≤ w.length ∧ w[j - 1]? = some a ∧ t.Sublist (w.drop j) := by
  induction hm : w.length - i generalizing i with
  | zero =>
      rw [List.drop_eq_nil_of_le (by omega)]
      constructor
      · intro h
        simp at h
      · rintro ⟨j, hj1, hj2, -, -⟩
        omega
  | succ m ih =>
      have hi : i < w.length := by omega
      rw [List.drop_eq_getElem_cons hi, List.sublist_cons_iff, ih (i + 1) (by omega)]
      constructor
      · rintro (⟨j, hj1, hj2, hjw, hjt⟩ | ⟨r, hr, hrt⟩)
        · exact ⟨j, by omega, hj2, hjw, hjt⟩
        · obtain ⟨rfl, rfl⟩ := List.cons.inj hr
          exact ⟨i + 1, le_rfl, hi, by simp [hi], hrt⟩
      · rintro ⟨j, hj1, hj2, hjw, hjt⟩
        rcases Nat.lt_or_ge (i + 1) j with h | h
        · exact Or.inl ⟨j, h, hj2, hjw, hjt⟩
        · have hj : j = i + 1 := by omega
          subst hj
          refine Or.inr ⟨t, ?_, hjt⟩
          have hsome : w[i]? = some a := by simpa using hjw
          rw [List.getElem?_eq_getElem hi, Option.some.injEq] at hsome
          rw [hsome]

/-- A pattern `l ++ c :: r` fits into `w` exactly when some position `j`
carries `c`, `l` fits before it and `r` after it. -/
theorem append_cons_sublist_iff (w l r : List σ) (c : σ) :
    (l ++ c :: r).Sublist w ↔
      ∃ j, 1 ≤ j ∧ j ≤ w.length ∧ l.Sublist (w.take (j - 1)) ∧ w[j - 1]? = some c ∧
        r.Sublist (w.drop j) := by
  constructor
  · intro h
    rw [← List.singleton_append, ← List.append_assoc, List.append_sublist_iff] at h
    obtain ⟨w₁, w₂, rfl, h₁, h₂⟩ := h
    rw [← List.take_length (l := w₁), sublist_snoc_take] at h₁
    obtain ⟨j, hj1, hj2, hjw, hjt⟩ := h₁
    refine ⟨j, hj1, by simp; omega, ?_, ?_, ?_⟩
    · rwa [List.take_append_of_le_length (by omega)]
    · rwa [List.getElem?_append_left (by omega)]
    · rw [List.drop_append_of_le_length hj2]
      exact h₂.trans (List.sublist_append_right _ _)
  · rintro ⟨j, hj1, hj2, hl, hc, hr⟩
    have hlt : j - 1 < w.length := by omega
    have hw : w = w.take (j - 1) ++ c :: w.drop j := by
      have hd := List.drop_eq_getElem_cons hlt
      rw [Nat.sub_add_cancel hj1] at hd
      rw [List.getElem?_eq_getElem hlt, Option.some.injEq] at hc
      rw [← hc, ← hd, List.take_append_drop]
    rw [hw]
    exact hl.append (List.Sublist.cons_cons c hr)

/-! ## The formulas -/

/-- `subseqAfter s` holds at `i` when `s` fits into `w[i+1:n]`: the mirror
image of `subseqStrict`, consuming the pattern from the left. -/
def subseqAfter : List σ → Form σ
  | [] => .neg (.lt .one .one)
  | a :: t => Form.exAfter (.and (.sym a) (subseqAfter t))

/-- The two-sided test for `s` with `k` symbols to the left of the middle:
one-sided when `s` has no `(k+1)`-st symbol, and otherwise a guess of the
position of that symbol, with `subseqStrict` to its left and `subseqAfter` to
its right. -/
def subseqTwoSided (k : ℕ) (s : List σ) : Form σ :=
  match s.drop k with
  | [] => subseqAt s.reverse
  | c :: r => Form.exAt (.and (.and (subseqStrict (s.take k).reverse) (.sym c)) (subseqAfter r))

@[simp] theorem depth_subseqAfter (s : List σ) : (subseqAfter s).depth = s.length := by
  induction s with
  | nil => rfl
  | cons a t ih => simp [subseqAfter, Form.depth, ih]

@[simp] theorem pnpFree_subseqAfter (s : List σ) : (subseqAfter s).pnpFree = true := by
  induction s with
  | nil => rfl
  | cons a t ih => simp [subseqAfter, Form.pnpFree, ih]

/-- The middle symbol costs one level on top of the longer half, so a pattern
of `2k + 1` symbols is tested at depth `k + 1`. -/
theorem depth_subseqTwoSided_le (k : ℕ) (s : List σ) :
    (subseqTwoSided k s).depth ≤ max (k + 1) (s.length - k) := by
  unfold subseqTwoSided
  split
  · next h =>
      rw [List.drop_eq_nil_iff] at h
      simp only [depth_subseqAt, List.length_reverse]
      omega
  · next c r h =>
      have hlen := congrArg List.length h
      rw [List.length_drop, List.length_cons] at hlen
      simp only [Form.depth_exAt, Form.depth, depth_subseqStrict, List.length_reverse,
        List.length_take, depth_subseqAfter]
      omega

/-- The two-sided test uses no Parikh numerical predicate. -/
theorem pnpFree_subseqTwoSided (k : ℕ) (s : List σ) : (subseqTwoSided k s).pnpFree = true := by
  unfold subseqTwoSided
  split <;> simp [Form.pnpFree]

section Semantics

variable [DecidableEq σ]

/-- `subseqAfter s` holds at `i` exactly when `s` fits into `w[i+1:n]`. -/
theorem sat_subseqAfter (s : List σ) (w : List σ) (i : ℕ) :
    (subseqAfter s).sat w i = true ↔ s.Sublist (w.drop i) := by
  induction s generalizing i with
  | nil => simp [subseqAfter, Form.sat, Term.val]
  | cons a t ih =>
      rw [subseqAfter, Form.sat_exAfter, cons_sublist_drop]
      constructor
      · rintro ⟨j, hj1, hj2, hj⟩
        rw [Form.sat, Bool.and_eq_true, ih] at hj
        exact ⟨j, hj1, hj2, by simpa [Form.sat] using hj.1, hj.2⟩
      · rintro ⟨j, hj1, hj2, hjw, hjt⟩
        exact ⟨j, hj1, hj2, by
          rw [Form.sat, Bool.and_eq_true, ih]
          exact ⟨by simpa [Form.sat] using hjw, hjt⟩⟩

/-- The two-sided test defines the 𝒥-expression of `s`, for every split. -/
theorem lang_subseqTwoSided (k : ℕ) (s : List σ) :
    (subseqTwoSided k s).lang = {w : List σ | s.Sublist w} := by
  unfold subseqTwoSided
  split
  · rw [lang_subseqAt, List.reverse_reverse]
  · next c r h =>
      have hs : s = s.take k ++ c :: r := by rw [← h, List.take_append_drop]
      ext w
      rw [Form.lang, Set.mem_ofPred_eq, Form.models, Form.sat_exAt, Set.mem_ofPred_eq,
        show s.Sublist w ↔ (s.take k ++ c :: r).Sublist w by rw [← hs],
        append_cons_sublist_iff]
      simp only [Form.sat, Bool.and_eq_true, sat_subseqStrict, List.reverse_reverse,
        sat_subseqAfter, decide_eq_true_eq, and_assoc]

end Semantics

end CRASP
end Transformer
