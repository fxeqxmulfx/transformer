/-
# Testing a fixed subsequence inside `TL[◁#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.4, the proof of `lem:piecewise_testable_depth`.

A 𝒥-expression asks whether a fixed pattern `σ₁ ⋯ σ_k` occurs as a
subsequence.  The formula testing it consumes the pattern from the *right*:
`σ₁ ⋯ σ_k` fits into the prefix `w[1:i]` exactly when some position `j ≤ i`
carries `σ_k` and `σ₁ ⋯ σ_{k-1}` already fits into `w[1:j-1]`.  Every
quantifier is then a bounded one over the past, `Form.exAt` or
`Form.exBefore`, and each of the `k` symbols costs one counting operator,
which is what puts the formula in `TL[◁#]_k`.

Patterns are therefore carried in reverse, last symbol first: `subseqAt s`
tests for `s.reverse`.
-/

import Transformer.CRASP.BoundedExists

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-! ## The combinatorics of a pattern consumed from the right -/

/-- Appending a letter on both sides: a pattern ending in `a` fits into
`u ++ [x]` either already inside `u`, or by matching `a` against `x`. -/
theorem sublist_snoc_snoc (u t : List σ) (a x : σ) :
    (t ++ [a]).Sublist (u ++ [x]) ↔ (t ++ [a]).Sublist u ∨ (x = a ∧ t.Sublist u) := by
  rw [List.sublist_append_iff]
  constructor
  · rintro ⟨l₁, l₂, heq, h₁, h₂⟩
    rcases List.sublist_singleton.mp h₂ with rfl | rfl
    · rw [List.append_nil] at heq
      exact Or.inl (by rw [heq]; exact h₁)
    · have hlen : t.length = l₁.length := by
        have hl := congrArg List.length heq
        simp only [List.length_append, List.length_cons, List.length_nil] at hl
        omega
      obtain ⟨rfl, ha⟩ := List.append_inj heq hlen
      exact Or.inr ⟨by simpa using ha.symm, h₁⟩
  · rintro (h | ⟨hxa, h⟩)
    · exact ⟨t ++ [a], [], by simp, h, List.nil_sublist _⟩
    · subst hxa
      exact ⟨t, [x], rfl, h, List.Sublist.refl _⟩

/-- **The step the formula mirrors.**  A pattern ending in `a` is a
subsequence of the prefix `w[1:n]` exactly when some position `1 ≤ j ≤ n`
carries `a` and the rest of the pattern fits into `w[1:j-1]`. -/
theorem sublist_snoc_take (w t : List σ) (a : σ) (n : ℕ) :
    (t ++ [a]).Sublist (w.take n) ↔
      ∃ j, 1 ≤ j ∧ j ≤ n ∧ w[j - 1]? = some a ∧ t.Sublist (w.take (j - 1)) := by
  induction n with
  | zero =>
      simp only [List.take_zero, List.sublist_nil]
      constructor
      · intro h; simp at h
      · rintro ⟨j, hj1, hj2, -, -⟩; omega
  | succ n ih =>
      rw [List.take_add_one]
      cases hx : w[n]? with
      | none =>
          rw [Option.toList_none, List.append_nil, ih]
          constructor
          · rintro ⟨j, hj1, hj2, hjw, hjt⟩
            exact ⟨j, hj1, Nat.le_succ_of_le hj2, hjw, hjt⟩
          · rintro ⟨j, hj1, hj2, hjw, hjt⟩
            refine ⟨j, hj1, ?_, hjw, hjt⟩
            by_contra hlt
            have hj : j - 1 = n := by omega
            rw [hj, hx] at hjw
            exact absurd hjw (by simp)
      | some x =>
          rw [Option.toList_some, sublist_snoc_snoc, ih]
          constructor
          · rintro (⟨j, hj1, hj2, hjw, hjt⟩ | ⟨hxa, hts⟩)
            · exact ⟨j, hj1, Nat.le_succ_of_le hj2, hjw, hjt⟩
            · exact ⟨n + 1, Nat.le_add_left 1 n, le_rfl, by simpa [hxa] using hx, by simpa using hts⟩
          · rintro ⟨j, hj1, hj2, hjw, hjt⟩
            rcases Nat.lt_or_ge j (n + 1) with h | h
            · exact Or.inl ⟨j, hj1, by omega, hjw, hjt⟩
            · have hj : j = n + 1 := by omega
              subst hj
              refine Or.inr ⟨?_, by simpa using hjt⟩
              have hsome : w[n]? = some a := by simpa using hjw
              rw [hx] at hsome
              exact Option.some.inj hsome

/-! ## The pattern formulas -/

/-- `subseqStrict s` holds at `i` when `s.reverse` fits into `w[1:i-1]`. -/
def subseqStrict : List σ → Form σ
  | [] => .neg (.lt .one .one)
  | a :: t => Form.exBefore (.and (.sym a) (subseqStrict t))

/-- `subseqAt s` holds at `i` when `s.reverse` fits into `w[1:i]`. -/
def subseqAt : List σ → Form σ
  | [] => .neg (.lt .one .one)
  | a :: t => Form.exAt (.and (.sym a) (subseqStrict t))

@[simp] theorem depth_subseqStrict (s : List σ) : (subseqStrict s).depth = s.length := by
  induction s with
  | nil => rfl
  | cons a t ih => simp [subseqStrict, Form.depth, ih]

@[simp] theorem depth_subseqAt (s : List σ) : (subseqAt s).depth = s.length := by
  cases s with
  | nil => rfl
  | cons a t => simp [subseqAt, Form.depth]

@[simp] theorem past_subseqStrict (s : List σ) : (subseqStrict s).past = true := by
  induction s with
  | nil => rfl
  | cons a t ih => simp [subseqStrict, Form.past, ih]

@[simp] theorem past_subseqAt (s : List σ) : (subseqAt s).past = true := by
  cases s with
  | nil => rfl
  | cons a t => simp [subseqAt, Form.past]

@[simp] theorem pnpFree_subseqStrict (s : List σ) : (subseqStrict s).pnpFree = true := by
  induction s with
  | nil => rfl
  | cons a t ih => simp [subseqStrict, Form.pnpFree, ih]

@[simp] theorem pnpFree_subseqAt (s : List σ) : (subseqAt s).pnpFree = true := by
  cases s with
  | nil => rfl
  | cons a t => simp [subseqAt, Form.pnpFree]

section Semantics

variable [DecidableEq σ]

/-- `subseqStrict s` holds at `i` exactly when `s.reverse` is a subsequence of
`w[1:i-1]`. -/
theorem sat_subseqStrict (s : List σ) (w : List σ) (i : ℕ) :
    (subseqStrict s).sat w i = true ↔ s.reverse.Sublist (w.take (i - 1)) := by
  induction s generalizing i with
  | nil => simp [subseqStrict, Form.sat, Term.val]
  | cons a t ih =>
      rw [subseqStrict, Form.sat_exBefore, List.reverse_cons, sublist_snoc_take]
      constructor
      · rintro ⟨j, hj1, hj2, hj⟩
        rw [Form.sat, Bool.and_eq_true, ih] at hj
        exact ⟨j, hj1, hj2, by simpa [Form.sat] using hj.1, hj.2⟩
      · rintro ⟨j, hj1, hj2, hjw, hjt⟩
        exact ⟨j, hj1, hj2, by rw [Form.sat, Bool.and_eq_true, ih]; exact ⟨by simpa [Form.sat] using hjw, hjt⟩⟩

/-- `subseqAt s` holds at `i` exactly when `s.reverse` is a subsequence of
`w[1:i]`. -/
theorem sat_subseqAt (s : List σ) (w : List σ) (i : ℕ) :
    (subseqAt s).sat w i = true ↔ s.reverse.Sublist (w.take i) := by
  cases s with
  | nil => simp [subseqAt, Form.sat, Term.val]
  | cons a t =>
      rw [subseqAt, Form.sat_exAt, List.reverse_cons, sublist_snoc_take]
      constructor
      · rintro ⟨j, hj1, hj2, hj⟩
        rw [Form.sat, Bool.and_eq_true, sat_subseqStrict] at hj
        exact ⟨j, hj1, hj2, by simpa [Form.sat] using hj.1, hj.2⟩
      · rintro ⟨j, hj1, hj2, hjw, hjt⟩
        refine ⟨j, hj1, hj2, ?_⟩
        rw [Form.sat, Bool.and_eq_true, sat_subseqStrict]
        exact ⟨by simpa [Form.sat] using hjw, hjt⟩

/-- The language of the pattern formula: exactly the strings containing
`s.reverse` as a subsequence. -/
theorem lang_subseqAt (s : List σ) : (subseqAt s).lang = {w : List σ | s.reverse.Sublist w} := by
  ext w
  rw [Form.lang, Set.mem_ofPred_eq, Form.models, sat_subseqAt, List.take_length]
  exact Iff.rfl

/-- **The paper's formula needs the strict count.**  The proof of
`lem:piecewise_testable_depth` tests `Σ*σ₁Σ*⋯Σ*σ_kΣ*` with
`◁#[⋯◁#[◁#[Q_σ₁] ≥ 1 ∧ Q_σ₂] ≥ 1 ⋯ ∧ Q_σ_k] ≥ 1`, the non-strict `◁#` at every
level, so two equal neighbours `σⱼ = σⱼ₊₁` can be matched at one position.
For `Σ*aΣ*aΣ*` that formula accepts the one-letter string `a`, which does not
contain `aa`.  The separating languages alternate their letters and are not
affected; `subseqStrict` counts strictly below the top level instead.

Source: arXiv:2506.16055v3, Appendix A, proof of `lem:piecewise_testable_depth`. -/
theorem nonstrict_subseq_formula_unsound (a : σ) :
    (Form.exAt (.and (Form.exAt (.sym a)) (.sym a))).models [a] ∧ ¬ [a, a].Sublist [a] := by
  refine ⟨?_, fun h => by simpa using h.length_le⟩
  rw [Form.models, Form.sat_exAt]
  refine ⟨1, le_rfl, le_rfl, ?_⟩
  rw [Form.sat, Bool.and_eq_true, Form.sat_exAt]
  exact ⟨⟨1, le_rfl, le_rfl, by simp [Form.sat]⟩, by simp [Form.sat]⟩

end Semantics

end CRASP
end Transformer
