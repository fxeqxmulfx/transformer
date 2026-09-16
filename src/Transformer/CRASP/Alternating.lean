/-
# Alternating words and the separating family `L_k`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.4, Equations `eq:altplus` and `eq:altsingle`.

The separating family of the paper is

    L_k = (a⁺b⁺)^{k/2}        k even
    L_k = (a⁺b⁺)^{(k-1)/2} a⁺  k odd

— strings of `k` alternating nonempty blocks, starting with `a`; written
recursively that is `altPlus`.  The alphabet is `Bool`, with `false` for `a`
and `true` for `b`, because the paper's `Σ = {a, b}` is exactly a two-element
alphabet and `altPlus_eq` below is false over a larger one.

`altPlus_eq` is the identity `lem:piecewise_testable` rests on,
`L_k = A_k ∖ B_k`, where `A_k` and `B_k` are the strings containing the
alternating word of length `k` starting with `a`, resp. with `b`, as a
subsequence.  It reads off the block decomposition: a word with `m` maximal
blocks whose first block carries the symbol `s` has an alternating
subsequence of length `m` starting with `s` and of length `m - 1` starting
with `!s`, so belonging to `A_k` and not to `B_k` pins `m` to `k`.  The
arguments below are the same count run without ever naming the block
decomposition.

**`0 < k` is needed** and the paper does not say so: at `k = 0` the two
𝒥-expressions are both `Σ*`, so the right-hand side is empty while
`L_0 = {ε}`.
-/

import Mathlib.Data.List.Basic
import Mathlib.Data.Set.Insert

namespace Transformer
namespace CRASP

/-- The alternating string of length `k` beginning with `s`: `ababab⋯`. -/
def altList (s : Bool) : ℕ → List Bool
  | 0 => []
  | k + 1 => s :: altList (!s) k

/-- The defining step of `altList`, as a rewrite rule. -/
theorem altList_succ (s : Bool) (k : ℕ) : altList s (k + 1) = s :: altList (!s) k := rfl

@[simp] theorem length_altList (s : Bool) (k : ℕ) : (altList s k).length = k := by
  induction k generalizing s with
  | zero => rfl
  | succ k ih => rw [altList_succ, List.length_cons, ih]

/-- The alternating word starting with `!s` peels its head without a double
negation left behind. -/
theorem altList_not_succ (s : Bool) (k : ℕ) :
    altList (!s) (k + 1) = (!s) :: altList s k := by
  rw [altList_succ, Bool.not_not]

/-- `A_k` (for `s = a`) and `B_k` (for `s = b`), the two 𝒥-expressions of
Equation `eq:altsingle`: the strings containing `k` alternating symbols
starting with `s` as a subsequence. -/
def altSingle (s : Bool) (k : ℕ) : Set (List Bool) := {w | (altList s k).Sublist w}

/-- `L_k` beginning with the symbol `s`: `k` alternating nonempty blocks
(Equation `eq:altplus`).  The paper's `L_k` is `altPlus false k`. -/
def altPlus (s : Bool) : ℕ → Set (List Bool)
  | 0 => {[]}
  | k + 1 => {w | ∃ m, 0 < m ∧ ∃ v ∈ altPlus (!s) k, w = List.replicate m s ++ v}

/-- `L_1 = a⁺`. -/
theorem altPlus_one (s : Bool) : altPlus s 1 = {w | ∃ m, 0 < m ∧ w = List.replicate m s} := by
  ext w
  constructor
  · rintro ⟨m, hm, v, hv, rfl⟩
    exact ⟨m, hm, by rw [Set.mem_singleton_iff.1 hv, List.append_nil]⟩
  · rintro ⟨m, hm, rfl⟩
    exact ⟨m, hm, [], rfl, by rw [List.append_nil]⟩

/-! ### Peeling a leading block off a subsequence -/

/-- A subsequence whose first symbol differs from the first symbol of the word
does not touch that first symbol. -/
theorem sublist_of_ne_cons {c a : Bool} {l u : List Bool}
    (h : (c :: l).Sublist (a :: u)) (hne : c ≠ a) : (c :: l).Sublist u := by
  rcases List.cons_sublist_cons'.mp h with h' | ⟨hca, _⟩
  · exact h'
  · exact absurd hca hne

/-- The hypotheses of `sublist_of_ne_cons` are satisfiable: `[a] <+ [b, a]`. -/
example : ([true] : List Bool).Sublist [false, true] ∧ (true : Bool) ≠ false := by
  refine ⟨?_, by decide⟩
  exact (List.sublist_cons_self false [true])

/-- The same, past a whole block of a single repeated symbol. -/
theorem sublist_of_ne_replicate {c a : Bool} {l u : List Bool} (m : ℕ)
    (h : (c :: l).Sublist (List.replicate m a ++ u)) (hne : c ≠ a) :
    (c :: l).Sublist u := by
  induction m with
  | zero => simpa using h
  | succ m ih =>
      rw [List.replicate_succ, List.cons_append] at h
      exact ih (sublist_of_ne_cons h hne)

/-- The hypotheses of `sublist_of_ne_replicate` are satisfiable: `[b] <+ aab`. -/
example : ([true] : List Bool).Sublist (List.replicate 2 false ++ [true])
    ∧ (true : Bool) ≠ false := by
  refine ⟨?_, by decide⟩
  exact List.sublist_append_right (List.replicate 2 false) [true]

/-! ### The two alternating words of a given length -/

/-- The alternating words starting with a fixed symbol grow by a subsequence
step. -/
theorem altList_sublist_succ (s : Bool) (k : ℕ) :
    (altList s k).Sublist (altList s (k + 1)) := by
  induction k generalizing s with
  | zero => exact List.nil_sublist _
  | succ k ih =>
      rw [altList_succ, altList_succ]
      exact List.Sublist.cons_cons s (ih (!s))

/-- **Both alternating words of length `j` force one of length `j + 1`.**

A word containing the alternating subsequence of length `j ≥ 1` in both
phases has `j + 1` alternating symbols in the phase of its own first symbol.
This is the step that pins the number of blocks from below in
`mem_altPlus_of_sublist`. -/
theorem altList_succ_of_both (s : Bool) (j : ℕ) (hj : 0 < j) (u : List Bool)
    (h1 : (altList s j).Sublist u) (h2 : (altList (!s) j).Sublist u) :
    (altList s (j + 1)).Sublist u ∨ (altList (!s) (j + 1)).Sublist u := by
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  cases u with
  | nil => exact absurd h1.length_le (by simp)
  | cons c u' =>
      rcases (show c = s ∨ c = !s by cases c <;> cases s <;> simp) with hc | hc
      · rw [hc] at h1 h2 ⊢
        rw [altList_not_succ] at h2
        have h2' : (altList (!s) (i + 1)).Sublist u' := by
          rw [altList_not_succ]
          exact sublist_of_ne_cons h2 (by cases s <;> decide)
        refine Or.inl ?_
        rw [altList_succ]
        exact List.Sublist.cons_cons s h2'
      · rw [hc] at h1 h2 ⊢
        rw [altList_succ] at h1
        have h1' : (altList s (i + 1)).Sublist u' := by
          rw [altList_succ]
          exact sublist_of_ne_cons h1 (by cases s <;> decide)
        refine Or.inr ?_
        rw [altList_not_succ]
        exact List.Sublist.cons_cons (!s) h1'

/-- The hypotheses of `altList_succ_of_both` are satisfiable: `ab` and `ba`
are both subsequences of `aba`. -/
example : 0 < 2 ∧ (altList false 2).Sublist [false, true, false]
    ∧ (altList true 2).Sublist [false, true, false] := by
  exact ⟨by decide, by decide, by decide⟩

/-! ### `L_k` is contained in `A_k ∖ B_k` -/

/-- A word with `k` alternating blocks starting with `s` contains the
alternating word of length `k` starting with `s`: one symbol from each
block. -/
theorem altList_sublist_of_mem_altPlus :
    ∀ (k : ℕ) (s : Bool) (w : List Bool), w ∈ altPlus s k → (altList s k).Sublist w := by
  intro k
  induction k with
  | zero =>
      intro s w hw
      rw [Set.mem_singleton_iff.1 hw]
      exact List.nil_sublist _
  | succ k ih =>
      rintro s w ⟨m, hm, v, hv, rfl⟩
      obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
      rw [altList_succ, List.replicate_succ, List.cons_append]
      exact List.Sublist.cons_cons s
        ((ih (!s) v hv).trans (List.sublist_append_right _ _))

/-- The hypothesis of `altList_sublist_of_mem_altPlus` is satisfiable: `ab`
has two alternating blocks starting with `a`. -/
example : ([false, true] : List Bool) ∈ altPlus false 2 :=
  ⟨1, Nat.one_pos, [true], ⟨1, Nat.one_pos, [], rfl, rfl⟩, rfl⟩

/-- **A word with `j` alternating blocks has no alternating subsequence of
length `j + 1` in the phase of its first block.**  Every symbol of such a
subsequence would have to come from a block of its own, and there are only
`j` of them. -/
theorem not_altList_succ_sublist :
    ∀ (j : ℕ) (t : Bool) (w : List Bool), w ∈ altPlus t j →
      ¬ (altList t (j + 1)).Sublist w := by
  intro j
  induction j with
  | zero =>
      intro t w hw h
      rw [Set.mem_singleton_iff.1 hw] at h
      exact absurd h.length_le (by simp)
  | succ j ih =>
      rintro t w ⟨m, hm, v, hv, rfl⟩ h
      rw [altList_succ] at h
      have hdrop : (altList (!t) (j + 1)).Sublist (List.replicate m t ++ v) :=
        (List.sublist_cons_self t _).trans h
      rw [altList_not_succ] at hdrop
      refine ih (!t) v hv ?_
      rw [altList_not_succ]
      exact sublist_of_ne_replicate m hdrop (by cases t <;> decide)

/-- The hypothesis of `not_altList_succ_sublist` is satisfiable: `aa ∈ a⁺`. -/
example : ([false, false] : List Bool) ∈ altPlus false 1 :=
  ⟨2, by omega, [], rfl, rfl⟩

/-- **A word with `k` alternating blocks starting with `s` has no alternating
subsequence of length `k` starting with `!s`.**  Such a subsequence cannot use
the first block, so it would need `k` blocks among the remaining `k - 1`. -/
theorem not_altList_not_sublist_of_mem_altPlus (k : ℕ) (hk : 0 < k) (s : Bool)
    (w : List Bool) (hw : w ∈ altPlus s k) : ¬ (altList (!s) k).Sublist w := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  obtain ⟨m, hm, v, hv, rfl⟩ := hw
  intro h
  rw [altList_not_succ] at h
  refine not_altList_succ_sublist j (!s) v hv ?_
  rw [altList_not_succ]
  exact sublist_of_ne_replicate m h (by cases s <;> decide)

/-- The hypotheses of `not_altList_not_sublist_of_mem_altPlus` are
satisfiable: `ab` has two alternating blocks starting with `a`. -/
example : 0 < 2 ∧ ([false, true] : List Bool) ∈ altPlus false 2 :=
  ⟨by omega, 1, Nat.one_pos, [true], ⟨1, Nat.one_pos, [], rfl, rfl⟩, rfl⟩

/-! ### `A_k ∖ B_k` is contained in `L_k` -/

/-- Repeating the first symbol does not change the number of blocks. -/
theorem cons_mem_altPlus_same {s : Bool} {k : ℕ} {u : List Bool}
    (h : u ∈ altPlus s (k + 1)) : (s :: u) ∈ altPlus s (k + 1) := by
  obtain ⟨m, hm, v, hv, rfl⟩ := h
  exact ⟨m + 1, by omega, v, hv, by rw [List.replicate_succ, List.cons_append]⟩

/-- The hypothesis of `cons_mem_altPlus_same` is satisfiable: `a ∈ a⁺`. -/
example : ([false] : List Bool) ∈ altPlus false 1 := ⟨1, Nat.one_pos, [], rfl, rfl⟩

/-- Prefixing one symbol of the opposite phase adds one block. -/
theorem cons_mem_altPlus_flip {s : Bool} {k : ℕ} {u : List Bool}
    (h : u ∈ altPlus (!s) k) : (s :: u) ∈ altPlus s (k + 1) :=
  ⟨1, Nat.one_pos, u, h, by rw [List.replicate_one, List.cons_append, List.nil_append]⟩

/-- The hypothesis of `cons_mem_altPlus_flip` is satisfiable: `b ∈ b⁺`. -/
example : ([true] : List Bool) ∈ altPlus true 1 := ⟨1, Nat.one_pos, [], rfl, rfl⟩

/-- **A word of `A_k ∖ B_k` has exactly `k` alternating blocks, the first of
them carrying `s`.**

The induction is on the word.  A leading symbol of the opposite phase is
impossible, because the alternating subsequence of length `k` starting with
`s` would then live in the tail and could be extended by that leading symbol
into one of length `k` starting with `!s`.  With the right leading symbol,
either the whole subsequence already lives in the tail — one fewer repetition
of the first block — or the tail carries `k - 1` blocks, and
`altList_succ_of_both` is what rules out its carrying more. -/
theorem mem_altPlus_of_sublist :
    ∀ (w : List Bool) (k : ℕ) (s : Bool), 0 < k →
      (altList s k).Sublist w → ¬ (altList (!s) k).Sublist w → w ∈ altPlus s k := by
  intro w
  induction w with
  | nil =>
      intro k s hk h1 _
      exact absurd h1.length_le (by simp; omega)
  | cons c u ih =>
      intro k s hk h1 h2
      obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
      rcases (show c = s ∨ c = !s by cases c <;> cases s <;> simp) with hc | hc
      · rw [hc] at h1 h2 ⊢
        rw [altList_succ] at h1
        have hb : (altList (!s) j).Sublist u := List.cons_sublist_cons.mp h1
        have h2' : ¬ (altList (!s) (j + 1)).Sublist u := fun hcon =>
          h2 (by rw [altList_not_succ] at hcon ⊢; exact hcon.cons s)
        by_cases hs : (altList s (j + 1)).Sublist u
        · exact cons_mem_altPlus_same (ih (j + 1) s (by omega) hs h2')
        · refine cons_mem_altPlus_flip ?_
          rcases Nat.eq_zero_or_pos j with rfl | hj
          · have hu : u = [] := by
              cases u with
              | nil => rfl
              | cons e u' =>
                  exfalso
                  rcases (show e = s ∨ e = !s by cases e <;> cases s <;> simp)
                    with he | he
                  · exact hs (by rw [altList_succ, he]
                                 exact List.Sublist.cons_cons s (List.nil_sublist u'))
                  · exact h2' (by rw [altList_not_succ, he]
                                  exact List.Sublist.cons_cons (!s) (List.nil_sublist u'))
            rw [hu]
            exact rfl
          · refine ih j (!s) hj hb ?_
            intro hcon
            rw [Bool.not_not] at hcon
            rcases altList_succ_of_both s j hj u hcon hb with h | h
            · exact hs h
            · exact h2' h
      · exfalso
        rw [hc] at h1 h2
        rw [altList_succ] at h1
        have h1' : (altList s (j + 1)).Sublist u := by
          rw [altList_succ]
          exact sublist_of_ne_cons h1 (by cases s <;> decide)
        refine h2 ?_
        rw [altList_not_succ]
        exact List.Sublist.cons_cons (!s) ((altList_sublist_succ s j).trans h1')

/-- The hypotheses of `mem_altPlus_of_sublist` are satisfiable: `ab` contains
`ab` and not `ba`. -/
example : 0 < 2 ∧ (altList false 2).Sublist [false, true]
    ∧ ¬ (altList true 2).Sublist [false, true] :=
  ⟨by omega, by decide, by decide⟩

/-- **The characterization behind `lem:piecewise_testable`.**  A string of
`{a, b}*` has `k ≥ 1` alternating blocks starting with `a` exactly when it
contains the alternating subsequence of length `k` starting with `a` but not
the one starting with `b`.

`0 < k` cannot be dropped: at `k = 0` both 𝒥-expressions are `Σ*`, so the
right-hand side is empty, whereas `L_0 = {ε}`.

Source: arXiv:2506.16055v3, §2.4, `lem:piecewise_testable`. -/
theorem altPlus_eq (s : Bool) (k : ℕ) (hk : 0 < k) :
    altPlus s k = altSingle s k \ altSingle (!s) k := by
  ext w
  constructor
  · intro hw
    exact ⟨altList_sublist_of_mem_altPlus k s w hw,
      not_altList_not_sublist_of_mem_altPlus k hk s w hw⟩
  · rintro ⟨h1, h2⟩
    exact mem_altPlus_of_sublist w k s hk h1 h2

/-- The hypothesis of `altPlus_eq` is satisfiable: `k = 1`. -/
example : 0 < 1 := Nat.one_pos

end CRASP
end Transformer
