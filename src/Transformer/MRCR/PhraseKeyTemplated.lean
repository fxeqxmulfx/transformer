module

public import Transformer.MRCR.PhraseKey

/-!
# Templated conversations whose key straddles the blocks

Source: MRCR of arXiv:2409.12640v2, `evals.tex`, line 127: "a long conversation between a user
and a model, in which the user requests writing (e.g. poems, riddles, essays) on different topics
proceeded by the model responses", and MRCR-V2 of arXiv:2507.06261v6, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, whose requests are phrases. The tokens and the
answer are those of `Transformer.MRCR.PhraseKey`.

A templated conversation is a list of turns, each the ask "`f` about `t`" answered by one or more
tokens, closed by a query (`Templated`). Cut it into blocks of `M` tokens that start at a topic:
the block of a bit (`turnBlock`) is the topic `t₁`, `M − 3` answers, and the start "`f_c` about"
of the next ask, `f_c` being `f₀` for `b` and `f₁` for `a`. After "`f₁` about" the blocks of any
bits, closed by `closing`, are a templated conversation (`templated_turnBlocks`). The close
answers the topic `t₀` by `v`, then asks "`f₀` about `t₀`", answers it by `v'` and queries the
first such ask. That ask is the one before the close when the last block is of `b`, answered by
`v`, and the one in the close otherwise, answered by `v'` (`phraseAnswer_closing_iff`): every
query is answered, and the answer is the last bit. `Transformer.MRCR.PhraseKeyTemplatedLimits`
uses this to show that positions read modulo a number do not answer these conversations.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP

variable {F N V R : ℕ}

/-- The words of a turn: the ask "`f` about `t`", then its answer `a`. -/
def turnWords (f : Fin F) (t : Fin N) (a : List (Fin V)) : List (PhraseTok F N V R) :=
  [.fmt f, .about, .topic t] ++ a.map .val

/-- **A templated conversation** (MRCR, arXiv:2409.12640v2, `evals.tex`, line 127): turns, each
the ask "`f` about `t`" answered by one or more tokens, then a query. -/
def Templated (w : List (PhraseTok F N V R)) : Prop :=
  ∃ (turns : List (Fin F × Fin N × List (Fin V))) (f : Fin F) (t : Fin N) (r : Fin R),
    (∀ x ∈ turns, x.2.2 ≠ []) ∧
      w = turns.flatMap (fun x => turnWords x.1 x.2.1 x.2.2) ++ [.query f t r]

/-- The block of the bit `c`: the topic `t₁`, `M − 3` answers `v`, and "`f_c` about", with
`f_c = f₀` for `b` and `f₁` for `a`. -/
def turnBlock (f₀ f₁ : Fin F) (t₁ : Fin N) (v : Fin V) (M : ℕ) (c : Bool) :
    List (PhraseTok F N V R) :=
  .topic t₁ :: List.replicate (M - 3) (.val v) ++ [.fmt (if c then f₀ else f₁), .about]

/-- The close: the topic `t₀` answered by `v`, the ask "`f₀` about `t₀`" answered by `v'`, and
the query for the first ask "`f₀` about `t₀`". -/
def closing [NeZero R] (f₀ : Fin F) (t₀ : Fin N) (v v' : Fin V) : List (PhraseTok F N V R) :=
  [.topic t₀, .val v, .fmt f₀, .about, .topic t₀, .val v', .query f₀ t₀ 0]

section Instance

variable {f₀ f₁ : Fin F} {t₀ t₁ : Fin N} {v v' : Fin V} {M : ℕ}

theorem length_turnBlock (hM : 3 ≤ M) (c : Bool) :
    (turnBlock (R := R) f₀ f₁ t₁ v M c).length = M := by
  simp only [turnBlock, List.cons_append, List.length_cons, List.length_append,
    List.length_replicate, List.length_nil]
  omega

/-- After "`f` about", blocks and a topic answered by `v` are turns. -/
theorem exists_turns_flatMap_turnBlock (hM : 4 ≤ M) (t : Fin N) (x : List Bool) :
    ∀ f : Fin F, ∃ turns : List (Fin F × Fin N × List (Fin V)), (∀ y ∈ turns, y.2.2 ≠ []) ∧
      [.fmt f, .about] ++ x.flatMap (turnBlock f₀ f₁ t₁ v M) ++ [.topic t, .val v] =
        turns.flatMap (fun y => turnWords (R := R) y.1 y.2.1 y.2.2) := by
  induction x with
  | nil => exact fun f => ⟨[(f, t, [v])], by simp, by simp [turnWords]⟩
  | cons c x ih =>
    intro f
    obtain ⟨turns, hne, he⟩ := ih (if c then f₀ else f₁)
    refine ⟨(f, t₁, List.replicate (M - 3) v) :: turns, ?_, ?_⟩
    · simp only [List.mem_cons, forall_eq_or_imp, ne_eq, List.replicate_eq_nil_iff]
      exact ⟨by omega, hne⟩
    · rw [List.flatMap_cons, List.flatMap_cons, ← he]
      simp [turnBlock, turnWords]

/-- **After "`f₁` about", the blocks of any bits and the close are a templated conversation.** -/
theorem templated_turnBlocks [NeZero R] (hM : 4 ≤ M) (x : List Bool) :
    Templated ([.fmt f₁, .about] ++ x.flatMap (turnBlock f₀ f₁ t₁ v M) ++
      closing (R := R) f₀ t₀ v v') := by
  obtain ⟨turns, hne, he⟩ := exists_turns_flatMap_turnBlock (R := R) hM t₀ x f₁ (f₀ := f₀)
    (f₁ := f₁) (t₁ := t₁) (v := v)
  refine ⟨turns ++ [(f₀, t₀, [v'])], f₀, t₀, 0, ?_, ?_⟩
  · simp only [List.mem_append, List.mem_singleton]
    rintro y (hy | rfl)
    · exact hne y hy
    · simp
  · rw [List.flatMap_append, ← he]
    simp [closing, turnWords]

/-- The topic `t₀` stands only in the close, if `A` lacks it. -/
theorem eq_of_getElem?_closing [NeZero R] {A : List (PhraseTok F N V R)} (hA : .topic t₀ ∉ A)
    (c : Bool) {k : ℕ}
    (h : (A ++ [.fmt (if c then f₀ else f₁), .about] ++ closing f₀ t₀ v v')[k]? =
      some (.topic t₀)) :
    k = A.length + 2 ∨ k = A.length + 6 := by
  rw [List.append_assoc] at h
  rcases lt_or_ge k A.length with hk | hk
  · rw [List.getElem?_append_left hk] at h
    exact absurd (List.mem_of_getElem? h) hA
  · rw [List.getElem?_append_right hk] at h
    rcases hm : k - A.length with _ | _ | _ | _ | _ | _ | _ | _ | _ | m <;>
      simp [hm, closing] at h <;> omega

/-- An ask "`f₀` about `t₀`" stands before the close for `b`, and in the close. -/
theorem askAt_closing [NeZero R] {A : List (PhraseTok F N V R)} (hA : .topic t₀ ∉ A)
    (hf : f₀ ≠ f₁) (c : Bool) {i : ℕ}
    (h : AskAt (A ++ [.fmt (if c then f₀ else f₁), .about] ++ closing f₀ t₀ v v') f₀ t₀ i) :
    (i = A.length ∧ c = true) ∨ i = A.length + 4 := by
  rcases eq_of_getElem?_closing hA c h.2.2 with hi | hi
  · obtain rfl : i = A.length := by omega
    have h₀ := h.1
    rw [List.append_assoc, List.getElem?_append_right le_rfl, Nat.sub_self] at h₀
    cases c
    · simp only [Bool.false_eq_true, ↓reduceIte, List.cons_append, List.getElem?_cons_zero,
        Option.some.injEq, PhraseTok.fmt.injEq] at h₀
      exact absurd h₀.symm hf
    · exact Or.inl ⟨rfl, rfl⟩
  · exact Or.inr (by omega)

/-- **The query is answered by `v` if the last block is of `b`, by `v'` otherwise.** -/
theorem phraseAnswer_closing_iff [NeZero R] {A : List (PhraseTok F N V R)} (hA : .topic t₀ ∉ A)
    (hf : f₀ ≠ f₁) (c : Bool) (u : Fin V) :
    PhraseAnswer (A ++ [.fmt (if c then f₀ else f₁), .about] ++ closing f₀ t₀ v v') u ↔
      u = if c then v else v' := by
  have hget (m : ℕ) : (A ++ [.fmt (if c then f₀ else f₁), .about] ++ closing f₀ t₀ v v')[
      A.length + m]? = ([.fmt (if c then f₀ else f₁), .about] ++ closing f₀ t₀ v v')[m]? := by
    rw [List.append_assoc, List.getElem?_append_right (Nat.le_add_right _ _),
      Nat.add_sub_cancel_left]
  have hget₀ := hget 0
  rw [Nat.add_zero] at hget₀
  have hl : List.getLast? (A ++ [PhraseTok.fmt (if c then f₀ else f₁), PhraseTok.about] ++
      closing f₀ t₀ v v') = some (.query f₀ t₀ 0) := by
    simp [closing]
  have hcount (i : ℕ) (hi : ∀ j < i, (j = A.length ∧ c = true) ∨ j = A.length + 4 → False) :
      Nat.count (AskAt (A ++ [.fmt (if c then f₀ else f₁), .about] ++ closing f₀ t₀ v v') f₀ t₀)
        i = ((0 : Fin R) : ℕ) := by
    rw [Fin.val_zero]
    exact Nat.count_of_forall_not fun j hj hask => hi j hj (askAt_closing hA hf c hask)
  constructor
  · rintro ⟨f, t, r, hq, i, hi, hc, hv⟩
    rw [hl, Option.some.injEq, PhraseTok.query.injEq] at hq
    obtain ⟨rfl, rfl, rfl⟩ := hq
    rcases askAt_closing hA hf c hi with ⟨rfl, rfl⟩ | rfl
    · rw [hget 3] at hv
      simpa [closing] using hv.symm
    · cases c
      · simp only [Nat.add_assoc, hget] at hv
        simpa [closing] using hv.symm
      · have h₀ : AskAt (A ++ [.fmt (if true then f₀ else f₁), .about] ++ closing f₀ t₀ v v') f₀ t₀
            A.length := by
          exact ⟨by rw [hget₀]; simp [closing], by rw [hget]; simp [closing],
            by rw [hget]; simp [closing]⟩
        rw [Fin.val_zero] at hc
        exact absurd h₀ (Nat.count_iff_forall_not.1 hc A.length (by omega))
  · rintro rfl
    refine ⟨f₀, t₀, 0, hl, ?_⟩
    cases c
    · refine ⟨A.length + 4, ⟨?_, ?_, ?_⟩, hcount _ fun j hj h => by grind, ?_⟩ <;>
        simp only [Nat.add_assoc, hget] <;> simp [closing]
    · refine ⟨A.length, ⟨?_, ?_, ?_⟩, hcount _ fun j hj h => by omega, ?_⟩
      · rw [hget₀]
        simp [closing]
      all_goals rw [hget]; simp [closing]

end Instance

/-- The hypotheses of the lemmas above are satisfiable: blocks of four tokens, and the empty
word before the close, with two formats. -/
example : (turnBlock (R := 1) (0 : Fin 2) 1 (0 : Fin 1) (0 : Fin 2) 4 true).length = 4 ∧
    Templated ([.fmt (1 : Fin 2), .about] ++ [true].flatMap (turnBlock 0 1 (0 : Fin 2) 0 4) ++
      closing (R := 1) 0 1 (0 : Fin 2) 1) ∧
    PhraseAnswer ([] ++ [.fmt (if true then (0 : Fin 2) else 1), .about] ++
      closing (R := 1) 0 (1 : Fin 2) (0 : Fin 2) 1) 0 :=
  ⟨length_turnBlock (by norm_num) true, templated_turnBlocks le_rfl [true],
    (phraseAnswer_closing_iff List.not_mem_nil (by decide) true 0).2 rfl⟩

/-- The hypotheses of `exists_turns_flatMap_turnBlock`, `eq_of_getElem?_closing` and
`askAt_closing` are satisfiable: the first turn of one block, and the ask in the close. -/
example : (∃ turns : List (Fin 2 × Fin 2 × List (Fin 2)), (∀ y ∈ turns, y.2.2 ≠ []) ∧
      [.fmt 1, .about] ++ [true].flatMap (turnBlock 0 1 1 0 4) ++ [.topic 0, .val 0] =
        turns.flatMap (fun y => turnWords (R := 1) y.1 y.2.1 y.2.2)) ∧
    (2 = ([] : List (PhraseTok 2 2 2 1)).length + 2 ∨
      2 = ([] : List (PhraseTok 2 2 2 1)).length + 6) ∧
    ((4 = ([] : List (PhraseTok 2 2 2 1)).length ∧ false = true) ∨
      4 = ([] : List (PhraseTok 2 2 2 1)).length + 4) :=
  ⟨exists_turns_flatMap_turnBlock le_rfl 0 [true] 1,
    eq_of_getElem?_closing (f₀ := 0) (f₁ := 1) (t₀ := 0) (v := 0) (v' := 1) List.not_mem_nil
      true (k := 2) (by simp [closing]),
    askAt_closing (f₀ := 0) (f₁ := 1) (t₀ := 0) (v := 0) (v' := 1) List.not_mem_nil (by decide)
      false (i := 4) ⟨by simp [closing], by simp [closing], by simp [closing]⟩⟩

end Transformer.MRCR
