/-
# Piecewise testable languages and the separating family

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.4 (`lem:piecewise_testable`, `lem:piecewise_testable_depth`).

A *𝒥-expression* is a language `Σ*σ₁Σ*σ₂Σ*⋯Σ*σ_kΣ*`, that is, the set of
strings containing `σ₁ ⋯ σ_k` as a subsequence; a language is *`k`-piecewise
testable* if it is a Boolean combination of 𝒥-expressions with at most `k`
fixed symbols.  `PT` is the syntax of such a Boolean combination and
`PT.width` counts its longest 𝒥-expression, so `KPiecewiseTestable` reads off
the definition directly.

The separating family of the whole paper is

    L_k = (a⁺b⁺)^{k/2}        k even
    L_k = (a⁺b⁺)^{(k-1)/2} a⁺  k odd

— strings of `k` alternating nonempty blocks, starting with `a`.  Written
recursively, that is `altPlus`.  The alphabet is `Bool` here, with `false`
for `a` and `true` for `b`, because the paper's `Σ = {a, b}` is exactly a
two-element alphabet and the characterization below is false over a larger
one.

`altPlus_eq` is the identity the paper's proof of `lem:piecewise_testable`
asserts, `L_k = (Σ* ∖ K_b^k) ∩ K_a^k`, with the two 𝒥-expressions being the
alternating subsequences of length `k` starting with `a` and with `b`.

**A typo.**  Equation `eq:altsingle` writes the even case of `K_a^k` as
`Σ*(aΣ*bΣ*)^k`, which fixes `2k` symbols and so is not `k`-piecewise
testable; the odd case, the companion `K_b^k`, and the use made of both
require `Σ*(aΣ*bΣ*)^{k/2}`.
-/

import Transformer.CRASP.Parikh

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- The syntax of a Boolean combination of 𝒥-expressions (§2.4). -/
inductive PT (σ : Type u) : Type u
  /-- The 𝒥-expression `Σ*σ₁Σ*⋯Σ*σ_kΣ*`, given by the list `σ₁ ⋯ σ_k`. -/
  | jexpr : List σ → PT σ
  /-- Complement. -/
  | neg : PT σ → PT σ
  /-- Intersection. -/
  | and : PT σ → PT σ → PT σ

/-- The language of a Boolean combination of 𝒥-expressions.  A string is in
the 𝒥-expression of `s` exactly when `s` is a subsequence of it. -/
def PT.lang : PT σ → Set (List σ)
  | .jexpr s => {w | s.Sublist w}
  | .neg e => e.langᶜ
  | .and e₁ e₂ => e₁.lang ∩ e₂.lang

/-- How many symbols the longest 𝒥-expression of the combination fixes. -/
def PT.width : PT σ → ℕ
  | .jexpr s => s.length
  | .neg e => e.width
  | .and e₁ e₂ => max e₁.width e₂.width

/-- A language is `k`-piecewise testable when it is a Boolean combination of
𝒥-expressions each fixing at most `k` symbols (§2.4). -/
def KPiecewiseTestable (k : ℕ) (L : Set (List σ)) : Prop :=
  ∃ e : PT σ, e.width ≤ k ∧ e.lang = L

/-- A language is piecewise testable when it is `k`-piecewise testable for
some `k` (§2.4). -/
def PiecewiseTestable (L : Set (List σ)) : Prop := ∃ k, KPiecewiseTestable k L

section Alternating

/-- The alternating string of length `k` beginning with `s`: `ababab⋯`. -/
def altList (s : Bool) : ℕ → List Bool
  | 0 => []
  | k + 1 => s :: altList (!s) k

@[simp] theorem length_altList (s : Bool) (k : ℕ) : (altList s k).length = k := by
  induction k generalizing s with
  | zero => rfl
  | succ k ih => rw [altList, List.length_cons, ih]

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

/-- **The characterization behind `lem:piecewise_testable`.**  A string of
`{a, b}*` has `k` alternating blocks starting with `a` exactly when it
contains the alternating subsequence of length `k` starting with `a` but not
the one starting with `b`. -/
theorem altPlus_eq (s : Bool) (k : ℕ) :
    altPlus s k = altSingle s k \ altSingle (!s) k :=
  sorry

/-- **Lemma `lem:piecewise_testable`.**  `L_k` is `k`-piecewise testable. -/
theorem kPiecewiseTestable_altPlus (k : ℕ) : KPiecewiseTestable k (altPlus false k) := by
  refine ⟨.and (.neg (.jexpr (altList true k))) (.jexpr (altList false k)), ?_, ?_⟩
  · simp only [PT.width, length_altList]
    omega
  · rw [altPlus_eq]
    ext w
    simp only [PT.lang, Set.mem_inter_iff, Set.mem_compl_iff, Set.mem_ofPred_eq,
      Set.mem_sdiff, altSingle, Bool.not_false]
    tauto

end Alternating

/-- **Lemma `lem:piecewise_testable_depth`.**  Any `k`-piecewise testable
language is definable in `TL[◁#]_k`. -/
theorem definableL_of_kPiecewiseTestable [DecidableEq σ] (k : ℕ) (L : Set (List σ))
    (h : KPiecewiseTestable k L) : DefinableL L k :=
  sorry

/-- **Lemma `lem:piecewise_testable_depth`, bidirectional half.**  Any
`(2k+1)`-piecewise testable language is definable in `TL[◁#, ▷#]_{k+1}`. -/
theorem definable_of_kPiecewiseTestable [DecidableEq σ] (k : ℕ) (L : Set (List σ))
    (h : KPiecewiseTestable (2 * k + 1) L) : Definable L (k + 1) :=
  sorry

/-- The hypotheses of the two definability lemmas are satisfiable: the
`0`-piecewise testable languages are `∅` and `Σ*`, and `Σ*` is one of them. -/
example : KPiecewiseTestable (σ := Bool) 0 Set.univ := by
  refine ⟨.jexpr [], by simp [PT.width], ?_⟩
  ext w
  simp [PT.lang]

end CRASP
end Transformer
