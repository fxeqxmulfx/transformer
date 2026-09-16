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

The separating family `L_k = altPlus false k` and the identity
`altPlus_eq` the paper's proof of `lem:piecewise_testable` asserts,
`L_k = (Σ* ∖ K_b^k) ∩ K_a^k`, are in `CRASP.Alternating`; what is left here is
the syntax of the Boolean combinations and the two definability lemmas.

`kPiecewiseTestable_altPlus` inherits the restriction `0 < k` of
`altPlus_eq`, and needs it for its own sake: at `k = 0` every 𝒥-expression of
width `0` is `Σ*`, so the `0`-piecewise testable languages are `∅` and `Σ*`,
neither of which is `L_0 = {ε}`.

**A typo.**  Equation `eq:altsingle` writes the even case of `K_a^k` as
`Σ*(aΣ*bΣ*)^k`, which fixes `2k` symbols and so is not `k`-piecewise
testable; the odd case, the companion `K_b^k`, and the use made of both
require `Σ*(aΣ*bΣ*)^{k/2}`.
-/

import Transformer.CRASP.Alternating
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

/-- **Lemma `lem:piecewise_testable`.**  `L_k` is `k`-piecewise testable, for
`k ≥ 1`.

Source: arXiv:2506.16055v3, §2.4, `lem:piecewise_testable`. -/
theorem kPiecewiseTestable_altPlus (k : ℕ) (hk : 0 < k) :
    KPiecewiseTestable k (altPlus false k) := by
  refine ⟨.and (.neg (.jexpr (altList true k))) (.jexpr (altList false k)), ?_, ?_⟩
  · simp only [PT.width, length_altList]
    omega
  · rw [altPlus_eq false k hk]
    ext w
    simp only [PT.lang, Set.mem_inter_iff, Set.mem_compl_iff, Set.mem_ofPred_eq,
      Set.mem_sdiff, altSingle, Bool.not_false]
    tauto

/-- The hypothesis of `kPiecewiseTestable_altPlus` is satisfiable: `k = 1`. -/
example : 0 < 1 := Nat.one_pos

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
