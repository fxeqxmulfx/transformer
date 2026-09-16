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
import Transformer.CRASP.SubsequenceTwoSided

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

/-- The formula of a Boolean combination of 𝒥-expressions, given a formula
`f s` testing each 𝒥-expression `s`: the Boolean connectives are kept (§2.4,
proof of `lem:piecewise_testable_depth`). -/
def PT.toForm (f : List σ → Form σ) : PT σ → Form σ
  | .jexpr s => f s
  | .neg e => .neg (e.toForm f)
  | .and e₁ e₂ => .and (e₁.toForm f) (e₂.toForm f)

/-- The Boolean connectives cost no depth, so a bound on the tests of the
𝒥-expressions of width at most `m` bounds the whole formula. -/
theorem PT.depth_toForm_le (f : List σ → Form σ) (m d : ℕ)
    (hf : ∀ s : List σ, s.length ≤ m → (f s).depth ≤ d) (e : PT σ) (he : e.width ≤ m) :
    (e.toForm f).depth ≤ d := by
  induction e with
  | jexpr s => exact hf s he
  | neg e ih => exact ih he
  | and e₁ e₂ ih₁ ih₂ =>
      rw [PT.width, max_le_iff] at he
      exact max_le (ih₁ he.1) (ih₂ he.2)

/-- The formula counts only over the past when the tests do. -/
theorem PT.past_toForm (f : List σ → Form σ) (hf : ∀ s : List σ, (f s).past = true)
    (e : PT σ) : (e.toForm f).past = true := by
  induction e with
  | jexpr s => exact hf s
  | neg e ih => simp [PT.toForm, Form.past, ih]
  | and e₁ e₂ ih₁ ih₂ => simp [PT.toForm, Form.past, ih₁, ih₂]

/-- The formula uses no Parikh numerical predicate when the tests use none. -/
theorem PT.pnpFree_toForm (f : List σ → Form σ) (hf : ∀ s : List σ, (f s).pnpFree = true)
    (e : PT σ) : (e.toForm f).pnpFree = true := by
  induction e with
  | jexpr s => exact hf s
  | neg e ih => simp [PT.toForm, Form.pnpFree, ih]
  | and e₁ e₂ ih₁ ih₂ => simp [PT.toForm, Form.pnpFree, ih₁, ih₂]

/-- The formula defines the language of the Boolean combination when each test
defines its 𝒥-expression. -/
theorem PT.lang_toForm [DecidableEq σ] (f : List σ → Form σ)
    (hf : ∀ s : List σ, (f s).lang = {w | s.Sublist w}) (e : PT σ) :
    (e.toForm f).lang = e.lang := by
  induction e with
  | jexpr s => exact hf s
  | neg e ih =>
      rw [PT.lang, ← ih]
      ext w
      simp [PT.toForm, Form.lang, Form.models, Form.sat]
  | and e₁ e₂ ih₁ ih₂ =>
      rw [PT.lang, ← ih₁, ← ih₂]
      ext w
      simp [PT.toForm, Form.lang, Form.models, Form.sat]

/-- The hypotheses of the four `PT.toForm` lemmas are satisfiable: the
one-sided subsequence test meets all of them, with `m = d`. -/
example [DecidableEq σ] (m : ℕ) :
    (∀ s : List σ, s.length ≤ m → (subseqAt s.reverse).depth ≤ m) ∧
      (∀ s : List σ, (subseqAt s.reverse).past = true) ∧
      (∀ s : List σ, (subseqAt s.reverse).pnpFree = true) ∧
      (∀ s : List σ, (subseqAt s.reverse).lang = {w | s.Sublist w}) ∧
      (PT.jexpr ([] : List σ)).width ≤ m :=
  ⟨fun s hs => by simpa using hs, fun _ => past_subseqAt _, fun _ => pnpFree_subseqAt _,
    fun s => by rw [lang_subseqAt, List.reverse_reverse], Nat.zero_le m⟩

/-- **Lemma `lem:piecewise_testable_depth`.**  Any `k`-piecewise testable
language is definable in `TL[◁#]_k`: `PT.toForm` with the one-sided test
`subseqAt` turns the Boolean combination into a past-only formula whose depth
is its width. -/
theorem definableL_of_kPiecewiseTestable [DecidableEq σ] (k : ℕ) (L : Set (List σ))
    (h : KPiecewiseTestable k L) : DefinableL L k := by
  obtain ⟨e, hwidth, rfl⟩ := h
  refine ⟨e.toForm fun s => subseqAt s.reverse, ⟨?_, ?_, ?_⟩, ?_⟩
  · exact PT.past_toForm _ (fun _ => past_subseqAt _) e
  · exact PT.pnpFree_toForm _ (fun _ => pnpFree_subseqAt _) e
  · exact PT.depth_toForm_le _ k k (fun s hs => by simpa using hs) e hwidth
  · exact PT.lang_toForm _ (fun s => by rw [lang_subseqAt, List.reverse_reverse]) e

/-- **Lemma `lem:piecewise_testable_depth`, bidirectional half.**  Any
`(2k+1)`-piecewise testable language is definable in `TL[◁#, ▷#]_{k+1}`:
`PT.toForm` with the two-sided test `subseqTwoSided k`, which guesses the
middle symbol with one `◁#` and checks the `k` symbols on either side of it at
depth `k`.  The halves `φ_L`, `φ_R` of the paper count non-strictly and so let
them reuse the middle position; `subseqTwoSided` counts strictly. -/
theorem definable_of_kPiecewiseTestable [DecidableEq σ] (k : ℕ) (L : Set (List σ))
    (h : KPiecewiseTestable (2 * k + 1) L) : Definable L (k + 1) := by
  obtain ⟨e, hwidth, rfl⟩ := h
  refine ⟨e.toForm (subseqTwoSided k), ⟨?_, ?_⟩, ?_⟩
  · exact PT.pnpFree_toForm _ (pnpFree_subseqTwoSided k) e
  · exact PT.depth_toForm_le _ (2 * k + 1) (k + 1)
      (fun s hs => (depth_subseqTwoSided_le k s).trans (max_le le_rfl (by omega))) e hwidth
  · exact PT.lang_toForm _ (lang_subseqTwoSided k) e

/-- The hypotheses of the two definability lemmas are satisfiable: the
`0`-piecewise testable languages are `∅` and `Σ*`, and `Σ*` is one of them. -/
example : KPiecewiseTestable (σ := Bool) 0 Set.univ := by
  refine ⟨.jexpr [], by simp [PT.width], ?_⟩
  ext w
  simp [PT.lang]

end CRASP
end Transformer
