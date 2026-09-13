/-
# Parikh vectors, intervals, and affix restrictions

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.3 (`def:Parikh_map`, `def:intervals`, `def:PNP`) and §4.1–4.2
(`def:affix_restriction`, `def:accommodating`, `def:constant`).

The depth-hierarchy proof works with the counts of symbols rather than with
strings: a string is a monotone path through `ℕ^|Σ|`, a prefix is a point on
that path, and a minimal depth-1 formula is a half-space.  This file fixes
the vocabulary.

An *interval* `[i⃗, j⃗]` is a box of Parikh vectors, and a *family of
intervals* attaches one to each `n⃗`.  An *affix restriction* `(λ, ϱ)` pins a
prefix and a suffix that depend only on `n⃗`, leaving free only the *middle*
of the string; its middle, as a family of intervals, is
`n⃗ ↦ [ℙ(λ(n⃗)), n⃗ - ℙ(ϱ(n⃗))]`.  The restriction is *accommodating* when that
middle is not too tight — "for any `s⃗` there is an interval `[i⃗, j⃗]` in the
image of `I` such that `s⃗ ≤ j⃗ - i⃗`" — which is what lets the depth-reduction
step be iterated.
-/

import Transformer.CRASP.Basic

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- A Parikh vector: a count for each symbol of the alphabet (§2.3). -/
abbrev PVec (σ : Type u) : Type u := σ → ℕ

/-- The length of a Parikh vector, `|v⃗| = v₁ + ⋯ + v_|Σ|` (§2.3). -/
def PVec.len [Fintype σ] (v : PVec σ) : ℕ := ∑ a, v a

/-- The componentwise order on Parikh vectors, `i⃗ ≤ j⃗` (Definition
`def:intervals`). -/
def PVec.le (i j : PVec σ) : Prop := ∀ a, i a ≤ j a

/-- An interval `[i⃗, j⃗]` in `ℕ^|Σ|`: the box between two Parikh vectors
(Definition `def:intervals`). -/
structure Interval (σ : Type u) : Type u where
  /-- The lower corner `i⃗`. -/
  lo : PVec σ
  /-- The upper corner `j⃗`. -/
  hi : PVec σ

/-- Membership in an interval. -/
def Interval.mem (I : Interval σ) (v : PVec σ) : Prop := PVec.le I.lo v ∧ PVec.le v I.hi

instance : Membership (PVec σ) (Interval σ) where
  mem I v := I.mem v

/-- One interval is contained in another. -/
def Interval.Subset (I J : Interval σ) : Prop := ∀ v, v ∈ I → v ∈ J

/-- A family of intervals: one interval for each Parikh vector (Definition
`def:intervals`). -/
abbrev IntervalFamily (σ : Type u) : Type u := PVec σ → Interval σ

/-- A family of intervals is *accommodating* when every `s⃗` fits inside one
of its intervals (Definition `def:accommodating`). -/
def Accommodating (I : IntervalFamily σ) : Prop :=
  ∀ s : PVec σ, ∃ n : PVec σ, ∀ a, s a + (I n).lo a ≤ (I n).hi a

/-- An interval *sticks to the top of* another when it sits inside it and
their upper corners agree at `b` (§4.3).  The paper takes `Σ = {a, b}` there
and reads a Parikh vector as a point of the plane; the definition is stated
here for whichever symbol is named. -/
def SticksTo (b : σ) (I' I : Interval σ) : Prop := I'.Subset I ∧ I'.hi b = I.hi b

/-- The companion of `SticksTo` at the lower corner: "sticking to the bottom"
for `b`, "sticking to the left" for `a` (§4.3, "analogously"). -/
def SticksToLo (b : σ) (I' I : Interval σ) : Prop := I'.Subset I ∧ I'.lo b = I.lo b

variable [DecidableEq σ]

/-- An affix restriction `(λ, ϱ)`: a prefix and a suffix, each a function of
the Parikh vector of the whole string (Definition `def:affix_restriction`). -/
structure Affix (σ : Type u) : Type u where
  /-- The prefix `λ`. -/
  pre : PVec σ → List σ
  /-- The suffix `ϱ`. -/
  suf : PVec σ → List σ

/-- The set of strings that match an affix restriction. -/
def Affix.matches (A : Affix σ) (w : List σ) : Prop :=
  ∃ w' : List σ, w = A.pre (parikh w) ++ w' ++ A.suf (parikh w)

/-- The restriction of a language to an affix restriction (Definition
`def:affix_restriction`). -/
def Affix.restrict (A : Affix σ) (L : Set (List σ)) : Set (List σ) :=
  {w | w ∈ L ∧ A.matches w}

/-- The *middle* of an affix restriction: the family of intervals
`n⃗ ↦ [ℙ(λ(n⃗)), n⃗ - ℙ(ϱ(n⃗))]` (§4.1). -/
def Affix.middle (A : Affix σ) : IntervalFamily σ :=
  fun n => ⟨parikh (A.pre n), fun a => n a - parikh (A.suf n) a⟩

/-- An affix restriction is *accommodating* when its middle is (Definition
`def:accommodating`). -/
def Affix.Accommodating (A : Affix σ) : Prop := CRASP.Accommodating A.middle

/-- A language is *commutative on the middle* of an affix restriction when
any two strings matching the restriction with the same Parikh vector are
either both in it or both out (§4.2). -/
def CommutativeOnMiddle (L : Set (List σ)) (A : Affix σ) : Prop :=
  ∀ w w' : List σ, A.matches w → A.matches w' → parikh w = parikh w' → (w ∈ L ↔ w' ∈ L)

/-- The Parikh vector of the prefix `w[1:i]`, which is the point of `ℕ^|Σ|`
that position `i` of `w` occupies. -/
def prefixVec (w : List σ) (i : ℕ) : PVec σ := parikh (w.take i)

/-- A formula is *constant* on a family of intervals when its truth value
depends on neither the string nor the position, as long as the string has the
given Parikh vector and the position sits inside the interval (Definition
`def:constant`). -/
def ConstantOn (φ : Form σ) (I : IntervalFamily σ) : Prop :=
  ∀ (n : PVec σ) (w w' : List σ) (i i' : ℕ), parikh w = n → parikh w' = n →
    prefixVec w i ∈ I n → prefixVec w' i' ∈ I n → φ.sat w i = φ.sat w' i'

mutual

/-- The Parikh numerical predicates occurring in a formula. -/
def Form.pnps : Form σ → List (Form σ)
  | .sym _ => []
  | .lt t₁ t₂ => t₁.pnps ++ t₂.pnps
  | .neg φ => φ.pnps
  | .and φ₁ φ₂ => φ₁.pnps ++ φ₂.pnps
  | .pnp π => [.pnp π]

/-- The Parikh numerical predicates occurring in a term. -/
def Term.pnps : Term σ → List (Form σ)
  | .countL φ => φ.pnps
  | .countR φ => φ.pnps
  | .add t₁ t₂ => t₁.pnps ++ t₂.pnps
  | .one => []

end

mutual

/-- The *minimal depth-1 subformulas* of a formula: the comparisons `t₁ < t₂`
whose terms already have depth `1`, so that they contain no depth-1 formula
of their own (§2.2). -/
def Form.minimalOne : Form σ → List (Form σ)
  | .sym _ => []
  | .lt t₁ t₂ =>
      if max t₁.depth t₂.depth = 1 then [.lt t₁ t₂] else t₁.minimalOne ++ t₂.minimalOne
  | .neg φ => φ.minimalOne
  | .and φ₁ φ₂ => φ₁.minimalOne ++ φ₂.minimalOne
  | .pnp _ => []

/-- The minimal depth-1 subformulas occurring in a term. -/
def Term.minimalOne : Term σ → List (Form σ)
  | .countL φ => φ.minimalOne
  | .countR φ => φ.minimalOne
  | .add t₁ t₂ => t₁.minimalOne ++ t₂.minimalOne
  | .one => []

end

/-- "The PNPs of `φ` are constant on `I`" (§4.2). -/
def PnpsConstantOn (φ : Form σ) (I : IntervalFamily σ) : Prop :=
  ∀ ψ ∈ φ.pnps, ConstantOn ψ I

/-- "The minimal depth-1 subformulas of `φ` are constant on `I`" (§4.3). -/
def MinimalOneConstantOn (φ : Form σ) (I : IntervalFamily σ) : Prop :=
  ∀ ψ ∈ φ.minimalOne, ConstantOn ψ I

/-- The trivial affix restriction, `λ(n⃗) = ϱ(n⃗) = ε`, is accommodating: its
middle is the whole box `[0⃗, n⃗]`, so any `s⃗` fits at `n⃗ = s⃗` (§4.1, "An
accommodating affix restriction is the trivial one"). -/
theorem accommodating_trivial : (⟨fun _ => [], fun _ => []⟩ : Affix σ).Accommodating := by
  intro s
  refine ⟨s, fun a => ?_⟩
  simp only [Affix.middle, parikh, List.filter_nil, List.length_nil]
  omega

/-- A formula with no counting is constant on every family of intervals only
when it also cannot read the current symbol; the constant predicates of the
paper are the ones a PNP can express.  Here is the simplest nontrivial
witness that `ConstantOn` is satisfiable: a PNP that ignores its arguments. -/
example (I : IntervalFamily σ) : ConstantOn (.pnp fun _ _ => true) I :=
  fun _ _ _ _ _ _ _ _ _ => rfl

end CRASP
end Transformer
