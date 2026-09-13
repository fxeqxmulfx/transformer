/-
# The depth hierarchy of `TL[◁#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.2–4.4 and §5.2: `lem:TLCP_commutative`, `lem:cropping_oneway`,
`lem:reduction`, `thm:TLCl_depth`, `def:prediction_task`,
`cor:prediction_task_depth`.

The argument runs downwards.  A depth-`k` formula defining `L_{k+1}` is peeled
one counting level at a time: the Cropping Lemma finds a sub-family of
intervals on which the minimal depth-1 subformulas are constant, the Reduction
Lemma then rewrites them away at the cost of restricting the language to an
affix restriction whose middle is that sub-family, and after `k-1` rounds what
is left is a depth-1 formula.  But a depth-1 formula defines a language
commutative on the middle (`lem:TLCP_commutative`), while `L_{k+1}` restricted
to those affixes is not — the two strings `λ b^{s_b} a^{s_a-1} b a ϱ` and
`λ b^{s_b} a^{s_a} ϱ b a` have the same Parikh vector and only one of them
alternates correctly.

From §4.3 on, the paper fixes `Σ = {a, b}`, so the statements below that use
the plane are over `Bool`, with `false` for `a` and `true` for `b`, matching
`CRASP.PiecewiseTestable`.

The cropping and reduction lemmas and the hierarchy itself are stated with
`sorry` in proof position: their proofs are the geometric content of
Appendices A.5–A.7 and are not carried over here.

**A typo.**  `lem:reduction` promises "a formula `φ'` of depth `(k-1)` of
`TL[◁#]^P_{k-1}` (or `TL[◁#,▷#]^P_k`, resp.)"; the parenthetical should read
`TL[◁#,▷#]^P_{k-1}`, as the sentence's own "of depth `(k-1)`" says and as the
proof of `thm:TLC_depth` uses it (it goes from depth `ℓ+1` to depth `ℓ`).
-/

import Transformer.CRASP.PiecewiseTestable

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-! ## Sticking to one side only -/

/-- An interval sticks *only to the top* of another when it shares its upper
`b`-corner and no other side (§4.3). -/
def SticksOnlyToTop (I' I : Interval Bool) : Prop :=
  SticksTo true I' I ∧ ¬ SticksTo false I' I ∧
    ¬ SticksToLo true I' I ∧ ¬ SticksToLo false I' I

/-- An interval sticks *only to the right* of another when it shares its upper
`a`-corner and no other side (§4.3). -/
def SticksOnlyToRight (I' I : Interval Bool) : Prop :=
  SticksTo false I' I ∧ ¬ SticksTo true I' I ∧
    ¬ SticksToLo true I' I ∧ ¬ SticksToLo false I' I

/-! ## Commutativity of depth 1 -/

section Commutative

variable [DecidableEq σ]

/-- **Lemma `lem:TLCP_commutative` (Commutativity of depth 1).**  For a
depth-1 formula of `TL[◁#,▷#]^P` — and so in particular of `TL[◁#]^P` — and an
affix restriction whose suffix is never empty, if the PNPs of the formula are
constant on the middle then the language it defines is commutative on the
middle.

The nonempty-suffix hypothesis is what stops `Q_σ` from reading the last
position, which is the position the whole formula is judged at. -/
theorem commutativeOnMiddle_of_mem_TLCP_one (φ : Form σ) (hφ : φ ∈ TLCP σ 1)
    (A : Affix σ) (hsuf : ∀ n : PVec σ, 1 ≤ (A.suf n).length)
    (hpnp : PnpsConstantOn φ A.middle) :
    CommutativeOnMiddle φ.lang A :=
  sorry

/-- The hypotheses of `commutativeOnMiddle_of_mem_TLCP_one` are satisfiable:
the formula `◁#[Q_a] < 1` has depth 1 and no PNPs, and the affix restriction
that pins a single `a` at the end has a nonempty suffix everywhere. -/
example (a : σ) :
    (Form.isZero (.countL (.sym a)) ∈ TLCP σ 1) ∧
      (∀ n : PVec σ, 1 ≤ ((⟨fun _ => [], fun _ => [a]⟩ : Affix σ).suf n).length) ∧
      PnpsConstantOn (Form.isZero (.countL (.sym a)))
        (⟨fun _ => [], fun _ => [a]⟩ : Affix σ).middle := by
  refine ⟨Nat.le_refl 1, fun _ => Nat.le_refl 1, ?_⟩
  intro ψ hψ
  simp [Form.isZero, Form.pnps, Term.pnps] at hψ

end Commutative

/-! ## Cropping and reduction -/

/-- **Lemma `lem:cropping_oneway` (Cropping Lemma for `TL[◁#]`).**  An
accommodating family of intervals on which the PNPs of a past-only formula are
constant can be cropped to an accommodating sub-family that sticks only to the
top of it and on which the minimal depth-1 subformulas are constant too. -/
theorem cropping_oneway (φ : Form Bool) (hφ : φ.past = true) (I : IntervalFamily Bool)
    (hI : Accommodating I) (hpnp : PnpsConstantOn φ I) :
    ∃ I' : IntervalFamily Bool, Accommodating I' ∧ (∀ n, SticksOnlyToTop (I' n) (I n)) ∧
      MinimalOneConstantOn φ I' ∧ PnpsConstantOn φ I' :=
  sorry

/-- **Lemma `lem:cropping_oneway`, second half.**  "Additionally, there exists
such an `I'` such that `I'(n⃗)` sticks only to the right of `I(n⃗)`." -/
theorem cropping_oneway_right (φ : Form Bool) (hφ : φ.past = true) (I : IntervalFamily Bool)
    (hI : Accommodating I) (hpnp : PnpsConstantOn φ I) :
    ∃ I' : IntervalFamily Bool, Accommodating I' ∧ (∀ n, SticksOnlyToRight (I' n) (I n)) ∧
      MinimalOneConstantOn φ I' ∧ PnpsConstantOn φ I' :=
  sorry

/-- The hypotheses of the cropping lemma are satisfiable: a PNP-free past-only
formula and the family `n⃗ ↦ [0⃗, n⃗]`, which is accommodating. -/
example (a : Bool) :
    (Form.isZero (.countL (.sym a))).past = true ∧
      Accommodating (fun n => ⟨fun _ => 0, n⟩ : IntervalFamily Bool) ∧
      PnpsConstantOn (Form.isZero (.countL (.sym a)))
        (fun n => ⟨fun _ => 0, n⟩ : IntervalFamily Bool) := by
  refine ⟨rfl, fun s => ⟨s, fun _ => by simp⟩, ?_⟩
  intro ψ hψ
  simp [Form.isZero, Form.pnps, Term.pnps] at hψ

section Reduction

variable [DecidableEq σ]

/-- **Lemma `lem:reduction` (Reduction Lemma), `TL[◁#]^P` version.**  A
depth-`k` formula whose PNPs and minimal depth-1 subformulas are constant on
the middle of an affix restriction can be traded for a depth-`(k-1)` formula
defining the affix-restricted language. -/
theorem reduction_past (k : ℕ) (hk : 0 < k) (φ : Form σ) (hφ : φ ∈ TLClP σ k)
    (A : Affix σ) (hpnp : PnpsConstantOn φ A.middle)
    (hmin : MinimalOneConstantOn φ A.middle) :
    ∃ φ' ∈ TLClP σ (k - 1), φ'.lang = A.restrict φ.lang ∧ PnpsConstantOn φ' A.middle :=
  sorry

/-- **Lemma `lem:reduction`, `TL[◁#,▷#]^P` version.**  The same statement
without the past-only restriction. -/
theorem reduction (k : ℕ) (hk : 0 < k) (φ : Form σ) (hφ : φ ∈ TLCP σ k)
    (A : Affix σ) (hpnp : PnpsConstantOn φ A.middle)
    (hmin : MinimalOneConstantOn φ A.middle) :
    ∃ φ' ∈ TLCP σ (k - 1), φ'.lang = A.restrict φ.lang ∧ PnpsConstantOn φ' A.middle :=
  sorry

/-- The hypotheses of the reduction lemmas are satisfiable at `k = 1`: `Q_a`
is past-only of depth `0`, and it has neither a PNP nor a minimal depth-1
subformula, so both constancy hypotheses hold under any affix restriction.
(The constancy hypotheses are the binding ones: `◁#[Q_a] < 1` *is* a minimal
depth-1 subformula of itself, and it is not constant on the middle of the
trivial restriction — which is exactly why the lemma restricts the language.) -/
example (a : σ) (A : Affix σ) :
    (0 < 1) ∧ (Form.sym a : Form σ) ∈ TLClP σ 1 ∧
      PnpsConstantOn (Form.sym a : Form σ) A.middle ∧
      MinimalOneConstantOn (Form.sym a : Form σ) A.middle := by
  refine ⟨Nat.one_pos, ⟨rfl, Nat.zero_le 1⟩, ?_, ?_⟩
  · intro ψ hψ
    simp [Form.pnps] at hψ
  · intro ψ hψ
    simp [Form.minimalOne] at hψ

end Reduction

/-! ## The hierarchy -/

/-- **Theorem `thm:TLCl_depth`.**  For `k > 0` the language `L_{k+1}` is
definable in `TL[◁#]_{k+1}` but not in `TL[◁#]_k`. -/
theorem definableL_altPlus (k : ℕ) (hk : 0 < k) :
    DefinableL (altPlus false (k + 1)) (k + 1) ∧ ¬ DefinableL (altPlus false (k + 1)) k :=
  sorry

/-- **Definition `def:prediction_task`.**  A `TL[◁#]` formula solves the
next-token prediction problem for `L` when, on every prefix of every string of
`L`, it says exactly whether that prefix is itself in `L`.  Only prefixes of
strings of `L` are considered, which is what separates prediction from
recognition. -/
def SolvesPrediction [DecidableEq σ] (φ : Form σ) (L : Set (List σ)) : Prop :=
  ∀ w ∈ L, ∀ i, 1 ≤ i → i ≤ w.length → (φ.models (w.take i) ↔ w.take i ∈ L)

/-- **Corollary `cor:prediction_task_depth`.**  A depth-`(k+1)` `TL[◁#]`
formula solves the next-token prediction problem for `L_{k+3}`, and no
depth-`k` formula does. -/
theorem prediction_task_depth (k : ℕ) (hk : 0 < k) :
    (∃ φ ∈ TLCl Bool (k + 1), SolvesPrediction φ (altPlus false (k + 3))) ∧
      ∀ φ ∈ TLCl Bool k, ¬ SolvesPrediction φ (altPlus false (k + 3)) :=
  sorry

/-- The prediction problem is not vacuous: the formula `⊤` — written `¬(1 < 1)`
— solves the prediction problem for `Σ*`, in which every prefix lies. -/
example : SolvesPrediction (σ := Bool) (.neg (.lt .one .one)) Set.univ :=
  fun _ _ _ _ _ => by simp [Form.models, Form.sat, Term.val]

end CRASP
end Transformer
