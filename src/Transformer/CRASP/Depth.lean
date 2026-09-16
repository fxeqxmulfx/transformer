/-
# The depth hierarchy of `TL[◁#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.4–4.5: `lem:cropping_oneway`, `lem:reduction` and
`thm:TLCl_depth`.

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

From §4.4 on, the paper fixes `Σ = {a, b}`, so the statements below that use
the plane are over `Bool`, with `false` for `a` and `true` for `b`, matching
`CRASP.PiecewiseTestable`.

`lem:TLCP_commutative` is proved in `Transformer.CRASP.Commutative`.
`lem:cropping_oneway` is false as stated, in both of its halves
(`cropping_oneway_unsound`, `cropping_oneway_right_unsound`): its proof reads a
minimal depth-1 subformula as a half-plane in the prefix vector, which forgets
the positions before the interval, where the PNPs are free
(`Transformer.CRASP.CroppingUnsound`).  `lem:reduction` is false already at
`k = 2`, in both of its versions (`reduction_past_unsound`,
`reduction_unsound`): no depth-1 formula whose PNPs are constant on the middle
checks the affix `ab`, because the middle contains the last position of the
prefix (`Transformer.CRASP.ReductionUnsound`).  The hierarchy itself is stated
with `sorry` in proof position: the paper derives it from those two lemmas.

**A typo.**  `lem:reduction` promises "a formula `φ'` of depth `(k-1)` of
`TL[◁#]^P_{k-1}` (or `TL[◁#,▷#]^P_k`, resp.)"; the parenthetical should read
`TL[◁#,▷#]^P_{k-1}`, as the sentence's own "of depth `(k-1)`" says and as the
proof of `thm:TLC_depth` uses it (it goes from depth `ℓ+1` to depth `ℓ`).
`reduction_unsound` refutes it in that reading.
-/

import Transformer.CRASP.Commutative
import Transformer.CRASP.CroppingUnsound
import Transformer.CRASP.ReductionUnsound
import Transformer.CRASP.PiecewiseTestable

namespace Transformer
namespace CRASP

/-! ## Sticking to one side only -/

/-- An interval sticks *only to the top* of another when it shares its upper
`b`-corner and no other side (§4.4). -/
def SticksOnlyToTop (I' I : Interval Bool) : Prop :=
  SticksTo true I' I ∧ ¬ SticksTo false I' I ∧
    ¬ SticksToLo true I' I ∧ ¬ SticksToLo false I' I

/-- An interval sticks *only to the right* of another when it shares its upper
`a`-corner and no other side (§4.4). -/
def SticksOnlyToRight (I' I : Interval Bool) : Prop :=
  SticksTo false I' I ∧ ¬ SticksTo true I' I ∧
    ¬ SticksToLo true I' I ∧ ¬ SticksToLo false I' I

/-! ## Cropping and reduction -/

/-- **`lem:cropping_oneway` (Cropping Lemma for `TL[◁#]`) is false.**  "For
any formula `φ` of `TL[◁#]^P` and any accommodating family of intervals `I`
such that the PNPs of `φ` are constant on `I`, there exists an accommodating
family of intervals `I'` such that `I'(n⃗)` sticks only to the top (and no other
side) of `I(n⃗)` for all `n⃗`, and all of the minimal depth-1 subformulas (and
PNPs) of `φ` are constant on `I'`."  `firstNotA` on `n⃗ ↦ [(1,1), n⃗]` meets the
hypotheses, and no accommodating family inside that one keeps its minimal
depth-1 subformula constant, whichever sides it sticks to.

Source: arXiv:2506.16055v3, §4.4, `lem:cropping_oneway`, and its proof in
Appendix C.2: "because any PNPs in each `ψ_ℓ` are constant on `I`, each `ψ_ℓ`
defines a half-plane". -/
theorem cropping_oneway_unsound :
    ¬ ∀ φ : Form Bool, φ.past = true → ∀ I : IntervalFamily Bool, Accommodating I →
      PnpsConstantOn φ I → ∃ I' : IntervalFamily Bool, Accommodating I' ∧
        (∀ n, SticksOnlyToTop (I' n) (I n)) ∧ MinimalOneConstantOn φ I' ∧ PnpsConstantOn φ I' :=
  fun h => by
    obtain ⟨I', hI', hstick, hmin, -⟩ :=
      h Form.firstNotA rfl _ accommodating_one pnpsConstantOn_firstNotA
    exact not_minimalOneConstantOn_firstNotA hI' (fun n => (hstick n).1.1) hmin

/-- **The second half of `lem:cropping_oneway` is false too.**  "Additionally,
there exists such an `I'` such that `I'(n⃗)` sticks only to the right of
`I(n⃗)`": the same `firstNotA` on `n⃗ ↦ [(1,1), n⃗]` refutes it.

Source: arXiv:2506.16055v3, §4.4, `lem:cropping_oneway`, and its proof in
Appendix C.2. -/
theorem cropping_oneway_right_unsound :
    ¬ ∀ φ : Form Bool, φ.past = true → ∀ I : IntervalFamily Bool, Accommodating I →
      PnpsConstantOn φ I → ∃ I' : IntervalFamily Bool, Accommodating I' ∧
        (∀ n, SticksOnlyToRight (I' n) (I n)) ∧ MinimalOneConstantOn φ I' ∧ PnpsConstantOn φ I' :=
  fun h => by
    obtain ⟨I', hI', hstick, hmin, -⟩ :=
      h Form.firstNotA rfl _ accommodating_one pnpsConstantOn_firstNotA
    exact not_minimalOneConstantOn_firstNotA hI' (fun n => (hstick n).1.1) hmin

/-- **`lem:reduction` (Reduction Lemma) is false**, already at `k = 2`.  "For
any depth-`k` formula `φ` of `TL[◁#]^P_k` and affix restriction `(λ, ϱ)`, if
the PNPs and minimal depth-1 subformulas of `φ` are constant on the middle of
`(λ, ϱ)`, then there is a formula `φ'` of depth `(k-1)` of `TL[◁#]^P_{k-1}`
that defines `L(φ)` restricted to `(λ, ϱ)`, and the PNPs of `φ'` are constant
on the middle of `(λ, ϱ)`."  Take `φ = ⊤`, written `¬(1 < 1)`, which has no
PNPs and no minimal depth-1 subformulas, and `(λ, ϱ) = (ab, ε)`: the restricted
language holds `abab` and not `aabb`, which no depth-1 formula with PNPs
constant on the middle tells apart (`Form.sat_abab_eq_aabb`).

Source: arXiv:2506.16055v3, §4.4, `lem:reduction`, and its proof in
Appendix C.3, where `Π_σ` is claimed constant on the middle. -/
theorem reduction_past_unsound :
    ¬ ∀ k : ℕ, 0 < k → ∀ φ : Form Bool, φ ∈ TLClP Bool k → ∀ A : Affix Bool,
      PnpsConstantOn φ A.middle → MinimalOneConstantOn φ A.middle →
      ∃ φ' ∈ TLClP Bool (k - 1), φ'.lang = A.restrict φ.lang ∧ PnpsConstantOn φ' A.middle :=
  fun h => by
    obtain ⟨φ', hφ', hlang, hpnp⟩ := h 2 two_pos (.neg (.lt .one .one)) ⟨rfl, Nat.zero_le 2⟩
      Affix.startAB (fun ψ hψ => by simp [Form.pnps, Term.pnps] at hψ)
      (fun ψ hψ => by simp [Form.minimalOne, Term.minimalOne, Term.depth] at hψ)
    exact not_lang_eq_restrict_startAB φ' hφ'.2 hpnp hlang

/-- **The `TL[◁#,▷#]^P` version of `lem:reduction` is false too**, by the same
`⊤` and `(ab, ε)`: `Form.sat_abab_eq_aabb` does not need the formula to be
past-only.

Source: arXiv:2506.16055v3, §4.4, `lem:reduction`, and its proof in
Appendix C.3. -/
theorem reduction_unsound :
    ¬ ∀ k : ℕ, 0 < k → ∀ φ : Form Bool, φ ∈ TLCP Bool k → ∀ A : Affix Bool,
      PnpsConstantOn φ A.middle → MinimalOneConstantOn φ A.middle →
      ∃ φ' ∈ TLCP Bool (k - 1), φ'.lang = A.restrict φ.lang ∧ PnpsConstantOn φ' A.middle :=
  fun h => by
    obtain ⟨φ', hφ', hlang, hpnp⟩ := h 2 two_pos (.neg (.lt .one .one)) (Nat.zero_le 2)
      Affix.startAB (fun ψ hψ => by simp [Form.pnps, Term.pnps] at hψ)
      (fun ψ hψ => by simp [Form.minimalOne, Term.minimalOne, Term.depth] at hψ)
    exact not_lang_eq_restrict_startAB φ' hφ' hpnp hlang

/-! ## The hierarchy -/

/-- **Theorem `thm:TLCl_depth`.**  For `k > 0` the language `L_{k+1}` is
definable in `TL[◁#]_{k+1}` but not in `TL[◁#]_k`.

The positive half asks for nothing beyond §2.4: `L_{k+1}` is `(k+1)`-piecewise
testable by `lem:piecewise_testable`, and `lem:piecewise_testable_depth` reads
every such language inside `TL[◁#]_{k+1}`.  The negative half is the one the
chapter is about, and only it is left open. -/
theorem definableL_altPlus (k : ℕ) (hk : 0 < k) :
    DefinableL (altPlus false (k + 1)) (k + 1) ∧ ¬ DefinableL (altPlus false (k + 1)) k :=
  ⟨definableL_of_kPiecewiseTestable (k + 1) _ (kPiecewiseTestable_altPlus (k + 1) k.succ_pos),
    sorry⟩

end CRASP
end Transformer
