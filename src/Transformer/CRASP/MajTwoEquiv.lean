/-
# `MAJ²`: the translations to and from `TL[◁#, ▷#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E: `thm:tlc_to_majtwo`, `thm:tlc_to_majtwo_closed`,
`thm:majtwo_to_tlc`, `thm:logical_inclusions`, `thm:ltc0_hierarchy`.

The two translations are depth-preserving in one direction and cost one level
in the other, so `MAJ²_k ⊆ TL[◁#,▷#]_k ⊆ MAJ²_{k+1}` and the two logics define
the same languages.

`thm:ltc0_hierarchy` concludes from this and `thm:TLC_depth` that the circuit
depth hierarchy for `FO[<]`-uniform `LTC⁰` is strict.  That last step is
Theorem 3 of Behle & Lange, about circuits, which this development does not
model; what is stated below is the half that lives in the logic — the `MAJ²`
depth hierarchy is strict.

Appendix E also carries a `lem:piecewise_testable_depth_majtwo` giving an
explicit `MAJ²_{k+1}` construction for `(2k+1)`-piecewise testable languages.
It sits inside an `\iffalse` block in the source and so is not part of the
paper; like `lem:bb` it is deliberately left out here.
-/

import Transformer.CRASP.MajTwoOfTLC

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

variable [DecidableEq σ]

/-- **Theorem `thm:tlc_to_majtwo`.**  A `TL[◁#,▷#]_k` formula is matched by a
`MAJ²_k` formula with one free variable `x`, read at the position `x` names. -/
theorem exists_majTwo_of_mem_TLC (k : ℕ) (φ : Form σ) (hφ : φ ∈ TLC σ k) :
    ∃ φ' ∈ MajTwo σ k, φ'.freeIn .y = false ∧
      ∀ (w : List σ) (i : ℕ), 1 ≤ i → i ≤ w.length →
        φ.sat w i = φ'.sat w (Function.update (fun _ => 0) Var.x i) := by
  refine ⟨φ.toMaj .x, (φ.depth_toMaj_le .x).trans hφ.2, φ.freeIn_toMaj .x, fun w i h₁ h₂ => ?_⟩
  rw [φ.sat_toMaj w .x _ hφ.1 (by rwa [Function.update_self]) (by rwa [Function.update_self]),
    Function.update_self]

/-- **Proposition `thm:tlc_to_majtwo_closed`.**  When the formula is
`#[ψ] > 0` — "at least one position satisfies `ψ`" — the `MAJ²` formula it
translates to is closed, which is what lets the translation be used on
languages rather than on formulas with a free variable. -/
theorem exists_closed_majTwo (k : ℕ) (ψ : Form σ) (hψ : ψ ∈ TLC σ k) :
    ∃ φ' ∈ MajTwo σ (k + 1), φ'.Closed ∧
      φ'.lang = {w : List σ | ∃ i, 1 ≤ i ∧ i ≤ w.length ∧ ψ.sat w i = true} := by
  refine ⟨.ex .x (ψ.toMaj .x), ?_, fun v => ?_, ?_⟩
  · simp only [MajTwo, Set.mem_ofPred_eq, Maj2.depth_ex]
    have := (ψ.depth_toMaj_le .x).trans hψ.2
    omega
  · have hy : (ψ.toMaj .x).freeIn .y = false := ψ.freeIn_toMaj .x
    cases v <;> simp [Maj2.ex, Maj2.freeIn, Maj2.top, Fin.exists_fin_two, hy]
  · ext w
    simp only [Maj2.lang, Maj2.models, Set.mem_ofPred_eq, Maj2.sat_ex, Finset.mem_Icc]
    refine exists_congr fun i => ?_
    constructor
    · rintro ⟨⟨h₁, h₂⟩, h⟩
      rw [ψ.sat_toMaj w .x _ hψ.1 (by rwa [Function.update_self])
        (by rwa [Function.update_self]), Function.update_self] at h
      exact ⟨h₁, h₂, h⟩
    · rintro ⟨h₁, h₂, h⟩
      refine ⟨⟨h₁, h₂⟩, ?_⟩
      rwa [ψ.sat_toMaj w .x _ hψ.1 (by rwa [Function.update_self])
        (by rwa [Function.update_self]), Function.update_self]

/-- **Theorem `thm:majtwo_to_tlc`.**  Conversely, a `MAJ²_k` formula with one
free variable `x` is matched by a `TL[◁#,▷#]_k` formula.

**What the source says and what is changed here.**  A hypothesis is added:
`φ` must not use the second variable `y`.  The paper says "a `MAJ²_k` formula
with one free variable", and `Maj2.freeIn φ .y = false` is what that means; the
formalization had `φ` range over all of `MajTwo σ k` and read it at the
valuation `x ↦ i`, `y ↦ 0`, which is a different and false claim.

*Why it is false without it.*  Take `φ = Q_a(y)`, of depth `0`, hence in
`MajTwo σ k` for every `k`.  At the valuation above `ξ y = 0`, and
`Maj2.sat` reads `w[ξ y - 1]?`, which is `w[0]?` in truncated subtraction — so
`φ` says "the first symbol of `w` is `a`", the same answer at every position
`i`.  That is not a `TL[◁#,▷#]_0` property: a depth-`0` formula is a Boolean
combination of `Q_b` and comparisons of constants, so it cannot see past the
current position, and `w = [a, b]`, `w' = [b, b]` read at `i = 2` agree on all
of them while requiring different answers.  Depth `1` does not suffice either
— `[a,b,b]` and `[b,a,b]` at `i = 3` agree on every `◁#`- and `▷#`-count of a
depth-`0` formula — and the property is first expressible at depth `2`, as
`◁#[Q_a ∧ ◁#[⊤] = 0] = 1`.  So no fixed `k` bounds the translation, and the
hypothesis is the paper's own reading of its statement.

The forward direction already produces formulas with this property:
`exists_majTwo_of_mem_TLC` returns `φ'` together with `φ'.freeIn .y = false`.

Source: arXiv:2506.16055v3, Appendix E, `thm:majtwo_to_tlc`. -/
theorem exists_mem_TLC_of_majTwo (k : ℕ) (φ : Maj2 σ) (hφ : φ ∈ MajTwo σ k)
    (hy : φ.freeIn .y = false) :
    ∃ φ' ∈ TLC σ k, ∀ (w : List σ) (i : ℕ), 1 ≤ i → i ≤ w.length →
      φ.sat w (Function.update (fun _ => 0) Var.x i) = φ'.sat w i :=
  sorry

/-- The hypotheses of the three translation theorems are satisfiable: `Q_a` is
a PNP-free formula of depth `0` on one side, and `Q_a(x)` is a `MAJ²` formula
of depth `0` on the other, in which `y` does not occur. -/
example (a : σ) (k : ℕ) :
    (Form.sym a : Form σ) ∈ TLC σ k ∧ (Maj2.sym a Var.x : Maj2 σ) ∈ MajTwo σ k ∧
      (Maj2.sym a Var.x : Maj2 σ).freeIn .y = false :=
  ⟨⟨rfl, Nat.zero_le k⟩, Nat.zero_le k, rfl⟩

/-- **Theorem `thm:logical_inclusions`, first inclusion.**  `TL[◁#,▷#]_k`
languages are `MAJ²_{k+1}` languages; the extra level pays for the end
satisfaction of the temporal logic, which becomes `∃x[¬∃y[y > x] ∧ φ'(x)]`.

Two things the paper leaves implicit are hypotheses here.  The depth has to be
positive, because "the position is the last one" — `Form.atEnd`, the `¬∃y[y>x]`
of the statement — is itself of depth `1`, and conjoining it is what turns end
satisfaction into the existential form `exists_closed_majTwo` produces; at
`k = 0` the inclusion is false, by `not_forall_closed_majTwo_of_definable`.
And the empty string, which no position of which satisfies anything, has to be
put back by hand: `¬∃x[⊤]` is the closed `MAJ²_1` formula that does it.

Source: arXiv:2506.16055v3, Appendix E, `thm:logical_inclusions`. -/
theorem exists_closed_majTwo_of_definable (k : ℕ) (hk : 0 < k) (L : Set (List σ))
    (h : Definable L k) :
    ∃ φ' ∈ MajTwo σ (k + 1), φ'.Closed ∧ φ'.lang = L := by
  obtain ⟨φ, ⟨hpnp, hdepth⟩, hlang⟩ := h
  have hmem : ∀ w : List σ, w ∈ L ↔ φ.sat w w.length = true := by
    intro w
    rw [← hlang]
    exact Iff.rfl
  obtain ⟨φ₀, hmem₀, hclosed₀, hlang₀⟩ :=
    exists_closed_majTwo k (φ.and Form.atEnd)
      ⟨by simp [Form.pnpFree, hpnp], by simp only [Form.depth, Form.depth_atEnd]; omega⟩
  have hdiff : {w : List σ | ∃ i, 1 ≤ i ∧ i ≤ w.length ∧ (φ.and Form.atEnd).sat w i = true}
      = L \ {[]} := by
    ext w
    simp only [Set.mem_ofPred_eq, Form.sat, Bool.and_eq_true, Form.sat_atEnd,
      decide_eq_true_eq, Set.mem_sdiff, Set.mem_singleton_iff]
    constructor
    · rintro ⟨i, hi₁, hi₂, hsat, hend⟩
      have hi : i = w.length := le_antisymm hi₂ hend
      subst hi
      exact ⟨(hmem w).mpr hsat, fun hw => by simp [hw] at hi₁⟩
    · rintro ⟨hw, hne⟩
      exact ⟨w.length, List.length_pos_iff.mpr hne, le_rfl, (hmem w).mp hw, le_rfl⟩
  by_cases hempty : ([] : List σ) ∈ L
  · have hex : ∀ w : List σ,
        (Maj2.ex Var.x (Maj2.top : Maj2 σ)).sat w (fun _ => 0) = true ↔ w ≠ [] := by
      intro w
      rw [Maj2.sat_ex]
      constructor
      · rintro ⟨i, hi, -⟩ rfl
        simp at hi
      · intro hne
        exact ⟨1, Finset.mem_Icc.mpr ⟨le_rfl, List.length_pos_iff.mpr hne⟩, by simp⟩
    refine ⟨.neg (.and (.neg φ₀) (Maj2.ex Var.x Maj2.top)), ?_, ?_, ?_⟩
    · have hone : (Maj2.ex Var.x (Maj2.top : Maj2 σ)).depth = 1 := by
        rw [Maj2.depth_ex, Maj2.top, Maj2.depth, Maj2.depth]
      simp only [MajTwo, Set.mem_ofPred_eq, Maj2.depth, hone, max_le_iff]
      exact ⟨hmem₀, by omega⟩
    · intro v
      simp only [Maj2.freeIn, Bool.or_eq_false_iff, hclosed₀ v, true_and]
      cases v <;> simp [Maj2.ex, Maj2.top, Maj2.freeIn, Fin.exists_fin_two]
    · ext w
      have hlang₀' : φ₀.sat w (fun _ => 0) = true ↔ w ∈ L ∧ w ≠ [] := by
        have hw : w ∈ φ₀.lang ↔ w ∈ L \ {[]} := by rw [hlang₀, hdiff]
        simpa [Maj2.lang, Maj2.models] using hw
      show (Maj2.neg (Maj2.and (Maj2.neg φ₀) (Maj2.ex Var.x Maj2.top)) : Maj2 σ).models w
        ↔ w ∈ L
      rw [Maj2.models]
      cases hb : φ₀.sat w (fun _ => 0)
      · simp only [Maj2.sat, hb, Bool.not_false, Bool.true_and]
        by_cases hc : w = []
        · subst hc
          have hnil : (Maj2.ex Var.x (Maj2.top : Maj2 σ)).sat [] (fun _ => 0) = false := by
            rcases Bool.eq_false_or_eq_true
              ((Maj2.ex Var.x (Maj2.top : Maj2 σ)).sat [] (fun _ => 0)) with h | h
            · exact absurd ((hex []).mp h) (by simp)
            · exact h
          simp [hnil, hempty]
        · rw [(hex w).mpr hc]
          simpa using fun hL => hc (by simpa [hb, hL] using hlang₀')
      · simp only [Maj2.sat, hb, Bool.not_true, Bool.false_and, Bool.not_false, true_iff]
        exact (hlang₀'.mp hb).1
  · exact ⟨φ₀, hmem₀, hclosed₀, by rw [hlang₀, hdiff, Set.sdiff_singleton_eq_self hempty]⟩

/-- **Theorem `thm:logical_inclusions`, second inclusion.**  `MAJ²_k`
languages are `TL[◁#,▷#]_k` languages. -/
theorem definable_of_closed_majTwo (k : ℕ) (φ : Maj2 σ) (hφ : φ ∈ MajTwo σ k)
    (hc : φ.Closed) : Definable φ.lang k :=
  sorry

/-- The hypotheses of the two inclusions are satisfiable: `k + 1` is positive,
`Σ*` is definable at every depth by `¬(1 < 1)`, and `¬(∃x[⊤] ∧ ¬∃x[⊤])` is a
closed `MAJ²` formula. -/
example (k : ℕ) : 0 < k + 1 ∧ Definable (σ := σ) Set.univ (k + 1) ∧
    (Maj2.closedTop : Maj2 σ) ∈ MajTwo σ 1 ∧ (Maj2.closedTop : Maj2 σ).Closed := by
  refine ⟨Nat.succ_pos k, ⟨.neg (.lt .one .one), ⟨rfl, Nat.zero_le _⟩, ?_⟩, ?_,
    Maj2.closed_closedTop⟩
  · ext w
    simp [Form.lang, Form.models, Form.sat, Term.val]
  · simp [MajTwo]

/-- **Corollary `thm:ltc0_hierarchy`, logical half.**  The `MAJ²` depth
hierarchy is strict: `D_k` is `MAJ²_k`-definable while `D_{k+1}` is not.  By
Theorem 3 of Behle & Lange this is equivalent to the strictness of the circuit
depth hierarchy for `FO[<]`-uniform `LTC⁰`, which is the corollary as stated in
the paper; circuits are not modelled here.

The second half is `definable_of_closed_majTwo` against `thm:TLC_depth` and is
proved.  The first half is not derivable from the two inclusions — they cost a
level, and `D_k` is `TL[◁#,▷#]`-definable at depth `k`, not at `k - 1` — so it
is exactly `lem:piecewise_testable_depth_majtwo`, the explicit `MAJ²_{k+1}`
construction for `(2k+1)`-piecewise testable languages that sits in an
`\iffalse` block of the source and is left out here with the rest of them. -/
theorem majTwo_depth_hierarchy (k : ℕ) (hk : 0 < k) :
    (∃ φ ∈ MajTwo Bool k, φ.Closed ∧ φ.lang = altPlusDouble k) ∧
      ∀ φ ∈ MajTwo Bool k, φ.Closed → φ.lang ≠ altPlusDouble (k + 1) := by
  refine ⟨sorry, fun φ hφ hc hlang => (definable_altPlusDouble k).2 ?_⟩
  exact hlang ▸ definable_of_closed_majTwo k φ hφ hc

end CRASP
end Transformer
