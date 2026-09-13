/-
# `MAJ²`: the translations to and from `TL[◁#, ▷#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D: `thm:tlc_to_majtwo`, `thm:tlc_to_majtwo_closed`,
`thm:majtwo_to_tlc`, `thm:logical_inclusions`, `thm:ltc0_hierarchy`.

The two translations are depth-preserving in one direction and cost one level
in the other, so `MAJ²_k ⊆ TL[◁#,▷#]_k ⊆ MAJ²_{k+1}` and the two logics define
the same languages.

`thm:ltc0_hierarchy` concludes from this and `thm:TLC_depth` that the circuit
depth hierarchy for `FO[<]`-uniform `LTC⁰` is strict.  That last step is
Theorem 3 of Behle & Lange, about circuits, which this development does not
model; what is stated below is the half that lives in the logic — the `MAJ²`
depth hierarchy is strict.

Appendix D also carries a `lem:piecewise_testable_depth_majtwo` giving an
explicit `MAJ²_{k+1}` construction for `(2k+1)`-piecewise testable languages.
It sits inside an `\iffalse` block in the source and so is not part of the
paper; like `lem:bb` it is deliberately left out here.
-/

import Transformer.CRASP.MajTwo

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
        φ.sat w i = φ'.sat w (Function.update (fun _ => 0) Var.x i) :=
  sorry

/-- **Proposition `thm:tlc_to_majtwo_closed`.**  When the formula is
`#[ψ] > 0` — "at least one position satisfies `ψ`" — the `MAJ²` formula it
translates to is closed, which is what lets the translation be used on
languages rather than on formulas with a free variable. -/
theorem exists_closed_majTwo (k : ℕ) (ψ : Form σ) (hψ : ψ ∈ TLC σ k) :
    ∃ φ' ∈ MajTwo σ (k + 1), φ'.Closed ∧
      φ'.lang = {w : List σ | ∃ i, 1 ≤ i ∧ i ≤ w.length ∧ ψ.sat w i = true} :=
  sorry

/-- **Theorem `thm:majtwo_to_tlc`.**  Conversely, a `MAJ²_k` formula with one
free variable `x` is matched by a `TL[◁#,▷#]_k` formula. -/
theorem exists_mem_TLC_of_majTwo (k : ℕ) (φ : Maj2 σ) (hφ : φ ∈ MajTwo σ k) :
    ∃ φ' ∈ TLC σ k, ∀ (w : List σ) (i : ℕ), 1 ≤ i → i ≤ w.length →
      φ.sat w (Function.update (fun _ => 0) Var.x i) = φ'.sat w i :=
  sorry

/-- The hypotheses of the three translation theorems are satisfiable: `Q_a` is
a PNP-free formula of depth `0` on one side, and `Q_a(x)` is a `MAJ²` formula
of depth `0` on the other. -/
example (a : σ) (k : ℕ) :
    (Form.sym a : Form σ) ∈ TLC σ k ∧ (Maj2.sym a Var.x : Maj2 σ) ∈ MajTwo σ k :=
  ⟨⟨rfl, Nat.zero_le k⟩, Nat.zero_le k⟩

/-- **Theorem `thm:logical_inclusions`, first inclusion.**  `TL[◁#,▷#]_k`
languages are `MAJ²_{k+1}` languages; the extra level pays for the end
satisfaction of the temporal logic, which becomes `∃x[¬∃y[y > x] ∧ φ'(x)]`. -/
theorem exists_closed_majTwo_of_definable (k : ℕ) (L : Set (List σ)) (h : Definable L k) :
    ∃ φ' ∈ MajTwo σ (k + 1), φ'.Closed ∧ φ'.lang = L :=
  sorry

/-- **Theorem `thm:logical_inclusions`, second inclusion.**  `MAJ²_k`
languages are `TL[◁#,▷#]_k` languages. -/
theorem definable_of_closed_majTwo (k : ℕ) (φ : Maj2 σ) (hφ : φ ∈ MajTwo σ k)
    (hc : φ.Closed) : Definable φ.lang k :=
  sorry

/-- The hypotheses of the two inclusions are satisfiable: `Σ*` is definable at
every depth, by `¬(1 < 1)`, and `¬(∃x[⊤] ∧ ¬∃x[⊤])` is a closed `MAJ²`
formula. -/
example (k : ℕ) : Definable (σ := σ) Set.univ k ∧
    (Maj2.closedTop : Maj2 σ) ∈ MajTwo σ 1 ∧ (Maj2.closedTop : Maj2 σ).Closed := by
  refine ⟨⟨.neg (.lt .one .one), ⟨rfl, Nat.zero_le k⟩, ?_⟩, ?_, Maj2.closed_closedTop⟩
  · ext w
    simp [Form.lang, Form.models, Form.sat, Term.val]
  · simp [MajTwo]

/-- **Corollary `thm:ltc0_hierarchy`, logical half.**  The `MAJ²` depth
hierarchy is strict: `D_k` is `MAJ²_k`-definable while `D_{k+1}` is not.  By
Theorem 3 of Behle & Lange this is equivalent to the strictness of the circuit
depth hierarchy for `FO[<]`-uniform `LTC⁰`, which is the corollary as stated in
the paper; circuits are not modelled here. -/
theorem majTwo_depth_hierarchy (k : ℕ) (hk : 0 < k) :
    (∃ φ ∈ MajTwo Bool k, φ.Closed ∧ φ.lang = altPlusDouble k) ∧
      ∀ φ ∈ MajTwo Bool k, φ.Closed → φ.lang ≠ altPlusDouble (k + 1) :=
  sorry

end CRASP
end Transformer
