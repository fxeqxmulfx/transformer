/-
# The reverse translation and language inclusion

arXiv:2506.16055v3, Appendix E, `thm:majtwo_to_tlc` and the reverse
inclusion of `thm:logical_inclusions`. Binary rectangle normalization gives
the temporal formula; the empty-word guard gives the language statement.
-/

import Transformer.CRASP.MajTwoTranslation

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]

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
      φ.sat w (Function.update (fun _ => 0) Var.x i) = φ'.sat w i := by
  obtain ⟨R, hR, hc⟩ := φ.exists_rectangles
  have hm := MajRects.diagonal_mem hR
  refine ⟨R.diagonal, ⟨hm.1, hm.2.trans hφ⟩, ?_⟩
  intro w i hi hn
  have heq : φ.sat w (Function.update (fun _ => 0) Var.x i) = φ.sat w (fun _ => i) := by
    apply φ.sat_eq_of_agree_free
    intro v hv
    cases v
    · simp
    · rw [hy] at hv; cases hv
  have hr := hc w (fun _ => i) (fun _ => ⟨hi, hn⟩)
  have hd : R.diagonal.sat w i = φ.sat w (fun _ => i) := by
    rw [MajRects.sat_diagonal, hr]
    cases φ.sat w (fun _ => i) <;> rfl
  exact heq.trans hd.symm

/-- **Theorem `thm:logical_inclusions`, second inclusion.** `MAJ²_k`
languages are `TL[◁#,▷#]_k` languages. The binary translation is read on
the diagonal; a depth-one position guard restores the empty word. A closed
`MAJ²` formula always has positive depth, so that guard costs no extra level.

Source: arXiv:2506.16055v3, Appendix E, `thm:logical_inclusions`. -/
theorem definable_of_closed_majTwo (k : ℕ) (φ : Maj2 σ) (hφ : φ ∈ MajTwo σ k)
    (hc : φ.Closed) : Definable φ.lang k := by
  have hd : φ.depth ≤ k := hφ
  have hk : 1 ≤ k := by
    by_contra h
    have hz : φ.depth = 0 := by omega
    exact Maj2.not_closed_of_depth_eq_zero φ hz hc
  obtain ⟨ψ, hψ, hs⟩ := exists_mem_TLC_of_majTwo k φ hφ (hc .y)
  let emptyTest : Form σ := if φ.sat [] (fun _ => 0) then Form.topAt 0 else .lt .one .one
  have he : emptyTest.pnpFree = true ∧ emptyTest.depth = 0 := by
    dsimp [emptyTest]
    split_ifs <;> exact ⟨rfl, rfl⟩
  refine ⟨Form.or (.and Form.pos ψ) (.and (.neg Form.pos) emptyTest), ⟨?_, ?_⟩, ?_⟩
  · simp [Form.or, Form.pnpFree, Form.pos, Form.topAt, Term.pnpFree, hψ.1, he.1]
  · simp only [Form.depth_or, Form.depth, Form.depth_pos, he.2, max_le_iff]
    exact ⟨⟨hk, hψ.2⟩, hk, Nat.zero_le k⟩
  · ext w
    by_cases hw : w = []
    · subst w
      have hempty : emptyTest.sat [] 0 = φ.sat [] (fun _ => 0) := by
        dsimp [emptyTest]
        cases φ.sat [] (fun _ => 0) <;> simp [Form.sat, Term.val]
      simp [Form.lang, Form.models, Form.sat, hempty, Maj2.lang, Maj2.models]
    · have hlen : 1 ≤ w.length := List.length_pos_iff.mpr hw
      have hpos : (Form.pos : Form σ).sat w w.length = true := by
        rw [Form.sat_pos]; simpa using (show 0 < w.length by omega)
      have hsat := hs w w.length hlen le_rfl
      rw [φ.sat_eq_of_closed hc w _ (fun _ => 0)] at hsat
      simp only [Form.lang, Form.models, Maj2.lang, Maj2.models, Set.mem_ofPred_eq,
        Form.sat_or, Form.sat, hpos, Bool.true_and, Bool.not_true, Bool.false_and,
        Bool.or_false]
      rw [← hsat]

/-- The translation and inclusion hypotheses have witnesses (Appendix E). -/
example : (Maj2.sym true .x : Maj2 Bool) ∈ MajTwo Bool 0 ∧
    (Maj2.sym true .x : Maj2 Bool).freeIn .y = false ∧
    (Maj2.closedTop : Maj2 Bool) ∈ MajTwo Bool 1 ∧ (Maj2.closedTop : Maj2 Bool).Closed :=
  ⟨by simp [MajTwo, Maj2.depth], rfl, by simp [MajTwo], Maj2.closed_closedTop⟩

end Transformer.CRASP
