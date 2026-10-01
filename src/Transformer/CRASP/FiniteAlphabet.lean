/-
# Why the transformer simulation needs a finite alphabet

arXiv:2506.16055v3, Section 2.3 (`def:Parikh_map`) fixes a finite alphabet,
and Appendix B.2 starts `thm:rtfr_to_TLCl` with a finite disjunction over
its letters. Omitting that condition is false: a depth-zero transformer
on natural-number tokens recognizes parity, whereas a temporal formula
can name only finitely many letters.
-/

import Transformer.CRASP.Transformers

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]

mutual

/-- Letters named by the symbol atoms of a formula (Appendix B.2). -/
def Form.symbols : Form σ → Finset σ
  | .sym a => {a}
  | .lt t u => t.symbols ∪ u.symbols
  | .neg φ => φ.symbols
  | .and φ ψ => φ.symbols ∪ ψ.symbols
  | .pnp _ => ∅

/-- Letters named by the symbol atoms beneath a term (Appendix B.2). -/
def Term.symbols : Term σ → Finset σ
  | .countL φ | .countR φ => φ.symbols
  | .add t u => t.symbols ∪ u.symbols
  | .one => ∅

end

mutual

/-- A PNP-free formula cannot distinguish unnamed singleton tokens.
Source: arXiv:2506.16055v3, Appendix B.2, base case of `thm:rtfr_to_TLCl`. -/
theorem Form.sat_singleton_of_avoids (a b : σ) : ∀ φ : Form σ,
    φ.pnpFree = true → a ∉ φ.symbols → b ∉ φ.symbols →
      ∀ i : ℕ, φ.sat [a] i = φ.sat [b] i
  | .sym c, _, ha, hb, i => by
      have hac : a ≠ c := by simpa [Form.symbols] using ha
      have hbc : b ≠ c := by simpa [Form.symbols] using hb
      cases i with
      | zero => simp [Form.sat, hac, hbc]
      | succ i => cases i <;> simp [Form.sat, hac, hbc]
  | .lt t u, hp, ha, hb, i => by
      simp only [Form.pnpFree, Bool.and_eq_true] at hp
      simp only [Form.symbols, Finset.mem_union, not_or] at ha hb
      rw [Form.sat, Form.sat, t.val_singleton_of_avoids a b hp.1 ha.1 hb.1 i,
        u.val_singleton_of_avoids a b hp.2 ha.2 hb.2 i]
  | .neg φ, hp, ha, hb, i => by
      rw [Form.sat, Form.sat, φ.sat_singleton_of_avoids a b hp ha hb i]
  | .and φ ψ, hp, ha, hb, i => by
      simp only [Form.pnpFree, Bool.and_eq_true] at hp
      simp only [Form.symbols, Finset.mem_union, not_or] at ha hb
      rw [Form.sat, Form.sat, φ.sat_singleton_of_avoids a b hp.1 ha.1 hb.1 i,
        ψ.sat_singleton_of_avoids a b hp.2 ha.2 hb.2 i]
  | .pnp _, hp, _, _, _ => by cases hp

/-- The corresponding counting terms also agree on unnamed singleton tokens.
Source: arXiv:2506.16055v3, Appendix B.2, `thm:rtfr_to_TLCl`. -/
theorem Term.val_singleton_of_avoids (a b : σ) : ∀ t : Term σ,
    t.pnpFree = true → a ∉ t.symbols → b ∉ t.symbols →
      ∀ i : ℕ, t.val [a] i = t.val [b] i
  | .countL φ, hp, ha, hb, i => by
      rw [Term.val, Term.val]
      exact congrArg List.length (List.filter_congr fun j _ =>
        φ.sat_singleton_of_avoids a b hp ha hb j)
  | .countR φ, hp, ha, hb, i => by
      rw [Term.val, Term.val]
      exact congrArg List.length (List.filter_congr fun j _ =>
        φ.sat_singleton_of_avoids a b hp ha hb j)
  | .add t u, hp, ha, hb, i => by
      simp only [Term.pnpFree, Bool.and_eq_true] at hp
      simp only [Term.symbols, Finset.mem_union, not_or] at ha hb
      rw [Term.val, Term.val, t.val_singleton_of_avoids a b hp.1 ha.1 hb.1 i,
        u.val_singleton_of_avoids a b hp.2 ha.2 hb.2 i]
  | .one, _, _, _, _ => rfl

end

/-- The unnamed-letter hypotheses are satisfiable (Appendix B.2). -/
example : (Form.sym (2 : ℕ)).pnpFree = true ∧ 0 ∉ (Form.sym (2 : ℕ)).symbols ∧
    1 ∉ (Form.sym (2 : ℕ)).symbols ∧ (Term.one : Term ℕ).pnpFree = true ∧
    0 ∉ (Term.one : Term ℕ).symbols ∧ 1 ∉ (Term.one : Term ℕ).symbols := by
  simp [Form.pnpFree, Form.symbols, Term.pnpFree, Term.symbols]

/-- The representable value `1` for a two-bit, integer-only model (B.1). -/
def Fx.unit2 : Fx 2 0 := ⟨1, by decide, by decide⟩

/-- A depth-zero model whose embedding marks even natural-number tokens.
The other projections are zero because no attention layer is reached.
Source: arXiv:2506.16055v3, Appendix B.1, `def:transformer`. -/
def RTfr.parityRecognizer : RTfr (Option ℕ) 2 0 1 0 where
  E a _ := match a with
    | none => 0
    | some n => if Even n then Fx.unit2 else 0
  WQ _ _ _ := 0
  WK _ _ _ := 0
  WV _ _ _ := 0
  ff _ _ _ := 0
  Wout h := h 0

/-- On a singleton the model accepts exactly the even tokens (Appendix B.1). -/
theorem RTfr.parityRecognizer_accepts (n : ℕ) :
    parityRecognizer.Accepts (bos [n]) ↔ Even n := by
  by_cases hn : Even n <;>
    simp [RTfr.Accepts, RTfr.out, RTfr.act, parityRecognizer, bos, hn, Fx.unit2, Fx.val]

/-- **The alphabet-free version of `thm:rtfr_to_TLCl` is false.**
`RTfr.parityRecognizer` has no PNP-free temporal definition at any depth.
Every such formula mentions finitely many natural-number tokens, so an even
and an odd token above all of them give the same singleton answer.

The manuscript assumes a finite alphabet in Section 2.3, `def:Parikh_map`,
and uses it explicitly in Appendix B.2's base case. This refutes the former
Lean signature without that assumption, not the finite-alphabet proposition.
Source: arXiv:2506.16055v3, Appendix B.2, `thm:rtfr_to_TLCl`. -/
theorem RTfr.parityRecognizer_not_definable :
    ¬ ∃ φ : Form ℕ, φ.pnpFree = true ∧
      φ.lang = {w | parityRecognizer.Accepts (bos w)} := by
  rintro ⟨φ, hp, hlang⟩
  let bound := φ.symbols.sup id
  let a := 2 * (bound + 1)
  let b := a + 1
  have ha : a ∉ φ.symbols := by
    intro hm
    have h : a ≤ bound := Finset.le_sup (f := id) hm
    dsimp [a] at h
    omega
  have hb : b ∉ φ.symbols := by
    intro hm
    have h : b ≤ bound := Finset.le_sup (f := id) hm
    dsimp [b, a] at h
    omega
  have heven : Even a := ⟨bound + 1, by dsimp [a]; omega⟩
  have hodd : ¬ Even b := by
    rintro ⟨j, hj⟩
    dsimp [b, a] at hj
    omega
  have hA : [a] ∈ φ.lang := by
    rw [hlang]
    exact (parityRecognizer_accepts a).mpr heven
  have hB : [b] ∈ φ.lang := by
    change φ.sat [b] 1 = true
    rw [← φ.sat_singleton_of_avoids a b hp ha hb 1]
    exact hA
  rw [hlang] at hB
  exact hodd ((parityRecognizer_accepts b).mp hB)

end Transformer.CRASP
