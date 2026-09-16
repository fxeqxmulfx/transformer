/-
# The Reduction Lemma is false as stated

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.4 `lem:reduction`, with its proof in Appendix C.3.

The lemma trades a depth-`k` formula for a depth-`(k-1)` formula `φ'` that
defines the affix-restricted language exactly, with the PNPs of `φ'` constant
on the middle.  The proof checks the affix with
`φ_aff = ◁#[(Π_a ∧ Q_a) ∨ (Π_b ∧ Q_b)] = ◁#[⊤]`, where `Π_σ` reads the letter
the affix puts at a position and is `⊤` in between, and claims that `Π_σ` is
constant on the middle.  It is not: the middle `[ℙ(λ(n⃗)), n⃗ - ℙ(ϱ(n⃗))]`
contains the prefix vector of position `|λ(n⃗)|`, where `Π_σ` reads the last
letter of `λ(n⃗)` and so is false for one of the two letters.

Nothing else can do the check at depth `1`.  Under `(λ, ϱ) = (ab, ε)` the
middle at `n⃗ = (2,2)` is `[(1,1), (2,2)]`, and `abab` enters it at position `2`
already, so a PNP constant on the middle takes one value at positions `2`, `3`
and `4` of every string with that Parikh vector.  A depth-0 formula then reads
positions `2, 3, 4` of `aabb` as it reads positions `3, 2, 4` of `abab`, the
counts of depth-1 formulas agree at the last position, and `abab`, which
matches the affix, cannot be told from `aabb`, which does not.
-/

import Transformer.CRASP.Commutative

namespace Transformer
namespace CRASP

/-- The affix restriction `(ab, ε)`: the strings that start with `ab`
(Definition `def:affix_restriction`). -/
def Affix.startAB : Affix Bool := ⟨fun _ => [false, true], fun _ => []⟩

/-- **A depth-1 formula with PNPs constant on the middle of `(ab, ε)` does not
tell `abab` from `aabb`** (arXiv:2506.16055v3, Appendix C.3, the claim that
`Π_σ` "is constant on the middle of `(λ, ϱ)`"). -/
theorem Form.sat_abab_eq_aabb (φ : Form Bool) (hφ : φ.depth ≤ 1)
    (hpnp : PnpsConstantOn φ Affix.startAB.middle) :
    φ.sat [false, true, false, true] 4 = φ.sat [false, false, true, true] 4 := by
  have hpar : parikh [false, true, false, true] = parikh [false, false, true, true] := by
    funext a
    cases a <;> rfl
  have hm : ∀ w i,
      prefixVec w i = parikh [false, true] ∨ prefixVec w i = parikh [false, false, true] →
        prefixVec w i ∈ Affix.startAB.middle (parikh [false, true, false, true]) := by
    rintro w i (h | h) <;> rw [h] <;>
      exact ⟨fun a => by cases a <;> decide, fun a => by cases a <;> decide⟩
  refine Form.sat_length_eq_of_depth_le_one (w := [false, true, false, true])
    (w' := [false, false, true, true]) rfl hpar rfl ?_ φ hφ hpnp
  intro χ hχ hχp
  have e1 := Form.sat_eq_of_depth_eq_zero (w := [false, true, false, true])
    (v := [false, false, true, true]) (i := 1) (j := 1) rfl χ hχ fun ψ hψ =>
      χ.sat_eq_of_mem_pnps hψ hpar 1
  have e4 := Form.sat_eq_of_depth_eq_zero (w := [false, true, false, true])
    (v := [false, false, true, true]) (i := 4) (j := 4) rfl χ hχ fun ψ hψ =>
      χ.sat_eq_of_mem_pnps hψ hpar 4
  have e2 := Form.sat_eq_of_depth_eq_zero (w := [false, true, false, true])
    (v := [false, false, true, true]) (i := 2) (j := 3) rfl χ hχ fun ψ hψ =>
      hχp ψ hψ _ _ _ 2 3 rfl hpar.symm (hm _ _ (Or.inl rfl)) (hm _ _ (Or.inr rfl))
  have e3 := Form.sat_eq_of_depth_eq_zero (w := [false, true, false, true])
    (v := [false, false, true, true]) (i := 3) (j := 2) rfl χ hχ fun ψ hψ =>
      (hχp ψ hψ _ _ _ 3 2 rfl rfl (hm _ _ (Or.inr (funext fun a => by cases a <;> rfl)))
        (hm _ _ (Or.inl rfl))).trans (χ.sat_eq_of_mem_pnps hψ hpar 2)
  show ((List.range' 1 4).filter fun j => χ.sat _ j).length =
    ((List.range' 1 4).filter fun j => χ.sat _ j).length
  simp only [show List.range' 1 4 = [1, 2, 3, 4] from rfl, List.filter_cons, List.filter_nil,
    e1, e2, e3, e4]
  cases χ.sat [false, false, true, true] 1 <;> cases χ.sat [false, false, true, true] 2 <;>
    cases χ.sat [false, false, true, true] 3 <;> cases χ.sat [false, false, true, true] 4 <;> rfl

/-- **No depth-1 formula with PNPs constant on the middle defines the strings
that start with `ab`** (arXiv:2506.16055v3, Appendix C.3): `abab` is one of
them and `aabb` is not. -/
theorem not_lang_eq_restrict_startAB (φ : Form Bool) (hφ : φ.depth ≤ 1)
    (hpnp : PnpsConstantOn φ Affix.startAB.middle) :
    φ.lang ≠ Affix.startAB.restrict (Form.neg (.lt .one .one)).lang := by
  intro h
  have hin :
      [false, true, false, true] ∈ Affix.startAB.restrict (Form.neg (.lt .one .one)).lang :=
    ⟨by simp [Form.lang, Form.models, Form.sat, Term.val], [false, true], rfl⟩
  have hout :
      [false, false, true, true] ∉ Affix.startAB.restrict (Form.neg (.lt .one .one)).lang :=
    fun ⟨_, w', hw'⟩ => by simp [Affix.startAB] at hw'
  rw [← h] at hin hout
  exact hout (show φ.sat _ 4 = true from (φ.sat_abab_eq_aabb hφ hpnp).symm.trans hin)

/-- The hypotheses of `Form.sat_abab_eq_aabb` and `not_lang_eq_restrict_startAB`
are satisfiable: `⊤`, written `¬(1 < 1)`, has depth `0` and no PNPs. -/
example : (Form.neg (.lt .one .one) : Form Bool).depth ≤ 1 ∧
    PnpsConstantOn (Form.neg (.lt .one .one) : Form Bool) Affix.startAB.middle :=
  ⟨Nat.zero_le 1, fun ψ hψ => by simp [Form.pnps, Term.pnps] at hψ⟩

end CRASP
end Transformer
