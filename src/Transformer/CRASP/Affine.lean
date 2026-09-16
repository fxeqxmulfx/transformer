/-
# Past a prefix, a count of constant formulas is affine in the letter counts

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.4 and Appendix C.2, the proof of `lem:cropping_oneway`: "each
`ψ_ℓ` defines a half-plane".

What the paper's claim needs, and what makes it true.  When every formula
under the counts of a term is constant, letter by letter, on the quadrant
`x ≤ W` after a prefix `u`, the term at a position of `u ++ m` is
`α + β·x + γ·y`, with `x` and `y` the numbers of `a`s and `b`s in `m` up to that
position (`Term.val_affine`): the count over `u` is a constant `α`, since a
past-only formula does not look ahead, and the count over `m` counts the
letters on which the formula holds.  A comparison of two such terms is then a
half-plane in `(x, y)`, and a formula over them is determined by the letter
and `(x, y)` (`Form.sat_eq_of_count_eq`).
-/

import Transformer.CRASP.Locality
import Transformer.CRASP.Middle
import Transformer.CRASP.Strip

namespace Transformer
namespace CRASP

/-- The letters of a string over `{a, b}` on which `g` holds number
`g(a)·x + g(b)·y`, with `x` and `y` the numbers of `a`s and `b`s (§4.4). -/
theorem countP_bool_eq (g : Bool → Bool) :
    ∀ l : List Bool, l.countP g = (g false).toNat * l.count false + (g true).toNat * l.count true
  | [] => rfl
  | c :: l => by
      rw [List.countP_cons, List.count_cons, List.count_cons, countP_bool_eq g l]
      cases c <;> cases g false <;> cases g true <;> simp <;> omega

/-- **Past `u`, a past-only term whose counted formulas are constant is affine
in the letter counts.**  At the position of `u ++ m` holding `m[i]` it is
`α + β·x + γ·y`, where `m[0..i]` holds `x` letters `a` and `y` letters `b`
(Appendix C.2, proof of `lem:cropping_oneway`, where each minimal depth-1
subformula "defines a half-plane"; the constant `α` is the count over the
positions before the interval, which the paper leaves out). -/
theorem Term.val_affine (u : List Bool) (W : ℕ) :
    ∀ t : Term Bool, t.past = true → t.pnpFree = true →
      (∀ ψ ∈ t.countSubs, ConstOnStrip ψ u 0 W 0) → ∃ α β γ : ℕ, ∀ m : List Bool,
        m.count false ≤ W → ∀ i, i < m.length → t.val (u ++ m) (u.length + i + 1) =
          α + β * (m.take (i + 1)).count false + γ * (m.take (i + 1)).count true
  | .countL ψ, hp, hf, h => by
      rw [Term.past] at hp
      rw [Term.pnpFree] at hf
      obtain ⟨g, hg⟩ := h ψ (by rw [Term.countSubs]; exact List.mem_cons_self ..)
      refine ⟨(List.range' 1 u.length).countP fun j => ψ.sat u j, (g false).toNat, (g true).toNat,
        fun m hm i hi => ?_⟩
      have hlen : (m.take (i + 1)).length = i + 1 := by rw [List.length_take]; omega
      have hm' := countP_range'_eq_countP g (m.take (i + 1))
        (fun j => ψ.sat (u ++ m) (u.length + j)) fun j a hj ha => by
          rw [List.getElem?_eq_some_iff] at ha
          obtain ⟨hj', rfl⟩ := ha
          rw [List.getElem_take]
          obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
          simpa [Nat.add_assoc] using
            hg m (by omega) j (by omega) (Nat.zero_le _) (Nat.zero_le _)
      rw [hlen] at hm'
      rw [Term.val, ← List.countP_eq_length_filter, Nat.add_assoc, countP_range'_add, hm',
        countP_bool_eq, List.countP_congr fun j hj => by
          rw [ψ.sat_append u m hp hf j (List.mem_range'_1.mp hj).1
            (by have := (List.mem_range'_1.mp hj).2; omega)], Nat.add_assoc]
  | .countR _, hp, _, _ => by rw [Term.past] at hp; exact absurd hp Bool.false_ne_true
  | .add t₁ t₂, hp, hf, h => by
      rw [Term.past, Bool.and_eq_true] at hp
      rw [Term.pnpFree, Bool.and_eq_true] at hf
      rw [Term.countSubs] at h
      obtain ⟨α₁, β₁, γ₁, h₁⟩ := Term.val_affine u W t₁ hp.1 hf.1 fun ψ hψ =>
        h ψ (List.mem_append_left _ hψ)
      obtain ⟨α₂, β₂, γ₂, h₂⟩ := Term.val_affine u W t₂ hp.2 hf.2 fun ψ hψ =>
        h ψ (List.mem_append_right _ hψ)
      exact ⟨α₁ + α₂, β₁ + β₂, γ₁ + γ₂, fun m hm i hi => by
        rw [Term.val, h₁ m hm i hi, h₂ m hm i hi, Nat.add_mul, Nat.add_mul]; omega⟩
  | .one, _, _, _ => ⟨1, 0, 0, fun _ _ _ _ => by rw [Term.val]; omega⟩

/-- The hypotheses of `Term.val_affine` are satisfiable: `◁#[Q_a]` after `ε`,
whose counted formula has depth `0`. -/
example : (Term.countL (.sym false) : Term Bool).past = true ∧
    (Term.countL (.sym false) : Term Bool).pnpFree = true ∧
      ∀ ψ ∈ (Term.countL (.sym false) : Term Bool).countSubs, ConstOnStrip ψ [] 0 1 0 := by
  refine ⟨rfl, rfl, fun ψ hψ => ?_⟩
  rw [Term.countSubs, Form.countSubs, List.mem_singleton] at hψ
  exact hψ ▸ Form.constOnStrip_of_depth_eq_zero [] 0 1 0 _ rfl rfl

/-- **Past `u`, a formula over constant counts reads the letter and the letter
counts.**  Two positions after `u` with the same letter and the same numbers of
`a`s and `b`s since `u` satisfy the same past-only formulas without PNPs whose
counted formulas are constant (Appendix C.2, proof of `lem:cropping_oneway`:
such a formula is a Boolean combination of half-planes and letters). -/
theorem Form.sat_eq_of_count_eq (u : List Bool) (W : ℕ) {m m' : List Bool} {i i' : ℕ}
    (hm : m.count false ≤ W) (hm' : m'.count false ≤ W) (hi : i < m.length)
    (hi' : i' < m'.length) (hx : (m.take (i + 1)).count false = (m'.take (i' + 1)).count false)
    (hy : (m.take (i + 1)).count true = (m'.take (i' + 1)).count true) (hc : m[i] = m'[i']) :
    ∀ φ : Form Bool, φ.past = true → φ.pnpFree = true →
      (∀ ψ ∈ φ.countSubs, ConstOnStrip ψ u 0 W 0) →
        φ.sat (u ++ m) (u.length + i + 1) = φ.sat (u ++ m') (u.length + i' + 1)
  | .sym _, _, _, _ => by
      rw [Form.sat, Form.sat, getElem?_append_length_add u m hi,
        getElem?_append_length_add u m' hi', hc]
  | .lt t₁ t₂, hp, hf, h => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.countSubs] at h
      obtain ⟨α₁, β₁, γ₁, h₁⟩ := Term.val_affine u W t₁ hp.1 hf.1 fun ψ hψ =>
        h ψ (List.mem_append_left _ hψ)
      obtain ⟨α₂, β₂, γ₂, h₂⟩ := Term.val_affine u W t₂ hp.2 hf.2 fun ψ hψ =>
        h ψ (List.mem_append_right _ hψ)
      rw [Form.sat, Form.sat, h₁ m hm i hi, h₁ m' hm' i' hi', h₂ m hm i hi, h₂ m' hm' i' hi', hx, hy]
  | .neg φ, hp, hf, h => by
      rw [Form.past] at hp
      rw [Form.pnpFree] at hf
      rw [Form.countSubs] at h
      rw [Form.sat, Form.sat, Form.sat_eq_of_count_eq u W hm hm' hi hi' hx hy hc φ hp hf h]
  | .and φ₁ φ₂, hp, hf, h => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.countSubs] at h
      rw [Form.sat, Form.sat,
        Form.sat_eq_of_count_eq u W hm hm' hi hi' hx hy hc φ₁ hp.1 hf.1 fun ψ hψ =>
          h ψ (List.mem_append_left _ hψ),
        Form.sat_eq_of_count_eq u W hm hm' hi hi' hx hy hc φ₂ hp.2 hf.2 fun ψ hψ =>
          h ψ (List.mem_append_right _ hψ)]
  | .pnp _, _, hf, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- The hypotheses of `Form.sat_eq_of_count_eq` are satisfiable: the last
positions of `aabb` and `abab`, which hold two `a`s and two `b`s each and end in
`b`, and `Q_a`. -/
example : ([false, false, true, true] : List Bool).count false ≤ 2 ∧
    ([false, true, false, true] : List Bool).count false ≤ 2 ∧
    3 < ([false, false, true, true] : List Bool).length ∧
    3 < ([false, true, false, true] : List Bool).length ∧
    (([false, false, true, true] : List Bool).take 4).count false =
      (([false, true, false, true] : List Bool).take 4).count false ∧
    (([false, false, true, true] : List Bool).take 4).count true =
      (([false, true, false, true] : List Bool).take 4).count true ∧
    ([false, false, true, true] : List Bool)[3] = ([false, true, false, true] : List Bool)[3] ∧
    (Form.sym false : Form Bool).past = true ∧ (Form.sym false : Form Bool).pnpFree = true ∧
    ∀ ψ ∈ (Form.sym false : Form Bool).countSubs, ConstOnStrip ψ [] 0 2 0 := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, rfl, rfl, rfl,
    fun ψ hψ => ?_⟩
  rw [Form.countSubs] at hψ
  exact absurd hψ List.not_mem_nil

end CRASP
end Transformer
