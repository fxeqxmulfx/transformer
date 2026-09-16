/-
# Formulas constant, letter by letter, on a strip past a prefix

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.4: "constant on `I`", the notion behind `lem:cropping_oneway`,
with the positions before the interval fixed.

The paper calls a formula constant on a family of intervals of Parikh vectors
when its value at a position depends only on the letter there while the prefix
vector ranges over the interval.  `lem:cropping_oneway` fails because a count
also reads the positions before the interval
(`Transformer.CRASP.CroppingUnsound`), so here those positions are a fixed
prefix `u`.  `ConstOnStrip ψ u p V q` says: at a position of `u ++ m` whose
prefix inside `m` holds `x` letters `a` and `y` letters `b`, with
`p ≤ x ≤ p + V` and `q ≤ y`, the value of `ψ` depends on the letter alone.  The
number of `a`s is bounded and that of `b`s is not, so the strip is open towards
`b`.

A strip after `u` becomes the whole quadrant `x ≤ V` after `u ++ μ` when `μ`
holds `p` letters `a` and `q` letters `b` (`ConstOnStrip.append`).  That is how
the argument climbs from one level of depth to the next.  A formula of depth
`0` without PNPs is constant on every strip
(`Form.constOnStrip_of_depth_eq_zero`).
-/

import Transformer.CRASP.DepthZero

namespace Transformer
namespace CRASP

/-- The position `|u| + i + 1` of `u ++ m` holds `m[i]` (Definition
`def:TLC_semantics`, positions numbered from `1`). -/
theorem getElem?_append_length_add {α : Type*} (u m : List α) {i : ℕ} (hi : i < m.length) :
    (u ++ m)[u.length + i + 1 - 1]? = some m[i] := by
  rw [Nat.add_sub_cancel, List.getElem?_append_right (Nat.le_add_right _ _),
    Nat.add_sub_cancel_left, List.getElem?_eq_getElem hi]

/-- The hypothesis of `getElem?_append_length_add` is satisfiable: `0 < |a|`. -/
example : 0 < [false].length := Nat.one_pos

/-- `ψ` is **constant, letter by letter, on the strip `p ≤ x ≤ p + V`, `q ≤ y`
after `u`**: at the position of `u ++ m` holding `m[i]`, where `m[0..i]` has `x`
letters `a` (`false`) and `y` letters `b` (`true`), the value of `ψ` is
`g m[i]`, for every `m` with at most `p + V` letters `a` (§4.4, "constant on
`I`", with the positions before the interval fixed to `u`). -/
def ConstOnStrip (ψ : Form Bool) (u : List Bool) (p V q : ℕ) : Prop :=
  ∃ g : Bool → Bool, ∀ m : List Bool, m.count false ≤ p + V → ∀ i (hi : i < m.length),
    p ≤ (m.take (i + 1)).count false → q ≤ (m.take (i + 1)).count true →
      ψ.sat (u ++ m) (u.length + i + 1) = g m[i]

/-- A formula constant on a strip is constant on every strip inside it (§4.4). -/
theorem ConstOnStrip.mono {ψ : Form Bool} {u : List Bool} {p V q p' V' q' : ℕ}
    (h : ConstOnStrip ψ u p V q) (hp : p ≤ p') (hV : p' + V' ≤ p + V) (hq : q ≤ q') :
    ConstOnStrip ψ u p' V' q' :=
  let ⟨g, hg⟩ := h
  ⟨g, fun m hm i hi hx hy => hg m (by omega) i hi (by omega) (by omega)⟩

/-- **A strip after `u` is the quadrant `x ≤ V` after `u ++ μ`**, when `μ` holds
`p` letters `a` and `q` letters `b`: the prefix vectors of the positions after
`u ++ μ` are those after `u` shifted by `(p, q)` (§4.4, where each round of
cropping moves the interval up and to the side). -/
theorem ConstOnStrip.append {ψ : Form Bool} {u μ : List Bool} {p V q : ℕ}
    (h : ConstOnStrip ψ u p V q) (hp : μ.count false = p) (hq : μ.count true = q) :
    ConstOnStrip ψ (u ++ μ) 0 V 0 := by
  obtain ⟨g, hg⟩ := h
  refine ⟨g, fun m hm i hi _ _ => ?_⟩
  have htake : (μ ++ m).take (μ.length + i + 1) = μ ++ m.take (i + 1) := by
    rw [Nat.add_assoc, List.take_length_add_append]
  have e := hg (μ ++ m) (by rw [List.count_append]; omega) (μ.length + i)
    (by rw [List.length_append]; omega) (by rw [htake, List.count_append]; omega)
    (by rw [htake, List.count_append]; omega)
  rw [List.append_assoc, List.length_append, Nat.add_assoc u.length, e,
    List.getElem_append_right (Nat.le_add_right _ _)]
  simp

/-- `¬ψ` is constant wherever `ψ` is (§4.4). -/
theorem ConstOnStrip.neg {ψ : Form Bool} {u : List Bool} {p V q : ℕ}
    (h : ConstOnStrip ψ u p V q) : ConstOnStrip (.neg ψ) u p V q :=
  let ⟨g, hg⟩ := h
  ⟨fun c => !g c, fun m hm i hi hx hy => by rw [Form.sat, hg m hm i hi hx hy]⟩

/-- `ψ₁ ∧ ψ₂` is constant wherever both are (§4.4). -/
theorem ConstOnStrip.and {ψ₁ ψ₂ : Form Bool} {u : List Bool} {p V q : ℕ}
    (h₁ : ConstOnStrip ψ₁ u p V q) (h₂ : ConstOnStrip ψ₂ u p V q) :
    ConstOnStrip (.and ψ₁ ψ₂) u p V q :=
  let ⟨g₁, hg₁⟩ := h₁
  let ⟨g₂, hg₂⟩ := h₂
  ⟨fun c => g₁ c && g₂ c, fun m hm i hi hx hy => by
    rw [Form.sat, hg₁ m hm i hi hx hy, hg₂ m hm i hi hx hy]⟩

/-- **A formula of depth `0` without PNPs is constant on every strip**: it reads
the letter at its position and nothing else (Definition `def:TLC_depth`;
Appendix C.1, the depth-0 cases of the proof of `lem:TLCP_commutative`). -/
theorem Form.constOnStrip_of_depth_eq_zero (u : List Bool) (p V q : ℕ) :
    ∀ ψ : Form Bool, ψ.pnpFree = true → ψ.depth = 0 → ConstOnStrip ψ u p V q
  | .sym a, _, _ => ⟨fun c => decide (some c = some a), fun m _ i hi _ _ => by
      rw [Form.sat, getElem?_append_length_add u m hi]⟩
  | .lt t₁ t₂, _, h => by
      rw [Form.depth] at h
      exact ⟨fun _ => decide (t₁.val [] 0 < t₂.val [] 0), fun m _ i _ _ _ => by
        rw [Form.sat, t₁.val_eq_of_depth_eq_zero _ [] _ 0 (by omega),
          t₂.val_eq_of_depth_eq_zero _ [] _ 0 (by omega)]⟩
  | .neg ψ, hf, h => by
      rw [Form.pnpFree] at hf
      rw [Form.depth] at h
      exact (Form.constOnStrip_of_depth_eq_zero u p V q ψ hf h).neg
  | .and ψ₁ ψ₂, hf, h => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.depth] at h
      exact (Form.constOnStrip_of_depth_eq_zero u p V q ψ₁ hf.1 (by omega)).and
        (Form.constOnStrip_of_depth_eq_zero u p V q ψ₂ hf.2 (by omega))
  | .pnp _, hf, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- The hypotheses of `Form.constOnStrip_of_depth_eq_zero` are satisfiable:
`Q_a` has depth `0` and no PNPs. -/
example : (Form.sym false : Form Bool).pnpFree = true ∧ (Form.sym false : Form Bool).depth = 0 :=
  ⟨rfl, rfl⟩

/-- The hypotheses of `ConstOnStrip.mono`, `ConstOnStrip.append`,
`ConstOnStrip.neg` and `ConstOnStrip.and` are satisfiable: `Q_a` on the strip
`x = 1`, `1 ≤ y` after `ε`, shifted by `ab`. -/
example : ConstOnStrip (.sym false) [] 1 0 1 ∧ 1 ≤ 1 ∧ 1 + 0 ≤ 1 + 0 ∧ 1 ≤ 1 ∧
    ([false, true] : List Bool).count false = 1 ∧ ([false, true] : List Bool).count true = 1 :=
  ⟨Form.constOnStrip_of_depth_eq_zero [] 1 0 1 _ rfl rfl, le_rfl, le_rfl, le_rfl, rfl, rfl⟩

end CRASP
end Transformer
