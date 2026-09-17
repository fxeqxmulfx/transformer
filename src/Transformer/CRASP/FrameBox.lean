/-
# A comparison is constant on a box of the middles of a frame

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D, the proof of `lem:cropping`: "a half-plane crossing the
interval is cropped away".

Inside the middle of a frame a comparison of two counts is a comparison of two
affine functions of the numbers `x` and `y` of `a`s and `b`s before the position
(`Term.val_affine_frame`), and on a box of side `V` inside a box of side `2V` it
is constant (`exists_box_lt`): each side of the comparison moves monotonically
in `x` and in `y` apart, so from the centre of the big box one of its four
quarters keeps the comparison as it is at the centre.  Unlike the one-sided
strip of `Transformer.CRASP.Shrink`, the box is bounded in `y` too: with the
whole Parikh vector fixed, `y` cannot grow past it.  `ConstOnBox` is the
property the cropping keeps, closed under Boolean combinations.
-/

import Transformer.CRASP.FrameAffine

namespace Transformer
namespace CRASP

/-- On a half of `[p₀, p₀ + 2V]`, `x ↦ β₁·x - β₂·x` stays below its value at the
centre `p₀ + V` (Appendix D, proof of `lem:cropping`). -/
theorem exists_half_mul_le (β₁ β₂ p₀ V : ℕ) :
    ∃ p, p₀ ≤ p ∧ p + V ≤ p₀ + 2 * V ∧
      ∀ x, p ≤ x → x ≤ p + V → β₁ * x + β₂ * (p₀ + V) ≤ β₁ * (p₀ + V) + β₂ * x := by
  rcases le_total β₁ β₂ with hβ | hβ
  · refine ⟨p₀ + V, le_add_right le_rfl, by omega, fun x hx _ => ?_⟩
    obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le hx
    have := Nat.mul_le_mul_right e hβ
    have h₁ : β₁ * (p₀ + V + e) = β₁ * (p₀ + V) + β₁ * e := Nat.mul_add ..
    have h₂ : β₂ * (p₀ + V + e) = β₂ * (p₀ + V) + β₂ * e := Nat.mul_add ..
    omega
  · refine ⟨p₀, le_rfl, by omega, fun x _ hx => ?_⟩
    obtain ⟨e, he⟩ := Nat.exists_eq_add_of_le hx
    have := Nat.mul_le_mul_right e hβ
    have h₁ : β₁ * (p₀ + V) = β₁ * x + β₁ * e := by rw [he, Nat.mul_add]
    have h₂ : β₂ * (p₀ + V) = β₂ * x + β₂ * e := by rw [he, Nat.mul_add]
    omega

/-- **A comparison of two affine functions is constant on a box.**  Inside
`[p₀, p₀ + 2V] × [q₀, q₀ + 2V]` there is a box `[p, p + V] × [q, q + V]` on which
`α₁ + β₁·x + γ₁·y < α₂ + β₂·x + γ₂·y` does not change (Appendix D, proof of
`lem:cropping`: a half-plane crossing the interval is cropped away). -/
theorem exists_box_lt (α₁ β₁ γ₁ α₂ β₂ γ₂ V p₀ q₀ : ℕ) :
    ∃ p q, p₀ ≤ p ∧ p + V ≤ p₀ + 2 * V ∧ q₀ ≤ q ∧ q + V ≤ q₀ + 2 * V ∧ ∃ b : Bool,
      ∀ x y, p ≤ x → x ≤ p + V → q ≤ y → y ≤ q + V →
        decide (α₁ + β₁ * x + γ₁ * y < α₂ + β₂ * x + γ₂ * y) = b := by
  by_cases hc : α₁ + β₁ * (p₀ + V) + γ₁ * (q₀ + V) < α₂ + β₂ * (p₀ + V) + γ₂ * (q₀ + V)
  · obtain ⟨p, hp, hpV, hx⟩ := exists_half_mul_le β₁ β₂ p₀ V
    obtain ⟨q, hq, hqV, hy⟩ := exists_half_mul_le γ₁ γ₂ q₀ V
    refine ⟨p, q, hp, hpV, hq, hqV, true, fun x y hpx hxp hqy hyq => ?_⟩
    have := hx x hpx hxp
    have := hy y hqy hyq
    rw [decide_eq_true_iff]
    omega
  · obtain ⟨p, hp, hpV, hx⟩ := exists_half_mul_le β₂ β₁ p₀ V
    obtain ⟨q, hq, hqV, hy⟩ := exists_half_mul_le γ₂ γ₁ q₀ V
    refine ⟨p, q, hp, hpV, hq, hqV, false, fun x y hpx hxp hqy hyq => ?_⟩
    have := hx x hpx hxp
    have := hy y hqy hyq
    rw [decide_eq_false_iff_not]
    omega

/-- A formula is **constant on the box** `[p, p + V] × [q, q + V]` of the middles
of a frame when, at the positions of a middle with `x` letters `a` and `y`
letters `b` before them in the middle, `(x, y)` in the box, it reads the letter
alone (Appendix D, "constant on `I`"). -/
def ConstOnBox (ψ : Form Bool) (u v : List Bool) (X Y p q V : ℕ) : Prop :=
  ∃ g : Bool → Bool, ∀ m : List Bool, m.count false = X → m.count true = Y →
    ∀ i (hi : i < m.length), p ≤ (m.take i).count false → (m.take i).count false ≤ p + V →
      q ≤ (m.take i).count true → (m.take i).count true ≤ q + V →
        ψ.sat (u ++ m ++ v) (u.length + i + 1) = g m[i]

/-- A formula constant on a box is constant on every box inside it
(Appendix D). -/
theorem ConstOnBox.mono {ψ : Form Bool} {u v : List Bool} {X Y p q V p' q' V' : ℕ}
    (h : ConstOnBox ψ u v X Y p q V) (hp : p ≤ p') (hpV : p' + V' ≤ p + V) (hq : q ≤ q')
    (hqV : q' + V' ≤ q + V) : ConstOnBox ψ u v X Y p' q' V' :=
  let ⟨g, hg⟩ := h
  ⟨g, fun m hX hY i hi hx hx' hy hy' =>
    hg m hX hY i hi (by omega) (by omega) (by omega) (by omega)⟩

/-- The hypotheses of `ConstOnBox.mono` are satisfiable: `Q_a` is constant on
every box, and `[1, 1] ⊆ [0, 2]`. -/
example : ConstOnBox (.sym false) [] [] 1 1 0 0 2 ∧ 0 ≤ 1 ∧ 1 + 0 ≤ 0 + 2 := by
  obtain ⟨⟨g, hg⟩, -⟩ := Form.constOnMiddle_of_depth_eq_zero [] [] 1 1 (.sym false) rfl rfl
  exact ⟨⟨g, fun m hX hY i hi _ _ _ _ => hg m hX hY i hi⟩, Nat.zero_le _, by omega⟩

/-- The negation of a formula constant on a box is constant on it
(Definition `def:TLC_semantics`). -/
theorem ConstOnBox.neg {ψ : Form Bool} {u v : List Bool} {X Y p q V : ℕ}
    (h : ConstOnBox ψ u v X Y p q V) : ConstOnBox (.neg ψ) u v X Y p q V :=
  let ⟨g, hg⟩ := h
  ⟨fun c => !g c, fun m hX hY i hi hx hx' hy hy' => by
    rw [Form.sat, hg m hX hY i hi hx hx' hy hy']⟩

/-- The conjunction of two formulas constant on a box is constant on it
(Definition `def:TLC_semantics`). -/
theorem ConstOnBox.and {ψ₁ ψ₂ : Form Bool} {u v : List Bool} {X Y p q V : ℕ}
    (h₁ : ConstOnBox ψ₁ u v X Y p q V) (h₂ : ConstOnBox ψ₂ u v X Y p q V) :
    ConstOnBox (.and ψ₁ ψ₂) u v X Y p q V :=
  let ⟨g₁, hg₁⟩ := h₁
  let ⟨g₂, hg₂⟩ := h₂
  ⟨fun c => g₁ c && g₂ c, fun m hX hY i hi hx hx' hy hy' => by
    rw [Form.sat, hg₁ m hX hY i hi hx hx' hy hy', hg₂ m hX hY i hi hx hx' hy hy']⟩

/-- The hypotheses of `ConstOnBox.neg` and `ConstOnBox.and` are satisfiable:
`Q_a` is constant on every box. -/
example : ConstOnBox (.sym false) [] [] 1 1 0 0 1 := by
  obtain ⟨⟨g, hg⟩, -⟩ := Form.constOnMiddle_of_depth_eq_zero [] [] 1 1 (.sym false) rfl rfl
  exact ⟨g, fun m hX hY i hi _ _ _ _ => hg m hX hY i hi⟩

end CRASP
end Transformer
