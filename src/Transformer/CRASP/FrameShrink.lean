/-
# Shrinking a box until a formula is constant on it

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D: `lem:cropping`, repaired.

The paper crops an interval of Parikh vectors until a formula of
`TL[◁#,▷#]` whose counted formulas are constant becomes constant itself.  With
the positions around the interval fixed to a frame
(`Transformer.CRASP.Frame`), that works.  A comparison is constant on a box of
half the side (`exists_box_lt`), once for each letter; Boolean combinations
shrink the box once per operand.  So a formula, and a list of formulas, is
constant on a box whose side depends on the formulas alone
(`Form.exists_constOnBox`, `exists_constOnBox_list`).
-/

import Transformer.CRASP.FrameBox

namespace Transformer
namespace CRASP

/-- **A formula whose counted formulas are constant on the middles is constant
on a box.**  For every side `V` there is a side `N` such that, on any frame on
whose middles the formulas under the counts of `ψ` are constant, every box of
side `N` holds a box of side `V` on which `ψ` is constant (Appendix D,
`lem:cropping`, with the positions around the interval fixed to the frame; `N`
depends on `ψ` alone). -/
theorem Form.exists_constOnBox :
    ∀ ψ : Form Bool, ψ.pnpFree = true → ∀ V, ∃ N, ∀ u v X Y p₀ q₀,
      (∀ χ ∈ ψ.countSubs, ConstOnMiddle χ u v X Y) →
        ∃ p q, p₀ ≤ p ∧ p + V ≤ p₀ + N ∧ q₀ ≤ q ∧ q + V ≤ q₀ + N ∧ ConstOnBox ψ u v X Y p q V
  | .sym a, _, V => ⟨V, fun u v X Y p₀ q₀ _ => by
      obtain ⟨⟨g, hg⟩, -⟩ := Form.constOnMiddle_of_depth_eq_zero u v X Y (.sym a) rfl rfl
      exact ⟨p₀, q₀, le_rfl, le_rfl, le_rfl, le_rfl,
        g, fun m hX hY i hi _ _ _ _ => hg m hX hY i hi⟩⟩
  | .lt t₁ t₂, _, V => by
      refine ⟨2 * (2 * V), fun u v X Y p₀ q₀ h => ?_⟩
      rw [Form.countSubs] at h
      obtain ⟨K₁, β₁, γ₁, β₁', γ₁', h₁⟩ := Term.val_affine_frame u v X Y t₁ fun χ hχ =>
        h χ (List.mem_append_left _ hχ)
      obtain ⟨K₂, β₂, γ₂, β₂', γ₂', h₂⟩ := Term.val_affine_frame u v X Y t₂ fun χ hχ =>
        h χ (List.mem_append_right _ hχ)
      obtain ⟨p₁, q₁, hp₁, hpV₁, hq₁, hqV₁, b₀, hb₀⟩ := exists_box_lt (K₁ false) (β₁ + β₂')
        (γ₁ + γ₂') (K₂ false) (β₂ + β₁') (γ₂ + γ₁') (2 * V) p₀ q₀
      obtain ⟨p₂, q₂, hp₂, hpV₂, hq₂, hqV₂, b₁, hb₁⟩ := exists_box_lt (K₁ true) (β₁ + β₂')
        (γ₁ + γ₂') (K₂ true) (β₂ + β₁') (γ₂ + γ₁') V p₁ q₁
      refine ⟨p₂, q₂, by omega, by omega, by omega, by omega, fun c => cond c b₁ b₀,
        fun m hX hY i hi hx hx' hy hy' => ?_⟩
      have e₁ := h₁ m hX hY i hi
      have e₂ := h₂ m hX hY i hi
      rw [Form.sat]
      generalize m[i] = c at e₁ e₂ ⊢
      generalize (m.take i).count false = x at e₁ e₂ hx hx'
      generalize (m.take i).count true = y at e₁ e₂ hy hy'
      cases c
      · show decide _ = b₀
        rw [← hb₀ x y (by omega) (by omega) (by omega) (by omega), decide_eq_decide,
          Nat.add_mul, Nat.add_mul, Nat.add_mul, Nat.add_mul]
        omega
      · show decide _ = b₁
        rw [← hb₁ x y (by omega) (by omega) (by omega) (by omega), decide_eq_decide,
          Nat.add_mul, Nat.add_mul, Nat.add_mul, Nat.add_mul]
        omega
  | .neg ψ, hf, V => by
      rw [Form.pnpFree] at hf
      obtain ⟨N, hN⟩ := Form.exists_constOnBox ψ hf V
      refine ⟨N, fun u v X Y p₀ q₀ h => ?_⟩
      rw [Form.countSubs] at h
      obtain ⟨p, q, hp, hpV, hq, hqV, hψ⟩ := hN u v X Y p₀ q₀ h
      exact ⟨p, q, hp, hpV, hq, hqV, hψ.neg⟩
  | .and ψ₁ ψ₂, hf, V => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      obtain ⟨N₂, hN₂⟩ := Form.exists_constOnBox ψ₂ hf.2 V
      obtain ⟨N₁, hN₁⟩ := Form.exists_constOnBox ψ₁ hf.1 N₂
      refine ⟨N₁, fun u v X Y p₀ q₀ h => ?_⟩
      rw [Form.countSubs] at h
      obtain ⟨p₁, q₁, hp₁, hpV₁, hq₁, hqV₁, h₁⟩ := hN₁ u v X Y p₀ q₀ fun χ hχ =>
        h χ (List.mem_append_left _ hχ)
      obtain ⟨p₂, q₂, hp₂, hpV₂, hq₂, hqV₂, h₂⟩ := hN₂ u v X Y p₁ q₁ fun χ hχ =>
        h χ (List.mem_append_right _ hχ)
      exact ⟨p₂, q₂, by omega, by omega, by omega, by omega,
        (h₁.mono hp₂ hpV₂ hq₂ hqV₂).and h₂⟩
  | .pnp _, hf, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- **Finitely many formulas are constant on one box** (Appendix D, proof of
`lem:cropping`, cropping once per formula). -/
theorem exists_constOnBox_list :
    ∀ S : List (Form Bool), (∀ ψ ∈ S, ψ.pnpFree = true) → ∀ V, ∃ N, ∀ u v X Y p₀ q₀,
      (∀ ψ ∈ S, ∀ χ ∈ ψ.countSubs, ConstOnMiddle χ u v X Y) →
        ∃ p q, p₀ ≤ p ∧ p + V ≤ p₀ + N ∧ q₀ ≤ q ∧ q + V ≤ q₀ + N ∧
          ∀ ψ ∈ S, ConstOnBox ψ u v X Y p q V
  | [], _, V => ⟨V, fun _ _ _ _ p₀ q₀ _ =>
      ⟨p₀, q₀, le_rfl, le_rfl, le_rfl, le_rfl, fun _ h => absurd h List.not_mem_nil⟩⟩
  | ψ :: S, hS, V => by
      obtain ⟨N, hN⟩ := exists_constOnBox_list S (fun χ hχ => hS χ (List.mem_cons_of_mem _ hχ)) V
      obtain ⟨Nψ, hNψ⟩ := Form.exists_constOnBox ψ (hS ψ (List.mem_cons_self ..)) N
      refine ⟨Nψ, fun u v X Y p₀ q₀ h => ?_⟩
      obtain ⟨p₁, q₁, hp₁, hpV₁, hq₁, hqV₁, h₁⟩ := hNψ u v X Y p₀ q₀ (h ψ (List.mem_cons_self ..))
      obtain ⟨p₂, q₂, hp₂, hpV₂, hq₂, hqV₂, h₂⟩ := hN u v X Y p₁ q₁ fun χ hχ =>
        h χ (List.mem_cons_of_mem _ hχ)
      refine ⟨p₂, q₂, by omega, by omega, by omega, by omega, fun χ hχ => ?_⟩
      rcases List.mem_cons.1 hχ with rfl | hχ
      · exact h₁.mono hp₂ hpV₂ hq₂ hqV₂
      · exact h₂ χ hχ

/-- The hypotheses of `Form.exists_constOnBox` and `exists_constOnBox_list` are
satisfiable: `▷#[Q_a] < 1` has no PNPs, and `Q_a` under its count is constant on
the middles of every frame. -/
example : (Form.lt (.countR (.sym false)) .one : Form Bool).pnpFree = true ∧
    ∀ u v X Y, ∀ χ ∈ (Form.lt (.countR (.sym false)) .one : Form Bool).countSubs,
      ConstOnMiddle χ u v X Y := by
  refine ⟨rfl, fun u v X Y χ hχ => ?_⟩
  simp only [Form.countSubs, Term.countSubs, List.append_nil, List.mem_singleton] at hχ
  exact hχ ▸ Form.constOnMiddle_of_depth_eq_zero u v X Y _ rfl rfl

end CRASP
end Transformer
