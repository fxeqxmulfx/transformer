/-
# Shrinking a strip until a formula is constant on it

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.4 and Appendix C.2: `lem:cropping_oneway`, repaired.

The paper crops an interval of Parikh vectors until a formula whose counted
formulas are constant becomes constant itself.  With the positions before the
interval fixed to a prefix `u` (`Transformer.CRASP.Strip`), that works.  A
comparison of two counts is a comparison of two affine functions of the
numbers `x` and `y` of `a`s and `b`s (`Term.val_affine`), and on a strip of
width `V` inside a strip of width `2V` it is constant once `y` is large enough
(`exists_strip_lt`): the coefficients of `y` decide it when they differ, and
otherwise the comparison is monotone in `x`, so it is constant on the lower or
the upper half.  Boolean combinations shrink the strip once per operand, so a
formula, and a list of formulas, is constant on a strip whose width depends on
the formulas alone (`Form.exists_constOnStrip`, `exists_constOnStrip_list`).
-/

import Transformer.CRASP.Affine

namespace Transformer
namespace CRASP

/-- **A comparison of two affine functions is constant on a strip.**  Inside
`p₀ ≤ x ≤ p₀ + 2V`, `q₀ ≤ y` there is a strip `p ≤ x ≤ p + V`, `q ≤ y` on which
`α₁ + β₁·x + γ₁·y < α₂ + β₂·x + γ₂·y` does not change (Appendix C.2, proof of
`lem:cropping_oneway`: a half-plane crossing the interval is cropped away). -/
theorem exists_strip_lt (α₁ β₁ γ₁ α₂ β₂ γ₂ V p₀ q₀ : ℕ) :
    ∃ p q, p₀ ≤ p ∧ p + V ≤ p₀ + 2 * V ∧ q₀ ≤ q ∧ ∃ b : Bool, ∀ x y, p ≤ x → x ≤ p + V →
      q ≤ y → decide (α₁ + β₁ * x + γ₁ * y < α₂ + β₂ * x + γ₂ * y) = b := by
  rcases lt_trichotomy γ₁ γ₂ with h | rfl | h
  · refine ⟨p₀, max q₀ (α₁ + β₁ * (p₀ + V) + 1), le_rfl, by omega, le_max_left _ _, true,
      fun x y _ hx hy => ?_⟩
    have h₁ : β₁ * x ≤ β₁ * (p₀ + V) := Nat.mul_le_mul_left _ hx
    have h₂ : (γ₁ + 1) * y ≤ γ₂ * y := Nat.mul_le_mul_right _ h
    have h₃ := le_max_right q₀ (α₁ + β₁ * (p₀ + V) + 1)
    rw [Nat.add_mul, Nat.one_mul] at h₂
    rw [decide_eq_true_iff]
    omega
  · rcases le_total β₁ β₂ with hβ | hβ <;>
      by_cases hP : α₁ + β₁ * (p₀ + V) < α₂ + β₂ * (p₀ + V)
    · refine ⟨p₀ + V, q₀, by omega, by omega, le_rfl, true, fun x y hx _ _ => ?_⟩
      obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le hx
      have := Nat.mul_le_mul_right e hβ
      have h₁ : β₁ * (p₀ + V + e) = β₁ * (p₀ + V) + β₁ * e := Nat.mul_add ..
      have h₂ : β₂ * (p₀ + V + e) = β₂ * (p₀ + V) + β₂ * e := Nat.mul_add ..
      rw [decide_eq_true_iff]
      omega
    · refine ⟨p₀, q₀, le_rfl, by omega, le_rfl, false, fun x y _ hx _ => ?_⟩
      obtain ⟨e, he⟩ := Nat.exists_eq_add_of_le hx
      have := Nat.mul_le_mul_right e hβ
      have h₁ : β₁ * (p₀ + V) = β₁ * x + β₁ * e := by rw [he, Nat.mul_add]
      have h₂ : β₂ * (p₀ + V) = β₂ * x + β₂ * e := by rw [he, Nat.mul_add]
      rw [decide_eq_false_iff_not]
      omega
    · refine ⟨p₀, q₀, le_rfl, by omega, le_rfl, true, fun x y _ hx _ => ?_⟩
      obtain ⟨e, he⟩ := Nat.exists_eq_add_of_le hx
      have := Nat.mul_le_mul_right e hβ
      have h₁ : β₁ * (p₀ + V) = β₁ * x + β₁ * e := by rw [he, Nat.mul_add]
      have h₂ : β₂ * (p₀ + V) = β₂ * x + β₂ * e := by rw [he, Nat.mul_add]
      rw [decide_eq_true_iff]
      omega
    · refine ⟨p₀ + V, q₀, by omega, by omega, le_rfl, false, fun x y hx _ _ => ?_⟩
      obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le hx
      have := Nat.mul_le_mul_right e hβ
      have h₁ : β₁ * (p₀ + V + e) = β₁ * (p₀ + V) + β₁ * e := Nat.mul_add ..
      have h₂ : β₂ * (p₀ + V + e) = β₂ * (p₀ + V) + β₂ * e := Nat.mul_add ..
      rw [decide_eq_false_iff_not]
      omega
  · refine ⟨p₀, max q₀ (α₂ + β₂ * (p₀ + V) + 1), le_rfl, by omega, le_max_left _ _, false,
      fun x y _ hx hy => ?_⟩
    have h₁ : β₂ * x ≤ β₂ * (p₀ + V) := Nat.mul_le_mul_left _ hx
    have h₂ : (γ₂ + 1) * y ≤ γ₁ * y := Nat.mul_le_mul_right _ h
    have h₃ := le_max_right q₀ (α₂ + β₂ * (p₀ + V) + 1)
    rw [Nat.add_mul, Nat.one_mul] at h₂
    rw [decide_eq_false_iff_not]
    omega

/-- **A formula whose counted formulas are constant is constant on a strip.**
For every width `V` there is a width `N` such that, after any prefix `u` on
whose quadrant `x ≤ W` the formulas under the counts of `ψ` are constant, every
strip of width `N` below `W` holds a strip of width `V` on which `ψ` is
constant, as far up in `y` as wanted (Appendix C.2, `lem:cropping_oneway`, with
the positions before the interval fixed to `u`; `N` depends on `ψ` alone). -/
theorem Form.exists_constOnStrip :
    ∀ ψ : Form Bool, ψ.past = true → ψ.pnpFree = true → ∀ V, ∃ N, ∀ u W p₀ q₀, p₀ + N ≤ W →
      (∀ χ ∈ ψ.countSubs, ConstOnStrip χ u 0 W 0) →
        ∃ p q, p₀ ≤ p ∧ p + V ≤ p₀ + N ∧ q₀ ≤ q ∧ ConstOnStrip ψ u p V q
  | .sym _, _, _, V => ⟨V, fun u _ p₀ q₀ _ _ =>
      ⟨p₀, q₀, le_rfl, le_rfl, le_rfl, Form.constOnStrip_of_depth_eq_zero u p₀ V q₀ _ rfl rfl⟩⟩
  | .lt t₁ t₂, hp, hf, V => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      refine ⟨2 * V, fun u W p₀ q₀ hW h => ?_⟩
      rw [Form.countSubs] at h
      obtain ⟨α₁, β₁, γ₁, h₁⟩ := Term.val_affine u W t₁ hp.1 hf.1 fun χ hχ =>
        h χ (List.mem_append_left _ hχ)
      obtain ⟨α₂, β₂, γ₂, h₂⟩ := Term.val_affine u W t₂ hp.2 hf.2 fun χ hχ =>
        h χ (List.mem_append_right _ hχ)
      obtain ⟨p, q, hp₀, hpV, hq₀, b, hb⟩ := exists_strip_lt α₁ β₁ γ₁ α₂ β₂ γ₂ V p₀ q₀
      refine ⟨p, q, hp₀, hpV, hq₀, fun _ => b, fun m hm i hi hx hy => ?_⟩
      rw [Form.sat, h₁ m (by omega) i hi, h₂ m (by omega) i hi]
      exact hb _ _ hx (((List.take_sublist _ m).count_le false).trans hm) hy
  | .neg ψ, hp, hf, V => by
      rw [Form.past] at hp
      rw [Form.pnpFree] at hf
      obtain ⟨N, hN⟩ := Form.exists_constOnStrip ψ hp hf V
      refine ⟨N, fun u W p₀ q₀ hW h => ?_⟩
      rw [Form.countSubs] at h
      obtain ⟨p, q, hp₀, hpV, hq₀, hψ⟩ := hN u W p₀ q₀ hW h
      exact ⟨p, q, hp₀, hpV, hq₀, hψ.neg⟩
  | .and ψ₁ ψ₂, hp, hf, V => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      obtain ⟨N₂, hN₂⟩ := Form.exists_constOnStrip ψ₂ hp.2 hf.2 V
      obtain ⟨N₁, hN₁⟩ := Form.exists_constOnStrip ψ₁ hp.1 hf.1 N₂
      refine ⟨N₁, fun u W p₀ q₀ hW h => ?_⟩
      rw [Form.countSubs] at h
      obtain ⟨p₁, q₁, hp₁, hpV₁, hq₁, h₁⟩ := hN₁ u W p₀ q₀ hW fun χ hχ =>
        h χ (List.mem_append_left _ hχ)
      obtain ⟨p₂, q₂, hp₂, hpV₂, hq₂, h₂⟩ := hN₂ u W p₁ q₁ (by omega) fun χ hχ =>
        h χ (List.mem_append_right _ hχ)
      exact ⟨p₂, q₂, by omega, by omega, by omega, (h₁.mono hp₂ hpV₂ hq₂).and h₂⟩
  | .pnp _, _, hf, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- **Finitely many formulas are constant on one strip** (Appendix C.2, proof of
`lem:cropping_oneway`, cropping once per formula). -/
theorem exists_constOnStrip_list :
    ∀ S : List (Form Bool), (∀ ψ ∈ S, ψ.past = true ∧ ψ.pnpFree = true) → ∀ V, ∃ N,
      ∀ u W p₀ q₀, p₀ + N ≤ W → (∀ ψ ∈ S, ∀ χ ∈ ψ.countSubs, ConstOnStrip χ u 0 W 0) →
        ∃ p q, p₀ ≤ p ∧ p + V ≤ p₀ + N ∧ q₀ ≤ q ∧ ∀ ψ ∈ S, ConstOnStrip ψ u p V q
  | [], _, V => ⟨V, fun _ _ p₀ q₀ _ _ =>
      ⟨p₀, q₀, le_rfl, le_rfl, le_rfl, fun _ h => absurd h List.not_mem_nil⟩⟩
  | ψ :: S, hS, V => by
      obtain ⟨N, hN⟩ := exists_constOnStrip_list S (fun χ hχ => hS χ (List.mem_cons_of_mem _ hχ)) V
      obtain ⟨Nψ, hNψ⟩ := Form.exists_constOnStrip ψ (hS ψ (List.mem_cons_self ..)).1
        (hS ψ (List.mem_cons_self ..)).2 N
      refine ⟨Nψ, fun u W p₀ q₀ hW h => ?_⟩
      obtain ⟨p₁, q₁, hp₁, hpV₁, hq₁, h₁⟩ := hNψ u W p₀ q₀ hW (h ψ (List.mem_cons_self ..))
      obtain ⟨p₂, q₂, hp₂, hpV₂, hq₂, h₂⟩ := hN u W p₁ q₁ (by omega) fun χ hχ =>
        h χ (List.mem_cons_of_mem _ hχ)
      refine ⟨p₂, q₂, by omega, by omega, by omega, fun χ hχ => ?_⟩
      rcases List.mem_cons.1 hχ with rfl | hχ
      · exact h₁.mono hp₂ hpV₂ hq₂
      · exact h₂ χ hχ

/-- The hypotheses of `Form.exists_constOnStrip` and `exists_constOnStrip_list`
are satisfiable: `◁#[Q_a] < 1` is past-only without PNPs, and `Q_a` under its
count is constant on every quadrant after `ε`. -/
example : (Form.lt (.countL (.sym false)) .one : Form Bool).past = true ∧
    (Form.lt (.countL (.sym false)) .one : Form Bool).pnpFree = true ∧
      ∀ W, ∀ χ ∈ (Form.lt (.countL (.sym false)) .one : Form Bool).countSubs,
        ConstOnStrip χ [] 0 W 0 := by
  refine ⟨rfl, rfl, fun W χ hχ => ?_⟩
  simp only [Form.countSubs, Term.countSubs, List.append_nil, List.mem_singleton] at hχ
  exact hχ ▸ Form.constOnStrip_of_depth_eq_zero [] 0 W 0 _ rfl rfl

end CRASP
end Transformer
