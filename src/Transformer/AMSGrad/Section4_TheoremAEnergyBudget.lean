import Transformer.AMSGrad.Section4_TheoremAEnergyBounds

/-
# AMSGrad — Theorem A: EnergyBudget

Alternative proof of the printed regret bound in arXiv:1904.03590v4, §1,
Theorem A. The paper’s telescoping step is invalid for varying schedules;
these estimates use the schedule-free projection potential instead.
-/

open Finset

namespace Transformer
namespace AMSGrad
namespace TheoremA

variable {d : ℕ}

/-- The first-moment and schedule-correction energies fit within the printed third term of Theorem A for every bounded schedule. Source: arXiv:1904.03590v4, §1, Theorem A; this is a new estimate. -/
theorem energy_budget {S : Setup d} {α β₁ : ℝ}
    (hα : 0 ≤ α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁) (hβ₁' : β₁ < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    (hγ : β₁ / Real.sqrt S.β₂ < 1) (T : ℕ) (i : Fin d) :
    ∑ t ∈ Icc 1 T,
        (S.α t * S.m amsgradRule t i ^ 2 /
          (2 * Real.sqrt (S.vhat amsgradRule t i))
        + S.β₁ t * S.α t *
          (S.g amsgradRule t i ^ 2 + S.m amsgradRule (t - 1) i ^ 2) /
            Real.sqrt (S.vhat amsgradRule t i)) ≤
      α * Real.sqrt (1 + Real.log T) /
        ((1 - β₁) ^ 2 * (1 - β₁ / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂)) *
          S.gnorm amsgradRule T i := by
  set B := 1 - β₁
  set Γ := 1 - β₁ / Real.sqrt S.β₂
  set c := Real.sqrt (1 - S.β₂)
  set K := α * Real.sqrt (1 + Real.log T) / c * S.gnorm amsgradRule T i
  set C := K / (B ^ 2 * Γ)
  set H := ∑ t ∈ Icc 1 T, S.α t * S.m amsgradRule t i ^ 2 /
      Real.sqrt (S.vhat amsgradRule t i)
  set J := ∑ t ∈ Icc 1 T, S.α t * S.g amsgradRule t i ^ 2 /
      Real.sqrt (S.vhat amsgradRule t i)
  set P := ∑ t ∈ Icc 1 T, S.α t * S.m amsgradRule (t - 1) i ^ 2 /
      Real.sqrt (S.vhat amsgradRule t i)
  have hβ₁0 : 0 ≤ β₁ := (hβ₁ 1 le_rfl).1.trans (hβ₁ 1 le_rfl).2
  have hB : 0 < B := by dsimp [B]; linarith
  have hB1 : B ≤ 1 := by dsimp [B]; linarith
  have hΓ : 0 < Γ := by dsimp [Γ]; linarith
  have hΓ1 : Γ ≤ 1 := by
    dsimp [Γ]
    have hq : 0 ≤ β₁ / Real.sqrt S.β₂ := by positivity
    linarith
  have hc : 0 < c := Real.sqrt_pos.mpr (by linarith)
  have hH := moment_energy_le hα hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ T i
  have hJ := gradient_energy_le le_amsgradRule hα hαt hβ₂ hβ₂' T i
  have hP : P ≤ H := previous_moment_energy_le hα hαt hβ₂ hβ₂' T i
  have hJK : J ≤ K := by simpa only [J, K] using hJ
  have hJ0 : 0 ≤ J := by
    dsimp [J]
    apply sum_nonneg
    intro t ht
    rw [hαt]
    positivity
  have hK : 0 ≤ K := hJ0.trans hJK
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hHC : H ≤ B * C := by
    have e : α * Real.sqrt (1 + Real.log T) / (B * Γ * c) *
        S.gnorm amsgradRule T i = B * C := by
      dsimp [C, K]
      field_simp
    rw [← e]
    simpa only [H, B, Γ, c] using hH
  have hKC : K ≤ B * C := by
    have e : B * C = K / (B * Γ) := by
      dsimp [C]
      field_simp
    rw [e]
    rw [le_div_iff₀ (mul_pos hB hΓ)]
    have hBG : B * Γ ≤ 1 := by nlinarith [mul_nonneg hB.le (sub_nonneg.mpr hΓ1)]
    nlinarith [mul_le_mul_of_nonneg_left hBG hK]
  have hJC : J ≤ B * C := hJK.trans hKC
  have hBbound : ∑ t ∈ Icc 1 T, S.β₁ t * S.α t *
      (S.g amsgradRule t i ^ 2 + S.m amsgradRule (t - 1) i ^ 2) /
        Real.sqrt (S.vhat amsgradRule t i) ≤ β₁ * (J + P) := by
    have e : J + P = ∑ t ∈ Icc 1 T, S.α t *
        (S.g amsgradRule t i ^ 2 + S.m amsgradRule (t - 1) i ^ 2) /
          Real.sqrt (S.vhat amsgradRule t i) := by
      rw [← sum_add_distrib]
      refine sum_congr rfl fun t _ => ?_
      ring

    rw [e, mul_sum]
    apply sum_le_sum
    intro t ht
    have hn : 0 ≤ S.α t * (S.g amsgradRule t i ^ 2 +
        S.m amsgradRule (t - 1) i ^ 2) /
          Real.sqrt (S.vhat amsgradRule t i) := by
      rw [hαt]; positivity
    have hb := (hβ₁ t (mem_Icc.mp ht).1).2
    convert mul_le_mul_of_nonneg_right hb hn using 1; ring
  have hsum : ∑ t ∈ Icc 1 T,
      (S.α t * S.m amsgradRule t i ^ 2 / (2 * Real.sqrt (S.vhat amsgradRule t i))
        + S.β₁ t * S.α t *
          (S.g amsgradRule t i ^ 2 + S.m amsgradRule (t - 1) i ^ 2) /
            Real.sqrt (S.vhat amsgradRule t i)) ≤ H / 2 + β₁ * (J + P) := by
    rw [sum_add_distrib]
    have e : ∑ t ∈ Icc 1 T,
        S.α t * S.m amsgradRule t i ^ 2 / (2 * Real.sqrt (S.vhat amsgradRule t i)) = H / 2 := by
      rw [sum_div]
      refine sum_congr rfl fun t _ => ?_
      ring
    rw [e]
    linarith [hBbound]
  have hcoef : B / 2 + 2 * β₁ * B ≤ 1 := by
    dsimp [B]
    nlinarith [sq_nonneg (2 * β₁ - 1)]
  have hfinal : H / 2 + β₁ * (J + P) ≤ C := by
    have hJP : J + P ≤ 2 * B * C := by linarith
    have h1 := mul_le_mul_of_nonneg_left hJP hβ₁0
    have h2 := mul_le_mul_of_nonneg_right hcoef hC
    nlinarith
  calc _ ≤ H / 2 + β₁ * (J + P) := hsum
    _ ≤ C := hfinal
    _ = _ := by
      dsimp [C, K, B, Γ, c]
      simp only [div_eq_mul_inv, mul_inv, pow_two]
      ring

/-- The energy-budget assumptions are realized with `α = 1`, zero
first-moment coefficients, and `β₂ = 1/2`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    (0 : ℝ) ≤ 1 ∧ S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧
      (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ 0) ∧
      (0 : ℝ) < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧
      0 / Real.sqrt S.β₂ < 1 :=
  ⟨zero_le_one, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    one_pos, by norm_num [zeroSetup], by norm_num [zeroSetup], by simp⟩

end TheoremA
end AMSGrad
end Transformer
