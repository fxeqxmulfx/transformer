import Transformer.AMSGrad.Section4_TheoremAEnergyBudget

/-
# AMSGrad — Theorem A: Potential

Alternative proof of the printed regret bound in arXiv:1904.03590v4, §1,
Theorem A. The paper’s telescoping step is invalid for varying schedules;
these estimates use the schedule-free projection potential instead.
-/

open Finset

namespace Transformer
namespace AMSGrad
namespace TheoremA

variable {d : ℕ}

/-- The schedule-free potential from projection fits within the printed first term of Theorem A. Source: arXiv:1904.03590v4, §1, Theorem A, and §3. -/
theorem potential_budget {S : Setup d} {F : Set (Vec d)} {D G α β₁ : ℝ}
    (hS : IsOnlineConvex S F D G) (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁0 : 0 ≤ β₁) (hβ₁' : β₁ < 1) {T : ℕ} (hT : 1 ≤ T)
    {xstar : Vec d} (hx : xstar ∈ F) (i : Fin d) :
    ∑ t ∈ Icc 1 T, Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t) *
      ((S.x amsgradRule t i - xstar i) ^ 2 -
        (S.x amsgradRule (t + 1) i - xstar i) ^ 2) ≤
      D ^ 2 * Real.sqrt T / (α * (1 - β₁)) *
        Real.sqrt (S.vhat amsgradRule T i) := by
  set a : ℕ → ℝ := fun t => Real.sqrt (S.vhat amsgradRule t i) / S.α t
  set u : ℕ → ℝ := fun t => (S.x amsgradRule t i - xstar i) ^ 2
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]
    exact div_pos hα (Real.sqrt_pos.mpr (by exact_mod_cast ht))
  have ha : ∀ t, 1 ≤ t → 0 ≤ a t := fun t ht =>
    div_nonneg (Real.sqrt_nonneg _) (hα' t ht).le
  have hmono : ∀ t, 2 ≤ t → a (t - 1) ≤ a t :=
    fun t ht => sqrt_vhat_div_mono hαt hα ht i
  have hu : ∀ t, 1 ≤ t → 0 ≤ u t ∧ u t ≤ D ^ 2 := by
    intro t ht
    refine ⟨sq_nonneg _, ?_⟩
    calc u t = |S.x amsgradRule t i - xstar i| ^ 2 := by simp [u, sq_abs]
      _ ≤ D ^ 2 := pow_le_pow_left₀ (abs_nonneg _)
        (hS.diam _ (x_mem hS _ t) _ hx i) 2
  have hpot := sum_mul_sub_le a u ha hmono hu hT
  have hterm : 0 ≤ a T * u (T + 1) :=
    mul_nonneg (ha T hT) (hu (T + 1) (by omega)).1
  have hhalf : ∑ t ∈ Icc 1 T, a t / 2 * (u t - u (t + 1)) ≤
      D ^ 2 * a T / 2 := by
    have e : ∑ t ∈ Icc 1 T, a t / 2 * (u t - u (t + 1)) =
        (∑ t ∈ Icc 1 T, a t * (u t - u (t + 1))) / 2 := by
      rw [sum_div]
      refine sum_congr rfl fun t _ => ?_
      ring
    rw [e]
    linarith
  have hcoeff : D ^ 2 * a T / 2 ≤
      D ^ 2 * Real.sqrt T / (α * (1 - β₁)) *
        Real.sqrt (S.vhat amsgradRule T i) := by
    have hX : 0 ≤ D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α := by
      positivity
    have e1 : D ^ 2 * a T / 2 =
        D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α / 2 := by
      dsimp [a]
      rw [hαt]
      simp only [div_div_eq_mul_div]
      ring
    have e2 : D ^ 2 * Real.sqrt T / (α * (1 - β₁)) *
        Real.sqrt (S.vhat amsgradRule T i) =
        D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α / (1 - β₁) := by
      simp only [div_eq_mul_inv, mul_inv]
      ring
    rw [e1, e2]
    exact div_le_div_of_nonneg_left hX (by linarith : 0 < 1 - β₁) (by linarith)
  calc _ = ∑ t ∈ Icc 1 T, a t / 2 * (u t - u (t + 1)) := by
        refine sum_congr rfl fun t _ => ?_
        dsimp [a, u]
        ring
    _ ≤ D ^ 2 * a T / 2 := hhalf
    _ ≤ _ := hcoeff

/-- The Young correction’s diameter term fits within the printed second term of Theorem A. Source: arXiv:1904.03590v4, §1, Theorem A. -/
theorem schedule_budget {S : Setup d} {D α β₁ : ℝ}
    (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁0 : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hb : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t) (T : ℕ) (i : Fin d) :
    ∑ t ∈ Icc 1 T, S.β₁ t *
        (Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t) * D ^ 2) ≤
      D ^ 2 / (2 * (1 - β₁)) *
        ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]
    exact div_pos hα (Real.sqrt_pos.mpr (by exact_mod_cast ht))
  have hsum : 0 ≤ ∑ t ∈ Icc 1 T,
      S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    apply sum_nonneg
    intro t ht
    have hb0 : 0 ≤ S.β₁ t := hb t (mem_Icc.mp ht).1
    have hat : 0 < S.α t := hα' t (mem_Icc.mp ht).1
    positivity
  have hcoef : D ^ 2 / 2 ≤ D ^ 2 / (2 * (1 - β₁)) :=
    div_le_div_of_nonneg_left (sq_nonneg D) (by linarith) (by linarith)
  have e : ∑ t ∈ Icc 1 T, S.β₁ t *
      (Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t) * D ^ 2) =
      D ^ 2 / 2 * ∑ t ∈ Icc 1 T,
        S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    rw [mul_sum]
    refine sum_congr rfl fun t _ => ?_
    ring
  rw [e]
  exact mul_le_mul_of_nonneg_right hcoef hsum

/-- The potential and schedule-budget hypotheses are satisfiable on the
zero-loss box run with `α = 1`, `β₁,ₜ = 0`, and `T = 1`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧
      (0 : ℝ) < 1 ∧ S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧
      (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (∀ t, 1 ≤ t → 0 ≤ S.β₁ t) ∧ 1 ≤ 1 ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, le_rfl, one_pos,
    fun _ _ => le_rfl, le_rfl, fun _ => by norm_num, fun _ => by norm_num⟩

end TheoremA
end AMSGrad
end Transformer
