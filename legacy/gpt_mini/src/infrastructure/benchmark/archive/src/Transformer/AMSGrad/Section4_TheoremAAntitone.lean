import Transformer.AMSGrad.Section3_Example
import Transformer.AMSGrad.Section4_TheoremAReduce

/-
# AMSGrad — Theorem A by Abel summation for a non-increasing schedule

This earlier proof obtains the printed constants of Theorem A of
arXiv:1904.03590v4, §1, by adding `β_{1,t}` non-increasing. That condition
makes the Abel step examined in §3 valid. The exact statement for every
bounded schedule is now proved by a different route in
`Section4_TheoremAFinite`; the extra condition here records the range of
the Abel argument.

Source: arXiv:1904.03590v4, §1, Theorem A, and §3.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- **The printed constant of Theorem A, non-increasing schedule.**  Under the hypotheses of
the global setup assumptions and `β_{1,t}` non-increasing in `t`, the bound holds with the second term
exactly as the source prints it, `D²/(2(1-β₁))`:

  `R(T) ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + D²/(2(1-β₁)) Σᵢ Σ_{t=1}^T β_{1,t}√v̂_{t,i}/α_t`
  `      + α√(1 + ln T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

This covers a constant `β_{1,t}` and the schedules `β₁λ^{t-1}` and `β₁/t` of §4.
The source assumes only `β_{1,t} ≤ β₁`; `theorem_A` now proves that case
without the added monotonicity.

Source: arXiv:1904.03590v4, §1, Theorem A; the hypothesis `hanti` is added. -/
theorem theorem_A_antitone {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hanti : ∀ t, 2 ≤ t → S.β₁ t ≤ S.β₁ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (2 * (1 - S.β₁ 1)) *
          ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  have hB : 0 < 1 - S.β₁ 1 := by linarith
  have hK := fun i : Fin d => abel_antitone
    (fun t => Real.sqrt (S.vhat amsgradRule t i) / S.α t) S.β₁
    (fun t => (S.x amsgradRule t i - xstar i) ^ 2) (E := D ^ 2) hβ₁'
    (fun t ht => div_nonneg (Real.sqrt_nonneg _) (hα' t ht).le)
    (fun t ht => sqrt_vhat_div_mono hαt hα ht i) hβ₁ hanti
    (fun t ht => ⟨sq_nonneg _, by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS _ t) _ hxstar i) 2⟩) hT
  have h := regret_le_of_abel hS hα hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ hT hxstar hK
  have hs : ∀ i, D ^ 2 / (2 * (1 - S.β₁ 1)) * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * Real.sqrt (S.vhat amsgradRule T i) := by
    intro i
    have hX : 0 ≤ D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α := by positivity
    have e1 : D ^ 2 / (2 * (1 - S.β₁ 1)) * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) =
        D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α / (2 * (1 - S.β₁ 1)) := by
      rw [hαt]; simp only; rw [div_div_eq_mul_div]
      generalize 1 - S.β₁ 1 = q; ring
    have e2 : D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * Real.sqrt (S.vhat amsgradRule T i) =
        D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α / (1 - S.β₁ 1) := by
      generalize 1 - S.β₁ 1 = q; ring
    rw [e1, e2]
    exact div_le_div_of_nonneg_left hX hB (by linarith)
  have hsh : ∀ i, ∑ t ∈ Icc 2 T, S.β₁ t * (Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1)) ≤
      ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    intro i
    refine (sum_le_sum fun t ht => ?_).trans (sum_le_sum_of_subset_of_nonneg
      (Icc_subset_Icc_left one_le_two) fun t ht _ => ?_)
    · rw [mul_div_assoc]
      exact mul_le_mul_of_nonneg_left (sqrt_vhat_div_mono hαt hα (mem_Icc.1 ht).1 i)
        (hβ₁ t (by have := (mem_Icc.1 ht).1; omega)).1
    · exact div_nonneg (mul_nonneg (hβ₁ t (mem_Icc.1 ht).1).1 (Real.sqrt_nonneg _))
        (hα' t (mem_Icc.1 ht).1).le
  have hsum : ∑ i, D ^ 2 / (2 * (1 - S.β₁ 1)) * (Real.sqrt (S.vhat amsgradRule T i) / S.α T +
        ∑ t ∈ Icc 2 T, S.β₁ t * (Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1))) ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (2 * (1 - S.β₁ 1)) *
          ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    have hc : 0 ≤ D ^ 2 / (2 * (1 - S.β₁ 1)) := by positivity
    simp only [mul_add, sum_add_distrib]
    rw [mul_sum, mul_sum]
    exact add_le_add (sum_le_sum fun i _ => hs i)
      (sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hsh i) hc)
  linarith

/-- The hypotheses of `theorem_A_antitone` are satisfiable, in the run of Example 3.2, whose
schedule `β_{1,t} = 0.9 · 0.001^{t-1}` is non-increasing, at `T = 1`, `x* = -1`. -/
example := theorem_A_antitone (S := exaSetup) (F := Set.Icc (fun _ => -1) (fun _ => 1))
    (D := 2) (G := 1010) (α := 0.001) isOnlineConvex_exa (by norm_num) rfl
    (fun t _ => ⟨by show (0 : ℝ) ≤ 0.9 * 0.001 ^ (t - 1); positivity,
      by show (0.9 : ℝ) * 0.001 ^ (t - 1) ≤ 0.9 * 0.001 ^ (1 - 1)
         exact mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (by norm_num) (by norm_num)
           (Nat.zero_le _)) (by norm_num)⟩)
    (by show (0.9 : ℝ) * 0.001 ^ (1 - 1) < 1; norm_num)
    (fun t _ => by
      show (0.9 : ℝ) * 0.001 ^ (t - 1) ≤ 0.9 * 0.001 ^ (t - 1 - 1)
      exact mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (by norm_num) (by norm_num)
        (Nat.sub_le _ _)) (by norm_num))
    (by show (0 : ℝ) < 0.999; norm_num) (by show (0.999 : ℝ) < 1; norm_num)
    (by
      show (0.9 : ℝ) * 0.001 ^ (1 - 1) / Real.sqrt 0.999 < 1
      rw [div_lt_one (Real.sqrt_pos.2 (by norm_num))]
      have : (0.9 : ℝ) < Real.sqrt 0.999 := Real.lt_sqrt (by norm_num) |>.2 (by norm_num)
      norm_num at this ⊢; linarith)
    (le_refl 1) (xstar := fun _ => -1) ⟨fun _ => le_rfl, fun _ => by norm_num⟩

end AMSGrad
end Transformer
