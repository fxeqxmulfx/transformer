import Transformer.AMSGrad.Section3_Example
import Transformer.AMSGrad.Section4_TheoremAReduce

/-
# AMSGrad — an Abel-based bound with a doubled second constant

Theorem A of arXiv:1904.03590v4, §1, is proved with its printed
constants in `Section4_TheoremAFinite`. This file records the weaker bound
obtained from the Abel-summation route examined in §3.

The original proof replaces `1/(1 - β_{1,t})` by `1/(1 - β₁)` in a signed
term; §3 shows that this is unjustified. `abel_general` yields the second
coefficient `D²/(1-β₁)` for every schedule, twice the printed one. The
specialized Abel estimates `theorem_A_antitone` and `theorem_A_sparse` retain
the printed coefficient under additional conditions. The scalar estimate
behind the original proof fails even on an AMSGrad run with the best
comparator (`not_abel_printed_actual_run`), while the alternative proof of
`theorem_A` uses a schedule-free projection potential and a Young bound on
`g_t - m_{t-1}`.

Source: arXiv:1904.03590v4, §1, Theorem A, and §3, Lemma 3.1.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- **A bound for every schedule, with the second constant doubled.**  For AMSGrad with
`α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`, `0 < β₂ < 1` and `γ = β₁/√β₂ < 1`, under
the standing assumptions, for `T ≥ 1` and every `x* ∈ F`,

  `R(T) ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + D²/(1-β₁) Σᵢ Σ_{t=1}^T β_{1,t}√v̂_{t,i}/α_t`
  `      + α√(1 + ln T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

The source prints `D²/(2(1-β₁))` in the second term, so this
Abel-based bound is twice as weak in that term. The printed theorem for
all schedules is `theorem_A` in `Section4_TheoremAFinite`.

Source: arXiv:1904.03590v4, §1, Theorem A (compare, do not identify). -/
theorem regret_general_schedule {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (1 - S.β₁ 1) *
          ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  have hB : 0 < 1 - S.β₁ 1 := by linarith
  have hK := fun i : Fin d => abel_general
    (fun t => Real.sqrt (S.vhat amsgradRule t i) / S.α t) S.β₁
    (fun t => (S.x amsgradRule t i - xstar i) ^ 2) (E := D ^ 2) hβ₁'
    (fun t ht => div_nonneg (Real.sqrt_nonneg _) (hα' t ht).le)
    (fun t ht => sqrt_vhat_div_mono hαt hα ht i) hβ₁
    (fun t ht => ⟨sq_nonneg _, by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS _ t) _ hxstar i) 2⟩) hT
  have h := regret_le_of_abel hS hα hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ hT hxstar hK
  have hs : ∀ i, D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / 2 ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * Real.sqrt (S.vhat amsgradRule T i) := by
    intro i
    have hX : 0 ≤ D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α := by positivity
    have e1 : D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / 2 =
        D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α / 2 := by
      rw [hαt]; simp only; rw [div_div_eq_mul_div]; ring
    have e2 : D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * Real.sqrt (S.vhat amsgradRule T i) =
        D ^ 2 * Real.sqrt T * Real.sqrt (S.vhat amsgradRule T i) / α / (1 - S.β₁ 1) := by
      generalize 1 - S.β₁ 1 = q; ring
    rw [e1, e2]
    exact div_le_div_of_nonneg_left hX hB (by linarith [(hβ₁ 1 le_rfl).1])
  have hsum : ∑ i, (D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / 2 +
        D ^ 2 / (1 - S.β₁ 1) *
          ∑ t ∈ Icc 1 T, S.β₁ t * (Real.sqrt (S.vhat amsgradRule t i) / S.α t)) ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (1 - S.β₁ 1) *
          ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    rw [sum_add_distrib, mul_sum, mul_sum]
    refine add_le_add (sum_le_sum fun i _ => hs i) (le_of_eq (sum_congr rfl fun i _ => ?_))
    congr 1
    exact sum_congr rfl fun t _ => (mul_div_assoc _ _ _).symm
  linarith

/-- The hypotheses of `regret_general_schedule` (and of `theorem_A`) are satisfiable, in the
run of Example 3.2 (`β_{1,t} = 0.9 · 0.001^{t-1}`, `β₂ = 0.999`, `α = 0.001`), at `T = 1`,
`x* = -1`. -/
example := regret_general_schedule (S := exaSetup) (F := Set.Icc (fun _ => -1) (fun _ => 1))
    (D := 2) (G := 1010) (α := 0.001) isOnlineConvex_exa (by norm_num) rfl
    (fun t _ => ⟨by show (0 : ℝ) ≤ 0.9 * 0.001 ^ (t - 1); positivity,
      by show (0.9 : ℝ) * 0.001 ^ (t - 1) ≤ 0.9 * 0.001 ^ (1 - 1)
         exact mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (by norm_num) (by norm_num)
           (Nat.zero_le _)) (by norm_num)⟩)
    (by show (0.9 : ℝ) * 0.001 ^ (1 - 1) < 1; norm_num)
    (by show (0 : ℝ) < 0.999; norm_num) (by show (0.999 : ℝ) < 1; norm_num)
    (by
      show (0.9 : ℝ) * 0.001 ^ (1 - 1) / Real.sqrt 0.999 < 1
      rw [div_lt_one (Real.sqrt_pos.2 (by norm_num))]
      have : (0.9 : ℝ) < Real.sqrt 0.999 := Real.lt_sqrt (by norm_num) |>.2 (by norm_num)
      norm_num at this ⊢; linarith)
    (le_refl 1) (xstar := fun _ => -1) ⟨fun _ => le_rfl, fun _ => by norm_num⟩

end AMSGrad
end Transformer
