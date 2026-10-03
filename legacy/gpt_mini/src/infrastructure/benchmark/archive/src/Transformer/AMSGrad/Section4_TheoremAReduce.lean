import Transformer.AMSGrad.Section4_TheoremAAbel
import Transformer.AMSGrad.Section4_Terms

/-
# AMSGrad — Theorem A: reduction of the regret to one scalar inequality per coordinate

Lemma 3.1 (`prepare_lem`) bounds the regret of AMSGrad by three sums; the second
is bounded by Lemma 4.4 (`eqsecond_le`), the first and third by an inequality
on the scalar sequences `a_t = √v̂_{t,i}/α_t`, `β_{1,t}`,
`u_t = (x_{t,i} - x*ᵢ)²`.  `regret_le_of_abel` takes that scalar bound as a
hypothesis. The full Theorem A is now proved by a different route in
`Section4_TheoremAFinite`.

Source: arXiv:1904.03590v4, §3, Lemma 3.1, and §4, proof of Theorem 4.1.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- `√v̂_{t,i}/α_t` is non-decreasing in `t` for AMSGrad with `α_t = α/√t`. -/
theorem sqrt_vhat_div_mono {S : Setup d} {α : ℝ} (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hα : 0 < α) {t : ℕ} (ht : 2 ≤ t) (i : Fin d) :
    Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1) ≤
      Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
  rw [hαt, div_div_eq_mul_div, div_div_eq_mul_div]
  have hv := vhat_le_succ S (t - 1) i
  rw [Nat.sub_add_cancel (by omega)] at hv
  gcongr
  exact Nat.sub_le t 1

/-- **Lemma 3.1 with the first and third terms bounded coordinatewise.**  For AMSGrad
with `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`, `0 < β₂ < 1` and `γ = β₁/√β₂ < 1`,
under the standing assumptions, if for every coordinate `i`

  `Σ_{t=1}^T a_t/(2(1-β_{1,t})) (u_t - u_{t+1}) + Σ_{t=2}^T β_{1,t} a_{t-1}/(2(1-β₁)) u_t ≤ Kᵢ`

with `a_t = √v̂_{t,i}/α_t`, `u_t = (x_{t,i} - x*ᵢ)²`, then

  `R(T) ≤ Σᵢ Kᵢ + α√(1 + ln T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.03590v4, §3, Lemma 3.1, and §4, (eqsecond2). -/
theorem regret_le_of_abel {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hxstar : xstar ∈ F) {K : Fin d → ℝ}
    (hK : ∀ i, ∑ t ∈ Icc 1 T, (Real.sqrt (S.vhat amsgradRule t i) / S.α t) /
            (2 * (1 - S.β₁ t)) *
          ((S.x amsgradRule t i - xstar i) ^ 2 - (S.x amsgradRule (t + 1) i - xstar i) ^ 2)
        + ∑ t ∈ Icc 2 T, S.β₁ t * (Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1)) /
            (2 * (1 - S.β₁ 1)) * (S.x amsgradRule t i - xstar i) ^ 2 ≤ K i) :
    S.regret amsgradRule xstar T ≤ ∑ i, K i +
      α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  have hp := prepare_lem hS hα' hβ₁ hβ₁' hT
    (fun _ _ _ h => m_eq_zero_of_vhat le_amsgradRule hβ₂ hβ₂' h) hxstar
  have h2 := eqsecond_le (R := amsgradRule) le_amsgradRule hα.le hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ T
  have hAC : ∑ i, ∑ t ∈ Icc 1 T, Real.sqrt (S.vhat amsgradRule t i) /
          (2 * S.α t * (1 - S.β₁ t)) *
        ((S.x amsgradRule t i - xstar i) ^ 2 - (S.x amsgradRule (t + 1) i - xstar i) ^ 2)
      + ∑ i, ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule (t - 1) i) /
          (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x amsgradRule t i - xstar i) ^ 2 ≤
      ∑ i, K i := by
    rw [← sum_add_distrib]
    refine sum_le_sum fun i _ => le_trans (le_of_eq ?_) (hK i)
    congr 1 <;> refine sum_congr rfl fun t _ => ?_
    · generalize 1 - S.β₁ t = q; ring
    · generalize 1 - S.β₁ 1 = q; ring
  have e : α * Real.sqrt (1 + Real.log T) /
        ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂)) =
      α * Real.sqrt (Real.log T + 1) /
        ((1 - S.β₁ 1) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - S.β₁ 1 / Real.sqrt S.β₂)) := by
    rw [add_comm (1 : ℝ) (Real.log T)]
    generalize 1 - S.β₁ 1 / Real.sqrt S.β₂ = r; generalize Real.sqrt (1 - S.β₂) = s
    ring
  rw [e]
  linarith

/-- The hypotheses of `regret_le_of_abel` are satisfiable: the zero cost on `[-1, 1]`,
`α = 1`, `β_{1,t} = 0`, `β₂ = 1/2`, `T = 1`, `x* = 0`, and `Kᵢ` the left-hand side of
the bound on `Kᵢ`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ S.β₁ 1 / Real.sqrt S.β₂ < 1 ∧ 1 ≤ 1 ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) ∧
      ∃ K : Fin 1 → ℝ, ∀ i, ∑ t ∈ Icc 1 1, (Real.sqrt (S.vhat amsgradRule t i) / S.α t) /
            (2 * (1 - S.β₁ t)) *
          ((S.x amsgradRule t i - (0 : Vec 1) i) ^ 2 -
            (S.x amsgradRule (t + 1) i - (0 : Vec 1) i) ^ 2)
        + ∑ t ∈ Icc 2 1, S.β₁ t * (Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1)) /
            (2 * (1 - S.β₁ 1)) * (S.x amsgradRule t i - (0 : Vec 1) i) ^ 2 ≤ K i :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    by simp [zeroSetup], le_rfl, ⟨fun _ => by norm_num, fun _ => by norm_num⟩,
    _, fun _ => le_rfl⟩

end AMSGrad
end Transformer
