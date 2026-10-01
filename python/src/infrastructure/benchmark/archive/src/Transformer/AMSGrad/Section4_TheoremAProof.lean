import Transformer.AMSGrad.Section4_TheoremAPotential

/-
# AMSGrad — the global-schedule core of Theorem A

The printed bound of Theorem A of arXiv:1904.03590v4, §1 (Theorem 4 of
Reddi et al.) is proved for schedules bounded at every step. The exact
finite-horizon statement, which requires assumptions on losses, steps, and
the schedule only through `T`, is `theorem_A` in `Section4_TheoremAFinite`.
The proof does not use the invalid telescoping step identified in §3. It
uses the schedule-free projection potential and pays for the difference
between the gradient and first moment from the printed schedule and gradient
terms. The original paper's §3 critique of the old proof remains valid.
-/

open Finset

namespace Transformer
namespace AMSGrad

open TheoremA

variable {d : ℕ}

/-- **Theorem A, global-schedule core.** For AMSGrad with
`α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`, `0 < β₂ < 1` and
`γ = β₁/√β₂ < 1`, under the standing online convex assumptions, the
printed three-term regret bound holds for every feasible comparator.

The source requires loss, step, and schedule assumptions only through `T`;
this core assumes them at every step. `theorem_A` removes the extra
requirements by continuing the data after `T`. The paper's §3
counterexample invalidates a step of the original proof; this result uses
a different argument.

Source: arXiv:1904.03590v4, §1, Theorem A and §3, Lemma 3.1. -/
theorem theorem_A_global {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
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
    rw [hαt]
    exact div_pos hα (Real.sqrt_pos.mpr (by exact_mod_cast ht))
  set P : ℕ → Fin d → ℝ := fun t i =>
    Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t) *
      ((S.x amsgradRule t i - xstar i) ^ 2 -
        (S.x amsgradRule (t + 1) i - xstar i) ^ 2)
  set M : ℕ → Fin d → ℝ := fun t i =>
    S.α t * S.m amsgradRule t i ^ 2 / (2 * Real.sqrt (S.vhat amsgradRule t i))
  set Q : ℕ → Fin d → ℝ := fun t i =>
    S.β₁ t * (Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t) * D ^ 2)
  set E : ℕ → Fin d → ℝ := fun t i =>
    S.β₁ t * S.α t *
      (S.g amsgradRule t i ^ 2 + S.m amsgradRule (t - 1) i ^ 2) /
        Real.sqrt (S.vhat amsgradRule t i)
  have hstep : ∀ t ∈ Icc 1 T, S.f t (S.x amsgradRule t) - S.f t xstar ≤
      ∑ i, (P t i + M t i + Q t i + E t i) := by
    intro t ht
    have ht1 := (mem_Icc.mp ht).1
    have hc := convex_first_order (hS.convexOn t) (hS.differentiable t)
      (S.x amsgradRule t) xstar
    change S.f t (S.x amsgradRule t) +
      ∑ i, S.g amsgradRule t i * (xstar i - S.x amsgradRule t i) ≤
        S.f t xstar at hc
    have hneg : ∑ i, S.g amsgradRule t i * (S.x amsgradRule t i - xstar i) =
        -∑ i, S.g amsgradRule t i * (xstar i - S.x amsgradRule t i) := by
      rw [← sum_neg_distrib]
      exact sum_congr rfl fun i _ => by ring
    have hconv : S.f t (S.x amsgradRule t) - S.f t xstar ≤
        ∑ i, S.g amsgradRule t i * (S.x amsgradRule t i - xstar i) := by
      linarith
    have hm := moment_step_ineq hS ht1 (hα' t ht1)
      (fun i h => m_eq_zero_of_vhat le_amsgradRule hβ₂ hβ₂' h) hxstar
    change (∑ i, S.m amsgradRule t i * (S.x amsgradRule t i - xstar i)) ≤
      ∑ i, (P t i + M t i) at hm
    have hcorr : ∑ i, S.g amsgradRule t i * (S.x amsgradRule t i - xstar i) ≤
        ∑ i, (S.m amsgradRule t i * (S.x amsgradRule t i - xstar i) +
          Q t i + E t i) := by
      apply sum_le_sum
      intro i _
      calc S.g amsgradRule t i * (S.x amsgradRule t i - xstar i)
          ≤ S.m amsgradRule t i * (S.x amsgradRule t i - xstar i) +
            S.β₁ t * (Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t) * D ^ 2 +
              S.α t * (S.g amsgradRule t i ^ 2 +
                S.m amsgradRule (t - 1) i ^ 2) /
                  Real.sqrt (S.vhat amsgradRule t i)) :=
            correction_step_ineq hS hβ₂ hβ₂' ht1 (hα' t ht1)
              (hβ₁ t ht1).1 hxstar i
        _ = _ := by dsimp [Q, E]; ring
    calc S.f t (S.x amsgradRule t) - S.f t xstar
        ≤ ∑ i, S.g amsgradRule t i * (S.x amsgradRule t i - xstar i) := hconv
      _ ≤ ∑ i, (S.m amsgradRule t i * (S.x amsgradRule t i - xstar i) +
          Q t i + E t i) := hcorr
      _ ≤ ∑ i, (P t i + M t i + Q t i + E t i) := by
        simp only [sum_add_distrib] at hm ⊢
        linarith
  have hregret : S.regret amsgradRule xstar T ≤
      ∑ i, ∑ t ∈ Icc 1 T, (P t i + M t i + Q t i + E t i) := by
    unfold Setup.regret
    calc _ ≤ ∑ t ∈ Icc 1 T, ∑ i, (P t i + M t i + Q t i + E t i) :=
          sum_le_sum hstep
      _ = _ := sum_comm
  have hcoord : ∀ i, ∑ t ∈ Icc 1 T, (P t i + M t i + Q t i + E t i) ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) *
        Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (2 * (1 - S.β₁ 1)) *
          ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) *
            Real.sqrt (1 - S.β₂)) * S.gnorm amsgradRule T i := by
    intro i
    have hp := potential_budget hS hα hαt (hβ₁ 1 le_rfl).1 hβ₁' hT hxstar i
    have hq := schedule_budget (D := D) hα hαt (hβ₁ 1 le_rfl).1 hβ₁'
      (fun t ht => (hβ₁ t ht).1) T i
    have he := energy_budget hα.le hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ T i
    change (∑ t ∈ Icc 1 T, P t i) ≤ _ at hp
    change (∑ t ∈ Icc 1 T, Q t i) ≤ _ at hq
    change (∑ t ∈ Icc 1 T, (M t i + E t i)) ≤ _ at he
    have e : ∑ t ∈ Icc 1 T, (P t i + M t i + Q t i + E t i) =
        (∑ t ∈ Icc 1 T, P t i) + (∑ t ∈ Icc 1 T, Q t i) +
          ∑ t ∈ Icc 1 T, (M t i + E t i) := by
      simp only [sum_add_distrib]
      ring
    rw [e]
    linarith
  have hsum := hregret.trans (sum_le_sum (fun i _ => hcoord i))
  calc S.regret amsgradRule xstar T
      ≤ ∑ i, (D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) *
          Real.sqrt (S.vhat amsgradRule T i)
        + D ^ 2 / (2 * (1 - S.β₁ 1)) *
            ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
        + α * Real.sqrt (1 + Real.log T) /
            ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) *
              Real.sqrt (1 - S.β₂)) * S.gnorm amsgradRule T i) := hsum
    _ = _ := by simp only [sum_add_distrib, ← mul_sum]

/-- The hypotheses of `theorem_A_global` are satisfiable: the zero cost on `[-1, 1]`,
`α = 1`, `β_{1,t} = 0`, `β₂ = 1/2`, `T = 1`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ S.β₁ 1 / Real.sqrt S.β₂ < 1 ∧ 1 ≤ 1 ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    by simp [zeroSetup], le_rfl, fun _ => by norm_num, fun _ => by norm_num⟩

end AMSGrad
end Transformer
