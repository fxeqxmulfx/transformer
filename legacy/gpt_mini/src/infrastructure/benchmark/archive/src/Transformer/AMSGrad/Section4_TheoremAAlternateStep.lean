import Transformer.AMSGrad.Section4_TheoremAReduce

/-
# AMSGrad — Theorem A: AlternateStep

Alternative proof of the printed regret bound in arXiv:1904.03590v4, §1,
Theorem A. The paper’s telescoping step is invalid for varying schedules;
these estimates use the schedule-free projection potential instead.
-/

open Finset

namespace Transformer
namespace AMSGrad
namespace TheoremA

variable {d : ℕ}

/-- For Algorithm 1, a zero accumulated second moment at a positive step forces that coordinate of the current gradient to vanish. Source: arXiv:1904.03590v4, Algorithm 1 and §3, Lemma 3.1. -/
theorem grad_eq_zero_of_vhat {S : Setup d} {R : Rule d}
    (hR : ∀ t a b, b ≤ R t a b) (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    {t : ℕ} (ht : 1 ≤ t) {i : Fin d}
    (h : Real.sqrt (S.vhat R t i) = 0) : S.g R t i = 0 := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  have hv0 : 0 ≤ S.v R n i := v_nonneg hβ₂.le hβ₂'.le R n i
  have hv1 : S.v R (n + 1) i ≤ S.vhat R (n + 1) i :=
    hR (n + 1) (S.vhat R n) (S.v R (n + 1)) i
  have hv2 : S.vhat R (n + 1) i ≤ 0 := Real.sqrt_eq_zero'.mp h
  have hr : S.v R (n + 1) i = S.β₂ * S.v R n i + (1 - S.β₂) * S.g R (n + 1) i ^ 2 := rfl
  have hsq : S.g R (n + 1) i ^ 2 = 0 := by
    have hpos : 0 ≤ S.β₂ * S.v R n i := mul_nonneg hβ₂.le hv0
    have hpos' : 0 ≤ (1 - S.β₂) * S.g R (n + 1) i ^ 2 :=
      mul_nonneg (by linarith) (sq_nonneg _)
    nlinarith
  nlinarith

/-- Projection gives a one-step potential bound for the first moment, with no factor depending on the first-moment schedule. Source: arXiv:1904.03590v4, §3, proof of Lemma 3.1. -/
theorem moment_step_ineq {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {R : Rule d} {t : ℕ} (ht : 1 ≤ t)
    (hα : 0 < S.α t)
    (hz : ∀ i, Real.sqrt (S.vhat R t i) = 0 → S.m R t i = 0)
    {xstar : Vec d} (hx : xstar ∈ F) :
    ∑ i, S.m R t i * (S.x R t i - xstar i) ≤
      ∑ i, (Real.sqrt (S.vhat R t i) / (2 * S.α t) *
          ((S.x R t i - xstar i) ^ 2 - (S.x R (t + 1) i - xstar i) ^ 2)
        + S.α t * S.m R t i ^ 2 / (2 * Real.sqrt (S.vhat R t i))) := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  set a := S.α (n + 1)
  set w : Vec d := vsqrt (S.vhat R (n + 1))
  set y : Vec d := S.x R (n + 1) - a • (S.m R (n + 1) / w)
  have hx1 : S.x R (n + 1 + 1) = S.proj w y := rfl
  have hp := proj_le hS.proj hS.convex (w := w) (fun i => Real.sqrt_nonneg _) y hx
  rw [← hx1] at hp
  set P : Fin d → ℝ := fun i => S.m R (n + 1) i * (S.x R (n + 1) i - xstar i)
  set Q : Fin d → ℝ := fun i => w i / (2 * a) *
      ((S.x R (n + 1) i - xstar i) ^ 2 - (S.x R (n + 1 + 1) i - xstar i) ^ 2)
    + a * S.m R (n + 1) i ^ 2 / (2 * w i)
  have key : ∀ i, 2 * a * P i + w i * (y i - xstar i) ^ 2 =
      2 * a * Q i + w i * (S.x R (n + 1 + 1) i - xstar i) ^ 2 := by
    intro i
    have hy : y i = S.x R (n + 1) i - a * (S.m R (n + 1) i / w i) := rfl
    simp only [P, Q]
    rw [hy]
    by_cases hw : w i = 0
    · have hm0 := hz i hw
      rw [hw, hm0]
      simp
    · field_simp
      ring
  have hsum := sum_congr rfl fun i (_ : i ∈ univ) => key i
  rw [sum_add_distrib, sum_add_distrib, ← mul_sum, ← mul_sum] at hsum
  show ∑ i, P i ≤ ∑ i, Q i
  refine le_of_mul_le_mul_left ?_ (by positivity : 0 < 2 * a)
  linarith

/-- The identity gₜ = mₜ + β₁,ₜ(gₜ − mₜ₋₁), followed by Young’s inequality, bounds the schedule correction. This is an alternative argument for arXiv:1904.03590v4, §1, Theorem A. -/
theorem correction_step_ineq {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    {t : ℕ} (ht : 1 ≤ t) (hα : 0 < S.α t) (hb : 0 ≤ S.β₁ t)
    {xstar : Vec d} (hx : xstar ∈ F) (i : Fin d) :
    S.g amsgradRule t i * (S.x amsgradRule t i - xstar i) ≤
      S.m amsgradRule t i * (S.x amsgradRule t i - xstar i)
      + S.β₁ t * (Real.sqrt (S.vhat amsgradRule t i) / (2 * S.α t) * D ^ 2
        + S.α t * (S.g amsgradRule t i ^ 2 + S.m amsgradRule (t - 1) i ^ 2) /
          Real.sqrt (S.vhat amsgradRule t i)) := by
  set w := Real.sqrt (S.vhat amsgradRule t i)
  set m := S.m amsgradRule t i
  set p := S.m amsgradRule (t - 1) i
  set g := S.g amsgradRule t i
  set q := S.x amsgradRule t i - xstar i
  have hw : 0 ≤ w := Real.sqrt_nonneg _
  have hq : q ^ 2 ≤ D ^ 2 := by
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS _ t) _ hx i) 2
  have hzero : w = 0 → g - p = 0 := by
    intro h
    have hg : g = 0 := grad_eq_zero_of_vhat le_amsgradRule hβ₂ hβ₂' ht h
    have hp : p = 0 := by
      obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
      have hprev : Real.sqrt (S.vhat amsgradRule n i) = 0 := by
        have hmono := vhat_le_succ S n i
        have hnonneg := vhat_nonneg S n i
        have hcur : S.vhat amsgradRule (n + 1) i ≤ 0 := Real.sqrt_eq_zero'.mp h
        apply Real.sqrt_eq_zero'.mpr
        linarith
      exact m_eq_zero_of_vhat le_amsgradRule hβ₂ hβ₂' hprev
    rw [hg, hp]; ring
  have hy := young hα hw hzero (z := q)
  have hqterm : w / (2 * S.α t) * q ^ 2 ≤ w / (2 * S.α t) * D ^ 2 :=
    mul_le_mul_of_nonneg_left hq (by positivity)
  have hsq : (g - p) ^ 2 ≤ 2 * (g ^ 2 + p ^ 2) := by nlinarith [sq_nonneg (g + p)]
  have hmterm : S.α t * (g - p) ^ 2 / (2 * w) ≤ S.α t * (g ^ 2 + p ^ 2) / w := by
    rcases hw.eq_or_lt with h0 | h0
    · rw [← h0]; simp
    · rw [div_le_div_iff₀ (by positivity) h0]
      nlinarith [mul_nonneg hα.le (sub_nonneg.mpr hsq)]
  have hcorr : (g - p) * q ≤ w / (2 * S.α t) * D ^ 2 +
      S.α t * (g ^ 2 + p ^ 2) / w := by linarith
  have heq : g = m + S.β₁ t * (g - p) := by
    obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
    change S.g amsgradRule (n + 1) i =
      S.β₁ (n + 1) * S.m amsgradRule n i +
        (1 - S.β₁ (n + 1)) * S.g amsgradRule (n + 1) i +
          S.β₁ (n + 1) *
            (S.g amsgradRule (n + 1) i - S.m amsgradRule n i)
    ring
  conv_lhs => rw [heq, add_mul]
  nlinarith [mul_le_mul_of_nonneg_left hcorr hb]

/-- The step hypotheses are realized by the zero loss on `[-1,1]`, with
AMSGrad, `α₁ = 1`, `β₁,₁ = 0`, and `β₂ = 1/2`. -/
example :
    let S := zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧
      (∀ t (a b : Vec 1), b ≤ amsgradRule t a b) ∧
      1 ≤ 1 ∧ 0 < S.α 1 ∧ 0 ≤ S.β₁ 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧
      (∀ i, Real.sqrt (S.vhat amsgradRule 1 i) = 0 → S.m amsgradRule 1 i = 0) ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, le_amsgradRule, le_rfl, one_pos, le_rfl,
    by norm_num [zeroSetup], by norm_num [zeroSetup],
    fun _ h => m_eq_zero_of_vhat le_amsgradRule (by norm_num [zeroSetup])
      (by norm_num [zeroSetup]) h,
    fun _ => by norm_num, fun _ => by norm_num⟩

end TheoremA
end AMSGrad
end Transformer
