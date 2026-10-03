import Transformer.AMSGrad.Section4_TheoremAAlternateStep

/-
# AMSGrad — Theorem A: EnergyBounds

Alternative proof of the printed regret bound in arXiv:1904.03590v4, §1,
Theorem A. The paper’s telescoping step is invalid for varying schedules;
these estimates use the schedule-free projection potential instead.
-/

open Finset

namespace Transformer
namespace AMSGrad
namespace TheoremA

variable {d : ℕ}

/-- The second moment controls the current gradient: g²/√v̂ ≤ |g|/√(1−β₂), including zero coordinates. Source: arXiv:1904.03590v4, Algorithm 1 and §4, Lemma 4.4. -/
theorem gradient_energy_point {S : Setup d} {R : Rule d}
    (hR : ∀ t a b, b ≤ R t a b) (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    {t : ℕ} (ht : 1 ≤ t) (i : Fin d) :
    S.g R t i ^ 2 / Real.sqrt (S.vhat R t i) ≤
      |S.g R t i| / Real.sqrt (1 - S.β₂) := by
  set c := Real.sqrt (1 - S.β₂)
  set g := S.g R t i
  set w := Real.sqrt (S.vhat R t i)
  have hc : 0 < c := Real.sqrt_pos.mpr (by linarith)
  have hg2 : (1 - S.β₂) * g ^ 2 ≤ S.vhat R t i := by
    obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
    have hv0 := v_nonneg hβ₂.le hβ₂'.le R n i
    change 0 ≤ S.v R n i at hv0
    have hv1 := hR (n + 1) (S.vhat R n) (S.v R (n + 1)) i
    change S.v R (n + 1) i ≤ S.vhat R (n + 1) i at hv1
    have hv : S.v R (n + 1) i = S.β₂ * S.v R n i +
        (1 - S.β₂) * S.g R (n + 1) i ^ 2 := rfl
    change (1 - S.β₂) * S.g R (n + 1) i ^ 2 ≤ S.vhat R (n + 1) i
    nlinarith [mul_nonneg hβ₂.le hv0]
  have hgw : c * |g| ≤ w := by
    calc c * |g| = Real.sqrt ((1 - S.β₂) * g ^ 2) := by
          rw [Real.sqrt_mul (by linarith : 0 ≤ 1 - S.β₂), Real.sqrt_sq_eq_abs]
      _ ≤ w := Real.sqrt_le_sqrt hg2
  by_cases hg : g = 0
  · simp [hg]
  have hga : 0 < |g| := abs_pos.mpr hg
  have hw : 0 < w := lt_of_lt_of_le (mul_pos hc hga) hgw
  rw [div_le_div_iff₀ hw hc]
  nlinarith [sq_abs g]

/-- Cauchy–Schwarz and the harmonic estimate bound the sum of current-gradient energy. Source: arXiv:1904.03590v4, §2, Lemmas 2.2 and 2.4, and §4. -/
theorem gradient_energy_le {S : Setup d} {R : Rule d}
    (hR : ∀ t a b, b ≤ R t a b) {α : ℝ} (hα : 0 ≤ α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (T : ℕ) (i : Fin d) :
    ∑ t ∈ Icc 1 T, S.α t * S.g R t i ^ 2 / Real.sqrt (S.vhat R t i) ≤
      α * Real.sqrt (1 + Real.log T) / Real.sqrt (1 - S.β₂) * S.gnorm R T i := by
  have hc : 0 < Real.sqrt (1 - S.β₂) := Real.sqrt_pos.mpr (by linarith)
  have hcs : ∑ t ∈ Icc 1 T, |S.g R t i| / Real.sqrt t ≤
      Real.sqrt (Real.log T + 1) * S.gnorm R T i := by
    have h := cauchy_schwarz T (fun t => 1 / Real.sqrt t) (fun t => |S.g R t i|)
    have e1 : ∀ t : ℕ, (1 / Real.sqrt t) ^ 2 = 1 / t := fun t => by
      rw [div_pow, Real.sq_sqrt (Nat.cast_nonneg _), one_pow]
    simp only [e1, sq_abs, one_div_mul_eq_div] at h
    calc ∑ t ∈ Icc 1 T, |S.g R t i| / Real.sqrt t
        ≤ Real.sqrt ((∑ t ∈ Icc 1 T, 1 / (t : ℝ)) * ∑ t ∈ Icc 1 T, S.g R t i ^ 2) :=
          (le_abs_self _).trans (Real.abs_le_sqrt h)
      _ = Real.sqrt (∑ t ∈ Icc 1 T, 1 / (t : ℝ)) * S.gnorm R T i :=
          Real.sqrt_mul (sum_nonneg fun _ _ => by positivity) _
      _ ≤ _ := mul_le_mul_of_nonneg_right (Real.sqrt_le_sqrt (harmonic_le T)) (Real.sqrt_nonneg _)
  calc ∑ t ∈ Icc 1 T, S.α t * S.g R t i ^ 2 / Real.sqrt (S.vhat R t i)
      ≤ ∑ t ∈ Icc 1 T, α / Real.sqrt t * (|S.g R t i| / Real.sqrt (1 - S.β₂)) := by
        apply sum_le_sum
        intro t ht
        rw [hαt]
        convert mul_le_mul_of_nonneg_left
          (gradient_energy_point hR hβ₂ hβ₂' (mem_Icc.mp ht).1 i)
          (by positivity : 0 ≤ α / Real.sqrt t) using 1
        ring
    _ = α / Real.sqrt (1 - S.β₂) * ∑ t ∈ Icc 1 T, |S.g R t i| / Real.sqrt t := by
        rw [mul_sum]
        refine sum_congr rfl fun t _ => ?_
        ring
    _ ≤ α / Real.sqrt (1 - S.β₂) *
          (Real.sqrt (Real.log T + 1) * S.gnorm R T i) := by
        gcongr
    _ = _ := by rw [add_comm (Real.log T) 1]; ring

/-- At the next step, the old moment has no larger energy weight: αₜ/√v̂ₜ ≤ αₜ₋₁/√v̂ₜ₋₁. Source: arXiv:1904.03590v4, Algorithm 1 and §4. -/
theorem previous_moment_energy_point {S : Setup d} {α : ℝ}
    (hα : 0 ≤ α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    {t : ℕ} (ht : 1 ≤ t) (i : Fin d) :
    S.α t * S.m amsgradRule (t - 1) i ^ 2 / Real.sqrt (S.vhat amsgradRule t i) ≤
      S.α (t - 1) * S.m amsgradRule (t - 1) i ^ 2 /
        Real.sqrt (S.vhat amsgradRule (t - 1) i) := by
  rcases eq_or_lt_of_le ht with h | h
  · subst t
    simp [show S.m amsgradRule 0 i = 0 from rfl]
  have htprev : 1 ≤ t - 1 := by omega
  set w := Real.sqrt (S.vhat amsgradRule t i)
  set p := Real.sqrt (S.vhat amsgradRule (t - 1) i)
  set m := S.m amsgradRule (t - 1) i
  have hp : 0 ≤ p := Real.sqrt_nonneg _
  have hpw : p ≤ w := by
    have := vhat_le_succ S (t - 1) i
    rw [Nat.sub_add_cancel (by omega)] at this
    exact Real.sqrt_le_sqrt this
  by_cases hp0 : p = 0
  · have hm : m = 0 := m_eq_zero_of_vhat le_amsgradRule hβ₂ hβ₂' hp0
    simp [hm]
  have hppos : 0 < p := hp.lt_of_ne fun h => hp0 h.symm
  have hq : m ^ 2 / w ≤ m ^ 2 / p :=
    div_le_div_of_nonneg_left (sq_nonneg _) hppos hpw
  have hαmono : S.α t ≤ S.α (t - 1) := by
    rw [hαt]
    exact div_le_div_of_nonneg_left hα
      (Real.sqrt_pos.mpr (by exact_mod_cast htprev))
      (Real.sqrt_le_sqrt (by exact_mod_cast Nat.sub_le t 1))
  have hleft : 0 ≤ m ^ 2 / w := by positivity
  have hright : 0 ≤ S.α (t - 1) := by rw [hαt]; positivity
  calc S.α t * m ^ 2 / w = S.α t * (m ^ 2 / w) := by ring
    _ ≤ S.α (t - 1) * (m ^ 2 / w) :=
        mul_le_mul_of_nonneg_right hαmono hleft
    _ ≤ S.α (t - 1) * (m ^ 2 / p) :=
        mul_le_mul_of_nonneg_left hq hright
    _ = _ := by ring

/-- The previous-moment energy sum is bounded by the current-moment energy sum after shifting indices. Source: arXiv:1904.03590v4, Algorithm 1 and §4. -/
theorem previous_moment_energy_le {S : Setup d} {α : ℝ}
    (hα : 0 ≤ α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (T : ℕ) (i : Fin d) :
    ∑ t ∈ Icc 1 T, S.α t * S.m amsgradRule (t - 1) i ^ 2 /
        Real.sqrt (S.vhat amsgradRule t i) ≤
      ∑ t ∈ Icc 1 T, S.α t * S.m amsgradRule t i ^ 2 /
        Real.sqrt (S.vhat amsgradRule t i) := by
  set U : ℕ → ℝ := fun t => S.α t * S.m amsgradRule t i ^ 2 /
    Real.sqrt (S.vhat amsgradRule t i)
  have hU0 : U 0 = 0 := by simp [U, show S.m amsgradRule 0 i = 0 from rfl]
  have hU : 0 ≤ U T := by
    dsimp [U]
    by_cases hT : T = 0
    · simp [hT, show S.m amsgradRule 0 i = 0 from rfl]
    · have hT' : 1 ≤ T := by omega
      rw [hαt]
      positivity
  have hshift : ∀ n, 1 ≤ n → ∑ t ∈ Icc 1 n, U (t - 1) + U n =
      ∑ t ∈ Icc 1 n, U t := by
    intro n hn
    induction n, hn using Nat.le_induction with
    | base => simp [hU0]
    | succ n hn ih =>
      rw [sum_Icc_succ_top (by omega), sum_Icc_succ_top (by omega), ← ih,
        Nat.add_sub_cancel]
  rcases Nat.eq_zero_or_pos T with hT | hT
  · simp [hT]
  have hsum := sum_le_sum (s := Icc 1 T) (fun t ht =>
    previous_moment_energy_point hα hαt hβ₂ hβ₂' (mem_Icc.mp ht).1 i)
  change (∑ t ∈ Icc 1 T, _) ≤ ∑ t ∈ Icc 1 T, U (t - 1) at hsum
  change (∑ t ∈ Icc 1 T, _) ≤ ∑ t ∈ Icc 1 T, U t
  linarith [hshift T hT]

/-- Lemma 4.4 bounds the current-moment energy for αₜ=α/√t. Source: arXiv:1904.03590v4, §4, Lemma 4.4. -/
theorem moment_energy_le {S : Setup d} {α β₁ : ℝ}
    (hα : 0 ≤ α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁) (hβ₁' : β₁ < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1)
    (hγ : β₁ / Real.sqrt S.β₂ < 1) (T : ℕ) (i : Fin d) :
    ∑ t ∈ Icc 1 T, S.α t * S.m amsgradRule t i ^ 2 /
        Real.sqrt (S.vhat amsgradRule t i) ≤
      α * Real.sqrt (1 + Real.log T) /
        ((1 - β₁) * (1 - β₁ / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂)) *
          S.gnorm amsgradRule T i := by
  have h := mainlem le_amsgradRule hβ₁ hβ₁' hβ₂ hβ₂' hγ T i
  have e : ∑ t ∈ Icc 1 T, S.α t * S.m amsgradRule t i ^ 2 /
      Real.sqrt (S.vhat amsgradRule t i) =
      α * ∑ t ∈ Icc 1 T,
        S.m amsgradRule t i ^ 2 / Real.sqrt (t * S.vhat amsgradRule t i) := by
    rw [mul_sum]
    refine sum_congr rfl fun t _ => ?_
    rw [hαt, Real.sqrt_mul (Nat.cast_nonneg _)]
    simp only [div_eq_mul_inv, mul_inv]
    ring
  rw [e]
  have hK : 0 ≤ α := hα
  have h' := mul_le_mul_of_nonneg_left h hK
  calc α * ∑ t ∈ Icc 1 T,
        S.m amsgradRule t i ^ 2 / Real.sqrt (t * S.vhat amsgradRule t i)
      ≤ α * (Real.sqrt (Real.log T + 1) /
        ((1 - β₁) * Real.sqrt (1 - S.β₂) *
          (1 - β₁ / Real.sqrt S.β₂)) * S.gnorm amsgradRule T i) := h'
    _ = _ := by rw [add_comm (Real.log T) 1]; ring

/-- The energy hypotheses hold for a zero-loss AMSGrad run with
`αₜ = 1/√t`, `β₁,ₜ = 0`, and `β₂ = 1/2`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    (∀ t (a b : Vec 1), b ≤ amsgradRule t a b) ∧
      (0 : ℝ) ≤ 1 ∧ S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧
      (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ 0) ∧
      (0 : ℝ) < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧
      0 / Real.sqrt S.β₂ < 1 :=
  ⟨le_amsgradRule, zero_le_one, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    one_pos, by norm_num [zeroSetup], by norm_num [zeroSetup], by simp⟩

end TheoremA
end AMSGrad
end Transformer
