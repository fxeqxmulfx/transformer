import Transformer.AMSGrad.Section4_Telescope
import Transformer.AMSGrad.Section4_Third

/-
# AMSGrad — the corrected convergence theorem

§4 of arXiv:1904.03590v4: Theorem 4.1 (`mainthm_change_beta1`) and
Corollary 4.5.  Lemma 4.4 is `Section4_MainLemma`.

**What the source says and what is carried here.**

* The source assumes `γ = β₁/√β₂ ≤ 1`; its bounds divide by `1 - γ`, so
  `γ < 1` is assumed instead, together with `0 ≤ β₁ < 1`, `0 < β₂ < 1` and
  `α > 0`, which the source assumes throughout.

* Theorem 4.1 asserts "there is some `1 ≤ t₀ ≤ T` such that … for all
  `T ≥ 1`".  Its `t₀` is the `t₀` of Lemma 4.3, which does not depend on `T`;
  it is stated as `∃ t₀ ≥ 1, ∀ T ≥ 1, ∀ x* ∈ F, R(T) ≤ …`.  The regret is
  bounded for every `x* ∈ F`, not only for the minimizer of `Σ_t f_t`.  One
  theorem per setting of `β_{1,t}`: `mainthm_lambda`, `mainthm_inv`.

* Corollary 4.5 asserts `lim R(T)/T = 0`.  Theorem 4.1 is an upper bound and
  gives only the upper half, `limsup R(T)/T ≤ 0`, which is what is stated here,
  `cor_lambda` and `cor_inv`.  The lower half is false: `Section4_Counter`.

Source: arXiv:1904.03590v4, §4, Theorem 4.1, Lemma 4.4, Corollary 4.5.
-/

open Finset Filter

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- **Theorem 4.1, `β_{1,t} = β₁λ^{t-1}`.**  With `α_t = α/√t`, `0 ≤ β₁ < 1`,
`0 < λ < 1`, `0 < β₂ < 1` and `γ < 1`, there is `t₀ ≥ 1` such that for all
`T ≥ 1` and `x* ∈ F`,

  `R(T) ≤ dD²G/(2α(1-β₁)) (Σ_{t=1}^{t₀} √t + √T) + dD²G/(2α(1-β₁)(1-λ)²)`
  `      + α√(ln T + 1)/((1-β₁)²√(1-β₂)(1-γ)) Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.03590v4, §4, Theorem 4.1. -/
theorem mainthm_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ lam : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    ∃ t₀, 1 ≤ t₀ ∧ ∀ T, 1 ≤ T → ∀ xstar ∈ F,
      S.regret amsgradRule xstar T ≤
        d * D ^ 2 * G / (2 * α * (1 - β₁)) * (∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t + Real.sqrt T)
        + d * D ^ 2 * G / (2 * α * (1 - β₁) * (1 - lam) ^ 2)
        + α * Real.sqrt (Real.log T + 1) /
            ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂))
            * ∑ i, S.gnorm amsgradRule T i := by
  have hb1 : S.β₁ 1 = β₁ := by rw [hb]; simp
  have hβt : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁ := fun t _ => by
    rw [hb]
    exact ⟨mul_nonneg hβ₁ (pow_nonneg hl.le _), mul_le_of_le_one_right hβ₁ (pow_le_one₀ hl.le hl'.le)⟩
  obtain ⟨t₀, ht₀1, ht₀⟩ := t_0_lambda (S := S) hβ₁ hβ₁' hl hl' hb
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  refine ⟨t₀, ht₀1, fun T hT xstar hx => ?_⟩
  have hp := prepare_lem hS (R := amsgradRule) (fun _ _ _ => le_sup_right) hα' (by rw [hb1]; exact hβt)
    (by rw [hb1]; exact hβ₁') hβ₂ hβ₂' hT hx
  have h1 := eqmain_le hS hα hαt hβt hβ₁' hβ₂.le hβ₂'.le ht₀ hT hx
  have h2 : ∑ i, ∑ t ∈ Icc 1 T, S.α t / (1 - S.β₁ 1) * S.m amsgradRule t i ^ 2 /
      Real.sqrt (S.vhat amsgradRule t i) ≤
      α * Real.sqrt (Real.log T + 1) /
        ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) *
          ∑ i, S.gnorm amsgradRule T i := by
    rw [hb1]
    exact eqsecond_le (R := amsgradRule) (fun _ _ _ => le_sup_right) hα.le hαt hβt hβ₁' hβ₂ hβ₂' hγ T
  have h3 := eqthird_lambda_le hS hα hαt hβ₁' hl.le hl' hb hβ₂.le hβ₂'.le T hx
  linarith

/-- **Theorem 4.1, `β_{1,t} = β₁/t`.**  Under the assumptions of
`mainthm_lambda` with `β_{1,t} = β₁/t`, there is `t₀ ≥ 1` such that for all
`T ≥ 1` and `x* ∈ F`,

  `R(T) ≤ dD²G/(2α(1-β₁)) (Σ_{t=1}^{t₀} √t + √T) + dD²G√T/(α(1-β₁))`
  `      + α√(ln T + 1)/((1-β₁)²√(1-β₂)(1-γ)) Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.03590v4, §4, Theorem 4.1. -/
theorem mainthm_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    ∃ t₀, 1 ≤ t₀ ∧ ∀ T, 1 ≤ T → ∀ xstar ∈ F,
      S.regret amsgradRule xstar T ≤
        d * D ^ 2 * G / (2 * α * (1 - β₁)) * (∑ t ∈ Icc (1 : ℕ) t₀, Real.sqrt t + Real.sqrt T)
        + d * D ^ 2 * G * Real.sqrt T / (α * (1 - β₁))
        + α * Real.sqrt (Real.log T + 1) /
            ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂))
            * ∑ i, S.gnorm amsgradRule T i := by
  have hb1 : S.β₁ 1 = β₁ := by rw [hb]; simp
  have hβt : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁ := fun t ht => by
    rw [hb]
    exact ⟨div_nonneg hβ₁ (Nat.cast_nonneg _), div_le_self hβ₁ (by exact_mod_cast ht)⟩
  obtain ⟨t₀, ht₀1, ht₀⟩ := t_0_inv (S := S) hβ₁ hβ₁' hb
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  refine ⟨t₀, ht₀1, fun T hT xstar hx => ?_⟩
  have hp := prepare_lem hS (R := amsgradRule) (fun _ _ _ => le_sup_right) hα' (by rw [hb1]; exact hβt)
    (by rw [hb1]; exact hβ₁') hβ₂ hβ₂' hT hx
  have h1 := eqmain_le hS hα hαt hβt hβ₁' hβ₂.le hβ₂'.le ht₀ hT hx
  have h2 : ∑ i, ∑ t ∈ Icc 1 T, S.α t / (1 - S.β₁ 1) * S.m amsgradRule t i ^ 2 /
      Real.sqrt (S.vhat amsgradRule t i) ≤
      α * Real.sqrt (Real.log T + 1) /
        ((1 - β₁) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) *
          ∑ i, S.gnorm amsgradRule T i := by
    rw [hb1]
    exact eqsecond_le (R := amsgradRule) (fun _ _ _ => le_sup_right) hα.le hαt hβt hβ₁' hβ₂ hβ₂' hγ T
  have h3 := eqthird_inv_le hS hα hαt hβ₁' hb hβ₂.le hβ₂'.le T hx
  linarith

/-- **Corollary 4.5, `β_{1,t} = β₁λ^{t-1}`, the upper half.**  Under the
assumptions of `mainthm_lambda`, for every `ε > 0`, eventually
`R(T)/T ≤ ε` for all `x* ∈ F`.  The source asserts `lim R(T)/T = 0`; see the
module docstring.

Source: arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem cor_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ lam : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret amsgradRule xstar T / T ≤ ε := by
  sorry

/-- **Corollary 4.5, `β_{1,t} = β₁/t`, the upper half.**  Under the
assumptions of `mainthm_inv`, for every `ε > 0`, eventually `R(T)/T ≤ ε` for
all `x* ∈ F`.

Source: arXiv:1904.03590v4, §4, Corollary 4.5. -/
theorem cor_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret amsgradRule xstar T / T ≤ ε := by
  sorry

/-- The hypotheses of Theorem 4.1 and Corollary 4.5, in both settings, are
satisfiable: the zero cost on `[-1, 1]`, `α = 1`, `β₁ = 0`, `λ = 1/2`,
`β₂ = 1/2`, `ε = 1`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (∀ t, S.β₁ t = 0 * (1 / 2) ^ (t - 1)) ∧
      (∀ t, S.β₁ t = 0 / t) ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ 0 / Real.sqrt S.β₂ < 1 ∧ (0 : ℝ) < 1 :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, le_rfl, one_pos, by norm_num, by norm_num,
    fun _ => by simp [zeroSetup], fun _ => by simp [zeroSetup], by norm_num [zeroSetup],
    by norm_num [zeroSetup], by simp, one_pos⟩

end AMSGrad
end Transformer
