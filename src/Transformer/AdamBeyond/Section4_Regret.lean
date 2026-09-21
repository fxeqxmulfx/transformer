import Transformer.AdamBeyond.Section4_AMSGrad
import Transformer.AdamBeyond.Section4_Abel

/-
# Adam and beyond — §4: Theorem 4

The proof of Theorem 4 of arXiv:1904.09237 (appendix, §"Proof of Theorem 4"):
the bound of Lemma 3.1 of arXiv:1904.03590 (`prepare_lem`), then per coordinate
Abel summation of the first term with the weights
`c_t = √v̂_{t,i}/(2α_t(1-β_{1,t}))`.  The source splits `c_t - c_{t-1}` at
`√v̂_{t,i}/(2α_t(1-β_{1,t-1}))`; `abel_beta_le` (`Section4_Abel`) instead
uses `c_t - c_{t-1} ≤ (a_t - a_{t-1} + β_{1,t}a_{t-1})/(2(1-β₁))` with
`a_t = √v̂_{t,i}/α_t`, which gives the stated bound.

`amsgrad_regret_moment` is the bound before the lemma of the proof is
applied, its third term `Σ_t α_t ‖V̂_t^{-1/4} m_t‖²/(1-β₁)`; Theorem 4 bounds
that sum by `amsgrad_moment_sum`, the `β₁/t` remark by
`amsgrad_moment_sum_sqrt`.

Source: arXiv:1904.09237, §4, Theorem 4 (thm:amsgrad-proof); appendix,
§"Proof of Theorem 4".
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- **The proof of Theorem 4, before its lemma.**  For AMSGrad with
`α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1` and `0 < β₂ < 1`, under the
standing assumptions, for every `x* ∈ F`,

  `R_T ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + D²/(1-β₁)² Σ_{t=1}^T Σᵢ β_{1,t}√v̂_{t,i}/α_t`
  `     + 1/(1-β₁) Σ_{t=1}^T α_t ‖V̂_t^{-1/4} m_t‖²`.

Its hypotheses are witnessed after `amsgrad_moment_sum`.
Source: arXiv:1904.09237, appendix, §"Proof of Theorem 4". -/
theorem amsgrad_regret_moment {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (T : ℕ) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (1 - S.β₁ 1) ^ 2 *
          ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + (∑ t ∈ Icc 1 T, S.α t *
          ∑ i, S.m amsgradRule t i ^ 2 / Real.sqrt (S.vhat amsgradRule t i)) / (1 - S.β₁ 1) := by
  rcases Nat.eq_zero_or_pos T with rfl | hT
  · simp [Setup.regret]
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  have hB : 0 < 1 - S.β₁ 1 := by linarith
  have hp := prepare_lem hS le_amsgradRule hα' hβ₁ hβ₁' hβ₂ hβ₂' hT hxstar
  have hmono : ∀ t, 2 ≤ t → ∀ i, Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1) ≤
      Real.sqrt (S.vhat amsgradRule t i) / S.α t := fun t ht2 i => by
    rw [hαt, div_div_eq_mul_div, div_div_eq_mul_div]
    have hv := vhat_le_succ S (t - 1) i
    rw [Nat.sub_add_cancel (by omega)] at hv
    gcongr
    exact Nat.sub_le t 1
  have hi := fun i : Fin d => abel_beta_le
    (fun t => Real.sqrt (S.vhat amsgradRule t i) / S.α t) S.β₁
    (fun t => (S.x amsgradRule t i - xstar i) ^ 2) (E := D ^ 2) hT hβ₁'
    (fun t ht => div_nonneg (Real.sqrt_nonneg _) (hα' t (mem_Icc.1 ht).1).le)
    (fun t ht => hmono t (mem_Icc.1 ht).1 i)
    (fun t ht => hβ₁ t (mem_Icc.1 ht).1)
    (fun t _ => ⟨sq_nonneg _, by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS _ t) _ hxstar i) 2⟩)
  have hmain : ∑ i, ∑ t ∈ Icc 1 T, Real.sqrt (S.vhat amsgradRule t i) /
        (2 * S.α t * (1 - S.β₁ t)) *
          ((S.x amsgradRule t i - xstar i) ^ 2 - (S.x amsgradRule (t + 1) i - xstar i) ^ 2)
      + ∑ i, ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule (t - 1) i) /
          (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x amsgradRule t i - xstar i) ^ 2 ≤
      ∑ i, D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / (2 * (1 - S.β₁ 1))
      + ∑ i, D ^ 2 / (1 - S.β₁ 1) * ∑ t ∈ Icc 2 T,
          S.β₁ t * (Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1)) := by
    rw [← sum_add_distrib, ← sum_add_distrib]
    refine sum_le_sum fun i _ => le_trans (le_of_eq ?_) (hi i)
    congr 1 <;> refine sum_congr rfl fun t _ => ?_
    · generalize 1 - S.β₁ t = q; ring
    · generalize 1 - S.β₁ 1 = q; ring
  have h1 : ∑ i, D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / (2 * (1 - S.β₁ 1)) ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i) := by
    rw [mul_sum]
    refine sum_le_sum fun i _ => ?_
    have hn : 0 ≤ D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) *
        Real.sqrt (S.vhat amsgradRule T i) := by positivity
    have e : D ^ 2 * (Real.sqrt (S.vhat amsgradRule T i) / S.α T) / (2 * (1 - S.β₁ 1)) =
        D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * Real.sqrt (S.vhat amsgradRule T i) / 2 := by
      rw [hαt]; simp only; rw [div_div_eq_mul_div]; generalize 1 - S.β₁ 1 = q; ring
    rw [e]; linarith
  have hB0 : 0 ≤ S.β₁ 1 := (hβ₁ 1 le_rfl).1
  have hcoef : D ^ 2 / (1 - S.β₁ 1) ≤ D ^ 2 / (1 - S.β₁ 1) ^ 2 :=
    div_le_div_of_nonneg_left (sq_nonneg D) (by positivity) (by nlinarith)
  have h2 : ∑ i, D ^ 2 / (1 - S.β₁ 1) * ∑ t ∈ Icc 2 T,
        S.β₁ t * (Real.sqrt (S.vhat amsgradRule (t - 1) i) / S.α (t - 1)) ≤
      D ^ 2 / (1 - S.β₁ 1) ^ 2 *
        ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t := by
    rw [sum_comm, mul_sum]
    refine sum_le_sum fun i _ => mul_le_mul hcoef ?_ ?_ (by positivity)
    · refine (sum_le_sum fun t ht => ?_).trans (sum_le_sum_of_subset_of_nonneg
        (Icc_subset_Icc_left one_le_two) fun t ht _ => ?_)
      · rw [mul_div_assoc]
        exact mul_le_mul_of_nonneg_left (hmono t (mem_Icc.1 ht).1 i)
          (hβ₁ t (by have := (mem_Icc.1 ht).1; omega)).1
      · exact div_nonneg (mul_nonneg (hβ₁ t (mem_Icc.1 ht).1).1 (Real.sqrt_nonneg _))
          (hα' t (mem_Icc.1 ht).1).le
    · refine sum_nonneg fun t ht => mul_nonneg
        (hβ₁ t (by have := (mem_Icc.1 ht).1; omega)).1 (div_nonneg (Real.sqrt_nonneg _) ?_)
      rw [hαt]; positivity
  have h3 : ∑ i, ∑ t ∈ Icc 1 T, S.α t / (1 - S.β₁ 1) * S.m amsgradRule t i ^ 2 /
      Real.sqrt (S.vhat amsgradRule t i) = (∑ t ∈ Icc 1 T, S.α t *
        ∑ i, S.m amsgradRule t i ^ 2 / Real.sqrt (S.vhat amsgradRule t i)) / (1 - S.β₁ 1) := by
    rw [sum_comm, sum_div]
    refine sum_congr rfl fun t _ => ?_
    rw [mul_sum, sum_div]
    exact sum_congr rfl fun i _ => by ring
  linarith

/-- **Theorem 4.**  For AMSGrad with `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`,
`0 < β₂ < 1` and `γ = β₁/√β₂ < 1`, under the standing assumptions, for every
`x* ∈ F`,

  `R_T ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + D²/(1-β₁)² Σ_{t=1}^T Σᵢ β_{1,t}√v̂_{t,i}/α_t`
  `     + α√(1 + log T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

Its hypotheses are witnessed after `amsgrad_moment_sum`.
Source: arXiv:1904.09237, §4, Theorem 4 (thm:amsgrad-proof). -/
theorem amsgrad_regret {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    (T : ℕ) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (1 - S.β₁ 1) ^ 2 *
          ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  have h := amsgrad_regret_moment hS hα hαt hβ₁ hβ₁' hβ₂ hβ₂' T hxstar
  have hm := amsgrad_moment_sum hα hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ T
  have hB : 0 < 1 - S.β₁ 1 := by linarith
  have e' : α * Real.sqrt (1 + Real.log T) /
        ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
        * ∑ i, S.gnorm amsgradRule T i = (α * Real.sqrt (1 + Real.log T) /
        ((1 - S.β₁ 1) * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
        * ∑ i, S.gnorm amsgradRule T i) / (1 - S.β₁ 1) := by
    generalize 1 - S.β₁ 1 = q; generalize 1 - S.β₁ 1 / Real.sqrt S.β₂ = r; ring
  rw [e']
  linarith [div_le_div_of_nonneg_right hm hB.le]

end AdamBeyond
end Transformer
