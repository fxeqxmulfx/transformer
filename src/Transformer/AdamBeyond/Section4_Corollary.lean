import Transformer.AdamBeyond.Section4_Regret

/-
# Adam and beyond — §4: Corollary 1 and the `β₁/t` remark

Corollary 1 of arXiv:1904.09237 and the remark after it, both consequences of
Theorem 4 (`amsgrad_regret`); the deviations are recorded in the docstring of
`Section4_AMSGrad`.  Their hypotheses are witnessed there.

Source: arXiv:1904.09237, §4, Corollary 1 (cor:t1-cor) and the paragraph
after it; appendix, the closing remark of §"Proof of Theorem 4".
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- `Σ_{t=1}^T λ^{t-1}√t ≤ 1/(1-λ)²` for `0 ≤ λ < 1`.
arXiv:1904.09237, §4, Corollary 1 (cor:t1-cor). -/
theorem sum_lambda_sqrt_le {lam : ℝ} (hl : 0 ≤ lam) (hl' : lam < 1) (T : ℕ) :
    ∑ t ∈ Icc 1 T, lam ^ (t - 1) * Real.sqrt t ≤ 1 / (1 - lam) ^ 2 := by
  have hsum : Summable fun n : ℕ => ((n : ℝ) + 1) * lam ^ n := by
    have hn : ‖lam‖ < 1 := by rwa [Real.norm_of_nonneg hl]
    have h := (hasSum_coe_mul_geometric_of_norm_lt_one hn).add
      (hasSum_geometric_of_lt_one hl hl')
    have he : (fun n : ℕ => ((n : ℝ) + 1) * lam ^ n) =
        fun n : ℕ => (n : ℝ) * lam ^ n + lam ^ n := by
      funext n; ring
    rw [he]; exact h.summable
  calc ∑ t ∈ Icc 1 T, lam ^ (t - 1) * Real.sqrt t
      ≤ ∑ n ∈ range T, ((n : ℝ) + 1) * lam ^ n := by
        rw [← Ico_add_one_right_eq_Icc, sum_Ico_eq_sum_range, Nat.add_sub_cancel]
        refine sum_le_sum fun n _ => ?_
        rw [show 1 + n - 1 = n by omega, mul_comm]
        push_cast
        gcongr
        rw [add_comm]
        exact Real.sqrt_le_self_iff.2 (Or.inr (by linarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]))
    _ ≤ ∑' n : ℕ, ((n : ℝ) + 1) * lam ^ n := hsum.sum_le_tsum _ fun n _ => by positivity
    _ = 1 / (1 - lam) ^ 2 := taylor_deriv hl hl'

/-- The hypotheses of `sum_lambda_sqrt_le` are satisfiable: `λ = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 := ⟨le_rfl, one_pos⟩

/-- **Corollary 1**, with the factor `d/α` of its second term restored.  Under
the assumptions of Theorem 4 with `β_{1,t} = β₁λ^{t-1}`, `0 ≤ λ < 1`, for
every `x* ∈ F`,

  `R_T ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + dβ₁D²G/(α(1-β₁)²(1-λ)²)`
  `     + α√(1 + log T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

The source's second term is `β₁D²G/((1-β₁)²(1-λ)²)`; see the module docstring.

Source: arXiv:1904.09237, §4, Corollary 1 (cor:t1-cor). -/
theorem amsgrad_regret_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α β₁ lam : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hl : 0 ≤ lam) (hl' : lam < 1) (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    (T : ℕ) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - β₁)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + d * β₁ * D ^ 2 * G / (α * (1 - β₁) ^ 2 * (1 - lam) ^ 2)
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - β₁) ^ 2 * (1 - β₁ / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  have hS1 : S.β₁ 1 = β₁ := by rw [hb]; simp
  have hβt : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1 := fun t _ => by
    rw [hb, hS1]
    exact ⟨by positivity, mul_le_of_le_one_right hβ₁ (pow_le_one₀ hl hl'.le)⟩
  have h4 := amsgrad_regret hS hα hαt hβt (hS1 ▸ hβ₁') hβ₂ hβ₂' (hS1 ▸ hγ) T hxstar
  rw [hS1] at h4
  have hdG : 0 ≤ (d : ℝ) * G := by
    rcases Nat.eq_zero_or_pos d with h | h
    · simp [h]
    · exact mul_nonneg (Nat.cast_nonneg _) ((abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem ⟨0, h⟩))
  have h2 : ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t ≤
      d * G * β₁ / α * ∑ t ∈ Icc 1 T, lam ^ (t - 1) * Real.sqrt t := by
    rw [mul_sum]
    refine sum_le_sum fun t _ => ?_
    have hc : 0 ≤ β₁ * lam ^ (t - 1) * Real.sqrt t / α := by positivity
    calc ∑ i, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
        = ∑ i, β₁ * lam ^ (t - 1) * Real.sqrt t / α * Real.sqrt (S.vhat amsgradRule t i) := by
          refine sum_congr rfl fun i _ => ?_
          rw [hb, hαt]; simp only; rw [div_div_eq_mul_div]; ring
      _ ≤ ∑ _i : Fin d, β₁ * lam ^ (t - 1) * Real.sqrt t / α * G :=
          sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (vt hS hβ₂.le hβ₂'.le t i) hc
      _ = _ := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have h2' := mul_le_mul_of_nonneg_left (sum_lambda_sqrt_le hl hl' T)
    (show 0 ≤ d * G * β₁ / α by positivity)
  have hk : 0 ≤ D ^ 2 / (1 - β₁) ^ 2 := by positivity
  have e : d * β₁ * D ^ 2 * G / (α * (1 - β₁) ^ 2 * (1 - lam) ^ 2) =
      D ^ 2 / (1 - β₁) ^ 2 * (d * G * β₁ / α * (1 / (1 - lam) ^ 2)) := by
    generalize 1 - β₁ = q; generalize 1 - lam = r; ring
  have := mul_le_mul_of_nonneg_left (h2.trans h2') hk
  rw [← e] at this
  linarith

/-- **`β_{1,t} = β₁/t` gives `O(√T)` regret.**  Under the assumptions of
Theorem 4 with `β_{1,t} = β₁/t`, there is `K` with `R_T ≤ K√T` for every `T`
and every `x* ∈ F`.

Source: arXiv:1904.09237, §4, the paragraph after Corollary 1; appendix, the
closing remark of §"Proof of Theorem 4". -/
theorem amsgrad_regret_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α β₁ : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    ∃ K : ℝ, ∀ T : ℕ, ∀ xstar ∈ F, S.regret amsgradRule xstar T ≤ K * Real.sqrt T := by
  sorry

end AdamBeyond
end Transformer
