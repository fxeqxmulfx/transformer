import Transformer.AMSGrad.Section2_Prelim

/-
# AdamX — the sums (last1), (last2)

§5 of arXiv:1904.03590v4, proof of Corollary 5.6: the sums
`Σ_{t=2}^T β_{1,t}√(t-1)` for `β_{1,t} = β₁λ^{t-1}`, (last1), and for
`β_{1,t} = β₁/t`, (last2).

Source: arXiv:1904.03590v4, §5, proof of Corollary 5.6.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-- **(last1).**  For `β₁ ≤ 1` and `0 ≤ λ < 1`,
`Σ_{t=2}^T β₁λ^{t-1}√(t-1) ≤ 1/(1-λ)²`.
Source: arXiv:1904.03590v4, §5, proof of Corollary 5.6, (last1). -/
theorem sum_lambda_le {β₁ lam : ℝ} (hβ₁' : β₁ ≤ 1) (hl : 0 ≤ lam)
    (hl' : lam < 1) (T : ℕ) :
    ∑ t ∈ Icc 2 T, β₁ * lam ^ (t - 1) * Real.sqrt ((t : ℝ) - 1) ≤ 1 / (1 - lam) ^ 2 := by
  have hg : ∀ n : ℕ, 0 ≤ ((n : ℝ) + 1) * lam ^ n := fun n => by positivity
  have key : ∀ T : ℕ, ∑ t ∈ Icc 2 T, β₁ * lam ^ (t - 1) * Real.sqrt ((t : ℝ) - 1) ≤
      ∑ n ∈ range T, ((n : ℝ) + 1) * lam ^ n := by
    intro T
    induction T with
    | zero => simp
    | succ T ih =>
      rw [sum_range_succ]
      rcases Nat.eq_zero_or_pos T with rfl | hT
      · simp
      rw [sum_Icc_succ_top (by omega)]
      have hs : Real.sqrt ((((T + 1 : ℕ) : ℝ)) - 1) ≤ (T : ℝ) + 1 := by
        push_cast; rw [add_sub_cancel_right]
        nlinarith [Real.sq_sqrt (Nat.cast_nonneg (α := ℝ) T), Real.sqrt_nonneg (T : ℝ),
          sq_nonneg (Real.sqrt (T : ℝ) - 1)]
      have hp : 0 ≤ lam ^ (T + 1 - 1) := by positivity
      have : β₁ * lam ^ (T + 1 - 1) * Real.sqrt ((((T + 1 : ℕ) : ℝ)) - 1) ≤
          ((T : ℝ) + 1) * lam ^ T := by
        rw [Nat.add_sub_cancel] at hp ⊢
        nlinarith [mul_le_mul_of_nonneg_left hs hp, Real.sqrt_nonneg ((((T + 1 : ℕ) : ℝ)) - 1),
          mul_nonneg (mul_nonneg (sub_nonneg.mpr hβ₁') hp)
            (Real.sqrt_nonneg ((((T + 1 : ℕ) : ℝ)) - 1))]
      linarith
  have hsum : Summable fun n : ℕ => ((n : ℝ) + 1) * lam ^ n := by
    have hn : ‖lam‖ < 1 := by rwa [Real.norm_of_nonneg hl]
    have h := (hasSum_coe_mul_geometric_of_norm_lt_one hn).add (hasSum_geometric_of_lt_one hl hl')
    have he : (fun n : ℕ => ((n : ℝ) + 1) * lam ^ n) = fun n : ℕ => (n : ℝ) * lam ^ n + lam ^ n := by
      funext n; ring
    rw [he]; exact h.summable
  calc _ ≤ _ := key T
    _ ≤ ∑' n : ℕ, ((n : ℝ) + 1) * lam ^ n := hsum.sum_le_tsum _ fun n _ => hg n
    _ = 1 / (1 - lam) ^ 2 := taylor_deriv hl hl'

/-- **(last2).**  For `β₁ ≤ 1`, `Σ_{t=2}^T (β₁/t)√(t-1) ≤ 2√T`.  The source
takes `β₁ = 1`.
Source: arXiv:1904.03590v4, §5, proof of Corollary 5.6, (last2). -/
theorem sum_inv_le {β₁ : ℝ} (hβ₁' : β₁ ≤ 1) (T : ℕ) :
    ∑ t ∈ Icc 2 T, β₁ / t * Real.sqrt ((t : ℝ) - 1) ≤ 2 * Real.sqrt T := by
  have hterm : ∀ t ∈ Icc 2 T, β₁ / t * Real.sqrt ((t : ℝ) - 1) ≤ 1 / Real.sqrt t := by
    intro t ht
    have ht1 : (1 : ℝ) ≤ t := by exact_mod_cast (show 1 ≤ t by simp at ht; omega)
    have hx : (0 : ℝ) < t := by linarith
    have hr : 0 < Real.sqrt (t : ℝ) := Real.sqrt_pos.mpr hx
    have h1 : Real.sqrt ((t : ℝ) - 1) ≤ Real.sqrt t := Real.sqrt_le_sqrt (by linarith)
    rw [div_mul_eq_mul_div, div_le_div_iff₀ hx hr]
    nlinarith [mul_le_mul_of_nonneg_right h1 hr.le, Real.mul_self_sqrt hx.le,
      mul_nonneg (sub_nonneg.mpr hβ₁') (mul_nonneg (Real.sqrt_nonneg ((t : ℝ) - 1)) hr.le)]
  calc _ ≤ ∑ t ∈ Icc 2 T, 1 / Real.sqrt t := sum_le_sum hterm
    _ ≤ ∑ t ∈ Icc 1 T, 1 / Real.sqrt t :=
      sum_le_sum_of_subset_of_nonneg (Icc_subset_Icc_left (by norm_num))
        fun _ _ _ => by positivity
    _ ≤ 2 * Real.sqrt T := sum_inv_sqrt_le T

/-- The hypotheses of `sum_lambda_le` and `sum_inv_le` are satisfiable. -/
example : (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

end AMSGrad
end Transformer
