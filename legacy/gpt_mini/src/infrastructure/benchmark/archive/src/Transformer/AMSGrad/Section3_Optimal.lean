import Transformer.AMSGrad.Section3_Example

/-
# AMSGrad — what Example 3.2 shows

§3 of arXiv:1904.03590v4: the optimum of Example 3.2, and the failure of the
step of Reddi et al. the example is built to exhibit.

**What the source says and what is carried here.**

* "The optimal solution is `x* = -1`" holds for every horizon `T ≥ 1`,
  `exa_optimal`.

* The point of the example is that the first ("red") inequality of Reddi et
  al., replacing `1/(1 - β_{1,t})` by `1/(1 - β₁)` in (3.1), is "not
  guaranteed".  It is false for this run already at `T = 2`, and this run
  satisfies every hypothesis of Theorem A (`isOnlineConvex_exa`):
  `not_red_ineq`.

Source: arXiv:1904.03590v4, §3, Example 3.2 and the paragraph before it.
-/

open Finset

namespace Transformer
namespace AMSGrad

/-! ### `x* = -1` is optimal for every horizon -/

/-- Over the first period the slopes sum to `1010 - 10(T - 1)`. -/
theorem exaCoef_sum_le_101 {T : ℕ} (h1 : 1 ≤ T) (h2 : T ≤ 101) :
    ∑ t ∈ Icc 1 T, exaCoef t = 1010 - 10 * ((T : ℝ) - 1) := by
  induction T, h1 using Nat.le_induction with
  | base => norm_num [exaCoef]
  | succ n hn ih =>
    rw [Finset.sum_Icc_succ_top (by omega), ih (by omega)]
    have : (n + 1) % 101 ≠ 1 := by omega
    simp only [exaCoef, this, ↓reduceIte]
    push_cast; ring

/-- The slopes are `101`-periodic and a full period sums to `10`. -/
theorem exaCoef_sum_add_101 (T : ℕ) :
    ∑ t ∈ Icc 1 (T + 101), exaCoef t = ∑ t ∈ Icc 1 T, exaCoef t + 10 := by
  induction T with
  | zero => rw [exaCoef_sum_le_101 (by norm_num) le_rfl]; norm_num
  | succ n ih =>
    rw [show n + 1 + 101 = n + 101 + 1 by ring, Finset.sum_Icc_succ_top (by omega), ih,
      Finset.sum_Icc_succ_top (by omega)]
    have : (n + 101 + 1) % 101 = (n + 1) % 101 := by omega
    simp only [exaCoef, this]
    ring

/-- The slopes up to any `T ≥ 1` sum to a positive number. -/
theorem exaCoef_sum_pos {T : ℕ} (hT : 1 ≤ T) : 0 < ∑ t ∈ Icc 1 T, exaCoef t := by
  induction T using Nat.strong_induction_on with
  | _ T ih =>
    rcases le_or_gt T 101 with h | h
    · rw [exaCoef_sum_le_101 hT h]
      have : (T : ℝ) ≤ 101 := by exact_mod_cast h
      linarith
    · obtain ⟨k, rfl⟩ : ∃ k, T = k + 101 := ⟨T - 101, by omega⟩
      rw [exaCoef_sum_add_101]
      linarith [ih k (by omega) (by omega)]

/-- **"The optimal solution is `x* = -1`."**  For every horizon `T ≥ 1`,
`x* = -1` minimizes `Σ_{t=1}^T f_t` over `F = [-1, 1]`.

Source: arXiv:1904.03590v4, §3, Example 3.2. -/
theorem exa_optimal {T : ℕ} (hT : 1 ≤ T) {x : Vec 1}
    (hx : x ∈ Set.Icc (fun _ => -1) (fun _ => 1)) :
    ∑ t ∈ Icc 1 T, exaSetup.f t (fun _ => -1) ≤ ∑ t ∈ Icc 1 T, exaSetup.f t x := by
  simp only [exaSetup, ← Finset.sum_mul]
  exact mul_le_mul_of_nonneg_left (hx.1 0) (exaCoef_sum_pos hT).le

/-- The hypotheses of `exa_optimal` are satisfiable. -/
example : 1 ≤ 1 ∧ (fun _ => -1 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨le_rfl, fun _ => le_rfl, fun _ => by norm_num⟩

/-! ### The red inequality fails -/

/-- **The first ("red") inequality of Reddi et al. is false.**  Replacing
`1/(1 - β_{1,t})` by `1/(1 - β₁)` in the first term (3.1) of Lemma 3.1 does not
give an upper bound: for the run of Example 3.2, which satisfies every
hypothesis of Theorem A (`isOnlineConvex_exa`), at `T = 2`, `x* = -1`,

  `Σ_{t=1}^2 √v̂_t/(2α_t(1-β_{1,t})) ((x_t - x*)² - (x_{t+1} - x*)²)`
  `  > Σ_{t=1}^2 √v̂_t/(2α_t(1-β₁)) ((x_t - x*)² - (x_{t+1} - x*)²)`,

since the `t = 2` difference is negative (`exa_sign_two`) and `β_{1,2} < β₁`.

Source: arXiv:1904.03590v4, §3, the paragraph before Example 3.2. -/
theorem not_red_ineq :
    ¬ (∑ i, ∑ t ∈ Icc 1 2, Real.sqrt (exaSetup.vhat amsgradRule t i) /
          (2 * exaSetup.α t * (1 - exaSetup.β₁ t)) *
          ((exaSetup.x amsgradRule t i + 1) ^ 2 - (exaSetup.x amsgradRule (t + 1) i + 1) ^ 2)
      ≤ ∑ i, ∑ t ∈ Icc 1 2, Real.sqrt (exaSetup.vhat amsgradRule t i) /
          (2 * exaSetup.α t * (1 - exaSetup.β₁ 1)) *
          ((exaSetup.x amsgradRule t i + 1) ^ 2 - (exaSetup.x amsgradRule (t + 1) i + 1) ^ 2)) := by
  have hv : 1020.1 ≤ exaSetup.vhat amsgradRule 2 0 := by
    show 1020.1 ≤ (exaSetup.step amsgradRule 2 (exaSetup.state amsgradRule 1)).vhat 0
    simp only [Setup.step, amsgradRule, Pi.sup_apply, exa_state_one.2.1]
    exact le_sup_left
  have hs : 0 < Real.sqrt (exaSetup.vhat amsgradRule 2 0) := Real.sqrt_pos.2 (by linarith)
  have hα : exaSetup.α 2 = 0.001 / Real.sqrt 2 := by simp [exaSetup]
  have hβ1 : exaSetup.β₁ 1 = 0.9 := by simp [exaSetup]
  have hβ2 : exaSetup.β₁ 2 = 0.9 * 0.001 := by simp [exaSetup]
  have hΔ := exa_sign_two
  have hI : Icc 1 2 = {1, 2} := by decide
  rw [Fin.sum_univ_one, Fin.sum_univ_one, hI, Finset.sum_pair (by norm_num),
    Finset.sum_pair (by norm_num), not_le, add_lt_add_iff_left, hα, hβ1, hβ2]
  set A := Real.sqrt (exaSetup.vhat amsgradRule 2 0)
  set Δ := (exaSetup.x amsgradRule 2 0 + 1) ^ 2 - (exaSetup.x amsgradRule (2 + 1) 0 + 1) ^ 2
  have hr : 0 < Real.sqrt 2 := by positivity
  rw [div_mul_eq_mul_div, div_mul_eq_mul_div, div_lt_div_iff₀ (by positivity) (by positivity)]
  have hk : 0 < 0.001 / Real.sqrt 2 := by positivity
  nlinarith [mul_neg_of_neg_of_pos (mul_neg_of_pos_of_neg hs hΔ) hk]

end AMSGrad
end Transformer
