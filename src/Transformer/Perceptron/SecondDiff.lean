/-
# Second differences and a discrete maximum principle

The two analytic facts behind the injectivity of `μ ↦ f_B^μ` in
`rem: general-attention` (`TransformMap`), stated without reference to
attention.

* `Σ_i (e^{t y_i} + e^{-t y_i} - 2) = t² + O(t⁴)` for `y` on the unit sphere
  (`abs_sum_exp_sub_le`): the second differences of `z ↦ e^{z·y}` along the
  coordinate axes add up to `t² ‖y‖² e^{z·y}`, the discrete form of
  `Δ e^{z·y} = ‖y‖² e^{z·y}`, which the source uses.
* A continuous `G` that is `≤ 0` on the unit sphere and whose second
  differences dominate `t² G - O(t⁴)` is `≤ 0` on the closed unit ball
  (`le_zero_of_secondDiff_ge`): the maximum principle for `Δ G ≥ G`, with
  second differences in place of the Laplacian, so that nothing has to be
  differentiated under an integral.

Source: arXiv:2601.21366v2, `rem: general-attention` (the Dirichlet problem
for `Δ - 1` on the ellipsoid, and `Δ e^{z·y} = ‖y‖² e^{z·y}`).
-/

import Transformer.Basic
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Topology.Order.Compact

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Second differences of the exponential -/

/-- `|e^s + e^{-s} - 2 - s²| ≤ s⁴` for `|s| ≤ 1`. -/
theorem abs_exp_add_exp_neg_sub_le {s : ℝ} (hs : |s| ≤ 1) :
    |exp s + exp (-s) - 2 - s ^ 2| ≤ s ^ 4 := by
  have h1 := Real.exp_bound hs (n := 4) (by norm_num)
  have h2 := Real.exp_bound (x := -s) (by rwa [abs_neg]) (n := 4) (by norm_num)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial, abs_neg,
    (by decide : Even 4).pow_abs] at h1 h2
  norm_num at h1 h2
  rw [abs_le] at h1 h2 ⊢
  constructor <;> nlinarith [pow_two_nonneg (s ^ 2)]

/-- The hypothesis of `abs_exp_add_exp_neg_sub_le` is satisfiable: `s = 0`. -/
example : |(0 : ℝ)| ≤ 1 := by simp

/-- **`Σ_i (e^{t y_i} + e^{-t y_i} - 2) = t² + O(t⁴)` on the sphere**: the
second differences of `e^{t·y}` along the coordinate axes add up to
`t² ‖y‖² = t²`, up to `t⁴`. -/
theorem abs_sum_exp_sub_le {y : EucSpace d} (hy : ‖y‖ = 1) {t : ℝ} (ht : |t| ≤ 1) :
    |∑ i, (exp (t * y i) + exp (-(t * y i)) - 2) - t ^ 2| ≤ t ^ 4 := by
  have hsum : ∑ i, y i ^ 2 = 1 := by
    rw [← EuclideanSpace.real_norm_sq_eq, hy, one_pow]
  have hyi : ∀ i, y i ^ 2 ≤ 1 := fun i =>
    hsum ▸ Finset.single_le_sum (fun j _ => sq_nonneg (y j)) (Finset.mem_univ i)
  have ht2 : t ^ 2 = ∑ i, (t * y i) ^ 2 := by
    simp_rw [mul_pow]
    rw [← Finset.mul_sum, hsum, mul_one]
  rw [ht2, ← Finset.sum_sub_distrib]
  calc |∑ i, (exp (t * y i) + exp (-(t * y i)) - 2 - (t * y i) ^ 2)|
      ≤ ∑ i, |exp (t * y i) + exp (-(t * y i)) - 2 - (t * y i) ^ 2| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, (t * y i) ^ 4 := Finset.sum_le_sum fun i _ => abs_exp_add_exp_neg_sub_le <| by
        rw [abs_mul]
        exact (mul_le_of_le_one_left (abs_nonneg _) ht).trans
          ((sq_le_one_iff_abs_le_one _).1 (hyi i))
    _ ≤ ∑ i, t ^ 4 * y i ^ 2 := Finset.sum_le_sum fun i _ => by
        rw [mul_pow]
        exact mul_le_mul_of_nonneg_left (by nlinarith [hyi i, sq_nonneg (y i)]) (by positivity)
    _ = t ^ 4 := by rw [← Finset.mul_sum, hsum, mul_one]

/-- The hypotheses of `abs_sum_exp_sub_le` are satisfiable: `y = e₀ ∈ 𝕊^0`,
`t = 0`. -/
example : ‖(EuclideanSpace.single 0 1 : EucSpace 1)‖ = 1 ∧ |(0 : ℝ)| ≤ 1 := by simp

/-! ### A discrete maximum principle -/

/-- **A discrete maximum principle for `Δ G ≥ G`.**  If `G` is continuous,
`≤ 0` on the unit sphere, and its second differences along some
`w_1, …, w_d` satisfy
`Σ_i [G(x + t w_i) + G(x - t w_i) - 2 G(x)] ≥ t² G(x) - t⁴ S(x)` for `|t| ≤ 1`,
then `G ≤ 0` on the closed unit ball.  At an interior positive maximum the
left side is `≤ 0`, while the right side is `> 0` once `0 < t² < G/S`. -/
theorem le_zero_of_secondDiff_ge {G S : EucSpace d → ℝ} (hG : Continuous G)
    (w : Fin d → EucSpace d)
    (hdiff : ∀ x t, |t| ≤ 1 →
      t ^ 2 * G x - t ^ 4 * S x ≤ ∑ i, (G (x + t • w i) + G (x - t • w i) - 2 * G x))
    (hsphere : ∀ x, ‖x‖ = 1 → G x ≤ 0) {x : EucSpace d} (hx : ‖x‖ ≤ 1) : G x ≤ 0 := by
  obtain ⟨x₀, hx₀, hmax⟩ := (isCompact_closedBall (0 : EucSpace d) 1).exists_isMaxOn
    ⟨0, Metric.mem_closedBall_self zero_le_one⟩ hG.continuousOn
  rw [isMaxOn_iff] at hmax
  refine (hmax x (mem_closedBall_zero_iff.2 hx)).trans (not_lt.1 fun hM => ?_)
  have hx₀1 : ‖x₀‖ < 1 := (mem_closedBall_zero_iff.1 hx₀).lt_of_ne fun h =>
    (hsphere x₀ h).not_gt hM
  have e1 : ∀ᶠ t in 𝓝 (0 : ℝ), |t| < 1 :=
    continuous_abs.continuousAt.eventually_lt continuousAt_const (by simp)
  have e2 : ∀ᶠ t in 𝓝 (0 : ℝ), t ^ 2 * S x₀ < G x₀ :=
    (Continuous.continuousAt (by fun_prop)).eventually_lt continuousAt_const (by simpa using hM)
  have e3 : ∀ᶠ t in 𝓝 (0 : ℝ), ∀ i, ‖x₀ + t • w i‖ < 1 ∧ ‖x₀ - t • w i‖ < 1 :=
    eventually_all.2 fun i =>
      ((Continuous.continuousAt (by fun_prop)).eventually_lt continuousAt_const
        (by simpa using hx₀1)).and
      ((Continuous.continuousAt (by fun_prop)).eventually_lt continuousAt_const
        (by simpa using hx₀1))
  obtain ⟨t, ⟨⟨ht1, htS⟩, htw⟩, ht0⟩ :=
    ((((e1.and e2).and e3).filter_mono nhdsWithin_le_nhds).and
      (self_mem_nhdsWithin : {0}ᶜ ∈ 𝓝[≠] (0 : ℝ))).exists
  have ht0' : t ≠ 0 := ht0
  have hsum : ∑ i, (G (x₀ + t • w i) + G (x₀ - t • w i) - 2 * G x₀) ≤ 0 :=
    Finset.sum_nonpos fun i _ => by
      have h1 := hmax _ (mem_closedBall_zero_iff.2 (htw i).1.le)
      have h2 := hmax _ (mem_closedBall_zero_iff.2 (htw i).2.le)
      linarith
  have ht2 : 0 < t ^ 2 := by positivity
  nlinarith [hdiff x₀ t ht1.le, mul_pos ht2 (sub_pos.2 htS)]

/-- The hypotheses of `le_zero_of_secondDiff_ge` are satisfiable: `G = S = 0`
and `w = 0`. -/
example : ∃ (G S : EucSpace 1 → ℝ) (w : Fin 1 → EucSpace 1), Continuous G ∧
    (∀ x t, |t| ≤ 1 →
      t ^ 2 * G x - t ^ 4 * S x ≤ ∑ i, (G (x + t • w i) + G (x - t • w i) - 2 * G x)) ∧
    ∀ x, ‖x‖ = 1 → G x ≤ 0 :=
  ⟨0, 0, 0, continuous_const, fun _ _ _ => by simp, fun _ _ => le_rfl⟩

end Perceptron
end Transformer
