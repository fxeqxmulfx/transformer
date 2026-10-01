/-
# Vanishing errors under a strict contraction

Analytic helper for the AMSGradW training extension of
arXiv:1904.03590v4, Algorithm 1 and §4. The later proof derives both
the recurrence and its vanishing forcing from the actual buffers.
-/

import Transformer.AMSGradW.Basic

open scoped Topology
open Filter

namespace Transformer.AMSGradW

/-- A nonnegative error with a contractive recurrence and forcing tending
to zero tends to zero. Source: analytic helper for arXiv:1904.03590v4,
§4, AMSGradW extension; no convergence conclusion is assumed of the error. -/
theorem contraction_forcing_tendsto_zero (q : ℝ) (u δ : ℕ → ℝ)
    (hq : 0 ≤ q) (hq' : q < 1) (hu : ∀ n, 0 ≤ u n)
    (hδ : Tendsto δ atTop (𝓝 0)) (hrec : ∀ n, u (n + 1) ≤ q * u n + δ n) :
    Tendsto u atTop (𝓝 0) := by
  apply tendsto_order.mpr
  constructor
  · intro a ha
    exact Eventually.of_forall fun n => ha.trans_le (hu n)
  · intro e he
    have hc : 0 < e * (1 - q) / 2 := by positivity
    obtain ⟨N, hN⟩ := eventually_atTop.mp ((tendsto_order.mp hδ).2 _ hc)
    have hb : ∀ k, u (N + k) ≤ q ^ k * u N + e / 2 := by
      intro k
      induction k with
      | zero => simp only [Nat.add_zero, pow_zero, one_mul]; linarith
      | succ k ih =>
        have hr := hrec (N + k)
        have hd := hN (N + k) (by omega)
        have hm := mul_le_mul_of_nonneg_left ih hq
        rw [pow_succ]
        simp only [Nat.add_succ] at *
        nlinarith
    have hp : Tendsto (fun k : ℕ => q ^ k * u N) atTop (𝓝 0) := by
      simpa only [zero_mul] using
        (tendsto_pow_atTop_nhds_zero_of_lt_one hq hq').mul_const (u N)
    obtain ⟨K, hK⟩ := eventually_atTop.mp ((tendsto_order.mp hp).2 (e / 2) (by linarith))
    filter_upwards [eventually_ge_atTop (N + K)] with n hn
    have hk : K ≤ n - N := by omega
    have hh := hb (n - N)
    have heq : N + (n - N) = n := by omega
    rw [heq] at hh
    linarith [hK (n - N) hk]

/-- The helper's assumptions hold for a nonzero geometric sequence.
Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ n : ℕ, 0 ≤ (1 / 2 : ℝ) ^ n) ∧
    Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (𝓝 0) ∧
    (∀ n : ℕ, (1 / 2 : ℝ) ^ (n + 1) ≤ (1 / 2) * (1 / 2) ^ n + 0) := by
  refine ⟨by norm_num, by norm_num, fun n => pow_nonneg (by norm_num) n,
    tendsto_const_nhds, fun n => ?_⟩
  rw [pow_succ]
  ring_nf
  exact le_rfl

end Transformer.AMSGradW
