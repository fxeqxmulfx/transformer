/-
# The number of modes of a Gaussian KDE — `lem:moments-p`, the covariance

§2.2 of arXiv:2412.09080v3, `lem:moments-p`: the three entries of the
covariance `Σ_t` of `(G(t), G'(t))`, along a sequence `(β_k, t_k)` with
`β_k → ∞` and `t_k²/β_k → 0` (see `Section2_MomentsP.lean` for the reading of
`~`).  All three are as the source writes them.

Each quotient of `Section5_CovAsymp.lean` is `A + b·Ψ`: `A` a continuous
function of `(1/β, t²/β)` equal to `1` at the origin, `b` a bounded function of
`t` (`t²e^{-t²/2}` and its kin, once `β ≥ 1`), and `Ψ` a continuous function of
`(1/β, t²/β)` vanishing at the origin — the product-of-means term, a power of
`1/β` smaller than the second moment.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:moments-p`; §5.2.
-/

import Transformer.Modes.Section2_MomentsP
import Transformer.Modes.Section5_CovAsymp

open Real Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-- A continuous function of `(1/β_k, t_k²/β_k)` tends to its value at the
origin. -/
theorem tendsto_comp_inv_sq_div {B t : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    (ht : Tendsto (fun k => t k ^ 2 / B k) atTop (𝓝 0)) {G : ℝ × ℝ → ℝ} {c : ℝ}
    (hG : ContinuousAt G (0, 0)) (hc : G (0, 0) = c) :
    Tendsto (fun k => G ((B k)⁻¹, t k ^ 2 / B k)) atTop (𝓝 c) :=
  hc ▸ hG.tendsto.comp ((tendsto_inv_atTop_zero.comp hB).prodMk_nhds ht)

/-- The hypotheses of `tendsto_comp_inv_sq_div` are satisfiable. -/
example : Tendsto (fun k : ℕ => (fun p : ℝ × ℝ => p.1) (((k : ℝ) + 1)⁻¹, 0 ^ 2 / ((k : ℝ) + 1)))
    atTop (𝓝 0) :=
  tendsto_comp_inv_sq_div tendsto_natSucc_atTop (by simp) continuousAt_fst rfl

/-- **Lemma (lem:moments-p), `Σ_{t,11}`.**  `Var G(t) ~ 2^{-5/2}β^{-3/2}e^{-t²/2} · 2`.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:moments-p`. -/
theorem moments_p_var_fst {B t : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    (ht : Tendsto (fun k => t k ^ 2 / B k) atTop (𝓝 0)) :
    (fun k => sqMeanG (B k) (t k) - meanG (B k) (t k) ^ 2)
      ~[atTop] fun k => 2 ^ (-(5 : ℝ) / 2) * momentScale (B k) (t k) * 2 := by
  have hA := tendsto_comp_inv_sq_div hB ht (c := 1)
    (G := fun p => Real.exp (p.2 * (1 / (2 + p.1)) / 2)
    * (1 + p.2 * (1 / (2 + p.1))) * (2 * √2) * (1 / (2 + p.1)) * √(1 / (2 + p.1)))
    (by fun_prop (disch := norm_num)) (by norm_num; field_simp)
  have hΨ := tendsto_comp_inv_sq_div hB ht (c := 0) (G := fun p => -(2 * √2)
    * Real.exp (p.2 * (1 / (1 + p.1)) / 2) ^ 2 * (1 / (1 + p.1)) ^ 3 * p.1 * √p.1)
    (by fun_prop (disch := norm_num)) (by simp)
  dsimp only at hA hΨ
  refine isEquivalent_of_eq_mul (tendsto_add_mul_of_bdd hA (C := 2)
    (b := fun k => t k ^ 2 * Real.exp (-(t k ^ 2) / 2))
    (Eventually.of_forall fun k => by
      rw [abs_of_nonneg (by positivity)]; exact sq_mul_exp_neg_half_sq_le _) hΨ) ?_
  filter_upwards [hB.eventually_gt_atTop 0] with k hk
  rw [varG_eq_mul hk, momentScale]
  ring

/-- `|e^{-t²/2}(1 - t² + ε)| ≤ 4` for `0 ≤ ε ≤ 1`. -/
theorem abs_exp_mul_one_sub_sq_add_le {ε : ℝ} (h0 : 0 ≤ ε) (h1 : ε ≤ 1) (t : ℝ) :
    |Real.exp (-(t ^ 2) / 2) * (1 - t ^ 2 + ε)| ≤ 4 := by
  have hg := Real.exp_pos (-(t ^ 2) / 2)
  have hg1 : Real.exp (-(t ^ 2) / 2) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg t])
  have h2 := sq_mul_exp_neg_half_sq_le t
  rw [abs_le]
  constructor <;> nlinarith [mul_le_mul_of_nonneg_left h1 hg.le, mul_nonneg hg.le h0]

/-- The hypotheses of `abs_exp_mul_one_sub_sq_add_le` are satisfiable. -/
example : |Real.exp (-(0 ^ 2) / 2) * (1 - 0 ^ 2 + 0)| ≤ 4 :=
  abs_exp_mul_one_sub_sq_add_le le_rfl zero_le_one 0

/-- `|e^{-t²/2}(1 + ε - t²)²| ≤ 24` for `0 ≤ ε ≤ 1`. -/
theorem abs_exp_mul_one_add_sub_sq_sq_le {ε : ℝ} (h0 : 0 ≤ ε) (h1 : ε ≤ 1) (t : ℝ) :
    |Real.exp (-(t ^ 2) / 2) * (1 + ε - t ^ 2) ^ 2| ≤ 24 := by
  have hg := Real.exp_pos (-(t ^ 2) / 2)
  have hg1 : Real.exp (-(t ^ 2) / 2) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg t])
  have h4 := pow_four_mul_exp_neg_half_sq_le t
  have hsq : (1 + ε - t ^ 2) ^ 2 ≤ 8 + 2 * t ^ 4 := by
    nlinarith [sq_nonneg (1 + ε + t ^ 2), mul_nonneg h0 h0]
  rw [abs_of_nonneg (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_left hsq hg.le]

/-- The hypotheses of `abs_exp_mul_one_add_sub_sq_sq_le` are satisfiable. -/
example : |Real.exp (-(0 ^ 2) / 2) * (1 + 0 - 0 ^ 2) ^ 2| ≤ 24 :=
  abs_exp_mul_one_add_sub_sq_sq_le le_rfl zero_le_one 0

/-- **Lemma (lem:moments-p), `Σ_{t,12}`.**
`Cov(G(t), G'(t)) ~ 2^{-5/2}β^{-3/2}e^{-t²/2} · (-t)`.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:moments-p`. -/
theorem moments_p_cov {B t : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    (ht : Tendsto (fun k => t k ^ 2 / B k) atTop (𝓝 0)) :
    (fun k => mulMeanGG' (B k) (t k) - meanG (B k) (t k) * meanG' (B k) (t k))
      ~[atTop] fun k => 2 ^ (-(5 : ℝ) / 2) * momentScale (B k) (t k) * (-t k) := by
  have hA := tendsto_comp_inv_sq_div hB ht (c := 1)
    (G := fun p => Real.exp (p.2 * (1 / (2 + p.1)) / 2)
    * (1 - p.1 / 2 + p.2 / 2 - p.1 ^ 2 / 2) * (8 * √2) * (1 / (2 + p.1)) ^ 3
    * √(1 / (2 + p.1))) (by fun_prop (disch := norm_num)) (by norm_num; field_simp)
  have hΨ := tendsto_comp_inv_sq_div hB ht (c := 0) (G := fun p => 4 * √2
    * Real.exp (p.2 * (1 / (1 + p.1)) / 2) ^ 2 * (1 / (1 + p.1)) ^ 4 * p.1 * √p.1)
    (by fun_prop (disch := norm_num)) (by simp)
  dsimp only at hA hΨ
  refine isEquivalent_of_eq_mul (tendsto_add_mul_of_bdd hA (C := 4)
    (b := fun k => Real.exp (-(t k ^ 2) / 2) * (1 - t k ^ 2 + (B k)⁻¹))
    ?_ hΨ) ?_
  · filter_upwards [hB.eventually_ge_atTop 1] with k hk
    exact abs_exp_mul_one_sub_sq_add_le (by positivity) (inv_le_one_of_one_le₀ hk) _
  filter_upwards [hB.eventually_gt_atTop 0] with k hk
  rw [covG_eq_mul hk, momentScale]
  ring

/-- **Lemma (lem:moments-p), `Σ_{t,22}`.**  `Var G'(t) ~ 2^{-5/2}β^{-3/2}e^{-t²/2} · 3β`.

Source: arXiv:2412.09080v3, `lem:moments-p`, `eq:moments-p`. -/
theorem moments_p_var_snd {B t : ℕ → ℝ} (hB : Tendsto B atTop atTop)
    (ht : Tendsto (fun k => t k ^ 2 / B k) atTop (𝓝 0)) :
    (fun k => sqMeanG' (B k) (t k) - meanG' (B k) (t k) ^ 2)
      ~[atTop] fun k => 2 ^ (-(5 : ℝ) / 2) * momentScale (B k) (t k) * (3 * B k) := by
  have hA := tendsto_comp_inv_sq_div hB ht (c := 1)
    (G := fun p => Real.exp (p.2 * (1 / (2 + p.1)) / 2)
    * (1 + (p.2 + 5 * p.1) / 3 + (p.2 ^ 2 - 2 * p.2 * p.1 + 15 * p.1 ^ 2) / 12
      - (p.2 * p.1 ^ 2 - 3 * p.1 ^ 3) / 6 + p.1 ^ 4 / 12)
    * (16 * √2) * (1 / (2 + p.1)) ^ 4 * √(1 / (2 + p.1)))
    (by fun_prop (disch := norm_num)) (by norm_num; field_simp)
  have hΨ := tendsto_comp_inv_sq_div hB ht (c := 0) (G := fun p => -(4 * √2 / 3)
    * Real.exp (p.2 * (1 / (1 + p.1)) / 2) ^ 2 * (1 / (1 + p.1)) ^ 5 * p.1 ^ 2 * √p.1)
    (by fun_prop (disch := norm_num)) (by simp)
  dsimp only at hA hΨ
  refine isEquivalent_of_eq_mul (tendsto_add_mul_of_bdd hA (C := 24)
    (b := fun k => Real.exp (-(t k ^ 2) / 2) * (1 + (B k)⁻¹ - t k ^ 2) ^ 2)
    ?_ hΨ) ?_
  · filter_upwards [hB.eventually_ge_atTop 1] with k hk
    exact abs_exp_mul_one_add_sub_sq_sq_le (by positivity) (inv_le_one_of_one_le₀ hk) _
  filter_upwards [hB.eventually_gt_atTop 0] with k hk
  rw [varG'_eq_mul hk, momentScale]
  ring

/-- The hypotheses of the three entries are satisfiable: `β_k = k + 1`, `t_k = 0`. -/
example : (fun k : ℕ => sqMeanG ((k : ℝ) + 1) 0 - meanG ((k : ℝ) + 1) 0 ^ 2)
      ~[atTop] fun k => 2 ^ (-(5 : ℝ) / 2) * momentScale ((k : ℝ) + 1) 0 * 2 :=
  moments_p_var_fst (t := fun _ => 0) tendsto_natSucc_atTop (by simp)

end Modes
end Transformer
