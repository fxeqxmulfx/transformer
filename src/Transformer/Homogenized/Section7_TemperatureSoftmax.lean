/-
# Genuine temperature derivatives of normalized softmax weights

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
part (b) of `eq:s_mu_small_beta_clean`. The original score row has
the form `β a`, with `a k = ⟨A x, y_k⟩`. Differentiation is with
respect to the actual inverse temperature, keeping the scores fixed.

The genuine first derivative retains the normalized softmax weight
and is bounded by `2 M w_k` when every original score is bounded by
`M`. Differentiating the genuine derivative gives a second derivative
bounded by `6 M² w_k`. Both estimates hold at every real temperature
and retain their original coordinate weight, so summing them costs
no factor from the number of tokens. They require no probabilistic
assumption and precede the source's Gaussian expectation and Taylor
remainder estimate.
-/

import Transformer.Homogenized.SoftmaxDerivatives
import Mathlib.Analysis.Calculus.Deriv.Prod

open Real
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual temperature derivative is the original softmax weight
times its centered score. Source: arXiv:2604.01978v1, §7,
the displayed formula for `w_k'` in the proof of `eq:s_mu_small_beta_clean`. -/
theorem hasDerivAt_softmaxWeight_temperature {n : ℕ} (hn : 0 < n)
    (a : Idx n → ℝ) (j : Idx n) (t : ℝ) :
    HasDerivAt (fun r : ℝ => Perspective.softmaxWeight (fun k => r * a k) j)
      (Perspective.softmaxWeight (fun k => t * a k) j *
        (a j - ∑ k : Idx n, Perspective.softmaxWeight (fun l => t * a l) k * a k)) t := by
  have hl : HasDerivAt (fun r : ℝ => fun k => r * a k) a t := by
    apply hasDerivAt_pi.mpr
    intro k
    simpa only [id_eq, one_mul] using (hasDerivAt_id t).mul_const (a k)
  have h := (hasFDerivAt_softmaxWeight hn (fun k => t * a k) j).comp_hasDerivAt t hl
  simpa only [Function.comp_def, smul_apply, sub_apply, sum_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul] using h

/-- A normalized softmax average preserves a uniform scalar bound.
Source: arXiv:2604.01978v1, §7, normalization in the `w_k'` estimate. -/
theorem abs_sum_softmaxWeight_mul_le {n : ℕ} (hn : 0 < n) (u a : Idx n → ℝ)
    (M : ℝ) (ha : ∀ k, |a k| ≤ M) :
    |∑ k : Idx n, Perspective.softmaxWeight u k * a k| ≤ M := by
  calc
    _ ≤ ∑ k : Idx n, |Perspective.softmaxWeight u k * a k| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k : Idx n, Perspective.softmaxWeight u k * M := by
      apply Finset.sum_le_sum
      intro k hk
      rw [abs_mul, abs_of_nonneg (Perspective.softmaxWeight_nonneg u k)]
      exact mul_le_mul_of_nonneg_left (ha k) (Perspective.softmaxWeight_nonneg u k)
    _ = M := by rw [← Finset.sum_mul, Perspective.sum_softmaxWeight hn u, one_mul]

/-- The actual centered score is bounded independently of token count.
Source: arXiv:2604.01978v1, §7, the pointwise temperature derivative bound. -/
theorem abs_softmax_centered_score_le {n : ℕ} (hn : 0 < n) (u a : Idx n → ℝ)
    (M : ℝ) (ha : ∀ k, |a k| ≤ M) (j : Idx n) :
    |a j - ∑ k : Idx n, Perspective.softmaxWeight u k * a k| ≤ 2 * M := by
  exact (abs_sub _ _).trans ((add_le_add (ha j) (abs_sum_softmaxWeight_mul_le hn u a M ha)).trans_eq (by ring))

/-- The genuine first temperature derivative keeps its original
coordinate weight. Source: arXiv:2604.01978v1, §7, the `w_k'` bound. -/
theorem abs_deriv_softmaxWeight_temperature_le {n : ℕ} (hn : 0 < n)
    (a : Idx n → ℝ) (M : ℝ) (ha : ∀ k, |a k| ≤ M) (j : Idx n) (t : ℝ) :
    |deriv (fun r : ℝ => Perspective.softmaxWeight (fun k => r * a k) j) t| ≤
      2 * M * Perspective.softmaxWeight (fun k => t * a k) j := by
  rw [(hasDerivAt_softmaxWeight_temperature hn a j t).deriv, abs_mul,
    abs_of_nonneg (Perspective.softmaxWeight_nonneg _ j)]
  exact (mul_le_mul_of_nonneg_left (abs_softmax_centered_score_le hn _ a M ha j)
    (Perspective.softmaxWeight_nonneg _ j)).trans_eq (by ring)

/-- Differentiating the actual first temperature derivative gives its
genuine centered-score second derivative. Source: arXiv:2604.01978v1,
§7, the second differentiation in `eq:s_mu_small_beta_clean`. -/
theorem hasDerivAt_deriv_softmaxWeight_temperature {n : ℕ} (hn : 0 < n)
    (a : Idx n → ℝ) (j : Idx n) (t : ℝ) :
    HasDerivAt (deriv (fun r : ℝ => Perspective.softmaxWeight (fun k => r * a k) j))
      (Perspective.softmaxWeight (fun k => t * a k) j *
          (a j - ∑ k : Idx n, Perspective.softmaxWeight (fun l => t * a l) k * a k) ^ 2 -
        Perspective.softmaxWeight (fun k => t * a k) j *
          ∑ k : Idx n, deriv (fun r : ℝ => Perspective.softmaxWeight (fun l => r * a l) k) t * a k) t := by
  have he : deriv (fun r : ℝ => Perspective.softmaxWeight (fun k => r * a k) j) =
      fun r => Perspective.softmaxWeight (fun k => r * a k) j *
        (a j - ∑ k : Idx n, Perspective.softmaxWeight (fun l => r * a l) k * a k) :=
    funext fun r => (hasDerivAt_softmaxWeight_temperature hn a j r).deriv
  rw [he]
  have hs := HasDerivAt.fun_sum (u := Finset.univ) fun k hk =>
    (hasDerivAt_softmaxWeight_temperature hn a k t).mul_const (a k)
  have h := (hasDerivAt_softmaxWeight_temperature hn a j t).fun_mul
    ((hasDerivAt_const t (a j)).fun_sub hs)
  simp_rw [(hasDerivAt_softmaxWeight_temperature hn a _ t).deriv]
  convert h using 1
  ring

/-- The actual derivative of the softmax score average has a uniform
quadratic score bound. Source: arXiv:2604.01978v1, §7, the second
temperature differentiation in `eq:s_mu_small_beta_clean`. -/
theorem abs_sum_deriv_softmaxWeight_temperature_mul_le {n : ℕ} (hn : 0 < n)
    (a : Idx n → ℝ) (M : ℝ) (ha : ∀ k, |a k| ≤ M) (t : ℝ) :
    |∑ k : Idx n, deriv (fun r : ℝ => Perspective.softmaxWeight (fun l => r * a l) k) t * a k| ≤
      2 * M ^ 2 := by
  have hM : 0 ≤ M := (abs_nonneg (a ⟨0, hn⟩)).trans (ha ⟨0, hn⟩)
  calc
    _ ≤ ∑ k : Idx n, |deriv (fun r : ℝ => Perspective.softmaxWeight (fun l => r * a l) k) t * a k| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k : Idx n, 2 * M ^ 2 * Perspective.softmaxWeight (fun l => t * a l) k := by
      apply Finset.sum_le_sum
      intro k hk
      rw [abs_mul]
      convert mul_le_mul (abs_deriv_softmaxWeight_temperature_le hn a M ha k t) (ha k)
        (abs_nonneg _) (mul_nonneg (mul_nonneg (by norm_num) hM)
          (Perspective.softmaxWeight_nonneg _ k)) using 1
      ring
    _ = _ := by rw [← Finset.mul_sum, Perspective.sum_softmaxWeight hn _, mul_one]

/-- **The genuine second temperature derivative is bounded by
`6 M² w_k`, uniformly in temperature and token count.**
Source: arXiv:2604.01978v1, §7, the second derivative estimate in
the proof of `eq:s_mu_small_beta_clean`, with an explicit constant. -/
theorem abs_deriv_deriv_softmaxWeight_temperature_le {n : ℕ} (hn : 0 < n)
    (a : Idx n → ℝ) (M : ℝ) (ha : ∀ k, |a k| ≤ M) (j : Idx n) (t : ℝ) :
    |deriv (deriv (fun r : ℝ => Perspective.softmaxWeight (fun k => r * a k) j)) t| ≤
      6 * M ^ 2 * Perspective.softmaxWeight (fun k => t * a k) j := by
  have hM : 0 ≤ M := (abs_nonneg (a j)).trans (ha j)
  have hc := abs_softmax_centered_score_le hn (fun k => t * a k) a M ha j
  have hs := abs_sum_deriv_softmaxWeight_temperature_mul_le hn a M ha t
  have hc2 : (a j - ∑ k : Idx n, Perspective.softmaxWeight (fun l => t * a l) k * a k) ^ 2 ≤
      4 * M ^ 2 := by
    have h := pow_le_pow_left₀ (abs_nonneg _) hc 2
    simpa only [sq_abs, mul_pow, show (2 : ℝ) ^ 2 = 4 by norm_num] using h
  rw [(hasDerivAt_deriv_softmaxWeight_temperature hn a j t).deriv]
  apply (abs_sub _ _).trans
  rw [abs_mul, abs_mul, abs_of_nonneg (Perspective.softmaxWeight_nonneg _ j),
    abs_of_nonneg (sq_nonneg _)]
  have hw := Perspective.softmaxWeight_nonneg (fun k => t * a k) j
  nlinarith [mul_le_mul_of_nonneg_left hc2 hw, mul_le_mul_of_nonneg_left hs hw]

/-- Each original softmax weight is smooth in the actual temperature.
Source: arXiv:2604.01978v1, §7, the temperature Taylor expansion
in the proof of `eq:s_mu_small_beta_clean`. -/
theorem contDiff_softmaxWeight_temperature {n : ℕ} (hn : 0 < n)
    (a : Idx n → ℝ) (j : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun r : ℝ => Perspective.softmaxWeight (fun k => r * a k) j) := by
  have hl : ContDiff ℝ (⊤ : ℕ∞) (fun r : ℝ => fun k : Idx n => r * a k) :=
    contDiff_pi.mpr fun k => by fun_prop
  exact (contDiff_softmaxWeight hn j).comp hl

/-- Nonconstant genuine scores, positive temperature and a nonzero
actual first derivative jointly realize all derivative and bound premises. -/
example : ∃ a : Idx 2 → ℝ, (0 : ℕ) < 2 ∧ (∀ k, |a k| ≤ 1) ∧
    deriv (fun r : ℝ => Perspective.softmaxWeight (fun k => r * a k) (0 : Idx 2)) 0 = 1 / 4 := by
  refine ⟨![1, 0], by decide, ?_, ?_⟩
  · intro k
    fin_cases k <;> norm_num
  · rw [(hasDerivAt_softmaxWeight_temperature (by decide) _ _ _).deriv]
    norm_num [Perspective.softmaxWeight, Fin.sum_univ_two]

end Transformer.Homogenized
