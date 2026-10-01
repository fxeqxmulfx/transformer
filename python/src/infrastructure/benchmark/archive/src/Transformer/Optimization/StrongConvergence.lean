/-
# Convergence of corrected training to a strongly convex minimum

User-requested extension of arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. The conclusions concern training time.
Strong convexity and the smooth quadratic model are additional objective
assumptions; neither manuscript asserts them for neural-network losses.
-/

import Transformer.Optimization.Descent
import Transformer.Optimization.StrongConvexity
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Real.Sqrt

open scoped InnerProductSpace Topology
open Filter

noncomputable section

namespace Transformer.Optimization

variable {E S : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Geometric objective-gap bound for the actual stateful corrected
training run. No assumption is made about the optimizer's auxiliary
states or candidate accuracy. Source: training extension of
arXiv:2502.16982, §2.1–2.2, and arXiv:2602.02016v2, §2–4. -/
theorem safeguardedRun_strong_rate (f : E → ℝ) (σ L μ : ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial star : E)
    (hf : SmoothObjective f L) (hstrong : StrongLowerModel f μ)
    (hL : 0 < L) (hμ : 0 < μ) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hrate : σ ^ 2 * μ ≤ L) (hstar : gradient f star = 0) (T : ℕ) :
    f (safeguardedRun σ L f propose initialState initial T).2 - f star ≤
      (1 - σ ^ 2 * μ / L) ^ T * (f initial - f star) := by
  let q := 1 - σ ^ 2 * μ / L
  let δ := σ ^ 2 / (2 * L)
  have hq : 0 ≤ q := by
    dsimp only [q]
    have h := (div_le_one hL).mpr hrate
    linarith
  have hδ : 0 ≤ δ := by dsimp only [δ]; positivity
  induction T with
  | zero => simp [safeguardedRun]
  | succ T ih =>
    let x := (safeguardedRun σ L f propose initialState initial T).2
    have hstep := safeguardedRun_descent f σ L propose initialState initial hf hL hσ hσ' T
    have hPL := (strong_gap_bounds f μ star x hstrong hμ hstar).2
    have hPLδ := mul_le_mul_of_nonneg_left hPL hδ
    calc
      _ ≤ f x - f star - δ * ‖gradient f x‖ ^ 2 := by linarith
      _ ≤ f x - f star - δ * (2 * μ * (f x - f star)) := by linarith
      _ = q * (f x - f star) := by dsimp only [q, δ]; field_simp
      _ ≤ q * (q ^ T * (f initial - f star)) := mul_le_mul_of_nonneg_left ih hq
      _ = _ := by dsimp only [q]; rw [pow_succ]; ring

/-- A nonconstant objective satisfies all rate hypotheses,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective quadratic 1 ∧ StrongLowerModel quadratic 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (1 / 2 : ℝ) ^ 2 * 1 ≤ 1 ∧ gradient quadratic 0 = 0 := by
  exact ⟨quadratic_smooth, quadratic_strong, by norm_num, by norm_num,
    by norm_num, by norm_num, by rw [quadratic_gradient]; rfl⟩

/-- The corrected training weights converge to a global minimizer, and
their loss converges to the minimum value, under the stated strongly
convex objective assumptions. This is stronger than stationarity and is
not a claim about general nonconvex neural-network training.
Source: training extension of arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. -/
theorem safeguardedRun_strong_convergence (f : E → ℝ) (σ L μ : ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial star : E)
    (hf : SmoothObjective f L) (hstrong : StrongLowerModel f μ)
    (hL : 0 < L) (hμ : 0 < μ) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hrate : σ ^ 2 * μ ≤ L) (hstar : gradient f star = 0) :
    (∀ x, f star ≤ f x) ∧
      Tendsto (fun t : ℕ => (safeguardedRun σ L f propose initialState initial t).2)
        atTop (𝓝 star) ∧
      Tendsto (fun t : ℕ => f (safeguardedRun σ L f propose initialState initial t).2)
        atTop (𝓝 (f star)) := by
  let x := fun t => (safeguardedRun σ L f propose initialState initial t).2
  let q := 1 - σ ^ 2 * μ / L
  have hmin : ∀ y, f star ≤ f y := by
    intro y
    have h := (strong_gap_bounds f μ star y hstrong hμ hstar).1
    have hn : 0 ≤ μ / 2 * ‖y - star‖ ^ 2 := by positivity
    linarith
  have hq : 0 ≤ q := by
    have h := (div_le_one hL).mpr hrate
    dsimp only [q]
    linarith
  have hq' : q < 1 := by
    have h : 0 < σ ^ 2 * μ / L := by positivity
    dsimp only [q]
    linarith
  have hpow := tendsto_pow_atTop_nhds_zero_of_abs_lt_one (by rwa [abs_of_nonneg hq])
  have hgap : Tendsto (fun t => f (x t) - f star) atTop (𝓝 0) := by
    apply squeeze_zero (fun t => sub_nonneg.mpr (hmin _))
      (fun t => safeguardedRun_strong_rate f σ L μ propose initialState initial star
        hf hstrong hL hμ hσ hσ' hrate hstar t)
    simpa only [zero_mul] using hpow.mul_const (f initial - f star)
  have hdist : Tendsto (fun t => ‖x t - star‖ ^ 2) atTop (𝓝 0) := by
    apply squeeze_zero (fun t => sq_nonneg _) (g := fun t => 2 / μ * (f (x t) - f star))
    · intro t
      have h := (strong_gap_bounds f μ star (x t) hstrong hμ hstar).1
      have hm := mul_le_mul_of_nonneg_left h (show 0 ≤ 2 / μ by positivity)
      have hc : 2 / μ * (μ / 2) = 1 := by field_simp
      nlinarith
    · simpa only [mul_zero] using hgap.const_mul (2 / μ)
  have hnorm := (Real.continuous_sqrt.tendsto 0).comp hdist
  refine ⟨hmin, tendsto_iff_norm_sub_tendsto_zero.mpr ?_, ?_⟩
  · simpa only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] using hnorm
  · simpa only [sub_add_cancel, zero_add] using hgap.add_const (f star)

/-- All minimum-convergence assumptions hold on a nonconstant quadratic,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective quadratic 1 ∧ StrongLowerModel quadratic 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (1 / 2 : ℝ) ^ 2 * 1 ≤ 1 ∧ gradient quadratic 0 = 0 := by
  exact ⟨quadratic_smooth, quadratic_strong, by norm_num, by norm_num,
    by norm_num, by norm_num, by rw [quadratic_gradient]; rfl⟩

end Transformer.Optimization
