/-
# Training convergence for a lower-bounded smooth objective

Additional deterministic convergence theorem for the corrected algorithms
of arXiv:2502.16982, §2.1–2.2, and arXiv:2602.02016v2, §2–4.
This is a learning-time limit, not a limit in inverse-root iterations.
The full gradient norms vanish, not merely a selected subsequence.
It does not assert convergence of nonconvex weights to a global minimum.
-/

import Transformer.Optimization.Descent
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Analysis.Real.Sqrt

open scoped InnerProductSpace BigOperators Topology
open Filter

noncomputable section

namespace Transformer.Optimization

variable {E S : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- All squared full gradients of the actual corrected training run are
summable for a fixed lower-bounded smooth objective. No reliability or
convergence premise is imposed on the candidate optimizer.
Source: training extension of arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. -/
theorem safeguardedRun_gradients_summable (f : E → ℝ) (σ L lower : ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial : E)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hlower : ∀ x, lower ≤ f x) :
    Summable (fun t : ℕ =>
      ‖gradient f (safeguardedRun σ L f propose initialState initial t).2‖ ^ 2) := by
  have hc : 0 < σ ^ 2 / (2 * L) := by positivity
  apply summable_of_sum_range_le (fun t => sq_nonneg _)
    (c := (f initial - lower) / (σ ^ 2 / (2 * L)))
  intro T
  apply (le_div_iff₀ hc).mpr
  have hb := safeguardedRun_gradient_budget f σ L propose initialState initial hf hL hσ hσ' T
  have hl := hlower (safeguardedRun σ L f propose initialState initial T).2
  nlinarith

/-- A nonconstant lower-bounded loss satisfies these assumptions,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (∀ x : ℝ, 0 ≤ quadratic x) := by
  refine ⟨quadratic_smooth, by norm_num, by norm_num, by norm_num, ?_⟩
  intro x
  exact div_nonneg (sq_nonneg x) (by norm_num)

/-- The actual full gradient norm tends to zero as the number of training
steps tends to infinity. This is stationarity of the corrected learning
algorithm, under stated loss assumptions, rather than convergence of its
matrix approximation to an exact root.
Source: training extension of arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. -/
theorem safeguardedRun_gradient_tendsto_zero (f : E → ℝ) (σ L lower : ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial : E)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ =>
      ‖gradient f (safeguardedRun σ L f propose initialState initial t).2‖) atTop (𝓝 0) := by
  have h := (safeguardedRun_gradients_summable f σ L lower propose initialState initial
    hf hL hσ hσ' hlower).tendsto_atTop_zero
  have hs := (Real.continuous_sqrt.tendsto 0).comp h
  simpa only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] using hs

/-- Nonconstant stationary-limit assumptions are satisfiable,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (∀ x : ℝ, 0 ≤ quadratic x) := by
  refine ⟨quadratic_smooth, by norm_num, by norm_num, by norm_num, ?_⟩
  intro x
  exact div_nonneg (sq_nonneg x) (by norm_num)

/-- Loss values along the actual corrected training run converge to a
finite infimum. Differentiability verifies that every vector called a
gradient along this run is the objective's genuine derivative.
Source: training extension of arXiv:2502.16982, §2.1–2.2, and
arXiv:2602.02016v2, §2–4. -/
theorem safeguardedRun_loss_convergence (f : E → ℝ) (σ L lower : ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial : E)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hlower : ∀ x, lower ≤ f x) :
    (∃ value : ℝ, Tendsto (fun t : ℕ =>
      f (safeguardedRun σ L f propose initialState initial t).2) atTop (𝓝 value)) ∧
    (∀ t : ℕ, HasGradientAt f
      (gradient f (safeguardedRun σ L f propose initialState initial t).2)
      (safeguardedRun σ L f propose initialState initial t).2) := by
  have hmono : Antitone (fun t : ℕ =>
      f (safeguardedRun σ L f propose initialState initial t).2) := by
    apply antitone_nat_of_succ_le
    intro t
    have h := safeguardedRun_descent f σ L propose initialState initial hf hL hσ hσ' t
    have hc : 0 ≤ σ ^ 2 / (2 * L) *
        ‖gradient f (safeguardedRun σ L f propose initialState initial t).2‖ ^ 2 := by positivity
    linarith
  have hbound : BddBelow (Set.range (fun t : ℕ =>
      f (safeguardedRun σ L f propose initialState initial t).2)) := by
    refine ⟨lower, ?_⟩
    rintro value ⟨t, rfl⟩
    exact hlower _
  exact ⟨⟨_, tendsto_atTop_ciInf hmono hbound⟩, fun t => (hf.1 _).hasGradientAt⟩

/-- The finite-loss-limit assumptions are satisfiable,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (∀ x : ℝ, 0 ≤ quadratic x) := by
  refine ⟨quadratic_smooth, by norm_num, by norm_num, by norm_num, ?_⟩
  intro x
  exact div_nonneg (sq_nonneg x) (by norm_num)

end Transformer.Optimization
