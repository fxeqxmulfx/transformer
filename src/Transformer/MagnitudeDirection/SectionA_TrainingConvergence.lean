/-
# MD training convergence to stationarity

User-requested deterministic training extension of arXiv:2606.25971v2,
§3.1 and Appendix A, Algorithm 2. These are limits in training time,
with the original base/gain optimizer computations retained as proposals.
The complete fused-step check is an explicit algorithm change. Smoothness
and a lower bound describe the loss; no convergence, bounded-gradient or
optimizer-alignment premise describes the iterates.
-/

import Transformer.MagnitudeDirection.SectionA_TrainingInvariants
import Transformer.Optimization.Stationarity
import Transformer.Optimization.Quadratic

open scoped BigOperators Topology
open Filter

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {m n : ℕ} {S : Type*}

/-- Actual corrected fused loss decreases by at least `σ²/(2L)` times
the squared full-gradient Frobenius norm. It checks the entire MD update,
including gains and projection. Source: training correction to
arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_descent (σ L c : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2) (t : ℕ) :
    f (safeguardedFullRun σ L c f propose initialMemory initial (t + 1)).2 ≤
      f (safeguardedFullRun σ L c f propose initialMemory initial t).2 -
        σ ^ 2 / (2 * L) *
          ‖gradient f (safeguardedFullRun σ L c f propose initialMemory initial t).2‖ ^ 2 := by
  exact safeguardedRun_descent f σ L _ _ _ hf hL hσ (by linarith) t

/-- A nonconstant Frobenius quadratic satisfies every descent assumption,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 :=
  ⟨energy_models.1, by norm_num, by norm_num, by norm_num⟩

/-- A finite-training bound on the sum of actual squared gradients.
No claim about minibatch noise or rounded arithmetic is included.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_gradient_budget (σ L c : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2) (T : ℕ) :
    σ ^ 2 / (2 * L) * ∑ t ∈ Finset.range T,
        ‖gradient f (safeguardedFullRun σ L c f propose initialMemory initial t).2‖ ^ 2 ≤
      f (fromMatrix initial.weight) -
        f (safeguardedFullRun σ L c f propose initialMemory initial T).2 := by
  exact safeguardedRun_gradient_budget f σ L _ _ _ hf hL hσ (by linarith) T

/-- The finite-budget assumptions are satisfiable,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 :=
  ⟨energy_models.1, by norm_num, by norm_num, by norm_num⟩

/-- Squared true gradients are summable along the complete corrected MD
recurrence for a fixed smooth lower-bounded objective. The base and gain
callbacks can have arbitrary evolving states. Source: training correction
to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_gradients_summable (σ L c lower : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hlower : ∀ x, lower ≤ f x) :
    Summable (fun t : ℕ =>
      ‖gradient f (safeguardedFullRun σ L c f propose initialMemory initial t).2‖ ^ 2) := by
  exact safeguardedRun_gradients_summable f σ L lower _ _ _ hf hL hσ (by linarith) hlower

/-- A genuine nonconstant lower-bounded objective satisfies these hypotheses,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- The full-gradient Frobenius norm tends to zero, along the entire
training sequence rather than a subsequence. This is actual fixed-loss
learning convergence for the corrected MD algorithm.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_stationarity (σ L c lower : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ =>
      ‖gradient f (safeguardedFullRun σ L c f propose initialMemory initial t).2‖)
      atTop (𝓝 0) := by
  exact safeguardedRun_gradient_tendsto_zero f σ L lower _ _ _
    hf hL hσ (by linarith) hlower

/-- Stationarity hypotheses hold on nonconstant matrix training,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- The actual fused loss values converge to a finite value. A global
minimum is not asserted for a general nonconvex loss.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_loss_convergence (σ L c lower : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hlower : ∀ x, lower ≤ f x) :
    ∃ value : ℝ, Tendsto (fun t : ℕ =>
      f (safeguardedFullRun σ L c f propose initialMemory initial t).2) atTop (𝓝 value) := by
  exact (safeguardedRun_loss_convergence f σ L lower _ _ _
    hf hL hσ (by linarith) hlower).1

/-- Finite-loss convergence assumptions are jointly satisfiable,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- The same vanishing-gradient result applies to the stored weight
actually exposed to the model. No separate idealized sequence is used.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_stored_stationarity (σ L c lower : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => ‖gradient f (fromMatrix
      (safeguardedFullRun σ L c f propose initialMemory initial t).1.2.weight)‖)
      atTop (𝓝 0) := by
  simpa only [safeguardedFullRun_storage σ L c f propose initialMemory initial hσ'] using
    safeguardedMD_stationarity σ L c lower f propose initialMemory initial hf hL hσ hσ' hlower

/-- Stored-gradient convergence hypotheses are satisfiable,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.MagnitudeDirection
