/-
# Convergence of corrected MD weights to a strongly convex minimum

User-requested training extension of arXiv:2606.25971v2, §3.1 and
Appendix A, Algorithm 2. Strong convexity is an additional loss assumption,
not an assertion about general neural-network objectives. The limits
concern fused weights and loss; redundant gains and optimizer memories
are not asserted to converge.
-/

import Transformer.MagnitudeDirection.SectionA_TrainingConvergence
import Transformer.Optimization.StrongConvergence

open scoped Topology
open Filter

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {m n : ℕ} {S : Type*}

/-- A smooth strongly convex loss gives convergence of the corrected MD
weights to its global minimizer, loss convergence and a geometric gap
bound. The complete base/gain proposals and memory updates are retained;
the fused-step safeguard supplies the needed descent conditions.
Source: training correction to arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem safeguardedMD_minimum_convergence (σ L c strong : ℝ)
    (f : MatrixSpace m n → ℝ) (propose : FullProposal S m n)
    (initialMemory : S) (initial : FullState m n) (star : MatrixSpace m n)
    (hf : SmoothObjective f L) (hstrong : StrongLowerModel f strong)
    (hL : 0 < L) (hs : 0 < strong) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hrate : σ ^ 2 * strong ≤ L) (hstar : gradient f star = 0) :
    (∀ x, f star ≤ f x) ∧
      Tendsto (fun t : ℕ => (safeguardedFullRun σ L c f propose initialMemory initial t).2)
        atTop (𝓝 star) ∧
      Tendsto (fun t : ℕ => f (safeguardedFullRun σ L c f propose initialMemory initial t).2)
        atTop (𝓝 (f star)) ∧
      (∀ t : ℕ, f (safeguardedFullRun σ L c f propose initialMemory initial t).2 - f star ≤
        (1 - σ ^ 2 * strong / L) ^ t * (f (fromMatrix initial.weight) - f star)) := by
  obtain ⟨hmin, hx, hloss⟩ := safeguardedRun_strong_convergence f σ L strong
    (trainingCandidate σ L c propose) (initialMemory, initial) (fromMatrix initial.weight)
    star hf hstrong hL hs hσ (by linarith) hrate hstar
  exact ⟨hmin, hx, hloss, fun t => safeguardedRun_strong_rate f σ L strong
    (trainingCandidate σ L c propose) (initialMemory, initial) (fromMatrix initial.weight)
    star hf hstrong hL hs hσ (by linarith) hrate hstar t⟩

/-- The minimum-convergence assumptions hold on a nonconstant matrix loss,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    StrongLowerModel (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧
    (1 / 4 : ℝ) ^ 2 * 1 ≤ 1 ∧ gradient (energy : MatrixSpace 1 1 → ℝ) 0 = 0 := by
  exact ⟨energy_models.1, energy_models.2, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by rw [energy_gradient]; rfl⟩

/-- Full valid-MD convergence statement: every finite iterate has a
nonzero fused weight and a fixed-norm recovered direction, while the
actual full gradient vanishes and the actual loss has a finite limit.
All assumptions concern the objective, initialization and positive guard
parameters. Source: training correction to arXiv:2606.25971v2, Appendix A. -/
theorem safeguardedMD_valid_convergence (σ L c lower : ℝ) (f : MatrixSpace m n → ℝ)
    (propose : FullProposal S m n) (initialMemory : S) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hc : 0 < c) (hD : frobeniusNorm (fullDirection softplus initial) = c)
    (hlower : ∀ x, lower ≤ f x) :
    (∀ t : ℕ, (safeguardedFullRun σ L c f propose initialMemory initial t).2 ≠ 0 ∧
      frobeniusNorm (fullDirection softplus
        (safeguardedFullRun σ L c f propose initialMemory initial t).1.2) = c) ∧
    Tendsto (fun t : ℕ => ‖gradient f
      (safeguardedFullRun σ L c f propose initialMemory initial t).2‖) atTop (𝓝 0) ∧
    (∃ value : ℝ, Tendsto (fun t : ℕ =>
      f (safeguardedFullRun σ L c f propose initialMemory initial t).2) atTop (𝓝 value)) := by
  refine ⟨fun t => ⟨?_, ?_⟩,
    safeguardedMD_stationarity σ L c lower f propose initialMemory initial hf hL hσ hσ' hlower,
    safeguardedMD_loss_convergence σ L c lower f propose initialMemory initial hf hL hσ hσ' hlower⟩
  · exact safeguardedFullRun_ne_zero σ L c f propose initialMemory initial hσ hσ' hL
      (fullState_ne_zero_of_norm c initial hc hD) t
  · exact safeguardedFullRun_direction_norm σ L c f propose initialMemory initial
      hσ hσ' hL hc hD t

/-- Every valid-convergence hypothesis holds on nonconstant matrix training,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    frobeniusNorm (fullDirection softplus (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 1 ∧
    (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  refine ⟨energy_models.1, by norm_num, by norm_num, by norm_num, by norm_num, ?_,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩
  rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
  norm_num

/-- Zero can be the limiting minimizer while every finite iterate remains
nonzero and on its positive sphere. Thus the correction does not need an
assumption excluding a zero optimum from the loss.
Source: extension of arXiv:2606.25971v2, §4.1.3 and Appendix A, Algorithm 2. -/
theorem safeguardedMD_zero_minimum (c : ℝ) (propose : FullProposal S m n)
    (initialMemory : S) (initial : FullState m n)
    (hc : 0 < c) (hD : frobeniusNorm (fullDirection softplus initial) = c) :
    (∀ t : ℕ, (safeguardedFullRun (1 / 4) 1 c energy propose initialMemory initial t).2 ≠ 0 ∧
      frobeniusNorm (fullDirection softplus
        (safeguardedFullRun (1 / 4) 1 c energy propose initialMemory initial t).1.2) = c) ∧
    Tendsto (fun t : ℕ =>
      (safeguardedFullRun (1 / 4) 1 c energy propose initialMemory initial t).2) atTop (𝓝 0) := by
  refine ⟨fun t => ⟨?_, ?_⟩, ?_⟩
  · exact safeguardedFullRun_ne_zero (1 / 4) 1 c energy propose initialMemory initial
      (by norm_num) (by norm_num) (by norm_num) (fullState_ne_zero_of_norm c initial hc hD) t
  · exact safeguardedFullRun_direction_norm (1 / 4) 1 c energy propose initialMemory initial
      (by norm_num) (by norm_num) (by norm_num) hc hD t
  · exact (safeguardedMD_minimum_convergence (1 / 4) 1 c 1 energy propose initialMemory
      initial 0 energy_models.1 energy_models.2 (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by rw [energy_gradient]; rfl)).2.1

/-- Positive-sphere initialization for the zero-minimum theorem exists,
arXiv:2606.25971v2, Appendix A, training extension. -/
example : (0 : ℝ) < 1 ∧
    frobeniusNorm (fullDirection softplus (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 1 := by
  rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
  norm_num

end Transformer.MagnitudeDirection
