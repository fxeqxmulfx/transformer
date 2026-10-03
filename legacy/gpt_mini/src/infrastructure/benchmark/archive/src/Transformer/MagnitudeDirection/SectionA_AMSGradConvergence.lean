/-
# Concrete safeguarded AMSGradMD convergence

User-requested extension of arXiv:2606.25971v2, §3.1 and Appendix A,
Algorithm 2, using the actual AMSGrad buffers of arXiv:1904.03590v4,
Algorithm 1 and §6, for the direction and both raw-gain optimizers.
The full fused-step guard and representation repair are explicit added
algorithm changes. No convergence claim for unmodified AMSGradMD follows
by inheriting the theorem for ordinary AMSGrad.
-/

import Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks
import Transformer.MagnitudeDirection.SectionA_TrainingMinimum

open scoped Topology
open Filter

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {m n : ℕ}

/-- The complete corrected AMSGradMD training run starts all three sets
of first/second/maximum histories at zero, computes actual factor gradients,
updates every history, and checks the resulting full fused step.
Source: arXiv:2606.25971v2, Appendix A, AMSGradMD training correction;
arXiv:1904.03590v4, Algorithm 1 and §6. -/
def safeguardedAMSGradMDRun (σ L c ε β β₂ etaW etaG : ℝ)
    (f : MatrixSpace m n → ℝ) (initial : FullState m n) :
    ℕ → (AMSGradMDMemory m n × FullState m n) × MatrixSpace m n :=
  safeguardedFullRun σ L c f (amsgradMDProposal ε β β₂ etaW etaG c)
    (zeroAMSGradMDMemory m n) initial

/-- All three actual running maxima are nondecreasing, even when the
candidate fused weight is rejected. This checks real optimizer-memory
evolution rather than a state-free surrogate.
Source: arXiv:1904.03590v4, Algorithm 1; arXiv:2606.25971v2, Appendix A,
AMSGradMD training correction. -/
theorem safeguardedAMSGradMD_maxima_mono (σ L c ε β β₂ etaW etaG : ℝ)
    (f : MatrixSpace m n → ℝ) (initial : FullState m n) (t : ℕ) :
    let old := (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).1.1
    let next := (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial (t + 1)).1.1
    (∀ p, old.1.maximum p ≤ next.1.maximum p) ∧
      (∀ i, old.2.1.maximum i ≤ next.2.1.maximum i) ∧
      (∀ j, old.2.2.maximum j ≤ next.2.2.maximum j) := by
  exact ⟨fun p => le_max_left _ _, fun i => le_max_left _ _, fun j => le_max_left _ _⟩

/-- Full concrete AMSGradMD convergence: valid nonzero sphere storage at
every finite time, full objective gradient tending to zero, and finite
loss limit. The guard supplies descent, so the proof needs no independent
momentum or preconditioner alignment assumption.
Source: arXiv:2606.25971v2, Appendix A, AMSGradMD training correction;
arXiv:1904.03590v4, Algorithm 1 and §6. -/
theorem safeguardedAMSGradMD_convergence (σ L c ε β β₂ etaW etaG lower : ℝ)
    (f : MatrixSpace m n → ℝ) (initial : FullState m n)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hc : 0 < c) (hD : frobeniusNorm (fullDirection softplus initial) = c)
    (hlower : ∀ x, lower ≤ f x) :
    (∀ t : ℕ, (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).2 ≠ 0 ∧
      frobeniusNorm (fullDirection softplus
        (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).1.2) = c) ∧
    Tendsto (fun t : ℕ => ‖gradient f
      (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).2‖) atTop (𝓝 0) ∧
    (∃ value : ℝ, Tendsto (fun t : ℕ =>
      f (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).2) atTop (𝓝 value)) := by
  exact safeguardedMD_valid_convergence σ L c lower f (amsgradMDProposal ε β β₂ etaW etaG c)
    (zeroAMSGradMDMemory m n) initial hf hL hσ hσ' hc hD hlower

/-- All concrete convergence assumptions hold on nonconstant matrix
training, arXiv:2606.25971v2, Appendix A, AMSGradMD extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧
    frobeniusNorm (fullDirection softplus (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ))) = 1 ∧
    (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  refine ⟨energy_models.1, by norm_num, by norm_num, by norm_num, by norm_num, ?_,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩
  rw [unitGainState_direction, frobeniusNorm_eq_sqrt]
  norm_num

/-- Under strong convexity the actual corrected AMSGradMD weights converge
to the global minimizer, with loss convergence and the geometric gap bound.
Source: arXiv:2606.25971v2, Appendix A, AMSGradMD training correction;
arXiv:1904.03590v4, Algorithm 1 and §6. -/
theorem safeguardedAMSGradMD_minimum_convergence (σ L c ε β β₂ etaW etaG strong : ℝ)
    (f : MatrixSpace m n → ℝ) (initial : FullState m n) (star : MatrixSpace m n)
    (hf : SmoothObjective f L) (hstrong : StrongLowerModel f strong)
    (hL : 0 < L) (hs : 0 < strong) (hσ : 0 < σ) (hσ' : σ ≤ 1 / 2)
    (hrate : σ ^ 2 * strong ≤ L) (hstar : gradient f star = 0) :
    (∀ x, f star ≤ f x) ∧
      Tendsto (fun t : ℕ =>
        (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).2) atTop (𝓝 star) ∧
      Tendsto (fun t : ℕ =>
        f (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).2) atTop (𝓝 (f star)) ∧
      (∀ t : ℕ, f (safeguardedAMSGradMDRun σ L c ε β β₂ etaW etaG f initial t).2 - f star ≤
        (1 - σ ^ 2 * strong / L) ^ t * (f (fromMatrix initial.weight) - f star)) := by
  exact safeguardedMD_minimum_convergence σ L c strong f (amsgradMDProposal ε β β₂ etaW etaG c)
    (zeroAMSGradMDMemory m n) initial star hf hstrong hL hs hσ hσ' hrate hstar

/-- Strong-convergence assumptions are jointly satisfiable,
arXiv:2606.25971v2, Appendix A, AMSGradMD training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    StrongLowerModel (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧
    (1 / 4 : ℝ) ^ 2 * 1 ≤ 1 ∧ gradient (energy : MatrixSpace 1 1 → ℝ) 0 = 0 := by
  exact ⟨energy_models.1, energy_models.2, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by rw [energy_gradient]; rfl⟩

end Transformer.MagnitudeDirection
