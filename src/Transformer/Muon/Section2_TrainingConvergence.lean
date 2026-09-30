/-
# Muon — proved convergence of the corrected learning algorithm

User-requested training extension of arXiv:2502.16982, §2.1–2.2.
These limits are in the number of training steps, with the source's
five Newton–Schulz steps fixed. The loss is fixed and deterministic;
stochastic minibatch training and floating-point rounding are not covered.
-/

import Transformer.Muon.Section2_TrainingModels
import Transformer.Optimization.Stationarity
import Transformer.Optimization.StrongConvergence
import Transformer.Optimization.Quadratic

open scoped Topology
open Filter

noncomputable section

namespace Transformer.Muon

open Optimization

variable {a b : ℕ}

/-- For every initial matrix, momentum coefficient and decay coefficient,
the corrected Muon run has vanishing full-gradient Frobenius norm on a
lower-bounded smooth objective. No alignment or exact-polar hypothesis
is imposed: the guard proves the needed direction conditions itself.
Source: training correction to arXiv:2502.16982, §2.1–2.2. -/
theorem safeguardedMuon_stationarity (σ L μ wd lower : ℝ)
    (f : MatrixSpace a b → ℝ) (initial : MatrixSpace a b)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => DASH.frobeniusNorm
      (toMatrix (gradient f (safeguardedMuonRun σ L μ wd f initial t).2)))
      atTop (𝓝 0) := by
  have h := safeguardedRun_gradient_tendsto_zero f σ L lower
    (muonTrainingCandidate μ wd) 0 initial hf hL hσ hσ' hlower
  simpa only [safeguardedMuonRun, ← fromMatrix_norm, fromMatrix_toMatrix] using h

/-- A nonconstant matrix objective satisfies all stationarity assumptions,
arXiv:2502.16982, §2.1–2.2, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- The corrected Muon's actual loss values converge to a finite value.
The limit is not asserted to be a global minimum without a convexity
assumption. Source: training correction to arXiv:2502.16982, §2.1–2.2. -/
theorem safeguardedMuon_loss_convergence (σ L μ wd lower : ℝ)
    (f : MatrixSpace a b → ℝ) (initial : MatrixSpace a b)
    (hf : SmoothObjective f L) (hL : 0 < L) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hlower : ∀ x, lower ≤ f x) :
    ∃ value : ℝ, Tendsto (fun t : ℕ =>
      f (safeguardedMuonRun σ L μ wd f initial t).2) atTop (𝓝 value) :=
  (safeguardedRun_loss_convergence f σ L lower (muonTrainingCandidate μ wd)
    0 initial hf hL hσ hσ' hlower).1

/-- The loss-limit assumptions hold on a nonconstant matrix quadratic,
arXiv:2502.16982, §2.1–2.2, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- With a strongly convex smooth objective and an existing stationary
point, the corrected Muon weights converge to that global minimizer,
and their objective gap has the stated geometric bound. This result
retains the actual printed five-step approximation; its matrix-solver
budget does not grow with training time.
Source: training correction to arXiv:2502.16982, §2.1–2.2. -/
theorem safeguardedMuon_minimum_convergence (σ L μ wd strong : ℝ)
    (f : MatrixSpace a b → ℝ) (initial star : MatrixSpace a b)
    (hf : SmoothObjective f L) (hstrong : StrongLowerModel f strong)
    (hL : 0 < L) (hs : 0 < strong) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hrate : σ ^ 2 * strong ≤ L) (hstar : gradient f star = 0) :
    (∀ x, f star ≤ f x) ∧
      Tendsto (fun t : ℕ => (safeguardedMuonRun σ L μ wd f initial t).2) atTop (𝓝 star) ∧
      Tendsto (fun t : ℕ => f (safeguardedMuonRun σ L μ wd f initial t).2)
        atTop (𝓝 (f star)) ∧
      (∀ t : ℕ, f (safeguardedMuonRun σ L μ wd f initial t).2 - f star ≤
        (1 - σ ^ 2 * strong / L) ^ t * (f initial - f star)) := by
  obtain ⟨hmin, hweights, hloss⟩ := safeguardedRun_strong_convergence f σ L strong
    (muonTrainingCandidate μ wd) 0 initial star hf hstrong hL hs hσ hσ' hrate hstar
  exact ⟨hmin, hweights, hloss, fun t => safeguardedRun_strong_rate f σ L strong
    (muonTrainingCandidate μ wd) 0 initial star hf hstrong hL hs hσ hσ' hrate hstar t⟩

/-- The minimum-convergence assumptions are satisfiable on genuine
nonconstant matrix training, arXiv:2502.16982, §2.1–2.2, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    StrongLowerModel (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (1 / 2 : ℝ) ^ 2 * 1 ≤ 1 ∧ gradient (energy : MatrixSpace 1 1 → ℝ) 0 = 0 := by
  exact ⟨energy_models.1, energy_models.2, by norm_num, by norm_num, by norm_num,
    by norm_num, by rw [energy_gradient]; rfl⟩

end Transformer.Muon
