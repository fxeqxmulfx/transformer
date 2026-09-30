/-
# DASH — proved training convergence of the safeguarded optimizer

User-requested extension of arXiv:2602.02016v2, §2–4. The full parameter
trajectory, actual EMA histories and grafting are included. The finite PI
and NDB budgets stay fixed as training time tends to infinity. The results
concern a fixed full-gradient objective in real arithmetic, not the paper's
stochastic training experiments or lower-precision implementation.
-/

import Transformer.DASH.Section2_TrainingModels
import Transformer.Optimization.Stationarity
import Transformer.Optimization.StrongConvergence
import Transformer.Optimization.Quadratic

open scoped Topology
open Filter

noncomputable section

namespace Transformer.DASH

open Optimization

variable {m n : ℕ} {κ : Type} [Fintype κ] [Nonempty κ]

/-- Every corrected DASH learning trajectory has vanishing actual
full-gradient Frobenius norm for a lower-bounded smooth objective.
The result is independent of the finite solver's accuracy: sufficient
alignment and bounded length are checked on its actual grafted output.
Source: training correction to arXiv:2602.02016v2, §2–4. -/
theorem safeguardedDash_stationarity (σ smooth β ν μ ε lower : ℝ)
    (startsL : κ → Fin m → ℝ) (startsR : κ → Fin n → ℝ) (piSteps rootSteps : ℕ)
    (f : MatrixSpace m n → ℝ) (initial : MatrixSpace m n)
    (hf : SmoothObjective f smooth) (hL : 0 < smooth) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => frobeniusNorm (toMatrix (gradient f
      (safeguardedDashRun σ smooth β ν μ ε startsL startsR piSteps rootSteps f initial t).2)))
      atTop (𝓝 0) := by
  have h := safeguardedRun_gradient_tendsto_zero f σ smooth lower
    (fun state _ g => dashTrainingCandidate β ν μ ε startsL startsR piSteps rootSteps state g)
    initialTrainingState initial hf hL hσ hσ' hlower
  simpa only [safeguardedDashRun, ← fromMatrix_norm, fromMatrix_toMatrix] using h

/-- Stationarity assumptions hold for a nonconstant matrix quadratic,
arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- The corrected full DASH training loss converges to a finite value.
Without strong convexity this value is not asserted to be a global
minimum. Source: training correction to arXiv:2602.02016v2, §2–4. -/
theorem safeguardedDash_loss_convergence (σ smooth β ν μ ε lower : ℝ)
    (startsL : κ → Fin m → ℝ) (startsR : κ → Fin n → ℝ) (piSteps rootSteps : ℕ)
    (f : MatrixSpace m n → ℝ) (initial : MatrixSpace m n)
    (hf : SmoothObjective f smooth) (hL : 0 < smooth) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hlower : ∀ x, lower ≤ f x) :
    ∃ value : ℝ, Tendsto (fun t : ℕ =>
      f (safeguardedDashRun σ smooth β ν μ ε startsL startsR piSteps rootSteps f initial t).2)
      atTop (𝓝 value) :=
  (safeguardedRun_loss_convergence f σ smooth lower
    (fun state _ g => dashTrainingCandidate β ν μ ε startsL startsR piSteps rootSteps state g)
    initialTrainingState initial hf hL hσ hσ' hlower).1

/-- Finite-loss convergence assumptions are satisfiable,
arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (∀ x : MatrixSpace 1 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- Corrected DASH converges to a global minimizer on a strongly convex
smooth fixed objective, with a geometric loss-gap bound. All finite
matrix computations and Adam grafting remain inside the actual training
recurrence; their convergence is not an input hypothesis.
Source: training correction to arXiv:2602.02016v2, §2–4. -/
theorem safeguardedDash_minimum_convergence (σ smooth β ν μ ε strong : ℝ)
    (startsL : κ → Fin m → ℝ) (startsR : κ → Fin n → ℝ) (piSteps rootSteps : ℕ)
    (f : MatrixSpace m n → ℝ) (initial star : MatrixSpace m n)
    (hf : SmoothObjective f smooth) (hstrong : StrongLowerModel f strong)
    (hL : 0 < smooth) (hs : 0 < strong) (hσ : 0 < σ) (hσ' : σ ≤ 1)
    (hrate : σ ^ 2 * strong ≤ smooth) (hstar : gradient f star = 0) :
    (∀ x, f star ≤ f x) ∧
      Tendsto (fun t : ℕ =>
        (safeguardedDashRun σ smooth β ν μ ε startsL startsR piSteps rootSteps f initial t).2)
        atTop (𝓝 star) ∧
      Tendsto (fun t : ℕ =>
        f (safeguardedDashRun σ smooth β ν μ ε startsL startsR piSteps rootSteps f initial t).2)
        atTop (𝓝 (f star)) ∧
      (∀ t : ℕ, f
        (safeguardedDashRun σ smooth β ν μ ε startsL startsR piSteps rootSteps f initial t).2 -
        f star ≤ (1 - σ ^ 2 * strong / smooth) ^ t * (f initial - f star)) := by
  obtain ⟨hmin, hweights, hloss⟩ := safeguardedRun_strong_convergence f σ smooth strong
    (fun state _ g => dashTrainingCandidate β ν μ ε startsL startsR piSteps rootSteps state g)
    initialTrainingState initial star hf hstrong hL hs hσ hσ' hrate hstar
  exact ⟨hmin, hweights, hloss, fun t => safeguardedRun_strong_rate f σ smooth strong
    (fun state _ g => dashTrainingCandidate β ν μ ε startsL startsR piSteps rootSteps state g)
    initialTrainingState initial star hf hstrong hL hs hσ hσ' hrate hstar t⟩

/-- Minimum-convergence assumptions hold on genuine matrix training,
arXiv:2602.02016v2, §2–4, training extension. -/
example : SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    StrongLowerModel (energy : MatrixSpace 1 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (1 / 2 : ℝ) ^ 2 * 1 ≤ 1 ∧ gradient (energy : MatrixSpace 1 1 → ℝ) 0 = 0 := by
  exact ⟨energy_models.1, energy_models.2, by norm_num, by norm_num, by norm_num,
    by norm_num, by rw [energy_gradient]; rfl⟩

end Transformer.DASH
