/-
# AdaFisher — finite dissipation budget of the full actual trajectory

arXiv:2405.16397v3, §3.4 and Appendix A.2, deterministic extension.
The budget is proved by telescoping actual state transitions and is
not the unestablished stochastic budget of the source's Proposition 3.4.
-/

import Transformer.AdaFisher.Section3_TrainingEnergy

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

open Optimization

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- The two positive quantities paid for by an actual step: squared
displacement and corrected-momentum error. Source: arXiv:2405.16397v3,
§3.4, deterministic training extension. -/
def trainingDissipation (η β δ : ℝ) (f : TrainingSpace a b → ℝ)
    (s next : OptimizerState a b) : ℝ :=
  δ / (4 * η) * ‖trainingVelocity η β δ next‖ ^ 2 +
    η * (1 - β ^ 2) / δ * ‖trainingError β f s‖ ^ 2

/-- Both dissipation weights are strictly positive in Algorithm 1's
momentum domain with positive damping and step. Source:
arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
theorem training_dissipation_coefficients (η β δ : ℝ)
    (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1) (hδ : 0 < δ) :
    0 < δ / (4 * η) ∧ 0 < η * (1 - β ^ 2) / δ := by
  have hb : 0 < 1 - β ^ 2 := by
    nlinarith [mul_pos (sub_pos.mpr hβ') (show 0 < 1 + β by linarith)]
  constructor <;> positivity

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 := by norm_num

/-- Actual dissipation is nonnegative. Source: arXiv:2405.16397v3,
Algorithm 1 and §3.4, deterministic extension. -/
theorem trainingDissipation_nonneg (η β δ : ℝ) (f : TrainingSpace a b → ℝ)
    (s next : OptimizerState a b) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1) (hδ : 0 < δ) :
    0 ≤ trainingDissipation η β δ f s next := by
  obtain ⟨hc, he⟩ := training_dissipation_coefficients η β δ hη hβ hβ' hδ
  exact add_nonneg (mul_nonneg hc.le (sq_nonneg _)) (mul_nonneg he.le (sq_nonneg _))

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 := by norm_num

/-- Telescoping gives a bound on every finite sum of actual dissipation.
Source: arXiv:2405.16397v3, §3.4 and Appendix A.2, deterministic full-state
extension. The actual initial momentum error is retained in the bound. -/
theorem trainingRun_dissipation_budget (η β γ δ L : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (hL : 0 < L) (hstep : 2 * L * η ≤ δ * (1 - β)) (T : ℕ) :
    (∑ t ∈ Finset.range T, trainingDissipation η β δ f
        (trainingRun η β γ δ f factors initial t)
        (trainingRun η β γ δ f factors initial (t + 1))) ≤
      f initial + η / (δ * (1 - β)) * ‖gradient f initial‖ ^ 2 -
        trainingEnergy η β δ f (trainingRun η β γ δ f factors initial T) := by
  induction T with
  | zero => simp [trainingEnergy_initial]
  | succ T ih =>
    rw [Finset.sum_range_succ]
    have h := trainingRun_energy_descent η β γ δ L f factors initial hf hgrad
      hη hβ hβ' hδ hL hstep T
    dsimp only [trainingDissipation] at ih ⊢
    linarith

/-- The finite-budget domain includes a nonconstant objective and nonzero
momentum, arXiv:2405.16397v3, §3.4, deterministic extension. -/
example : SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) := by
  exact ⟨energy_models.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num⟩

end Transformer.AdaFisher
