/-
# Muon — the gate-factor calculation and its variance interpretation

arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. Sorting sigmoid scores is
equivalent to sorting their logits. For nonempty selected logits the
normalized gate weights are probabilities. The output-variance statement
requires equal, uncorrelated expert-output variances; it is not an
unconditional statement about the trained experts.
-/

import Transformer.Muon.AppendixC_GateScaling
import Transformer.Muon.AppendixC_Routing
import Mathlib.Analysis.SpecialFunctions.Sigmoid

open scoped BigOperators

noncomputable section

namespace Transformer.Muon

variable {k : ℕ}

/-- The sigmoid and logits have identical top selections,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`, “select topk logits”. -/
theorem sigmoid_top_selection_iff (logits : Fin k → ℝ) (S : Finset (Fin k)) :
    IsTopSelection (fun i => Real.sigmoid (logits i)) S ↔ IsTopSelection logits S := by
  simp only [IsTopSelection, Real.sigmoid_strictMono.le_iff_le]

/-- Nonempty selected sigmoid logits have positive total mass,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
theorem sigmoid_selected_mass_pos (logits : Fin k → ℝ) (hk : 0 < k) :
    0 < ∑ i, Real.sigmoid (logits i) := by
  apply Finset.sum_pos (fun i _ => Real.sigmoid_pos (logits i))
  exact ⟨⟨0, hk⟩, Finset.mem_univ _⟩

/-- A nonempty selection exists, arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
example : 0 < (2 : ℕ) := by norm_num

/-- The code's normalized sigmoid weights are a probability vector,
arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`, “renormalize”. -/
theorem sigmoid_gate_probability (logits : Fin k → ℝ) (hk : 0 < k) :
    (∀ i, 0 ≤ normalizedGateWeights (fun j => Real.sigmoid (logits j)) i) ∧
      (∑ i, normalizedGateWeights (fun j => Real.sigmoid (logits j)) i) = 1 := by
  exact normalizedGateWeights_probability _ (fun i => (Real.sigmoid_pos (logits i)).le)
    (sigmoid_selected_mass_pos logits hk)

/-- A nonempty selection exists, arXiv:2502.16982, Appendix C, `fig:gate_scaling_code`. -/
example : 0 < (2 : ℕ) := by norm_num

/-- The variance of a weighted, scaled mixture in terms of its covariance
matrix, the mathematical model underlying arXiv:2502.16982, Appendix C,
“Gate Scaling Factor”. -/
def mixtureVariance (p : Fin k → ℝ) (cov : Fin k → Fin k → ℝ) (scale : ℝ) : ℝ :=
  scale ^ 2 * ∑ i, ∑ j, p i * p j * cov i j

/-- The source's factor preserves a common expert variance in the equal,
uncorrelated covariance model. The source reports a numerical factor for
“similar output rms”; the covariance condition is made explicit here and
is not asserted for its actual trained experts.
Source: arXiv:2502.16982, Appendix C, “Gate Scaling Factor”, `fig:gate_scaling_code`. -/
theorem gateScale_preserves_variance (p : Fin k → ℝ) (cov : Fin k → Fin k → ℝ) (v : ℝ)
    (hp : ∀ i, 0 ≤ p i) (hs : ∑ i, p i = 1)
    (hcov : ∀ i j, cov i j = if i = j then v else 0) :
    mixtureVariance p cov (gateScale p) = v := by
  have hsum : (∑ i, ∑ j, p i * p j * cov i j) = (∑ i, p i ^ 2) * v := by
    simp [hcov, mul_ite, pow_two, Finset.sum_mul]
  rw [mixtureVariance, hsum, ← mul_assoc, gateScale_normalizes_energy p hp hs, one_mul]

/-- Probability gates and an equal diagonal covariance coexist,
arXiv:2502.16982, Appendix C, the explicit variance model. -/
example : (∀ i : Fin 2, (0 : ℝ) ≤ (fun _ : Fin 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Fin 2, (fun _ : Fin 2 => (1 / 2 : ℝ)) i) = 1 ∧
    (∀ i j : Fin 2, (fun i j : Fin 2 => if i = j then (1 : ℝ) else 0) i j =
      if i = j then (1 : ℝ) else 0) := by
  norm_num

end Transformer.Muon
