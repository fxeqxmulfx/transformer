/-
# Muon — centering the expert-routing bias

arXiv:2502.16982, Appendix C, “Auxfree Bias Update”. Subtracting the mean
sign shifts all expert scores by the same constant, preserving every
ordering and top-k selection. It also preserves the sum of the biases.
-/

import Transformer.Muon.Section2_Models

open scoped BigOperators

noncomputable section

namespace Transformer.Muon

variable {n : ℕ}

/-- The real sign of an expert's violation, arXiv:2502.16982, Appendix C. -/
def violationSign (x : ℝ) : ℝ := if 0 < x then 1 else if x < 0 then -1 else 0

/-- Mean of the expert-violation signs, arXiv:2502.16982, Appendix C. -/
def meanViolationSign (e : Fin n → ℝ) : ℝ := (∑ i, violationSign (e i)) / n

/-- The original routing bias update, arXiv:2502.16982, Appendix C. -/
def originalBiasStep (u : ℝ) (b e : Fin n → ℝ) (i : Fin n) : ℝ :=
  b i + u * violationSign (e i)

/-- The centered routing bias update, arXiv:2502.16982, Appendix C. -/
def centeredBiasStep (u : ℝ) (b e : Fin n → ℝ) (i : Fin n) : ℝ :=
  b i + u * (violationSign (e i) - meanViolationSign e)

/-- Membership in a top selection, with all outside scores bounded by all
inside scores. Ties are allowed, as in the source's top-k routing logic.
Source: arXiv:2502.16982, Appendix C. -/
def IsTopSelection (scores : Fin n → ℝ) (S : Finset (Fin n)) : Prop :=
  ∀ i ∈ S, ∀ j ∉ S, scores j ≤ scores i

/-- Centering applies one common shift to the updated biases,
arXiv:2502.16982, Appendix C, “Auxfree Bias Update”. -/
theorem centeredBiasStep_eq (u : ℝ) (b e : Fin n → ℝ) (i : Fin n) :
    centeredBiasStep u b e i = originalBiasStep u b e i - u * meanViolationSign e := by
  unfold centeredBiasStep originalBiasStep
  ring

/-- Every score comparison, including ties, is preserved by centering,
arXiv:2502.16982, Appendix C, “Auxfree Bias Update”. -/
theorem centered_scores_le_iff (u : ℝ) (scores b e : Fin n → ℝ) (i j : Fin n) :
    scores i + centeredBiasStep u b e i ≤ scores j + centeredBiasStep u b e j ↔
      scores i + originalBiasStep u b e i ≤ scores j + originalBiasStep u b e j := by
  rw [centeredBiasStep_eq, centeredBiasStep_eq]
  constructor <;> intro h <;> linarith

/-- Centered and original bias updates yield exactly the same top-k
selections, for every `k` and even with ties. This is the invariant claimed
in arXiv:2502.16982, Appendix C, “Auxfree Bias Update”. -/
theorem centered_topK_iff (u : ℝ) (scores b e : Fin n → ℝ) (k : ℕ) (S : Finset (Fin n)) :
    (S.card = k ∧ IsTopSelection (fun i => scores i + centeredBiasStep u b e i) S) ↔
      (S.card = k ∧ IsTopSelection (fun i => scores i + originalBiasStep u b e i) S) := by
  simp only [IsTopSelection, centered_scores_le_iff]

/-- The sum of routing biases is preserved by the centered update.
The number of experts must be positive to interpret the mean.
Source: arXiv:2502.16982, Appendix C, “Auxfree Bias Update”. -/
theorem centeredBiasStep_sum (u : ℝ) (b e : Fin n → ℝ) (hn : 0 < n) :
    (∑ i, centeredBiasStep u b e i) = ∑ i, b i := by
  have hn' : (n : ℝ) ≠ 0 := by positivity
  simp only [centeredBiasStep, mul_sub, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold meanViolationSign
  field_simp
  ring

/-- There can be a positive number of experts, arXiv:2502.16982, Appendix C. -/
example : 0 < (2 : ℕ) := by norm_num

end Transformer.Muon
