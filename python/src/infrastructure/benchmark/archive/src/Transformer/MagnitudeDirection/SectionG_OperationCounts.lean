/-
# Algebra of the optimizer overhead model

arXiv:2606.25971v2, Appendix G. Elementwise gain work is proportional
to parameter count, forward/backward work to parameters times tokens.
These theorems verify that model, not measured distributed throughput.
-/

import Mathlib.Tactic

noncomputable section

namespace Transformer.MagnitudeDirection

/-- Extra gain operations relative to forward/backward work in the stated
operation-count model, arXiv:2606.25971v2, Appendix G. -/
def overheadFraction (gainCost trainingCost parameters tokens : ℝ) : ℝ :=
  (gainCost * parameters) / (trainingCost * parameters * tokens)

/-- Parameter count cancels from the relative elementwise overhead,
arXiv:2606.25971v2, Appendix G. A nonzero parameter count is necessary
for this cancellation with Lean's total division. -/
theorem overheadFraction_eq (g a P T : ℝ) (hP : P ≠ 0) :
    overheadFraction g a P T = g / (a * T) := by
  unfold overheadFraction
  field_simp

/-- Nonzero parameter counts exist, arXiv:2606.25971v2, Appendix G. -/
example : (1 : ℝ) ≠ 0 := by norm_num

/-- Increasing tokens per step decreases the operation-count overhead,
arXiv:2606.25971v2, Appendix G. The proportionality coefficients are
explicit; the statement does not assume that all architectures have the
same measured wall-clock cost. -/
theorem overheadFraction_antitone (g a P T₁ T₂ : ℝ) (hg : 0 ≤ g)
    (ha : 0 < a) (hP : P ≠ 0) (hT : 0 < T₁) (hTT : T₁ ≤ T₂) :
    overheadFraction g a P T₂ ≤ overheadFraction g a P T₁ := by
  rw [overheadFraction_eq g a P T₂ hP, overheadFraction_eq g a P T₁ hP]
  exact div_le_div_of_nonneg_left hg (mul_pos ha hT)
    (mul_le_mul_of_nonneg_left hTT ha.le)

/-- The overhead comparison has satisfiable cost assumptions,
arXiv:2606.25971v2, Appendix G. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 ∧
    (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 := by norm_num

/-- Doubling batch tokens halves the algebraic gain overhead,
arXiv:2606.25971v2, Appendix G. This is not an assertion that the printed
throughput slowdowns halve, since communication and other costs differ. -/
theorem overheadFraction_double_tokens (g a P T : ℝ) :
    overheadFraction g a P (2 * T) = overheadFraction g a P T / 2 := by
  unfold overheadFraction
  ring

/-- In the dense square-matrix Muon model, quadratic elementwise gain work
relative to cubic orthogonalization work scales inversely with hidden size.
Source: arXiv:2606.25971v2, Appendix G, `O(d³)` Newton--Schulz comparison. -/
theorem gain_to_cubic_cost (g a d : ℝ) (hd : d ≠ 0) :
    (g * d ^ 2) / (a * d ^ 3) = g / (a * d) := by
  field_simp

/-- Nonzero hidden dimensions exist, arXiv:2606.25971v2, Appendix G. -/
example : (512 : ℝ) ≠ 0 := by norm_num

end Transformer.MagnitudeDirection
