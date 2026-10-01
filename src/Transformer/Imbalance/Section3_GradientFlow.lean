/-
# Matrix gradient flow and its class-frequency-dependent rate

arXiv:2402.19449v2, Theorem 3 and Appendix I, Lemmas 4--6.
The explicit integral-equation solution is verified against the matrix
field, and uniqueness identifies every global zero-initialized solution.
-/

import Transformer.Imbalance.Section3_SimpleModel
import Transformer.GlobalFlow.Existence
import Mathlib.Analysis.Calculus.ContDiff.RCLike

open Filter Asymptotics

noncomputable section

namespace Transformer.Imbalance

variable {c : ℕ}

/-- At least two classes means `z=c-1>0`; Appendix I's use of `1/z`. -/
theorem otherClasses_pos (c : ℕ) (hc : 2 ≤ c) : 0 < (c : ℝ) - 1 := by
  have h : (2 : ℝ) ≤ c := by exact_mod_cast hc
  linarith

/-- Nonvacuity of Appendix I's nontrivial class-count assumption. -/
example : 2 ≤ 3 := by decide

/-- Diagonal parameter in Lemmas 4 and 5: `a=(1-1/c)u`. -/
def gdDiagonal (c : ℕ) (hc : 2 ≤ c) (π t : ℝ) : ℝ :=
  (1 - 1 / c) * gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) π t

/-- Off-diagonal parameter in Lemmas 4 and 5: `b=-u/c`. -/
def gdOffDiagonal (c : ℕ) (hc : 2 ≤ c) (π t : ℝ) : ℝ :=
  -(1 / c) * gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) π t

/-- Explicit full parameter trajectory in Appendix I, Lemmas 4 and 5. -/
def gdParameters (c : ℕ) (hc : 2 ≤ c) (π : Fin c → ℝ) (t : ℝ) : Parameters c c :=
  fun i k => if i = k then gdDiagonal c hc (π k) t else gdOffDiagonal c hc (π k) t

/-- Diagonal-minus-off-diagonal equals the solved margin; Lemma 5. -/
theorem gdDiagonal_eq (c : ℕ) (hc : 2 ≤ c) (π t : ℝ) :
    gdDiagonal c hc π t = gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) π t +
      gdOffDiagonal c hc π t := by
  unfold gdDiagonal gdOffDiagonal
  ring

/-- All class-count hypotheses in the explicit solution are realizable. -/
example : 2 ≤ 2 := le_rfl

/-- Probabilities along the actual gradient-flow trajectory; Lemma 4. -/
theorem gdParameters_probability (c : ℕ) (hc : 2 ≤ c) (π : Fin c → ℝ)
    (t : ℝ) (i k : Fin c) :
    Perspective.softmaxWeight (fun j => gdParameters c hc π t j k) i =
      (if i = k then Real.exp (gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) (π k) t)
        else 1) /
      (Real.exp (gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) (π k) t) + ((c : ℝ) - 1)) := by
  unfold gdParameters
  rw [gdDiagonal_eq]
  exact softmax_twoLevel _ _ k i

/-- Hypothesis witness for the column-probability formula; Lemma 4. -/
example : 2 ≤ 3 := by decide

/-- The explicit solution follows the gradient of the cross-entropy
objective, with zero initialization; Appendix I, Lemmas 4 and 5. -/
theorem gdParameters_flow (c : ℕ) (hc : 2 ≤ c) (π : Fin c → ℝ) :
    IsGradientFlow π (gdParameters c hc π) := by
  have hc0 : (c : ℝ) ≠ 0 := by have := otherClasses_pos c hc; linarith
  constructor
  · ext i k
    simp [gdParameters, gdDiagonal, gdOffDiagonal, gdMargin_zero]
  · intro t i k
    have hd := hasDerivAt_gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) (π k) t
    have hden : Real.exp (gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) (π k) t) +
        ((c : ℝ) - 1) ≠ 0 := by have := otherClasses_pos c hc; positivity
    unfold gradientField
    rw [gdParameters_probability]
    by_cases hi : i = k
    · simp only [gdParameters, hi, ite_true]
      unfold gdDiagonal
      convert hd.const_mul (1 - 1 / (c : ℝ)) using 1
      field_simp
      ring
    · simp only [gdParameters, hi, ite_false]
      unfold gdOffDiagonal
      convert hd.const_mul (-(1 / (c : ℝ))) using 1
      field_simp
      ring

/-- Witness for actual zero-initialized gradient dynamics; Lemmas 4 and 5. -/
example : 2 ≤ 3 ∧ IsGradientFlow (fun _ : Fin 3 => (1 / 3 : ℝ))
    (gdParameters 3 (by decide) (fun _ => 1 / 3)) :=
  ⟨by decide, gdParameters_flow 3 (by decide) _⟩

/-- The matrix field is smooth, giving the uniqueness invoked by the
separation argument of Appendix I, Lemma 4. -/
theorem gradientField_contDiff (π : Fin c → ℝ) :
    ContDiff ℝ 1 (gradientField π) := by
  apply contDiff_pi.2
  intro i
  apply contDiff_pi.2
  intro k
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le k.val) k.isLt
  unfold gradientField Perspective.softmaxWeight
  apply ContDiff.mul contDiff_const
  apply ContDiff.sub contDiff_const
  apply ContDiff.div (by fun_prop) (by fun_prop)
  intro W
  exact (Perspective.softmaxPartition_pos hc (fun j => W j k)).ne'

/-- Lemmas 4 and 5, including uniqueness: every global zero-initialized
gradient trajectory is the explicitly solved matrix trajectory. -/
theorem gradientFlow_unique (hc : 2 ≤ c) (π : Fin c → ℝ)
    (W : ℝ → Parameters c c) (hW : IsGradientFlow π W) : W = gdParameters c hc π := by
  have hg := gdParameters_flow c hc π
  apply GlobalFlow.eq_of_hasDerivAt (gradientField_contDiff π).locallyLipschitz
  · intro t
    exact hasDerivAt_pi.2 fun i => hasDerivAt_pi.2 fun k => hW.2 t i k
  · intro t
    exact hasDerivAt_pi.2 fun i => hasDerivAt_pi.2 fun k => hg.2 t i k
  · exact hW.1.trans hg.1.symm

/-- All uniqueness hypotheses are satisfiable; Appendix I, Lemmas 4 and 5. -/
example : 2 ≤ 2 ∧ IsGradientFlow (fun _ : Fin 2 => (1 / 2 : ℝ))
    (gdParameters 2 le_rfl (fun _ => 1 / 2)) :=
  ⟨le_rfl, gdParameters_flow 2 le_rfl _⟩

/-- Loss along the matrix solution, expressed by its margin; Lemma 6. -/
theorem gdParameters_loss (c : ℕ) (hc : 2 ≤ c) (π : Fin c → ℝ) (k : Fin c) (t : ℝ) :
    sampleLoss (gdParameters c hc π t) (Pi.single k 1) k =
      marginLoss ((c : ℝ) - 1) (gdMargin ((c : ℝ) - 1) (otherClasses_pos c hc) (π k) t) := by
  rw [sampleLoss_basis]
  unfold gdParameters
  rw [gdDiagonal_eq]
  exact crossEntropy_twoLevel _ _ k

/-- Nonvacuity of the class-count hypothesis for the matrix loss. -/
example : 2 ≤ 2 := le_rfl

/-- Gradient-flow conclusion of Theorem 3, with positive class frequency
and at least two classes stated explicitly. This is the original rate. -/
theorem gradient_flow_rate (hc : 2 ≤ c) (π : Fin c → ℝ)
    (W : ℝ → Parameters c c) (hW : IsGradientFlow π W) (k : Fin c) (hπ : 0 < π k) :
    (fun t => sampleLoss (W t) (Pi.single k 1) k) =Θ[atTop] (fun t => 1 / (π k * t)) := by
  rw [gradientFlow_unique hc π W hW]
  simp_rw [gdParameters_loss]
  exact gradient_loss_rate _ (otherClasses_pos c hc) _ hπ

/-- Nonvacuity of all hypotheses of the gradient part of Theorem 3. -/
example : 2 ≤ 3 ∧ IsGradientFlow (fun _ : Fin 3 => (1 / 3 : ℝ))
    (gdParameters 3 (by decide) (fun _ => 1 / 3)) ∧ (0 : ℝ) < 1 / 3 :=
  ⟨by decide, gdParameters_flow 3 (by decide) _, by norm_num⟩

end Transformer.Imbalance
