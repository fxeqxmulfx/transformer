/-
# The original concrete AMSGradMD scalar step can stay in one component

arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2, with the explicitly
specified AMSGrad callbacks of arXiv:1904.03590v4, Algorithm 1 and §6.
With epsilon one, both moment coefficients zero and LR one quarter,
the actual AMSGrad direction step from positive unit direction remains
positive. Neither moment histories nor factor gradients are idealized.
-/

import Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks
import Transformer.MagnitudeDirection.SectionA_TrainingCounterexampleStep

noncomputable section

namespace Transformer.MagnitudeDirection

/-- For a positive gradient, the actual epsilon-stabilized AMSGrad update
from a unit scalar direction stays positive with LR one quarter. Its
new maximum is at least the current squared gradient; no premise about
previous buffer values is required.
Source: arXiv:1904.03590v4, Algorithm 1 and §6; arXiv:2606.25971v2,
Appendix A, concrete AMSGradMD training counterexample. -/
theorem amsgradMatrixCallback_unit_positive (s : AMSGradBuffers (Fin 1 × Fin 1))
    (G : Matrix (Fin 1) (Fin 1) ℝ) (hG : 0 < G 0 0) :
    0 < (amsgradMatrixCallback 1 0 0 s 1 G (1 / 4)).2 0 0 := by
  have hroot : G 0 0 ≤ Real.sqrt (max (s.maximum (0, 0)) (G 0 0 ^ 2)) := by
    calc
      _ = Real.sqrt (G 0 0 ^ 2) := (Real.sqrt_sq hG.le).symm
      _ ≤ _ := Real.sqrt_le_sqrt (le_max_right _ _)
  have hden : 0 < 1 + Real.sqrt (max (s.maximum (0, 0)) (G 0 0 ^ 2)) := by
    linarith
  have hquot : (1 / 4 : ℝ) * G 0 0 /
      (1 + Real.sqrt (max (s.maximum (0, 0)) (G 0 0 ^ 2))) < 1 / 4 := by
    apply (div_lt_iff₀ hden).mpr
    linarith
  dsimp [amsgradMatrixCallback, amsgradVectorCallback]
  norm_num only [zero_mul, sub_zero, one_mul, zero_add, Matrix.one_apply, ite_true]
  linarith

/-- The genuine nonzero-gradient premise is satisfiable,
arXiv:2606.25971v2, Appendix A, AMSGradMD training counterexample. -/
example : (0 : ℝ) < (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 := by norm_num

/-- The complete concrete AMSGradMD proposal preserves positive fused
weight and positive unit direction from that component, for a positive
actual loss gradient. All three first/second/maximum histories are
updated before the weight is reassembled.
Source: arXiv:2606.25971v2, Appendix A, AMSGradMD training counterexample;
arXiv:1904.03590v4, Algorithm 1 and §6. -/
theorem amsgradMDProposal_positive (memory : AMSGradMDMemory 1 1)
    (s : FullState 1 1) (G : Matrix (Fin 1) (Fin 1) ℝ)
    (hD : fullDirection softplus s = 1) (hG : 0 < G 0 0) :
    0 < (amsgradMDProposal 1 0 0 (1 / 4) (1 / 4) 1 memory s G).2.weight 0 0 ∧
      fullDirection softplus
        (amsgradMDProposal 1 0 0 (1 / 4) (1 / 4) 1 memory s G).2 = 1 := by
  let row := fun i => softplus (s.rawRow i)
  let col := fun j => softplus (s.rawCol j)
  have hDG : 0 < directionGradient row col G 0 0 :=
    mul_pos (mul_pos (softplus_pos _) hG) (softplus_pos _)
  have hP := matrixProject_single_positive _
    (amsgradMatrixCallback_unit_positive memory.1 (directionGradient row col G) hDG)
  rw [amsgradMDProposal, statefulFullProposal_eq]
  constructor
  · dsimp only [fullStep, fullCandidate]
    rw [hD, hP]
    simp only [fuse, Matrix.one_apply, ite_true, mul_one]
    exact mul_pos (softplus_pos _) (softplus_pos _)
  · dsimp only [fullDirection, fullStep, fullCandidate]
    rw [unfuse_fuse _ _ _ (fun i => (softplus_pos _).ne') (fun j => (softplus_pos _).ne')]
    change matrixProject 1 ((amsgradMatrixCallback 1 0 0 memory.1
      (fullDirection softplus s) (directionGradient row col G) (1 / 4)).2) = 1
    rw [hD]
    exact hP

/-- Positive unit storage and a nonzero actual gradient satisfy the full
proposal premises. Source: arXiv:2606.25971v2, Appendix A, AMSGradMD counterexample. -/
example : fullDirection softplus (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)) = 1 ∧
    (0 : ℝ) < (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 :=
  ⟨unitGainState_direction _, by norm_num⟩

end Transformer.MagnitudeDirection
