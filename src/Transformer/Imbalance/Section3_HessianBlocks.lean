/-
# Hessian block balance and diagonal dominance

arXiv:2402.19449v2, Appendix H.1. The exact algebra implies diagonal
dominance of the matrix of block traces. An orders-of-magnitude separation
is an empirical observation, not a universal consequence of the identity.
-/

import Transformer.Imbalance.Section3_AssignmentBounds

open scoped BigOperators

noncomputable section

namespace Transformer.Imbalance

variable {c d : ℕ}

/-- Sample Hessian block trace, Appendix H.1. -/
def sampleBlockTrace (W : Parameters c d) (x : Fin d → ℝ) (y k j : Fin c) : ℝ :=
  ∑ r, sampleHessian W x y k j r r

/-- Each row of class Hessian blocks sums to zero; Appendix H.1,
the identity H_kk = -sum_{j≠k} H_kj. -/
theorem sampleHessian_row_sum (W : Parameters c d) (x : Fin d → ℝ)
    (y k : Fin c) (r s : Fin d) : (∑ j, sampleHessian W x y k j r s) = 0 := by
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le k.val) k.isLt
  have hsum : (∑ j, ((if k = j then (1 : ℝ) else 0) - probability W x j)) = 0 := by
    rw [Finset.sum_sub_distrib]
    simp only [probability, ← Perspective.sum_softmaxWeight hc (scores W x)]
    simp [Perspective.sum_softmaxWeight hc]
  calc
    _ = probability W x k * (∑ j, ((if k = j then 1 else 0) - probability W x j)) * x r * x s := by
      simp only [sampleHessian_eq, Finset.mul_sum, Finset.sum_mul]
    _ = 0 := by rw [hsum]; ring

/-- Nonnegative diagonal traces in Appendix H.1. -/
theorem sampleBlockTrace_diagonal_nonneg (W : Parameters c d) (x : Fin d → ℝ) (y k : Fin c) :
    0 ≤ sampleBlockTrace W x y k k := by
  apply Finset.sum_nonneg
  intro r _
  rw [sampleHessian_eq, ite_eq_left rfl]
  have hp := probability_bounds W x k
  have hcoef := mul_nonneg hp.1 (show 0 ≤ 1 - probability W x k by linarith)
  nlinarith [sq_nonneg (x r)]

/-- Nonpositive off-diagonal traces in Appendix H.1. -/
theorem sampleBlockTrace_offDiagonal_nonpos (W : Parameters c d) (x : Fin d → ℝ)
    (y k j : Fin c) (hkj : k ≠ j) : sampleBlockTrace W x y k j ≤ 0 := by
  apply Finset.sum_nonpos
  intro r _
  rw [sampleHessian_eq, ite_eq_right hkj]
  have hp := mul_nonneg (probability_bounds W x k).1 (probability_bounds W x j).1
  nlinarith [sq_nonneg (x r)]

/-- Nonvacuity of the distinct-block hypothesis; Appendix H.1. -/
example : (0 : Fin 2) ≠ 1 := by decide

/-- The matrix of Hessian block traces has zero row sums; Appendix H.1. -/
theorem sampleBlockTrace_row_sum (W : Parameters c d) (x : Fin d → ℝ) (y k : Fin c) :
    (∑ j, sampleBlockTrace W x y k j) = 0 := by
  unfold sampleBlockTrace
  rw [Finset.sum_comm]
  simp only [sampleHessian_row_sum, Finset.sum_const_zero]

/-- Exact diagonal dominance of the block-trace matrix; Appendix H.1.
The diagonal equals the sum of absolute off-diagonal entries in its row. -/
theorem sampleBlockTrace_diagonal_dominance (W : Parameters c d) (x : Fin d → ℝ) (y k : Fin c) :
    |sampleBlockTrace W x y k k| =
      ∑ j ∈ Finset.univ.erase k, |sampleBlockTrace W x y k j| := by
  rw [abs_of_nonneg (sampleBlockTrace_diagonal_nonneg W x y k)]
  have he := Finset.sum_erase_add Finset.univ (fun j => sampleBlockTrace W x y k j)
    (Finset.mem_univ k)
  rw [sampleBlockTrace_row_sum] at he
  have habs : (∑ j ∈ Finset.univ.erase k, |sampleBlockTrace W x y k j|) =
      -(∑ j ∈ Finset.univ.erase k, sampleBlockTrace W x y k j) := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro j hj
    exact abs_of_nonpos (sampleBlockTrace_offDiagonal_nonpos W x y k j
      (Finset.ne_of_mem_erase hj).symm)
  rw [habs]
  linarith

/-- A two-class witness shows why the empirical orders-of-magnitude
comparison in Appendix H.1 cannot be a strict universal inequality. -/
theorem two_class_blocks_equal_magnitude :
    sampleBlockTrace (fun _ _ => 0 : Parameters 2 1) (fun _ => 1) 0 0 0 = 1 / 4 ∧
    |sampleBlockTrace (fun _ _ => 0 : Parameters 2 1) (fun _ => 1) 0 0 1| = 1 / 4 := by
  norm_num [sampleBlockTrace, sampleHessian_eq, probability_zero]

end Transformer.Imbalance
