/-
# Connecting the simple matrix field to the actual objective

arXiv:2402.19449v2, Section 3.2, equation (4), and Appendix I, Lemma 4.
The ODE field is the negative partial derivative of the weighted loss,
not an independent surrogate for the optimization problem.
-/

import Transformer.Imbalance.Section3_SimpleModel

open scoped BigOperators

namespace Transformer.Imbalance

variable {c d : ℕ}

/-- Differentiability of a sample along a matrix coordinate;
Section 3.2, equation (4). -/
theorem hasDerivAt_sampleLoss_coordinate (W : Parameters c d) (x : Fin d → ℝ)
    (y k : Fin c) (r : Fin d) :
    HasDerivAt (fun t => sampleLoss (coordinateUpdate W k r t) x y)
      (sampleGradient W x y k r) 0 := by
  have hf : (fun t => sampleLoss (coordinateUpdate W k r t) x y) =
      (fun t => crossEntropy (fun j => scores W x j +
        t * (if j = k then x r else 0)) y) := by
    funext t
    unfold sampleLoss
    congr 1
    funext j
    exact scores_coordinateUpdate W x k r t j
  have hd : DifferentiableAt ℝ (fun t => sampleLoss (coordinateUpdate W k r t) x y) 0 := by
    rw [hf]
    exact (hasDerivAt_crossEntropy_line (scores W x)
      (fun j => if j = k then x r else 0) y 0).differentiableAt
  exact hd.hasDerivAt

/-- The negative coordinate derivative of the actual weighted objective
is precisely the vector field of Appendix I, Lemma 4. -/
theorem gradientField_eq_negative_deriv (π : Fin c → ℝ) (W : Parameters c c)
    (i k : Fin c) :
    gradientField π W i k =
      -deriv (fun t => simpleLoss π (coordinateUpdate W i k t)) 0 := by
  have hd := HasDerivAt.fun_sum (u := Finset.univ) fun j _ =>
    (hasDerivAt_sampleLoss_coordinate W (Pi.single j 1) j i k).const_mul (π j)
  change gradientField π W i k = -deriv
    (fun t => ∑ j, π j * sampleLoss (coordinateUpdate W i k t) (Pi.single j 1) j) 0
  rw [hd.deriv]
  simp only [sampleGradient_eq, probability]
  have hs : ∀ j, scores W (Pi.single j 1) = fun l => W l j := by
    intro j
    funext l
    exact scores_basis W j l
  simp_rw [hs]
  simp only [Pi.single_apply, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq,
    Finset.mem_univ, ite_true]
  unfold gradientField
  by_cases hi : i = k
  · simp [hi]
    ring
  · simp [hi, eq_comm]

end Transformer.Imbalance
