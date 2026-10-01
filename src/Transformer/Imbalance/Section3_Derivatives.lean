/-
# The actual gradient and Hessian

arXiv:2402.19449v2, Section 3.2, equation (4), and Appendix H.1.
The paper writes `(1_{y=k}-p_k)x` for the gradient of `-log p_y`.
Its sign is reversed. The definitions below use actual Lean derivatives
and prove the corrected formula `(p_k-1_{y=k})x`.
-/

import Transformer.Imbalance.Section3_Model

open scoped BigOperators

namespace Transformer.Imbalance

variable {c d : ℕ}

/-- Derivative of log-sum-exp along a line; Section 3.2, equation (3). -/
theorem hasDerivAt_crossEntropy_line (z v : Fin c → ℝ) (y : Fin c) (t : ℝ) :
    HasDerivAt (fun s => crossEntropy (fun j => z j + s * v j) y)
      ((∑ j, Perspective.softmaxWeight (fun l => z l + t * v l) j * v j) - v y) t := by
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le y.val) y.isLt
  have hZ := Perspective.softmaxPartition_pos hc (fun j => z j + t * v j)
  have hd : HasDerivAt (fun s => ∑ j, Real.exp (z j + s * v j))
      (∑ j, Real.exp (z j + t * v j) * v j) t := by
    apply HasDerivAt.fun_sum
    intro j _
    simpa only [id_eq, one_mul, mul_one, mul_comm] using
      (((hasDerivAt_id t).mul_const (v j)).const_add (z j)).exp
  simp_rw [crossEntropy_eq]
  have hh : HasDerivAt (fun s => Real.log (∑ j, Real.exp (z j + s * v j)) -
      (z y + s * v y))
      ((∑ j, Real.exp (z j + t * v j) * v j) /
        (∑ j, Real.exp (z j + t * v j)) - v y) t := by
    simpa only [id_eq, one_mul] using
      (hd.log hZ.ne').fun_sub (((hasDerivAt_id t).mul_const (v y)).const_add (z y))
  convert hh using 1
  simp only [Perspective.softmaxWeight]
  congr 1
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Derivative of a softmax probability along an arbitrary logit direction;
Section 3.2, differentiation of equation (3), used in equation (4). -/
theorem hasDerivAt_softmax_line (z v : Fin c → ℝ) (k : Fin c) (t : ℝ) :
    HasDerivAt (fun s => Perspective.softmaxWeight (fun j => z j + s * v j) k)
      (Perspective.softmaxWeight (fun j => z j + t * v j) k *
        (v k - ∑ j, Perspective.softmaxWeight (fun l => z l + t * v l) j * v j)) t := by
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le k.val) k.isLt
  have hZ := Perspective.softmaxPartition_pos hc (fun j => z j + t * v j)
  have hd : HasDerivAt (fun s => ∑ j, Real.exp (z j + s * v j))
      (∑ j, Real.exp (z j + t * v j) * v j) t := by
    apply HasDerivAt.fun_sum
    intro j _
    simpa only [id_eq, one_mul, mul_one, mul_comm] using
      (((hasDerivAt_id t).mul_const (v j)).const_add (z j)).exp
  have hn : HasDerivAt (fun s => Real.exp (z k + s * v k))
      (Real.exp (z k + t * v k) * v k) t := by
    simpa only [id_eq, one_mul] using
      (((hasDerivAt_id t).mul_const (v k)).const_add (z k)).exp
  unfold Perspective.softmaxWeight
  convert hn.div hd hZ.ne' using 1
  have hs : (∑ j, Real.exp (z j + t * v j) /
      (∑ l, Real.exp (z l + t * v l)) * v j) =
      (∑ j, Real.exp (z j + t * v j) * v j) /
      (∑ l, Real.exp (z l + t * v l)) := by
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hs]
  field_simp

/-- Actual partial derivative of cross-entropy, with the sign corrected from
Section 3.2, equation (4), and Appendix H. -/
theorem sampleGradient_eq (W : Parameters c d) (x : Fin d → ℝ)
    (y k : Fin c) (r : Fin d) :
    sampleGradient W x y k r =
      (probability W x k - if y = k then 1 else 0) * x r := by
  have h := hasDerivAt_crossEntropy_line (scores W x)
    (fun j => if j = k then x r else 0) y 0
  have hf : (fun t => sampleLoss (coordinateUpdate W k r t) x y) =
      (fun t => crossEntropy (fun j => scores W x j +
        t * (if j = k then x r else 0)) y) := by
    funext t
    unfold sampleLoss
    congr 1
    funext j
    exact scores_coordinateUpdate W x k r t j
  rw [sampleGradient, hf, h.deriv]
  simp only [zero_mul, add_zero]
  simp only [mul_ite, mul_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  unfold probability
  split_ifs <;> ring

/-- Actual Hessian blocks. The diagonal is `p_k(1-p_k)xxᵀ` and an
off-diagonal block is `-p_k p_j xxᵀ`; equation (4) and Appendix H.1. -/
theorem sampleHessian_eq (W : Parameters c d) (x : Fin d → ℝ)
    (y k j : Fin c) (r s : Fin d) :
    sampleHessian W x y k j r s =
      probability W x k * ((if k = j then 1 else 0) - probability W x j) *
        x r * x s := by
  have h := hasDerivAt_softmax_line (scores W x)
    (fun l => if l = j then x s else 0) k 0
  have hf : (fun t => probability (coordinateUpdate W j s t) x k) =
      (fun t => Perspective.softmaxWeight (fun l => scores W x l +
        t * (if l = j then x s else 0)) k) := by
    funext t
    unfold probability
    congr 1
    funext l
    exact scores_coordinateUpdate W x j s t l
  have hp : HasDerivAt (fun t => probability (coordinateUpdate W j s t) x k)
      (Perspective.softmaxWeight (fun l => scores W x l +
        0 * (if l = j then x s else 0)) k *
        ((if k = j then x s else 0) -
          ∑ l, Perspective.softmaxWeight (fun q => scores W x q +
            0 * (if q = j then x s else 0)) l * (if l = j then x s else 0))) 0 := by
    rw [hf]
    exact h
  simp_rw [sampleHessian, sampleGradient_eq]
  rw [((hp.sub_const (if y = k then 1 else 0)).mul_const (x r)).deriv]
  simp only [zero_mul, add_zero]
  simp only [mul_ite, mul_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  unfold probability
  split_ifs <;> ring

/-- Counterexample to the sign of the gradient in equation (4) and
Proposition 2, equation (1): with two classes and `x=1`, the correct-label
derivative at zero is `-1/2`, whereas the paper's expression is `+1/2`. -/
theorem paper_gradient_sign_false :
    sampleGradient (fun _ _ => 0 : Parameters 2 1) (fun _ => 1) 0 0 0 = -1 / 2 ∧
      sampleGradient (fun _ _ => 0 : Parameters 2 1) (fun _ => 1) 0 0 0 ≠
        (1 - probability (fun _ _ => 0 : Parameters 2 1) (fun _ => 1) 0) := by
  simp only [sampleGradient_eq, probability_zero]
  norm_num

end Transformer.Imbalance
