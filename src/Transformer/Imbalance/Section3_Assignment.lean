/-
# Exact finite-data form of the assignment mechanism

arXiv:2402.19449v2, Section 3.2, Assumption 1 and Proposition 2,
equation (2), proof in Appendix H. The gradient's leading term is
`(p-1)π_k mean_k`, correcting the paper's sign. Remainders are actual
contributions from other classes, not uninterpreted error parameters.
-/

import Transformer.Imbalance.Section3_Empirical

open scoped BigOperators

noncomputable section

namespace Transformer.Imbalance

variable {c d n : ℕ}

/-- Finite version of correct assignment (Assumption 1): probability p
on class-k samples, and a uniform bound q on other samples. Asymptotic
O(1/c) is obtained by taking q=C/c with fixed C. -/
def CorrectAssignment (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (p q : ℝ) : Prop :=
  (∀ i, y i = k → probability W (x i) k = p) ∧
    (∀ i, y i ≠ k → probability W (x i) k ≤ q)

/-- Incorrect-label contribution to the empirical gradient;
Appendix H, the vector d_k divided by c, with the gradient sign corrected. -/
def gradientRemainder (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r : Fin d) : ℝ :=
  (∑ i ∈ Finset.univ.filter (fun i => y i ≠ k), probability W (x i) k * x i r) / n

/-- Incorrect-label contribution to the empirical Hessian;
Appendix H, the matrix D_k divided by c. -/
def hessianRemainder (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r s : Fin d) : ℝ :=
  (∑ i ∈ Finset.univ.filter (fun i => y i ≠ k),
    probability W (x i) k * (1 - probability W (x i) k) * x i r * x i s) / n

/-- Exact gradient decomposition behind Proposition 2, equation (2).
The paper's leading term has the opposite sign. -/
theorem assignment_gradient (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r : Fin d) (p q : ℝ)
    (hn : 0 < n) (hk : (classSamples y k).Nonempty)
    (h : CorrectAssignment W x y k p q) :
    empiricalGradient W x y k r =
      (p - 1) * frequency y k * classMean x y k r + gradientRemainder W x y k r := by
  have hcorrect : (∑ i ∈ classSamples y k, sampleGradient W (x i) (y i) k r) =
      (p - 1) * ∑ i ∈ classSamples y k, x i r := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    have hy : y i = k := (Finset.mem_filter.1 hi).2
    rw [sampleGradient_eq, h.1 i hy, ite_eq_left hy]
  have hother : (∑ i ∈ Finset.univ.filter (fun i => y i ≠ k),
      sampleGradient W (x i) (y i) k r) =
      ∑ i ∈ Finset.univ.filter (fun i => y i ≠ k), probability W (x i) k * x i r := by
    apply Finset.sum_congr rfl
    intro i hi
    have hy := (Finset.mem_filter.1 hi).2
    simp only [sampleGradient_eq, ite_eq_right hy, sub_zero]
  unfold empiricalGradient
  rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i => y i = k)]
  change ((∑ i ∈ classSamples y k, sampleGradient W (x i) (y i) k r) +
    (∑ i ∈ Finset.univ.filter (fun i => y i ≠ k), sampleGradient W (x i) (y i) k r)) / n = _
  rw [hcorrect, hother]
  rw [mul_assoc, frequency_mul_classMean x y k r hn hk]
  unfold gradientRemainder
  ring

/-- Nonvacuity of the full finite-data assignment hypotheses;
Assumption 1 and Proposition 2. -/
example : 0 < 2 ∧ (classSamples (id : Fin 2 → Fin 2) 0).Nonempty ∧
    CorrectAssignment (fun _ _ => 0 : Parameters 2 1) (fun _ _ => 1) id 0 (1 / 2) (1 / 2) := by
  refine ⟨by decide, ⟨0, by simp [classSamples]⟩, ?_⟩
  constructor <;> intro i hi <;> simp only [probability_zero] <;> norm_num

/-- Exact diagonal-Hessian decomposition behind Proposition 2, equation (2).
This retains the paper's factor p(1-p) and the actual class second moment. -/
theorem assignment_hessian (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r s : Fin d) (p q : ℝ)
    (hn : 0 < n) (hk : (classSamples y k).Nonempty)
    (h : CorrectAssignment W x y k p q) :
    empiricalHessian W x y k k r s =
      p * (1 - p) * frequency y k * classSecondMoment x y k r s +
        hessianRemainder W x y k r s := by
  have hcorrect : (∑ i ∈ classSamples y k, sampleHessian W (x i) (y i) k k r s) =
      (p * (1 - p)) * ∑ i ∈ classSamples y k, x i r * x i s := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    have hy : y i = k := (Finset.mem_filter.1 hi).2
    rw [sampleHessian_eq, h.1 i hy, ite_eq_left rfl]
    ring
  unfold empiricalHessian
  rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i => y i = k)]
  change ((∑ i ∈ classSamples y k, sampleHessian W (x i) (y i) k k r s) +
    (∑ i ∈ Finset.univ.filter (fun i => y i ≠ k), sampleHessian W (x i) (y i) k k r s)) / n = _
  rw [hcorrect]
  simp only [sampleHessian_eq, ite_true]
  rw [mul_assoc (p * (1 - p)), frequency_mul_classSecondMoment x y k r s hn hk]
  unfold hessianRemainder
  ring

/-- Nonvacuity of all hypotheses of the assignment-Hessian formula. -/
example : 0 < 2 ∧ (classSamples (id : Fin 2 → Fin 2) 0).Nonempty ∧
    CorrectAssignment (fun _ _ => 0 : Parameters 2 1) (fun _ _ => 1) id 0 (1 / 2) (1 / 2) := by
  refine ⟨by decide, ⟨0, by simp [classSamples]⟩, ?_⟩
  constructor <;> intro i hi <;> simp only [probability_zero] <;> norm_num

end Transformer.Imbalance
