/-
# Empirical derivatives and initialization

arXiv:2402.19449v2, Section 3.2, Proposition 2, equation (1), Appendix H.
The paper's initialization gradient has its sign reversed. The Hessian
formula is unchanged. Derivatives are averages of actual sample derivatives.
-/

import Transformer.Imbalance.Section3_Objective

open scoped BigOperators

noncomputable section

namespace Transformer.Imbalance

variable {c d n : ℕ}

/-- Empirical gradient block, averaging actual partial derivatives;
Proposition 2 and Appendix H. -/
def empiricalGradient (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r : Fin d) : ℝ :=
  (∑ i, sampleGradient W (x i) (y i) k r) / n

/-- Empirical Hessian block, averaging actual second partial derivatives;
Proposition 2 and Appendix H.1. -/
def empiricalHessian (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k j : Fin c) (r s : Fin d) : ℝ :=
  (∑ i, sampleHessian W (x i) (y i) k j r s) / n

/-- Differentiating the actual empirical loss gives the averaged gradient;
Section 3.2, Proposition 2, proof in Appendix H. -/
theorem empiricalGradient_eq_deriv (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r : Fin d) :
    empiricalGradient W x y k r =
      deriv (fun t => empiricalLoss (coordinateUpdate W k r t) x y) 0 := by
  have hd := (HasDerivAt.fun_sum (u := Finset.univ) fun i _ =>
    hasDerivAt_sampleLoss_coordinate W (x i) (y i) k r).div_const (n : ℝ)
  exact hd.deriv.symm

/-- Class-conditional sums equal frequency times class means;
Section 3.2, the moment definitions before Assumption 1. -/
theorem frequency_mul_classMean (x : Fin n → Fin d → ℝ) (y : Fin n → Fin c)
    (k : Fin c) (r : Fin d) (hn : 0 < n) (hk : (classSamples y k).Nonempty) :
    frequency y k * classMean x y k r = (∑ i ∈ classSamples y k, x i r) / n := by
  have hN : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hK : ((classSamples y k).card : ℝ) ≠ 0 := by
    exact_mod_cast hk.card_pos.ne'
  unfold frequency classMean
  field_simp

/-- Nonvacuity of the class-mean hypotheses; Section 3.2. -/
example : 0 < 1 ∧ (classSamples (fun _ : Fin 1 => (0 : Fin 1)) 0).Nonempty := by
  refine ⟨by decide, ⟨0, ?_⟩⟩
  simp [classSamples]

/-- Class second moments have the same frequency conversion;
Section 3.2, before Assumption 1. -/
theorem frequency_mul_classSecondMoment (x : Fin n → Fin d → ℝ) (y : Fin n → Fin c)
    (k : Fin c) (r s : Fin d) (hn : 0 < n) (hk : (classSamples y k).Nonempty) :
    frequency y k * classSecondMoment x y k r s =
      (∑ i ∈ classSamples y k, x i r * x i s) / n := by
  have hN : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hK : ((classSamples y k).card : ℝ) ≠ 0 := by
    exact_mod_cast hk.card_pos.ne'
  unfold frequency classSecondMoment
  field_simp

/-- Nonvacuity of the class-second-moment hypotheses; Section 3.2. -/
example : 0 < 1 ∧ (classSamples (fun _ : Fin 1 => (0 : Fin 1)) 0).Nonempty := by
  refine ⟨by decide, ⟨0, ?_⟩⟩
  simp [classSamples]

/-- Corrected initialization gradient of Proposition 2, equation (1).
The source writes `π_k mean_k - mean/c`, the negative of the derivative.
Both input vectors and class frequencies retain their original dependence. -/
theorem empiricalGradient_zero (x : Fin n → Fin d → ℝ) (y : Fin n → Fin c)
    (k : Fin c) (r : Fin d) (hn : 0 < n) (hk : (classSamples y k).Nonempty) :
    empiricalGradient (fun _ _ => 0) x y k r =
      (1 / (c : ℝ)) * meanInput x r - frequency y k * classMean x y k r := by
  rw [frequency_mul_classMean x y k r hn hk]
  unfold empiricalGradient meanInput
  simp only [sampleGradient_eq, probability_zero, sub_mul, ite_mul, one_mul, zero_mul,
    Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp only [classSamples, Finset.sum_filter]
  ring

/-- Nonvacuity of Proposition 2's initialization-gradient hypotheses. -/
example : 0 < 2 ∧ (classSamples (fun i : Fin 2 => i) 0).Nonempty := by
  refine ⟨by decide, ⟨0, ?_⟩⟩
  simp [classSamples]

/-- Initialization Hessian of Proposition 2, equation (1).
Every diagonal class block is `(1/c)(1-1/c)` times the same second moment. -/
theorem empiricalHessian_zero (x : Fin n → Fin d → ℝ) (y : Fin n → Fin c)
    (k : Fin c) (r s : Fin d) :
    empiricalHessian (fun _ _ => 0) x y k k r s =
      (1 / (c : ℝ)) * (1 - 1 / (c : ℝ)) * secondMoment x r s := by
  unfold empiricalHessian secondMoment
  simp only [sampleHessian_eq, probability_zero, ite_true]
  simp_rw [mul_assoc]
  rw [← Finset.mul_sum, ← Finset.mul_sum]
  ring

end Transformer.Imbalance
