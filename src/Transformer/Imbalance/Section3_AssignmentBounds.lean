/-
# Quantitative errors in the assignment mechanism

arXiv:2402.19449v2, Proposition 2, equation (2), Appendix H.
The paper concludes that incorrect-class contributions are O(1/c)
from probability bounds alone. Uniform bounds on data moments are also
needed. Here |x_ir|≤B and off-class probabilities ≤q give explicit errors.
-/

import Transformer.Imbalance.Section3_Assignment

open scoped BigOperators

namespace Transformer.Imbalance

variable {c d n : ℕ}

/-- Probability bounds used in Appendix H's remainder estimate. -/
theorem probability_bounds (W : Parameters c d) (x : Fin d → ℝ) (k : Fin c) :
    0 ≤ probability W x k ∧ probability W x k ≤ 1 := by
  have hc : 0 < c := lt_of_le_of_lt (Nat.zero_le k.val) k.isLt
  unfold probability
  refine ⟨Perspective.softmaxWeight_nonneg _ _, ?_⟩
  rw [← Perspective.sum_softmaxWeight hc (scores W x)]
  exact Finset.single_le_sum (fun i _ => Perspective.softmaxWeight_nonneg (scores W x) i)
    (Finset.mem_univ k)

/-- Corrected finite-data error bound for the gradient in Proposition 2.
Taking q=C/c gives C B/c. Data boundedness is explicit. -/
theorem gradientRemainder_bound (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r : Fin d) (p q B : ℝ)
    (hn : 0 < n) (hq : 0 ≤ q) (hB : 0 ≤ B)
    (hx : ∀ i s, |x i s| ≤ B) (h : CorrectAssignment W x y k p q) :
    |gradientRemainder W x y k r| ≤ q * B := by
  have hN : (0 : ℝ) < n := by exact_mod_cast hn
  have hb : ∀ i, |if y i ≠ k then probability W (x i) k * x i r else 0| ≤ q * B := by
    intro i
    by_cases hy : y i ≠ k
    · rw [ite_eq_left hy, abs_mul, abs_of_nonneg (probability_bounds W (x i) k).1]
      exact mul_le_mul (h.2 i hy) (hx i r) (abs_nonneg _) hq
    · simp only [ite_eq_right hy, abs_zero]
      positivity
  have hs : |∑ i, if y i ≠ k then probability W (x i) k * x i r else 0| ≤ n * (q * B) := by
    calc
      _ ≤ ∑ i, |if y i ≠ k then probability W (x i) k * x i r else 0| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin n, q * B := Finset.sum_le_sum fun i _ => hb i
      _ = n * (q * B) := by simp
  unfold gradientRemainder
  rw [Finset.sum_filter, abs_div, abs_of_pos hN]
  apply (div_le_iff₀ hN).2
  simpa only [mul_comm] using hs

/-- Nonvacuity of the complete gradient-remainder assumptions; Proposition 2. -/
example : 0 < 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ i : Fin 2, ∀ r : Fin 1, |(fun _ : Fin 2 => fun _ : Fin 1 => (1 : ℝ)) i r| ≤ 1) ∧
    CorrectAssignment (fun _ _ => 0 : Parameters 2 1) (fun _ _ => 1) id 0 (1 / 2) (1 / 2) := by
  refine ⟨by decide, by norm_num, zero_le_one, fun _ _ => by norm_num, ?_⟩
  constructor <;> intro i hi <;> simp only [probability_zero] <;> norm_num

/-- Corrected finite-data error bound for Hessian entries in Proposition 2.
Taking q=C/c gives C B²/c, with the original second moments retained. -/
theorem hessianRemainder_bound (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) (r s : Fin d) (p q B : ℝ)
    (hn : 0 < n) (hq : 0 ≤ q) (hB : 0 ≤ B)
    (hx : ∀ i l, |x i l| ≤ B) (h : CorrectAssignment W x y k p q) :
    |hessianRemainder W x y k r s| ≤ q * B ^ 2 := by
  have hN : (0 : ℝ) < n := by exact_mod_cast hn
  have hb : ∀ i, |if y i ≠ k then
      probability W (x i) k * (1 - probability W (x i) k) * x i r * x i s else 0|
      ≤ q * B ^ 2 := by
    intro i
    by_cases hy : y i ≠ k
    · have hp := probability_bounds W (x i) k
      have hcoef : 0 ≤ probability W (x i) k * (1 - probability W (x i) k) :=
        mul_nonneg hp.1 (by linarith)
      have hcoef' : probability W (x i) k * (1 - probability W (x i) k) ≤ q := by
        nlinarith [h.2 i hy]
      have hprod : |x i r| * |x i s| ≤ B ^ 2 := by
        nlinarith [hx i r, hx i s, abs_nonneg (x i r), abs_nonneg (x i s)]
      rw [ite_eq_left hy, abs_mul, abs_mul, abs_of_nonneg hcoef]
      rw [mul_assoc]
      exact mul_le_mul hcoef' hprod (by positivity) hq
    · simp only [ite_eq_right hy, abs_zero]
      positivity
  have hs : |∑ i, if y i ≠ k then
      probability W (x i) k * (1 - probability W (x i) k) * x i r * x i s else 0|
      ≤ n * (q * B ^ 2) := by
    calc
      _ ≤ ∑ i, |if y i ≠ k then
          probability W (x i) k * (1 - probability W (x i) k) * x i r * x i s else 0| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin n, q * B ^ 2 := Finset.sum_le_sum fun i _ => hb i
      _ = n * (q * B ^ 2) := by simp
  unfold hessianRemainder
  rw [Finset.sum_filter, abs_div, abs_of_pos hN]
  apply (div_le_iff₀ hN).2
  simpa only [mul_comm] using hs

/-- Nonvacuity of all Hessian-remainder hypotheses; Proposition 2. -/
example : 0 < 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ i : Fin 2, ∀ r : Fin 1, |(fun _ : Fin 2 => fun _ : Fin 1 => (1 : ℝ)) i r| ≤ 1) ∧
    CorrectAssignment (fun _ _ => 0 : Parameters 2 1) (fun _ _ => 1) id 0 (1 / 2) (1 / 2) := by
  refine ⟨by decide, by norm_num, zero_le_one, fun _ _ => by norm_num, ?_⟩
  constructor <;> intro i hi <;> simp only [probability_zero] <;> norm_num

end Transformer.Imbalance
