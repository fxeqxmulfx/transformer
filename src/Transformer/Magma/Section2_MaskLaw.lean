/-
# Independent block masks

Formalization of arXiv:2602.15322v1, Section 2, eq:rsu_def, and Appendix A.1.
Conditioning on the current parameters and base update leaves a finite
product Bernoulli law. The law is constructed, not postulated through moment
identities. It covers p=1; probability statements require 0<=p<=1.
-/

import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Bernoulli mass, with true meaning a surviving update.
Source: arXiv:2602.15322v1, Section 2, eq:rsu_def. -/
def bitMass (p : ℝ) (bit : Bool) : ℝ := if bit then p else 1 - p

/-- Actual independent product law over the block masks.
Source: arXiv:2602.15322v1, Section 2 and Appendix A.1. -/
def maskMass (p : ℝ) (mask : ι → Bool) : ℝ := ∏ b, bitMass p (mask b)

/-- Expectation over the finite product law after fixing the base update.
Source: arXiv:2602.15322v1, Section 2 and Appendix A.1. -/
def maskExpectation (p : ℝ) (f : (ι → Bool) → ℝ) : ℝ :=
  ∑ mask, maskMass p mask * f mask

/-- The product law factors expectations of blockwise products.
Source: arXiv:2602.15322v1, Appendix A.1, independence calculation. -/
theorem maskExpectation_prod (p : ℝ) (f : ι → Bool → ℝ) :
    maskExpectation p (fun mask => ∏ b, f b (mask b)) =
      ∏ b, ((1 - p) * f b false + p * f b true) := by
  unfold maskExpectation maskMass
  simp only [← Finset.prod_mul_distrib]
  rw [← Fintype.prod_sum (fun (b : ι) (bit : Bool) => bitMass p bit * f b bit)]
  apply Finset.prod_congr rfl
  intro b hb
  simp only [Fintype.sum_bool, bitMass, Bool.false_eq_true, ite_false, ite_true]
  ring

/-- Mask masses sum to one, including the degenerate p=1 case.
Source: arXiv:2602.15322v1, Section 2, Bernoulli(p). -/
theorem maskMass_sum (p : ℝ) : ∑ mask : ι → Bool, maskMass p mask = 1 := by
  have h := maskExpectation_prod p (fun (_ : ι) (_ : Bool) => (1 : ℝ))
  simpa [maskExpectation, show 1 - p + p = 1 by ring] using h

omit [DecidableEq ι] in
/-- On the probability domain every mask mass is nonnegative.
Source: arXiv:2602.15322v1, Section 2, p in (0,1]. -/
theorem maskMass_nonneg (p : ℝ) (hp : 0 ≤ p) (hp' : p ≤ 1) (mask : ι → Bool) :
    0 ≤ maskMass p mask := by
  apply Finset.prod_nonneg
  intro b hb
  unfold bitMass
  split <;> linarith

/-- The probability domain is nonempty. Source: arXiv:2602.15322v1,
Algorithm 1, p=1/2. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Marginal expectation of a block's mask statistic.
Source: arXiv:2602.15322v1, Appendix A.1. -/
theorem maskExpectation_coordinate (p : ℝ) (b : ι) (f : Bool → ℝ) :
    maskExpectation p (fun mask => f (mask b)) = (1 - p) * f false + p * f true := by
  have h := maskExpectation_prod p (fun j bit => if j = b then f bit else 1)
  have hprod (mask : ι → Bool) :
      (∏ j, if j = b then f (mask j) else 1) = f (mask b) := by simp
  simp only [hprod] at h
  rw [h]
  have heq (j : ι) :
      (1 - p) * (if j = b then f false else 1) + p * (if j = b then f true else 1) =
        if j = b then (1 - p) * f false + p * f true else 1 := by
    split <;> simp_all
  simp only [heq]
  simp

/-- Statistics on distinct block bits are independent.
Source: arXiv:2602.15322v1, Appendix A.1, off-diagonal Hessian term. -/
theorem maskExpectation_pair (p : ℝ) (b c : ι) (hbc : b ≠ c) (f g : Bool → ℝ) :
    maskExpectation p (fun mask => f (mask b) * g (mask c)) =
      ((1 - p) * f false + p * f true) * ((1 - p) * g false + p * g true) := by
  have h := maskExpectation_prod p
    (fun j bit => (if j = b then f bit else 1) * (if j = c then g bit else 1))
  have hprod (mask : ι → Bool) :
      (∏ j, (if j = b then f (mask j) else 1) * (if j = c then g (mask j) else 1)) =
        f (mask b) * g (mask c) := by rw [Finset.prod_mul_distrib]; simp
  simp only [hprod] at h
  rw [h]
  have heq (j : ι) :
      (1 - p) * ((if j = b then f false else 1) * (if j = c then g false else 1)) +
        p * ((if j = b then f true else 1) * (if j = c then g true else 1)) =
        (if j = b then (1 - p) * f false + p * f true else 1) *
          (if j = c then (1 - p) * g false + p * g true else 1) := by
    by_cases hb : j = b <;> by_cases hc : j = c <;> simp_all
  simp only [heq]
  rw [Finset.prod_mul_distrib]
  simp

/-- Distinct block indices exist. Source: arXiv:2602.15322v1, Section 2. -/
example : (0 : Fin 2) ≠ 1 := by decide

/-- Finite expectation is linear in sums. Source: arXiv:2602.15322v1,
Appendix A.1, taking expectation of the Taylor expansion. -/
theorem maskExpectation_add (p : ℝ) (f g : (ι → Bool) → ℝ) :
    maskExpectation p (fun mask => f mask + g mask) =
      maskExpectation p f + maskExpectation p g := by
  simp [maskExpectation, mul_add, Finset.sum_add_distrib]

/-- Finite expectation commutes with scalar multiplication. Source:
arXiv:2602.15322v1, Appendix A.1. -/
theorem maskExpectation_mul (p a : ℝ) (f : (ι → Bool) → ℝ) :
    maskExpectation p (fun mask => a * f mask) = a * maskExpectation p f := by
  simp only [maskExpectation]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro mask hm
  ring

/-- Constants retain their value because the law has total mass one.
Source: arXiv:2602.15322v1, Appendix A.1. -/
theorem maskExpectation_const (p a : ℝ) :
    maskExpectation (ι := ι) p (fun _ => a) = a := by
  simp only [maskExpectation, ← Finset.sum_mul, maskMass_sum, one_mul]

end Transformer.Magma
