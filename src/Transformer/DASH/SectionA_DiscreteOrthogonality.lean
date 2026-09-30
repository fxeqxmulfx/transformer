/-
# DASH — discrete orthogonality of the fitting transform

arXiv:2602.02016v2, Appendix A, coefficient-fitting algorithm.
The cosine modes at the source's midpoint angles are orthogonal
when their degrees are strictly below the number of samples.
-/

import Transformer.DASH.SectionA_Coefficients
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- Every nonconstant cosine mode below the first alias has zero sample sum.
Source: arXiv:2602.02016v2, Appendix A, the angles used for coefficient fitting. -/
theorem chebAngle_cos_sum_zero (N k : ℕ) (hk : 0 < k) (hkN : k < 2 * N) :
    ∑ i : Fin N, Real.cos ((k : ℝ) * chebAngle N i) = 0 := by
  have hN : 0 < N := by omega
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  have hkN' : (k : ℝ) < 2 * N := by exact_mod_cast hkN
  let a : ℝ := (k : ℝ) * Real.pi / N
  have ha : 0 < a / 2 := by dsimp [a]; positivity
  have ha' : a / 2 < Real.pi := by
    dsimp [a]
    apply (div_lt_iff₀ (by norm_num : (0 : ℝ) < 2)).2
    apply (div_lt_iff₀ (by exact_mod_cast hN : (0 : ℝ) < N)).2
    nlinarith [Real.pi_pos]
  have hs := Real.sin_mul_sum_cos N a (a / 2)
  have harg : ∀ i : ℕ,
      a * (i : ℝ) + a / 2 = (k : ℝ) * ((2 * (i : ℝ) + 1) * Real.pi / (2 * N)) := by
    intro i
    dsimp [a]
    field_simp
  have hfirst : (N : ℝ) * a / 2 = (k : ℝ) * Real.pi / 2 := by
    dsimp [a]
    field_simp
  have hlast : ((N : ℝ) - 1) * a / 2 + a / 2 = (k : ℝ) * Real.pi / 2 := by
    dsimp [a]
    field_simp
    ring
  have hz : Real.sin ((k : ℝ) * Real.pi / 2) *
      Real.cos ((k : ℝ) * Real.pi / 2) = 0 := by
    have h := Real.two_mul_sin_mul_cos ((k : ℝ) * Real.pi / 2)
      ((k : ℝ) * Real.pi / 2)
    rw [sub_self, Real.sin_zero] at h
    have hdouble : (k : ℝ) * Real.pi / 2 + (k : ℝ) * Real.pi / 2 =
        (k : ℝ) * Real.pi := by ring
    rw [hdouble, Real.sin_nat_mul_pi] at h
    linarith
  rw [hfirst, hlast, hz] at hs
  have hsum : (∑ i : Fin N, Real.cos ((k : ℝ) * chebAngle N i)) =
      ∑ i ∈ Finset.range N, Real.cos (a * (i : ℝ) + a / 2) := by
    simp only [harg, chebAngle]
    exact Fin.sum_univ_eq_sum_range
      (fun i : ℕ => Real.cos ((k : ℝ) *
        ((2 * (i : ℝ) + 1) * Real.pi / (2 * N)))) N
  rw [hsum]
  exact (mul_eq_zero.mp hs).resolve_left (Real.sin_pos_of_pos_of_lt_pi ha ha').ne'

/-- Nonconstant modes in the admissible frequency range exist,
arXiv:2602.02016v2, Appendix A. -/
example : 0 < (1 : ℕ) ∧ (1 : ℕ) < 2 * 2 := by norm_num

/-- Distinct fitting modes have a zero sum of their difference mode.
Source: arXiv:2602.02016v2, Appendix A, discrete cosine orthogonality. -/
theorem chebAngle_cos_difference_sum_zero (N k l : ℕ)
    (hk : k < N) (hl : l < N) (hkl : k ≠ l) :
    ∑ i : Fin N, Real.cos (((k : ℝ) - l) * chebAngle N i) = 0 := by
  rcases lt_or_gt_of_ne hkl with hlt | hgt
  · have hcast : (k : ℝ) - l = -((l - k : ℕ) : ℝ) := by
      rw [Nat.cast_sub hlt.le]
      ring
    simp only [hcast, neg_mul, Real.cos_neg]
    apply chebAngle_cos_sum_zero N (l - k) <;> omega
  · have hcast : (k : ℝ) - l = ((k - l : ℕ) : ℝ) := by
      rw [Nat.cast_sub hgt.le]
    rw [hcast]
    apply chebAngle_cos_sum_zero N (k - l) <;> omega

/-- Distinct admissible modes exist, arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℕ) < 2 ∧ (1 : ℕ) < 2 ∧ (0 : ℕ) ≠ 1 := by norm_num

/-- The fitting cosine columns have diagonal Gram matrix: the constant
column has squared norm `N`, and the remaining columns have squared norm
`N/2`. The missing restriction `k,l<N` is required for the source's
orthogonality explanation of the halved constant coefficient.
Source: arXiv:2602.02016v2, Appendix A, coefficient-fitting algorithm. -/
theorem chebAngle_cos_orthogonality (N k l : ℕ) (hk : k < N) (hl : l < N) :
    ∑ i : Fin N, Real.cos ((k : ℝ) * chebAngle N i) *
      Real.cos ((l : ℝ) * chebAngle N i) =
      if k = l then (if k = 0 then (N : ℝ) else (N : ℝ) / 2) else 0 := by
  have hproduct : 2 * (∑ i : Fin N, Real.cos ((k : ℝ) * chebAngle N i) *
      Real.cos ((l : ℝ) * chebAngle N i)) =
      (∑ i : Fin N, Real.cos (((k : ℝ) - l) * chebAngle N i)) +
      ∑ i : Fin N, Real.cos (((k : ℝ) + l) * chebAngle N i) := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    simpa only [sub_mul, add_mul, mul_assoc] using
      Real.two_mul_cos_mul_cos ((k : ℝ) * chebAngle N i) ((l : ℝ) * chebAngle N i)
  split_ifs with hkl hk0
  · subst l
    subst k
    simp
  · subst l
    have hz := chebAngle_cos_sum_zero N (2 * k) (by omega) (by omega)
    simp only [Nat.cast_mul, Nat.cast_ofNat] at hz
    have hdouble : (k : ℝ) + k = 2 * k := by ring
    rw [sub_self] at hproduct
    simp only [zero_mul, Real.cos_zero, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, mul_one, hdouble, hz, add_zero] at hproduct
    linarith
  · have hsum := chebAngle_cos_sum_zero N (k + l) (by omega) (by omega)
    simp only [Nat.cast_add] at hsum
    rw [chebAngle_cos_difference_sum_zero N k l hk hl hkl, hsum] at hproduct
    linarith

/-- A nonempty orthogonal cosine family exists,
arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℕ) < 1000 ∧ (60 : ℕ) < 1000 := by norm_num

end Transformer.DASH
