/-
# Uniform bounds for real roots of analytic monic families

The elementary monic root bound gives a compact interval containing all
real roots at sufficiently small parameters. It applies after Newton
normalization, where the central polynomial need not be a pure power.
-/

import Transformer.Normalization.PolynomialFamily
import Mathlib.Algebra.Order.BigOperators.Group.Finset

open Filter Finset
open scoped BigOperators

namespace Transformer.Normalization

/-- Every real root of a positive-degree monic polynomial is bounded by
one plus the sum of the absolute lower coefficients. Auxiliary compactness
input for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_root_abs_bound {n : ℕ} (a : Fin (n + 1) → ℝ → ℝ)
    (t y : ℝ) (hroot : polynomialFamily (n + 1) a (t, y) = 0) :
    |y| ≤ 1 + ∑ i : Fin (n + 1), |a i t| := by
  let S : ℝ := ∑ i : Fin (n + 1), |a i t|
  have hS : 0 ≤ S := sum_nonneg (fun i _ => abs_nonneg _)
  by_contra hn
  have hylarge : 1 + S < |y| := lt_of_not_ge hn
  have hyone : 1 ≤ |y| := by linarith
  have hypos : 0 < |y| := by linarith
  have habs : |y| ^ (n + 1) = |∑ i : Fin (n + 1), a i t * y ^ i.val| := by
    have heq : y ^ (n + 1) = -(∑ i : Fin (n + 1), a i t * y ^ i.val) := by
      dsimp only [polynomialFamily] at hroot
      linarith
    rw [← abs_pow, heq, abs_neg]
  have hsum : |y| ^ (n + 1) ≤ S * |y| ^ n := by
    rw [habs]
    calc
      |∑ i : Fin (n + 1), a i t * y ^ i.val| ≤
          ∑ i : Fin (n + 1), |a i t * y ^ i.val| := abs_sum_le_sum_abs _ _
      _ = ∑ i : Fin (n + 1), |a i t| * |y| ^ i.val := by
        simp only [abs_mul, abs_pow]
      _ ≤ ∑ i : Fin (n + 1), |a i t| * |y| ^ n := by
        apply sum_le_sum
        intro i hi
        exact mul_le_mul_of_nonneg_left (pow_le_pow_right₀ hyone (by omega)) (abs_nonneg _)
      _ = S * |y| ^ n := by rw [sum_mul]
  have hpowpos : 0 < |y| ^ n := pow_pos hypos n
  rw [pow_succ] at hsum
  have hyle : |y| ≤ S := (mul_le_mul_iff_of_pos_left hpowpos).mp (by
    simpa only [mul_comm] using hsum)
  linarith

/-- Analytic coefficient germs give a positive uniform bound for every
real root at every sufficiently small parameter. All roots, including
repeated roots, lie in one compact interval. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_polynomial_roots_bounded {n : ℕ} (a : Fin (n + 1) → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) :
    ∃ B > 0, ∀ᶠ t in nhds (0 : ℝ), ∀ y : ℝ,
      polynomialFamily (n + 1) a (t, y) = 0 → |y| ≤ B := by
  let S : ℝ → ℝ := fun t => ∑ i : Fin (n + 1), |a i t|
  have hS : ContinuousAt S 0 := by
    exact tendsto_finsetSum Finset.univ (fun i hi => (ha i).continuousAt.abs.tendsto)
  have hnear : ∀ᶠ t in nhds (0 : ℝ), S t < S 0 + 1 :=
    hS.eventually (Iio_mem_nhds (by linarith))
  refine ⟨S 0 + 2, by dsimp only [S]; positivity, ?_⟩
  filter_upwards [hnear] with t ht
  intro y hy
  have h := polynomial_root_abs_bound a t y hy
  change |y| ≤ 1 + S t at h
  linarith

/-- The cubic `y³ - t²` has analytic coefficients and real roots for
positive parameters. Its positive root at parameter one simultaneously
exercises the elementary root bound and the analytic uniform bound.
Auxiliary example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then -(t ^ 2) else 0
    (|(1 : ℝ)| ≤ 1 + ∑ i : Fin 3, |a i 1|) ∧
      ∃ B > 0, ∀ᶠ t in nhds (0 : ℝ), ∀ y : ℝ,
        polynomialFamily 3 a (t, y) = 0 → |y| ≤ B := by
  constructor
  · apply polynomial_root_abs_bound
    norm_num [polynomialFamily, Fin.sum_univ_succ]
  · apply analytic_polynomial_roots_bounded
    intro i
    split_ifs
    · exact (analyticAt_id.fun_pow 2).fun_neg
    · exact analyticAt_const

end Transformer.Normalization
