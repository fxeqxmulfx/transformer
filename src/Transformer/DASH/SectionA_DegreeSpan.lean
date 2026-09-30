/-
# DASH — degree-preserving Chebyshev expansions

arXiv:2602.02016v2, Appendix A. A degree-`d` reference polynomial can
be expressed using exactly the permitted modes `T_0,...,T_d`, which
connects polynomial approximation bounds to the source's fitted arrays.
-/

import Transformer.DASH.SectionA_FitApproximation
import Mathlib.LinearAlgebra.Finsupp.LinearCombination

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- Every polynomial of degree at most `d` is in the first `d+1` Chebyshev
span; the finite expansion preserves its degree bound.
Source: arXiv:2602.02016v2, Appendix A, the degree-`d` finite basis expansion. -/
theorem exists_chebyshev_degree_expansion (p : Polynomial ℝ) (d : ℕ)
    (hp : p.natDegree ≤ d) :
    ∃ c : Fin (d + 1) → ℝ,
      p = ∑ j : Fin (d + 1), c j • Polynomial.Chebyshev.T ℝ (j.val : ℤ) := by
  let S := Polynomial.Chebyshev.chebyshevTsequence ℝ
  have hset : Set.range (fun j : Fin (d + 1) => Polynomial.Chebyshev.T ℝ (j.val : ℤ)) =
      S '' Set.Iic d := by
    ext q
    constructor
    · rintro ⟨j, rfl⟩
      exact ⟨j.val, Nat.le_of_lt_succ j.isLt, rfl⟩
    · rintro ⟨j, hj, rfl⟩
      exact ⟨⟨j, Nat.lt_succ_of_le hj⟩, rfl⟩
  have hspan : Submodule.span ℝ
      (Set.range (fun j : Fin (d + 1) => Polynomial.Chebyshev.T ℝ (j.val : ℤ))) =
      Polynomial.degreeLE ℝ d := by
    rw [hset]
    apply S.span_degreeLE
    intro j hj
    change IsUnit (Polynomial.Chebyshev.T ℝ (j : ℤ)).leadingCoeff
    rw [Polynomial.Chebyshev.leadingCoeff_T]
    exact isUnit_iff_ne_zero.2 (pow_ne_zero _ (by norm_num))
  have hmem : p ∈ Submodule.span ℝ
      (Set.range (fun j : Fin (d + 1) => Polynomial.Chebyshev.T ℝ (j.val : ℤ))) := by
    rw [hspan, Polynomial.mem_degreeLE]
    exact Polynomial.natDegree_le_iff_degree_le.1 hp
  obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).1 hmem
  exact ⟨c, hc.symm⟩

/-- The degree constraint has a nonconstant instance,
arXiv:2602.02016v2, Appendix A. -/
example : (Polynomial.X : Polynomial ℝ).natDegree ≤ 1 := by simp

/-- A degree-bounded polynomial has a Chebyshev array evaluating to it at
every real coordinate. Source: arXiv:2602.02016v2, Appendix A, polynomial evaluation. -/
theorem exists_chebyshev_degree_eval (p : Polynomial ℝ) (d : ℕ)
    (hp : p.natDegree ≤ d) :
    ∃ c : Fin (d + 1) → ℝ, ∀ t : ℝ, p.eval t =
      ∑ j : Fin (d + 1), c j * chebT t j := by
  obtain ⟨c, hc⟩ := exists_chebyshev_degree_expansion p d hp
  refine ⟨c, ?_⟩
  intro t
  rw [hc]
  simp only [Polynomial.eval_finsetSum, Polynomial.eval_smul, smul_eq_mul,
    chebT_eq_eval]

/-- A nonzero reference polynomial has bounded degree,
arXiv:2602.02016v2, Appendix A. -/
example : (Polynomial.C (1 : ℝ)).natDegree ≤ 0 := by simp

/-- A degree-bounded polynomial approximation controls the fitted array
after interval mapping, without having to supply its Chebyshev coefficients.
Source: arXiv:2602.02016v2, Appendix A, fitting on a fixed scalar interval. -/
theorem chebFitEval_polynomial_approximation (f : ℝ → ℝ) (a b x η : ℝ) (N d : ℕ)
    (p : Polynomial ℝ) (hab : a < b) (hx : a ≤ x) (hx' : x ≤ b) (hd : d < N)
    (hη : 0 ≤ η) (hp : p.natDegree ≤ d)
    (happrox : ∀ y : ℝ, a ≤ y → y ≤ b →
      |p.eval (chebToCoordinate a b y) - f y| ≤ η) :
    |chebFitEval f a b N d x - f x| ≤ (2 * (d : ℝ) + 2) * η := by
  obtain ⟨c, hc⟩ := exists_chebyshev_degree_eval p d hp
  apply chebFitEval_approximation f a b x η N d c hab hx hx' hd hη
  intro y hy hy'
  rw [← hc]
  exact happrox y hy hy'

/-- Polynomial-fit assumptions hold for an exact constant reference,
arXiv:2602.02016v2, Appendix A. -/
example : (1 : ℝ) < 2 ∧ (1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧ (0 : ℕ) < 10 ∧
    (0 : ℝ) ≤ 0 ∧ (Polynomial.C (1 : ℝ)).natDegree ≤ 0 ∧
    (∀ y : ℝ, 1 ≤ y → y ≤ 2 →
      |(Polynomial.C (1 : ℝ)).eval (chebToCoordinate 1 2 y) - 1| ≤ 0) := by simp

end Transformer.DASH
