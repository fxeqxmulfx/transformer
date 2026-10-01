/-
# Eliminating finite polynomial sign constraints at real roots

A finite recursion combines Sturm--Tarski queries. The resulting
expression contains no quantified root variable. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.SignMasks

noncomputable section
open Polynomial Transformer.Normalization
open scoped BigOperators Classical

namespace Transformer.Sturm

/-- A finite arithmetic expression for a real-root query restricted by
the given signs. The multiplier accumulates query products, and every
terminal leaf is the proved arithmetic Sturm--Tarski construction.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def signConstraintQuery (p multiplier : Polynomial ℝ) :
    List (Polynomial ℝ × AnalyticSignRequirement) → ℤ
  | [] => realRootQuery p multiplier
  | (q, requirement) :: rest =>
      signMaskCoefficients requirement 0 * signConstraintQuery p multiplier rest +
      signMaskCoefficients requirement 1 * signConstraintQuery p (multiplier * q) rest +
      signMaskCoefficients requirement 2 * signConstraintQuery p (multiplier * q ^ 2) rest

/-- The finite query recursion equals the weighted sum over the actual
distinct real roots. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem signConstraintQuery_eq_sum {p : Polynomial ℝ} (hp : p ≠ 0)
    (multiplier : Polynomial ℝ) (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    signConstraintQuery p multiplier conditions = ∑ x ∈ p.roots.toFinset,
      (SignType.sign (multiplier.eval x) : ℤ) * signConstraintWeight conditions x := by
  induction conditions generalizing multiplier with
  | nil => simpa only [signConstraintQuery, signConstraintWeight, mul_one] using
      realRootQuery_eq hp multiplier
  | cons c rest ih =>
      rcases c with ⟨q, requirement⟩
      rw [signConstraintQuery, ih, ih, ih]
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro x _
      rw [signConstraintWeight, ← signMaskCoefficients_formula]
      simp only [eval_mul, eval_pow, sign_mul, sign_pow, SignType.coe_mul, SignType.coe_pow]
      ring

/-- With multiplier one, the finite expression is `2^m` times the
number of real roots satisfying all `m` sign requirements. Repeated
roots, identically zero constraints, and weak inequalities are retained
exactly. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem signConstraintQuery_eq_count {p : Polynomial ℝ} (hp : p ≠ 0)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    signConstraintQuery p 1 conditions = (2 : ℤ) ^ conditions.length *
      ((p.roots.toFinset.filter (fun x =>
        ∀ c ∈ conditions, c.2.Holds (c.1.eval x))).card : ℤ) := by
  rw [signConstraintQuery_eq_sum hp]
  simp only [eval_one, sign_one, SignType.coe_one, one_mul, signConstraintWeight_value]
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
  exact mul_comm _ _

/-- Exact existential elimination on a nonzero polynomial's real
roots with arbitrary finite sign constraints. The right side consists
only of finitely many polynomial operations and sign variations.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_root_with_signs_iff {p : Polynomial ℝ} (hp : p ≠ 0)
    (conditions : List (Polynomial ℝ × AnalyticSignRequirement)) :
    (∃ x : ℝ, p.eval x = 0 ∧ ∀ c ∈ conditions, c.2.Holds (c.1.eval x)) ↔
      0 < signConstraintQuery p 1 conditions := by
  rw [signConstraintQuery_eq_count hp, mul_pos_iff_of_pos_left (pow_pos (by norm_num) _),
    Nat.cast_pos, Finset.card_pos]
  simp only [Finset.Nonempty, Finset.mem_filter, Multiset.mem_toFinset, mem_roots hp,
    Polynomial.IsRoot]

/-- A polynomial with a repeated root and simultaneous zero, strict,
and weak sign requirements is a valid nonzero input. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X ^ 4 : Polynomial ℝ) ≠ 0 ∧
    ∃ x : ℝ, (X ^ 4 : Polynomial ℝ).eval x = 0 ∧
      ∀ c ∈ ([(X, .zero), (1, .positive), (X ^ 2, .nonnegative)] :
        List (Polynomial ℝ × AnalyticSignRequirement)), c.2.Holds (c.1.eval x) := by
  refine ⟨pow_ne_zero _ X_ne_zero, 0, by simp, ?_⟩
  simp [AnalyticSignRequirement.Holds]

end Transformer.Sturm
