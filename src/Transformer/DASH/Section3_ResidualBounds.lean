/-
# DASH — quantitative consequences of the Newton recurrence

arXiv:2602.02016v2, §3.2–3.4. Exact residual identities explain
quadratic convergence near one and the slow escape from zero.
-/

import Transformer.DASH.Section3_CoupledNewton

noncomputable section

namespace Transformer.DASH

/-- Exact order-two residual recurrence. It also applies to `Y_k Z_k`
in NDB, whose scalar product follows the same polynomial.
Source: arXiv:2602.02016v2, §3.2–3.4, `equation:CN-X-M` and `equation:NDB-Y-Z`. -/
theorem cn_two_residual_identity (x : ℝ) :
    1 - cnMap 2 x = (1 - x) ^ 2 * (4 - x) / 4 := by
  norm_num [cnMap, cnFactor]
  ring

/-- For `0≤x≤1` the next residual lies between 3/4 and 1 times the
squared current residual. This is a precise quadratic-convergence bound.
Source: arXiv:2602.02016v2, §3.4, the discussion of eigenvalue convergence. -/
theorem cn_two_residual_bounds (x : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 1) :
    3 / 4 * (1 - x) ^ 2 ≤ 1 - cnMap 2 x ∧
      1 - cnMap 2 x ≤ (1 - x) ^ 2 := by
  rw [cn_two_residual_identity]
  constructor <;> nlinarith [sq_nonneg (1 - x), mul_nonneg hx (sq_nonneg (1 - x)),
    mul_nonneg (sub_nonneg.mpr hx') (sq_nonneg (1 - x))]

/-- The quadratic region is nonempty, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Unit-scaled iterates stay in the unit interval,
arXiv:2602.02016v2, §3.2–3.4, the scalar Newton recurrence. -/
theorem cn_two_unit_bounds (a : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (k : ℕ) :
    0 ≤ cnScalarM 2 a 1 k ∧ cnScalarM 2 a 1 k ≤ 1 := by
  induction k with
  | zero => simpa [cnScalarM] using And.intro ha ha'
  | succ k ih =>
    exact ⟨mul_nonneg ih.1 (sq_nonneg _),
      cnMap_le_one 2 _ (by norm_num) ih.1 (by norm_num; linarith [ih.2])⟩

/-- Unit-scaling assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- The residual is at most `(1-a)^(2^k)` in the unit-scaled recurrence.
This is a proved accuracy bound, separate from the paper's plotted step counts.
Source: arXiv:2602.02016v2, §3.4, fixed-precision convergence. -/
theorem cn_two_residual_bound (a : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (k : ℕ) :
    1 - cnScalarM 2 a 1 k ≤ (1 - a) ^ (2 ^ k) := by
  induction k with
  | zero => simp [cnScalarM]
  | succ k ih =>
    have hm := cn_two_unit_bounds a ha ha' k
    calc
      1 - cnScalarM 2 a 1 (k + 1) ≤ (1 - cnScalarM 2 a 1 k) ^ 2 :=
        (cn_two_residual_bounds _ hm.1 hm.2).2
      _ ≤ ((1 - a) ^ (2 ^ k)) ^ 2 :=
        pow_le_pow_left₀ (sub_nonneg.mpr hm.2) ih 2
      _ = (1 - a) ^ (2 ^ (k + 1)) := by
        rw [show 2 ^ (k + 1) = 2 ^ k * 2 from pow_succ 2 k, pow_mul]

/-- Accuracy-bound assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- A product iterate cannot grow faster than `(9/4)^k a` on `[0,1]`.
Small initial eigenvalues therefore remain far from one for a fixed budget.
Source: arXiv:2602.02016v2, §3.4, Frobenius normalization pushing eigenvalues to zero. -/
theorem cn_two_growth_bound (a : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (k : ℕ) :
    cnScalarM 2 a 1 k ≤ (9 / 4 : ℝ) ^ k * a := by
  induction k with
  | zero => simp [cnScalarM]
  | succ k ih =>
    have hm := cn_two_unit_bounds a ha ha' k
    have hfactor : cnFactor 2 (cnScalarM 2 a 1 k) ^ 2 ≤ (9 / 4 : ℝ) := by
      norm_num [cnFactor]
      nlinarith [mul_nonneg hm.1 (sub_nonneg.mpr hm.2)]
    calc
      cnScalarM 2 a 1 (k + 1) ≤ (9 / 4 : ℝ) * cnScalarM 2 a 1 k := by
        exact (mul_le_mul_of_nonneg_left hfactor hm.1).trans_eq (mul_comm _ _)
      _ ≤ (9 / 4 : ℝ) * ((9 / 4 : ℝ) ^ k * a) :=
        mul_le_mul_of_nonneg_left ih (by norm_num)
      _ = (9 / 4 : ℝ) ^ (k + 1) * a := by rw [pow_succ]; ring

/-- Growth-bound assumptions are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Two calls to the exact NDB limit have inverse-fourth-root exponent.
Source: arXiv:2602.02016v2, §3.3, `(A^(1/2))^(-1/2)=A^(-1/4)`. -/
theorem inverse_fourth_root_scalar (a : ℝ) (ha : 0 ≤ a) :
    (a ^ (1 / 2 : ℝ)) ^ (-(1 / 2 : ℝ)) = a ^ (-(1 / 4 : ℝ)) := by
  rw [← Real.rpow_mul ha]
  norm_num

/-- Nonnegative inputs exist, arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) ≤ 1 := by norm_num

/-- The order-two Newton product map preserves order below one, including
the source's unit interval. Source: arXiv:2602.02016v2, §3.4, how smaller
normalized eigenvalues delay convergence. -/
theorem cn_two_map_monotone (x y : ℝ) (hxy : x ≤ y) (hy : y ≤ 1) :
    cnMap 2 x ≤ cnMap 2 y := by
  have hx' : 0 ≤ 1 - x := by linarith
  have hy' : 0 ≤ 1 - y := by linarith
  have hid : cnMap 2 y - cnMap 2 x = (y - x) *
      (3 * (1 - x) + 3 * (1 - y) + (1 - x) ^ 2 +
        (1 - x) * (1 - y) + (1 - y) ^ 2) / 4 := by
    norm_num [cnMap, cnFactor]
    ring
  have hnonneg : 0 ≤ (y - x) *
      (3 * (1 - x) + 3 * (1 - y) + (1 - x) ^ 2 +
        (1 - x) * (1 - y) + (1 - y) ^ 2) / 4 := by
    apply div_nonneg _ (by norm_num)
    apply mul_nonneg (sub_nonneg.mpr hxy)
    positivity
  linarith

/-- Ordered unit-interval eigenvalues exist, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Every fixed-budget product iterate is ordered by its initial eigenvalue.
Source: arXiv:2602.02016v2, §3.4, the convergence effect of matrix scaling. -/
theorem cn_two_iterate_monotone (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b)
    (hb : b ≤ 1) (k : ℕ) : cnScalarM 2 a 1 k ≤ cnScalarM 2 b 1 k := by
  have hb' : 0 ≤ b := le_trans ha hab
  induction k with
  | zero => simpa [cnScalarM] using hab
  | succ k ih =>
    exact cn_two_map_monotone _ _ ih
      (cn_two_unit_bounds b hb' hb k).2

/-- Iteration-order hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- If the smaller eigenvalue meets the identity-residual stopping criterion,
the larger one already meets it at the same iteration. This rigorously supports
the scaling discussion for CN/NDB product residuals; it does not assert the
source's informal inverse proportionality for plotted absolute root errors.
Source: arXiv:2602.02016v2, §3.2 (early stopping) and §3.4 (scaling). -/
theorem cn_two_residual_stopping_order (a b δ : ℝ) (ha : 0 ≤ a) (hab : a ≤ b)
    (hb : b ≤ 1) (k : ℕ) (hstop : 1 - cnScalarM 2 a 1 k ≤ δ) :
    1 - cnScalarM 2 b 1 k ≤ δ := by
  linarith [cn_two_iterate_monotone a b ha hab hb k]

/-- The ordered stopping assumptions are satisfiable,
arXiv:2602.02016v2, §3.2 and §3.4. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    1 - cnScalarM 2 (1 / 4) 1 0 ≤ (3 / 4 : ℝ) := by norm_num [cnScalarM]

end Transformer.DASH
