/-
# Actual primitives of the original polynomial activation branches

ArXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, and §5.1,
`lem: quadpol` (i). A branch of the original activation is an actual
polynomial. Its primitive below is the actual finite coefficient sum
with the original powers increased by one and their coefficients divided
by the resulting positive integer.

Finite differentiation recovers that same original polynomial. Composing
with each actual neuron coordinate gives its actual ambient derivative
and an original spherical polynomial of the resulting degree bound.
Actual finite weighted sums preserve one common original degree bound.
No derivative, analyticity, or polynomial inverse is built into the data.
-/

import Transformer.Perceptron.TransformPoly
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Analytic.Polynomial

open Real Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The actual primitive with constant coefficient zero of an original
polynomial branch. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
Its actual derivative and degree are proved below. -/
noncomputable def polynomialPrimitive (P : Polynomial ℝ) (s : ℝ) : ℝ :=
  ∑ i ∈ Finset.range (P.natDegree + 1), P.coeff i / (i + 1) * s ^ (i + 1)

/-- The actual primitive is the evaluation of its original coefficient
polynomial. Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
Every original coefficient, positive denominator and raised power agrees. -/
theorem polynomialPrimitive_eq_eval (P : Polynomial ℝ) (s : ℝ) :
    polynomialPrimitive P s = (∑ i ∈ Finset.range (P.natDegree + 1),
      Polynomial.C (P.coeff i / (i + 1)) * Polynomial.X ^ (i + 1)).eval s := by
  rw [Polynomial.eval_finsetSum]
  unfold polynomialPrimitive
  apply Finset.sum_congr rfl
  intro i _
  simp only [Polynomial.eval_mul, Polynomial.eval_C, Polynomial.eval_pow, Polynomial.eval_X]

/-- The actual primitive's derivative is the original polynomial value.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, the polynomial
first-variation branch. Every actual coefficient is differentiated. -/
theorem hasDerivAt_polynomialPrimitive (P : Polynomial ℝ) (s : ℝ) :
    HasDerivAt (polynomialPrimitive P) (P.eval s) s := by
  have h := HasDerivAt.fun_sum (u := Finset.range (P.natDegree + 1))
    (fun i _ => (hasDerivAt_pow (i + 1) s).const_mul (P.coeff i / (i + 1)))
  apply h.congr_deriv
  rw [Polynomial.eval_eq_sum_range]
  apply Finset.sum_congr rfl
  intro i _
  simp only [Nat.add_sub_cancel, Nat.cast_add, Nat.cast_one]
  have hne : (i + 1 : ℝ) ≠ 0 := by positivity
  field_simp

/-- The original branch primitive has actual value zero at zero.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
Its otherwise arbitrary additive constant is absorbed in the branch's
constant first-variation value. -/
theorem polynomialPrimitive_zero (P : Polynomial ℝ) : polynomialPrimitive P 0 = 0 := by
  unfold polynomialPrimitive
  apply Finset.sum_eq_zero
  intro i _
  simp

/-- The original neuron coordinate composed with the actual primitive
has its actual covector derivative represented by the original neuron.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
The scalar derivative above and the original linear coordinate give it. -/
theorem hasFDerivAt_polynomialPrimitive_inner {d : ℕ} (P : Polynomial ℝ)
    (a x : EucSpace d) :
    HasFDerivAt (fun z => polynomialPrimitive P (inner (𝕜 := ℝ) a z))
      (innerSL ℝ ((P.eval (inner (𝕜 := ℝ) a x)) • a)) x := by
  have h := (hasDerivAt_polynomialPrimitive P (inner (𝕜 := ℝ) a x)).comp_hasFDerivAt x
    (innerSL ℝ a).hasFDerivAt
  apply h.congr_fderiv
  ext v
  change P.eval (inner (𝕜 := ℝ) a x) * inner (𝕜 := ℝ) a v =
    inner (𝕜 := ℝ) (P.eval (inner (𝕜 := ℝ) a x) • a) v
  rw [real_inner_smul_left]

/-- An original polynomial branch evaluated at an actual neuron
coordinate is analytic on the actual ambient space.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, the branch field.
The actual finite polynomial expansion supplies analyticity. -/
theorem analyticOnNhd_polynomial_eval_inner {d : ℕ} (P : Polynomial ℝ) (a : EucSpace d) :
    AnalyticOnNhd ℝ (fun x => P.eval (inner (𝕜 := ℝ) a x)) univ := by
  intro x _
  have heq : (fun z : EucSpace d => P.eval (inner (𝕜 := ℝ) a z)) =
      fun z => ∑ i ∈ Finset.range (P.natDegree + 1), P.coeff i * inner (𝕜 := ℝ) a z ^ i := by
    funext z
    exact Polynomial.eval_eq_sum_range _
  rw [heq]
  apply Finset.analyticAt_fun_sum
  intro i _
  exact analyticAt_const.fun_mul (((innerSL ℝ a).analyticAt x).pow i)

/-- The actual original branch primitive is analytic along each actual
neuron coordinate. Source: arXiv:2601.21366v2, §5.2,
`rem:piecewise-poly`, the analytic scalar first variation on a branch. -/
theorem analyticOnNhd_polynomialPrimitive_inner {d : ℕ} (P : Polynomial ℝ) (a : EucSpace d) :
    AnalyticOnNhd ℝ (fun x => polynomialPrimitive P (inner (𝕜 := ℝ) a x)) univ := by
  intro x _
  unfold polynomialPrimitive
  apply Finset.analyticAt_fun_sum
  intro i _
  exact analyticAt_const.fun_mul (((innerSL ℝ a).analyticAt x).pow (i + 1))

/-- Each actual original branch primitive on the original sphere has
degree at most one more than the original branch polynomial.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`, with §5.1,
`lem: quadpol` (i). Every actual finite power has its original degree. -/
theorem isPolyOfDegreeLE_polynomialPrimitive_inner {d : ℕ} (P : Polynomial ℝ) (a : EucSpace d) :
    IsPolyOfDegreeLE (P.natDegree + 1)
      (fun x : SSphere d => polynomialPrimitive P (inner (𝕜 := ℝ) a (x : EucSpace d))) := by
  unfold polynomialPrimitive
  apply isPolyOfDegreeLE_finsetSum
  intro i hi
  have hin : i < P.natDegree + 1 := Finset.mem_range.mp hi
  exact isPolyOfDegreeLE_const_mul (P.coeff i / (i + 1))
    (isPolyOfDegreeLE_mono (isPolyOfDegreeLE_pow (isPolyOfDegreeLE_inner a) (i + 1)) (by omega))

/-- An actual finite weighted family of original primitive branches
retains its common original polynomial degree bound.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`.
No original neuron, weight or branch is dropped from the finite sum. -/
theorem isPolyOfDegreeLE_sum_polynomialPrimitives {d k : ℕ} {ι : Type*} [Fintype ι]
    (ω : ι → ℝ) (a : ι → EucSpace d) (P : ι → Polynomial ℝ)
    (hP : ∀ j, (P j).natDegree + 1 ≤ k) :
    IsPolyOfDegreeLE k (fun x : SSphere d =>
      ∑ j, ω j * polynomialPrimitive (P j) (inner (𝕜 := ℝ) (a j) (x : EucSpace d))) := by
  apply isPolyOfDegreeLE_finsetSum
  intro j _
  exact isPolyOfDegreeLE_const_mul (ω j)
    (isPolyOfDegreeLE_mono (isPolyOfDegreeLE_polynomialPrimitive_inner (P j) (a j)) (hP j))

/-- All actual degree inputs hold jointly for original linear branches
at both original circle neurons. The primitive itself has the actual
derivative of the linear activation at a nonzero original input.
Source: arXiv:2601.21366v2, §5.2, `rem:piecewise-poly`. -/
example : (∀ j : Idx 2, ((fun _ : Idx 2 => (Polynomial.X : Polynomial ℝ)) j).natDegree + 1 ≤ 2) ∧
    HasDerivAt (polynomialPrimitive Polynomial.X) (1 : ℝ) 1 := by
  refine ⟨fun _ => by simp, ?_⟩
  simpa using hasDerivAt_polynomialPrimitive Polynomial.X 1

end Transformer.Perceptron
