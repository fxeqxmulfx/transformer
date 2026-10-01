/-
# Arithmetic elimination of the scalar fiber minimum comparison

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.ConstrainedMinima
import Transformer.AnalyticPreparation.FiniteFibers
import Mathlib.Data.List.OfFn

noncomputable section
open Polynomial
open scoped BigOperators

namespace Transformer.Normalization

open AnalyticPreparation Sturm

/-- The polynomial of degree below `d` representing an analytic
remainder at fixed base coordinates. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
def fiberRemainderPolynomial {n d : ℕ} (b : Fin d → Base n → ℝ) (z : Base n) :
    Polynomial ℝ := ∑ i : Fin d, C (b i z) * X ^ (i : ℕ)

/-- Evaluation is exactly the remainder value used by analytic
division. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem fiberRemainderPolynomial_eval {n d : ℕ} (b : Fin d → Base n → ℝ)
    (z : Base n) (y : ℝ) :
    (fiberRemainderPolynomial b z).eval y = ∑ i : Fin d, b i z * y ^ (i : ℕ) := by
  simp [fiberRemainderPolynomial, eval_finsetSum]

/-- All indexed sign remainders retained as a finite list of actual
polynomial constraints. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def fiberPolynomialConditions {n d m : ℕ} (b : Fin m → Fin d → Base n → ℝ)
    (requirement : Fin m → AnalyticSignRequirement) (z : Base n) :
    List (Polynomial ℝ × AnalyticSignRequirement) :=
  List.ofFn (fun j => (fiberRemainderPolynomial (b j) z, requirement j))

/-- The finite-list polynomial signs are equivalent to every original
indexed remainder sign. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem fiberPolynomialConditions_iff {n d m : ℕ} (b : Fin m → Fin d → Base n → ℝ)
    (requirement : Fin m → AnalyticSignRequirement) (z : Base n) (y : ℝ) :
    (∀ c ∈ fiberPolynomialConditions b requirement z, c.2.Holds (c.1.eval y)) ↔
      ∀ j, (requirement j).Holds (∑ i : Fin d, b j i z * y ^ (i : ℕ)) := by
  simp only [fiberPolynomialConditions, List.forall_mem_ofFn_iff, fiberRemainderPolynomial_eval]

/-- The finite arithmetic query for the absence of a strictly better
admissible root in the actual open scalar box. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def fiberMinimumQuery {n d m : ℕ} (a : Fin d → Base n → ℝ)
    (b : Fin m → Fin d → Base n → ℝ) (v : Fin d → Base n → ℝ)
    (requirement : Fin m → AnalyticSignRequirement) (z : Base n) (y r : ℝ) : ℤ :=
  signConstraintQuery (distinguishedPolynomial d a z) 1
    (betterRootConditions (fiberRemainderPolynomial v z) y (-r) r
      (fiberPolynomialConditions b requirement z))

/-- The scalar universal comparison on a prepared fiber is equivalent
to one arithmetic query being zero. All finite sign constraints and
the strict scalar box are kept; no curve in the base coordinates is
assumed. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem fiberMinimumQuery_iff {n d m : ℕ} (a : Fin d → Base n → ℝ)
    (b : Fin m → Fin d → Base n → ℝ) (v : Fin d → Base n → ℝ)
    (requirement : Fin m → AnalyticSignRequirement) (z : Base n) (y r : ℝ) :
    (∀ t : ℝ, |t| < r → preparedPolynomial d a (z, t) = 0 →
      (∀ j, (requirement j).Holds (∑ i : Fin d, b j i z * t ^ (i : ℕ))) →
        (∑ i : Fin d, v i z * y ^ (i : ℕ)) ≤ ∑ i : Fin d, v i z * t ^ (i : ℕ)) ↔
      fiberMinimumQuery a b v requirement z y r = 0 := by
  have h := polynomial_root_minimum_iff (distinguishedPolynomial_monic d a z).ne_zero
    (fiberRemainderPolynomial v z) (fiberPolynomialConditions b requirement z) y (-r) r
  simpa only [distinguishedPolynomial_eval, fiberRemainderPolynomial_eval,
    fiberPolynomialConditions_iff, Set.mem_Ioo, ← abs_lt, fiberMinimumQuery] using h

end Transformer.Normalization
