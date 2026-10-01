/-
# Common objective thresholds on prepared scalar fibers

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.PolynomialFiberArithmetic
import Transformer.Sturm.Thresholds

noncomputable section
open scoped BigOperators

namespace Transformer.Normalization

open AnalyticPreparation Sturm

/-- The strict-below-threshold query uses a threshold independent of
the competitor's base coordinates. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
def fiberLowerBoundQuery {n d m : ℕ} (a : Fin d → Base n → ℝ)
    (b : Fin m → Fin d → Base n → ℝ) (v : Fin d → Base n → ℝ)
    (requirement : Fin m → AnalyticSignRequirement) (z : Base n) (level r : ℝ) : ℤ :=
  signConstraintQuery (distinguishedPolynomial d a z) 1
    (belowThresholdConditions (fiberRemainderPolynomial v z) level (-r) r
      (fiberPolynomialConditions b requirement z))

/-- A common threshold is below every admissible scalar root exactly
when its finite arithmetic query vanishes. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem fiberLowerBoundQuery_iff {n d m : ℕ} (a : Fin d → Base n → ℝ)
    (b : Fin m → Fin d → Base n → ℝ) (v : Fin d → Base n → ℝ)
    (requirement : Fin m → AnalyticSignRequirement) (z : Base n) (level r : ℝ) :
    (∀ t : ℝ, |t| < r → preparedPolynomial d a (z, t) = 0 →
      (∀ j, (requirement j).Holds (∑ i : Fin d, b j i z * t ^ (i : ℕ))) →
        level ≤ ∑ i : Fin d, v i z * t ^ (i : ℕ)) ↔
      fiberLowerBoundQuery a b v requirement z level r = 0 := by
  have h := polynomial_root_lower_bound_iff (distinguishedPolynomial_monic d a z).ne_zero
    (fiberRemainderPolynomial v z) (fiberPolynomialConditions b requirement z) level (-r) r
  simpa only [distinguishedPolynomial_eval, fiberRemainderPolynomial_eval,
    fiberPolynomialConditions_iff, Set.mem_Ioo, ← abs_lt, fiberLowerBoundQuery] using h

end Transformer.Normalization
