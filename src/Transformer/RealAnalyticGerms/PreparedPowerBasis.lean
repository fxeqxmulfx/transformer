/-
# Real analytic germs: PreparedPowerBasis

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, FiniteProjection/PreparedQuotient.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.PreparedQuotient

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem remainderPolynomialGerm_basisFun {n d : ℕ} (i : Fin d) :
    remainderPolynomialGerm
        (Pi.basisFun (AnalyticGerm n) (Fin d) i) =
      lastCoordinateGerm n ^ (i : ℕ) := by
  classical
  unfold remainderPolynomialGerm
  rw [Pi.basisFun_apply, Finset.sum_eq_single i]
  · simp
  · intro j _ hji
    simp [hji]
  · simp

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem preparedGermDivisionRemainder_lastCoordinate_pow {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (i : Fin d) :
    preparedGermDivisionRemainder a ha ha0
        (lastCoordinateGerm n ^ (i : ℕ)) =
      Pi.basisFun (AnalyticGerm n) (Fin d) i := by
  let e := Pi.basisFun (AnalyticGerm n) (Fin d) i
  have hdivision : IsPreparedGermDivision a ha
      (lastCoordinateGerm n ^ (i : ℕ)) 0 e := by
    unfold IsPreparedGermDivision
    simp only [zero_mul, zero_add]
    exact (remainderPolynomialGerm_basisFun i).symm
  exact (preparedGermDivision_eq_of_isDivision a ha ha0
    (lastCoordinateGerm n ^ (i : ℕ)) 0 e hdivision).2

/-- The explicit power basis of the prepared quotient.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def preparedQuotientBasis {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    Module.Basis (Fin d) (AnalyticGerm n)
      (AnalyticGerm (n + 1) ⧸ preparedPolynomialIdeal a ha) :=
  (Pi.basisFun (AnalyticGerm n) (Fin d)).map
    (preparedQuotientRemainderEquiv a ha ha0).symm

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem preparedQuotientBasis_apply {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (i : Fin d) :
    preparedQuotientBasis a ha ha0 i =
      Ideal.Quotient.mk (preparedPolynomialIdeal a ha)
        (lastCoordinateGerm n ^ (i : ℕ)) := by
  apply (preparedQuotientRemainderEquiv a ha ha0).injective
  rw [preparedQuotientBasis, Module.Basis.map_apply,
    LinearEquiv.apply_symm_apply,
    preparedQuotientRemainderEquiv_mk,
    preparedGermDivisionRemainder_lastCoordinate_pow]

/-- The prepared quotient is free over the lower-dimensional analytic
germ ring.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedQuotient_moduleFree {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    Module.Free (AnalyticGerm n)
      (AnalyticGerm (n + 1) ⧸ preparedPolynomialIdeal a ha) :=
  Module.Free.of_basis (preparedQuotientBasis a ha ha0)

/-- The prepared quotient is finite over the lower-dimensional analytic
germ ring.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedQuotient_moduleFinite {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    Module.Finite (AnalyticGerm n)
      (AnalyticGerm (n + 1) ⧸ preparedPolynomialIdeal a ha) :=
  Module.Finite.of_basis (preparedQuotientBasis a ha ha0)

/-- Combined finite-free conclusion for the quotient by a prepared
polynomial.  The preceding `preparedQuotientBasis` specifies its rank-`d`
power basis.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedQuotient_finiteFree {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    Module.Free (AnalyticGerm n)
        (AnalyticGerm (n + 1) ⧸ preparedPolynomialIdeal a ha) ∧
      Module.Finite (AnalyticGerm n)
        (AnalyticGerm (n + 1) ⧸ preparedPolynomialIdeal a ha) :=
  ⟨preparedQuotient_moduleFree a ha ha0,
    preparedQuotient_moduleFinite a ha ha0⟩

end Transformer.RealAnalyticGerms
