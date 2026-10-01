/-
# Real analytic germs: DivisionMaps

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/GermDivision.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.DivisionUniqueness

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- The canonical quotient, chosen from existence and made intrinsic by
`preparedGermDivision_unique`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def preparedGermDivisionQuotient {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (h : AnalyticGerm (n + 1)) :
    AnalyticGerm (n + 1) :=
  Classical.choose (exists_preparedGermDivision a ha ha0 h)

/-- The canonical coefficient vector of the degree-`< d` remainder.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def preparedGermDivisionRemainder {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (h : AnalyticGerm (n + 1)) :
    Fin d → AnalyticGerm n :=
  Classical.choose
    (Classical.choose_spec (exists_preparedGermDivision a ha ha0 h))

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedGermDivision_spec {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (h : AnalyticGerm (n + 1)) :
    IsPreparedGermDivision a ha h
      (preparedGermDivisionQuotient a ha ha0 h)
      (preparedGermDivisionRemainder a ha ha0 h) :=
  Classical.choose_spec
    (Classical.choose_spec (exists_preparedGermDivision a ha ha0 h))

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedGermDivision_eq_of_isDivision {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (h q : AnalyticGerm (n + 1))
    (r : Fin d → AnalyticGerm n)
    (hdivision : IsPreparedGermDivision a ha h q r) :
    preparedGermDivisionQuotient a ha ha0 h = q ∧
      preparedGermDivisionRemainder a ha ha0 h = r :=
  preparedGermDivision_unique a ha ha0 h _ _ _ _
    (preparedGermDivision_spec a ha ha0 h) hdivision

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem remainderPolynomialGerm_zero {n d : ℕ} :
    remainderPolynomialGerm (0 : Fin d → AnalyticGerm n) = 0 := by
  simp [remainderPolynomialGerm]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem remainderPolynomialGerm_add {n d : ℕ}
    (r s : Fin d → AnalyticGerm n) :
    remainderPolynomialGerm (r + s) =
      remainderPolynomialGerm r + remainderPolynomialGerm s := by
  simp only [remainderPolynomialGerm, Pi.add_apply, map_add, add_mul,
    Finset.sum_add_distrib]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem remainderPolynomialGerm_smul {n d : ℕ}
    (c : AnalyticGerm n) (r : Fin d → AnalyticGerm n) :
    remainderPolynomialGerm (c • r) = c • remainderPolynomialGerm r := by
  simp only [remainderPolynomialGerm, Pi.smul_apply, smul_eq_mul,
    map_mul, analyticGermSucc_smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem preparedGermDivisionRemainder_zero {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    preparedGermDivisionRemainder a ha ha0 0 = 0 := by
  have hzero : IsPreparedGermDivision a ha (0 : AnalyticGerm (n + 1)) 0
      (0 : Fin d → AnalyticGerm n) := by
    simp [IsPreparedGermDivision]
  exact (preparedGermDivision_eq_of_isDivision a ha ha0 0 0 0 hzero).2

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedGermDivisionRemainder_add {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (h k : AnalyticGerm (n + 1)) :
    preparedGermDivisionRemainder a ha ha0 (h + k) =
      preparedGermDivisionRemainder a ha ha0 h +
        preparedGermDivisionRemainder a ha ha0 k := by
  let qh := preparedGermDivisionQuotient a ha ha0 h
  let qk := preparedGermDivisionQuotient a ha ha0 k
  let rh := preparedGermDivisionRemainder a ha ha0 h
  let rk := preparedGermDivisionRemainder a ha ha0 k
  have hh := preparedGermDivision_spec a ha ha0 h
  have hk := preparedGermDivision_spec a ha ha0 k
  have hadd : IsPreparedGermDivision a ha (h + k) (qh + qk) (rh + rk) := by
    unfold IsPreparedGermDivision at hh hk ⊢
    rw [hh, hk, remainderPolynomialGerm_add]
    ring
  exact (preparedGermDivision_eq_of_isDivision a ha ha0 (h + k)
    (qh + qk) (rh + rk) hadd).2

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedGermDivisionRemainder_smul {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (c : AnalyticGerm n)
    (h : AnalyticGerm (n + 1)) :
    preparedGermDivisionRemainder a ha ha0 (c • h) =
      c • preparedGermDivisionRemainder a ha ha0 h := by
  let q := preparedGermDivisionQuotient a ha ha0 h
  let r := preparedGermDivisionRemainder a ha ha0 h
  have hh := preparedGermDivision_spec a ha ha0 h
  have hsmul : IsPreparedGermDivision a ha (c • h) (c • q) (c • r) := by
    unfold IsPreparedGermDivision at hh ⊢
    rw [hh, remainderPolynomialGerm_smul]
    simp only [analyticGermSucc_smul_eq_mul]
    ring
  exact (preparedGermDivision_eq_of_isDivision a ha ha0 (c • h)
    (c • q) (c • r) hsmul).2

/-- The base-linear coefficient-remainder map supplied by analytic
Weierstrass division.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def preparedGermDivisionRemainderLinearMap {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    AnalyticGerm (n + 1) →ₗ[AnalyticGerm n]
      (Fin d → AnalyticGerm n) where
  toFun := preparedGermDivisionRemainder a ha ha0
  map_add' := preparedGermDivisionRemainder_add a ha ha0
  map_smul' := preparedGermDivisionRemainder_smul a ha ha0


end Transformer.RealAnalyticGerms
