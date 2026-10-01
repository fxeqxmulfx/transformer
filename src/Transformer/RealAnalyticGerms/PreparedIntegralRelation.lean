/-
# Integral relations on a prepared real analytic hypersurface

The actual finite power basis of a prepared quotient makes every analytic
function on it integral over the parameter-germ ring. Hence its values obey
a monic polynomial relation with analytic parameter coefficients.
-/

import Transformer.RealAnalyticGerms.PreparedPowerBasis
import Mathlib.LinearAlgebra.Charpoly.Basic

noncomputable section
namespace Transformer.RealAnalyticGerms

open AnalyticPreparation

/-- The characteristic polynomial of multiplication by an analytic germ
gives a monic relation whose degree is exactly the degree of the prepared
equation. This uses its actual rank-`d` power basis, including repeated
roots. Auxiliary for finite analytic elimination in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem prepared_analytic_germ_relation_degree {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (g : AnalyticGerm (n + 1)) :
    ∃ p : Polynomial (AnalyticGerm n), p.Monic ∧ p.natDegree = d ∧
      Polynomial.aeval g p ∈ preparedPolynomialIdeal a ha := by
  let I := preparedPolynomialIdeal a ha
  let : Module.Finite (AnalyticGerm n) (AnalyticGerm (n + 1) ⧸ I) :=
    preparedQuotient_moduleFinite a ha ha0
  let : Module.Free (AnalyticGerm n) (AnalyticGerm (n + 1) ⧸ I) :=
    preparedQuotient_moduleFree a ha ha0
  let q := Ideal.Quotient.mkₐ (AnalyticGerm n) I
  let f := Algebra.lmul (AnalyticGerm n) (AnalyticGerm (n + 1) ⧸ I) (q g)
  refine ⟨f.charpoly, f.charpoly_monic, ?_, ?_⟩
  · rw [LinearMap.charpoly_natDegree,
      Module.finrank_eq_card_basis (preparedQuotientBasis a ha ha0), Fintype.card_fin]
  · apply Ideal.Quotient.eq_zero_iff_mem.mp
    change q (Polynomial.aeval g f.charpoly) = 0
    rw [← Polynomial.aeval_algHom_apply]
    exact Algebra.aeval_self_charpoly_lmul (q g)

/-- Every real analytic germ obeys a monic polynomial relation modulo a
prepared equation. The relation has coefficients in the actual ring of
convergent real analytic parameter germs. Auxiliary for finite analytic
elimination in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem prepared_analytic_germ_integral_relation {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (g : AnalyticGerm (n + 1)) :
    ∃ p : Polynomial (AnalyticGerm n), p.Monic ∧
      Polynomial.aeval g p ∈ preparedPolynomialIdeal a ha := by
  obtain ⟨p, hp, _, hmem⟩ := prepared_analytic_germ_relation_degree a ha ha0 g
  exact ⟨p, hp, hmem⟩

end Transformer.RealAnalyticGerms
