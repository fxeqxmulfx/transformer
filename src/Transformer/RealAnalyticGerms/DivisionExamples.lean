/-
# Joint witnesses for germ division and its finite power basis

The divisor `y² - z₀` has a two-element power basis over the actual base
germ ring. Dividing `y³` computes the nonzero remainder `z₀ y`. Auxiliary
examples for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.RealAnalyticGerms.PreparedIntegralRelation

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open AnalyticPreparation

/-- Nonconstant coefficients, a cubic dividend, and a nonzero degree-one
remainder jointly witness germ division, intrinsic quotient and remainder,
the exact kernel, the finite power basis, and integrality. Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ (a : Fin 2 → Base 1 → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (r : Fin 2 → AnalyticGerm 1),
    r 0 = 0 ∧ r 1 ≠ 0 ∧
      IsPreparedGermDivision a ha (lastCoordinateGerm 1 ^ 3) (lastCoordinateGerm 1) r ∧
      preparedGermDivisionQuotient a ha ha0 (lastCoordinateGerm 1 ^ 3) = lastCoordinateGerm 1 ∧
      preparedGermDivisionRemainder a ha ha0 (lastCoordinateGerm 1 ^ 3) = r ∧
      LinearMap.ker (preparedGermDivisionRemainderLinearMap a ha ha0) =
        (preparedPolynomialIdeal a ha).restrictScalars (AnalyticGerm 1) ∧
      Module.Free (AnalyticGerm 1) (AnalyticGerm 2 ⧸ preparedPolynomialIdeal a ha) ∧
      Module.Finite (AnalyticGerm 1) (AnalyticGerm 2 ⧸ preparedPolynomialIdeal a ha) ∧
      preparedQuotientRemainderEquiv a ha ha0
        (Ideal.Quotient.mk (preparedPolynomialIdeal a ha) (lastCoordinateGerm 1 ^ 3)) = r ∧
      preparedQuotientBasis a ha ha0 (1 : Fin 2) =
        Ideal.Quotient.mk (preparedPolynomialIdeal a ha) (lastCoordinateGerm 1) ∧
      ∃ p : Polynomial (AnalyticGerm 1), p.Monic ∧ p.natDegree = 2 ∧
        Polynomial.aeval (lastCoordinateGerm 1 ^ 3) p ∈ preparedPolynomialIdeal a ha := by
  let a : Fin 2 → Base 1 → ℝ := fun i z => if i = 0 then -z 0 else 0
  let b : Fin 2 → Base 1 → ℝ := fun i z => if i = 0 then 0 else z 0
  have hz : AnalyticAt ℝ (fun z : Base 1 => z 0) 0 :=
    (ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0
  have ha : ∀ i, AnalyticAt ℝ (a i) 0 := by
    intro i
    dsimp only [a]
    split_ifs
    · exact hz.neg
    · exact analyticAt_const
  have ha0 : ∀ i, a i 0 = 0 := by intro i; simp only [a]; split_ifs <;> simp
  have hb : ∀ i, AnalyticAt ℝ (b i) 0 := by
    intro i
    dsimp only [b]
    split_ifs
    · exact analyticAt_const
    · exact hz
  let r : Fin 2 → AnalyticGerm 1 := fun i => AnalyticGerm.ofFunction (b i) (hb i)
  have hr0 : r 0 = 0 := by apply Subtype.ext; rfl
  have hr1 : r 1 ≠ 0 := by
    intro hzero
    have heq : (fun z : Base 1 => z 0) =ᶠ[𝓝 0] (fun _ => 0) := by
      apply Filter.Germ.coe_eq.mp
      exact congrArg (fun g : AnalyticGerm 1 => (g : FunctionGerm 1)) hzero
    let line : ℝ →L[ℝ] Base 1 := ContinuousLinearMap.toSpanSingleton ℝ (fun _ => 1)
    have ht : Tendsto line (𝓝 0) (𝓝 0) := by
      simpa only [map_zero] using line.continuous.continuousAt.tendsto (x := 0)
    have hid : (fun t : ℝ => t) =ᶠ[𝓝 0] (fun _ => 0) := by
      simpa [line, Function.comp_def] using heq.comp_tendsto ht
    obtain ⟨delta, hd, hball⟩ := Metric.eventually_nhds_iff.mp hid
    have hzero := hball (y := delta / 2) (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hd)] using half_lt_self hd)
    exact (half_pos hd).ne' hzero
  have hdivision : IsPreparedGermDivision a ha
      (lastCoordinateGerm 1 ^ 3) (lastCoordinateGerm 1) r := by
    unfold IsPreparedGermDivision
    apply Subtype.ext
    refine Eq.trans ?_ (coe_preparedDivisionExpression_ofFunction a ha (lastCoordinateCLM 1)
      ((lastCoordinateCLM 1).analyticAt 0) b hb).symm
    apply Filter.Germ.coe_eq.mpr
    apply Eventually.of_forall
    intro x
    change (lastCoordinateCLM 1 x) ^ 3 =
      lastCoordinateCLM 1 x * preparedPolynomialFunction a x +
        ∑ i : Fin 2, b i (baseProjectionCLM 1 x) * lastCoordinateCLM 1 x ^ (i : ℕ)
    simp [preparedPolynomialFunction, preparedPolynomial, wptAmbientEquiv_apply,
      baseProjectionCLM_apply, lastCoordinateCLM_apply, a, b, Fin.sum_univ_succ]
    ring
  have hcanonical := preparedGermDivision_eq_of_isDivision a ha ha0
    (lastCoordinateGerm 1 ^ 3) (lastCoordinateGerm 1) r hdivision
  refine ⟨a, ha, ha0, r, hr0, hr1, hdivision, hcanonical.1, hcanonical.2,
    preparedGermDivisionRemainderLinearMap_ker a ha ha0,
    preparedQuotient_moduleFree a ha ha0, preparedQuotient_moduleFinite a ha ha0, ?_, ?_,
    prepared_analytic_germ_relation_degree a ha ha0 (lastCoordinateGerm 1 ^ 3)⟩
  · exact hcanonical.2
  · rw [preparedQuotientBasis_apply]
    norm_num

end Transformer.RealAnalyticGerms
