/-
# Real analytic preparation: AnalyticDivision

Adapted from LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/Division.lean.
The scalar field is real; only the required division infrastructure is copied.
Apache-2.0 licenses: third_party/classical-complex-wpt/LICENSE and
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.AnalyticDivisionFunctions

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/--
Fixed-radius analytic Weierstrass division by a prepared polynomial.  The
radius and ambient power-series witness remain explicit so downstream germ
bridges can reuse the constructed quotient and remainder functions.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticWeierstrassDivision_fixedRadius {n d : ℕ}
    {h : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ)
    (hp : HasFPowerSeriesAt h p 0)
    (r : ℝ≥0) (hr0 : 0 < r) (hrp : (r : ℝ≥0∞) < p.radius)
    (a : Fin d → Base n → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    AnalyticAt ℝ (analyticDivisionQuotient p r a) 0 ∧
      (∀ i, AnalyticAt ℝ (analyticDivisionRemainderCoefficient p r a i) 0) ∧
      h =ᶠ[𝓝 0] fun x ↦
        analyticDivisionQuotient p r a x * preparedPolynomial d a x +
          ∑ i : Fin d,
            analyticDivisionRemainderCoefficient p r a i x.1 * x.2 ^ (i : ℕ) := by
  refine ⟨analyticAt_analyticDivisionQuotient p r hrp a ha ha0,
    fun i ↦ analyticAt_analyticDivisionRemainderCoefficient p r hrp a ha ha0 i, ?_⟩
  have hsmallBase : ∀ᶠ z in 𝓝 (0 : Base n), ‖preparedTailSeq r d a z‖ < 1 :=
    eventually_norm_preparedTailSeq_lt_one r a ha ha0
  have hsmall : ∀ᶠ x in 𝓝 (0 : Ambient n),
      ‖preparedTailSeq r d a x.1‖ < 1 := by
    exact (continuousAt_fst : ContinuousAt (fun x : Ambient n ↦ x.1) 0).eventually
      hsmallBase
  have hw : ∀ᶠ x in 𝓝 (0 : Ambient n), ‖(r : ℝ)⁻¹ * x.2‖ < 1 := by
    have hc : ContinuousAt (fun x : Ambient n ↦ (r : ℝ)⁻¹ * x.2) 0 := by
      fun_prop
    have hb := Metric.ball_mem_nhds (0 : ℝ) (by norm_num : (0 : ℝ) < 1)
    have hb' : Metric.ball (0 : ℝ) 1 ∈
        𝓝 ((fun x : Ambient n ↦ (r : ℝ)⁻¹ * x.2) 0) := by
      simpa [ambient_zero_eq] using hb
    have he := hc.eventually hb'
    simpa only [Metric.mem_ball, dist_zero_right] using he
  have hreconstruct :=
    eventually_eval_weightedCoefficientSeries_eq_of_hasFPowerSeriesAt
      p r hp hr0 hrp
  filter_upwards [hsmall, hw, hreconstruct] with x hxsmall hxw hxreconstruct
  simpa using analyticDivision_pointwise p r hr0 a x.1 x.2 hxsmall hxw hxreconstruct

/--
Arbitrary-dividend analytic Weierstrass division.  This is the function-level
adapter missing from the pinned WPT public surface: it returns an analytic
quotient and analytic coefficients of a remainder of distinguished degree
strictly below `d`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_analyticWeierstrassDivision {n d : ℕ}
    (h : Ambient n → ℝ) (hh : AnalyticAt ℝ h 0)
    (a : Fin d → Base n → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    ∃ (q : Ambient n → ℝ) (remainder : Fin d → Base n → ℝ),
      AnalyticAt ℝ q 0 ∧
      (∀ i, AnalyticAt ℝ (remainder i) 0) ∧
      h =ᶠ[𝓝 0] fun x ↦
        q x * preparedPolynomial d a x +
          ∑ i : Fin d, remainder i x.1 * x.2 ^ (i : ℕ) := by
  obtain ⟨p, hp⟩ := hh
  rcases ENNReal.lt_iff_exists_nnreal_btwn.1 hp.radius_pos with ⟨r, hr0E, hrp⟩
  have hr0 : 0 < r := by exact_mod_cast hr0E
  let q : Ambient n → ℝ := analyticDivisionQuotient p r a
  let remainder : Fin d → Base n → ℝ :=
    fun i ↦ analyticDivisionRemainderCoefficient p r a i
  have hdivision := analyticWeierstrassDivision_fixedRadius p hp r hr0 hrp a ha ha0
  exact ⟨q, remainder, hdivision.1, hdivision.2.1, hdivision.2.2⟩

end Transformer.AnalyticPreparation
