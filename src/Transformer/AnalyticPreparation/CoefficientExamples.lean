/-
# Joint witnesses for analytic preparation hypotheses

The energy `w² + z₀²` supplies a nonconstant, singular analytic example.
The same convergent series and positive radius satisfy the exact-order,
majorant, normalization, analytic quotient, and analytic remainder inputs.
Auxiliary examples for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.AnalyticPreparation.SequenceExamples

open Filter
open scoped ENNReal NNReal Topology

namespace Transformer.AnalyticPreparation

/-- The quadratic energy simultaneously supplies an actual analytic
power series, exact order two, and all origin normalization and division
hypotheses. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ p : FormalMultilinearSeries ℝ (Ambient 1) ℝ,
    HasFPowerSeriesAt (fun x : Ambient 1 => x.2 ^ 2 + (x.1 0) ^ 2) p 0 ∧
    ExactOrderInLastVariable (fun x : Ambient 1 => x.2 ^ 2 + (x.1 0) ^ 2) 2 ∧
    ∃ (r : ℝ≥0) (hr : (r : ℝ≥0∞) < p.radius), 0 < r ∧
      originWeightedCoeffs p r hr 2 ≠ 0 ∧
      (∀ k < 2, originWeightedCoeffs p r hr k = 0) ∧
      ‖analyticNormalizedCoefficientMap p r hr 2 0 - monomialSeq 2‖ < (1 : ℝ) / 2 ∧
      AnalyticAt ℝ (analyticNormalizedCoefficientMap p r hr 2) 0 ∧
      (∀ᶠ z in nhds (0 : Base 1),
        ‖analyticNormalizedCoefficientMap p r hr 2 z - monomialSeq 2‖ < 1) ∧
      ‖normalizedPreparationTail p r hr 2 0‖ < 1 ∧
      AnalyticAt ℝ (normalizedPreparationQuotient p r hr 2) 0 ∧
      AnalyticAt ℝ (normalizedPreparationRemainder p r hr 2) 0 ∧
      normalizedPreparationRemainder p r hr 2 0 = 0 ∧
      normalizedPreparationQuotient p r hr 2 0 0 = 1 := by
  let f : Ambient 1 → ℝ := fun x => x.2 ^ 2 + (x.1 0) ^ 2
  have hz : AnalyticAt ℝ (fun x : Ambient 1 => x.1 0) 0 := by
    let L : Ambient 1 →L[ℝ] ℝ :=
      (ContinuousLinearMap.proj (0 : Fin 1)).comp (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
    exact L.analyticAt 0
  have hf : AnalyticAt ℝ f 0 := (analyticAt_snd.fun_pow 2).add (hz.fun_pow 2)
  have hslice : lastSlice f = fun t : ℝ => t ^ 2 := by funext t; simp [lastSlice, f]
  have hord : analyticOrderAt (fun t : ℝ => t ^ 2) 0 = 2 := by
    simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := 0)) 2
  have horder : ExactOrderInLastVariable f 2 := by
    rw [ExactOrderInLastVariable, hslice]
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero
      (analyticAt_id.fun_pow 2)).mp hord
  obtain ⟨p, hp⟩ := hf
  obtain ⟨r, hr, hr0, htop, han, hclose, hnear⟩ :=
    exists_normalizedAnalyticCoefficientMap p hp horder
  have hsmall : ‖normalizedPreparationTail p r hr 2 0‖ < 1 :=
    hclose.trans (by norm_num)
  exact ⟨p, hp, horder, r, hr, hr0, htop,
    fun k hk => originWeightedCoeffs_low_eq_zero p hp horder r hr hk,
    hclose, han, hnear, hsmall, analyticAt_normalizedPreparationQuotient p r hr 2 hsmall,
    analyticAt_normalizedPreparationRemainder p r hr 2 hsmall,
    normalizedPreparationRemainder_zero p hp horder r hr0 hr hsmall,
    normalizedPreparationQuotient_zero_apply_zero p hp horder r hr0 hr hsmall⟩

/-- A linear series has infinite radius, so positive weights and both
simultaneous reconstruction ball conditions are satisfiable. The
majorants and moving-coefficient evaluation are exercised at the origin.
Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ (p : FormalMultilinearSeries ℝ (Ambient 1) ℝ) (r : ℝ≥0),
    0 < r ∧ (r : ℝ≥0∞) < p.radius ∧
    HasFPowerSeriesAt (fun x : Ambient 1 => x.2) p 0 ∧
    Summable (fun k : ℕ => ‖p.changeOriginSeries k 0‖₊ * r ^ k) ∧
    (0 : Base 1) ∈ Metric.eball 0 (weightedCoefficientSeries p r).radius ∧
    ((0 : Base 1), (0 : ℝ)) ∈ Metric.eball (0 : Ambient 1) p.radius ∧
    evalL1PowerSeries ((weightedCoefficientSeries p r).sum 0)
      ((r : ℝ)⁻¹ * 0) = 0 := by
  let L : Ambient 1 →L[ℝ] ℝ := ContinuousLinearMap.snd ℝ (Base 1) ℝ
  let p := L.fpowerSeries 0
  have hp : HasFPowerSeriesAt (fun x : Ambient 1 => x.2) p 0 := L.hasFPowerSeriesAt 0
  obtain ⟨r, hr0E, hr⟩ := ENNReal.lt_iff_exists_nnreal_btwn.mp hp.radius_pos
  have hr0 : 0 < r := by exact_mod_cast hr0E
  have hq : (0 : Base 1) ∈ Metric.eball 0 (weightedCoefficientSeries p r).radius := by
    simpa only [Metric.mem_eball, edist_self] using radius_weightedCoefficientSeries_pos p r hr
  have hball : ((0 : Base 1), (0 : ℝ)) ∈ Metric.eball (0 : Ambient 1) p.radius := by
    simpa only [← ambient_zero_eq, Metric.mem_eball, edist_self] using hp.radius_pos
  have hrec := eventually_eval_weightedCoefficientSeries_eq_of_hasFPowerSeriesAt p r hp hr0 hr
  refine ⟨p, r, hr0, hr, hp, summable_nnnorm_changeOriginSeries_mul_pow p r hr 0,
    hq, hball, ?_⟩
  simpa only [ambient_zero_eq] using hrec.self_of_nhds

/-- Degree zero is a genuine nonvanishing unit case, and is also
satisfiable in the public preparation theorem. Appendix D.1 of
arXiv:2510.22026v2. -/
example : ∃ (a : Fin 0 → Base 1 → ℝ) (u : Ambient 1 → ℝ),
    IsWeierstrassPreparation (fun _ : Ambient 1 => (1 : ℝ)) 0 a u := by
  apply exists_isWeierstrassPreparation analyticAt_const
  constructor
  · intro k hk
    omega
  · simp [lastSlice]

end Transformer.AnalyticPreparation
