/-
# Real analytic germs: DivisionUniqueness

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/GermDivision.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.DivisionExistence

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Germ-level uniqueness.  In particular, the quotient and coefficient
germs do not depend on any analytic representatives used to construct them.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedGermDivision_unique {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0)
    (h q q' : AnalyticGerm (n + 1))
    (r r' : Fin d → AnalyticGerm n)
    (hdivision : IsPreparedGermDivision a ha h q r)
    (hdivision' : IsPreparedGermDivision a ha h q' r') :
    q = q' ∧ r = r' := by
  obtain ⟨qf, hqf, hqrep⟩ := AnalyticGerm.exists_rep q
  obtain ⟨qf', hqf', hqrep'⟩ := AnalyticGerm.exists_rep q'
  choose rf hrf hrfrep using fun i ↦ AnalyticGerm.exists_rep (r i)
  choose rf' hrf' hrfrep' using fun i ↦ AnalyticGerm.exists_rep (r' i)
  have hqGerm : AnalyticGerm.ofFunction qf hqf = q := by
    apply Subtype.ext
    exact hqrep
  have hqGerm' : AnalyticGerm.ofFunction qf' hqf' = q' := by
    apply Subtype.ext
    exact hqrep'
  have hrGerm : (fun i ↦ AnalyticGerm.ofFunction (rf i) (hrf i)) = r := by
    funext i
    apply Subtype.ext
    exact hrfrep i
  have hrGerm' : (fun i ↦ AnalyticGerm.ofFunction (rf' i) (hrf' i)) = r' := by
    funext i
    apply Subtype.ext
    exact hrfrep' i
  have hdecomposition :
      q * preparedPolynomialGerm a ha + remainderPolynomialGerm r =
        q' * preparedPolynomialGerm a ha + remainderPolynomialGerm r' :=
    hdivision.symm.trans hdivision'
  rw [← hqGerm, ← hqGerm', ← hrGerm, ← hrGerm'] at hdecomposition
  have hdecompositionCoe := congrArg
    (fun g : AnalyticGerm (n + 1) ↦ (g : FunctionGerm (n + 1)))
    hdecomposition
  rw [coe_preparedDivisionExpression_ofFunction a ha qf hqf rf hrf,
    coe_preparedDivisionExpression_ofFunction a ha qf' hqf' rf' hrf']
      at hdecompositionCoe
  have hstandard := Filter.Germ.coe_eq.mp hdecompositionCoe
  let qWPT : Ambient n → ℝ := fun x ↦ qf ((wptAmbientEquiv n).symm x)
  let qWPT' : Ambient n → ℝ := fun x ↦ qf' ((wptAmbientEquiv n).symm x)
  have hqWPT : AnalyticAt ℝ qWPT 0 := by
    change AnalyticAt ℝ (qf ∘ (wptAmbientEquiv n).symm) 0
    have hqf0 : AnalyticAt ℝ qf ((wptAmbientEquiv n).symm 0) := by
      rw [map_zero]
      exact hqf
    simpa using hqf0.compContinuousLinearMap
      (u := ((wptAmbientEquiv n).symm :
        Ambient n →L[ℝ] Base (n + 1))) (x := 0)
  have hqWPT' : AnalyticAt ℝ qWPT' 0 := by
    change AnalyticAt ℝ (qf' ∘ (wptAmbientEquiv n).symm) 0
    have hqf0 : AnalyticAt ℝ qf' ((wptAmbientEquiv n).symm 0) := by
      rw [map_zero]
      exact hqf'
    simpa using hqf0.compContinuousLinearMap
      (u := ((wptAmbientEquiv n).symm :
        Ambient n →L[ℝ] Base (n + 1))) (x := 0)
  have hequivSymm : Tendsto (wptAmbientEquiv n).symm (𝓝 0) (𝓝 0) := by
    have hc : Tendsto (wptAmbientEquiv n).symm (𝓝 0)
        (𝓝 ((wptAmbientEquiv n).symm 0)) :=
      (wptAmbientEquiv n).symm.continuous.continuousAt
    rw [map_zero] at hc
    exact hc
  have hWPT : (fun x : Ambient n ↦
      qWPT x * preparedPolynomial d a x +
        ∑ i : Fin d, rf i x.1 * x.2 ^ (i : ℕ)) =ᶠ[𝓝 0]
      fun x ↦ qWPT' x * preparedPolynomial d a x +
        ∑ i : Fin d, rf' i x.1 * x.2 ^ (i : ℕ) := by
    have hstandard' := hstandard.comp_tendsto hequivSymm
    filter_upwards [hstandard'] with x hx
    change
      qf ((wptAmbientEquiv n).symm x) *
          preparedPolynomialFunction a ((wptAmbientEquiv n).symm x) +
          ∑ i : Fin d,
            rf i (baseProjectionCLM n ((wptAmbientEquiv n).symm x)) *
              lastCoordinateCLM n ((wptAmbientEquiv n).symm x) ^ (i : ℕ) =
        qf' ((wptAmbientEquiv n).symm x) *
          preparedPolynomialFunction a ((wptAmbientEquiv n).symm x) +
          ∑ i : Fin d,
            rf' i (baseProjectionCLM n ((wptAmbientEquiv n).symm x)) *
              lastCoordinateCLM n ((wptAmbientEquiv n).symm x) ^ (i : ℕ) at hx
    simpa [qWPT, qWPT', preparedPolynomialFunction, baseProjectionCLM,
      lastCoordinateCLM] using hx
  let dividend : Ambient n → ℝ := fun x ↦
    qWPT x * preparedPolynomial d a x +
      ∑ i : Fin d, rf i x.1 * x.2 ^ (i : ℕ)
  have hunique := analyticWeierstrassDivision_unique
    dividend qWPT qWPT' rf rf' a hqWPT hqWPT' ha ha0
    (Filter.Eventually.of_forall fun _ ↦ rfl) hWPT
  have hequiv : Tendsto (wptAmbientEquiv n) (𝓝 0) (𝓝 0) := by
    have hc : Tendsto (wptAmbientEquiv n) (𝓝 0)
        (𝓝 (wptAmbientEquiv n 0)) :=
      (wptAmbientEquiv n).continuous.continuousAt
    rw [map_zero] at hc
    exact hc
  constructor
  · have hqStandard := hunique.1.comp_tendsto hequiv
    have hqFunctions : qf =ᶠ[𝓝 0] qf' := by
      filter_upwards [hqStandard] with x hx
      change qf ((wptAmbientEquiv n).symm (wptAmbientEquiv n x)) =
        qf' ((wptAmbientEquiv n).symm (wptAmbientEquiv n x)) at hx
      rw [(wptAmbientEquiv n).symm_apply_apply] at hx
      exact hx
    have hchosen :
        AnalyticGerm.ofFunction qf hqf =
          AnalyticGerm.ofFunction qf' hqf' :=
      Subtype.ext (Filter.Germ.coe_eq.mpr hqFunctions)
    exact hqGerm.symm.trans (hchosen.trans hqGerm')
  · apply funext
    intro i
    have hrFunctions := hunique.2 i
    have hchosen :
        AnalyticGerm.ofFunction (rf i) (hrf i) =
          AnalyticGerm.ofFunction (rf' i) (hrf' i) :=
      Subtype.ext (Filter.Germ.coe_eq.mpr hrFunctions)
    exact (congrFun hrGerm i).symm.trans (hchosen.trans (congrFun hrGerm' i))


end Transformer.RealAnalyticGerms
