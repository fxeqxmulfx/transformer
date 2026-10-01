/-
# Real analytic germs: DivisionExistence

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/GermDivision.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.PolynomialGerms

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Every analytic germ admits a quotient and degree-`< d` remainder by a
fixed prepared polynomial.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_preparedGermDivision {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (h : AnalyticGerm (n + 1)) :
    ∃ (q : AnalyticGerm (n + 1)) (r : Fin d → AnalyticGerm n),
      IsPreparedGermDivision a ha h q r := by
  obtain ⟨f, hf, hrep⟩ := AnalyticGerm.exists_rep h
  let fWPT : Ambient n → ℝ := fun x ↦ f ((wptAmbientEquiv n).symm x)
  have hfWPT : AnalyticAt ℝ fWPT 0 := by
    change AnalyticAt ℝ (f ∘ (wptAmbientEquiv n).symm) 0
    have hf' : AnalyticAt ℝ f ((wptAmbientEquiv n).symm 0) := by
      rw [map_zero]
      exact hf
    simpa using hf'.compContinuousLinearMap
      (u := ((wptAmbientEquiv n).symm :
        Ambient n →L[ℝ] Base (n + 1))) (x := 0)
  obtain ⟨q, r, hq, hr, hfactor⟩ :=
    exists_analyticWeierstrassDivision fWPT hfWPT a ha ha0
  let qStandard : Base (n + 1) → ℝ :=
    fun x ↦ q (wptAmbientEquiv n x)
  have hqStandard : AnalyticAt ℝ qStandard 0 := by
    change AnalyticAt ℝ (q ∘ (wptAmbientEquiv n)) 0
    simpa using hq.compContinuousLinearMap
      (u := (wptAmbientEquiv n :
        Base (n + 1) →L[ℝ] Ambient n)) (x := 0)
  let qGerm : AnalyticGerm (n + 1) :=
    AnalyticGerm.ofFunction qStandard hqStandard
  let rGerm : Fin d → AnalyticGerm n :=
    fun i ↦ AnalyticGerm.ofFunction (r i) (hr i)
  have hequiv : Tendsto (wptAmbientEquiv n) (𝓝 0) (𝓝 0) := by
    have hc : Tendsto (wptAmbientEquiv n) (𝓝 0)
        (𝓝 (wptAmbientEquiv n 0)) :=
      (wptAmbientEquiv n).continuous.continuousAt
    rw [map_zero] at hc
    exact hc
  have hfactorStandard : f =ᶠ[𝓝 0] fun x ↦
      qStandard x * preparedPolynomialFunction a x +
        ∑ i : Fin d, r i (wptAmbientEquiv n x).1 *
          (wptAmbientEquiv n x).2 ^ (i : ℕ) := by
    have hfactor' := hfactor.comp_tendsto hequiv
    filter_upwards [hfactor'] with x hx
    change f ((wptAmbientEquiv n).symm (wptAmbientEquiv n x)) =
      qStandard x * preparedPolynomialFunction a x +
        ∑ i : Fin d, r i (wptAmbientEquiv n x).1 *
          (wptAmbientEquiv n x).2 ^ (i : ℕ) at hx
    rw [(wptAmbientEquiv n).symm_apply_apply] at hx
    exact hx
  refine ⟨qGerm, rGerm, ?_⟩
  unfold IsPreparedGermDivision
  apply Subtype.ext
  rw [← hrep]
  simp only [remainderPolynomialGerm, Subring.coe_add, Subring.coe_mul,
    Subring.coe_pow, AddSubmonoidClass.coe_finsetSum]
  change (f : FunctionGerm (n + 1)) =
    (qGerm : FunctionGerm (n + 1)) *
      (preparedPolynomialGerm a ha : FunctionGerm (n + 1)) +
      ∑ i : Fin d,
        (lowerDimensionalInclusion n (rGerm i) : FunctionGerm (n + 1)) *
          (lastCoordinateGerm n : FunctionGerm (n + 1)) ^ (i : ℕ)
  rw [show (qGerm : FunctionGerm (n + 1)) = qStandard from rfl]
  rw [show (preparedPolynomialGerm a ha : FunctionGerm (n + 1)) =
    preparedPolynomialFunction a from rfl]
  simp only [analyticGermPullbackHom_coe, lowerDimensionalInclusion]
  simp_rw [show ∀ i, (rGerm i : FunctionGerm n) = r i from fun _ ↦ rfl]
  simp only [functionGermPullbackHom_coe]
  rw [show (lastCoordinateGerm n : FunctionGerm (n + 1)) =
    lastCoordinateCLM n from rfl]
  have hfactorStandard' : f =ᶠ[𝓝 0]
      qStandard * preparedPolynomialFunction a +
        ∑ i : Fin d,
          (r i ∘ baseProjectionCLM n) * (lastCoordinateCLM n) ^ (i : ℕ) := by
    filter_upwards [hfactorStandard] with x hx
    simpa using hx
  have hgerm := Filter.Germ.coe_eq.mpr hfactorStandard'
  have hsum :
      (((∑ i : Fin d,
          (r i ∘ baseProjectionCLM n) * (lastCoordinateCLM n) ^ (i : ℕ)) :
          Base (n + 1) → ℝ) : FunctionGerm (n + 1)) =
        ∑ i : Fin d,
          ((r i ∘ baseProjectionCLM n) : FunctionGerm (n + 1)) *
            ((lastCoordinateCLM n : Base (n + 1) → ℝ) :
              FunctionGerm (n + 1)) ^ (i : ℕ) := by
    calc
      _ = ∑ i : Fin d,
          (((r i ∘ baseProjectionCLM n) *
            (lastCoordinateCLM n) ^ (i : ℕ) :
              Base (n + 1) → ℝ) : FunctionGerm (n + 1)) := by
          exact map_sum (Filter.Germ.coeRingHom
            (𝓝 (0 : Base (n + 1))))
              (fun i : Fin d ↦
                (r i ∘ baseProjectionCLM n) *
                  (lastCoordinateCLM n) ^ (i : ℕ)) Finset.univ
      _ = _ := by
        simp only [Filter.Germ.coe_mul, Filter.Germ.coe_pow]
  simp only [Filter.Germ.coe_add, Filter.Germ.coe_mul] at hgerm
  rw [hsum] at hgerm
  convert hgerm using 1
  congr 1


end Transformer.RealAnalyticGerms
