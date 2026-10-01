/-
# Expectations with arbitrary random parameters from the Brownian past

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The next full vector increment has its genuine product Gaussian law,
independent of any bundle of jointly adapted parameters.
-/

import Transformer.BatchSize.Section4_BrownianVectorIncrement

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Averaging the actual next Brownian increment conditional on any
adapted parameter bundle is exactly standard Gaussian averaging with
the elapsed-time rescaling, Section 4.3 (2)--(3). Both sides are actual
expectations; independence and integrable Fubini are established here. -/
theorem vectorBrownianIncrement_integral_adapted {d : ℕ} {E : Type*}
    [TopologicalSpace E] [TopologicalSpace.PseudoMetrizableSpace E] [MeasurableSpace E] [BorelSpace E]
    (s t : ℝ≥0) (hst : s ≤ t) (H : BrownianSample d → E)
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (g : E × (Fin d → ℝ) → ℝ) (hg : Measurable g) (C : ℝ)
    (hC : ∀ ω z, |g (H ω, z)| ≤ C) :
    (∫ ω, g (H ω, vectorBrownianIncrement d s t ω) ∂brownianNoiseLaw d) =
      ∫ ω, ∫ z, g (H ω, fun k => Real.sqrt ((t : ℝ) - s) * z k)
        ∂standardGaussianVectorLaw d ∂brownianNoiseLaw d := by
  let P := brownianNoiseLaw d
  let Q := Measure.pi (fun _ : Fin d => gaussianReal 0 (t - s))
  have hHm : Measurable H := (hH.mono ((brownianFiltration d).le s)).measurable
  let : IsProbabilityMeasure (P.map H) :=
    (Measure.isProbabilityMeasure_map_iff hHm.aemeasurable).mpr inferInstance
  have hΔ := vectorBrownianIncrement_hasLaw d s t hst
  have hi := vectorBrownianIncrement_independent_adapted s t hst hH
  have hpair : HasLaw (fun ω => (H ω, vectorBrownianIncrement d s t ω))
      ((P.map H).prod Q) P := by
    refine ⟨(hHm.prodMk (vectorBrownianIncrement_measurable d s t)).aemeasurable, ?_⟩
    rw [hi.map_prod_eq_prod_map_map hHm.aemeasurable
      (vectorBrownianIncrement_measurable d s t).aemeasurable, hΔ.map_eq]
  have hMP : MeasurePreserving H P (P.map H) := ⟨hHm, rfl⟩
  have hMPprod := hMP.prod (MeasurePreserving.id Q)
  have hgactual : Integrable (g ∘ Prod.map H id) (P.prod Q) :=
    (integrable_const C).mono'
      (hg.comp ((hHm.comp measurable_fst).prodMk measurable_snd)).stronglyMeasurable.aestronglyMeasurable
      (Eventually.of_forall fun p => by
        simpa only [Real.norm_eq_abs, Function.comp_def, Prod.map, id_eq] using hC p.1 p.2)
  have hgint : Integrable g ((P.map H).prod Q) :=
    (hMPprod.integrable_comp hg.stronglyMeasurable.aestronglyMeasurable).mp hgactual
  have hrescale := standardGaussianVector_rescale_hasLaw d (t - s)
  have hscale : ((t - s : ℝ≥0) : ℝ) = (t : ℝ) - s := NNReal.coe_sub hst
  have hmRes : Measurable (fun p : E × (Fin d → ℝ) =>
      g (p.1, fun k => Real.sqrt ((t : ℝ) - s) * p.2 k)) := by
    apply hg.comp
    exact measurable_fst.prodMk (Measurable.of_eval fun k =>
      measurable_const.mul ((measurable_pi_apply k).comp measurable_snd))
  have hmInner : StronglyMeasurable (fun h : E => ∫ z,
      g (h, fun k => Real.sqrt ((t : ℝ) - s) * z k) ∂standardGaussianVectorLaw d) :=
    hmRes.stronglyMeasurable.integral_prod_right'
  calc
    _ = ∫ p, g p ∂(P.map H).prod Q := hpair.integral_comp hg.stronglyMeasurable.aestronglyMeasurable
    _ = ∫ h, ∫ z, g (h, z) ∂Q ∂P.map H := integral_prod g hgint
    _ = ∫ h, ∫ z, g (h, fun k => Real.sqrt ((t : ℝ) - s) * z k)
          ∂standardGaussianVectorLaw d ∂P.map H := by
      apply integral_congr_ae
      exact Eventually.of_forall fun h => by
        have hs := hrescale.integral_comp (f := fun z => g (h, z))
          (hg.comp ((measurable_const (a := h)).prodMk measurable_id)).stronglyMeasurable.aestronglyMeasurable
        simpa only [Function.comp_def, hscale] using hs.symm
    _ = _ := integral_map hHm.aemeasurable hmInner.aestronglyMeasurable

/-- Joint nonvacuity of the adapted Gaussian averaging hypotheses,
Section 4.3: a genuine past state and a bounded nonconstant observable
depending on both that state and the next innovation. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (vectorBrownian 1 1) ∧
    Measurable (fun p : EucSpace 1 × (Fin 1 → ℝ) => Real.sin (p.1 0 + p.2 0)) ∧
    (∀ p : EucSpace 1 × (Fin 1 → ℝ), |Real.sin (p.1 0 + p.2 0)| ≤ 1) := by
  refine ⟨by norm_num, Filtration.stronglyAdapted_natural
    (fun t => (vectorBrownian_measurable 1 t).stronglyMeasurable) 1, ?_,
    fun p => Real.abs_sin_le_one _⟩
  exact Real.continuous_sin.measurable.comp
    (((PiLp.continuous_apply 2 (fun _ : Fin 1 => ℝ) 0).measurable.comp measurable_fst).add
      ((measurable_pi_apply 0).comp measurable_snd))

end Transformer.BatchSize
