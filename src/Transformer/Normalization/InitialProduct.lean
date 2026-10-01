/-
# Integrals over independent initialization coordinates

Finite-product integration used in Appendix B, proof of Theorem 4.2 of
arXiv:2510.22026v2. The marginal space is compact, as is the unit sphere.
-/

import Transformer.Normalization.InitialSphereMoments
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Bochner.Set

open MeasureTheory Set

namespace Transformer.Normalization

variable {X E : Type*} [TopologicalSpace X] [CompactSpace X]
  [MeasurableSpace X] [BorelSpace X] [SecondCountableTopology X]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Splitting the first independent coordinate turns a finite product
integral into an iterated integral. Source: arXiv:2510.22026v2, Appendix B,
conditioning on a token in the proof of Theorem 4.2. -/
theorem integral_pi_cons (μ : Measure X) [IsProbabilityMeasure μ] (n : ℕ)
    (f : (Fin (n + 1) → X) → E) (hf : Continuous f) :
    (∫ x, f x ∂Measure.pi (fun _ : Fin (n + 1) => μ)) =
      ∫ a, ∫ x, f (Fin.cons a x) ∂Measure.pi (fun _ : Fin n => μ) ∂μ := by
  let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => X) 0
  have hp := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => μ) 0
  change MeasurePreserving e (Measure.pi (fun _ : Fin (n + 1) => μ))
    (μ.prod (Measure.pi (fun _ : Fin n => μ))) at hp
  have he : ∀ p : X × (Fin n → X), e.symm p = Fin.cons p.1 p.2 := by
    intro p
    ext i
    cases i using Fin.cases <;>
      simp [e, MeasurableEquiv.piFinSuccAbove_symm_apply]
  have hc : Continuous (fun p : X × (Fin n → X) => f (Fin.cons p.1 p.2)) := by
    apply hf.comp
    apply continuous_pi
    intro i
    cases i using Fin.cases <;> fun_prop
  have h := hp.integral_comp e.measurableEmbedding (fun p => f (e.symm p))
  change (∫ x, f (e.symm (e x)) ∂Measure.pi (fun _ : Fin (n + 1) => μ)) =
    ∫ p, f (e.symm p) ∂μ.prod (Measure.pi (fun _ : Fin n => μ)) at h
  simp only [MeasurableEquiv.symm_apply_apply] at h
  simp only [he] at h
  rw [h, integral_prod _ (Perspective.integrable_of_continuous_compact hc _)]

/-- A constant integrand on a one-point product is continuous. -/
example : Continuous (fun _ : Fin 2 → Unit => (1 : ℝ)) := continuous_const

/-- Averaging the remaining coordinates preserves continuity of the first
coordinate. Source: arXiv:2510.22026v2, Appendix B, conditional means. -/
theorem continuous_integral_pi_cons [FirstCountableTopology X] [LocallyCompactSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ] (n : ℕ)
    (f : (Fin (n + 1) → X) → ℝ) (hf : Continuous f) :
    Continuous (fun a => ∫ x, f (Fin.cons a x) ∂Measure.pi (fun _ : Fin n => μ)) := by
  have hc : Continuous (fun p : X × (Fin n → X) => f (Fin.cons p.1 p.2)) := by
    apply hf.comp
    apply continuous_pi
    intro i
    cases i using Fin.cases <;> fun_prop
  simpa only [setIntegral_univ] using
    (continuous_parametric_integral_of_continuous hc (s := Set.univ) isCompact_univ)

/-- Constant integrands meet the continuity hypothesis. -/
example : Continuous (fun _ : Fin 2 → Unit => (1 : ℝ)) := continuous_const

omit [TopologicalSpace X] [CompactSpace X] [BorelSpace X] [SecondCountableTopology X] in
/-- A uniform bound on every section bounds the probability of the whole
event. Source: arXiv:2510.22026v2, Appendix B, conditioning on `θ_j` in
the proof of Theorem 4.2. -/
theorem measure_pi_insertNth_le (μ : Measure X) [IsProbabilityMeasure μ]
    (n : ℕ) (j : Fin (n + 1)) (s : Set (Fin (n + 1) → X)) (hs : MeasurableSet s)
    (p : ENNReal) (hsection : ∀ a : X,
      Measure.pi (fun _ : Fin n => μ) {x | (Fin.insertNth j a x : Fin (n + 1) → X) ∈ s} ≤ p) :
    Measure.pi (fun _ : Fin (n + 1) => μ) s ≤ p := by
  let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => X) j
  have hp := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => μ) j
  change MeasurePreserving e (Measure.pi (fun _ : Fin (n + 1) => μ))
    (μ.prod (Measure.pi (fun _ : Fin n => μ))) at hp
  have hs' : MeasurableSet (e.symm ⁻¹' s) := hs.preimage e.symm.measurable
  have hsymm : ∀ (a : X) (x : Fin n → X),
      e.symm (a, x) = (Fin.insertNth j a x : Fin (n + 1) → X) := by
    intro a x
    rfl
  have he : e ⁻¹' (e.symm ⁻¹' s) = s := by ext x; simp
  have hm := hp.measure_preimage hs'.nullMeasurableSet
  rw [he] at hm
  rw [hm, Measure.prod_apply hs']
  calc
    _ ≤ (∫⁻ _ : X, p ∂μ) := by
      apply lintegral_mono
      intro a
      change Measure.pi (fun _ : Fin n => μ) {x | e.symm (a, x) ∈ s} ≤ p
      simpa only [hsymm] using hsection a
    _ = p := by simp

/-- The full event over a one-point product meets both hypotheses. -/
example : MeasurableSet (Set.univ : Set (Fin 2 → Unit)) ∧
    ∀ a : Unit, Measure.pi (fun _ : Fin 1 => Measure.dirac ())
      {x | (Fin.insertNth (0 : Fin 2) a x : Fin 2 → Unit) ∈ Set.univ} ≤ 1 := by
  simp

end Transformer.Normalization
