module

public import Transformer.Metastability.Section4_SequentialApproximation
public import Mathlib.MeasureTheory.Measure.Prod
public import Mathlib.MeasureTheory.Integral.Lebesgue.Add

/-!
# Conditioning the original independent sample on its actual prefix

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The original finite product of sphere laws splits into the newest
point and its actual earlier sample prefix. Fubini integrates the
actual event sections, rather than assuming that a random frame is
independent of the points used to construct it.

If an actual next-step bad event has probability at most `p` over
each successful prefix, its total probability is bounded by the
actual prefix failure probability plus `p`. Failed prefixes use the
probability bound one. The indicator of prefix failure therefore
dominates the conditional contribution left over from the uniform
one-step bound. All event measurability and finite-product identities
are proved for the original sample law.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Metastability

/-- Appending the newest actual sphere point is jointly continuous.
Source: arXiv:2410.06833v1, §4, the original sample-prefix induction. -/
theorem continuous_sphere_snoc (d m : ℕ) :
    Continuous (fun p : SSphere d × (Fin m → SSphere d) =>
      (Fin.snoc p.2 p.1 : Fin (m + 1) → SSphere d)) := by
  apply continuous_pi
  intro i
  cases i using Fin.lastCases with
  | last => simpa only [Fin.snoc_last] using continuous_fst
  | cast i => simpa only [Fin.snoc_castSucc, Function.comp_def] using (continuous_apply i).comp continuous_snd

/-- Appending the newest actual sphere point is jointly measurable.
Source: arXiv:2410.06833v1, §4, conditioning on the actual original prefix. -/
theorem measurable_sphere_snoc (d m : ℕ) :
    Measurable (fun p : SSphere d × (Fin m → SSphere d) =>
      (Fin.snoc p.2 p.1 : Fin (m + 1) → SSphere d)) :=
  (continuous_sphere_snoc d m).measurable

/-- Splitting the original independent sample at its last coordinate
preserves the actual probability of every measurable event.
Source: arXiv:2410.06833v1, §4, the original independent-sample
conditioning; arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem iidSphere_snoc_event (d m : ℕ) (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (B : Set (Fin (m + 1) → SSphere d)) (hB : MeasurableSet B) :
    iidSphere d (m + 1) ν B =
      (ν.prod (iidSphere d m ν)) ((fun p : SSphere d × (Fin m → SSphere d) =>
        Fin.snoc p.2 p.1) ⁻¹' B) := by
  let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (m + 1) => SSphere d) (Fin.last m)
  have hp := measurePreserving_piFinSuccAbove (fun _ : Fin (m + 1) => ν) (Fin.last m)
  change MeasurePreserving e (iidSphere d (m + 1) ν) (ν.prod (iidSphere d m ν)) at hp
  have he (p : SSphere d × (Fin m → SSphere d)) : e.symm p = Fin.snoc p.2 p.1 := by
    simp [e, MeasurableEquiv.piFinSuccAbove_symm_apply]
    rfl
  have hs := hB.preimage e.symm.measurable
  have hpre : e ⁻¹' (e.symm ⁻¹' B) = B := by ext X; simp
  have h := hp.measure_preimage hs.nullMeasurableSet
  rw [hpre] at h
  have hefun : e.symm = (fun p : SSphere d × (Fin m → SSphere d) =>
      (Fin.snoc p.2 p.1 : Fin (m + 1) → SSphere d)) := funext he
  rw [hefun] at h
  exact h

/-- Fubini integrates the actual newest-point event sections against
the actual original prefix law. Source: arXiv:2410.06833v1, §4,
the original independent-sample conditioning step. -/
theorem iidSphere_event_eq_lintegral_snoc (d m : ℕ)
    (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (B : Set (Fin (m + 1) → SSphere d)) (hB : MeasurableSet B) :
    iidSphere d (m + 1) ν B =
      ∫⁻ X, ν {x : SSphere d | Fin.snoc X x ∈ B} ∂iidSphere d m ν := by
  let : IsProbabilityMeasure (iidSphere d m ν) := by unfold iidSphere; infer_instance
  rw [iidSphere_snoc_event d m ν B hB,
    Measure.prod_apply_symm (hB.preimage (measurable_sphere_snoc d m))]
  rfl

/-- The original finite product law is a probability law.
Source: arXiv:2410.06833v1, §4, the original i.i.d. uniform initialization. -/
theorem isProbabilityMeasure_iidSphere (d m : ℕ) (ν : Measure (SSphere d)) [IsProbabilityMeasure ν] :
    IsProbabilityMeasure (iidSphere d m ν) := by
  unfold iidSphere
  infer_instance

/-- A bound on each actual successful prefix bounds the original
next-step event by prefix failure plus the one-step bound.
Source: arXiv:2410.06833v1, §4, the original independent-sample union
bound. The section bound is an explicit input to this general Fubini lemma. -/
theorem iidSphere_event_le_prefix_compl_add (d m : ℕ) (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (A : Set (Fin m → SSphere d)) (hA : MeasurableSet A)
    (B : Set (Fin (m + 1) → SSphere d)) (hB : MeasurableSet B)
    (p : ENNReal) (hsection : ∀ X ∈ A, ν {x : SSphere d | Fin.snoc X x ∈ B} ≤ p) :
    iidSphere d (m + 1) ν B ≤ iidSphere d m ν Aᶜ + p := by
  let : IsProbabilityMeasure (iidSphere d m ν) := isProbabilityMeasure_iidSphere d m ν
  rw [iidSphere_event_eq_lintegral_snoc d m ν B hB]
  calc
    _ ≤ ∫⁻ X, Aᶜ.indicator (fun _ => (1 : ENNReal)) X + p ∂iidSphere d m ν := by
      apply lintegral_mono
      intro X
      change ν {x : SSphere d | Fin.snoc X x ∈ B} ≤ Aᶜ.indicator (fun _ => (1 : ENNReal)) X + p
      by_cases hX : X ∈ A
      · have h := hsection X hX
        simpa only [indicator_of_notMem (show X ∉ Aᶜ from fun h => h hX), zero_add] using h
      · have h := prob_le_one (μ := ν) (s := {x : SSphere d | Fin.snoc X x ∈ B})
        rw [indicator_of_mem (show X ∈ Aᶜ from hX)]
        exact h.trans (le_add_right le_rfl)
    _ = _ := by
      rw [lintegral_add_left (measurable_const.indicator hA.compl)]
      rw [lintegral_indicator hA.compl]
      simp

/-- The same original prefix failure bound holds for real-valued
probabilities. Source: arXiv:2410.06833v1, §4, the numerical probability
bound in `prop: concentration unif`. Finite probability measures justify
both conversions from the actual nonnegative integral estimate. -/
theorem iidSphere_real_event_le_prefix_compl_add (d m : ℕ)
    (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (A : Set (Fin m → SSphere d)) (hA : MeasurableSet A)
    (B : Set (Fin (m + 1) → SSphere d)) (hB : MeasurableSet B)
    (p : ℝ) (hp : 0 ≤ p)
    (hsection : ∀ X ∈ A, ν.real {x : SSphere d | Fin.snoc X x ∈ B} ≤ p) :
    (iidSphere d (m + 1) ν).real B ≤ (iidSphere d m ν).real Aᶜ + p := by
  let : IsProbabilityMeasure (iidSphere d m ν) := isProbabilityMeasure_iidSphere d m ν
  have h := iidSphere_event_le_prefix_compl_add d m ν A hA B hB (ENNReal.ofReal p) (by
    intro X hX
    have hsec := ENNReal.ofReal_le_ofReal (hsection X hX)
    simpa only [ofReal_measureReal (measure_ne_top ν _)] using hsec)
  have hfin : iidSphere d m ν Aᶜ + ENNReal.ofReal p ≠ ⊤ :=
    ENNReal.add_ne_top.mpr ⟨measure_ne_top _ _, ENNReal.ofReal_ne_top⟩
  have hreal := ENNReal.toReal_mono hfin h
  rw [ENNReal.toReal_add (measure_ne_top _ _) ENNReal.ofReal_ne_top, ENNReal.toReal_ofReal hp] at hreal
  exact hreal

/-- A concrete prefix event and a concrete next-step event, the
original uniform circle law, and its numerical section bound jointly
witness the hypotheses of both original conditioning lemmas. -/
example : IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    MeasurableSet (univ : Set (Fin 1 → SSphere 2)) ∧
    MeasurableSet (univ : Set (Fin 2 → SSphere 2)) ∧ (0 : ℝ) ≤ 1 ∧
    ∀ X ∈ (univ : Set (Fin 1 → SSphere 2)),
      (MeanField.uniformLaw 2).real {x : SSphere 2 | Fin.snoc X x ∈ (univ : Set (Fin 2 → SSphere 2))} ≤ 1 := by
  have : IsProbabilityMeasure (MeanField.uniformLaw 2) := (isUniformOn_uniformLaw (by norm_num)).1
  exact ⟨isUniformOn_uniformLaw (by norm_num), MeasurableSet.univ, MeasurableSet.univ,
    zero_le_one, fun X hX => by simp⟩

end Transformer.Metastability
