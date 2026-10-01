/-
# Adapted coordinate noises for the joint Brownian filtration

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The filtration contains the histories of all coordinates, so a noise
coefficient can depend on the entire evolving state.
-/

import Transformer.BatchSize.Section4_BrownianIndependence

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual vector driver's natural filtration, Section 4.3 (2)--(3). -/
def brownianFiltration (d : ℕ) : Filtration ℝ≥0 (inferInstance : MeasurableSpace (BrownianSample d)) :=
  Filtration.natural (vectorBrownian d) (fun t => (vectorBrownian_measurable d t).stronglyMeasurable)

/-- Every scalar coordinate is a Brownian motion relative to the full
vector filtration, Section 4.3 (2)--(3). In particular its future increments
are independent of every event determined by the joint past. -/
theorem coordinateBrownian_filtered {d : ℕ} (k : Fin d) :
    IsFilteredPreBrownian (coordinateBrownian k) (brownianFiltration d) (brownianNoiseLaw d) := by
  refine ⟨(coordinateBrownian_isBrownian k).toIsPreBrownianReal, ?_, ?_⟩
  · intro t
    have hv := Filtration.stronglyAdapted_natural
      (fun t => (vectorBrownian_measurable d t).stronglyMeasurable) t
    exact (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable hv
  · intro s t hst
    have hi := (IndepFun_iff_Indep _ _ _).mp
      (coordinateBrownian_increment_past_independent k s t hst)
    apply indep_of_indep_of_le_right hi
    change (⨆ u ≤ s, MeasurableSpace.comap (vectorBrownian d u) inferInstance) ≤ _
    refine iSup_le fun u => iSup_le fun hu => ?_
    let past : BrownianSample d → Fin d → Set.Iic s → ℝ :=
      fun ω j r => coordinateBrownian j r ω
    let pick : (Fin d → Set.Iic s → ℝ) → EucSpace d :=
      fun p => WithLp.toLp 2 (fun j => p j ⟨u, hu⟩)
    have hpick : Measurable pick := by
      apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.measurable.comp
      apply Measurable.of_eval
      intro j
      simpa only [Function.comp_def] using
        (measurable_pi_apply (⟨u, hu⟩ : Set.Iic s)).comp (measurable_pi_apply j)
    have heq : vectorBrownian d u = pick ∘ past := rfl
    rw [heq, ← MeasurableSpace.comap_comp]
    exact MeasurableSpace.comap_mono (Measurable.comap_le hpick)

/-- The actual Brownian coordinates are martingales relative to the
joint past, Section 4.3 (2)--(3). This follows from the constructed Gaussian
increments and their independence, not from a postulated SDE solution. -/
theorem coordinateBrownian_martingale {d : ℕ} (k : Fin d) :
    Martingale (coordinateBrownian k) (brownianFiltration d) (brownianNoiseLaw d) := by
  let := coordinateBrownian_filtered k
  exact IsPreBrownianReal.isMartingale (coordinateBrownian k)
    (brownianFiltration d) (brownianNoiseLaw d)

end Transformer.BatchSize
