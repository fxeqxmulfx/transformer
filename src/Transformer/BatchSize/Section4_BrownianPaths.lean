/-
# The constructed Brownian probability measure on continuous vector paths

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
Real-time paths are extended constantly to negative time. The resulting
measure lives on exactly the path space used by the diffusion law.
-/

import Transformer.BatchSize.Section4_BrownianFiltration
import Transformer.BatchSize.Section4_DiffusionPaths

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The constructed real-time Brownian path, Section 4.3 (2)--(3).
Nonnegative time is Brownian; negative time uses the value at zero. -/
def brownianPath (d : ℕ) (ω : BrownianSample d) : DiffusionPath d :=
  ⟨fun t => vectorBrownian d t.toNNReal ω,
    (vectorBrownian_continuous d ω).comp continuous_real_toNNReal⟩

/-- Measurability into the continuous-path space, Section 4.3 (2)--(3),
proved from the constructed coordinate evaluations. -/
theorem brownianPath_measurable (d : ℕ) : Measurable (brownianPath d) := by
  apply ContinuousMap.measurable_iff_eval.mpr
  intro t
  exact vectorBrownian_measurable d t.toNNReal

/-- The Brownian driver as an actual measure on the continuous path
space of Section 4.3 (2)--(3). -/
def brownianPathLaw (d : ℕ) : Measure (DiffusionPath d) :=
  (brownianNoiseLaw d).map (brownianPath d)

/-- Normalization of the continuous vector path law, Section 4.3 (2)--(3). -/
instance brownianPathLaw_probability (d : ℕ) : IsProbabilityMeasure (brownianPathLaw d) :=
  (Measure.isProbabilityMeasure_map_iff (brownianPath_measurable d).aemeasurable).mpr
    (by infer_instance)

/-- The constructed path law starts at zero almost surely,
Section 4.3 (2)--(3). -/
theorem brownianPathLaw_zero (d : ℕ) : ∀ᵐ ω ∂brownianPathLaw d, ω 0 = 0 := by
  have hevent : MeasurableSet {ω : DiffusionPath d | ω 0 = 0} :=
    measurableSet_eq_fun (ContinuousMap.measurable_eval (0 : ℝ)) measurable_const
  rw [brownianPathLaw, ae_map_iff (brownianPath_measurable d).aemeasurable
    hevent]
  simpa only [brownianPath, ContinuousMap.coe_mk, Real.toNNReal_zero] using vectorBrownian_zero d

/-- The actual coordinate marginal of the continuous path measure is
the centered Gaussian with variance max(t,0), Section 4.3 (2)--(3). -/
theorem brownianPathLaw_coordinate_hasLaw {d : ℕ} (k : Fin d) (t : ℝ) :
    HasLaw (fun ω : DiffusionPath d => ω t k) (gaussianReal 0 t.toNNReal) (brownianPathLaw d) := by
  have heval : Measurable (fun ω : DiffusionPath d => ω t k) := by
    exact (PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).measurable.comp
      (ContinuousMap.measurable_eval t)
  refine ⟨heval.aemeasurable, ?_⟩
  rw [brownianPathLaw, Measure.map_map heval (brownianPath_measurable d)]
  exact (coordinateBrownian_hasLaw k t.toNNReal).map_eq

end Transformer.BatchSize
