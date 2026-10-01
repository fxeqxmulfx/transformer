/-
# Constructed independent Brownian noises

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The underlying measure and continuous processes are constructed, rather
than assuming the existence of a Brownian motion in a structure field.
The retained external construction is attributed in third_party/Stochastic.
-/

import BrownianMotion.Gaussian.BrownianMotion
import Transformer.BatchSize.Section4_DiffusionModel

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Independent coordinate sample spaces for the driving Brownian motion
in Section 4.3 (2)--(3). Continuity is supplied by its modification map. -/
abbrev BrownianSample (d : ℕ) := Fin d → ℝ≥0 → ℝ

/-- Product of the constructed Gaussian projective-limit measures,
Section 4.3 (2)--(3). It is an actual probability measure for every dimension. -/
def brownianNoiseLaw (d : ℕ) : Measure (BrownianSample d) :=
  Measure.pi (fun _ => gaussianLimit)

/-- Normalization of the constructed noise measure, Section 4.3 (2)--(3). -/
instance brownianNoiseLaw_probability (d : ℕ) : IsProbabilityMeasure (brownianNoiseLaw d) := by
  dsimp [brownianNoiseLaw]
  infer_instance

/-- Coordinate Brownian process, Section 4.3 (2)--(3). Its value is a
continuous modification of the canonical coordinate process. -/
def coordinateBrownian {d : ℕ} (k : Fin d) (t : ℝ≥0) (ω : BrownianSample d) : ℝ :=
  brownian t (ω k)

/-- Each coordinate is measurable on the product probability space,
Section 4.3 (2)--(3). -/
theorem coordinateBrownian_measurable {d : ℕ} (k : Fin d) (t : ℝ≥0) :
    Measurable (coordinateBrownian k t) :=
  (measurable_brownian t).comp (measurable_pi_apply k)

/-- The constructed coordinate has the actual centered Gaussian law
with variance t, Section 4.3 (2)--(3). -/
theorem coordinateBrownian_hasLaw {d : ℕ} (k : Fin d) (t : ℝ≥0) :
    HasLaw (coordinateBrownian k t) (gaussianReal 0 t) (brownianNoiseLaw d) :=
  hasLaw_brownian_eval.comp ((measurePreserving_eval (fun _ : Fin d => gaussianLimit) k).hasLaw)

/-- Every coordinate of the constructed driving process is Brownian,
including continuity and all finite-dimensional distributions,
Section 4.3 (2)--(3). -/
theorem coordinateBrownian_isBrownian {d : ℕ} (k : Fin d) :
    IsBrownianReal (coordinateBrownian k) (brownianNoiseLaw d) := by
  refine ⟨⟨fun I => ?_⟩, ae_of_all _ (fun ω => continuous_brownian (ω k))⟩
  exact (isBrownianReal_brownian.hasLaw I).comp
    ((measurePreserving_eval (fun _ : Fin d => gaussianLimit) k).hasLaw)

/-- Independence is across entire coordinate processes, not only their
values at one time, Section 4.3's diagonal Brownian driver. -/
theorem coordinateBrownian_independent (d : ℕ) :
    iIndepFun (fun k (ω : BrownianSample d) (t : ℝ≥0) => coordinateBrownian k t ω)
      (brownianNoiseLaw d) := by
  exact iIndepFun_pi (μ := fun _ : Fin d => gaussianLimit)
    (X := fun _ ω t => brownian t ω)
      (fun _ => (Measurable.of_eval measurable_brownian).aemeasurable)

/-- The Euclidean Brownian driver in Section 4.3 (2)--(3), assembled
from the independent scalar coordinates. -/
def vectorBrownian (d : ℕ) (t : ℝ≥0) (ω : BrownianSample d) : EucSpace d :=
  WithLp.toLp 2 (fun k => coordinateBrownian k t ω)

/-- The vector process has continuous sample paths for every sample,
Section 4.3 (2)--(3), including the modification on exceptional samples. -/
theorem vectorBrownian_continuous (d : ℕ) (ω : BrownianSample d) :
    Continuous (fun t => vectorBrownian d t ω) := by
  exact (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp
    (continuous_pi fun k => continuous_brownian (ω k))

/-- Measurability of the actual Euclidean noise, Section 4.3 (2)--(3). -/
theorem vectorBrownian_measurable (d : ℕ) (t : ℝ≥0) :
    Measurable (vectorBrownian d t) := by
  exact (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.measurable.comp
    (Measurable.of_eval fun k => coordinateBrownian_measurable k t)

/-- The constructed Brownian driver starts at zero almost surely,
Section 4.3 (2)--(3). -/
theorem vectorBrownian_zero (d : ℕ) :
    ∀ᵐ ω ∂brownianNoiseLaw d, vectorBrownian d 0 ω = 0 := by
  have hz (k : Fin d) := (coordinateBrownian_isBrownian k).toIsPreBrownianReal.eval_zero_ae_eq_zero
  filter_upwards [ae_all_iff.mpr hz] with ω hω
  ext k
  exact hω k

end Transformer.BatchSize
