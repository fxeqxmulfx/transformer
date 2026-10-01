/-
# The genuine frozen-coefficient Brownian Euler interval

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The random left state determines all coefficients; the next full
Brownian increment supplies the actual future noise.
-/

import Transformer.BatchSize.Section4_BrownianVectorExpectation
import Transformer.BatchSize.Section4_GaussianParameters

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual frozen Euler state starting at a random state at s,
Section 4.3 (2)--(3), using the constructed Brownian increment s..t. -/
def frozenBrownianState {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (X : BrownianSample d → EucSpace d)
    (s t : ℝ≥0) (ω : BrownianSample d) : EucSpace d :=
  gaussianAffineState (X ω + ((t : ℝ) - s) • b (X ω)) (a (X ω))
    (vectorBrownianIncrement d s t ω)

/-- The actual frozen Brownian state is measurable for measurable
left states and continuous coefficients, Section 4.3 (2)--(3). -/
theorem frozenBrownianState_measurable {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (X : BrownianSample d → EucSpace d)
    (hX : Measurable X) (s t : ℝ≥0) : Measurable (frozenBrownianState b a X s t) :=
  gaussianAffineState_measurable_parameters _ _ _
    (hX.add ((hb.measurable.comp hX).const_smul ((t : ℝ) - s)))
    (fun k => (ha k).measurable.comp hX)
    (fun k => (coordinateBrownian_measurable k t).sub (coordinateBrownian_measurable k s))

/-- The frozen interval has continuous actual samples in elapsed
time, with negative elapsed times held at its initial state,
Section 4.3 (2)--(3). -/
theorem frozenBrownianState_continuous_time {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (X : BrownianSample d → EucSpace d)
    (s : ℝ≥0) (ω : BrownianSample d) :
    Continuous (fun u : ℝ => frozenBrownianState b a X s (s + u.toNNReal) ω) := by
  have htime : Continuous (fun u : ℝ => s + u.toNNReal) := continuous_const.add continuous_real_toNNReal
  have hdelta : Continuous (fun u : ℝ => ((s + u.toNNReal : ℝ≥0) : ℝ) - s) :=
    (NNReal.continuous_coe.comp htime).sub continuous_const
  have hnoise : Continuous (fun u : ℝ => vectorBrownianIncrement d s (s + u.toNNReal) ω) := by
    apply continuous_pi
    intro k
    have hB : Continuous (fun t : ℝ≥0 => coordinateBrownian k t ω) := continuous_brownian (ω k)
    exact (hB.comp htime).sub continuous_const
  exact (continuous_const.add (hdelta.smul continuous_const)).add
    ((gaussianDiagonalMap (a (X ω))).continuous.comp hnoise)

/-- Joint measurability of elapsed time and sample for the actual
frozen Brownian interval, Section 4.3 (2)--(3). -/
theorem frozenBrownianState_measurable_time_sample {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (X : BrownianSample d → EucSpace d)
    (hX : Measurable X) (s : ℝ≥0) : Measurable (fun p : ℝ × BrownianSample d =>
      frozenBrownianState b a X s (s + p.1.toNNReal) p.2) :=
  measurable_uncurry_of_continuous_of_measurable
    (frozenBrownianState_continuous_time b a X s)
    (fun u => frozenBrownianState_measurable b a hb ha X hX s (s + u.toNNReal))

/-- Joint nonvacuity of the frozen Brownian interval hypotheses,
Section 4.3: a genuine random left state and unit diagonal noise. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ k : Fin 1, Continuous (fun _ : EucSpace 1 => (k : ℝ) + 1)) ∧
    Measurable (vectorBrownian 1 1) :=
  ⟨continuous_const, fun _ => continuous_const, vectorBrownian_measurable 1 1⟩

end Transformer.BatchSize
