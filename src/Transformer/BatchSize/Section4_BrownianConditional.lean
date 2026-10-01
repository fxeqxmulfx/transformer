/-
# Conditional moments with respect to the joint Brownian past

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
These conditional identities use integrable Gaussian increments and the
constructed joint filtration. Conditional expectation's default value for
nonintegrable inputs is never used to manufacture a martingale identity.
-/

import Transformer.BatchSize.Section4_BrownianMoments

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A future Brownian increment has conditional mean zero given the
entire joint past, Section 4.3 (2)--(3). -/
theorem brownianIncrement_conditional_mean {d : ℕ} (k : Fin d) (s t : ℝ≥0) (hst : s ≤ t) :
    (brownianNoiseLaw d)[brownianIncrement k s t | brownianFiltration d s] =ᵐ[brownianNoiseLaw d] 0 := by
  have hp := (coordinateBrownian_isBrownian k).toIsPreBrownianReal
  have hs := condExp_of_stronglyMeasurable ((brownianFiltration d).le s)
    ((coordinateBrownian_filtered k).stronglyAdapted s) (hp.integrable_eval s)
  have hsub := condExp_sub (hp.integrable_eval t) (hp.integrable_eval s) (brownianFiltration d s)
  rw [hs] at hsub
  have ht := (coordinateBrownian_martingale k).condExp_ae_eq hst
  filter_upwards [hsub, ht] with ω hω hωt
  change ((brownianNoiseLaw d)[coordinateBrownian k t - coordinateBrownian k s |
    brownianFiltration d s]) ω = 0
  simpa only [Pi.sub_apply, hωt, sub_self] using hω

/-- Conditional Brownian variance equals elapsed time, Section 4.3
(2)--(3). Square integrability is exhibited together with the identity. -/
theorem brownianIncrement_conditional_secondMoment {d : ℕ} (k : Fin d)
    (s t : ℝ≥0) (hst : s ≤ t) :
    Integrable (fun ω => brownianIncrement k s t ω ^ 2) (brownianNoiseLaw d) ∧
      (brownianNoiseLaw d)[fun ω => brownianIncrement k s t ω ^ 2 | brownianFiltration d s]
        =ᵐ[brownianNoiseLaw d] fun _ => (t : ℝ) - s := by
  refine ⟨(brownianIncrement_memLp k s t).integrable_sq, ?_⟩
  have hm : Measurable (brownianIncrement k s t) :=
    (coordinateBrownian_measurable k t).sub (coordinateBrownian_measurable k s)
  have hsquare : StronglyMeasurable[MeasurableSpace.comap (brownianIncrement k s t) inferInstance]
      (fun ω => brownianIncrement k s t ω ^ 2) :=
    (continuous_pow 2).comp_stronglyMeasurable (comap_measurable (brownianIncrement k s t)).stronglyMeasurable
  have h := condExp_indep_eq (Measurable.comap_le hm) ((brownianFiltration d).le s)
    hsquare ((coordinateBrownian_filtered k).indep s t hst)
  rw [brownianIncrement_secondMoment k s t hst] at h
  exact h

/-- An L2 coefficient measurable in the joint past gives conditional
mean zero for its future noise update, Section 4.3 (2)--(3). -/
theorem brownianIncrement_adapted_conditional_mean {d : ℕ} (k : Fin d)
    (H : BrownianSample d → ℝ) (s t : ℝ≥0) (hst : s ≤ t)
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHL2 : MemLp H 2 (brownianNoiseLaw d)) :
    (brownianNoiseLaw d)[fun ω => H ω * brownianIncrement k s t ω | brownianFiltration d s]
      =ᵐ[brownianNoiseLaw d] 0 := by
  have hint := (brownianIncrement_adapted_mean k s t hst hH
    (hHL2.integrable (by norm_num))).1
  have hpull := condExp_mul_of_stronglyMeasurable_left hH hint
    ((brownianIncrement_memLp k s t).integrable (by norm_num))
  simp only [Pi.mul_def] at hpull
  have hzero := brownianIncrement_conditional_mean k s t hst
  filter_upwards [hpull, hzero] with ω hω hωzero
  simpa only [Pi.mul_apply, hωzero, Pi.zero_apply, mul_zero] using hω

/-- Joint nonvacuity of all conditional moment hypotheses, Section 4.3:
a coefficient from another coordinate of the actual joint past. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 2 1] (coordinateBrownian (1 : Fin 2) 1) ∧
    MemLp (coordinateBrownian (1 : Fin 2) 1) 2 (brownianNoiseLaw 2) :=
  ⟨by norm_num, (coordinateBrownian_filtered (1 : Fin 2)).stronglyAdapted 1,
    ((coordinateBrownian_isBrownian (1 : Fin 2)).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two⟩

end Transformer.BatchSize
