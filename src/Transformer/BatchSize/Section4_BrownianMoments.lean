/-
Copyright (c) 2026 Raphael Coelho. All rights reserved.
Released under Apache 2.0 license as described in third_party/Stochastic/LICENSE.
Adapted from proofs by Raphael Coelho.

# Brownian moment identities with coefficients measurable in the joint past

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The factorization proofs adapt MathFin/Foundations/ItoIsometryAdapted.lean
by Raphael Coelho, Apache License 2.0:
https://github.com/formal-applied-math/formal-mathfin.
Here adaptedness is relative to the entire vector driver's filtration.
-/

import Transformer.BatchSize.Section4_BrownianFiltration

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

noncomputable section

namespace Transformer.BatchSize

/-- A forward coordinate increment of the driver of Section 4.3 (2)--(3). -/
def brownianIncrement {d : ℕ} (k : Fin d) (s t : ℝ≥0) (ω : BrownianSample d) : ℝ :=
  coordinateBrownian k t ω - coordinateBrownian k s ω

/-- The increment has its actual Gaussian law, Section 4.3 (2)--(3). -/
theorem brownianIncrement_hasLaw {d : ℕ} (k : Fin d) (s t : ℝ≥0) :
    HasLaw (brownianIncrement k s t) (gaussianReal 0 (nndist (t : ℝ) (s : ℝ)))
      (brownianNoiseLaw d) :=
  (coordinateBrownian_isBrownian k).toIsPreBrownianReal.hasLaw_sub t s

/-- Every increment is square integrable, Section 4.3 (2)--(3). -/
theorem brownianIncrement_memLp {d : ℕ} (k : Fin d) (s t : ℝ≥0) :
    MemLp (brownianIncrement k s t) 2 (brownianNoiseLaw d) :=
  (coordinateBrownian_isBrownian k).isGaussianProcess.hasGaussianLaw_sub.memLp_two

/-- The increment has mean zero, Section 4.3 (2)--(3). -/
theorem brownianIncrement_mean {d : ℕ} (k : Fin d) (s t : ℝ≥0) :
    (∫ ω, brownianIncrement k s t ω ∂brownianNoiseLaw d) = 0 := by
  rw [(brownianIncrement_hasLaw k s t).integral_eq, integral_id_gaussianReal]

/-- The forward increment's second moment is its elapsed time,
Section 4.3 (2)--(3). -/
theorem brownianIncrement_secondMoment {d : ℕ} (k : Fin d) (s t : ℝ≥0)
    (hst : s ≤ t) :
    (∫ ω, brownianIncrement k s t ω ^ 2 ∂brownianNoiseLaw d) = (t : ℝ) - s := by
  have hvar := variance_of_integral_eq_zero
    (μ := gaussianReal 0 (nndist (t : ℝ) (s : ℝ))) aemeasurable_id (by simp)
  rw [variance_id_gaussianReal] at hvar
  simp only [id_eq] at hvar
  have h := (brownianIncrement_hasLaw k s t).integral_comp
    (continuous_pow 2).aestronglyMeasurable
  rw [← hvar] at h
  simpa only [Function.comp_def, coe_nndist, Real.dist_eq,
    abs_of_nonneg (sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst))] using h

/-- A coefficient measurable in the joint past is independent of the
next scalar increment, Section 4.3 (2)--(3). -/
theorem brownianIncrement_independent_adapted {d : ℕ} (k : Fin d) (s t : ℝ≥0)
    (hst : s ≤ t) {H : BrownianSample d → ℝ}
    (hH : StronglyMeasurable[brownianFiltration d s] H) :
    IndepFun H (brownianIncrement k s t) (brownianNoiseLaw d) := by
  apply (IndepFun_iff_Indep _ _ _).mpr
  exact (indep_of_indep_of_le_right
    ((coordinateBrownian_filtered k).indep s t hst)
    (Measurable.comap_le hH.measurable)).symm

/-- Integrable adapted coefficients give a genuinely integrable
martingale difference, Section 4.3 (2)--(3). -/
theorem brownianIncrement_adapted_mean {d : ℕ} (k : Fin d) (s t : ℝ≥0)
    (hst : s ≤ t) {H : BrownianSample d → ℝ}
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHint : Integrable H (brownianNoiseLaw d)) :
    Integrable (fun ω => H ω * brownianIncrement k s t ω) (brownianNoiseLaw d) ∧
      (∫ ω, H ω * brownianIncrement k s t ω ∂brownianNoiseLaw d) = 0 := by
  have hi := brownianIncrement_independent_adapted k s t hst hH
  have hΔ := brownianIncrement_memLp k s t
  refine ⟨hi.integrable_mul hHint (hΔ.integrable (by norm_num)), ?_⟩
  rw [hi.integral_fun_mul_eq_mul_integral hHint.aestronglyMeasurable hΔ.aestronglyMeasurable,
    brownianIncrement_mean, mul_zero]

/-- The adapted second-moment kernel of the Itô isometry,
Section 4.3 (2)--(3), for coefficients depending on all past coordinates. -/
theorem brownianIncrement_adapted_secondMoment {d : ℕ} (k : Fin d) (s t : ℝ≥0)
    (hst : s ≤ t) {H : BrownianSample d → ℝ}
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHint : Integrable H (brownianNoiseLaw d)) :
    Integrable (fun ω => H ω * brownianIncrement k s t ω ^ 2) (brownianNoiseLaw d) ∧
      (∫ ω, H ω * brownianIncrement k s t ω ^ 2 ∂brownianNoiseLaw d) =
        (∫ ω, H ω ∂brownianNoiseLaw d) * ((t : ℝ) - s) := by
  have hi := (brownianIncrement_independent_adapted k s t hst hH).comp
    (φ := id) (ψ := fun z : ℝ => z ^ 2) measurable_id (continuous_pow 2).measurable
  simp only [Function.comp_def, id_eq] at hi
  have hΔ := (brownianIncrement_memLp k s t).integrable_sq
  refine ⟨hi.integrable_mul hHint hΔ, ?_⟩
  rw [hi.integral_fun_mul_eq_mul_integral hHint.aestronglyMeasurable hΔ.aestronglyMeasurable,
    brownianIncrement_secondMoment k s t hst]

/-- Multiplying a square-integrable adapted coefficient by an independent
future increment preserves square integrability, Section 4.3 (2)--(3). -/
theorem brownianIncrement_adapted_memLp {d : ℕ} (k : Fin d) (s t : ℝ≥0)
    (hst : s ≤ t) {H : BrownianSample d → ℝ}
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHL2 : MemLp H 2 (brownianNoiseLaw d)) :
    MemLp (fun ω => H ω * brownianIncrement k s t ω) 2 (brownianNoiseLaw d) := by
  have hm : StronglyMeasurable H := hH.mono ((brownianFiltration d).le s)
  have hΔm := (coordinateBrownian_measurable k t).sub (coordinateBrownian_measurable k s)
  have hi := (brownianIncrement_independent_adapted k s t hst hH).comp
    (φ := fun z : ℝ => z ^ 2) (ψ := fun z : ℝ => z ^ 2)
    (continuous_pow 2).measurable (continuous_pow 2).measurable
  refine (memLp_two_iff_integrable_sq (hm.mul hΔm.stronglyMeasurable).aestronglyMeasurable).mpr ?_
  convert hi.integrable_mul hHL2.integrable_sq
    (brownianIncrement_memLp k s t).integrable_sq using 1
  ext ω
  simp only [Function.comp_def, Pi.mul_apply, Pi.sub_apply, brownianIncrement, mul_pow]

/-- Joint nonvacuity of the adapted moment hypotheses, Section 4.3:
the coefficient is a different Brownian coordinate observed at time one. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 2 1] (coordinateBrownian (1 : Fin 2) 1) ∧
    MemLp (coordinateBrownian (1 : Fin 2) 1) 2 (brownianNoiseLaw 2) ∧
    Integrable (coordinateBrownian (1 : Fin 2) 1) (brownianNoiseLaw 2) := by
  have hp := (coordinateBrownian_isBrownian (1 : Fin 2)).toIsPreBrownianReal
  exact ⟨by norm_num, (coordinateBrownian_filtered (1 : Fin 2)).stronglyAdapted 1,
    (hp.isGaussianProcess.hasGaussianLaw_eval 1).memLp_two, hp.integrable_eval 1⟩

end Transformer.BatchSize
