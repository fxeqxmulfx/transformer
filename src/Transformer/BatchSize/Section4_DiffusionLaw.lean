/-
# Constructing the paper's actual optimizer diffusion laws

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The continuous Brownian Euler limit satisfies the genuine generator
martingale problem with all bounded measurable past-cylinder tests.
-/

import Transformer.BatchSize.Section4_PathObservables

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The constructed continuous optimizer Euler law really is a
diffusion law for precisely the paper's drift and covariance,
Section 4.3 (2)--(3), Theorem 1. The full bounded-cylinder martingale
problem, normalization and initial condition are proved. -/
theorem optimizerEulerPathLaw_isDiffusionLaw {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d) :
    IsDiffusionLaw method η B f σ x₀ (optimizerEulerPathLaw method η B f σ hη hB hmodel x₀) := by
  refine ⟨optimizerEulerPathLaw_probability method η B f σ hη hB hmodel x₀,
    optimizerEulerPathLaw_initial method η B f σ hη hB hmodel x₀, ?_⟩
  intro φ hφ s t hs hst n times htimes F hFm hF1
  let Z := optimizerEulerPath method η B f σ hη hB hmodel x₀
  let Q (ω : DiffusionPath d) := F (fun i => ω (times i)) *
    compensatedIncrement method η B f σ φ s t ω
  have hQm : Measurable Q :=
    (hFm.comp (Measurable.of_eval fun i => ContinuousMap.measurable_eval (times i))).mul
      (compensatedIncrement_measurable method η B f σ hη hB hmodel φ hφ s t hst)
  obtain ⟨H, hH, hH1, hFH⟩ := optimizerEulerPath_cylinder_adapted method η B f σ hη hB hmodel x₀
    s hs times htimes F hFm hF1
  have hst' : s.toNNReal ≤ t.toNNReal := by
    apply NNReal.coe_le_coe.mp
    simpa only [Real.coe_toNNReal s hs, Real.coe_toNNReal t (hs.trans hst)] using hst
  have h := optimizerEulerPath_weighted_compensation method η B f σ hη hB hmodel x₀
    s.toNNReal t.toNNReal hst' H hH hH1 φ hφ
  simp only [Real.coe_toNNReal s hs, Real.coe_toNNReal t (hs.trans hst)] at h
  have heq : (fun ω => Q (Z ω)) =ᵐ[brownianNoiseLaw d]
      (fun ω => H ω * compensatedIncrement method η B f σ φ s t (Z ω)) := by
    filter_upwards [hFH] with ω hω
    exact congrArg (fun w => w * compensatedIncrement method η B f σ φ s t (Z ω)) hω
  have hQi : Integrable (fun ω => Q (Z ω)) (brownianNoiseLaw d) :=
    (integrable_congr heq).mpr h.1
  have hQzero : (∫ ω, Q (Z ω) ∂brownianNoiseLaw d) = 0 :=
    (integral_congr_ae heq).trans h.2
  have hZm := optimizerEulerPath_measurable method η B f σ hη hB hmodel x₀
  refine ⟨(integrable_map_measure hQm.aestronglyMeasurable hZm.aemeasurable).mpr hQi, ?_⟩
  rw [optimizerEulerPathLaw, integral_map hZm.aemeasurable hQm.aestronglyMeasurable]
  exact hQzero

/-- Joint nonvacuity of the constructed diffusion-law hypotheses,
Section 4.3: positive rate and batch, flat loss and unit noise. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) :=
  ⟨by norm_num, by norm_num, regularGaussianModel_flat 1⟩

end Transformer.BatchSize
