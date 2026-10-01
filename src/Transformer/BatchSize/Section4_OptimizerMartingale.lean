/-
# The constructed optimizer paths satisfy the actual martingale identity

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The previously constructed continuous Euler modification is identified
with the paper's generator using its true adapted Brownian approximations.
-/

import Transformer.BatchSize.Section4_OptimizerGenerator
import Transformer.BatchSize.Section4_EulerLimitGenerator
import Transformer.BatchSize.Section4_EulerLimitLaw
import Transformer.BatchSize.Section4_PathCompensation

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual continuous optimizer Euler path satisfies the
weighted generator identity with every bounded past weight,
Section 4.3 (2)--(3). Existence, adaptation and all state limits
come from proved Brownian constructions rather than an SDE premise. -/
theorem optimizerEulerPath_weighted_generator_identity {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d)
    (s t : NNReal) (hst : s ≤ t) (F : BrownianSample d → ℝ)
    (hF : StronglyMeasurable[brownianFiltration d s] F) (hF1 : ∀ ω, |F ω| ≤ 1)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    (∫ ω, F ω * φ (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω t) ∂brownianNoiseLaw d) -
      (∫ ω, F ω * φ (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω s) ∂brownianNoiseLaw d) =
      ∫ u in (s : ℝ)..(t : ℝ), ∫ ω, F ω * diffusionGenerator method η B f σ φ
        (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω u) ∂brownianNoiseLaw d := by
  obtain ⟨Kb, Ka, M, A, hb, ha, hbM, haA⟩ :=
    optimizerDiffusion_bounded_lipschitz method η B f σ hη hB hmodel
  have h := eulerLimit_weighted_generator_identity _ _ Kb Ka M A hb ha hbM haA x₀ s t hst
    (optimizerEulerLimit method η B f σ hη hB hmodel x₀)
    (optimizerEulerLimit_memLp method η B f σ hη hB hmodel x₀)
    (optimizerEulerLimit_meanSquare method η B f σ hη hB hmodel x₀)
    (optimizerEulerPath method η B f σ hη hB hmodel x₀)
    (optimizerEulerPath_eval_ae_eq method η B f σ hη hB hmodel x₀) F hF hF1 φ hφ
  simp_rw [← diffusionGenerator_eq_frozen method η B f σ hη φ] at h
  exact h

/-- Integrability and zero mean of the true optimizer compensated
observable with every bounded Brownian-past weight,
Section 4.3 (2)--(3). The actual path law will inherit this identity. -/
theorem optimizerEulerPath_weighted_compensation {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d)
    (s t : NNReal) (hst : s ≤ t) (F : BrownianSample d → ℝ)
    (hF : StronglyMeasurable[brownianFiltration d s] F) (hF1 : ∀ ω, |F ω| ≤ 1)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Integrable (fun ω => F ω * compensatedIncrement method η B f σ φ s t
      (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω)) (brownianNoiseLaw d) ∧
      (∫ ω, F ω * compensatedIncrement method η B f σ φ s t
        (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω) ∂brownianNoiseLaw d) = 0 := by
  obtain ⟨hg, K, hK, hbound⟩ := diffusionGenerator_continuous_bounded method η B f σ hη hB hmodel φ hφ
  exact continuousPath_weighted_compensation (brownianNoiseLaw d) _
    (optimizerEulerPath_measurable method η B f σ hη hB hmodel x₀) F
    (hF.mono ((brownianFiltration d).le s)).measurable hF1 φ hφ _ hg K hbound s t
    (NNReal.coe_le_coe.mpr hst)
    (optimizerEulerPath_weighted_generator_identity method η B f σ hη hB hmodel x₀ s t hst F hF hF1 φ hφ)

/-- Joint nonvacuity of the actual optimizer martingale hypotheses,
Section 4.3: positive rate and batch, flat loss and unit noise,
positive interval, unit past weight and normalized nonzero C2 test. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (1 : NNReal) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    |(1 : ℝ)| ≤ 1 ∧ BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨by norm_num, by norm_num, regularGaussianModel_flat 1, by norm_num,
    stronglyMeasurable_const, by norm_num, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
