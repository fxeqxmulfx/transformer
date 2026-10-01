/-
# The actual optimizer generator and bounded diagonal coefficients

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The square of the constructed Brownian amplitude is precisely the paper's
covariance, including eta. No martingale or approximation premise is used.
-/

import Transformer.BatchSize.Section4_GaussianParameters
import Transformer.BatchSize.Section4_RandomGaussianIdentity
import Transformer.BatchSize.Section4_BoundedCoefficients

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Actual global Lipschitz and boundedness constants for both
optimizer diffusion coefficients, Section 4.3 (2)--(3).
The noise amplitude bound includes the true square-root eta factor. -/
theorem optimizerDiffusion_bounded_lipschitz {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ Kb Ka M A : NNReal,
      LipschitzWith Kb (diffusionDrift method B f σ) ∧
      (∀ k, LipschitzWith Ka (fun x => diffusionNoiseScale method η B f σ x k)) ∧
      (∀ x, ‖diffusionDrift method B f σ x‖ ≤ M) ∧
      (∀ x k, |diffusionNoiseScale method η B f σ x k| ≤ A) := by
  obtain ⟨Kb, hb⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨Ka, ha⟩ := diffusionNoiseScale_lipschitz method B f σ hmodel
  obtain ⟨M, hbM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  obtain ⟨A, haA⟩ := diffusionNoiseScale_uniform_bound method B f σ hB hmodel
  refine ⟨Kb, ‖Real.sqrt η‖₊ * Ka, M, ‖Real.sqrt η‖₊ * A, hb, ha η hη, hbM, ?_⟩
  simpa only [NNReal.coe_mul, coe_nnnorm, Real.norm_eq_abs,
    abs_of_nonneg (Real.sqrt_nonneg _)] using haA η hη

/-- The paper's actual optimizer generator is the diagonal
Brownian generator at its current state, Section 4.3 (2)--(3).
The eta factor remains inside the squared noise amplitude. -/
theorem diffusionGenerator_eq_frozen {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (φ : EucSpace d → ℝ) (x : EucSpace d) :
    diffusionGenerator method η B f σ φ x = frozenGaussianGenerator
      (diffusionDrift method B f σ x) (diffusionNoiseScale method η B f σ x) φ x := by
  unfold diffusionGenerator frozenGaussianGenerator
  simp_rw [diffusionNoiseScale_sq method η B f σ x _ hη]

/-- The true optimizer generator is continuous and globally bounded
on every normalized C2 test, Section 4.3 (2)--(3).
These are the actual coefficients, with their model hypotheses proved. -/
theorem diffusionGenerator_continuous_bounded {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Continuous (diffusionGenerator method η B f σ φ) ∧
      ∃ K : ℝ, 0 ≤ K ∧ ∀ x, |diffusionGenerator method η B f σ φ x| ≤ K := by
  obtain ⟨Kb, Ka, M, A, hb, ha, hbM, haA⟩ :=
    optimizerDiffusion_bounded_lipschitz method η B f σ hη hB hmodel
  have heq : diffusionGenerator method η B f σ φ = fun x => frozenGaussianGenerator
      (diffusionDrift method B f σ x) (diffusionNoiseScale method η B f σ x) φ x := by
    funext x
    exact diffusionGenerator_eq_frozen method η B f σ hη φ x
  rw [heq]
  exact ⟨frozenGaussianGenerator_continuous_parameters _ _ _ hb.continuous
      (fun k => (ha k).continuous) continuous_id φ hφ,
    (M : ℝ) + (d : ℝ) * (A : ℝ) ^ 2 / 2, by positivity,
    fun x => frozenGaussianGenerator_uniform_bound _ _ M A (hbM x) (haA x) φ hφ x⟩

/-- Joint nonvacuity of all optimizer generator hypotheses,
Section 4.3: positive rate and batch, flat loss and unit diagonal
noise, with a normalized nonzero C2 observable. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨by norm_num, by norm_num, regularGaussianModel_flat 1, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
