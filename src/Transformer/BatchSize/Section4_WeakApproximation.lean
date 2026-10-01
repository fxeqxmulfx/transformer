/-
# First-order weak approximation by the constructed diffusion law

arXiv:2506.12543v1, Section 4.3, Theorem 1 and equations (2)--(3).
The coefficient computations do not by themselves prove weak convergence.
Here the continuous process is specified by the generator martingale problem
on continuous paths, and the error compares actual optimizer expectations.
-/

import Transformer.BatchSize.Section4_DiffusionLaw
import Transformer.BatchSize.Section4_DiscreteDeterministicComparison
import Transformer.BatchSize.Section4_OptimizerDeterministicComparison

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Corrected precise form of Section 4.3, Theorem 1, together with its
SGD counterpart (2): one SDE law approximates every discrete time k*eta
on a fixed finite horizon, with expectation error O(eta).

The paper omits the smoothness, nondegeneracy, test-function and horizon
conditions. They are explicit here. Sigma is diagonal with entries sigma_k^2;
the coefficients are exactly (2)--(3), without momentum or bias correction.
Uniform discrete one-step consistency, the Markov contraction and the
global telescope are proved in the accompanying modules. The actual
Brownian Euler limit has continuous paths, the prescribed initial state
and the full bounded-cylinder generator martingale identity. Both its
expectations and the true discrete law are compared with the same mean
Euler trajectory using uniform finite-horizon C2 backward test bounds.
The result holds for every positive eta; no eta <= 1 restriction is needed. -/
theorem sgd_sign_first_order_weak_approximation {d : ℕ}
    (method : UpdateKind) (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (x₀ : EucSpace d) (T : ℝ) (hT : 0 ≤ T) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η : ℝ, 0 < η →
      ∃ P : Measure (DiffusionPath d), IsDiffusionLaw method η B f σ x₀ P ∧
        ∀ φ, BoundedSmoothTest 6 φ → ∀ k : ℕ, (k : ℝ) * η ≤ T →
          |(∫ x, φ x ∂discreteLaw method η B f σ x₀ k) -
            ∫ ω, φ (ω ((k : ℝ) * η)) ∂P| ≤ C * η := by
  obtain ⟨C₁, hC₁, hdiscrete⟩ :=
    sgd_sign_discrete_deterministic_weak_error method B f σ hB hmodel T hT
  obtain ⟨C₂, hC₂, hpath⟩ :=
    optimizerEulerPath_deterministic_weak_error method B f σ hB hmodel T hT
  refine ⟨C₁ + C₂, add_nonneg hC₁ hC₂, ?_⟩
  intro η hη
  let ε : NNReal := ⟨η, hη.le⟩
  let P := optimizerEulerPathLaw method η B f σ hη.le hB hmodel x₀
  refine ⟨P, optimizerEulerPathLaw_isDiffusionLaw method η B f σ hη.le hB hmodel x₀, ?_⟩
  intro φ hφ k hk
  have hφ2 : BoundedSmoothTest 2 φ :=
    ⟨hφ.1.of_le (by norm_num), fun j hj y => hφ.2 j (by omega) y⟩
  have hd := hdiscrete ε φ hφ2 k hk x₀
  have hp := hpath ε hη φ hφ2 k hk x₀
  have hmap : (∫ ω, φ (ω ((k : ℝ) * η)) ∂P) =
      ∫ ω, φ (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω ((k : ℝ) * η))
        ∂brownianNoiseLaw d := by
    dsimp only [P]
    rw [optimizerEulerPathLaw]
    have hm : Measurable (fun ω : DiffusionPath d => φ (ω ((k : ℝ) * η))) :=
      hφ.1.continuous.measurable.comp (ContinuousMap.measurable_eval ((k : ℝ) * η))
    exact integral_map
      (optimizerEulerPath_measurable method η B f σ hη.le hB hmodel x₀).aemeasurable
      hm.aestronglyMeasurable
  rw [hmap]
  calc
    _ ≤ |(∫ x, φ x ∂discreteLaw method η B f σ x₀ k) -
        deterministicEulerTest (diffusionDrift method B f σ) ε φ k x₀| +
      |deterministicEulerTest (diffusionDrift method B f σ) ε φ k x₀ -
        (∫ ω, φ (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω ((k : ℝ) * η))
          ∂brownianNoiseLaw d)| := abs_sub_le _ _ _
    _ ≤ C₁ * η + C₂ * η := add_le_add hd (by rw [abs_sub_comm]; exact hp)
    _ = _ := by ring

/-- Nonvacuity of all hypotheses of the corrected weak-approximation
claim, Section 4.3: a flat loss with unit Gaussian noise on a unit horizon. -/
example : 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
      (fun _ => EuclideanSpace.single (0 : Fin 1) 1) ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨by norm_num, ?_, by norm_num⟩
  refine ⟨contDiff_const, contDiff_const, 1, 1, by norm_num, le_rfl, ?_, ?_, ?_⟩
  · intro x k
    fin_cases k
    simp [EuclideanSpace.single]
  · intro j hj hupper x
    rw [iteratedFDeriv_const_of_ne (by omega)]
    simp
  · intro j hj hupper x
    rw [iteratedFDeriv_const_of_ne (by omega)]
    simp

end Transformer.BatchSize
