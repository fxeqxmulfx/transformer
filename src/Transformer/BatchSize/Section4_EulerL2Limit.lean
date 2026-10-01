/-
# Constructed adapted L2 limits of the optimizer Euler approximations

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The limits here are constructed at each actual time. Continuous sample
paths and the generator martingale identity remain separate obligations.
-/

import Transformer.BatchSize.Section4_L2Limits
import Transformer.BatchSize.Section4_BoundedCoefficients

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- Bounded globally Lipschitz coefficients produce an actual
square-integrable Euler limit measurable in the joint Brownian past,
Section 4.3 (2)--(3). The limit is obtained by completeness, with no
existence premise for an SDE or a martingale problem. -/
theorem dyadicEuler_exists_adapted_limit {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka M A : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (T : ℝ≥0) :
    ∃ X : BrownianSample d → EucSpace d,
      StronglyMeasurable[brownianFiltration d T] X ∧ MemLp X 2 (brownianNoiseLaw d) ∧
        Tendsto (fun m => ∫ ω, ‖dyadicEuler b a x₀ T m ω - X ω‖ ^ 2 ∂brownianNoiseLaw d)
          atTop (𝓝 0) := by
  have hL2 (m : ℕ) : MemLp (dyadicEuler b a x₀ T m) 2 (brownianNoiseLaw d) :=
    eulerChain_memLp b a Kb Ka hb ha _ (dyadicGrid_monotone T m) x₀ _
  have hmeas (m : ℕ) : StronglyMeasurable[brownianFiltration d T] (dyadicEuler b a x₀ T m) := by
    have h := eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous)
      (dyadicGrid T m) (dyadicGrid_monotone T m) x₀ (dyadicEndpointIndex T m)
    rwa [dyadicGrid_endpoint] at h
  obtain ⟨K, hK, hbound⟩ := dyadicEuler_refinement_bound b a Kb Ka M A hb ha hbM haA x₀ T
  have hbound' (m : ℕ) : (∫ ω, ‖dyadicEuler b a x₀ T m ω - dyadicEuler b a x₀ T (m + 1) ω‖ ^ 2
      ∂brownianNoiseLaw d) ≤ K * (1 / 2 : ℝ) ^ m := by
    simpa only [norm_sub_rev] using hbound m
  obtain ⟨X, hX, hlim⟩ := meanSquare_geometric_limit (brownianNoiseLaw d)
    (dyadicEuler b a x₀ T) hL2 K hK hbound'
  have hm := meanSquare_limit_adapted (brownianNoiseLaw d) (brownianFiltration d T)
    ((brownianFiltration d).le T) (dyadicEuler b a x₀ T) hL2 hmeas X hX hlim
  refine ⟨hm.mk X, hm.stronglyMeasurable_mk, (memLp_congr_ae hm.ae_eq_mk).mp hX, ?_⟩
  have heq (m : ℕ) : (∫ ω, ‖dyadicEuler b a x₀ T m ω - hm.mk X ω‖ ^ 2 ∂brownianNoiseLaw d) =
      ∫ ω, ‖dyadicEuler b a x₀ T m ω - X ω‖ ^ 2 ∂brownianNoiseLaw d := by
    apply integral_congr_ae
    filter_upwards [hm.ae_eq_mk] with ω hω
    rw [hω]
  simpa only [heq] using hlim

/-- Joint nonvacuity of the bounded Lipschitz limit hypotheses,
Section 4.3: zero drift and positive constant diagonal amplitude. -/
example : LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, LipschitzWith 0 (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : ℝ≥0)) :=
  ⟨LipschitzWith.const _, fun _ => LipschitzWith.const _, by simp, by simp⟩

/-- The actual SGD and SignSGD coefficients admit adapted L2 Euler
limits at every nonnegative time, Section 4.3 (2)--(3), Theorem 1. -/
theorem optimizerEuler_exists_adapted_limit {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (x₀ : EucSpace d) (T : ℝ≥0) :
    ∃ X : BrownianSample d → EucSpace d,
      StronglyMeasurable[brownianFiltration d T] X ∧ MemLp X 2 (brownianNoiseLaw d) ∧
        Tendsto (fun m => ∫ ω,
          ‖dyadicEuler (diffusionDrift method B f σ) (diffusionNoiseScale method η B f σ) x₀ T m ω - X ω‖ ^ 2
          ∂brownianNoiseLaw d) atTop (𝓝 0) := by
  obtain ⟨Kb, hb⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨Ka, ha⟩ := diffusionNoiseScale_lipschitz method B f σ hmodel
  obtain ⟨M, hbM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  obtain ⟨A, haA⟩ := diffusionNoiseScale_uniform_bound method B f σ hB hmodel
  exact dyadicEuler_exists_adapted_limit _ _ Kb (‖Real.sqrt η‖₊ * Ka) M
    (‖Real.sqrt η‖₊ * A) hb (ha η hη) hbM
    (by simpa only [NNReal.coe_mul, coe_nnnorm, Real.norm_eq_abs,
      abs_of_nonneg (Real.sqrt_nonneg _)] using haA η hη) x₀ T

/-- Joint nonvacuity of the optimizer-limit hypotheses, Section 4.3:
positive batch and rate, flat loss and unit noise. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) :=
  ⟨by norm_num, by norm_num, regularGaussianModel_flat 2⟩

end Transformer.BatchSize
