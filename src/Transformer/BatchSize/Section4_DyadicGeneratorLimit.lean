/-
# Passing the actual frozen Euler generator to its state limit

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Both the preceding node and the interpolated Brownian state converge in
measure; bounded continuous observables pass their generator expectations.
-/

import Transformer.BatchSize.Section4_DyadicNeighborLimits
import Transformer.BatchSize.Section4_BoundedObservableLimits

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- The genuine weighted frozen-generator expectation converges
to the actual limit-state generator at every observation time,
Section 4.3 (2)--(3). Any almost-surely equal state representative
can be used at the limit, including the continuous modification. -/
theorem dyadicGenerator_expectation_tendsto {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ) (Kb Ka M A : NNReal)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (u T : NNReal) (huT : u ≤ T)
    (Y : BrownianSample d → EucSpace d) (hY : MemLp Y 2 (brownianNoiseLaw d))
    (hlim : Tendsto (fun m => ∫ ω, ‖dyadicEuler b a x₀ u m ω - Y ω‖ ^ 2
      ∂brownianNoiseLaw d) atTop (𝓝 0))
    (Z : BrownianSample d → EucSpace d) (hZY : Z =ᵐ[brownianNoiseLaw d] Y)
    (F : BrownianSample d → ℝ) (hFm : Measurable F) (hF1 : ∀ ω, |F ω| ≤ 1)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Tendsto (fun m => ∫ ω, F ω * dyadicGeneratorValue b a x₀ T m φ u ω ∂brownianNoiseLaw d)
      atTop (𝓝 (∫ ω, F ω * frozenGaussianGenerator (b (Z ω)) (a (Z ω)) φ (Z ω)
        ∂brownianNoiseLaw d)) := by
  let L (m : ℕ) := dyadicLeftState b a x₀ T m u
  let S (m : ℕ) := fun ω => eulerPathValue b a (dyadicGrid T m) x₀
    (dyadicEndpointIndex T m) ω u
  have hL := (dyadicEuler_neighbors_meanSquare b a Kb Ka M A hb ha hbM haA x₀ u T huT Y hY hlim).1
  have hLin := meanSquare_tendstoInMeasure (brownianNoiseLaw d) L hL.1 Y hY hL.2
  have hSeq (m : ℕ) : S m = dyadicEuler b a x₀ u m := by
    funext ω
    exact (dyadicEuler_eq_pathValue b a x₀ u T huT m ω).symm
  have hS2 (m : ℕ) : MemLp (S m) 2 (brownianNoiseLaw d) := by
    rw [hSeq]
    exact eulerChain_memLp b a Kb Ka hb ha _ (dyadicGrid_monotone u m) x₀ _
  have hSin := meanSquare_tendstoInMeasure (brownianNoiseLaw d) S hS2 Y hY
    (by simpa only [hSeq] using hlim)
  let g (ω : BrownianSample d) (p : EucSpace d × EucSpace d) :=
    F ω * frozenGaussianGenerator (b p.1) (a p.1) φ p.2
  have hgc (ω : BrownianSample d) : Continuous (g ω) :=
    continuous_const.mul (frozenGaussianGenerator_continuous_parameters _ _ _
      (hb.continuous.comp continuous_fst) (fun k => (ha k).continuous.comp continuous_fst)
      continuous_snd φ hφ)
  have hgm (m : ℕ) : AEStronglyMeasurable (fun ω => g ω (L m ω, S m ω))
      (brownianNoiseLaw d) :=
    (hFm.mul ((dyadicGeneratorValue_measurable b a hb.continuous (fun k => (ha k).continuous)
      x₀ T m φ hφ).comp (measurable_const.prodMk measurable_id))).aestronglyMeasurable
  have hbound (ω : BrownianSample d) (p : EucSpace d × EucSpace d) :
      |g ω p| ≤ (M : ℝ) + (d : ℝ) * (A : ℝ) ^ 2 / 2 := by
    dsimp only [g]
    rw [abs_mul]
    exact (mul_le_mul (hF1 ω) (frozenGaussianGenerator_uniform_bound _ _ M A
      (hbM _) (haA _) φ hφ _) (abs_nonneg _) (by norm_num)).trans_eq (one_mul _)
  have h := integral_bounded_pair_observable_tendsto (brownianNoiseLaw d) L Y hLin S Y hSin
    g hgc hgm ((M : ℝ) + (d : ℝ) * (A : ℝ) ^ 2 / 2) hbound
  have heq : (∫ ω, g ω (Y ω, Y ω) ∂brownianNoiseLaw d) =
      ∫ ω, F ω * frozenGaussianGenerator (b (Z ω)) (a (Z ω)) φ (Z ω) ∂brownianNoiseLaw d := by
    apply integral_congr_ae
    filter_upwards [hZY] with ω hω
    rw [hω]
  rw [heq] at h
  exact h

/-- Joint nonvacuity of the two generator-state convergence
hypotheses, Section 4.3: constant bounded coefficients with a
nonzero initial state and its actual zero-time limit. -/
example : LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, LipschitzWith 0 (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (0 : NNReal) ≤ 1 ∧
    MemLp (fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 2
      (brownianNoiseLaw 1) ∧
    Tendsto (fun _ : ℕ => ∫ ω : BrownianSample 1,
      ‖(fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ω -
        EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ ^ 2 ∂brownianNoiseLaw 1) atTop (𝓝 0) ∧
    ((fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) =ᵐ[brownianNoiseLaw 1]
      fun _ => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) :=
  ⟨LipschitzWith.const _, fun _ => LipschitzWith.const _, by norm_num, memLp_const _,
    by simp, Eventually.of_forall fun _ => rfl⟩

end Transformer.BatchSize
