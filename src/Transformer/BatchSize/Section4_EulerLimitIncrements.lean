/-
# Arbitrary increment moments of the constructed optimizer limits

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Consistent dyadic interpolations give mesh-uniform increment estimates.
Mean-square convergence and Fatou transfer these to the actual limits.
-/

import Transformer.BatchSize.Section4_EulerIncrementMoments
import Transformer.BatchSize.Section4_EulerLimitMoments

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- Stopped dyadic Euler values at two arbitrary times have the same
fourth-moment increment bound as their common interpolation,
Section 4.3 (2)--(3). -/
theorem dyadicEuler_increment_fourthMoment {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (M A : ℝ≥0)
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (s v : ℝ≥0) (hsv : s ≤ v) (m : ℕ) :
    MemLp (fun ω => dyadicEuler b a x₀ v m ω - dyadicEuler b a x₀ s m ω) 4 (brownianNoiseLaw d) ∧
    (∫ ω, ‖dyadicEuler b a x₀ v m ω - dyadicEuler b a x₀ s m ω‖ ^ 4 ∂brownianNoiseLaw d) ≤
      8 * M ^ 4 * ((v : ℝ) - s) ^ 4 + 24 * (d : ℝ) ^ 2 * A ^ 4 * ((v : ℝ) - s) ^ 2 := by
  have heq : (fun ω => dyadicEuler b a x₀ v m ω - dyadicEuler b a x₀ s m ω) =
      (fun ω => eulerPathValue b a (dyadicGrid v m) x₀ (dyadicEndpointIndex v m) ω v -
        eulerPathValue b a (dyadicGrid v m) x₀ (dyadicEndpointIndex v m) ω s) := by
    funext ω
    rw [dyadicEuler_eq_pathValue b a x₀ s v hsv, dyadicEuler_eq_pathValue b a x₀ v v le_rfl]
  change MemLp (fun ω => dyadicEuler b a x₀ v m ω - dyadicEuler b a x₀ s m ω) 4 _ ∧ _
  rw [heq]
  simp_rw [dyadicEuler_eq_pathValue b a x₀ s v hsv m, dyadicEuler_eq_pathValue b a x₀ v v le_rfl m]
  exact eulerPathValue_increment_fourthMoment b a hb ha M A hbM haA
    (dyadicGrid v m) (dyadicGrid_monotone v m) x₀ s v hsv (dyadicEndpointIndex v m)

/-- Joint nonvacuity of dyadic increment hypotheses, Section 4.3:
constant bounded coefficients and two distinct nonnegative times. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, Continuous (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : ℝ≥0)) ∧ (1 : ℝ≥0) ≤ 2 :=
  ⟨continuous_const, fun _ => continuous_const, by simp, by simp, by norm_num⟩

/-- The actual optimizer limits satisfy arbitrary-time fourth-moment
increment estimates, Section 4.3 (2)--(3), with constants independent
of eta, the initial state, and the dyadic resolution. -/
theorem optimizerEulerLimit_increment_fourthMoment {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ M A : NNReal, ∀ (η : ℝ) (hη : 0 ≤ η) (x₀ : EucSpace d) (s v : ℝ≥0), s ≤ v →
      Integrable (fun ω =>
        ‖optimizerEulerLimit method η B f σ hη hB hmodel x₀ v ω -
          optimizerEulerLimit method η B f σ hη hB hmodel x₀ s ω‖ ^ 4) (brownianNoiseLaw d) ∧
      (∫ ω, ‖optimizerEulerLimit method η B f σ hη hB hmodel x₀ v ω -
          optimizerEulerLimit method η B f σ hη hB hmodel x₀ s ω‖ ^ 4 ∂brownianNoiseLaw d) ≤
        8 * M ^ 4 * ((v : ℝ) - s) ^ 4 + 24 * (d : ℝ) ^ 2 * η ^ 2 * A ^ 4 * ((v : ℝ) - s) ^ 2 := by
  obtain ⟨Kb, hb⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨Ka, ha⟩ := diffusionNoiseScale_lipschitz method B f σ hmodel
  obtain ⟨M, hbM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  obtain ⟨A, haA⟩ := diffusionNoiseScale_uniform_bound method B f σ hB hmodel
  refine ⟨M, A, ?_⟩
  intro η hη x₀ s v hsv
  let b := diffusionDrift method B f σ
  let a := diffusionNoiseScale method η B f σ
  let Aη : NNReal := ‖Real.sqrt η‖₊ * A
  have hAη : ∀ x k, |a x k| ≤ Aη := by
    simpa only [Aη, NNReal.coe_mul, coe_nnnorm, Real.norm_eq_abs,
      abs_of_nonneg (Real.sqrt_nonneg _)] using haA η hη
  let X (m : ℕ) := dyadicEuler b a x₀ v m
  let Z (m : ℕ) := dyadicEuler b a x₀ s m
  let Y := optimizerEulerLimit method η B f σ hη hB hmodel x₀ v
  let W := optimizerEulerLimit method η B f σ hη hB hmodel x₀ s
  have hX2 (m : ℕ) : MemLp (X m) 2 (brownianNoiseLaw d) :=
    eulerChain_memLp b a Kb (‖Real.sqrt η‖₊ * Ka) hb (ha η hη) _ (dyadicGrid_monotone v m) x₀ _
  have hZ2 (m : ℕ) : MemLp (Z m) 2 (brownianNoiseLaw d) :=
    eulerChain_memLp b a Kb (‖Real.sqrt η‖₊ * Ka) hb (ha η hη) _ (dyadicGrid_monotone s m) x₀ _
  have hY2 := optimizerEulerLimit_memLp method η B f σ hη hB hmodel x₀ v
  have hW2 := optimizerEulerLimit_memLp method η B f σ hη hB hmodel x₀ s
  have hlim := meanSquare_sub_tendsto (brownianNoiseLaw d) X Z hX2 hZ2 Y W hY2 hW2
    (optimizerEulerLimit_meanSquare method η B f σ hη hB hmodel x₀ v)
    (optimizerEulerLimit_meanSquare method η B f σ hη hB hmodel x₀ s)
  let R : ℝ := 8 * M ^ 4 * ((v : ℝ) - s) ^ 4 + 24 * (d : ℝ) ^ 2 * Aη ^ 4 * ((v : ℝ) - s) ^ 2
  have hR : 0 ≤ R := by dsimp [R]; positivity
  have hstep (m : ℕ) := dyadicEuler_increment_fourthMoment b a hb.continuous
    (fun k => (ha η hη k).continuous) M Aη hbM hAη x₀ s v hsv m
  obtain ⟨hint, hmoment⟩ := fourthMoment_le_of_meanSquare_tendsto (brownianNoiseLaw d)
    (fun m ω => X m ω - Z m ω) (fun m => (hX2 m).sub (hZ2 m)) (fun m => (hstep m).1)
    (fun ω => Y ω - W ω) (hY2.sub hW2) hlim R hR (fun m => (hstep m).2)
  have hroot : Real.sqrt η ^ 4 = η ^ 2 := by
    rw [show Real.sqrt η ^ 4 = (Real.sqrt η ^ 2) ^ 2 by ring, Real.sq_sqrt hη]
  have hscale : (Aη : ℝ) ^ 4 = η ^ 2 * (A : ℝ) ^ 4 := by
    simp only [Aη, NNReal.coe_mul, coe_nnnorm, Real.norm_eq_abs,
      abs_of_nonneg (Real.sqrt_nonneg _), mul_pow, hroot]
  refine ⟨hint, hmoment.trans_eq ?_⟩
  dsimp only [R]
  rw [hscale]
  ring

/-- Joint nonvacuity of optimizer increment hypotheses, Section 4.3:
the flat regular model and a positive batch. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) :=
  ⟨by norm_num, regularGaussianModel_flat 2⟩

end Transformer.BatchSize
