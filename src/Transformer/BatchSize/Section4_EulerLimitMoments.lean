/-
# Fourth-moment bounds for the constructed optimizer Euler limits

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Fatou transfers actual, grid-uniform Euler estimates to the constructed
limit, retaining the eta-squared factor in the fourth noise moment.
-/

import Transformer.BatchSize.Section4_EulerLimitProcess
import Transformer.BatchSize.Section4_EulerFourthMoment
import Transformer.BatchSize.Section4_LimitMoments

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- Constructed optimizer Euler limits have uniformly bounded fourth
displacement moments with their actual small-noise scaling,
Section 4.3 (2)--(3). The constants are independent of eta, the initial
state, and Euler grid resolution. -/
theorem optimizerEulerLimit_fourthMoment {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ M A : NNReal, ∀ (η : ℝ) (hη : 0 ≤ η) (x₀ : EucSpace d) (t : ℝ≥0),
      Integrable (fun ω => ‖optimizerEulerLimit method η B f σ hη hB hmodel x₀ t ω - x₀‖ ^ 4)
        (brownianNoiseLaw d) ∧
      (∫ ω, ‖optimizerEulerLimit method η B f σ hη hB hmodel x₀ t ω - x₀‖ ^ 4 ∂brownianNoiseLaw d) ≤
        8 * M ^ 4 * (t : ℝ) ^ 4 + 24 * (d : ℝ) ^ 2 * η ^ 2 * A ^ 4 * (t : ℝ) ^ 2 := by
  obtain ⟨Kb, hb⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨Ka, ha⟩ := diffusionNoiseScale_lipschitz method B f σ hmodel
  obtain ⟨M, hbM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  obtain ⟨A, haA⟩ := diffusionNoiseScale_uniform_bound method B f σ hB hmodel
  refine ⟨M, A, ?_⟩
  intro η hη x₀ t
  let b := diffusionDrift method B f σ
  let a := diffusionNoiseScale method η B f σ
  let Aη : NNReal := ‖Real.sqrt η‖₊ * A
  have hAη : ∀ x k, |a x k| ≤ Aη := by
    simpa only [Aη, NNReal.coe_mul, coe_nnnorm, Real.norm_eq_abs,
      abs_of_nonneg (Real.sqrt_nonneg _)] using haA η hη
  let X (m : ℕ) (ω : BrownianSample d) := dyadicEuler b a x₀ t m ω - x₀
  let Y (ω : BrownianSample d) := optimizerEulerLimit method η B f σ hη hB hmodel x₀ t ω - x₀
  have hX2 (m : ℕ) : MemLp (X m) 2 (brownianNoiseLaw d) :=
    (eulerChain_memLp b a Kb (‖Real.sqrt η‖₊ * Ka) hb (ha η hη) _
      (dyadicGrid_monotone t m) x₀ (dyadicEndpointIndex t m)).sub (memLp_const x₀)
  have hX4 (m : ℕ) : MemLp (X m) 4 (brownianNoiseLaw d) :=
    (eulerChain_memLp_four b a hb.continuous (fun k => (ha η hη k).continuous) M Aη hbM hAη
      _ (dyadicGrid_monotone t m) x₀ (dyadicEndpointIndex t m)).sub (memLp_const x₀)
  have hY2 : MemLp Y 2 (brownianNoiseLaw d) :=
    (optimizerEulerLimit_memLp method η B f σ hη hB hmodel x₀ t).sub (memLp_const x₀)
  have hlim : Tendsto (fun m => ∫ ω, ‖X m ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) atTop (𝓝 0) := by
    simpa only [X, Y, sub_sub_sub_cancel_right] using
      optimizerEulerLimit_meanSquare method η B f σ hη hB hmodel x₀ t
  let R : ℝ := 8 * M ^ 4 * (t : ℝ) ^ 4 + 24 * (d : ℝ) ^ 2 * Aη ^ 4 * (t : ℝ) ^ 2
  have hR : 0 ≤ R := by dsimp [R]; positivity
  have hbound (m : ℕ) : (∫ ω, ‖X m ω‖ ^ 4 ∂brownianNoiseLaw d) ≤ R := by
    have h := eulerChain_fourthMoment_bound b a hb.continuous (fun k => (ha η hη k).continuous)
      M Aη hbM hAη (dyadicGrid t m) (dyadicGrid_monotone t m) x₀ (dyadicEndpointIndex t m)
    rw [dyadicGrid_endpoint, dyadicGrid_zero] at h
    simpa only [NNReal.coe_zero, sub_zero, X, dyadicEuler, R] using h
  obtain ⟨hint, hmoment⟩ := fourthMoment_le_of_meanSquare_tendsto (brownianNoiseLaw d) X hX2 hX4 Y hY2
    hlim R hR hbound
  have hroot : Real.sqrt η ^ 4 = η ^ 2 := by
    rw [show Real.sqrt η ^ 4 = (Real.sqrt η ^ 2) ^ 2 by ring, Real.sq_sqrt hη]
  have hscale : (Aη : ℝ) ^ 4 = η ^ 2 * (A : ℝ) ^ 4 := by
    simp only [Aη, NNReal.coe_mul, coe_nnnorm, Real.norm_eq_abs,
      abs_of_nonneg (Real.sqrt_nonneg _), mul_pow, hroot]
  refine ⟨hint, hmoment.trans_eq ?_⟩
  dsimp only [R]
  rw [hscale]
  ring

/-- Joint nonvacuity of optimizer fourth-moment hypotheses,
Section 4.3: a positive batch and the flat regular model with unit noise. -/
example : 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) :=
  ⟨by norm_num, regularGaussianModel_flat 2⟩

end Transformer.BatchSize
