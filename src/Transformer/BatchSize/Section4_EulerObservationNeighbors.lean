/-
# Actual neighboring Euler states converge at every observation time

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The arbitrary-window moment estimate controls the difference between a
chosen adjacent grid state and the Brownian interpolation at that time.
-/

import Transformer.BatchSize.Section4_EulerIntervals
import Transformer.BatchSize.Section4_EulerWindowSecondMoment
import Transformer.BatchSize.Section4_DyadicObservationTimes
import Transformer.BatchSize.Section4_LimitMoments

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- Every actual dyadic grid state within one mesh of an observation
has the same genuine mean-square limit as the interpolated state,
Section 4.3 (2)--(3). This includes both floor and ceiling neighbors. -/
theorem dyadicEuler_neighbor_meanSquare {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ) (Kb Ka M A : NNReal)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (u T : NNReal) (huT : u ≤ T) (J : ℕ → ℕ)
    (hJ : ∀ m, J m ≤ dyadicEndpointIndex T m)
    (htime : ∀ m, |(dyadicGrid T m (J m) : ℝ) - u| ≤ (1 / 2 : ℝ) ^ m)
    (Y : BrownianSample d → EucSpace d) (hY : MemLp Y 2 (brownianNoiseLaw d))
    (hlim : Tendsto (fun m => ∫ ω, ‖dyadicEuler b a x₀ u m ω - Y ω‖ ^ 2
      ∂brownianNoiseLaw d) atTop (𝓝 0)) :
    (∀ m, MemLp (eulerChain b a (dyadicGrid T m) x₀ (J m)) 2 (brownianNoiseLaw d)) ∧
      Tendsto (fun m => ∫ ω, ‖eulerChain b a (dyadicGrid T m) x₀ (J m) ω - Y ω‖ ^ 2
        ∂brownianNoiseLaw d) atTop (𝓝 0) := by
  let X (m : ℕ) := eulerChain b a (dyadicGrid T m) x₀ (J m)
  let Z := dyadicEuler b a x₀ u
  have hX2 (m : ℕ) : MemLp (X m) 2 (brownianNoiseLaw d) :=
    eulerChain_memLp b a Kb Ka hb ha _ (dyadicGrid_monotone T m) x₀ (J m)
  have hZ2 (m : ℕ) : MemLp (Z m) 2 (brownianNoiseLaw d) :=
    eulerChain_memLp b a Kb Ka hb ha _ (dyadicGrid_monotone u m) x₀ _
  have hXZ (m : ℕ) : (∫ ω, ‖X m ω - Z m ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      2 * M ^ 2 * ((1 / 2 : ℝ) ^ m) ^ 2 + 2 * (d : ℝ) * A ^ 2 * (1 / 2 : ℝ) ^ m := by
    have h := eulerPathValue_increment_meanSquare_abs b a hb.continuous
      (fun k => (ha k).continuous) M A hbM haA _ (dyadicGrid_monotone T m) x₀
      u (dyadicGrid T m (J m)) (dyadicEndpointIndex T m)
    simp_rw [eulerPathValue_grid b a _ (dyadicGrid_monotone T m) x₀ _ _ (hJ m),
      ← dyadicEuler_eq_pathValue b a x₀ u T huT m] at h
    exact h.2.trans (by gcongr; exact htime m; exact htime m)
  have hbound (m : ℕ) : (∫ ω, ‖X m ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      2 * (2 * M ^ 2 * ((1 / 2 : ℝ) ^ m) ^ 2 + 2 * (d : ℝ) * A ^ 2 * (1 / 2 : ℝ) ^ m) +
        2 * ∫ ω, ‖Z m ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d := by
    have h := integral_mono ((hX2 m).sub hY |>.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      ((((hX2 m).sub (hZ2 m)).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul 2 |>.add
        (((hZ2 m).sub hY).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0) |>.const_mul 2))
      (fun ω => norm_sub_split_sq_le (X m ω) (Z m ω) (Y ω))
    simp only [Pi.add_apply] at h
    rw [integral_add
      ((((hX2 m).sub (hZ2 m)).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul 2)
      ((((hZ2 m).sub hY).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul 2),
      integral_const_mul, integral_const_mul] at h
    exact h.trans (add_le_add
      (mul_le_mul_of_nonneg_left (hXZ m) (by norm_num : (0 : ℝ) ≤ 2)) le_rfl)
  have hp : Tendsto (fun m : ℕ => (1 / 2 : ℝ) ^ m) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  refine ⟨hX2, squeeze_zero (fun m => integral_nonneg (fun ω => sq_nonneg _)) hbound ?_⟩
  have h := ((hp.pow 2 |>.const_mul (2 * (M : ℝ) ^ 2)).add
    (hp.const_mul (2 * (d : ℝ) * (A : ℝ) ^ 2))).const_mul 2 |>.add (hlim.const_mul 2)
  simpa only [zero_pow (by norm_num : (2 : ℕ) ≠ 0), mul_zero, add_zero] using h

/-- Joint nonvacuity of neighbor convergence hypotheses, Section 4.3:
the actual zero-time neighbor and a nonzero constant mean-square limit. -/
example : (0 : NNReal) ≤ 1 ∧
    (∀ m : ℕ, (0 : ℕ) ≤ dyadicEndpointIndex 1 m) ∧
    (∀ m : ℕ, |(dyadicGrid 1 m 0 : ℝ) - 0| ≤ (1 / 2 : ℝ) ^ m) ∧
    MemLp (fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 2
      (brownianNoiseLaw 1) ∧
    Tendsto (fun _ : ℕ => ∫ ω : BrownianSample 1,
      ‖(fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ω -
        EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ ^ 2 ∂brownianNoiseLaw 1) atTop (𝓝 0) := by
  refine ⟨by norm_num, fun m => Nat.zero_le _, ?_, memLp_const _, by simp⟩
  intro m
  simp only [dyadicGrid_zero, NNReal.coe_zero, sub_self, abs_zero]
  positivity

end Transformer.BatchSize
