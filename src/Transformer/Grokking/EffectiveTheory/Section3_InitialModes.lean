import Transformer.Grokking.EffectiveTheory.Section3_InitialDomain
import Mathlib.Order.Filter.AtTopBot.Field

/-!
# Spectral decay and effective-loss convergence from initial ground data

Source: Liu et al., arXiv:2205.10343v2, section 3.2, "Time towards the
linear structure". The source's raw linear-flow deduction is false for
the quotient loss. The corrected single-constraint model has an exact
exponential law for the residual relative to the ground coordinate.

The hypotheses now specify differentiable paths obeying the actual ODE
off the origin and a nonzero initial ground coordinate. Domain preservation,
norm conservation and ground nonvanishing are derived. A bound on the
ground coordinate then proves that the normalized loss tends to zero.
The law does not establish classifier accuracy or AdamW convergence.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- Initial ground data suffices for all-time ground nonvanishing without
an assumed nonzero representation domain. Source correction:
arXiv:2205.10343v2, section 3.2, normalized embedding dynamics. -/
theorem ground_nonzero_off_origin (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z) (h0 : x 0 - z 0 ≠ 0) :
    ∀ t, x t - z t ≠ 0 := by
  have hZ0 := squaredNorm_ne_zero_of_ground (x 0) (y 0) (z 0) h0
  have hZ := nonzero_domain_from_initial x y z hdx hdy hdz hflow hZ0
  have hf := full_gradient_flow_from_initial x y z hdx hdy hdz hflow hZ0
  exact ground_nonzero_from_initial x y z hZ hf h0

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) ∧ (1 : ℝ) - (-1) ≠ 0 := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_, by norm_num⟩
  intro t hZ
  have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
  refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- The corrected exact spectral law from initial data only. Source
correction: arXiv:2205.10343v2, section 3.2; the exponential is for a
relative mode, with rate twelve for this one-parallelogram convention. -/
theorem relativeMode_exponential_initial (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z) (h0 : x 0 - z 0 ≠ 0) (t : ℝ) :
    relativeMode (x t) (y t) (z t) = relativeMode (x 0) (y 0) (z 0) *
      Real.exp (-(12 / squaredNorm (x 0) (y 0) (z 0)) * t) := by
  have hZ0 := squaredNorm_ne_zero_of_ground (x 0) (y 0) (z 0) h0
  have hZ := nonzero_domain_from_initial x y z hdx hdy hdz hflow hZ0
  have hf := full_gradient_flow_from_initial x y z hdx hdy hdz hflow hZ0
  exact relativeMode_exponential_from_initial_ground x y z hZ hf h0 t

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) ∧ (1 : ℝ) - (-1) ≠ 0 := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_, by norm_num⟩
  intro t hZ
  have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
  refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- The ground coordinate cannot carry more than twice the norm energy.
Source specialization: arXiv:2205.10343v2, section 3.2, `Z₀`; this
finite-dimensional bound turns relative-mode decay into loss decay. -/
theorem ground_square_le_two_norm (x y z : ℝ) :
    (x - z) ^ 2 ≤ 2 * squaredNorm x y z := by
  unfold squaredNorm
  nlinarith [sq_nonneg (x + z), sq_nonneg y]

/-- The actual normalized loss is nonnegative, including Lean's total
division at the origin. Source: arXiv:2205.10343v2, section 3.2,
equation `eq:l_eff`; the flow itself is specified only off the origin. -/
theorem normalizedLoss_nonneg (x y z : ℝ) : 0 ≤ normalizedLoss x y z := by
  unfold normalizedLoss numerator squaredNorm
  positivity

/-- Small relative residual controls the actual quotient loss. Source
specialization: arXiv:2205.10343v2, section 3.2. A nonzero ground
coordinate supplies the positive denominator and allows cancellation. -/
theorem normalizedLoss_le_relative_energy (x y z : ℝ) (hA : x - z ≠ 0) :
    normalizedLoss x y z ≤ 2 * relativeMode x y z ^ 2 := by
  have hZ := squaredNorm_ne_zero_of_ground x y z hA
  have hn : 0 ≤ squaredNorm x y z := by unfold squaredNorm; positivity
  have hp := lt_of_le_of_ne hn (Ne.symm hZ)
  have hN : numerator x y z = relativeMode x y z ^ 2 * (x - z) ^ 2 := by
    unfold numerator relativeMode
    field_simp [hA]
  unfold normalizedLoss
  apply (div_le_iff₀ hp).mpr
  rw [hN]
  calc
    relativeMode x y z ^ 2 * (x - z) ^ 2 ≤
        relativeMode x y z ^ 2 * (2 * squaredNorm x y z) :=
      mul_le_mul_of_nonneg_left (ground_square_le_two_norm x y z) (sq_nonneg _)
    _ = 2 * relativeMode x y z ^ 2 * squaredNorm x y z := by ring

example : (1 : ℝ) - 0 ≠ 0 := by norm_num

/-- Exponential upper bound for the corrected model's actual normalized
loss. Source correction: arXiv:2205.10343v2, section 3.2, spectral
relaxation; energy decays at twice the relative-amplitude rate. -/
theorem normalizedLoss_exponential_bound (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z) (h0 : x 0 - z 0 ≠ 0) (t : ℝ) :
    normalizedLoss (x t) (y t) (z t) ≤ 2 * relativeMode (x 0) (y 0) (z 0) ^ 2 *
      Real.exp (-(24 / squaredNorm (x 0) (y 0) (z 0)) * t) := by
  have hA := ground_nonzero_off_origin x y z hdx hdy hdz hflow h0
  have hl := normalizedLoss_le_relative_energy (x t) (y t) (z t) (hA t)
  rw [relativeMode_exponential_initial x y z hdx hdy hdz hflow h0 t, mul_pow] at hl
  have he : Real.exp (-(12 / squaredNorm (x 0) (y 0) (z 0)) * t) ^ 2 =
      Real.exp (-(24 / squaredNorm (x 0) (y 0) (z 0)) * t) := by
    rw [pow_two, ← Real.exp_add]
    congr 1
    ring
  simpa only [he, mul_assoc] using hl

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) ∧ (1 : ℝ) - (-1) ≠ 0 := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_, by norm_num⟩
  intro t hZ
  have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
  refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- Convergence to zero effective loss under initial nonzero ground data.
Source correction: arXiv:2205.10343v2, section 3.2; this is derived
from the actual quotient flow, not the source's frozen linear field. -/
theorem normalizedLoss_tendsto_zero (x y z : ℝ → ℝ)
    (hdx : Differentiable ℝ x) (hdy : Differentiable ℝ y) (hdz : Differentiable ℝ z)
    (hflow : FollowsGradientOffOrigin x y z) (h0 : x 0 - z 0 ≠ 0) :
    Filter.Tendsto (fun t => normalizedLoss (x t) (y t) (z t)) Filter.atTop (nhds 0) := by
  have hZ := squaredNorm_ne_zero_of_ground (x 0) (y 0) (z 0) h0
  have hn : 0 ≤ squaredNorm (x 0) (y 0) (z 0) := by unfold squaredNorm; positivity
  have hp := lt_of_le_of_ne hn (Ne.symm hZ)
  have hr : 0 < 24 / squaredNorm (x 0) (y 0) (z 0) := by positivity
  have ht : Filter.Tendsto (fun t : ℝ => 24 / squaredNorm (x 0) (y 0) (z 0) * t)
      Filter.atTop Filter.atTop :=
    (Filter.tendsto_const_mul_atTop_of_pos hr).mpr Filter.tendsto_id
  have he := Real.tendsto_exp_neg_atTop_nhds_zero.comp ht
  have hu : Filter.Tendsto
      (fun t : ℝ => 2 * relativeMode (x 0) (y 0) (z 0) ^ 2 *
        Real.exp (-(24 / squaredNorm (x 0) (y 0) (z 0)) * t)) Filter.atTop (nhds 0) := by
    simpa only [Function.comp_apply, neg_mul, mul_zero] using
      (tendsto_const_nhds (x := 2 * relativeMode (x 0) (y 0) (z 0) ^ 2)).mul he
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hu
    (fun t => normalizedLoss_nonneg (x t) (y t) (z t))
    (normalizedLoss_exponential_bound x y z hdx hdy hdz hflow h0)

example : Differentiable ℝ (fun _ : ℝ => (1 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (0 : ℝ)) ∧
    Differentiable ℝ (fun _ : ℝ => (-1 : ℝ)) ∧
    FollowsGradientOffOrigin (fun _ => 1) (fun _ => 0) (fun _ => -1) ∧ (1 : ℝ) - (-1) ≠ 0 := by
  refine ⟨differentiable_const 1, differentiable_const 0, differentiable_const (-1), ?_, by norm_num⟩
  intro t hZ
  have hg := ground_gradient_zero 1 0 (-1) hZ (by unfold residual; norm_num)
  refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

end Transformer.Grokking.EffectiveTheory
