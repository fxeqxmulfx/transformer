module

public import Transformer.Perceptron.KernelHess
public import Transformer.Perceptron.Section5_InteractionCurvature

/-!
# Uniform curvature bounds at the actual cluster scale

ArXiv:2601.21366v2, §5.4, `eq:dist.cond`, `eq:T-in-bound`,
and `eq:T-out-bound`. The cluster scale for `λ=1/2` is exactly
`1/(2 sqrt β)`. Direct trigonometric inequalities give a stronger
negative bound at this scale than the source's asymptotic argument:
`K_β''/β ≤ -(1/2) exp β` for `β ≥ 1`.

The source's global positive upper bound gives
`K_β''/β ≤ (3/5) exp β` for `β ≥ 4`. These two convenient rational
constants will suffice for the printed cluster-mass constant
`0.5742`, without replacing the kernel or changing the cluster scale.
The statements also transfer the bounds to the actual spherical
interaction curvature used by the angular first variation.
-/

@[expose] public section

open Real

namespace Transformer.Perceptron

/-- **The actual cluster scale controls the squared angular separation.**
Source: arXiv:2601.21366v2, §5.4, `eq:dist.cond` at `λ=1/2`.
This uses the stated scale directly, rather than identifying it
with an asymptotic equivalent involving `θ_c`. -/
theorem cluster_scale_square_le (β : ℝ) (hβ : 1 ≤ β) (θ : ℝ)
    (hθ : |θ| ≤ 1 / (2 * Real.sqrt β)) : β * θ ^ 2 ≤ 1 / 4 := by
  have hβ0 : 0 < β := by linarith
  have hs : 0 < Real.sqrt β := Real.sqrt_pos.mpr hβ0
  have hmul := (le_div_iff₀ (show 0 < 2 * Real.sqrt β by positivity)).mp hθ
  have hhalf : |θ| * Real.sqrt β ≤ 1 / 2 := by linarith
  have hsq := pow_le_pow_left₀ (mul_nonneg (abs_nonneg θ) hs.le) hhalf 2
  rw [mul_pow, sq_abs, Real.sq_sqrt hβ0.le] at hsq
  nlinarith

/-- **A uniform negative curvature bound at the source's actual scale.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-in-bound`.
This strengthens its constant for `λ=1/2`: direct bounds give
`β sin²θ ≤ 1/4`, `cos θ ≥ 7/8`, and
`exp(β cos θ) ≥ (7/8) exp β` whenever `β ≥ 1`. -/
theorem kernelK2_normalized_le_cluster (β : ℝ) (hβ : 1 ≤ β) (θ : ℝ)
    (hθ : |θ| ≤ 1 / (2 * Real.sqrt β)) :
    kernelK2 β θ / β ≤ -(1 / 2 : ℝ) * Real.exp β := by
  have hβ0 : 0 < β := by linarith
  have hsq := cluster_scale_square_le β hβ θ hθ
  have hθsq : θ ^ 2 ≤ 1 / 4 := by nlinarith [sq_nonneg θ]
  have hc := Real.one_sub_sq_div_two_le_cos (x := θ)
  have hs := Real.sin_sq_le_sq (x := θ)
  have hbracket : β * Real.sin θ ^ 2 - Real.cos θ ≤ -(5 / 8 : ℝ) := by
    nlinarith [mul_le_mul_of_nonneg_left hs hβ0.le]
  have harg : β - 1 / 8 ≤ β * Real.cos θ := by
    nlinarith [mul_le_mul_of_nonneg_left hc hβ0.le]
  have he : (7 / 8 : ℝ) * Real.exp β ≤ Real.exp (β * Real.cos θ) := by
    calc (7 / 8 : ℝ) * Real.exp β ≤ Real.exp (-1 / 8) * Real.exp β := by
          nlinarith [mul_le_mul_of_nonneg_right (Real.add_one_le_exp (-1 / 8))
            (Real.exp_pos β).le]
      _ = Real.exp (β - 1 / 8) := by rw [← Real.exp_add]; congr 1; ring
      _ ≤ Real.exp (β * Real.cos θ) := Real.exp_le_exp.mpr harg
  have hformula : kernelK2 β θ / β =
      Real.exp (β * Real.cos θ) * (β * Real.sin θ ^ 2 - Real.cos θ) := by
    unfold kernelK2
    field_simp
  rw [hformula]
  have h1 := mul_le_mul_of_nonneg_left hbracket (Real.exp_pos (β * Real.cos θ)).le
  have h2 := mul_le_mul_of_nonpos_right he (show -(5 / 8 : ℝ) ≤ 0 by norm_num)
  nlinarith [Real.exp_pos β]

/-- **The positive curvature is uniformly at most `(3/5) exp β`.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-out-bound`.
The source gives `2 exp(β-3/2)` after dividing by `β`.
The quadratic exponential lower bound at `3/2` gives the rational
upper bound used here, at every angle and from `β=4` on. -/
theorem kernelK2_normalized_le_global (β : ℝ) (hβ : 4 ≤ β) (θ : ℝ) :
    kernelK2 β θ / β ≤ (3 / 5 : ℝ) * Real.exp β := by
  have hβ0 : 0 < β := by linarith
  have he := Real.quadratic_le_exp_of_nonneg (show (0 : ℝ) ≤ 3 / 2 by norm_num)
  have hfactor : (2 : ℝ) ≤ (3 / 5) * Real.exp (3 / 2) := by linarith
  have hupper : 2 * Real.exp (β - 3 / 2) ≤ (3 / 5 : ℝ) * Real.exp β := by
    rw [Real.exp_sub, ← mul_div_assoc, div_le_iff₀ (Real.exp_pos (3 / 2))]
    nlinarith [mul_le_mul_of_nonneg_left hfactor (Real.exp_pos β).le]
  apply (div_le_iff₀ hβ0).mpr
  have h := kernelK2_le_of_four_le β hβ θ
  nlinarith [mul_le_mul_of_nonneg_left hupper hβ0.le]

/-- **The negative cluster bound is for the actual force derivative.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-in-bound`.
Both circle points are the actual sphere chart points; the second
angle's difference from the first lies in the stated cluster scale. -/
theorem circleInteractionCurvature_le_cluster (β : ℝ) (hβ : 1 ≤ β) (θ φ : ℝ)
    (hsep : |φ - θ| ≤ 1 / (2 * Real.sqrt β)) :
    circleInteractionCurvature β θ (circlePoint φ) ≤ -(1 / 2 : ℝ) * Real.exp β := by
  have hβ0 : 0 < β := by linarith
  rw [circleInteractionCurvature_eq_kernel β hβ0.ne']
  exact kernelK2_normalized_le_cluster β hβ (φ - θ) hsep

/-- **The positive bound applies to every actual interacting sphere point.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-out-bound`.
An actual angular representative exists for every integration point. -/
theorem circleInteractionCurvature_le_global (β : ℝ) (hβ : 4 ≤ β)
    (θ : ℝ) (y : SSphere 2) :
    circleInteractionCurvature β θ y ≤ (3 / 5 : ℝ) * Real.exp β := by
  obtain ⟨φ, -, rfl⟩ := exists_circlePoint_eq y
  have hβ0 : 0 < β := by linarith
  rw [circleInteractionCurvature_eq_kernel β hβ0.ne']
  exact kernelK2_normalized_le_global β hβ (φ - θ)

/-- **Changing an angle by an integer number of turns preserves the point.**
Source: arXiv:2601.21366v2, §5.4, the periodic representatives
in `eq:dist.cond`. This is an identity of actual sphere points. -/
theorem circlePoint_int_mul_period (θ : ℝ) (k : ℤ) :
    circlePoint (θ + 2 * π * (k : ℝ)) = circlePoint θ := by
  have harg : θ + 2 * π * (k : ℝ) = θ + (k : ℝ) * (2 * π) := by ring
  apply Subtype.ext
  ext i
  fin_cases i
  · change (circlePoint (θ + 2 * π * (k : ℝ)) : EucSpace 2) 0 =
      (circlePoint θ : EucSpace 2) 0
    rw [circlePoint_coord_zero, circlePoint_coord_zero, harg, Real.cos_add_int_mul_two_pi]
  · change (circlePoint (θ + 2 * π * (k : ℝ)) : EucSpace 2) 1 =
      (circlePoint θ : EucSpace 2) 1
    rw [circlePoint_coord_one, circlePoint_coord_one, harg, Real.sin_add_int_mul_two_pi]

/-- The actual angular velocity is also unchanged by integer turns.
Source: arXiv:2601.21366v2, §5.4, the tangent speeds attached
to periodic representatives in `eq:dist.cond`. -/
theorem circleVel_int_mul_period (θ : ℝ) (k : ℤ) :
    circleVel (θ + 2 * π * (k : ℝ)) = circleVel θ := by
  have harg : θ + 2 * π * (k : ℝ) = θ + (k : ℝ) * (2 * π) := by ring
  rw [circleVel, circleVel, harg, Real.sin_add_int_mul_two_pi, Real.cos_add_int_mul_two_pi]

/-- The actual kernel curvature is unchanged by integer turns.
Source: arXiv:2601.21366v2, §5.4, using `eq:dist.cond` in
`eq:T-in-bound`. Both the sine square and cosine are preserved. -/
theorem kernelK2_int_mul_period (β θ : ℝ) (k : ℤ) :
    kernelK2 β (θ + 2 * π * (k : ℝ)) = kernelK2 β θ := by
  have harg : θ + 2 * π * (k : ℝ) = θ + (k : ℝ) * (2 * π) := by ring
  rw [kernelK2, kernelK2, harg, Real.sin_add_int_mul_two_pi, Real.cos_add_int_mul_two_pi]

/-- The scale, threshold and separation hypotheses are jointly
realized at `β=4` and two copies of the actual angle zero.
The angular-period identity also includes nonzero integer turns. -/
example : (1 : ℝ) ≤ 4 ∧ (4 : ℝ) ≤ 4 ∧
    |(0 : ℝ)| ≤ 1 / (2 * Real.sqrt 4) ∧
    |(0 : ℝ) - 0| ≤ 1 / (2 * Real.sqrt 4) ∧
    circlePoint (0 + 2 * π * (1 : ℤ)) = circlePoint 0 := by
  refine ⟨by norm_num, le_rfl, by norm_num, by norm_num, ?_⟩
  exact circlePoint_int_mul_period 0 1

end Transformer.Perceptron
