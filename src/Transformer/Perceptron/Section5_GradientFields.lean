/-
# Smooth gradient directions in actual circle coordinates

ArXiv:2601.21366v2, §2.2, `eq:otto.tangent`, and §5.2,
the localized directions in the proof of `thm: circle.gelu`.
The spherical gradient of a smooth ambient function has an actual
scalar angular speed in the continuous quarter-turn frame.

The frame identity below identifies the projected actual gradient
with that speed times the unit tangent. Smoothness makes the speed
continuous, and linear combinations remain smooth gradient fields.
These are the directions tested by the already proved strict
Hessian inequality; no closure or assumed Hessian formula is used.
-/

import Transformer.Perceptron.Section5_CircleRotation
import Mathlib.Analysis.Calculus.ContDiff.Basic
import Mathlib.Analysis.Calculus.Gradient.Basic

open Real

namespace Transformer.Perceptron

/-- The actual angular speed of a spherical gradient. Source:
arXiv:2601.21366v2, §2.2, `eq:otto.tangent`, and §5.2,
the scalar tangent fields in `eq:Hess-bilinear`. -/
noncomputable def circleGradientSpeed (f : EucSpace 2 → ℝ) (x : SSphere 2) : ℝ :=
  inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2)) (gradient f (x : EucSpace 2))

/-- A constant test potential has zero actual gradient speed.
Source: arXiv:2601.21366v2, §2.2, the smooth gradient subspace
in `eq:otto.tangent`. -/
@[simp] theorem circleGradientSpeed_const (c : ℝ) :
    circleGradientSpeed (fun _ : EucSpace 2 => c) = 0 := by
  funext x
  simp [circleGradientSpeed, gradient]

/-- The projected gradient is exactly its speed times the actual unit
tangent. Source: arXiv:2601.21366v2, §5.2, the scalar representation
of `eq:otto.tangent` in `eq:Hess-bilinear`. -/
theorem proj_gradient_eq_circleGradientSpeed (f : EucSpace 2 → ℝ) (x : SSphere 2) :
    proj 2 (x : EucSpace 2) (gradient f (x : EucSpace 2)) =
      circleGradientSpeed f x • circleTangent (x : EucSpace 2) := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  unfold circleGradientSpeed
  rw [circleTangent_circlePoint, proj]
  have h := circle_frame_decomposition θ (gradient f (circlePoint θ : EucSpace 2))
  calc
    _ = (inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2)
      (gradient f (circlePoint θ : EucSpace 2)) • (circlePoint θ : EucSpace 2) +
      inner (𝕜 := ℝ) (circleVel θ) (gradient f (circlePoint θ : EucSpace 2)) • circleVel θ) -
      inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2)
        (gradient f (circlePoint θ : EucSpace 2)) • (circlePoint θ : EucSpace 2) :=
      congrArg (fun v => v - inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2)
        (gradient f (circlePoint θ : EucSpace 2)) • (circlePoint θ : EucSpace 2)) h
    _ = _ := by abel

/-- A smooth function has a continuous actual ambient gradient.
Source: arXiv:2601.21366v2, §5.2, the smooth test potentials used
to construct the localized tangent directions. -/
theorem continuous_gradient_of_contDiff (f : EucSpace 2 → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) : Continuous (gradient f) := by
  exact (InnerProductSpace.toDual ℝ (EucSpace 2)).symm.continuous.comp
    (hf.continuous_fderiv (by simp))

/-- The angular speed of a smooth spherical gradient is continuous.
Source: arXiv:2601.21366v2, §5.2, the smooth directions in
the proof of `thm: circle.gelu`. -/
theorem continuous_circleGradientSpeed (f : EucSpace 2 → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) : Continuous (circleGradientSpeed f) :=
  (continuous_circleTangent.comp continuous_subtype_val).inner
    ((continuous_gradient_of_contDiff f hf).comp continuous_subtype_val)

/-- These speeds give actual smooth gradient directions of the source.
Source: arXiv:2601.21366v2, §2.2, `eq:otto.tangent`. -/
theorem isGradientField_circleGradientSpeed (f : EucSpace 2 → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) :
    IsGradientField (fun x => circleGradientSpeed f x • circleTangent (x : EucSpace 2)) :=
  ⟨f, hf, fun x => (proj_gradient_eq_circleGradientSpeed f x).symm⟩

/-- The actual squared tangent norm is the speed squared.
Source: arXiv:2601.21366v2, §5.2, the `L²` normalization of
the localized directions in `eq:Hess-bilinear`. -/
theorem norm_sq_circleGradientSpeed (f : EucSpace 2 → ℝ) (x : SSphere 2) :
    ‖proj 2 (x : EucSpace 2) (gradient f (x : EucSpace 2))‖ ^ 2 =
      circleGradientSpeed f x ^ 2 := by
  rw [proj_gradient_eq_circleGradientSpeed, norm_smul, norm_circleTangent,
    mem_sphere_zero_iff_norm.mp x.2, mul_one, Real.norm_eq_abs, sq_abs]

/-- Scalar multiplication of a smooth test potential multiplies its
actual gradient speed. Source: arXiv:2601.21366v2, §5.2,
the linear combinations of localized gradient directions. -/
theorem circleGradientSpeed_const_mul (f : EucSpace 2 → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) (c : ℝ) (x : SSphere 2) :
    circleGradientSpeed (fun z => c * f z) x = c * circleGradientSpeed f x := by
  unfold circleGradientSpeed gradient
  rw [fderiv_const_mul (hf.differentiable (by simp) (x : EucSpace 2)) c]
  simp only [map_smulₛₗ, starRingEnd_apply, star_trivial, real_inner_smul_right]

/-- Subtraction of smooth test potentials subtracts their actual speeds.
Source: arXiv:2601.21366v2, §5.2, the mean-zero combination
of localized directions in the proof of `thm: circle.gelu`. -/
theorem circleGradientSpeed_sub (f g : EucSpace 2 → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) (hg : ContDiff ℝ (⊤ : ℕ∞) g) (x : SSphere 2) :
    circleGradientSpeed (fun z => f z - g z) x =
      circleGradientSpeed f x - circleGradientSpeed g x := by
  unfold circleGradientSpeed gradient
  rw [fderiv_fun_sub (hf.differentiable (by simp) (x : EucSpace 2))
    (hg.differentiable (by simp) (x : EucSpace 2)), map_sub, inner_sub_right]

/-- The speed is uniquely determined by its actual tangent field.
Source: arXiv:2601.21366v2, §5.2, the unit scalar tangent frame. -/
theorem speed_eq_circleGradientSpeed (f : EucSpace 2 → ℝ) (s : SSphere 2 → ℝ)
    (hfield : ∀ x, s x • circleTangent (x : EucSpace 2) =
      proj 2 (x : EucSpace 2) (gradient f (x : EucSpace 2))) :
    s = circleGradientSpeed f := by
  funext x
  have h := congrArg (fun v => inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2)) v)
    ((hfield x).trans (proj_gradient_eq_circleGradientSpeed f x))
  have hnorm : ‖circleTangent (x : EucSpace 2)‖ = 1 :=
    (norm_circleTangent _).trans (mem_sphere_zero_iff_norm.mp x.2)
  simpa only [real_inner_smul_right, real_inner_self_eq_norm_sq, hnorm, one_pow, mul_one] using h

/-- A scalar combination of actual smooth gradient fields remains one.
Source: arXiv:2601.21366v2, §5.2, the construction of `ξ₀`
with zero measure mean in the proof of `thm: circle.gelu`. -/
theorem isGradientField_speed_sub_const_mul (s₁ s₂ : SSphere 2 → ℝ)
    (h₁ : IsGradientField (fun x => s₁ x • circleTangent (x : EucSpace 2)))
    (h₂ : IsGradientField (fun x => s₂ x • circleTangent (x : EucSpace 2))) (c : ℝ) :
    IsGradientField (fun x => (s₁ x - c * s₂ x) • circleTangent (x : EucSpace 2)) := by
  obtain ⟨f, hf, hfieldf⟩ := h₁
  obtain ⟨g, hg, hfieldg⟩ := h₂
  have hsf := speed_eq_circleGradientSpeed f s₁ hfieldf
  have hsg := speed_eq_circleGradientSpeed g s₂ hfieldg
  have hcg : ContDiff ℝ (⊤ : ℕ∞) (fun z => c * g z) := contDiff_const.mul hg
  have hfg := isGradientField_circleGradientSpeed (fun z => f z - c * g z)
    (hf.sub hcg)
  have heq : circleGradientSpeed (fun z => f z - c * g z) = fun x => s₁ x - c * s₂ x := by
    funext x
    rw [circleGradientSpeed_sub f (fun z => c * g z) hf hcg,
      circleGradientSpeed_const_mul g hg c, ← hsf, ← hsg]
  rwa [heq] at hfg

/-- The regularity, gradient-field and field-identification inputs
are jointly realized by the zero smooth potential and its actual
zero speed on the nonempty circle. -/
example : ContDiff ℝ (⊤ : ℕ∞) (fun _ : EucSpace 2 => (0 : ℝ)) ∧
    IsGradientField (fun x => (0 : ℝ) • circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, (0 : ℝ) • circleTangent (x : EucSpace 2) =
      proj 2 (x : EucSpace 2) (gradient (fun _ : EucSpace 2 => (0 : ℝ)) (x : EucSpace 2))) := by
  refine ⟨contDiff_const, ?_, fun x => by simp [gradient, proj]⟩
  simpa only [zero_smul] using (isGradientField_zero (d := 2))

end Transformer.Perceptron
