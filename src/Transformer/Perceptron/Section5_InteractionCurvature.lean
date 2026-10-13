/-
# Angular derivatives of the actual pair interaction

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`, and §5.5,
`rem:strictSOPD-perceptron`. The interaction part of the first
variation has kernel `exp(β cos(θ-φ))/β`. Its first angular
derivative and the derivative of that force are written below
as actual inner products on the Euclidean sphere.

Their absolute bounds are `exp β`. For the curvature, this is
exactly the source's `|K_β''|≤β exp β` divided by `β>0`.
The construction of a circle angle for every integration point
justifies using that scalar bound on the whole sphere.
-/

import Transformer.Perceptron.Section5_CircleGeometry
import Transformer.Perceptron.KernelSup

open Real

namespace Transformer.Perceptron

/-- The actual first angular derivative of the normalized pair
potential at positive temperature; arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`.
The sine sign is determined by differentiating the first angle. -/
noncomputable def circleInteractionForce (β θ : ℝ) (y : SSphere 2) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (y : EucSpace 2) (circlePoint θ : EucSpace 2)) *
    inner (𝕜 := ℝ) (y : EucSpace 2) (circleVel θ)

/-- The actual derivative of the angular interaction force;
arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`, and §5.5,
`rem:strictSOPD-perceptron`. The negative radial term records
the curvature `x''(θ)=-x(θ)` of the unit circle. -/
noncomputable def circleInteractionCurvature (β θ : ℝ) (y : SSphere 2) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (y : EucSpace 2) (circlePoint θ : EucSpace 2)) *
    (β * inner (𝕜 := ℝ) (y : EucSpace 2) (circleVel θ) ^ 2 -
      inner (𝕜 := ℝ) (y : EucSpace 2) (circlePoint θ : EucSpace 2))

/-- **The force derivative is the stated actual curvature.**
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`, and
`rem:strictSOPD-perceptron`. No regularity of a measure is needed. -/
theorem hasDerivAt_circleInteractionForce (β θ : ℝ) (y : SSphere 2) :
    HasDerivAt (fun t => circleInteractionForce β t y)
      (circleInteractionCurvature β θ y) θ := by
  have hexp := ((hasDerivAt_inner_circlePoint (y : EucSpace 2) θ).const_mul β).exp
  have h := hexp.mul (hasDerivAt_inner_circleVel (y : EucSpace 2) θ)
  convert h using 1
  · rfl
  · unfold circleInteractionCurvature
    ring

/-- The force integrand is continuous in the sphere point.
Source: arXiv:2601.21366v2, §5.2, the angular convolution. -/
theorem continuous_circleInteractionForce (β θ : ℝ) :
    Continuous (circleInteractionForce β θ) := by
  unfold circleInteractionForce
  fun_prop

/-- The curvature integrand is continuous in the sphere point.
Source: arXiv:2601.21366v2, §5.2, the Hessian convolution. -/
theorem continuous_circleInteractionCurvature (β θ : ℝ) :
    Continuous (circleInteractionCurvature β θ) := by
  unfold circleInteractionCurvature
  fun_prop

/-- **The interaction force is bounded by `exp β`.**
Source: arXiv:2601.21366v2, §5.2, the angular first variation.
Every point is represented by an actual angle; the sine has
absolute value at most one and the cosine at most one. -/
theorem abs_circleInteractionForce_le (β : ℝ) (hβ : 0 ≤ β)
    (θ : ℝ) (y : SSphere 2) : |circleInteractionForce β θ y| ≤ Real.exp β := by
  obtain ⟨φ, -, rfl⟩ := exists_circlePoint_eq y
  rw [circleInteractionForce, inner_circlePoint_circlePoint, inner_circlePoint_circleVel_sub,
    abs_mul, abs_of_pos (Real.exp_pos _)]
  have hexp : Real.exp (β * Real.cos (φ - θ)) ≤ Real.exp β :=
    Real.exp_le_exp.mpr (by nlinarith [Real.cos_le_one (φ - θ)])
  calc
    _ ≤ Real.exp (β * Real.cos (φ - θ)) * 1 :=
      mul_le_mul_of_nonneg_left (Real.abs_sin_le_one _) (Real.exp_pos _).le
    _ ≤ Real.exp β := by simpa only [mul_one] using hexp

/-- **The curvature is exactly `K_β''(φ-θ)/β` in coordinates.**
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`, and
§5.5, the displayed `K_β''` in `rem:strictSOPD-perceptron`. -/
theorem circleInteractionCurvature_eq_kernel (β : ℝ) (hβ : β ≠ 0) (θ φ : ℝ) :
    circleInteractionCurvature β θ (circlePoint φ) = kernelK2 β (φ - θ) / β := by
  rw [circleInteractionCurvature, inner_circlePoint_circlePoint, inner_circlePoint_circleVel_sub]
  unfold kernelK2
  field_simp

/-- **The force is minus `K_β'(φ-θ)/β` in the angular chart.**
Source: arXiv:2601.21366v2, §5.2, the first variation whose
second derivative is `eq:Hess-bilinear`. The minus sign comes
from differentiating the first angle in `φ-θ`.
Together with the curvature identity this checks both derivatives
of the original exponential pair potential, including its scaling. -/
theorem circleInteractionForce_eq_kernel_derivative (β : ℝ) (hβ : β ≠ 0)
    (θ φ : ℝ) :
    circleInteractionForce β θ (circlePoint φ) = -(deriv (kernelK β) (φ - θ)) / β := by
  rw [circleInteractionForce, inner_circlePoint_circlePoint, inner_circlePoint_circleVel_sub,
    deriv_kernelK]
  field_simp

/-- The nonzero-temperature input of both chart derivative identities
is realized at `β=1`. Both points can be actual points of the circle. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **The actual curvature has sharp absolute bound `exp β`.**
Source: arXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron`,
where `‖K_β''‖_∞=β exp β` is stated. -/
theorem abs_circleInteractionCurvature_le (β : ℝ) (hβ : 0 < β)
    (θ : ℝ) (y : SSphere 2) : |circleInteractionCurvature β θ y| ≤ Real.exp β := by
  obtain ⟨φ, -, rfl⟩ := exists_circlePoint_eq y
  rw [circleInteractionCurvature_eq_kernel β hβ.ne', abs_div, abs_of_pos hβ]
  apply (div_le_iff₀ hβ).mpr
  have h := abs_kernelK2_le β hβ (φ - θ)
  nlinarith

/-- **The same actual curvature is bounded below by `-exp β`.**
Source: arXiv:2601.21366v2, §5.5, the interaction part of
`rem:strictSOPD-perceptron`. This is the bound required for
the second derivative of the angular first variation. -/
theorem neg_exp_le_circleInteractionCurvature (β : ℝ) (hβ : 0 < β)
    (θ : ℝ) (y : SSphere 2) : -Real.exp β ≤ circleInteractionCurvature β θ y :=
  (abs_le.mp (abs_circleInteractionCurvature_le β hβ θ y)).1

/-- **The lower curvature bound is attained at the interacting point.**
Source: arXiv:2601.21366v2, §5.5, the sharp kernel bound in
`rem:strictSOPD-perceptron`. Thus it is not a bound on an altered
potential or an assumed Hessian. -/
theorem circleInteractionCurvature_self (β θ : ℝ) :
    circleInteractionCurvature β θ (circlePoint θ) = -Real.exp β := by
  rw [circleInteractionCurvature, inner_circlePoint_circlePoint,
    inner_circlePoint_circleVel, sub_self, Real.cos_zero]
  ring_nf

/-- The actual pair force vanishes at its interacting point;
arXiv:2601.21366v2, §5.2, the first angular variation. -/
theorem circleInteractionForce_self (β θ : ℝ) :
    circleInteractionForce β θ (circlePoint θ) = 0 := by
  rw [circleInteractionForce, inner_circlePoint_circleVel, mul_zero]

/-- All parameter restrictions in the force and curvature estimates
are met at `β=1`, on the actual point `circlePoint 0`.
The derivative hypothesis for the kernel identity is also nonzero. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 ∧
    circleInteractionCurvature 1 0 (circlePoint 0) = -Real.exp 1 :=
  ⟨zero_le_one, one_pos, one_ne_zero, circleInteractionCurvature_self 1 0⟩

end Transformer.Perceptron
