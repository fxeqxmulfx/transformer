/-
# Pair-kernel derivatives along arbitrary angular transports

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. The angular
speed of each point is fixed at the start of the transport. The
derivatives of the actual transported exponential pair kernel
therefore carry the speed difference and its square.

The force and curvature are explicit formulas in the actual
Euclidean sphere coordinates. Their equality to the derivatives
is proved below. Continuous bounded speeds give global domination,
which will justify differentiation against an arbitrary measure.
-/

import Transformer.Perceptron.Section5_CircleRotation
import Transformer.Perceptron.Section5_InteractionCurvature
import Transformer.Perceptron.Section5_FiniteVariation

open Real

namespace Transformer.Perceptron

/-- The normalized angular pair force. Source: arXiv:2601.21366v2,
§5.2, the first differentiation of `eq:Hess-bilinear.std`. -/
noncomputable def circlePairForce (β : ℝ) (x y : SSphere 2) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) *
    inner (𝕜 := ℝ) (y : EucSpace 2) (circleTangent (x : EucSpace 2))

/-- The normalized pair curvature, including the radial acceleration.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
noncomputable def circlePairCurvature (β : ℝ) (x y : SSphere 2) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2)) *
    (β * inner (𝕜 := ℝ) (y : EucSpace 2) (circleTangent (x : EucSpace 2)) ^ 2 -
      inner (𝕜 := ℝ) (x : EucSpace 2) (y : EucSpace 2))

/-- The pair force is exactly the chart kernel's derivative, including
its temperature factor. Source: arXiv:2601.21366v2, §5.2. -/
theorem circlePairForce_circlePoint (β θ φ : ℝ) :
    β * circlePairForce β (circlePoint θ) (circlePoint φ) =
      deriv (kernelK β) (θ - φ) := by
  rw [circlePairForce, circleTangent_circlePoint, inner_circlePoint_circlePoint,
    inner_circlePoint_circleVel_sub, deriv_kernelK,
    show φ - θ = -(θ - φ) by ring, Real.sin_neg]
  ring

/-- The pair curvature is the actual second chart derivative divided
by temperature. Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`.
The multiplied form also makes sense at zero temperature. -/
theorem circlePairCurvature_circlePoint (β θ φ : ℝ) :
    β * circlePairCurvature β (circlePoint θ) (circlePoint φ) =
      kernelK2 β (θ - φ) := by
  rw [circlePairCurvature, circleTangent_circlePoint, inner_circlePoint_circlePoint,
    inner_circlePoint_circleVel_sub, show φ - θ = -(θ - φ) by ring,
    Real.sin_neg, neg_sq, kernelK2]
  ring

/-- The normalized curvature is symmetric in its actual sphere points.
Source: arXiv:2601.21366v2, §5.2, the symmetry used in `eq:Hess-bilinear`. -/
theorem circlePairCurvature_swap (β : ℝ) (x y : SSphere 2) :
    circlePairCurvature β x y = circlePairCurvature β y x := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  obtain ⟨φ, -, rfl⟩ := exists_circlePoint_eq y
  simp only [circlePairCurvature, circleTangent_circlePoint,
    inner_circlePoint_circlePoint, inner_circlePoint_circleVel_sub]
  rw [show φ - θ = -(θ - φ) by ring, Real.cos_neg, Real.sin_neg, neg_sq]

/-- The pair force is jointly continuous in its two actual sphere points.
Source: arXiv:2601.21366v2, §5.2, the integrand before `eq:Hess-bilinear.std`. -/
theorem continuous_circlePairForce (β : ℝ) :
    Continuous (fun p : SSphere 2 × SSphere 2 => circlePairForce β p.1 p.2) := by
  have hJ := continuous_circleTangent
  unfold circlePairForce
  fun_prop

/-- The curvature integrand is jointly continuous as well. Source:
arXiv:2601.21366v2, §5.2, the pair kernel in `eq:Hess-bilinear.std`. -/
theorem continuous_circlePairCurvature (β : ℝ) :
    Continuous (fun p : SSphere 2 × SSphere 2 => circlePairCurvature β p.1 p.2) := by
  have hJ := continuous_circleTangent
  unfold circlePairCurvature
  fun_prop

/-- Positive-temperature pair forces have absolute bound `exp β`.
Source: arXiv:2601.21366v2, §5.2, the first kernel differentiation. -/
theorem abs_circlePairForce_le (β : ℝ) (hβ : 0 ≤ β) (x y : SSphere 2) :
    |circlePairForce β x y| ≤ Real.exp β := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  have heq : circlePairForce β (circlePoint θ) y = circleInteractionForce β θ y := by
    rw [circlePairForce, circleInteractionForce, circleTangent_circlePoint,
      real_inner_comm (y : EucSpace 2) (circlePoint θ : EucSpace 2)]
  rw [heq]
  exact abs_circleInteractionForce_le β hβ θ y

/-- The normalized pair curvature has the sharp absolute bound `exp β`.
Source: arXiv:2601.21366v2, §5.2 and §5.5, the bound on `K_β''/β`. -/
theorem abs_circlePairCurvature_le (β : ℝ) (hβ : 0 < β) (x y : SSphere 2) :
    |circlePairCurvature β x y| ≤ Real.exp β := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  have heq : circlePairCurvature β (circlePoint θ) y =
      circleInteractionCurvature β θ y := by
    rw [circlePairCurvature, circleInteractionCurvature, circleTangent_circlePoint,
      real_inner_comm (y : EucSpace 2) (circlePoint θ : EucSpace 2)]
  rw [heq]
  exact abs_circleInteractionCurvature_le β hβ θ y

/-- The actual transported pair kernel has the source's first derivative.
Source: arXiv:2601.21366v2, §5.2, the differentiation preceding
`eq:Hess-bilinear.std`. No regularity of the speed is needed pointwise. -/
theorem hasDerivAt_rotated_pair_kernel (β : ℝ) (s : SSphere 2 → ℝ)
    (x y : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => Real.exp (β * inner (𝕜 := ℝ)
      (circleRotation s r x : EucSpace 2) (circleRotation s r y : EucSpace 2)))
      (β * circlePairForce β (circleRotation s t x) (circleRotation s t y) * (s x - s y))
      t := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  obtain ⟨φ, -, rfl⟩ := exists_circlePoint_eq y
  have hK : Differentiable ℝ (kernelK β) :=
    fun u => (hasDerivAt_kernelK β u).differentiableAt
  convert hasDerivAt_affine_comp (kernelK β) hK (θ - φ)
    (s (circlePoint θ) - s (circlePoint φ)) t using 1
  · funext r
    rw [circleRotation_circlePoint, circleRotation_circlePoint, inner_circlePoint_circlePoint]
    unfold kernelK
    congr 3
    ring
  · rw [circleRotation_circlePoint, circleRotation_circlePoint, circlePairForce_circlePoint]
    congr 2
    ring

/-- The actual first-derivative integrand has the source's second derivative.
Source: arXiv:2601.21366v2, §5.2, the speed difference squared in
`eq:Hess-bilinear.std`. The curvature is derived from the original kernel. -/
theorem hasDerivAt_rotated_pair_force (β : ℝ) (s : SSphere 2 → ℝ)
    (x y : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => β * circlePairForce β
      (circleRotation s r x) (circleRotation s r y) * (s x - s y))
      (β * circlePairCurvature β
        (circleRotation s t x) (circleRotation s t y) * (s x - s y) ^ 2) t := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  obtain ⟨φ, -, rfl⟩ := exists_circlePoint_eq y
  have ha : HasDerivAt (fun r : ℝ => θ - φ +
      r * (s (circlePoint θ) - s (circlePoint φ)))
      (s (circlePoint θ) - s (circlePoint φ)) t := by
    simpa only [id_eq, one_mul] using
      ((hasDerivAt_id t).mul_const (s (circlePoint θ) - s (circlePoint φ))).const_add (θ - φ)
  have h := ((hasDerivAt_deriv_kernelK β _).comp t ha).mul_const
    (s (circlePoint θ) - s (circlePoint φ))
  convert h using 1
  · funext r
    rw [circleRotation_circlePoint, circleRotation_circlePoint, circlePairForce_circlePoint]
    congr 2
    ring
  · rw [circleRotation_circlePoint, circleRotation_circlePoint, circlePairCurvature_circlePoint]
    rw [show θ + t * s (circlePoint θ) - (φ + t * s (circlePoint φ)) =
      θ - φ + t * (s (circlePoint θ) - s (circlePoint φ)) by ring]
    ring

/-- All temperature inputs in the force and curvature bounds are met
at `β=1`, with actual points of the nonempty circle. Arbitrary angular
speeds are permitted in the two pointwise differentiation identities. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    |circlePairCurvature 1 (circlePoint 0) (circlePoint 0)| = Real.exp 1 := by
  have heq : circlePairCurvature 1 (circlePoint 0) (circlePoint 0) = -Real.exp 1 := by
    simp only [circlePairCurvature, circleTangent_circlePoint, inner_circlePoint_circlePoint,
      sub_self, Real.cos_zero, inner_circlePoint_circleVel, mul_one, sq, mul_zero]
    ring
  rw [heq, abs_neg, abs_of_pos (Real.exp_pos 1)]
  exact ⟨zero_le_one, one_pos, rfl⟩

end Transformer.Perceptron
