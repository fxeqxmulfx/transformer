module

public import Transformer.Perceptron.Section5_CircleGeometry
public import Transformer.Perceptron.CircleDeriv
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# A uniform Lipschitz bound for the actual angular potential force

ArXiv:2601.21366v2, §5.4, `eq:T-field-bound`. The source bounds
one half of the potential's second angular derivative by
`C_ϑ=2 Σ_j |ω_j| ‖a_j‖²` when the activation is one-Lipschitz
and vanishes at zero. A Lipschitz activation need not be
differentiable at a support point of an atomic measure.

Here the actual force, one half of the first angular derivative
of the potential, is globally Lipschitz with that same constant.
This gives an increment bound at every pair of angles, including
angles where the activation is not differentiable. It will allow
the stationary force equations at two distinct cluster endpoints
to be compared without differentiating the activation.

The geometric estimates use the actual Euclidean circle chart and
unit angular velocity, and the force identity uses `φ'=2σ`.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Perceptron

/-- The actual chart point has Euclidean norm one.
Source: arXiv:2601.21366v2, §5.4, the unit vectors in
the estimate leading to `eq:T-field-bound`. -/
theorem norm_circlePoint_eq_one (θ : ℝ) : ‖(circlePoint θ : EucSpace 2)‖ = 1 :=
  mem_sphere_zero_iff_norm.mp (circlePoint θ).2

/-- **The actual circle chart has speed at most one.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`.
The global increment estimate follows from its actual unit derivative. -/
theorem norm_circlePoint_sub_le (u v : ℝ) :
    ‖(circlePoint u : EucSpace 2) - (circlePoint v : EucSpace 2)‖ ≤ |u - v| := by
  have h := Convex.norm_image_sub_le_of_norm_deriv_le
    (f := fun r : ℝ => (circlePoint r : EucSpace 2)) (s := Set.univ) (x := v) (y := u) (C := 1)
    (fun r hr => (hasDerivAt_circlePoint r).differentiableAt)
    (fun r hr => by rw [(hasDerivAt_circlePoint r).deriv, norm_circleVel])
    (convex_univ : Convex ℝ (Set.univ : Set ℝ)) (Set.mem_univ v) (Set.mem_univ u)
  simpa only [one_mul, Real.norm_eq_abs] using h

/-- **The actual unit angular velocity has speed at most one.**
Source: arXiv:2601.21366v2, §5.4, `x''(θ)=-x(θ)` in
the estimate leading to `eq:T-field-bound`. -/
theorem norm_circleVel_sub_le (u v : ℝ) : ‖circleVel u - circleVel v‖ ≤ |u - v| := by
  have h := Convex.norm_image_sub_le_of_norm_deriv_le
    (f := circleVel) (s := Set.univ) (x := v) (y := u) (C := 1)
    (fun r hr => (hasDerivAt_circleVel r).differentiableAt)
    (fun r hr => by rw [(hasDerivAt_circleVel r).deriv, norm_neg, norm_circlePoint_eq_one])
    (convex_univ : Convex ℝ (Set.univ : Set ℝ)) (Set.mem_univ v) (Set.mem_univ u)
  simpa only [one_mul, Real.norm_eq_abs] using h

/-- The input projection is uniformly bounded by the neuron's norm.
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`. -/
theorem abs_inner_circlePoint_le (a : EucSpace 2) (θ : ℝ) :
    |inner (𝕜 := ℝ) a (circlePoint θ : EucSpace 2)| ≤ ‖a‖ := by
  simpa only [norm_circlePoint_eq_one, mul_one] using
    abs_real_inner_le_norm a (circlePoint θ : EucSpace 2)

/-- The tangent projection is uniformly bounded by the neuron's norm.
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`. -/
theorem abs_inner_circleVel_le (a : EucSpace 2) (θ : ℝ) :
    |inner (𝕜 := ℝ) a (circleVel θ)| ≤ ‖a‖ := by
  simpa only [norm_circleVel, mul_one] using abs_real_inner_le_norm a (circleVel θ)

/-- The projected chart increment has Lipschitz constant `‖a‖`.
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`. -/
theorem abs_inner_circlePoint_sub_le (a : EucSpace 2) (u v : ℝ) :
    |inner (𝕜 := ℝ) a (circlePoint u : EucSpace 2) -
      inner (𝕜 := ℝ) a (circlePoint v : EucSpace 2)| ≤ ‖a‖ * |u - v| := by
  rw [← inner_sub_right]
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_left (norm_circlePoint_sub_le u v) (norm_nonneg a))

/-- The projected tangent increment has the same Lipschitz constant.
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`. -/
theorem abs_inner_circleVel_sub_le (a : EucSpace 2) (u v : ℝ) :
    |inner (𝕜 := ℝ) a (circleVel u) - inner (𝕜 := ℝ) a (circleVel v)| ≤ ‖a‖ * |u - v| := by
  rw [← inner_sub_right]
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_left (norm_circleVel_sub_le u v) (norm_nonneg a))

/-- **The activation on the chart is bounded by the input norm.**
Source: arXiv:2601.21366v2, §5.4, `σ(0)=0` and the one-Lipschitz
activation in the estimate leading to `eq:T-field-bound`. -/
theorem abs_activation_circlePoint_le (σ : ℝ → ℝ) (hσ : LipschitzWith 1 σ)
    (hzero : σ 0 = 0) (a : EucSpace 2) (θ : ℝ) :
    |σ (inner (𝕜 := ℝ) a (circlePoint θ : EucSpace 2))| ≤ ‖a‖ := by
  have h : |σ (inner (𝕜 := ℝ) a (circlePoint θ : EucSpace 2))| ≤
      |inner (𝕜 := ℝ) a (circlePoint θ : EucSpace 2)| := by
    simpa [hzero, Real.norm_eq_abs] using
      hσ.norm_sub_le (inner (𝕜 := ℝ) a (circlePoint θ : EucSpace 2)) 0
  exact h.trans (abs_inner_circlePoint_le a θ)

/-- **A single neuron's angular force is globally Lipschitz.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`.
The product increment uses only the Lipschitz activation and
the actual chart and tangent increments, not an activation derivative. -/
theorem activation_tangent_product_sub_le (σ : ℝ → ℝ) (hσ : LipschitzWith 1 σ)
    (hzero : σ 0 = 0) (a : EucSpace 2) (u v : ℝ) :
    |σ (inner (𝕜 := ℝ) a (circlePoint u : EucSpace 2)) * inner (𝕜 := ℝ) a (circleVel u) -
      σ (inner (𝕜 := ℝ) a (circlePoint v : EucSpace 2)) * inner (𝕜 := ℝ) a (circleVel v)| ≤
        2 * ‖a‖ ^ 2 * |u - v| := by
  let A := fun r => σ (inner (𝕜 := ℝ) a (circlePoint r : EucSpace 2))
  let B := fun r => inner (𝕜 := ℝ) a (circleVel r)
  have hA : |A u - A v| ≤ ‖a‖ * |u - v| := by
    have h : |A u - A v| ≤ |inner (𝕜 := ℝ) a (circlePoint u : EucSpace 2) -
        inner (𝕜 := ℝ) a (circlePoint v : EucSpace 2)| := by
      simpa [A, Real.norm_eq_abs] using hσ.norm_sub_le
        (inner (𝕜 := ℝ) a (circlePoint u : EucSpace 2))
        (inner (𝕜 := ℝ) a (circlePoint v : EucSpace 2))
    exact h.trans (abs_inner_circlePoint_sub_le a u v)
  change |A u * B u - A v * B v| ≤ _
  rw [show A u * B u - A v * B v = (A u - A v) * B u + A v * (B u - B v) by ring]
  calc
    _ ≤ |A u - A v| * |B u| + |A v| * |B u - B v| := by
      simpa only [abs_mul] using abs_add_le ((A u - A v) * B u) (A v * (B u - B v))
    _ ≤ (‖a‖ * |u - v|) * ‖a‖ + ‖a‖ * (‖a‖ * |u - v|) := by
      gcongr
      · exact abs_inner_circleVel_le a u
      · exact abs_activation_circlePoint_le σ hσ hzero a v
      · exact abs_inner_circleVel_sub_le a u v
    _ = _ := by ring

/-- The actual force is one half of the potential's first angular
derivative; arXiv:2601.21366v2, §5.4, `eq:T-field-bound`. -/
noncomputable def angularPotentialForce (φ : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (θ : ℝ) : ℝ :=
  (1 / 2 : ℝ) * deriv (fun t => potential φ ω a (circlePoint t : EucSpace 2)) θ

/-- The primitive identity gives the source's angular force sum.
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`, with `φ'=2σ`. -/
theorem angularPotentialForce_eq_sum (φ σ : ℝ → ℝ)
    (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    angularPotentialForce φ ω a θ = ∑ j : Idx 2, ω j *
      (σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel θ)) := by
  rw [angularPotentialForce, deriv_potential_circlePoint φ σ hφ]
  ring

/-- **The actual potential force is globally Lipschitz with `C_ϑ`.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-field-bound`.
The bound holds at every pair of angles for a Lipschitz activation,
without requiring its derivative at any atomic support point. -/
theorem abs_angularPotentialForce_sub_le (φ σ : ℝ → ℝ)
    (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (hσ : LipschitzWith 1 σ) (hzero : σ 0 = 0)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (u v : ℝ) :
    |angularPotentialForce φ ω a u - angularPotentialForce φ ω a v| ≤
      (2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2) * |u - v| := by
  rw [angularPotentialForce_eq_sum φ σ hφ ω a u,
    angularPotentialForce_eq_sum φ σ hφ ω a v, ← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ j : Idx 2, |ω j * (σ (inner (𝕜 := ℝ) (a j) (circlePoint u : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel u)) -
      ω j * (σ (inner (𝕜 := ℝ) (a j) (circlePoint v : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel v))| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j : Idx 2, |ω j| * (2 * ‖a j‖ ^ 2 * |u - v|) := by
      apply Finset.sum_le_sum
      intro j hj
      rw [← mul_sub, abs_mul]
      exact mul_le_mul_of_nonneg_left
        (activation_tangent_product_sub_le σ hσ hzero (a j) u v) (abs_nonneg _)
    _ = _ := by
      rw [Finset.mul_sum, Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro j hj
      ring

/-- All primitive, Lipschitz and zero-value hypotheses are jointly
realized by the quadratic primitive and the linear activation. -/
example : (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id r) r) ∧
    LipschitzWith 1 (id : ℝ → ℝ) ∧ id (0 : ℝ) = 0 :=
  ⟨fun r => by simpa using hasDerivAt_pow 2 r, LipschitzWith.id, rfl⟩

end Transformer.Perceptron
