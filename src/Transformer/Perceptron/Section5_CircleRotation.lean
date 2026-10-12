/-
# Continuous angular transport on the actual circle

ArXiv:2601.21366v2, §2.2, the exponential-map geodesic, and §5.2,
`eq:Hess-bilinear.std`. The source writes a tangent field as a scalar
angular speed. A quarter-turn of the actual Euclidean coordinates
provides its unit tangent without choosing a discontinuous angle.

The rotation below is an actual sphere-valued map. For a continuous
speed it is measurable, its pushforward is a probability measure,
and the resulting curve is precisely the Monge geodesic from §2.2.
These facts hold for arbitrary probability measures on the circle;
no finite support or density is assumed.
-/

import Transformer.Perceptron.Section5_CircleGeodesics

open Real MeasureTheory

namespace Transformer.Perceptron

/-- The positively oriented quarter-turn `(-x₁,x₀)`. Source:
arXiv:2601.21366v2, §5.2, the unit angular tangent `x'(θ)`. -/
noncomputable def circleTangent (x : EucSpace 2) : EucSpace 2 :=
  -x 1 • ((basePoint 1 : SSphere 2) : EucSpace 2) + x 0 • secondAxis

/-- The two coordinates of the actual quarter-turn; arXiv:2601.21366v2,
§5.2, `x'(θ)=(-sin θ,cos θ)`. -/
@[simp] theorem circleTangent_coord_zero (x : EucSpace 2) :
    circleTangent x 0 = -x 1 := by
  simp [circleTangent, basePoint, secondAxis]

/-- The second coordinate of the same quarter-turn; source §5.2. -/
@[simp] theorem circleTangent_coord_one (x : EucSpace 2) :
    circleTangent x 1 = x 0 := by
  simp [circleTangent, basePoint, secondAxis]

/-- The quarter-turn gives exactly the source's angular tangent.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
@[simp] theorem circleTangent_circlePoint (θ : ℝ) :
    circleTangent (circlePoint θ : EucSpace 2) = circleVel θ := by
  ext i
  fin_cases i
  · change circleTangent (circlePoint θ : EucSpace 2) 0 = circleVel θ 0
    rw [circleTangent_coord_zero, circleVel_coord_zero, circlePoint_coord_one]
  · change circleTangent (circlePoint θ : EucSpace 2) 1 = circleVel θ 1
    rw [circleTangent_coord_one, circleVel_coord_one, circlePoint_coord_zero]

/-- The quarter-turn is continuous in the actual sphere coordinates.
Source: arXiv:2601.21366v2, §5.2, the scalar tangent field. -/
theorem continuous_circleTangent : Continuous circleTangent := by
  unfold circleTangent
  fun_prop

/-- A quarter-turn preserves the Euclidean norm. Source:
arXiv:2601.21366v2, §5.2, the identification of tangent speeds. -/
theorem norm_circleTangent (x : EucSpace 2) : ‖circleTangent x‖ = ‖x‖ := by
  have hs : ‖circleTangent x‖ ^ 2 = ‖x‖ ^ 2 := by
    simp only [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two,
      circleTangent_coord_zero, circleTangent_coord_one, neg_sq]
    ring
  nlinarith [norm_nonneg (circleTangent x), norm_nonneg x]

/-- The quarter-turn is tangent to the circle at every unit vector.
Source: arXiv:2601.21366v2, §2.2, `eq:otto.tangent`. -/
theorem inner_circleTangent (x : EucSpace 2) :
    inner (𝕜 := ℝ) x (circleTangent x) = 0 := by
  rw [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_two]
  change circleTangent x 0 * x 0 + circleTangent x 1 * x 1 = 0
  rw [circleTangent_coord_zero, circleTangent_coord_one]
  ring

/-- The rotation formula has unit norm. Source: arXiv:2601.21366v2,
§2.2, the exponential-map footnote. Its norm is proved before
using the formula as a map into the actual sphere. -/
theorem norm_circle_rotation (x : SSphere 2) (r : ℝ) :
    ‖Real.cos r • (x : EucSpace 2) + Real.sin r • circleTangent (x : EucSpace 2)‖ = 1 := by
  have hx : ‖(x : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have ht : ‖circleTangent (x : EucSpace 2)‖ = 1 := (norm_circleTangent _).trans hx
  have hs : ‖Real.cos r • (x : EucSpace 2) +
      Real.sin r • circleTangent (x : EucSpace 2)‖ ^ 2 = 1 := by
    rw [norm_add_sq_real, real_inner_smul_left, real_inner_smul_right,
      inner_circleTangent, mul_zero, mul_zero, norm_smul, norm_smul,
      hx, ht, mul_one, mul_one, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, sq_abs]
    nlinarith [Real.sin_sq_add_cos_sq r]
  nlinarith [norm_nonneg (Real.cos r • (x : EucSpace 2) +
    Real.sin r • circleTangent (x : EucSpace 2))]

/-- The actual transport with angular speed `s`. Source:
arXiv:2601.21366v2, §2.2, `T(t)(x)=exp_x(t ξ(x))`. -/
noncomputable def circleRotation (s : SSphere 2 → ℝ) (t : ℝ) (x : SSphere 2) :
    SSphere 2 :=
  ⟨Real.cos (t * s x) • (x : EucSpace 2) +
    Real.sin (t * s x) • circleTangent (x : EucSpace 2),
    mem_sphere_zero_iff_norm.mpr (norm_circle_rotation x (t * s x))⟩

/-- Transport starts at the actual point. Source: arXiv:2601.21366v2,
§2.2, the geodesic issued from the original measure. -/
@[simp] theorem circleRotation_zero (s : SSphere 2 → ℝ) (x : SSphere 2) :
    circleRotation s 0 x = x := by
  apply Subtype.ext
  simp [circleRotation]

/-- The rotation is the exact angular shift in every representative.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`. -/
theorem circleRotation_circlePoint (s : SSphere 2 → ℝ) (t θ : ℝ) :
    circleRotation s t (circlePoint θ) = circlePoint (θ + t * s (circlePoint θ)) := by
  apply Subtype.ext
  change Real.cos (t * s (circlePoint θ)) • (circlePoint θ : EucSpace 2) +
    Real.sin (t * s (circlePoint θ)) • circleTangent (circlePoint θ : EucSpace 2) = _
  rw [circleTangent_circlePoint, circlePoint_add_frame]

/-- The rotation agrees with the actual spherical exponential.
Source: arXiv:2601.21366v2, §2.2, the Monge geodesic. -/
theorem circleRotation_eq_sphereExp (s : SSphere 2 → ℝ) (t : ℝ) (x : SSphere 2) :
    (circleRotation s t x : EucSpace 2) =
      sphereExp (x : EucSpace 2) (t • (s x • circleTangent (x : EucSpace 2))) := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  rw [smul_smul, circleTangent_circlePoint, sphereExp_circleVel,
    circleRotation_circlePoint]

/-- Continuous angular speeds yield continuous sphere transport.
Source: arXiv:2601.21366v2, §2.2, the measurable Monge map. -/
theorem continuous_circleRotation (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    Continuous (circleRotation s t) := by
  apply Continuous.subtype_mk
  exact ((Real.continuous_cos.comp (hs.const_mul t)).smul continuous_subtype_val).add
    ((Real.continuous_sin.comp (hs.const_mul t)).smul
      (continuous_circleTangent.comp continuous_subtype_val))

/-- The actual pushforward curve realizes the source's geodesic.
Source: arXiv:2601.21366v2, §2.2, the curve used to define the
Hessian in `eq:strict2order`. This constructs the curve for an
arbitrary measure, including a measure with infinite support. -/
theorem isGeodesicFrom_circleRotation (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    IsGeodesicFrom (fun x => s x • circleTangent (x : EucSpace 2)) μ
      (fun t => μ.map (circleRotation s t)) := by
  intro t
  exact ⟨circleRotation s t, (continuous_circleRotation s hs t).measurable,
    circleRotation_eq_sphereExp s t, rfl⟩

/-- All continuity, tangency, norm and geodesic inputs are jointly
realized by constant unit angular speed on the actual circle.
No gradient-field assertion is needed for these transport facts. -/
example : Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) ∧
    (∀ x : SSphere 2, ‖circleTangent (x : EucSpace 2)‖ = 1) ∧
    IsGeodesicFrom (fun x => (1 : ℝ) • circleTangent (x : EucSpace 2))
      (Perspective.diracProb 2 (basePoint 1))
      (fun t => (Perspective.diracProb 2 (basePoint 1)).map
        (circleRotation (fun _ => 1) t)) := by
  refine ⟨continuous_const, fun x => inner_circleTangent (x : EucSpace 2), ?_, ?_⟩
  · intro x
    exact (norm_circleTangent _).trans (mem_sphere_zero_iff_norm.mp x.2)
  · exact isGeodesicFrom_circleRotation _ _ continuous_const

end Transformer.Perceptron
