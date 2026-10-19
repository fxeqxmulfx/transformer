module

public import Transformer.Perceptron.Section5_IsometryLaws

/-!
# Original neuron reflection geometry on the circle

ArXiv:2601.21366v2, §5.2, `thm: circle`, and §6,
the symmetry mechanism of `rem:unifiedlog-B`.
An isometry negating the first neuron and fixing the second reverses
exactly the first original input sign. Its two eigendirections are
orthogonal. At a point with nonzero input in the fixed direction,
the projection of the nonzero negated neuron cannot vanish.

An actual infinite compact angular pattern contains a point where
both original nonzero neurons have nonzero input: their two finite
boundary sets cannot cover it. The explicit coordinate reflection
below supplies joint original witnesses without assuming a reflection
or a strict-pattern point as part of any neuron definition.
-/

@[expose] public section

open Real Set

namespace Transformer.Perceptron

/-- Actual isometric negation and fixation reverse exactly the first
original neuron input. Source: arXiv:2601.21366v2, §5.2 and §6,
the neuron-pattern comparison in `rem:unifiedlog-B`. -/
theorem inner_neuron_reflection_inputs {d : ℕ} (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (u v : EucSpace d) (hu : U u = -u) (hv : U v = v) (x : EucSpace d) :
    inner (𝕜 := ℝ) u (U x) = -inner (𝕜 := ℝ) u x ∧
      inner (𝕜 := ℝ) v (U x) = inner (𝕜 := ℝ) v x := by
  have h₀ := U.inner_map_map u x
  have h₁ := U.inner_map_map v x
  rw [hu, inner_neg_left] at h₀
  rw [hv] at h₁
  exact ⟨by linarith, h₁⟩

/-- The negated and fixed original neuron directions are orthogonal.
Source: arXiv:2601.21366v2, §6, the isometric reflection case of
the symmetry argument in `rem:unifiedlog-B`. -/
theorem inner_eq_zero_of_isometry_neg_fixed {d : ℕ} (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (u v : EucSpace d) (hu : U u = -u) (hv : U v = v) : inner (𝕜 := ℝ) u v = 0 := by
  have h := (inner_neuron_reflection_inputs U u v hu hv v).1
  rw [hv] at h
  linarith

/-- A nonzero original neuron has nonzero projected direction when
an orthogonal test neuron has nonzero input at the original point.
Source: arXiv:2601.21366v2, §2, the projector in `eq:WGF-main`,
used in §5.2, `thm: circle`. The displayed algebra holds in every dimension. -/
theorem proj_ne_zero_of_perpendicular_test {d : ℕ} (u v x : EucSpace d) (hu : u ≠ 0)
    (huv : inner (𝕜 := ℝ) u v = 0) (hvx : inner (𝕜 := ℝ) v x ≠ 0) : proj d x u ≠ 0 := by
  intro hz
  change u - inner (𝕜 := ℝ) x u • x = 0 at hz
  have heq : u = inner (𝕜 := ℝ) x u • x := sub_eq_zero.mp hz
  have hvu : inner (𝕜 := ℝ) v u = 0 := (real_inner_comm u v).trans huv
  have h := congrArg (fun z => inner (𝕜 := ℝ) v z) heq
  rw [hvu, real_inner_smul_right] at h
  have hcoef := (mul_eq_zero.mp h.symm).resolve_right hvx
  exact hu (by rw [heq, hcoef, zero_smul])

/-- Two nonzero original neurons have a common strict-input point in
any infinite original compact angular set.
Source: arXiv:2601.21366v2, §5.2, the finite boundary set `𝒵`
in the proof of `thm: circle`. No original angle is replaced. -/
theorem exists_two_inner_ne_zero_in_infinite_angles (u v : EucSpace 2)
    (hu : u ≠ 0) (hv : v ≠ 0) (S : Set ℝ) (hS : S.Infinite) (hbound : S ⊆ Icc (-π) π) :
    ∃ θ ∈ S, inner (𝕜 := ℝ) u (circlePoint θ : EucSpace 2) ≠ 0 ∧
      inner (𝕜 := ℝ) v (circlePoint θ : EucSpace 2) ≠ 0 := by
  let A := {θ : ℝ | θ ∈ Icc (-π) π ∧ inner (𝕜 := ℝ) u (circlePoint θ : EucSpace 2) = 0}
  let B := {θ : ℝ | θ ∈ Icc (-π) π ∧ inner (𝕜 := ℝ) v (circlePoint θ : EucSpace 2) = 0}
  have hA : A.Finite := finite_inner_zero_angles u hu
  have hB : B.Finite := finite_inner_zero_angles v hv
  obtain ⟨θ, hθ, hout⟩ := (hS.sdiff (hA.union hB)).nonempty
  refine ⟨θ, hθ, ?_, ?_⟩
  · intro hz
    exact hout (Or.inl ⟨hbound hθ, hz⟩)
  · intro hz
    exact hout (Or.inr ⟨hbound hθ, hz⟩)

/-- The actual plane reflection in the second coordinate axis.
Source: arXiv:2601.21366v2, §6, the isometric sign-change
mechanism of `rem:unifiedlog-B`. Its actual eigenvectors are proved below. -/
noncomputable def circleCoordinateReflection : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2 :=
  (ℝ ∙ secondAxis).reflection

/-- The first original coordinate neuron is negated by this reflection.
Source: arXiv:2601.21366v2, §6, the original sign-change step. -/
theorem circleCoordinateReflection_first :
    circleCoordinateReflection (basePoint 1 : EucSpace 2) = -(basePoint 1 : EucSpace 2) := by
  apply Submodule.reflection_mem_subspace_orthogonalComplement_eq_neg
  exact Submodule.mem_orthogonal_singleton_iff_inner_left.mpr inner_basePoint_secondAxis

/-- The second original coordinate neuron is fixed by this reflection.
Source: arXiv:2601.21366v2, §6, the original sign-change step. -/
theorem circleCoordinateReflection_second : circleCoordinateReflection secondAxis = secondAxis :=
  Submodule.reflection_mem_subspace_eq_self (Submodule.mem_span_singleton_self secondAxis)

/-- Its actual action at any original plane point negates the first
coordinate and retains the second. Source: arXiv:2601.21366v2,
§6, the isometric neuron sign-change argument. -/
theorem circleCoordinateReflection_apply (x : EucSpace 2) :
    circleCoordinateReflection x = (-x 0) • (basePoint 1 : EucSpace 2) + x 1 • secondAxis := by
  have hx : x = x 0 • (basePoint 1 : EucSpace 2) + x 1 • secondAxis := by
    ext j
    fin_cases j <;> simp [basePoint, secondAxis]
  calc
    _ = circleCoordinateReflection (x 0 • (basePoint 1 : EucSpace 2) + x 1 • secondAxis) := congrArg _ hx
    _ = _ := by
      rw [map_add, map_smul, map_smul, circleCoordinateReflection_first, circleCoordinateReflection_second,
        smul_neg, neg_smul]

/-- The first actual reflected coordinate has the opposite sign.
Source: arXiv:2601.21366v2, §6, the isometric sign-change step. -/
@[simp] theorem circleCoordinateReflection_coord_zero (x : EucSpace 2) :
    circleCoordinateReflection x 0 = -x 0 := by
  rw [circleCoordinateReflection_apply]
  simp [basePoint, secondAxis]

/-- The second actual reflected coordinate is unchanged.
Source: arXiv:2601.21366v2, §6, the isometric sign-change step. -/
@[simp] theorem circleCoordinateReflection_coord_one (x : EucSpace 2) :
    circleCoordinateReflection x 1 = x 1 := by
  rw [circleCoordinateReflection_apply]
  simp [basePoint, secondAxis]

/-- Reflecting an actual point twice returns that original point.
Source: arXiv:2601.21366v2, §6, the two reflected support components. -/
theorem circleCoordinateReflection_involutive : Function.Involutive circleCoordinateReflection := by
  intro x
  ext j
  fin_cases j <;> simp

/-- All geometric input hypotheses hold jointly for the actual
coordinate reflection, actual nonzero coordinate neurons, a strict
input point and the full original compact infinite angle interval. -/
example : circleCoordinateReflection (basePoint 1 : EucSpace 2) = -(basePoint 1 : EucSpace 2) ∧
    circleCoordinateReflection secondAxis = secondAxis ∧ (basePoint 1 : EucSpace 2) ≠ 0 ∧
    secondAxis ≠ 0 ∧ inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) secondAxis = 0 ∧
    inner (𝕜 := ℝ) secondAxis ((basePoint 1 : EucSpace 2) + secondAxis) ≠ 0 ∧
    (Icc (-π) π).Infinite ∧ (Icc (-π) π ⊆ Icc (-π) π) := by
  refine ⟨circleCoordinateReflection_first, circleCoordinateReflection_second, ?_, ?_,
    inner_basePoint_secondAxis, ?_, Icc_infinite (by linarith [Real.pi_pos]), subset_rfl⟩
  · intro hz
    have h := norm_basePoint_one
    rw [hz, norm_zero] at h
    exact zero_ne_one h
  · intro hz
    have h := norm_secondAxis
    rw [hz, norm_zero] at h
    exact zero_ne_one h
  · simp [basePoint, secondAxis, inner_add_right, EuclideanSpace.inner_single_left]

end Transformer.Perceptron
