/-
# Actual coordinate derivatives along uniform spherical rotations

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The exact uniform projection moments used in concentration can
be derived by differentiating actual rotational invariance twice.
Here the original coordinate of the original orthogonal orbit
and its actual two derivatives are computed before integration.

The coordinate has absolute value at most one on the original
unit sphere. Its actual velocity is minus twice the second rotated
coordinate and has absolute value at most two. The acceleration
is minus four times the first coordinate. These bounds hold at
every real parameter, including the original identity parameter.
-/

import Transformer.Metastability.Section4_OrthogonalRotations

open Real MeasureTheory

namespace Transformer.Metastability

/-- The original first coordinate of the actual two-reflection orbit.
Source: arXiv:2410.06833v1, §4, the uniform projection-moment estimate.
The equality to the original orthogonal orbit is proved below. -/
noncomputable def rotationCoordinate {d : ℕ} (u v : EucSpace d) (t : ℝ) (z : EucSpace d) : ℝ :=
  Real.cos (2 * t) * inner (𝕜 := ℝ) u z - Real.sin (2 * t) * inner (𝕜 := ℝ) v z

/-- The actual velocity of the original rotated coordinate.
Source: arXiv:2410.06833v1, §4, differentiating the original uniform moments. -/
noncomputable def rotationCoordinateVelocity {d : ℕ}
    (u v : EucSpace d) (t : ℝ) (z : EucSpace d) : ℝ :=
  -2 * Real.sin (2 * t) * inner (𝕜 := ℝ) u z -
    2 * Real.cos (2 * t) * inner (𝕜 := ℝ) v z

/-- The coordinate is exactly the original orthogonal orbit's coordinate.
Source: arXiv:2410.06833v1, §4, the original rotation-invariant law. -/
theorem rotationCoordinate_eq_inner {d : ℕ} {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (t : ℝ) (z : EucSpace d) :
    rotationCoordinate u v t z = inner (𝕜 := ℝ) u (planeRotation u v t z) :=
  (inner_planeRotation_left hu hv huv t z).symm

/-- The actual velocity is minus twice the original second rotated
coordinate. Source: arXiv:2410.06833v1, §4, uniform moment differentiation. -/
theorem rotationCoordinateVelocity_eq_inner {d : ℕ} {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (t : ℝ) (z : EucSpace d) :
    rotationCoordinateVelocity u v t z = -2 * inner (𝕜 := ℝ) v (planeRotation u v t z) := by
  rw [inner_planeRotation_right hu hv huv]
  unfold rotationCoordinateVelocity
  ring

/-- The first actual coordinate derivative along the original orbit.
Source: arXiv:2410.06833v1, §4, the first uniform moment differentiation. -/
theorem hasDerivAt_rotationCoordinate {d : ℕ} (u v z : EucSpace d) (t : ℝ) :
    HasDerivAt (fun r => rotationCoordinate u v r z) (rotationCoordinateVelocity u v t z) t := by
  have ht := (hasDerivAt_id' (x := t)).const_mul 2
  have h := (ht.cos.mul_const (inner (𝕜 := ℝ) u z)).sub
    (ht.sin.mul_const (inner (𝕜 := ℝ) v z))
  apply h.congr_deriv
  simp only [rotationCoordinateVelocity, mul_one]
  ring

/-- The second actual coordinate derivative is the original radial
acceleration factor. Source: arXiv:2410.06833v1, §4,
the second uniform moment differentiation used for concentration. -/
theorem hasDerivAt_rotationCoordinateVelocity {d : ℕ} (u v z : EucSpace d) (t : ℝ) :
    HasDerivAt (fun r => rotationCoordinateVelocity u v r z) (-4 * rotationCoordinate u v t z) t := by
  have ht := (hasDerivAt_id' (x := t)).const_mul 2
  have h := ((ht.sin.const_mul (-2)).mul_const (inner (𝕜 := ℝ) u z)).sub
    ((ht.cos.const_mul 2).mul_const (inner (𝕜 := ℝ) v z))
  apply h.congr_deriv
  simp only [rotationCoordinate, mul_one]
  ring

/-- The original coordinate is jointly continuous in time and point.
Source: arXiv:2410.06833v1, §4, the actual uniform moment integrands. -/
theorem continuous_rotationCoordinate_joint {d : ℕ} (u v : EucSpace d) :
    Continuous (fun p : ℝ × SSphere d => rotationCoordinate u v p.1 (p.2 : EucSpace d)) := by
  have ht : Continuous (fun p : ℝ × SSphere d => 2 * p.1) := continuous_fst.const_mul 2
  have hz : Continuous (fun p : ℝ × SSphere d => (p.2 : EucSpace d)) :=
    continuous_subtype_val.comp continuous_snd
  exact ((Real.continuous_cos.comp ht).mul (continuous_const.inner hz)).sub
    ((Real.continuous_sin.comp ht).mul (continuous_const.inner hz))

/-- The actual velocity is jointly continuous in time and original point.
Source: arXiv:2410.06833v1, §4, domination for moment differentiation. -/
theorem continuous_rotationCoordinateVelocity_joint {d : ℕ} (u v : EucSpace d) :
    Continuous (fun p : ℝ × SSphere d => rotationCoordinateVelocity u v p.1 (p.2 : EucSpace d)) := by
  have ht : Continuous (fun p : ℝ × SSphere d => 2 * p.1) := continuous_fst.const_mul 2
  have hz : Continuous (fun p : ℝ × SSphere d => (p.2 : EucSpace d)) :=
    continuous_subtype_val.comp continuous_snd
  exact (((Real.continuous_sin.comp ht).const_mul (-2)).mul (continuous_const.inner hz)).sub
    (((Real.continuous_cos.comp ht).const_mul 2).mul (continuous_const.inner hz))

/-- Each original fixed-time coordinate integrand is continuous.
Source: arXiv:2410.06833v1, §4, finite uniform moment integrability. -/
theorem continuous_rotationCoordinate {d : ℕ} (u v : EucSpace d) (t : ℝ) :
    Continuous (fun x : SSphere d => rotationCoordinate u v t (x : EucSpace d)) := by
  have hu : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) u (x : EucSpace d)) :=
    continuous_const.inner continuous_subtype_val
  have hv : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) v (x : EucSpace d)) :=
    continuous_const.inner continuous_subtype_val
  exact (hu.const_mul (Real.cos (2 * t))).sub (hv.const_mul (Real.sin (2 * t)))

/-- Each original fixed-time velocity integrand is continuous.
Source: arXiv:2410.06833v1, §4, first-derivative moment domination. -/
theorem continuous_rotationCoordinateVelocity {d : ℕ} (u v : EucSpace d) (t : ℝ) :
    Continuous (fun x : SSphere d => rotationCoordinateVelocity u v t (x : EucSpace d)) := by
  have hu : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) u (x : EucSpace d)) :=
    continuous_const.inner continuous_subtype_val
  have hv : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) v (x : EucSpace d)) :=
    continuous_const.inner continuous_subtype_val
  exact (hu.const_mul (-2 * Real.sin (2 * t))).sub (hv.const_mul (2 * Real.cos (2 * t)))

/-- The original unit coordinate remains bounded by one at every time.
Source: arXiv:2410.06833v1, §4, domination of the original uniform moments. -/
theorem abs_rotationCoordinate_le {d : ℕ} {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (t : ℝ) (x : SSphere d) : |rotationCoordinate u v t (x : EucSpace d)| ≤ 1 := by
  rw [rotationCoordinate_eq_inner hu hv huv]
  have h := abs_real_inner_le_norm u (planeRotation u v t (x : EucSpace d))
  simpa only [hu, LinearIsometryEquiv.norm_map, norm_eq_of_mem_sphere, one_mul] using h

/-- The actual coordinate velocity is bounded by two at every time.
Source: arXiv:2410.06833v1, §4, domination of the actual first moment derivative. -/
theorem abs_rotationCoordinateVelocity_le {d : ℕ} {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (t : ℝ) (x : SSphere d) : |rotationCoordinateVelocity u v t (x : EucSpace d)| ≤ 2 := by
  have h := abs_real_inner_le_norm v (planeRotation u v t (x : EucSpace d))
  have hbound : |inner (𝕜 := ℝ) v (planeRotation u v t (x : EucSpace d))| ≤ 1 := by
    simpa only [hv, LinearIsometryEquiv.norm_map, norm_eq_of_mem_sphere, one_mul] using h
  rw [rotationCoordinateVelocity_eq_inner hu hv huv, abs_mul]
  rw [abs_neg, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
  linarith

/-- At the original identity parameter the coordinate is unchanged.
Source: arXiv:2410.06833v1, §4, the projection-moment recurrence at zero. -/
@[simp] theorem rotationCoordinate_zero {d : ℕ} (u v z : EucSpace d) :
    rotationCoordinate u v 0 z = inner (𝕜 := ℝ) u z := by
  simp [rotationCoordinate]

/-- The actual initial velocity retains the original second coordinate.
Source: arXiv:2410.06833v1, §4, the actual moment derivative at zero. -/
@[simp] theorem rotationCoordinateVelocity_zero {d : ℕ} (u v z : EucSpace d) :
    rotationCoordinateVelocity u v 0 z = -2 * inner (𝕜 := ℝ) v z := by
  simp [rotationCoordinateVelocity]

/-- The original circle coordinate axes jointly realize all nontrivial
unit-vector and orthogonality inputs of this actual coordinate calculus. -/
example : ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    ‖(EuclideanSpace.single (1 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)
      (EuclideanSpace.single (1 : Fin 2) 1) = 0 := by
  exact ⟨by simp, by simp, by simp [EuclideanSpace.inner_single_left]⟩

end Transformer.Metastability
