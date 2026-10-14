/-
# Actual differentiated polynomial moments of uniform rotations

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The exact uniform projection moments can be recovered by twice
differentiating the original coordinate powers along actual
orthogonal rotations. This module computes both genuine
derivatives, including the original radial acceleration term.

On the original unit sphere the coordinate and velocity bounds
give global domination constants `2n` and `4n²`, independent of
the rotation parameter. Continuity of the actual integrands on
the original compact sphere supplies the remaining inputs for
two dominated differentiations under its original probability law.
-/

import Transformer.Metastability.Section4_RotationCalculus

open Real MeasureTheory

namespace Transformer.Metastability

/-- The original coordinate power along the original orthogonal orbit.
Source: arXiv:2410.06833v1, §4, the uniform projection-moment estimate. -/
noncomputable def rotatedPower {d : ℕ} (u v : EucSpace d) (n : ℕ) (t : ℝ) (z : EucSpace d) : ℝ :=
  rotationCoordinate u v t z ^ n

/-- The actual first derivative of the original power moment integrand.
Source: arXiv:2410.06833v1, §4, differentiating uniform rotational invariance. -/
noncomputable def rotatedPowerVelocity {d : ℕ}
    (u v : EucSpace d) (n : ℕ) (t : ℝ) (z : EucSpace d) : ℝ :=
  (n : ℝ) * rotationCoordinate u v t z ^ (n - 1) * rotationCoordinateVelocity u v t z

/-- The actual second derivative for degrees at least two, with the
original radial acceleration. Source: arXiv:2410.06833v1, §4,
the uniform moment recurrence used in the projection estimate. -/
noncomputable def rotatedPowerCurvature {d : ℕ}
    (u v : EucSpace d) (n : ℕ) (t : ℝ) (z : EucSpace d) : ℝ :=
  (n : ℝ) * ((n - 1 : ℕ) : ℝ) * rotationCoordinate u v t z ^ (n - 2) *
    rotationCoordinateVelocity u v t z ^ 2 - 4 * (n : ℝ) * rotationCoordinate u v t z ^ n

/-- The first genuine derivative retains the original power and degree.
Source: arXiv:2410.06833v1, §4, the first uniform moment differentiation. -/
theorem hasDerivAt_rotatedPower {d : ℕ} (u v z : EucSpace d) (n : ℕ) (t : ℝ) :
    HasDerivAt (fun r => rotatedPower u v n r z) (rotatedPowerVelocity u v n t z) t :=
  (hasDerivAt_rotationCoordinate u v z t).pow n

/-- A second genuine differentiation gives the original moment curvature.
Source: arXiv:2410.06833v1, §4, the moment recurrence used for concentration. -/
theorem hasDerivAt_rotatedPowerVelocity {d : ℕ} (u v z : EucSpace d)
    (n : ℕ) (hn : 2 ≤ n) (t : ℝ) :
    HasDerivAt (fun r => rotatedPowerVelocity u v n r z) (rotatedPowerCurvature u v n t z) t := by
  have h := (((hasDerivAt_rotationCoordinate u v z t).pow (n - 1)).const_mul (n : ℝ)).mul
    (hasDerivAt_rotationCoordinateVelocity u v z t)
  apply h.congr_deriv
  simp only [rotatedPowerCurvature, Nat.sub_sub, Pi.pow_apply]
  have hp : rotationCoordinate u v t z ^ (n - 1) * rotationCoordinate u v t z =
      rotationCoordinate u v t z ^ n := by
    rw [← pow_succ, Nat.sub_add_cancel (by omega : 1 ≤ n)]
  linear_combination -4 * (n : ℝ) * hp

/-- The original power's actual derivative function has the computed
second derivative. Source: arXiv:2410.06833v1, §4, the original
uniform moment recurrence used for projection concentration. -/
theorem hasDerivAt_deriv_rotatedPower {d : ℕ} (u v z : EucSpace d)
    (n : ℕ) (hn : 2 ≤ n) (t : ℝ) :
    HasDerivAt (deriv (fun r => rotatedPower u v n r z)) (rotatedPowerCurvature u v n t z) t := by
  have heq : deriv (fun r => rotatedPower u v n r z) = fun r => rotatedPowerVelocity u v n r z :=
    funext fun r => (hasDerivAt_rotatedPower u v z n r).deriv
  rw [heq]
  exact hasDerivAt_rotatedPowerVelocity u v z n hn t

/-- The original power integrand is continuous on the original sphere.
Source: arXiv:2410.06833v1, §4, actual uniform moment integrability. -/
theorem continuous_rotatedPower {d : ℕ} (u v : EucSpace d) (n : ℕ) (t : ℝ) :
    Continuous (fun x : SSphere d => rotatedPower u v n t (x : EucSpace d)) :=
  (continuous_rotationCoordinate u v t).pow n

/-- The actual first derivative integrand is continuous on the sphere.
Source: arXiv:2410.06833v1, §4, first-derivative moment domination. -/
theorem continuous_rotatedPowerVelocity {d : ℕ} (u v : EucSpace d) (n : ℕ) (t : ℝ) :
    Continuous (fun x : SSphere d => rotatedPowerVelocity u v n t (x : EucSpace d)) :=
  (((continuous_rotationCoordinate u v t).pow (n - 1)).const_mul (n : ℝ)).mul
    (continuous_rotationCoordinateVelocity u v t)

/-- The actual second derivative integrand is continuous on the sphere.
Source: arXiv:2410.06833v1, §4, second-derivative moment domination. -/
theorem continuous_rotatedPowerCurvature {d : ℕ} (u v : EucSpace d) (n : ℕ) (t : ℝ) :
    Continuous (fun x : SSphere d => rotatedPowerCurvature u v n t (x : EucSpace d)) :=
  (((((continuous_rotationCoordinate u v t).pow (n - 2)).const_mul
    ((n : ℝ) * ((n - 1 : ℕ) : ℝ))).mul ((continuous_rotationCoordinateVelocity u v t).pow 2)).sub
      (((continuous_rotationCoordinate u v t).pow n).const_mul (4 * (n : ℝ))))

/-- Every original coordinate power is bounded by one along the actual
unit-sphere orbit. Source: arXiv:2410.06833v1, §4, uniform moment domination. -/
theorem abs_rotationCoordinate_pow_le {d : ℕ} {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (n : ℕ) (t : ℝ) (x : SSphere d) : |rotationCoordinate u v t (x : EucSpace d) ^ n| ≤ 1 := by
  rw [abs_pow]
  simpa only [one_pow] using pow_le_pow_left₀ (abs_nonneg _)
    (abs_rotationCoordinate_le hu hv huv t x) n

/-- The first actual derivative has the global original domination
constant `2n`. Source: arXiv:2410.06833v1, §4, uniform moment differentiation. -/
theorem abs_rotatedPowerVelocity_le {d : ℕ} {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (n : ℕ) (t : ℝ) (x : SSphere d) : |rotatedPowerVelocity u v n t (x : EucSpace d)| ≤ 2 * n := by
  rw [rotatedPowerVelocity, abs_mul, abs_mul, abs_of_nonneg (Nat.cast_nonneg n)]
  calc
    _ ≤ (n : ℝ) * 1 * 2 := by
      gcongr
      · exact abs_rotationCoordinate_pow_le hu hv huv (n - 1) t x
      · exact abs_rotationCoordinateVelocity_le hu hv huv t x
    _ = _ := by ring

/-- The second actual derivative has the global original domination
constant `4n²`. Source: arXiv:2410.06833v1, §4, uniform moment differentiation. -/
theorem abs_rotatedPowerCurvature_le {d : ℕ} {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (n : ℕ) (hn : 2 ≤ n) (t : ℝ) (x : SSphere d) :
    |rotatedPowerCurvature u v n t (x : EucSpace d)| ≤ 4 * (n : ℝ) ^ 2 := by
  have hV : rotationCoordinateVelocity u v t (x : EucSpace d) ^ 2 ≤ 4 := by
    have h := abs_rotationCoordinateVelocity_le hu hv huv t x
    nlinarith [sq_abs (rotationCoordinateVelocity u v t (x : EucSpace d)), abs_nonneg
      (rotationCoordinateVelocity u v t (x : EucSpace d))]
  have hP₁ := abs_rotationCoordinate_pow_le hu hv huv (n - 2) t x
  have hP₂ := abs_rotationCoordinate_pow_le hu hv huv n t x
  calc
    _ ≤ |(n : ℝ) * ((n - 1 : ℕ) : ℝ) * rotationCoordinate u v t (x : EucSpace d) ^ (n - 2) *
        rotationCoordinateVelocity u v t (x : EucSpace d) ^ 2| +
        |4 * (n : ℝ) * rotationCoordinate u v t (x : EucSpace d) ^ n| := by
      simpa only [rotatedPowerCurvature, sub_zero, zero_sub, abs_neg] using abs_sub_le
        ((n : ℝ) * ((n - 1 : ℕ) : ℝ) * rotationCoordinate u v t (x : EucSpace d) ^ (n - 2) *
          rotationCoordinateVelocity u v t (x : EucSpace d) ^ 2) 0
        (4 * (n : ℝ) * rotationCoordinate u v t (x : EucSpace d) ^ n)
    _ ≤ (n : ℝ) * ((n - 1 : ℕ) : ℝ) * 1 * 4 + 4 * (n : ℝ) * 1 := by
      simp only [abs_mul, abs_of_nonneg (Nat.cast_nonneg n : (0 : ℝ) ≤ n),
        abs_of_nonneg (Nat.cast_nonneg (n - 1) : (0 : ℝ) ≤ ((n - 1 : ℕ) : ℝ)),
        abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 4),
        abs_of_nonneg (sq_nonneg (rotationCoordinateVelocity u v t (x : EucSpace d)))]
      gcongr
    _ = _ := by rw [Nat.cast_sub (by omega : 1 ≤ n)]; push_cast; ring

/-- Degree two and the actual circle axes jointly witness all degree,
unit-vector and orthogonality inputs of these derivative bounds. -/
example : (2 : ℕ) ≤ 2 ∧ ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    ‖(EuclideanSpace.single (1 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)
      (EuclideanSpace.single (1 : Fin 2) 1) = 0 :=
  ⟨le_rfl, by simp, by simp, by simp [EuclideanSpace.inner_single_left]⟩

end Transformer.Metastability
