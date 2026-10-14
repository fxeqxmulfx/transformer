/-
# Actual uniform mixed-moment identity from rotational invariance

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The uniform projection estimate can be obtained from exact
polynomial moments. Two dominated differentiations of the actual
rotated coordinate powers are justified by the global bounds
from `Section4_RotatedPowers` on the original compact sphere.

The actual original uniform law makes each power integral constant
along the original orthogonal orbit. Its proved second derivative
therefore vanishes. At the identity this gives the original
mixed-moment relation `E a^n = (n-1) E a^(n-2)b²` for orthonormal
coordinate vectors and degrees at least two. No moment value or
concentration inequality is supplied as an unproved input.
-/

import Transformer.Metastability.Section4_RotatedPowers
import Transformer.Perceptron.Section5_DominatedVariation

open Real MeasureTheory

namespace Transformer.Metastability

/-- First actual differentiation of the original coordinate-power
integral. Source: arXiv:2410.06833v1, §4, the uniform projection moments
used in `prop: concentration unif`. Finite measures suffice for calculus. -/
theorem hasDerivAt_integral_rotatedPower {d : ℕ} (ν : Measure (SSphere d)) [IsFiniteMeasure ν]
    {u v : EucSpace d} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1)
    (huv : inner (𝕜 := ℝ) u v = 0) (n : ℕ) (t : ℝ) :
    HasDerivAt (fun r => ∫ x : SSphere d, rotatedPower u v n r (x : EucSpace d) ∂ν)
      (∫ x : SSphere d, rotatedPowerVelocity u v n t (x : EucSpace d) ∂ν) t := by
  exact Perceptron.hasDerivAt_integral_on_compact ν
    (fun r x => rotatedPower u v n r (x : EucSpace d))
    (fun r x => rotatedPowerVelocity u v n r (x : EucSpace d))
    (continuous_rotatedPower u v n) (continuous_rotatedPowerVelocity u v n)
    (2 * n) (abs_rotatedPowerVelocity_le hu hv huv n)
    (fun r x => hasDerivAt_rotatedPower u v (x : EucSpace d) n r) t

/-- Two dominated differentiations establish the actual second derivative
of the original moment integral. Source: arXiv:2410.06833v1, §4,
the moment recurrence underlying the original projection concentration. -/
theorem hasDerivAt_deriv_integral_rotatedPower {d : ℕ}
    (ν : Measure (SSphere d)) [IsFiniteMeasure ν] {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (n : ℕ) (hn : 2 ≤ n) (t : ℝ) :
    HasDerivAt (deriv (fun r => ∫ x : SSphere d, rotatedPower u v n r (x : EucSpace d) ∂ν))
      (∫ x : SSphere d, rotatedPowerCurvature u v n t (x : EucSpace d) ∂ν) t := by
  exact Perceptron.hasDerivAt_deriv_integral_on_compact ν
    (fun r x => rotatedPower u v n r (x : EucSpace d))
    (fun r x => rotatedPowerVelocity u v n r (x : EucSpace d))
    (fun r x => rotatedPowerCurvature u v n r (x : EucSpace d))
    (continuous_rotatedPower u v n) (continuous_rotatedPowerVelocity u v n)
    (continuous_rotatedPowerCurvature u v n) (2 * n) (4 * (n : ℝ) ^ 2)
    (abs_rotatedPowerVelocity_le hu hv huv n) (abs_rotatedPowerCurvature_le hu hv huv n hn)
    (fun r x => hasDerivAt_rotatedPower u v (x : EucSpace d) n r)
    (fun r x => hasDerivAt_rotatedPowerVelocity u v (x : EucSpace d) n hn r) t

/-- The computed original curvature integral is the actual iterated
derivative value. Source: arXiv:2410.06833v1, §4, the uniform
projection-moment recurrence used in `prop: concentration unif`. -/
theorem secondDeriv_integral_rotatedPower {d : ℕ}
    (ν : Measure (SSphere d)) [IsFiniteMeasure ν] {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (n : ℕ) (hn : 2 ≤ n) (t : ℝ) :
    deriv^[2] (fun r => ∫ x : SSphere d, rotatedPower u v n r (x : EucSpace d) ∂ν) t =
      ∫ x : SSphere d, rotatedPowerCurvature u v n t (x : EucSpace d) ∂ν :=
  (hasDerivAt_deriv_integral_rotatedPower ν hu hv huv n hn t).deriv

/-- The original uniform law makes the actual rotated power integral
constant. Source: arXiv:2410.06833v1, §4, the rotational invariance
behind `prop: concentration unif` and its projection estimate. -/
theorem integral_rotatedPower_eq {d : ℕ} (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    {u v : EucSpace d} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1)
    (huv : inner (𝕜 := ℝ) u v = 0) (n : ℕ) (t : ℝ) :
    (∫ x : SSphere d, rotatedPower u v n t (x : EucSpace d) ∂ν) =
      ∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ n ∂ν := by
  simp_rw [rotatedPower, rotationCoordinate_eq_inner hu hv huv]
  exact integral_inner_pow_planeRotation_eq ν hν u v t n

/-- Constancy and the proved actual second derivative force the original
curvature integral to vanish. Source: arXiv:2410.06833v1, §4,
the uniform polynomial moment identity used for projection concentration. -/
theorem integral_rotatedPowerCurvature_eq_zero {d : ℕ}
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (n : ℕ) (hn : 2 ≤ n) (t : ℝ) :
    (∫ x : SSphere d, rotatedPowerCurvature u v n t (x : EucSpace d) ∂ν) = 0 := by
  let : IsProbabilityMeasure ν := hν.1
  have heq : (fun r => ∫ x : SSphere d, rotatedPower u v n r (x : EucSpace d) ∂ν) =
      fun _ => ∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ n ∂ν :=
    funext fun r => integral_rotatedPower_eq ν hν hu hv huv n r
  have h := hasDerivAt_deriv_integral_rotatedPower ν hu hv huv n hn t
  rw [heq] at h
  simpa using h.deriv.symm

/-- At the original identity parameter the actual curvature gives the
original coordinate and mixed-coordinate powers, with all degree factors.
Source: arXiv:2410.06833v1, §4, the actual uniform moment recurrence. -/
theorem rotatedPowerCurvature_zero {d : ℕ} (u v z : EucSpace d) (n : ℕ) :
    rotatedPowerCurvature u v n 0 z = 4 * (n : ℝ) *
      (((n - 1 : ℕ) : ℝ) * inner (𝕜 := ℝ) u z ^ (n - 2) * inner (𝕜 := ℝ) v z ^ 2 -
        inner (𝕜 := ℝ) u z ^ n) := by
  simp only [rotatedPowerCurvature, rotationCoordinate_zero, rotationCoordinateVelocity_zero]
  ring

/-- The actual uniform mixed-moment identity retains the original law,
coordinate vectors and power indices.
Source: arXiv:2410.06833v1, §4, the exact polynomial moments used for
the original concentration estimate. It is derived from the actual
orthogonal orbit and actual differentiated integral, with no moment hypothesis. -/
theorem integral_uniform_power_eq_mixed {d : ℕ}
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) {u v : EucSpace d}
    (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (huv : inner (𝕜 := ℝ) u v = 0)
    (n : ℕ) (hn : 2 ≤ n) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ n ∂ν) =
      ((n - 1 : ℕ) : ℝ) * (∫ x : SSphere d, inner (𝕜 := ℝ) u (x : EucSpace d) ^ (n - 2) *
        inner (𝕜 := ℝ) v (x : EucSpace d) ^ 2 ∂ν) := by
  let : IsProbabilityMeasure ν := hν.1
  have hf : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) u (x : EucSpace d) ^ (n - 2) *
      inner (𝕜 := ℝ) v (x : EucSpace d) ^ 2) :=
    ((continuous_const.inner continuous_subtype_val).pow (n - 2)).mul
      ((continuous_const.inner continuous_subtype_val).pow 2)
  have hg : Continuous (fun x : SSphere d => inner (𝕜 := ℝ) u (x : EucSpace d) ^ n) :=
    (continuous_const.inner continuous_subtype_val).pow n
  have h0 := integral_rotatedPowerCurvature_eq_zero ν hν hu hv huv n hn 0
  simp_rw [rotatedPowerCurvature_zero] at h0
  have hfun : (fun x : SSphere d => ((n - 1 : ℕ) : ℝ) * inner (𝕜 := ℝ) u (x : EucSpace d) ^ (n - 2) *
      inner (𝕜 := ℝ) v (x : EucSpace d) ^ 2 - inner (𝕜 := ℝ) u (x : EucSpace d) ^ n) =
      fun x : SSphere d => ((n - 1 : ℕ) : ℝ) * (inner (𝕜 := ℝ) u (x : EucSpace d) ^ (n - 2) *
        inner (𝕜 := ℝ) v (x : EucSpace d) ^ 2) - inner (𝕜 := ℝ) u (x : EucSpace d) ^ n := by
    funext x
    ring
  rw [integral_const_mul, hfun,
    integral_sub ((hf.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)).const_mul
      ((n - 1 : ℕ) : ℝ))
      (hg.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)), integral_const_mul] at h0
  have hnpos : 0 < (n : ℝ) := by exact_mod_cast (by omega : 0 < n)
  have hcoeff : 4 * (n : ℝ) ≠ 0 := by positivity
  have hd := (mul_eq_zero.mp h0).resolve_left hcoeff
  linarith

/-- The actual uniform circle, its original unit axes and degree two
jointly witness every finite-law, uniformity, orthogonality and degree
input of these actual differentiated moment identities. -/
example : IsUniformOn 2 (MeanField.uniformLaw 2) ∧ (2 : ℕ) ≤ 2 ∧
    ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    ‖(EuclideanSpace.single (1 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)
      (EuclideanSpace.single (1 : Fin 2) 1) = 0 :=
  ⟨isUniformOn_uniformLaw (by norm_num), le_rfl, by simp, by simp,
    by simp [EuclideanSpace.inner_single_left]⟩

end Transformer.Metastability
