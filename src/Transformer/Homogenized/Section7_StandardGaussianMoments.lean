module

public import Transformer.Homogenized.Section7_GaussianConditioning

/-!
# Actual standard Gaussian coordinate and vector moments

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The covariance square-root representation
reduces the original integration-by-parts argument to actual standard
Gaussian coordinates. Each has mean zero and variance one; their
actual vector cross moment is the corresponding coordinate vector.

The latter identity proves the actual covariance of a coordinate
with any continuous linear form. Subtracting its own coordinate
component therefore leaves an actual Gaussian residual uncorrelated
with that coordinate. These are the moment inputs to the independent
parameter argument, rather than hypotheses about Gaussian coordinates.
All vector integrals are proved finite by actual Gaussian second
moments and Hölder's inequality before commuting the linear forms.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Every actual standard Gaussian coordinate has an actual Gaussian
law. Source: arXiv:2604.01978v1, §7, the standard-coordinate
integration-by-parts reduction of `lem:gaussian_interp`. -/
theorem hasGaussianLaw_stdGaussian_coordinate {m : ℕ} (i : Fin m) :
    HasGaussianLaw (fun z : EucSpace m => z i) (stdGaussian (EucSpace m)) :=
  (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace m))).map
    (EuclideanSpace.proj (𝕜 := ℝ) i)

/-- The actual standard coordinate mean is zero.
Source: arXiv:2604.01978v1, §7, the centered original Gaussian
coordinate factors used in `lem:gaussian_interp`. -/
theorem integral_stdGaussian_coordinate {m : ℕ} (i : Fin m) :
    (∫ z : EucSpace m, z i ∂stdGaussian (EucSpace m)) = 0 := by
  have h := integral_inner (𝕜 := ℝ) (IsGaussian.integrable_fun_id (μ := stdGaussian (EucSpace m)))
    (EuclideanSpace.single i (1 : ℝ))
  simpa only [EuclideanSpace.inner_single_left, map_one, one_mul, integral_id_stdGaussian,
    inner_zero_right] using h

/-- Actual standard-coordinate second moments are the original
identity covariance entries. Source: arXiv:2604.01978v1, §7,
the genuine Gaussian covariance reduction in `lem:gaussian_interp`. -/
theorem integral_stdGaussian_coordinate_mul {m : ℕ} (i j : Fin m) :
    (∫ z : EucSpace m, z i * z j ∂stdGaussian (EucSpace m)) = if i = j then 1 else 0 := by
  have h := congrArg (fun B => B (EuclideanSpace.single i (1 : ℝ))
    (EuclideanSpace.single j (1 : ℝ))) (covarianceBilin_stdGaussian (E := EucSpace m))
  rw [covarianceBilin_apply (IsGaussian.memLp_two_id (μ := stdGaussian (EucSpace m)))] at h
  simpa only [id_eq, integral_id_stdGaussian, sub_zero, EuclideanSpace.inner_single_left,
    map_one, one_mul, innerSL_apply_apply ℝ, PiLp.single_apply] using h

/-- Each actual standard Gaussian coordinate has variance one.
Source: arXiv:2604.01978v1, §7, the actual scalar variance factor
in the integration-by-parts proof of `lem:gaussian_interp`. -/
theorem variance_stdGaussian_coordinate {m : ℕ} (i : Fin m) :
    variance (fun z : EucSpace m => z i) (stdGaussian (EucSpace m)) = 1 := by
  rw [variance_eq_integral (hasGaussianLaw_stdGaussian_coordinate i).aemeasurable,
    integral_stdGaussian_coordinate i]
  simp only [sub_zero, pow_two, integral_stdGaussian_coordinate_mul, ite_true]

/-- The actual coordinate pushforward is the actual scalar standard
Gaussian law. Source: arXiv:2604.01978v1, §7, the scalar Gaussian
integration-by-parts coordinate used in `lem:gaussian_interp`. -/
theorem stdGaussian_map_coordinate {m : ℕ} (i : Fin m) :
    (stdGaussian (EucSpace m)).map (fun z : EucSpace m => z i) = gaussianReal 0 1 := by
  have h := (hasGaussianLaw_stdGaussian_coordinate i).map_eq_gaussianReal
  rw [integral_stdGaussian_coordinate i, variance_stdGaussian_coordinate i] at h
  simpa only [Real.toNNReal_one] using h

/-- The original coordinate times the actual standard vector is
integrable. Source: arXiv:2604.01978v1, §7, the vector cross moment
used to condition `lem:gaussian_interp` on the original coordinate. -/
theorem integrable_stdGaussian_coordinate_smul {m : ℕ} (i : Fin m) :
    Integrable (fun z : EucSpace m => z i • z) (stdGaussian (EucSpace m)) := by
  have h : MemLp (fun z : EucSpace m => z i • z) 1 (stdGaussian (EucSpace m)) :=
    (hasGaussianLaw_stdGaussian_coordinate i).memLp_two.smul
      (IsGaussian.memLp_two_fun_id (μ := stdGaussian (EucSpace m)))
  exact memLp_one_iff_integrable.mp h

/-- The actual vector cross moment equals the original coordinate
vector. Source: arXiv:2604.01978v1, §7, the covariance identity
behind the original Gaussian integration-by-parts conditioning. -/
theorem integral_stdGaussian_coordinate_smul {m : ℕ} (i : Fin m) :
    (∫ z : EucSpace m, z i • z ∂stdGaussian (EucSpace m)) = EuclideanSpace.single i 1 := by
  ext j
  have h := (EuclideanSpace.proj (𝕜 := ℝ) j).integral_comp_comm
    (integrable_stdGaussian_coordinate_smul i)
  change (∫ z : EucSpace m, z i * z j ∂stdGaussian (EucSpace m)) =
    (∫ z : EucSpace m, z i • z ∂stdGaussian (EucSpace m)) j at h
  rw [← h, integral_stdGaussian_coordinate_mul]
  simp only [PiLp.single_apply, eq_comm]

/-- Every actual continuous linear form has mean zero under the actual
standard Gaussian. Source: arXiv:2604.01978v1, §7, the original
centered Gaussian vectors in `lem:gaussian_interp`. -/
theorem integral_stdGaussian_linearForm {m : ℕ} (L : EucSpace m →L[ℝ] ℝ) :
    (∫ z : EucSpace m, L z ∂stdGaussian (EucSpace m)) = 0 := by
  rw [L.integral_comp_comm (IsGaussian.integrable_fun_id (μ := stdGaussian (EucSpace m))),
    integral_id_stdGaussian, map_zero]

/-- The actual coordinate times any actual linear form is integrable.
Source: arXiv:2604.01978v1, §7, the original finite coordinate
covariance used in the Gaussian integration-by-parts computation. -/
theorem integrable_stdGaussian_coordinate_linearForm {m : ℕ}
    (i : Fin m) (L : EucSpace m →L[ℝ] ℝ) :
    Integrable (fun z : EucSpace m => z i * L z) (stdGaussian (EucSpace m)) :=
  (hasGaussianLaw_stdGaussian_coordinate i).memLp_two.integrable_mul
    ((IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace m))).map L).memLp_two

/-- Pairing an actual coordinate with an actual linear form gives
the form on the corresponding original coordinate vector.
Source: arXiv:2604.01978v1, §7, the actual cross covariance needed
for independent-coordinate integration by parts in `lem:gaussian_interp`. -/
theorem integral_stdGaussian_coordinate_linearForm {m : ℕ}
    (i : Fin m) (L : EucSpace m →L[ℝ] ℝ) :
    (∫ z : EucSpace m, z i * L z ∂stdGaussian (EucSpace m)) = L (EuclideanSpace.single i 1) := by
  have h := L.integral_comp_comm (integrable_stdGaussian_coordinate_smul i)
  rw [integral_stdGaussian_coordinate_smul i] at h
  simpa only [map_smul, smul_eq_mul] using h

/-- The actual standard coordinate and linear-form covariance is
the original coordinate coefficient. Source: arXiv:2604.01978v1,
§7, the uncorrelated Gaussian residual used in `lem:gaussian_interp`. -/
theorem covariance_stdGaussian_coordinate_linearForm {m : ℕ}
    (i : Fin m) (L : EucSpace m →L[ℝ] ℝ) :
    covariance (fun z : EucSpace m => z i) (fun z => L z) (stdGaussian (EucSpace m)) =
      L (EuclideanSpace.single i 1) := by
  rw [covariance, integral_stdGaussian_coordinate i, integral_stdGaussian_linearForm L]
  simp only [sub_zero]
  exact integral_stdGaussian_coordinate_linearForm i L

/-- The original coordinate covariances are the actual Kronecker
coefficients. Source: arXiv:2604.01978v1, §7, the original Gaussian
coordinate conditioning in `lem:gaussian_interp`. -/
theorem covariance_stdGaussian_coordinates {m : ℕ} (i j : Fin m) :
    covariance (fun z : EucSpace m => z i) (fun z => z j) (stdGaussian (EucSpace m)) =
      if i = j then 1 else 0 := by
  rw [covariance, integral_stdGaussian_coordinate i, integral_stdGaussian_coordinate j]
  simp only [sub_zero]
  exact integral_stdGaussian_coordinate_mul i j

/-- The original two-dimensional standard law, a nonzero actual
coordinate and a nonzero actual linear form jointly witness the
coordinate and form inputs; its coordinate variance is genuinely one. -/
example : HasGaussianLaw (fun z : EucSpace 2 => z (0 : Fin 2)) (stdGaussian (EucSpace 2)) ∧
    variance (fun z : EucSpace 2 => z (0 : Fin 2)) (stdGaussian (EucSpace 2)) = 1 ∧
    (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)) (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  exact ⟨hasGaussianLaw_stdGaussian_coordinate 0, variance_stdGaussian_coordinate 0, by rfl⟩

end Transformer.Homogenized
