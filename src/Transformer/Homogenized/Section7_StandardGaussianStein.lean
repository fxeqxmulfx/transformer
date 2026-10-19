module

public import Transformer.Homogenized.Section7_GaussianResidual

/-!
# Integration by parts for the actual standard Gaussian vector

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. Fixing the actual orthogonal residual
of a standard Gaussian coordinate turns the original vector test
into a scalar test on the original coordinate line.

The derivative of that scalar test is the actual Fréchet derivative
of the original function evaluated on the original coordinate axis.
The coordinate and residual independence already proved from their
joint Gaussian law permits the scalar Stein identity with this
independent parameter. Their actual reconstruction then gives the
vector identity without an assumed integration-by-parts formula.
Bounded first derivatives justify each original expectation.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The original test restricted to an actual affine coordinate line.
Source: arXiv:2604.01978v1, §7, coordinate conditioning in
the integration-by-parts proof of `lem:gaussian_interp`. -/
noncomputable def gaussianCoordinateSlice {m : ℕ} (g : EucSpace m → ℝ)
    (i : Fin m) (p : ℝ × EucSpace m) : ℝ :=
  g (p.1 • EuclideanSpace.single i 1 + p.2)

/-- The actual derivative on that same original coordinate line.
Source: arXiv:2604.01978v1, §7, the original Gaussian
coordinate derivative in `lem:gaussian_interp`. -/
noncomputable def gaussianCoordinateSliceDerivative {m : ℕ} (g : EucSpace m → ℝ)
    (i : Fin m) (p : ℝ × EucSpace m) : ℝ :=
  fderiv ℝ g (p.1 • EuclideanSpace.single i 1 + p.2) (EuclideanSpace.single i 1)

/-- The actual scalar slice is jointly continuous in its original
coordinate and residual. Source: arXiv:2604.01978v1, §7,
the Fubini input for `lem:gaussian_interp`. -/
theorem continuous_gaussianCoordinateSlice {m : ℕ} (g : EucSpace m → ℝ)
    (hg : ContDiff ℝ 1 g) (i : Fin m) : Continuous (gaussianCoordinateSlice g i) := by
  apply hg.continuous.comp
  fun_prop

/-- The actual coordinate derivative is also jointly continuous.
Source: arXiv:2604.01978v1, §7, the original derivative
integrated after coordinate conditioning in `lem:gaussian_interp`. -/
theorem continuous_gaussianCoordinateSliceDerivative {m : ℕ} (g : EucSpace m → ℝ)
    (hg : ContDiff ℝ 1 g) (i : Fin m) :
    Continuous (gaussianCoordinateSliceDerivative g i) := by
  have haff : Continuous (fun p : ℝ × EucSpace m =>
      p.1 • EuclideanSpace.single i 1 + p.2) := by fun_prop
  exact ((hg.continuous_fderiv (by decide)).comp haff).clm_apply continuous_const

/-- The scalar derivative is the original Fréchet derivative on the
actual axis, by the chain rule along the actual affine line.
Source: arXiv:2604.01978v1, §7, the coordinate integration
by parts used in `lem:gaussian_interp`. -/
theorem gaussianCoordinateSlice_hasDerivAt {m : ℕ} (g : EucSpace m → ℝ)
    (hg : ContDiff ℝ 1 g) (i : Fin m) (y : EucSpace m) (r : ℝ) :
    HasDerivAt (fun q => gaussianCoordinateSlice g i (q, y))
      (gaussianCoordinateSliceDerivative g i (r, y)) r := by
  have haff : HasDerivAt (fun q : ℝ => q • EuclideanSpace.single i 1 + y)
      (EuclideanSpace.single i 1) r := by
    simpa only [id_eq, one_smul] using
      ((hasDerivAt_id r).smul_const (EuclideanSpace.single i (1 : ℝ))).add_const y
  exact ((hg.differentiable (by decide)) _).hasFDerivAt.comp_hasDerivAt r haff

/-- The original derivative bound controls the actual scalar slice
derivative because its original coordinate axis has norm one.
Source: arXiv:2604.01978v1, §7, domination in the integration
by parts computation of `lem:gaussian_interp`. -/
theorem norm_gaussianCoordinateSliceDerivative_le {m : ℕ} (g : EucSpace m → ℝ)
    (i : Fin m) (C' : ℝ) (hbound : ∀ z, ‖fderiv ℝ g z‖ ≤ C')
    (p : ℝ × EucSpace m) : ‖gaussianCoordinateSliceDerivative g i p‖ ≤ C' := by
  have h := (fderiv ℝ g (p.1 • EuclideanSpace.single i 1 + p.2)).le_opNorm
    (EuclideanSpace.single i 1)
  simpa only [gaussianCoordinateSliceDerivative, PiLp.norm_single, Real.norm_eq_abs,
    abs_one, mul_one] using h.trans (by simpa using hbound (p.1 • EuclideanSpace.single i 1 + p.2))

/-- The original bounded test weighted by its actual standard
coordinate has a finite expectation. Source: arXiv:2604.01978v1,
§7, the original coordinate integral in `lem:gaussian_interp`. -/
theorem integrable_stdGaussian_coordinate_test {m : ℕ} (g : EucSpace m → ℝ)
    (hg : ContDiff ℝ 1 g) (i : Fin m) (C : ℝ) (hbound : ∀ z, ‖g z‖ ≤ C) :
    Integrable (fun z : EucSpace m => z i * g z) (stdGaussian (EucSpace m)) := by
  have h := integrable_gaussian_parameter_weighted_test (stdGaussian (EucSpace m))
    (fun z => z i) (hasGaussianLaw_stdGaussian_coordinate i)
    (fun z => standardGaussianResidual i z) (standardGaussianResidual i).continuous.aemeasurable
    (gaussianCoordinateSlice g i) (continuous_gaussianCoordinateSlice g hg i) C
    (fun p => hbound (p.1 • EuclideanSpace.single i 1 + p.2))
  simpa only [gaussianCoordinateSlice, standardGaussianResidual_reconstruct] using h

/-- The actual coordinate Fréchet derivative is integrable under the
original standard Gaussian law. Source: arXiv:2604.01978v1, §7,
the finite derivative expectation in `lem:gaussian_interp`. -/
theorem integrable_stdGaussian_coordinate_derivative {m : ℕ} (g : EucSpace m → ℝ)
    (hg : ContDiff ℝ 1 g) (i : Fin m) (C' : ℝ) (hbound : ∀ z, ‖fderiv ℝ g z‖ ≤ C') :
    Integrable (fun z : EucSpace m => fderiv ℝ g z (EuclideanSpace.single i 1))
      (stdGaussian (EucSpace m)) := by
  apply Integrable.of_bound ((hg.continuous_fderiv (by decide)).clm_apply
    continuous_const).aestronglyMeasurable C'
  exact ae_of_all _ fun z => by
    simpa only [gaussianCoordinateSliceDerivative, zero_smul, zero_add] using
      norm_gaussianCoordinateSliceDerivative_le g i C' hbound (0, z)

/-- Integration by parts for each actual coordinate of the original
standard Gaussian vector. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp`. The independent actual residual, the actual
scalar derivative and variance one give the identity for the original
test and its genuine Fréchet derivative, including every dimension. -/
theorem stdGaussian_stein_coordinate {m : ℕ} (g : EucSpace m → ℝ)
    (hg : ContDiff ℝ 1 g) (i : Fin m) (C C' : ℝ)
    (hbound : ∀ z, ‖g z‖ ≤ C) (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C') :
    (∫ z : EucSpace m, z i * g z ∂stdGaussian (EucSpace m)) =
      ∫ z : EucSpace m, fderiv ℝ g z (EuclideanSpace.single i 1) ∂stdGaussian (EucSpace m) := by
  have h := gaussian_stein_independent_parameter (stdGaussian (EucSpace m))
    (fun z => z i) (hasGaussianLaw_stdGaussian_coordinate i) (integral_stdGaussian_coordinate i)
    (fun z => standardGaussianResidual i z) (standardGaussianResidual i).continuous.aemeasurable
    (indepFun_stdGaussian_coordinate_residual i)
    (gaussianCoordinateSlice g i) (gaussianCoordinateSliceDerivative g i)
    (continuous_gaussianCoordinateSlice g hg i) (continuous_gaussianCoordinateSliceDerivative g hg i)
    (gaussianCoordinateSlice_hasDerivAt g hg i) C C'
    (fun p => hbound (p.1 • EuclideanSpace.single i 1 + p.2))
    (norm_gaussianCoordinateSliceDerivative_le g i C' hbound')
  simpa only [gaussianCoordinateSlice, gaussianCoordinateSliceDerivative,
    standardGaussianResidual_reconstruct, variance_stdGaussian_coordinate, one_mul] using h

/-- A nonconstant sine of an actual coordinate jointly witnesses the
smoothness and both boundedness inputs. Its derivative on its original
axis at zero is one under the nondegenerate two-dimensional law. -/
example : ∃ g : EucSpace 2 → ℝ, ContDiff ℝ 1 g ∧ (∀ z, ‖g z‖ ≤ 1) ∧
    (∀ z, ‖fderiv ℝ g z‖ ≤ ‖EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)‖) ∧
    fderiv ℝ g 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  let L := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  let g : EucSpace 2 → ℝ := fun z => Real.sin (L z)
  have hderiv (z : EucSpace 2) : fderiv ℝ g z = Real.cos (L z) • L :=
    L.hasFDerivAt.sin.fderiv
  refine ⟨g, L.contDiff.sin, ?_, ?_, ?_⟩
  · intro z
    exact Real.abs_sin_le_one (L z)
  · intro z
    rw [hderiv, norm_smul]
    calc
      ‖Real.cos (L z)‖ * ‖L‖ ≤ 1 * ‖L‖ :=
        mul_le_mul_of_nonneg_right (Real.abs_cos_le_one (L z)) (norm_nonneg L)
      _ = ‖L‖ := one_mul _
  · rw [hderiv]
    simp [L]

end Transformer.Homogenized
