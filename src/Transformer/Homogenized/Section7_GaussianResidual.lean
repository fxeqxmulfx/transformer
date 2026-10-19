module

public import Transformer.Homogenized.Section7_StandardGaussianMoments
public import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Independence

/-!
# Independence of an actual standard Gaussian coordinate and its residual

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. Removing one actual coordinate component
of the standard Gaussian vector gives its actual orthogonal residual.
The coordinate and this residual are jointly Gaussian, as genuine
linear images of the original vector, and are independent.

The cross covariance against every scalar and vector continuous
linear form is computed from the actual standard Gaussian moments.
Joint Gaussianity then supplies independence. No independence,
coordinate distribution or residual distribution is assumed.
The residual remains in the original ambient dimension, with its
selected coordinate zero, so the construction also retains its
genuine degenerate law without choosing a lower-dimensional density.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The actual linear orthogonal residual after removing one original
coordinate. Source: arXiv:2604.01978v1, §7, the actual coordinate
conditioning in the integration-by-parts proof of `lem:gaussian_interp`. -/
noncomputable def standardGaussianResidual {m : ℕ} (i : Fin m) : EucSpace m →L[ℝ] EucSpace m :=
  ContinuousLinearMap.id ℝ (EucSpace m) -
    (EuclideanSpace.proj (𝕜 := ℝ) i).smulRight (EuclideanSpace.single i 1)

/-- The actual residual subtracts exactly its original coordinate
component. Source: arXiv:2604.01978v1, §7, the original Gaussian
coordinate decomposition used in `lem:gaussian_interp`. -/
theorem standardGaussianResidual_apply {m : ℕ} (i : Fin m) (z : EucSpace m) :
    standardGaussianResidual i z = z - z i • EuclideanSpace.single i 1 := rfl

/-- The selected coordinate of the actual residual is zero.
Source: arXiv:2604.01978v1, §7, the actual Gaussian coordinate
decomposition in `lem:gaussian_interp`. -/
theorem standardGaussianResidual_coordinate {m : ℕ} (i : Fin m) (z : EucSpace m) :
    (standardGaussianResidual i z) i = 0 := by
  rw [standardGaussianResidual_apply]
  simp

/-- Every other actual coordinate is preserved by the residual.
Source: arXiv:2604.01978v1, §7, the original Gaussian coordinate
decomposition used in `lem:gaussian_interp`. -/
theorem standardGaussianResidual_other_coordinate {m : ℕ} (i j : Fin m)
    (hji : j ≠ i) (z : EucSpace m) : (standardGaussianResidual i z) j = z j := by
  rw [standardGaussianResidual_apply]
  simp [hji]

/-- The actual coordinate and residual reconstruct the original
vector. Source: arXiv:2604.01978v1, §7, the original Gaussian
conditioning in `lem:gaussian_interp`. -/
theorem standardGaussianResidual_reconstruct {m : ℕ} (i : Fin m) (z : EucSpace m) :
    z i • EuclideanSpace.single i 1 + standardGaussianResidual i z = z := by
  rw [standardGaussianResidual_apply]
  abel

/-- The actual residual is orthogonal to its original coordinate
axis. Source: arXiv:2604.01978v1, §7, the actual decomposition
underlying Gaussian coordinate integration by parts. -/
theorem inner_standardGaussianResidual {m : ℕ} (i : Fin m) (z : EucSpace m) :
    inner (𝕜 := ℝ) (EuclideanSpace.single i 1) (standardGaussianResidual i z) = 0 := by
  rw [EuclideanSpace.inner_single_left, map_one, one_mul, standardGaussianResidual_coordinate]

/-- The actual residual satisfies the original exact orthogonal
decomposition of its squared norm. Source: arXiv:2604.01978v1,
§7, the coordinate decomposition used in `lem:gaussian_interp`. -/
theorem norm_standardGaussianResidual_sq {m : ℕ} (i : Fin m) (z : EucSpace m) :
    ‖standardGaussianResidual i z‖ ^ 2 = ‖z‖ ^ 2 - (z i) ^ 2 := by
  have he : ‖(EuclideanSpace.single i 1 : EucSpace m)‖ = 1 := by simp
  rw [standardGaussianResidual_apply, norm_sub_sq_real, real_inner_smul_right,
    EuclideanSpace.inner_single_right, RCLike.conj_to_real, one_mul, norm_smul,
    he, mul_one, Real.norm_eq_abs, sq_abs]
  ring

/-- The actual residual never increases the original vector norm.
Source: arXiv:2604.01978v1, §7, the norm domination of the actual
coordinate conditioning in `lem:gaussian_interp`. -/
theorem norm_standardGaussianResidual_le {m : ℕ} (i : Fin m) (z : EucSpace m) :
    ‖standardGaussianResidual i z‖ ≤ ‖z‖ := by
  have h := norm_standardGaussianResidual_sq i z
  nlinarith [sq_nonneg (z i), norm_nonneg z, norm_nonneg (standardGaussianResidual i z)]

/-- The actual residual has a Gaussian law, including its original
zero coordinate. Source: arXiv:2604.01978v1, §7, the Gaussian
conditioning in `lem:gaussian_interp`, including degenerate laws. -/
theorem hasGaussianLaw_standardGaussianResidual {m : ℕ} (i : Fin m) :
    HasGaussianLaw (fun z : EucSpace m => standardGaussianResidual i z) (stdGaussian (EucSpace m)) :=
  (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace m))).map (standardGaussianResidual i)

/-- The actual coordinate and residual are jointly Gaussian, as the
actual product linear image of the original standard vector.
Source: arXiv:2604.01978v1, §7, the original Gaussian integration
by parts used in `lem:gaussian_interp`. -/
theorem hasGaussianLaw_standardGaussianCoordinate_residual {m : ℕ} (i : Fin m) :
    HasGaussianLaw (fun z : EucSpace m => (z i, standardGaussianResidual i z))
      (stdGaussian (EucSpace m)) :=
  (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace m))).map
    ((EuclideanSpace.proj (𝕜 := ℝ) i).prod (standardGaussianResidual i))

/-- The original coordinate is uncorrelated with every actual linear
form of its actual residual. Source: arXiv:2604.01978v1, §7,
the covariance computation proving the original Gaussian conditioning. -/
theorem covariance_stdGaussian_coordinate_residual {m : ℕ}
    (i : Fin m) (L : EucSpace m →L[ℝ] ℝ) :
    covariance (fun z : EucSpace m => z i) (fun z => L (standardGaussianResidual i z))
      (stdGaussian (EucSpace m)) = 0 := by
  let c : ℝ := L (EuclideanSpace.single i 1)
  have hfun : (fun z : EucSpace m => L (standardGaussianResidual i z)) =
      (fun z => L z) - (fun z => c * z i) := by
    funext z
    rw [standardGaussianResidual_apply, map_sub, map_smul]
    simp only [Pi.sub_apply, smul_eq_mul]
    dsimp only [c]
    ring
  have hcoord := (hasGaussianLaw_stdGaussian_coordinate i).memLp_two
  have hform : MemLp (fun z : EucSpace m => L z) 2 (stdGaussian (EucSpace m)) :=
    ((IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace m))).map L).memLp_two
  rw [hfun, covariance_sub_right hcoord hform (hcoord.const_mul c),
    covariance_stdGaussian_coordinate_linearForm i L, covariance_const_mul_right c,
    covariance_stdGaussian_coordinates i i]
  simp only [ite_true, mul_one, c, sub_self]

/-- The original standard Gaussian coordinate is independent of the
actual residual vector. Source: arXiv:2604.01978v1, §7,
the original Gaussian coordinate conditioning in `lem:gaussian_interp`.
Joint Gaussianity and every actual cross covariance supply the proof. -/
theorem indepFun_stdGaussian_coordinate_residual {m : ℕ} (i : Fin m) :
    IndepFun (fun z : EucSpace m => z i) (fun z => standardGaussianResidual i z)
      (stdGaussian (EucSpace m)) := by
  apply (hasGaussianLaw_standardGaussianCoordinate_residual i).indepFun_of_covariance_strongDual
  intro L₁ L₂
  have hscalar (r : ℝ) : L₁ r = L₁ 1 * r := by
    have h := L₁.map_smul r (1 : ℝ)
    simpa only [smul_eq_mul, mul_one, mul_comm] using h
  change covariance (fun z : EucSpace m => L₁ (z i))
    (fun z => L₂ (standardGaussianResidual i z)) (stdGaussian (EucSpace m)) = 0
  have hfun : (fun z : EucSpace m => L₁ (z i)) = (fun z => L₁ 1 * z i) :=
    funext fun z => hscalar (z i)
  rw [hfun, covariance_const_mul_left, covariance_stdGaussian_coordinate_residual i L₂, mul_zero]

/-- Both actual coordinates occur in the nondegenerate original
two-dimensional standard law. Its nonzero vector has a nonzero residual,
and its original coordinate is genuinely independent of that residual. -/
example : (1 : Fin 2) ≠ 0 ∧
    HasGaussianLaw (fun z : EucSpace 2 => (z (0 : Fin 2), standardGaussianResidual 0 z))
      (stdGaussian (EucSpace 2)) ∧
    IndepFun (fun z : EucSpace 2 => z (0 : Fin 2)) (fun z => standardGaussianResidual 0 z)
      (stdGaussian (EucSpace 2)) ∧
    standardGaussianResidual (0 : Fin 2)
      (EuclideanSpace.single 0 1 + EuclideanSpace.single 1 1) = EuclideanSpace.single 1 1 := by
  refine ⟨by decide, hasGaussianLaw_standardGaussianCoordinate_residual 0,
    indepFun_stdGaussian_coordinate_residual 0, ?_⟩
  ext j
  fin_cases j <;> simp [standardGaussianResidual_apply]

end Transformer.Homogenized
