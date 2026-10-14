/-
# The original Gaussian interpolation covariance–Hessian derivative

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. Vector integration by parts identifies
each original vector gradient expectation with its original covariance
contraction, multiplied by that vector's actual square-root coefficient.
Symmetry of the original second moments gives the original Hessian
argument order in the covariance trace.

The actual interior path derivative has the inverse of twice each
of those original square-root coefficients. Their exact cancellation
therefore gives the original factor one half and the original difference
of covariance matrices. All original expectations and finite sum
exchanges are justified by the original Gaussian moments and test bounds.
The construction retains both original vectors and includes singular
original covariance matrices without a density assumption.
-/

import Transformer.Homogenized.Section7_GaussianGradientMoments

open Real MeasureTheory ProbabilityTheory Set
open scoped BigOperators

namespace Transformer.Homogenized

/-- The original vector gradient expectation is its actual original
covariance trace, with the exact original blending coefficient.
Source: arXiv:2604.01978v1, §7, the Gaussian integration-by-parts
step in `lem:gaussian_interp`. Actual covariance symmetry supplies the
original Hessian argument order in the trace. -/
theorem integral_gaussianInterpolation_vector_gradient_covariance
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω)
    (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0)
    (hcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = S i j)
    (Y : Ω → EucSpace m) (hY : AEMeasurable Y P) (hindep : IndepFun X Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) (t : ℝ) :
    (∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (X ω) ∂P) =
      Real.sqrt t * ∫ ω, gaussianCovarianceTrace φ S
        (gaussianInterpolationPoint t (X ω) (Y ω)) ∂P := by
  let : IsProbabilityMeasure P := hX.isProbabilityMeasure
  have hstein (i : Fin m) :
      (∫ ω, X ω i * fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
        (EuclideanSpace.single i 1) ∂P) = Real.sqrt t * ∫ ω, ∑ j : Fin m,
        S i j * iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
          ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] ∂P := by
    simpa only [gaussianInterpolationPoint] using gaussian_stein_blend_gradient P X hX S hcenter hcov
      Y hY hindep φ hφ C hbdd (Real.sqrt t) (Real.sqrt (1 - t)) i
  have hint (i j : Fin m) :=
    (integrable_gaussianInterpolation_hessian_entry P X Y hX.aemeasurable hY φ hφ C hbdd t j i).const_mul (S i j)
  have hsum : (∑ i : Fin m, ∫ ω, ∑ j : Fin m, S i j *
      iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
        ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] ∂P) =
      ∫ ω, ∑ i : Fin m, ∑ j : Fin m, S i j *
        iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
          ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] ∂P :=
    (integral_finsetSum _ (fun i hi => integrable_finsetSum _ (fun j hj => hint i j))).symm
  calc
    _ = Real.sqrt t * ∑ i : Fin m, ∫ ω, ∑ j : Fin m, S i j *
        iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
          ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] ∂P := by
      rw [integral_gaussianInterpolation_vector_gradient P X hX Y hY φ hφ C hbdd t]
      simp only [hstein, Finset.mul_sum]
    _ = Real.sqrt t * ∫ ω, ∑ i : Fin m, ∑ j : Fin m, S i j *
        iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
          ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] ∂P := congrArg (fun r => Real.sqrt t * r) hsum
    _ = _ := by
      congr 1
      apply integral_congr_ae
      apply ae_of_all
      intro ω
      change (∑ i : Fin m, ∑ j : Fin m, S i j *
        iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
          ![EuclideanSpace.single j 1, EuclideanSpace.single i 1]) = _
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by
        rw [gaussian_secondMoment_symmetric P X S hcov j i]

/-- The actual original scalar path derivative expectation equals the
original half covariance-difference trace of its actual original Hessian.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
The original square-root factors cancel exactly at every interior time;
both singular Gaussian laws and the original covariance entries are retained. -/
theorem integral_gaussianInterpolation_velocity_covariance
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) (X Y : Ω → EucSpace m)
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (SX SY : Matrix (Fin m) (Fin m) ℝ)
    (hXc : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0) (hYc : ∀ i : Fin m, ∫ ω, Y ω i ∂P = 0)
    (hXcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = SX i j)
    (hYcov : ∀ i j : Fin m, ∫ ω, Y ω i * Y ω j ∂P = SY i j) (hindep : IndepFun X Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) :
    (∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
      (gaussianInterpolationVelocity t (X ω) (Y ω)) ∂P) =
      (1 / 2 : ℝ) * ∫ ω, gaussianCovarianceTrace φ (SX - SY)
        (gaussianInterpolationPoint t (X ω) (Y ω)) ∂P := by
  let : IsProbabilityMeasure P := hX.isProbabilityMeasure
  have hswap : 1 - (1 - t) = t := by ring
  have hxint := integrable_gaussianInterpolation_vector_gradient P X hX Y hY.aemeasurable φ hφ C hbdd t
  have hyint : Integrable (fun ω => fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (Y ω)) P := by
    simpa only [gaussianInterpolationPoint, hswap, add_comm] using
      integrable_gaussianInterpolation_vector_gradient P Y hY X hX.aemeasurable φ hφ C hbdd (1 - t)
  have hx := integral_gaussianInterpolation_vector_gradient_covariance P X hX SX hXc hXcov
    Y hY.aemeasurable hindep φ hφ C hbdd t
  have hy : (∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (Y ω) ∂P) =
      Real.sqrt (1 - t) * ∫ ω, gaussianCovarianceTrace φ SY
        (gaussianInterpolationPoint t (X ω) (Y ω)) ∂P := by
    simpa only [gaussianInterpolationPoint, hswap, add_comm] using
      integral_gaussianInterpolation_vector_gradient_covariance P Y hY SY hYc hYcov
        X hX.aemeasurable hindep.symm φ hφ C hbdd (1 - t)
  have ha : (1 / (2 * Real.sqrt t)) * Real.sqrt t = (1 / 2 : ℝ) := by
    field_simp [Real.sqrt_ne_zero'.mpr ht.1]
  have hsqY : Real.sqrt (1 - t) ≠ 0 := Real.sqrt_ne_zero'.mpr (by linarith [ht.2])
  have hb : (1 / (2 * Real.sqrt (1 - t))) * Real.sqrt (1 - t) = (1 / 2 : ℝ) := by
    field_simp [hsqY]
  have hTX := integrable_gaussianInterpolation_covarianceTrace P X Y hX.aemeasurable hY.aemeasurable φ hφ C hbdd SX t
  have hTY := integrable_gaussianInterpolation_covarianceTrace P X Y hX.aemeasurable hY.aemeasurable φ hφ C hbdd SY t
  calc
    _ = (1 / (2 * Real.sqrt t)) * ∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (X ω) ∂P -
        (1 / (2 * Real.sqrt (1 - t))) * ∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (Y ω) ∂P := by
      simp only [gaussianInterpolation_velocity_gradient_expansion]
      rw [integral_sub (hxint.const_mul _) (hyint.const_mul _), integral_const_mul, integral_const_mul]
    _ = (1 / 2 : ℝ) * ((∫ ω, gaussianCovarianceTrace φ SX (gaussianInterpolationPoint t (X ω) (Y ω)) ∂P) -
        ∫ ω, gaussianCovarianceTrace φ SY (gaussianInterpolationPoint t (X ω) (Y ω)) ∂P) := by
      rw [hx, hy, ← mul_assoc, ha, ← mul_assoc, hb]
      ring
    _ = _ := by
      rw [← integral_sub hTX hTY]
      congr 1
      exact integral_congr_ae (ae_of_all _ fun ω =>
        (gaussianCovarianceTrace_sub φ SX SY (gaussianInterpolationPoint t (X ω) (Y ω))).symm)

/-- A genuine nondegenerate original Gaussian vector, the original
zero Gaussian vector and the nonconstant bounded sine jointly witness
every original centering, covariance, independence and regularity premise. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    HasGaussianLaw (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ _z : EucSpace 2, (0 : EucSpace 2) i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ _z : EucSpace 2, (0 : EucSpace 2) i * (0 : EucSpace 2) j ∂stdGaussian (EucSpace 2) =
      (0 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    IndepFun id (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    ContDiff ℝ 2 φ ∧ (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧ fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  obtain ⟨φ, C, hφ, hbdd, hn⟩ := gaussian_interp_test_witness
  refine ⟨φ, C, IsGaussian.hasGaussianLaw_id, integral_stdGaussian_coordinate, ?_,
    (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace 2))).map
      (0 : EucSpace 2 →L[ℝ] EucSpace 2), fun i => by simp, fun i j => by simp,
    indepFun_const_right _ _, hφ, hbdd, by norm_num, hn⟩
  intro i j
  simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j

end Transformer.Homogenized
