/-
# The actual blended gradient under polynomial Gaussian domination

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:gaussian_interp_0`. The coordinate gradient at the actual linear blend
has polynomial growth in the original pair of Gaussian vectors. Its full
derivative has polynomial growth from the original Hessian and the actual
chain rule. The proved polynomial vector Stein identity therefore applies.

The original covariance contraction retains the exact coefficient of the
original Gaussian vector, for cancellation against the interpolation
velocity. Coordinate-weighted gradients and genuine Hessian entries are
integrable under the original joint law. Singular Gaussian laws remain
included. The source's polynomial bounds on its bilinear softmax test,
rather than score-variable uniform bounds, justify these expectations.
-/

import Transformer.Homogenized.Section7_GaussianPolynomialTestBounds

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual blended coordinate gradient has a polynomial bound
on the original pair. Source: arXiv:2604.01978v1, §7, the
genuine Stein test in the proof of `lem:gaussian_interp`. -/
theorem norm_gaussianBlendGradient_polynomial_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (a b : ℝ) (i : Fin m) (z : EucSpace m × EucSpace m) :
    ‖gaussianBlendGradient φ a b i z‖ ≤
      (C * (1 + ‖gaussianBlendLinearMap (m := m) a b‖) ^ p) * (1 + ‖z‖) ^ p := by
  simpa only [norm_zero, add_zero, gaussianBlendGradient, Function.comp_def] using
    norm_polynomial_gaussian_affine_le (gaussianBlendLinearMap a b) 0 (gaussianCoordinateGradient φ i)
      C (gaussian_test_polynomial_bound_nonneg φ C p hbdd) p
      (norm_gaussianCoordinateGradient_polynomial_le φ C p hbdd i) z

/-- The actual full derivative of the blended gradient obeys the
original polynomial Hessian bound through the genuine blending map.
Source: arXiv:2604.01978v1, §7, the actual derivative in the Stein step. -/
theorem norm_fderiv_gaussianBlendGradient_polynomial_le {m : ℕ}
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (a b : ℝ) (i : Fin m) (z : EucSpace m × EucSpace m) :
    ‖fderiv ℝ (gaussianBlendGradient φ a b i) z‖ ≤
      (C * (1 + ‖gaussianBlendLinearMap (m := m) a b‖) ^ p * ‖gaussianBlendLinearMap (m := m) a b‖) *
        (1 + ‖z‖) ^ p := by
  let L := gaussianBlendLinearMap (m := m) a b
  have h := norm_polynomial_gaussian_affine_le L 0
    (fun w => ‖fderiv ℝ (gaussianCoordinateGradient φ i) w‖) C
    (gaussian_test_polynomial_bound_nonneg φ C p hbdd) p
    (fun w => by simpa only [norm_norm] using
      norm_fderiv_gaussianCoordinateGradient_polynomial_le φ hφ C p hbdd i w) z
  simp only [norm_zero, add_zero, norm_norm] at h
  rw [fderiv_gaussianBlendGradient φ hφ a b i z]
  calc
    _ ≤ ‖fderiv ℝ (gaussianCoordinateGradient φ i) (L z)‖ * ‖L‖ :=
      (fderiv ℝ (gaussianCoordinateGradient φ i) (L z)).opNorm_comp_le L
    _ ≤ (C * (1 + ‖L‖) ^ p) * (1 + ‖z‖) ^ p * ‖L‖ :=
      mul_le_mul_of_nonneg_right h (norm_nonneg L)
    _ = _ := by ring

/-- The original coordinate-weighted blended gradient is integrable
on its actual original probability space. Source: arXiv:2604.01978v1,
§7, the finite original gradient moments used in `lem:gaussian_interp`. -/
theorem integrable_gaussian_blend_coordinate_gradient_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (a b : ℝ) (i : Fin m) :
    Integrable (fun ω => X ω i * fderiv ℝ φ (a • X ω + b • Y ω) (EuclideanSpace.single i 1)) P := by
  exact integrable_gaussian_pair_coordinate_test_polynomial P X Y hX hY hindep
    (gaussianBlendGradient φ a b i) (contDiff_gaussianBlendGradient φ hφ a b i).continuous i
    (C * (1 + ‖gaussianBlendLinearMap (m := m) a b‖) ^ p) p
    (norm_gaussianBlendGradient_polynomial_le φ C p hbdd a b i)

/-- Each actual Hessian entry at the original blended vector has a
finite original expectation. Source: arXiv:2604.01978v1, §7,
the actual original covariance trace in `lem:gaussian_interp`. -/
theorem integrable_gaussian_blend_hessian_entry_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (a b : ℝ) (i j : Fin m) :
    Integrable (fun ω => iteratedFDeriv ℝ 2 φ (a • X ω + b • Y ω)
      ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]) P := by
  let g : EucSpace m → ℝ := fun z => iteratedFDeriv ℝ 2 φ z
    ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]
  have hD : ContDiff ℝ 1 (fderiv ℝ φ) := hφ.fderiv_right (by norm_num)
  have hD2 : Continuous (fderiv ℝ (fderiv ℝ φ)) := hD.continuous_fderiv (by decide)
  have hg : Continuous g := by
    dsimp only [g]
    simp only [iteratedFDeriv_two_apply, Matrix.cons_val_zero, Matrix.cons_val_one]
    fun_prop
  exact integrable_gaussian_pair_test_polynomial P X Y hX hY hindep
    (g ∘ gaussianBlendLinearMap a b) (hg.comp (gaussianBlendLinearMap a b).continuous)
    (C * (1 + ‖gaussianBlendLinearMap (m := m) a b‖) ^ p) p (fun z => by
      simpa only [Function.comp_def, norm_zero, add_zero] using
        norm_polynomial_gaussian_affine_le (gaussianBlendLinearMap a b) 0 g C
          (gaussian_test_polynomial_bound_nonneg φ C p hbdd) p
          (norm_gaussian_test_hessian_entry_polynomial_le φ C p hbdd i j) z)

/-- The original coordinate-weighted blended gradient equals its
actual covariance contraction with the original Hessian, with the exact
original scalar coefficient. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp` and `eq:gaussian_interp_0`, for polynomial growth. -/
theorem gaussian_stein_blend_gradient_polynomial {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0)
    (hcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = S i j)
    (Y : Ω → EucSpace m) (hY : HasGaussianLaw Y P) (hindep : IndepFun X Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (a b : ℝ) (i : Fin m) :
    (∫ ω, X ω i * fderiv ℝ φ (a • X ω + b • Y ω) (EuclideanSpace.single i 1) ∂P) =
      a * ∫ ω, ∑ j : Fin m, S i j * iteratedFDeriv ℝ 2 φ (a • X ω + b • Y ω)
        ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] ∂P := by
  have h := gaussian_stein_vector_independent_parameter_polynomial P X hX S hcenter hcov Y hY hindep
    (gaussianBlendGradient φ a b i) (contDiff_gaussianBlendGradient φ hφ a b i) i
    (C * (1 + ‖gaussianBlendLinearMap (m := m) a b‖) ^ p)
    (C * (1 + ‖gaussianBlendLinearMap (m := m) a b‖) ^ p * ‖gaussianBlendLinearMap (m := m) a b‖) p
    (norm_gaussianBlendGradient_polynomial_le φ C p hbdd a b i)
    (norm_fderiv_gaussianBlendGradient_polynomial_le φ hφ C p hbdd a b i)
  simp only [fderiv_gaussianBlendGradient_first_axis φ hφ] at h
  rw [← integral_const_mul]
  refine h.trans ?_
  apply integral_congr_ae
  apply ae_of_all
  intro ω
  simp only [Finset.mul_sum]
  exact Finset.sum_congr rfl fun j hj => by ring

/-- A genuine quadratic test, original nondegenerate Gaussian and
independent zero Gaussian jointly witness every original moment,
independence, regularity and polynomial bound hypothesis. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    HasGaussianLaw (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧ ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 := by
  obtain ⟨φ, C, hφ, hbdd, hline, hhess⟩ := gaussian_interp_polynomial_test_witness
  refine ⟨φ, C, IsGaussian.hasGaussianLaw_id, ?_, indepFun_const_right _ _,
    integral_stdGaussian_coordinate, ?_, hφ, hbdd, hline, hhess⟩
  · simpa only [Function.comp_def, zero_apply] using
      (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace 2))).map
        (0 : EucSpace 2 →L[ℝ] EucSpace 2)
  · intro i j
    simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j

end Transformer.Homogenized
