module

public import Transformer.Homogenized.Section7_GaussianVectorConditioning

/-!
# Actual derivatives of the bounded Gaussian interpolation test

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The source states the identity for every
`C²` test without a growth condition. The corrected formal statement
uses a uniform bound on the original test and its first two derivatives
to make all original expectations finite and justify differentiation.

The coordinate gradient below is the original Fréchet derivative on
the actual coordinate axis. Its derivative is the actual second
Fréchet derivative with the appropriate argument order. Operator norms
of these original derivatives are identified with the actual iterated
derivative norms occurring in the corrected statement.
A nonconstant coordinate sine witnesses all of these hypotheses with
a nonzero original derivative; no derivative or bound is assumed for it.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The actual original coordinate gradient of the original test.
Source: arXiv:2604.01978v1, §7, the original gradient used in
the integration-by-parts proof of `lem:gaussian_interp`. -/
noncomputable def gaussianCoordinateGradient {m : ℕ} (φ : EucSpace m → ℝ)
    (i : Fin m) (z : EucSpace m) : ℝ := fderiv ℝ φ z (EuclideanSpace.single i 1)

/-- The actual zeroth derivative bound bounds the original test.
Source: arXiv:2604.01978v1, §7, the finite expectation in
the corrected bounded-test statement of `lem:gaussian_interp`. -/
theorem norm_gaussian_test_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) (z : EucSpace m) :
    ‖φ z‖ ≤ C := by simpa only [norm_iteratedFDeriv_zero] using hbdd 0 (by omega) z

/-- The uniform original test bound is necessarily nonnegative.
Source: arXiv:2604.01978v1, §7, domination in the corrected
bounded-test statement of `lem:gaussian_interp`. -/
theorem gaussian_test_bound_nonneg {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) : 0 ≤ C :=
  (norm_nonneg (φ 0)).trans (norm_gaussian_test_le φ C hbdd 0)

/-- The original first iterated derivative bounds the actual Fréchet
derivative. Source: arXiv:2604.01978v1, §7, domination of
the path derivative in the corrected `lem:gaussian_interp`. -/
theorem norm_gaussian_test_fderiv_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) (z : EucSpace m) :
    ‖fderiv ℝ φ z‖ ≤ C := by simpa only [norm_iteratedFDeriv_one] using hbdd 1 (by omega) z

/-- The original second iterated derivative bounds the actual second
Fréchet derivative. Source: arXiv:2604.01978v1, §7,
the genuine Hessian expectation in the corrected `lem:gaussian_interp`. -/
theorem norm_gaussian_test_second_fderiv_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) (z : EucSpace m) :
    ‖fderiv ℝ (fderiv ℝ φ) z‖ ≤ C := by
  rw [← norm_iteratedFDeriv_one (fderiv ℝ φ), norm_iteratedFDeriv_fderiv]
  exact hbdd 2 (by omega) z

/-- The actual coordinate gradient of the original `C²` test is `C¹`.
Source: arXiv:2604.01978v1, §7, the original test to which
Gaussian integration by parts applies in `lem:gaussian_interp`. -/
theorem contDiff_gaussianCoordinateGradient {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (i : Fin m) : ContDiff ℝ 1 (gaussianCoordinateGradient φ i) := by
  change ContDiff ℝ 1 (fun z : EucSpace m => fderiv ℝ φ z (EuclideanSpace.single i 1))
  have hD : ContDiff ℝ 1 (fderiv ℝ φ) := hφ.fderiv_right (by norm_num)
  exact hD.clm_apply contDiff_const

/-- The derivative of the original coordinate gradient is the actual
second derivative with that original coordinate as its second argument.
Source: arXiv:2604.01978v1, §7, the original Hessian contraction
in the integration-by-parts proof of `lem:gaussian_interp`. -/
theorem fderiv_gaussianCoordinateGradient {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (i : Fin m) (z : EucSpace m) :
    fderiv ℝ (gaussianCoordinateGradient φ i) z =
      (fderiv ℝ (fderiv ℝ φ) z).flip (EuclideanSpace.single i 1) := by
  change fderiv ℝ (fun y : EucSpace m => fderiv ℝ φ y (EuclideanSpace.single i 1)) z = _
  have hD : ContDiff ℝ 1 (fderiv ℝ φ) := hφ.fderiv_right (by norm_num)
  have h := (((hD.differentiable (by decide)) z).hasFDerivAt.clm_apply
    (hasFDerivAt_const (EuclideanSpace.single i (1 : ℝ)) z)).fderiv
  simpa only [ContinuousLinearMap.comp_zero, zero_add] using h

/-- The actual original coordinate gradient obeys the first derivative
bound, since its original coordinate axis has norm one.
Source: arXiv:2604.01978v1, §7, the bounded original Stein test. -/
theorem norm_gaussianCoordinateGradient_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (i : Fin m) (z : EucSpace m) : ‖gaussianCoordinateGradient φ i z‖ ≤ C := by
  have h := (fderiv ℝ φ z).le_opNorm (EuclideanSpace.single i 1)
  simpa only [gaussianCoordinateGradient, PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one]
    using h.trans (by simpa using norm_gaussian_test_fderiv_le φ C hbdd z)

/-- The actual coordinate-gradient derivative obeys the original
Hessian norm bound. Source: arXiv:2604.01978v1, §7,
the derivative domination for vector Stein in `lem:gaussian_interp`. -/
theorem norm_fderiv_gaussianCoordinateGradient_le {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (i : Fin m) (z : EucSpace m) : ‖fderiv ℝ (gaussianCoordinateGradient φ i) z‖ ≤ C := by
  rw [fderiv_gaussianCoordinateGradient φ hφ i z]
  have h := (fderiv ℝ (fderiv ℝ φ) z).flip.le_opNorm (EuclideanSpace.single i 1)
  simpa only [PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one, ContinuousLinearMap.opNorm_flip]
    using h.trans (by simpa using norm_gaussian_test_second_fderiv_le φ C hbdd z)

/-- Every original coordinate Hessian entry obeys the original uniform
second derivative bound. Source: arXiv:2604.01978v1, §7,
the bounded original trace expectation in `lem:gaussian_interp`. -/
theorem norm_gaussian_test_hessian_entry_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (i j : Fin m) (z : EucSpace m) :
    ‖iteratedFDeriv ℝ 2 φ z ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]‖ ≤ C := by
  have h := (iteratedFDeriv ℝ 2 φ z).le_opNorm
    ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]
  have hnorm : (∏ k : Fin 2,
      ‖(![EuclideanSpace.single i 1, EuclideanSpace.single j 1] k : EucSpace m)‖) = 1 := by simp
  exact h.trans (by rw [hnorm, mul_one]; exact hbdd 2 (by omega) z)

/-- A nonconstant test jointly satisfies every corrected original
bounded `C²` hypothesis. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp`. Its actual derivative on its own axis at zero
is one, and all three original derivative bounds are proved explicitly. -/
theorem gaussian_interp_test_witness : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    ContDiff ℝ 2 φ ∧ (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  let A := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  let φ : EucSpace 2 → ℝ := fun z => Real.sin (A z)
  let C : ℝ := max 1 (max ‖A‖ (‖A‖ * ‖A‖))
  have h1 : 1 ≤ C := le_max_left _ _
  have hA : ‖A‖ ≤ C := (le_max_left _ _).trans (le_max_right _ _)
  have hAA : ‖A‖ * ‖A‖ ≤ C := (le_max_right _ _).trans (le_max_right _ _)
  have hd : fderiv ℝ φ = fun z => Real.cos (A z) • A := funext fun z => A.hasFDerivAt.sin.fderiv
  have hd2 (z : EucSpace 2) : fderiv ℝ (fderiv ℝ φ) z = (-Real.sin (A z) • A).smulRight A := by
    rw [hd]
    exact (A.hasFDerivAt.cos.smul_const A).fderiv
  refine ⟨φ, C, A.contDiff.sin, ?_, ?_⟩
  · intro k hk z
    interval_cases k
    · rw [norm_iteratedFDeriv_zero]
      exact (Real.abs_sin_le_one (A z)).trans h1
    · rw [norm_iteratedFDeriv_one, hd, norm_smul]
      have h : ‖Real.cos (A z)‖ * ‖A‖ ≤ ‖A‖ := by
        simpa only [Real.norm_eq_abs, one_mul] using
          mul_le_mul_of_nonneg_right (Real.abs_cos_le_one (A z)) (norm_nonneg A)
      exact h.trans hA
    · rw [← norm_iteratedFDeriv_fderiv (n := 1), norm_iteratedFDeriv_one, hd2,
        ContinuousLinearMap.norm_smulRight_apply, norm_smul]
      have h : ‖-Real.sin (A z)‖ * ‖A‖ ≤ ‖A‖ := by
        simpa only [norm_neg, Real.norm_eq_abs, one_mul] using
          mul_le_mul_of_nonneg_right (Real.abs_sin_le_one (A z)) (norm_nonneg A)
      exact (mul_le_mul_of_nonneg_right h (norm_nonneg A)).trans hAA
  · rw [hd]; simp [A]

/-- The nonconstant original sine test is a joint witness for every
boundedness and regularity hypothesis of the coordinate-gradient lemmas. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ), ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := gaussian_interp_test_witness

end Transformer.Homogenized
