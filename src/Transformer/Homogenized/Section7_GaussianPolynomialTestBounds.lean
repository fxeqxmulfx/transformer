module

public import Transformer.Homogenized.Section7_GaussianPolynomialParameter
public import Transformer.Homogenized.Section7_GaussianBlendGradient

/-!
# Polynomial bounds on the actual Gaussian interpolation derivatives

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and the polynomial
growth justification following `eq:interpolation_interm`. Bounds on the
original iterated derivatives give polynomial bounds on the actual test,
its genuine Fréchet derivative, coordinate gradients and Hessian entries.

The source's bilinear scores produce this kind of derivative growth;
uniform softmax bounds in the score variables do not bound the derivatives
in the original Gaussian vectors. The bounds here remain in those original
vectors. An actual quadratic test witnesses the hypotheses with an
unbounded test and gradient and with a nonzero actual Hessian.
All derivatives are the original derivatives, not auxiliary definitions
that replace a derivative or the function being differentiated.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- Zeroth derivative control bounds the actual polynomial test.
Source: arXiv:2604.01978v1, §7, the original expectation in
`lem:gaussian_interp`, with polynomial growth control. -/
theorem norm_gaussian_test_polynomial_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (z : EucSpace m) : ‖φ z‖ ≤ C * (1 + ‖z‖) ^ p := by
  simpa only [norm_iteratedFDeriv_zero] using hbdd 0 (by omega) z

/-- The polynomial envelope constant is nonnegative, from the
actual test bound at zero. Source: arXiv:2604.01978v1, §7,
domination following `eq:interpolation_interm`. -/
theorem gaussian_test_polynomial_bound_nonneg {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p) :
    0 ≤ C := by
  have h := (norm_nonneg (φ 0)).trans (norm_gaussian_test_polynomial_le φ C p hbdd 0)
  simpa only [norm_zero, add_zero, one_pow, mul_one] using h

/-- The actual first derivative obeys the original polynomial bound.
Source: arXiv:2604.01978v1, §7, domination of the genuine path
derivative in `lem:gaussian_interp`. -/
theorem norm_gaussian_test_fderiv_polynomial_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (z : EucSpace m) : ‖fderiv ℝ φ z‖ ≤ C * (1 + ‖z‖) ^ p := by
  simpa only [norm_iteratedFDeriv_one] using hbdd 1 (by omega) z

/-- The actual second Fréchet derivative has the same polynomial
bound. Source: arXiv:2604.01978v1, §7, the genuine Hessian
expectation in `lem:gaussian_interp`. -/
theorem norm_gaussian_test_second_fderiv_polynomial_le {m : ℕ}
    (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (z : EucSpace m) : ‖fderiv ℝ (fderiv ℝ φ) z‖ ≤ C * (1 + ‖z‖) ^ p := by
  rw [← norm_iteratedFDeriv_one (fderiv ℝ φ), norm_iteratedFDeriv_fderiv]
  exact hbdd 2 (by omega) z

/-- The actual coordinate gradient obeys the original first derivative
bound, since the original axis has norm one.
Source: arXiv:2604.01978v1, §7, the genuine vector Stein test. -/
theorem norm_gaussianCoordinateGradient_polynomial_le {m : ℕ}
    (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (i : Fin m) (z : EucSpace m) : ‖gaussianCoordinateGradient φ i z‖ ≤ C * (1 + ‖z‖) ^ p := by
  have h := (fderiv ℝ φ z).le_opNorm (EuclideanSpace.single i 1)
  simpa only [gaussianCoordinateGradient, PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one]
    using h.trans (by simpa using norm_gaussian_test_fderiv_polynomial_le φ C p hbdd z)

/-- The actual derivative of the original coordinate gradient obeys
the polynomial Hessian norm bound. Source: arXiv:2604.01978v1,
§7, derivative domination in the original Gaussian Stein step. -/
theorem norm_fderiv_gaussianCoordinateGradient_polynomial_le {m : ℕ}
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (i : Fin m) (z : EucSpace m) :
    ‖fderiv ℝ (gaussianCoordinateGradient φ i) z‖ ≤ C * (1 + ‖z‖) ^ p := by
  rw [fderiv_gaussianCoordinateGradient φ hφ i z]
  have h := (fderiv ℝ (fderiv ℝ φ) z).flip.le_opNorm (EuclideanSpace.single i 1)
  simpa only [PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one, ContinuousLinearMap.opNorm_flip]
    using h.trans (by simpa using norm_gaussian_test_second_fderiv_polynomial_le φ C p hbdd z)

/-- Every actual coordinate Hessian entry retains the original
polynomial bound. Source: arXiv:2604.01978v1, §7, the original
covariance trace in `lem:gaussian_interp`. -/
theorem norm_gaussian_test_hessian_entry_polynomial_le {m : ℕ}
    (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (i j : Fin m) (z : EucSpace m) :
    ‖iteratedFDeriv ℝ 2 φ z ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]‖ ≤ C * (1 + ‖z‖) ^ p := by
  have h := (iteratedFDeriv ℝ 2 φ z).le_opNorm
    ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]
  have hn : (∏ k : Fin 2,
      ‖(![EuclideanSpace.single i 1, EuclideanSpace.single j 1] k : EucSpace m)‖) = 1 := by simp
  exact h.trans (by rw [hn, mul_one]; exact hbdd 2 (by omega) z)

/-- A genuine quadratic test has polynomially controlled original
derivatives and a nonzero actual Hessian. Source: arXiv:2604.01978v1,
§7, a witness for the growth-controlled `lem:gaussian_interp`. -/
theorem gaussian_interp_polynomial_test_witness : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    ContDiff ℝ 2 φ ∧ (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 := by
  let A := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  let φ : EucSpace 2 → ℝ := fun z => (A z) ^ 2
  let C : ℝ := 2 * ‖A‖ ^ 2
  have hd : fderiv ℝ φ = fun z => (2 * A z) • A := by
    funext z
    simpa only [φ, Nat.reduceSub, pow_one, two_nsmul, ← two_mul] using (A.hasFDerivAt.pow 2).fderiv
  have hd2 (z : EucSpace 2) : fderiv ℝ (fderiv ℝ φ) z = ((2 : ℝ) • A).smulRight A := by
    rw [hd]
    exact ((A.hasFDerivAt.const_mul 2).smul_const A).fderiv
  refine ⟨φ, C, A.contDiff.pow 2, ?_, ?_, ?_⟩
  · intro k hk z
    have hC : 0 ≤ C := by dsimp only [C]; positivity
    have hn : 0 ≤ ‖z‖ := norm_nonneg z
    interval_cases k
    · rw [norm_iteratedFDeriv_zero]
      change ‖(A z) ^ 2‖ ≤ C * (1 + ‖z‖) ^ 2
      rw [norm_pow]
      calc
        _ ≤ (‖A‖ * ‖z‖) ^ 2 := pow_le_pow_left₀ (norm_nonneg _) (A.le_opNorm z) 2
        _ = ‖A‖ ^ 2 * ‖z‖ ^ 2 := mul_pow _ _ _
        _ ≤ ‖A‖ ^ 2 * (2 * (1 + ‖z‖) ^ 2) := mul_le_mul_of_nonneg_left
          (by nlinarith) (sq_nonneg ‖A‖)
        _ = _ := by dsimp only [C]; ring
    · rw [norm_iteratedFDeriv_one, hd, norm_smul, norm_mul]
      norm_num
      have he : ‖A z‖ ≤ ‖A‖ * (1 + ‖z‖) ^ 2 := (A.le_opNorm z).trans
        (mul_le_mul_of_nonneg_left (by nlinarith) (norm_nonneg A))
      calc
        _ = ‖A z‖ * (2 * ‖A‖) := by simp only [Real.norm_eq_abs]; ring
        _ ≤ (‖A‖ * (1 + ‖z‖) ^ 2) * (2 * ‖A‖) :=
          mul_le_mul_of_nonneg_right he (by positivity)
        _ = _ := by simp only [C]; ring
    · rw [← norm_iteratedFDeriv_fderiv (n := 1), norm_iteratedFDeriv_one, hd2,
        ContinuousLinearMap.norm_smulRight_apply, norm_smul]
      norm_num
      have h := mul_le_mul_of_nonneg_left (show 1 ≤ (1 + ‖z‖) ^ 2 by nlinarith) hC
      simpa only [C, mul_one, pow_two, mul_assoc] using h
  · intro r
    simp [φ, A]
  · rw [iteratedFDeriv_two_apply, hd2]
    simp [A]

/-- The actual quadratic test jointly witnesses every polynomial
bound and regularity hypothesis with an unbounded test and gradient. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ), ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 :=
  gaussian_interp_polynomial_test_witness

end Transformer.Homogenized
