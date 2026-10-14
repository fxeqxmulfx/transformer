/-
# Differentiation of the original polynomial Gaussian expectation

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:gaussian_interp_0`. On the original interpolation interval, the actual
blended vector is bounded by twice the norm of the original pair. On an
explicit neighborhood of an interior time, the actual path velocity has
a fixed bound in that same pair. Polynomial derivative growth therefore
gives an explicit locally uniform polynomial bound of one higher degree.

Actual moments of the original independent Gaussian pair make this
dominator integrable. The original chain rule and genuine dominated
differentiation then differentiate the original test expectation.
The derivative is the expectation of the actual original path derivative;
no derivative-under-integral identity is an input. Singular original
Gaussian laws and every polynomial degree are included.
The source's bare `C²` interpolation statement is supplemented by
polynomial growth conditions on the original derivatives, as used
in its actual bilinear softmax application.
This analytic step precedes the covariance contraction in the Gram-stability proof.
-/

import Transformer.Homogenized.Section7_GaussianPolynomialContinuity

open Real MeasureTheory ProbabilityTheory Filter Set
open scoped Topology

namespace Transformer.Homogenized

/-- The actual interpolation point is controlled by its original
pair on the original interval. Source: arXiv:2604.01978v1, §7,
the square-root blend in `eq:interpolation_gaussian`. -/
theorem norm_gaussianInterpolationPoint_pair_le {m : ℕ} (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1)
    (x y : EucSpace m) : ‖gaussianInterpolationPoint t x y‖ ≤ 2 * ‖(x, y)‖ := by
  have hx : Real.sqrt t ≤ 1 := Real.sqrt_le_one.mpr ht.2
  have hy : Real.sqrt (1 - t) ≤ 1 := Real.sqrt_le_one.mpr (by linarith [ht.1])
  have h := norm_add_le (Real.sqrt t • x) (Real.sqrt (1 - t) • y)
  simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg (Real.sqrt_nonneg _)] at h
  apply h.trans
  calc
    _ ≤ ‖x‖ + ‖y‖ := add_le_add
      (by simpa only [one_mul] using mul_le_mul_of_nonneg_right hx (norm_nonneg x))
      (by simpa only [one_mul] using mul_le_mul_of_nonneg_right hy (norm_nonneg y))
    _ ≤ _ := by linarith [norm_fst_le (x, y), norm_snd_le (x, y)]

/-- The actual scalar path derivative has a locally uniform
polynomial dominator in the actual original pair.
Source: arXiv:2604.01978v1, §7, domination before the Gaussian
integration-by-parts step of `lem:gaussian_interp`. -/
theorem norm_gaussianInterpolation_test_derivative_local_polynomial_le {m : ℕ}
    (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) (r : ℝ) (hr : r ∈ Ioo (t / 2) ((t + 1) / 2))
    (x y : EucSpace m) :
    ‖fderiv ℝ φ (gaussianInterpolationPoint r x y) (gaussianInterpolationVelocity r x y)‖ ≤
      (C * 2 ^ p * ((1 / (2 * Real.sqrt (t / 2))) +
        (1 / (2 * Real.sqrt ((1 - t) / 2))))) * (1 + ‖(x, y)‖) ^ (p + 1) := by
  have hC : 0 ≤ C := gaussian_test_polynomial_bound_nonneg φ C p hbdd
  have hrI : r ∈ Icc (0 : ℝ) 1 := ⟨by linarith [ht.1, hr.1], by linarith [ht.2, hr.2]⟩
  have hp : (1 + ‖gaussianInterpolationPoint r x y‖) ^ p ≤ 2 ^ p * (1 + ‖(x, y)‖) ^ p := by
    rw [← mul_pow]
    apply pow_le_pow_left₀ (by positivity)
    linarith [norm_gaussianInterpolationPoint_pair_le r hrI x y]
  have hg : ‖fderiv ℝ φ (gaussianInterpolationPoint r x y)‖ ≤
      (C * 2 ^ p) * (1 + ‖(x, y)‖) ^ p :=
    (norm_gaussian_test_fderiv_polynomial_le φ C p hbdd _).trans
      (by simpa only [mul_assoc] using mul_le_mul_of_nonneg_left hp hC)
  let u := 1 / (2 * Real.sqrt (t / 2))
  let v := 1 / (2 * Real.sqrt ((1 - t) / 2))
  have hu : 0 ≤ u := by dsimp only [u]; positivity
  have hv : 0 ≤ v := by dsimp only [v]; positivity
  have hvel : ‖gaussianInterpolationVelocity r x y‖ ≤ (u + v) * (1 + ‖(x, y)‖) := by
    calc
      _ ≤ u * ‖x‖ + v * ‖y‖ := norm_gaussianInterpolationVelocity_local_le t ht r hr x y
      _ ≤ u * ‖(x, y)‖ + v * ‖(x, y)‖ := add_le_add
        (mul_le_mul_of_nonneg_left (norm_fst_le (x, y)) hu)
        (mul_le_mul_of_nonneg_left (norm_snd_le (x, y)) hv)
      _ = (u + v) * ‖(x, y)‖ := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (by linarith) (add_nonneg hu hv)
  calc
    _ ≤ ‖fderiv ℝ φ (gaussianInterpolationPoint r x y)‖ * ‖gaussianInterpolationVelocity r x y‖ :=
      (fderiv ℝ φ _).le_opNorm _
    _ ≤ ((C * 2 ^ p) * (1 + ‖(x, y)‖) ^ p) * ((u + v) * (1 + ‖(x, y)‖)) :=
      mul_le_mul hg hvel (norm_nonneg _) (by positivity)
    _ = _ := by rw [pow_succ]; ring

/-- The original polynomial Gaussian expectation has the derivative
given by its actual original scalar path derivative expectation.
Source: arXiv:2604.01978v1, §7, the actual differentiation-under-
expectation step in `lem:gaussian_interp` and `eq:gaussian_interp_0`. -/
theorem gaussianInterpolation_differentiate_gaussian_integral_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) :
    Integrable (fun ω => fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
      (gaussianInterpolationVelocity t (X ω) (Y ω))) P ∧
    HasDerivAt (fun r : ℝ => ∫ ω, φ (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P)
      (∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
        (gaussianInterpolationVelocity t (X ω) (Y ω)) ∂P) t := by
  have hs : Ioo (t / 2) ((t + 1) / 2) ∈ 𝓝 t :=
    isOpen_Ioo.mem_nhds ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have hmeas (r : ℝ) := aestronglyMeasurable_gaussianInterpolation_test P X Y
    hX.aemeasurable hY.aemeasurable φ hφ.continuous r
  have htestint := integrable_gaussianInterpolation_test_polynomial P X Y hX hY hindep
    φ hφ.continuous C p (norm_gaussian_test_polynomial_le φ C p hbdd) t
  have hderivmeas := aestronglyMeasurable_gaussianInterpolation_test_derivative P X Y
    hX.aemeasurable hY.aemeasurable φ hφ t
  have hXY := hasGaussianLaw_independent_gaussian_pair P X Y hX hY hindep
  have hdom := (integrable_gaussian_polynomial_envelope P _ hXY (p + 1)).const_mul
    (C * 2 ^ p * ((1 / (2 * Real.sqrt (t / 2))) + (1 / (2 * Real.sqrt ((1 - t) / 2)))))
  have hbound : ∀ᵐ ω ∂P, ∀ r ∈ Ioo (t / 2) ((t + 1) / 2),
      ‖fderiv ℝ φ (gaussianInterpolationPoint r (X ω) (Y ω))
        (gaussianInterpolationVelocity r (X ω) (Y ω))‖ ≤
      (C * 2 ^ p * ((1 / (2 * Real.sqrt (t / 2))) +
        (1 / (2 * Real.sqrt ((1 - t) / 2))))) * (1 + ‖(X ω, Y ω)‖) ^ (p + 1) :=
    ae_of_all _ fun ω r hr => norm_gaussianInterpolation_test_derivative_local_polynomial_le φ C p hbdd t ht r hr _ _
  have hdiff : ∀ᵐ ω ∂P, ∀ r ∈ Ioo (t / 2) ((t + 1) / 2),
      HasDerivAt (fun q => φ (gaussianInterpolationPoint q (X ω) (Y ω)))
        (fderiv ℝ φ (gaussianInterpolationPoint r (X ω) (Y ω))
          (gaussianInterpolationVelocity r (X ω) (Y ω))) r := by
    apply ae_of_all
    intro ω r hr
    apply hasDerivAt_gaussianInterpolation_test φ hφ (X ω) (Y ω) r
    exact ⟨by linarith [ht.1, hr.1], by linarith [ht.2, hr.2]⟩
  exact hasDerivAt_integral_of_dominated_loc_of_deriv_le hs (Eventually.of_forall hmeas)
    htestint hderivmeas hbound hdom hdiff

/-- A genuine quadratic, an original nondegenerate Gaussian and
independent zero Gaussian satisfy the original polynomial differentiation
hypotheses at an actual interior time and its actual local neighborhood. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    HasGaussianLaw (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    ContDiff ℝ 2 φ ∧ (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    (1 / 2 : ℝ) ∈ Ioo ((1 / 2 : ℝ) / 2) (((1 / 2 : ℝ) + 1) / 2) ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 := by
  obtain ⟨φ, C, hφ, hbdd, hline, hhess⟩ := gaussian_interp_polynomial_test_witness
  refine ⟨φ, C, IsGaussian.hasGaussianLaw_id, ?_, indepFun_const_right _ _, hφ, hbdd,
    by norm_num, by norm_num, by norm_num, hline, hhess⟩
  simpa only [Function.comp_def, zero_apply] using
    (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace 2))).map
      (0 : EucSpace 2 →L[ℝ] EucSpace 2)

end Transformer.Homogenized
