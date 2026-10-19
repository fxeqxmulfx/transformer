module

public import Transformer.Homogenized.GaussianInterp
public import Transformer.Homogenized.Section7_GaussianPolynomialCovariance
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Gaussian interpolation with polynomial growth of the actual derivatives

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:gaussian_interp_0`. The source states interpolation for every `C²`
test, without growth control; that does not ensure finite expectations.
This corrected version allows polynomial growth of the actual test and
its actual derivatives through order two. It includes the type of growth
used for the source's bilinear softmax test after `eq:interpolation_interm`.

Both actual original Gaussian laws and actual measurability are carried
by `HasGaussianLaw`. The covariance entries are the original coordinate
second moments, with actual centering explicit. Genuine density integration
by parts, polynomial Gaussian moments, Fubini and dominated differentiation
prove the identity for the original expectation on its original space.
Continuity of that expectation and of its actual covariance trace gives
the one-sided derivative at both endpoints. Singular laws are included.

The quantitative consequence bounds the actual expectation difference
by a bound on the actual covariance trace along the original interpolation.
The source uses this analytic step after its specific softmax Hessian
estimates to obtain `eq:Lispchitz_bound_softmax`; those estimates remain
separate inputs to this general consequence.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Set
open scoped BigOperators

namespace Transformer.Homogenized

/-- Every original expectation appearing in polynomial interpolation
is finite. Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
The actual polynomial growth conditions justify both original endpoint
test expectations and the original covariance trace at every time. -/
theorem gaussian_interp_polynomial_integrable {m : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (S : Matrix (Fin m) (Fin m) ℝ) (t : ℝ) :
    Integrable (fun ω => φ (X ω)) P ∧ Integrable (fun ω => φ (Y ω)) P ∧
      Integrable (fun ω => gaussianCovarianceTrace φ S (gaussInterp X Y t ω)) P := by
  have hb := norm_gaussian_test_polynomial_le φ C p hbdd
  refine ⟨?_, ?_, integrable_gaussianInterpolation_covarianceTrace_polynomial P X Y hX hY hindep
    φ hφ C p hbdd S t⟩
  · exact integrable_of_gaussian_polynomial_bound P X hX _
      (hφ.continuous.aestronglyMeasurable.comp_aemeasurable hX.aemeasurable) C p
      (ae_of_all _ fun ω => hb (X ω))
  · exact integrable_of_gaussian_polynomial_bound P Y hY _
      (hφ.continuous.aestronglyMeasurable.comp_aemeasurable hY.aemeasurable) C p
      (ae_of_all _ fun ω => hb (Y ω))

/-- Gaussian interpolation for the actual original `C²` test with
polynomial growth through order two, including both one-sided endpoint
derivatives and singular original covariance matrices.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
The source's unrestricted `C²` wording is corrected by polynomial growth
and actual measurability, retained in the original `HasGaussianLaw` inputs. -/
theorem gaussian_interp_polynomial {m : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (SX SY : Matrix (Fin m) (Fin m) ℝ)
    (hXc : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0) (hYc : ∀ i : Fin m, ∫ ω, Y ω i ∂P = 0)
    (hXcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = SX i j)
    (hYcov : ∀ i j : Fin m, ∫ ω, Y ω i * Y ω j ∂P = SY i j) (hindep : IndepFun X Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    HasDerivWithinAt (fun r => ∫ ω, φ (gaussInterp X Y r ω) ∂P)
      ((1 / 2 : ℝ) * ∫ ω, ∑ i : Fin m, ∑ j : Fin m, (SX i j - SY i j) *
        iteratedFDeriv ℝ 2 φ (gaussInterp X Y t ω)
          ![EuclideanSpace.single i 1, EuclideanSpace.single j 1] ∂P) (Icc 0 1) t := by
  have hf := continuous_expectation_gaussian_blend_polynomial P X Y hX hY hindep
    φ hφ.continuous C p (norm_gaussian_test_polynomial_le φ C p hbdd)
  have hg := (continuous_expectation_gaussianCovarianceTrace_polynomial P X Y hX hY hindep
    φ hφ C p hbdd (SX - SY)).const_mul (1 / 2 : ℝ)
  have hd (r : ℝ) (hr : r ∈ Ioo (0 : ℝ) 1) :
      HasDerivAt (fun q : ℝ => ∫ ω, φ (gaussianInterpolationPoint q (X ω) (Y ω)) ∂P)
        ((1 / 2 : ℝ) * ∫ ω, gaussianCovarianceTrace φ (SX - SY)
          (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P) r := by
    have h := (gaussianInterpolation_differentiate_gaussian_integral_polynomial P X Y hX hY hindep
      φ hφ C p hbdd r hr).2
    rw [integral_gaussianInterpolation_velocity_covariance_polynomial P X Y hX hY SX SY hXc hYc
      hXcov hYcov hindep φ hφ C p hbdd r hr] at h
    exact h
  have h := interpolation_hasDerivWithinAt_endpoints _ _ hf hg hd t ht
  simpa only [gaussInterp, gaussianInterpolationPoint, gaussianCovarianceTrace, Matrix.sub_apply] using h

/-- Bounding the actual covariance trace bounds the difference of
the original test expectations. Source: arXiv:2604.01978v1, §7,
integration of `eq:steins_lemma` to obtain `eq:Lispchitz_bound_softmax`.
This general analytic consequence keeps the actual trace estimate as
an explicit input; the original interpolation identity is proved above. -/
theorem gaussian_interp_expectation_difference_le {m : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (SX SY : Matrix (Fin m) (Fin m) ℝ)
    (hXc : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0) (hYc : ∀ i : Fin m, ∫ ω, Y ω i ∂P = 0)
    (hXcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = SX i j)
    (hYcov : ∀ i j : Fin m, ∫ ω, Y ω i * Y ω j ∂P = SY i j) (hindep : IndepFun X Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (L : ℝ) (htrace : ∀ t ∈ Icc (0 : ℝ) 1,
      |(1 / 2 : ℝ) * ∫ ω, gaussianCovarianceTrace φ (SX - SY) (gaussInterp X Y t ω) ∂P| ≤ L) :
    |(∫ ω, φ (X ω) ∂P) - ∫ ω, φ (Y ω) ∂P| ≤ L := by
  have hd (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
      HasDerivWithinAt (fun r => ∫ ω, φ (gaussInterp X Y r ω) ∂P)
        ((1 / 2 : ℝ) * ∫ ω, gaussianCovarianceTrace φ (SX - SY) (gaussInterp X Y t ω) ∂P) (Icc 0 1) t := by
    simpa only [gaussianCovarianceTrace, Matrix.sub_apply] using gaussian_interp_polynomial P X Y hX hY
      SX SY hXc hYc hXcov hYcov hindep φ hφ C p hbdd t ht
  have h := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le hd
    (fun t ht => by simpa only [Real.norm_eq_abs] using htrace t ht)
    (convex_Icc (0 : ℝ) 1) (by norm_num : (0 : ℝ) ∈ Icc (0 : ℝ) 1)
    (by norm_num : (1 : ℝ) ∈ Icc (0 : ℝ) 1)
  simpa [gaussInterp, Real.norm_eq_abs] using h

/-- A genuine unbounded quadratic, original standard Gaussian and
independent zero Gaussian witness every original moment and polynomial
hypothesis at both actual interpolation endpoints. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    HasGaussianLaw (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) = (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    (∀ i : Fin 2, ∫ _z : EucSpace 2, (0 : EucSpace 2) i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ _z : EucSpace 2, (0 : EucSpace 2) i * (0 : EucSpace 2) j ∂stdGaussian (EucSpace 2) =
      (0 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    ContDiff ℝ 2 φ ∧ (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 := by
  obtain ⟨φ, C, hφ, hbdd, hline, hhess⟩ := gaussian_interp_polynomial_test_witness
  refine ⟨φ, C, IsGaussian.hasGaussianLaw_id, ?_, indepFun_const_right _ _, integral_stdGaussian_coordinate,
    ?_, fun i => by simp, fun i j => by simp, hφ, hbdd, by norm_num, by norm_num, hline, hhess⟩
  · simpa only [Function.comp_def, zero_apply] using
      (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace 2))).map (0 : EucSpace 2 →L[ℝ] EucSpace 2)
  · intro i j
    simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j

/-- Two actual independent zero Gaussian vectors and a genuine
quadratic satisfy the expectation-difference hypotheses with trace bound
zero, including the original centering and covariance premises. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (fun _ : Unit => (0 : EucSpace 2)) (Measure.dirac ()) ∧
    IndepFun (fun _ : Unit => (0 : EucSpace 2)) (fun _ : Unit => (0 : EucSpace 2)) (Measure.dirac ()) ∧
    (∀ i : Fin 2, ∫ _ω : Unit, (0 : EucSpace 2) i ∂Measure.dirac () = 0) ∧
    (∀ i j : Fin 2, ∫ _ω : Unit, (0 : EucSpace 2) i * (0 : EucSpace 2) j ∂Measure.dirac () =
      (0 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧ ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 ∧
    (∀ t : ℝ, |(1 / 2 : ℝ) * ∫ _ω : Unit,
      gaussianCovarianceTrace φ ((0 : Matrix (Fin 2) (Fin 2) ℝ) - 0)
        (gaussInterp (fun _ : Unit => (0 : EucSpace 2)) (fun _ : Unit => (0 : EucSpace 2)) t ())
      ∂Measure.dirac ()| ≤ 0) := by
  obtain ⟨φ, C, hφ, hbdd, hline, hhess⟩ := gaussian_interp_polynomial_test_witness
  refine ⟨φ, C, ?_, indepFun_const_left _ _, fun i => by simp, fun i j => by simp,
    hφ, hbdd, hline, hhess, fun t => ?_⟩
  · refine ⟨measurable_const.aemeasurable, ?_⟩
    rw [Measure.map_const]
    simp
    infer_instance
  · simp [gaussianCovarianceTrace]

end Transformer.Homogenized
