module

public import Transformer.Homogenized.Section7_GaussianPolynomialBlendGradient
public import Transformer.Homogenized.Section7_GaussianInterpolationCovariance

/-!
# Continuity of the original polynomial Gaussian interpolation expectations

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The original square-root blending map is
continuous in time. Its operator norm is bounded in a neighborhood of
each original time, including both endpoints of the interpolation interval.
Polynomial growth therefore gives a locally uniform integrable envelope
from genuine moments of the actual original independent Gaussian pair.

Dominated convergence proves continuity of the original test expectation.
Actual coordinate Hessian bounds give the same control on the original
covariance trace, so its original expectation is also continuous. These
are endpoint inputs for interpolation with polynomially growing derivatives,
without replacing the original variables or discarding singular laws.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter Set
open scoped BigOperators Topology

namespace Transformer.Homogenized

/-- The actual polynomial test at every original time is integrable.
Source: arXiv:2604.01978v1, §7, the original test expectation in
`lem:gaussian_interp`, justified by polynomial Gaussian moments. -/
theorem integrable_gaussianInterpolation_test_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (g : EucSpace m → ℝ) (hg : Continuous g) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p) (t : ℝ) :
    Integrable (fun ω => g (gaussianInterpolationPoint t (X ω) (Y ω))) P := by
  have hC : 0 ≤ C := (norm_nonneg (g 0)).trans (by simpa using hbound 0)
  let L := gaussianBlendLinearMap (m := m) (Real.sqrt t) (Real.sqrt (1 - t))
  exact integrable_gaussian_pair_test_polynomial P X Y hX hY hindep (g ∘ L)
    (hg.comp L.continuous) (C * (1 + ‖L‖) ^ p) p (fun z => by
      simpa only [norm_zero, add_zero, Function.comp_def] using
        norm_polynomial_gaussian_affine_le L 0 g C hC p hbound z)

/-- The original polynomial test expectation is continuous at every
original time, including interpolation endpoints. Source:
arXiv:2604.01978v1, §7, dominated convergence in `lem:gaussian_interp`. -/
theorem continuous_expectation_gaussian_blend_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (g : EucSpace m → ℝ) (hg : Continuous g) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p) :
    Continuous (fun t : ℝ => ∫ ω, g (gaussianInterpolationPoint t (X ω) (Y ω)) ∂P) := by
  have hC : 0 ≤ C := (norm_nonneg (g 0)).trans (by simpa using hbound 0)
  have hXY := hasGaussianLaw_independent_gaussian_pair P X Y hX hY hindep
  let L (t : ℝ) := gaussianBlendLinearMap (m := m) (Real.sqrt t) (Real.sqrt (1 - t))
  have hL : Continuous L := by unfold L gaussianBlendLinearMap; fun_prop
  apply continuous_iff_continuousAt.mpr
  intro t
  have hnear : ∀ᶠ r in 𝓝 t, ‖L r‖ < ‖L t‖ + 1 :=
    hL.norm.continuousAt.eventually (isOpen_Iio.mem_nhds (by linarith : ‖L t‖ < ‖L t‖ + 1))
  have hdom := (integrable_gaussian_polynomial_envelope P _ hXY p).const_mul
    (C * (2 + ‖L t‖) ^ p)
  have hbound : ∀ᶠ r in 𝓝 t, ∀ᵐ ω ∂P, ‖g (gaussianInterpolationPoint r (X ω) (Y ω))‖ ≤
      (C * (2 + ‖L t‖) ^ p) * (1 + ‖(X ω, Y ω)‖) ^ p := by
    filter_upwards [hnear] with r hr
    apply ae_of_all
    intro ω
    have h := norm_polynomial_gaussian_affine_le (L r) 0 g C hC p hbound (X ω, Y ω)
    simp only [norm_zero, add_zero] at h
    change ‖g (gaussianInterpolationPoint r (X ω) (Y ω))‖ ≤ _ at h
    apply h.trans
    apply mul_le_mul_of_nonneg_right _ (by positivity)
    exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (by positivity) (by linarith) p) hC
  have hcont : ∀ᵐ ω ∂P, ContinuousAt (fun r => g (gaussianInterpolationPoint r (X ω) (Y ω))) t :=
    ae_of_all _ fun ω => (hg.comp (continuous_gaussianInterpolationPoint (X ω) (Y ω))).continuousAt
  exact continuousAt_of_dominated
    (Eventually.of_forall fun r => aestronglyMeasurable_gaussianInterpolation_test P X Y
      hX.aemeasurable hY.aemeasurable g hg r) hbound hdom hcont

/-- Every actual coordinate Hessian entry along the original path has
a finite original expectation. Source: arXiv:2604.01978v1, §7,
the original Hessian contraction in `lem:gaussian_interp`. -/
theorem integrable_gaussianInterpolation_hessian_entry_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (t : ℝ) (i j : Fin m) :
    Integrable (fun ω => iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
      ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]) P :=
  integrable_gaussian_blend_hessian_entry_polynomial P X Y hX hY hindep φ hφ C p hbdd
    (Real.sqrt t) (Real.sqrt (1 - t)) i j

/-- The original covariance trace inherits the original polynomial
Hessian bound. Source: arXiv:2604.01978v1, §7, the genuine trace
expectation in `lem:gaussian_interp` and `eq:gaussian_interp_0`. -/
theorem norm_gaussianCovarianceTrace_polynomial_le {m : ℕ}
    (φ : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (S : Matrix (Fin m) (Fin m) ℝ) (z : EucSpace m) :
    ‖gaussianCovarianceTrace φ S z‖ ≤ ((∑ i : Fin m, ∑ j : Fin m, ‖S i j‖) * C) * (1 + ‖z‖) ^ p := by
  unfold gaussianCovarianceTrace
  calc
    _ ≤ ∑ i : Fin m, ‖∑ j : Fin m, S i j * iteratedFDeriv ℝ 2 φ z
        ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]‖ := norm_sum_le _ _
    _ ≤ ∑ i : Fin m, ∑ j : Fin m, ‖S i j‖ * (C * (1 + ‖z‖) ^ p) := by
      apply Finset.sum_le_sum
      intro i hi
      apply (norm_sum_le _ _).trans
      apply Finset.sum_le_sum
      intro j hj
      rw [norm_mul]
      exact mul_le_mul_of_nonneg_left (norm_gaussian_test_hessian_entry_polynomial_le φ C p hbdd i j z)
        (norm_nonneg (S i j))
    _ = _ := by simp only [Finset.sum_mul, mul_assoc]

/-- The original covariance trace at each original time is integrable.
Source: arXiv:2604.01978v1, §7, the original Hessian expectation
in `lem:gaussian_interp`, with polynomial growth domination. -/
theorem integrable_gaussianInterpolation_covarianceTrace_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (S : Matrix (Fin m) (Fin m) ℝ) (t : ℝ) :
    Integrable (fun ω => gaussianCovarianceTrace φ S (gaussianInterpolationPoint t (X ω) (Y ω))) P :=
  integrable_gaussianInterpolation_test_polynomial P X Y hX hY hindep (gaussianCovarianceTrace φ S)
    (continuous_gaussianCovarianceTrace φ hφ S) ((∑ i : Fin m, ∑ j : Fin m, ‖S i j‖) * C) p
    (norm_gaussianCovarianceTrace_polynomial_le φ C p hbdd S) t

/-- The original covariance trace expectation is continuous, including
both original endpoint values. Source: arXiv:2604.01978v1, §7,
the endpoint extension of the growth-controlled `lem:gaussian_interp`. -/
theorem continuous_expectation_gaussianCovarianceTrace_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X Y : Ω → EucSpace m) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hindep : IndepFun X Y P) (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (S : Matrix (Fin m) (Fin m) ℝ) :
    Continuous (fun t : ℝ => ∫ ω, gaussianCovarianceTrace φ S (gaussianInterpolationPoint t (X ω) (Y ω)) ∂P) :=
  continuous_expectation_gaussian_blend_polynomial P X Y hX hY hindep (gaussianCovarianceTrace φ S)
    (continuous_gaussianCovarianceTrace φ hφ S) ((∑ i : Fin m, ∑ j : Fin m, ‖S i j‖) * C) p
    (norm_gaussianCovarianceTrace_polynomial_le φ C p hbdd S)

/-- The original nondegenerate Gaussian, independent zero Gaussian
and genuine quadratic jointly witness the original measurability,
independence, growth and Hessian regularity hypotheses. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    HasGaussianLaw (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    ContDiff ℝ 2 φ ∧ (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 := by
  obtain ⟨φ, C, hφ, hbdd, hline, hhess⟩ := gaussian_interp_polynomial_test_witness
  refine ⟨φ, C, IsGaussian.hasGaussianLaw_id, ?_, indepFun_const_right _ _, hφ, hbdd, hline, hhess⟩
  simpa only [Function.comp_def, zero_apply] using
    (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace 2))).map
      (0 : EucSpace 2 →L[ℝ] EucSpace 2)

end Transformer.Homogenized
