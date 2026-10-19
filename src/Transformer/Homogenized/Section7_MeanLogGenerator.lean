module

public import Transformer.Homogenized.Section7_MeanLogCalculus
public import Transformer.Homogenized.Section7_SmallTemperatureMeanDrift

/-!
# The actual Gaussian generator and noise of the regularized log mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:m_decomp_clean` and the mean bracket calculation after
`eq:gronwall_e_clean`. The auxiliary logarithm is applied to the
original nonnegative empirical mean, with positive regularization.

The genuine sphere Hessian gives its original Gaussian generator
as `g/(m+a) - q/(2(m+a)²)`, where `g` is the actual original mean
drift and `q` is its actual noise second moment. Every original
Gaussian Hessian and noise-square expectation is proved integrable
before splitting its integral.

The logarithm's noise is its actual derivative on the original noise
field. Its second moment is exactly `q/(m+a)²`, retaining the genuine
regularization and the original Gaussian query/value law. The full
adapted continuous sphere martingale problem supplies the selected
logarithmic test identity; it is not replaced by an assumed tail bound.
-/

@[expose] public section

open Real MeasureTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The genuine logarithmic generator keeps its negative quadratic
correction. Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`
and Appendix, `def:ito.formula`. The two actual expectations are
explicit integrability hypotheses for this arbitrary head law. -/
theorem sphGenerator_logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (ρ : Measure (HeadParam d)) (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d) (X : Idx n → EucSpace d)
    (hH : Integrable (fun θ => sphHess meanOverlap X (G θ X)) ρ)
    (hQ : Integrable (fun θ => (meanOverlapForm X (G θ X)) ^ 2) ρ) :
    sphGenerator ρ B G (logMeanOverlap a) X =
      (meanOverlap X + a)⁻¹ * sphGenerator ρ B G meanOverlap X -
        ((meanOverlap X + a) ^ 2)⁻¹ / 2 * ∫ θ, (meanOverlapForm X (G θ X)) ^ 2 ∂ρ := by
  rw [sphGenerator, fderiv_logMeanOverlap ha]
  simp only [smul_apply, smul_eq_mul]
  simp_rw [sphHess_logMeanOverlap ha]
  rw [integral_sub (hH.const_mul _) (hQ.const_mul _), integral_const_mul, integral_const_mul,
    sphGenerator, fderiv_meanOverlap]
  ring

/-- The actual original mean differential squared is integrable
along the genuine Gaussian noise field.
Source: arXiv:2604.01978v1, §7, the original mean bracket integrand. -/
theorem integrable_meanOverlapForm_noise_sq_gaussian {d n : ℕ} (hn : 0 < n) (β : ℝ)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    Integrable (fun θ => (meanOverlapForm X (noiseField β 1 1 ρ θ X)) ^ 2) ρ :=
  integrable_meanOverlapNoiseIntegrand_sq_gaussian hn β hρ X hX

/-- The genuine logarithmic sphere Hessian has an integrable
original Gaussian expectation. Source: arXiv:2604.01978v1, §7,
the original Itô generator and its mean bracket calculation. -/
theorem integrable_sphHess_logMeanOverlap_gaussian {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    Integrable (fun θ => sphHess (logMeanOverlap a) X (noiseField β 1 1 ρ θ X)) ρ := by
  simp_rw [sphHess_logMeanOverlap ha]
  exact ((integrable_sphHess_meanOverlap_gaussian β hρ X hX).const_mul _).sub
    ((integrable_meanOverlapForm_noise_sq_gaussian hn β hρ X hX).const_mul _)

/-- **The genuine original Gaussian log-mean generator is its
actual logarithmic drift, including the true quadratic correction.**
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean` and its
bracket calculation, at §2.3.3's adopted value variance `1/d`. -/
theorem sphGenerator_logMeanOverlap_gaussian {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X =
      (meanOverlap X + a)⁻¹ * meanOverlapDrift β ρ X -
        ((meanOverlap X + a) ^ 2)⁻¹ / 2 * ∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ := by
  rw [sphGenerator_logMeanOverlap ha ρ (bField β ρ) (noiseField β 1 1 ρ) X
    (integrable_sphHess_meanOverlap_gaussian β hρ X hX)
    (integrable_meanOverlapForm_noise_sq_gaussian hn β hρ X hX),
    sphGenerator_meanOverlap_gaussian β hρ hσ X hX]
  rfl

/-- The genuine logarithmic mean differential applied to the original
Gaussian noise field. Source: arXiv:2604.01978v1, §7, the original
mean bracket integrand and its logarithmic transformation. -/
noncomputable def logMeanOverlapNoiseIntegrand {d n : ℕ} (a β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) (θ : HeadParam d) : ℝ :=
  fderiv ℝ (logMeanOverlap a) X (noiseField β 1 1 ρ θ X)

/-- The true logarithmic noise is the true original mean noise
divided by its positive regularized mean.
Source: arXiv:2604.01978v1, §7, the original mean bracket calculation. -/
theorem logMeanOverlapNoiseIntegrand_eq {d n : ℕ} {a : ℝ} (ha : 0 < a) (β : ℝ)
    (ρ : Measure (HeadParam d)) (X : Idx n → EucSpace d) (θ : HeadParam d) :
    logMeanOverlapNoiseIntegrand a β ρ X θ = (meanOverlap X + a)⁻¹ * meanOverlapNoiseIntegrand β ρ X θ := by
  rw [logMeanOverlapNoiseIntegrand, fderiv_logMeanOverlap ha, smul_apply, smul_eq_mul]
  rfl

/-- The genuine logarithmic noise has a finite second moment under
the actual original Gaussian head law.
Source: arXiv:2604.01978v1, §7, the original mean-noise variance estimate. -/
theorem integrable_logMeanOverlapNoiseIntegrand_sq_gaussian {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    Integrable (fun θ => logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2) ρ := by
  simp_rw [logMeanOverlapNoiseIntegrand_eq ha, mul_pow]
  exact (integrable_meanOverlapNoiseIntegrand_sq_gaussian hn β hρ X hX).const_mul _

/-- The actual logarithmic noise second moment is exactly the
original moment divided by the square of the positive regularized mean.
Source: arXiv:2604.01978v1, §7, the original mean bracket and Itô's formula. -/
theorem integral_logMeanOverlapNoiseIntegrand_sq {d n : ℕ} {a : ℝ} (ha : 0 < a) (β : ℝ)
    (ρ : Measure (HeadParam d)) (X : Idx n → EucSpace d) :
    (∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ) =
      ((meanOverlap X + a) ^ 2)⁻¹ * ∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ := by
  simp_rw [logMeanOverlapNoiseIntegrand_eq ha, mul_pow, inv_pow]
  exact integral_const_mul _ _

/-- The full actual sphere martingale problem supplies the genuine
compensated logarithmic mean test.
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean` and Appendix, `def:ito.formula`. -/
theorem IsSphereMartingaleSolution.logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a)
    {ρ : Measure (HeadParam d)} {B : (Idx n → EucSpace d) → Idx n → EucSpace d}
    {G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d}
    {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ m}
    {T : ℝ} {X : ℝ → Ω → (Idx n → EucSpace d)}
    (h : IsSphereMartingaleSolution ρ B G P ℱ T X) :
    IsMartingaleOn T ℱ P (fun t ω => logMeanOverlap a (X t ω) - logMeanOverlap a (X 0 ω) -
      ∫ s in (0 : ℝ)..t, sphGenerator ρ B G (logMeanOverlap a) (X s ω)) :=
  h.2.2.2 _ (contDiff_logMeanOverlap ha)

/-- Positive regularization, nonempty unit queries and the original
Gaussian law jointly realize every actual generator integrability premise. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℝ) < 1 ∧ (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧
    Integrable (fun θ => sphHess meanOverlap X (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) θ X))
      (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    Integrable (fun θ => (meanOverlapForm X (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) θ X)) ^ 2)
      (gaussHeadLaw 2 (stdSigmaV 2) 1) := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hX : ∀ k, ‖X k‖ = 1 := fun k => by simp [X]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 1
  exact ⟨X, one_pos, by decide, hX, hρ, stdSigmaV_sq 2,
    integrable_sphHess_meanOverlap_gaussian 0 hρ X hX,
    integrable_meanOverlapForm_noise_sq_gaussian (by decide) 0 hρ X hX⟩

/-- Positive regularization and a genuine full-test sphere solution
jointly satisfy the logarithmic martingale selector's hypotheses. -/
example : ∃ ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit), (0 : ℝ) < 1 ∧
    IsSphereMartingaleSolution (Measure.dirac (0 : HeadParam 2))
      (bField 1 (Measure.dirac 0)) (noiseField 1 1 1 (Measure.dirac 0)) (Measure.dirac ()) ℱ 1
      (fun _ _ => (fun i : Idx 2 => EuclideanSpace.single i (1 : ℝ))) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  exact ⟨ℱ, one_pos, isSphereMartingaleSolution_dirac_zero (Measure.dirac ()) ℱ 1 1 1 1 _ (fun i => by simp)⟩

end Transformer.Homogenized
