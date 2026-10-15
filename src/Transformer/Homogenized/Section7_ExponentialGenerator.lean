/-
# The actual generator of the exponential overlap

ArXiv:2604.01978v1, §7, lem:Ito_formula and the maximal
inequality after eq:Gronwall_E_selfcontained.
The proved sphere Hessian chain rule gives
L exp(cR)=exp(cR)(cD+c²q/2), with the original generator,
original overlap drift and original noise second moment q.

Every Gaussian expectation in this identity is proved integrable.
At the source's standard value variance, the actual bound q≤3/d
therefore gives the required exponential generator inequality.
No exponential supermartingale or tail bound is assumed here.
The example has positive value variance and generator value 1/4.
-/

import Transformer.Homogenized.Section7_ExponentialObservable
import Transformer.Homogenized.Section7_OverlapNoiseBounds

open scoped BigOperators NNReal
open MeasureTheory

namespace Transformer.Homogenized

/-- The actual exponential chain rule for the original spherical generator.
Source: arXiv:2604.01978v1, §7, def:ito.formula and the maximal inequality.
The two needed head expectations are explicit integrability premises. -/
theorem sphGenerator_exponentialOverlap {d n : ℕ} (c : ℝ) (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    (X : Idx n → EucSpace d) (i j : Idx n)
    (hH : Integrable (fun θ => sphHess
      (fun Y : Idx n → EucSpace d => inner (𝕜 := ℝ) (Y i) (Y j)) X (G θ X)) ρ)
    (hQ : Integrable (fun θ => (overlapForm i j X (G θ X)) ^ 2) ρ) :
    sphGenerator ρ B G (exponentialOverlap c i j) X = exponentialOverlap c i j X *
      (c * sphGenerator ρ B G (fun Y : Idx n → EucSpace d => inner (𝕜 := ℝ) (Y i) (Y j)) X +
        c ^ 2 / 2 * ∫ θ, (overlapForm i j X (G θ X)) ^ 2 ∂ρ) := by
  rw [sphGenerator, fderiv_exponentialOverlap]
  simp only [smul_apply, smul_eq_mul]
  simp_rw [sphHess_exponentialOverlap]
  rw [integral_const_mul, integral_add (hH.const_mul c) (hQ.const_mul (c ^ 2)),
    integral_const_mul, integral_const_mul, sphGenerator, (hasFDerivAt_overlap i j X).fderiv]
  ring

/-- The original overlap Hessian is integrable along the original Gaussian
noise. Source: arXiv:2604.01978v1, §7, lem:Ito_formula. -/
theorem integrable_sphHess_overlap_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    Integrable (fun θ => sphHess
      (fun Y : Idx n → EucSpace d => inner (𝕜 := ℝ) (Y i) (Y j)) X (noiseField β 1 1 ρ θ X)) ρ := by
  have hij := integrable_inner_Gfield_cross_gaussian β hρ hX i j
  have hi : Integrable (fun θ => ‖Gfield β ρ X θ i‖ ^ 2) ρ := by
    simpa only [real_inner_self_eq_norm_sq] using integrable_inner_Gfield_cross_gaussian β hρ hX i i
  have hj : Integrable (fun θ => ‖Gfield β ρ X θ j‖ ^ 2) ρ := by
    simpa only [real_inner_self_eq_norm_sq] using integrable_inner_Gfield_cross_gaussian β hρ hX j j
  have hsum : Integrable (fun θ => ‖Gfield β ρ X θ i‖ ^ 2 + ‖Gfield β ρ X θ j‖ ^ 2) ρ := hi.add hj
  have hfinal : Integrable (fun θ => 2 * inner (𝕜 := ℝ) (Gfield β ρ X θ i) (Gfield β ρ X θ j) -
      inner (𝕜 := ℝ) (X i) (X j) * (‖Gfield β ρ X θ i‖ ^ 2 + ‖Gfield β ρ X θ j‖ ^ 2)) ρ :=
    (hij.const_mul 2).sub (hsum.const_mul _)
  simpa only [sphHess_overlap, noiseField_one_one] using hfinal

/-- The derivative-square term of the original generator is genuinely
integrable. Source: arXiv:2604.01978v1, §7, eq:martingale_Mij_clean. -/
theorem integrable_overlapForm_noise_sq_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    Integrable (fun θ => (overlapForm i j X (noiseField β 1 1 ρ θ X)) ^ 2) ρ := by
  simpa only [overlapForm_apply, noiseField_one_one, overlapNoiseIntegrand] using
    integrable_overlapNoiseIntegrand_sq_gaussian β hρ X hX i j

/-- The exponential test's actual sphere Hessian is integrable at Gaussian
heads. Source: arXiv:2604.01978v1, §7, def:ito.formula and the maximal inequality. -/
theorem integrable_sphHess_exponentialOverlap_gaussian {d n : ℕ} (c β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    Integrable (fun θ => sphHess (exponentialOverlap c i j) X (noiseField β 1 1 ρ θ X)) ρ := by
  have hH := integrable_sphHess_overlap_gaussian β hρ X hX i j
  have hQ := integrable_overlapForm_noise_sq_gaussian β hρ X hX i j
  have hsum : Integrable (fun θ => c * sphHess
      (fun Y : Idx n → EucSpace d => inner (𝕜 := ℝ) (Y i) (Y j)) X (noiseField β 1 1 ρ θ X) +
      c ^ 2 * (overlapForm i j X (noiseField β 1 1 ρ θ X)) ^ 2) ρ :=
    (hH.const_mul c).add (hQ.const_mul _)
  simp_rw [sphHess_exponentialOverlap]
  exact hsum.const_mul _

/-- At the original Gaussian law and standard value variance the original
exponential generator is exp(cR)(cD+c²q/2), with its actual noise density q.
Source: arXiv:2604.01978v1, §7, lem:Ito_formula and the maximal inequality. -/
theorem sphGenerator_exponentialOverlap_gaussian {d n : ℕ} (c β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) X =
      exponentialOverlap c i j X * (c * overlapDrift β ρ X i j +
        c ^ 2 / 2 * ∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) := by
  rw [sphGenerator_exponentialOverlap c ρ (bField β ρ) (noiseField β 1 1 ρ) X i j
    (integrable_sphHess_overlap_gaussian β hρ X hX i j)
    (integrable_overlapForm_noise_sq_gaussian β hρ X hX i j),
    ito_formula_overlap β σV σA ρ hρ hσ X hX i j]
  simp only [overlapForm_apply, noiseField_one_one, overlapNoiseIntegrand]

/-- The actual noise second-moment integral is nonnegative at every law.
Source: arXiv:2604.01978v1, §7, the original overlap bracket density. -/
theorem integral_overlapNoiseIntegrand_sq_nonneg {d n : ℕ} (β : ℝ)
    (ρ : Measure (HeadParam d)) (X : Idx n → EucSpace d) (i j : Idx n) :
    0 ≤ ∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ := by
  exact integral_nonneg fun θ => sq_nonneg _

/-- The proved actual noise bound gives the exponential generator upper
bound with coefficient 3/d, for either sign of c.
Source: arXiv:2604.01978v1, §7, the maximal inequality after the bracket bound. -/
theorem sphGenerator_exponentialOverlap_le {d n : ℕ} (c β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) X ≤
      exponentialOverlap c i j X * (c * overlapDrift β ρ X i j + c ^ 2 / 2 * (3 / (d : ℝ))) := by
  rw [sphGenerator_exponentialOverlap_gaussian c β hρ hσ X hX]
  apply mul_le_mul_of_nonneg_left _ (exponentialOverlap_pos c i j X).le
  have hc : 0 ≤ c ^ 2 / 2 := div_nonneg (sq_nonneg c) (by norm_num)
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (integral_overlapNoiseIntegrand_sq_le_three_div_dim β hρ hσ X hX i j) hc)

/-- Original positive value variance, actual unit basis tokens and standard
scaling have generator 1/4 for the exponential test, with every required
original Hessian and derivative-square expectation integrable. -/
example : ∃ X : Idx 2 → EucSpace 2,
    IsGaussianHeadLaw 2 (stdSigmaV 2) 0 (gaussHeadLaw 2 (stdSigmaV 2) 0) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ (∀ k, ‖X k‖ = 1) ∧
    Integrable (fun θ => sphHess
      (fun Y : Idx 2 → EucSpace 2 => inner (𝕜 := ℝ) (Y 0) (Y 1)) X
        (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 0) θ X)) (gaussHeadLaw 2 (stdSigmaV 2) 0) ∧
    Integrable (fun θ => (overlapForm 0 1 X
      (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 0) θ X)) ^ 2) (gaussHeadLaw 2 (stdSigmaV 2) 0) ∧
    sphGenerator (gaussHeadLaw 2 (stdSigmaV 2) 0) (bField 1 (gaussHeadLaw 2 (stdSigmaV 2) 0))
      (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 0)) (exponentialOverlap 1 0 1) X = 1 / 4 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 0
  have hX : ∀ k, ‖X k‖ = 1 := fun k => by simp [X]
  refine ⟨X, hρ, stdSigmaV_sq 2, hX,
    integrable_sphHess_overlap_gaussian 1 hρ X hX 0 1,
    integrable_overlapForm_noise_sq_gaussian 1 hρ X hX 0 1, ?_⟩
  rw [sphGenerator_exponentialOverlap_gaussian 1 1 hρ (stdSigmaV_sq 2) X hX]
  have hnoise : (∫ θ, overlapNoiseIntegrand 1 (gaussHeadLaw 2 (stdSigmaV 2) 0) X 0 1 θ ^ 2
      ∂(gaussHeadLaw 2 (stdSigmaV 2) 0)) = 1 / 2 := by
    rw [show X = (fun k : Idx 2 => (EuclideanSpace.single k 1 : EucSpace 2)) from rfl,
      integral_overlapNoiseIntegrand_sq_basis_zero_attention, stdSigmaV_sq]
    norm_num
  rw [hnoise]
  norm_num [exponentialOverlap, overlapDrift, X, EuclideanSpace.inner_single_left, PiLp.single_apply]

end Transformer.Homogenized
