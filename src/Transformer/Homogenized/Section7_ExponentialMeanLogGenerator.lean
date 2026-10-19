module

public import Transformer.Homogenized.Section7_ExponentialMeanLog
public import Transformer.Homogenized.Section7_MeanLogDriftBounds

/-!
# Actual exponential generator control for the original logarithmic mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the exponential maximal argument after the original mean bracket,
and Appendix, `def:ito.formula`. The genuine logarithmic exponential
test has generator

  `L exp(cψ) = exp(cψ) (c Lψ + c² qψ/2)`.

Every expectation in this identity is proved integrable under the
actual original Gaussian head. The genuine bound `qψ ≤ 4/d` gives
the exponential generator inequality for both signs of `c`.

The fixed small regularization also gives the actual uniform
generator domination needed for the integrating-factor argument.
The full adapted continuous original sphere martingale problem
supplies the selected exponential test identity. Neither that
weighted supermartingale nor its maximal probability bound is
introduced as a replacement stochastic hypothesis.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped NNReal

namespace Transformer.Homogenized

/-- The genuine exponential chain rule for the original sphere
generator, with its two actual expectation integrability premises.
Source: arXiv:2604.01978v1, §7 and Appendix, `def:ito.formula`. -/
theorem sphGenerator_exponentialLogMeanOverlap {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (ρ : Measure (HeadParam d)) (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d) (X : Idx n → EucSpace d)
    (hH : Integrable (fun θ => sphHess (logMeanOverlap a) X (G θ X)) ρ)
    (hQ : Integrable (fun θ => (fderiv ℝ (logMeanOverlap a) X (G θ X)) ^ 2) ρ) :
    sphGenerator ρ B G (exponentialLogMeanOverlap c a) X = exponentialLogMeanOverlap c a X *
      (c * sphGenerator ρ B G (logMeanOverlap a) X +
        c ^ 2 / 2 * ∫ θ, (fderiv ℝ (logMeanOverlap a) X (G θ X)) ^ 2 ∂ρ) := by
  rw [sphGenerator, fderiv_exponentialLogMeanOverlap c ha]
  simp only [smul_apply, smul_eq_mul]
  simp_rw [sphHess_exponentialLogMeanOverlap c ha]
  rw [integral_const_mul, integral_add (hH.const_mul c) (hQ.const_mul (c ^ 2)),
    integral_const_mul, integral_const_mul, sphGenerator]
  ring

/-- The actual exponential-logarithmic sphere Hessian is integrable
under the original Gaussian head law.
Source: arXiv:2604.01978v1, §7, the genuine exponential maximal test. -/
theorem integrable_sphHess_exponentialLogMeanOverlap_gaussian {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    Integrable (fun θ => sphHess (exponentialLogMeanOverlap c a) X (noiseField β 1 1 ρ θ X)) ρ := by
  have hH := integrable_sphHess_logMeanOverlap_gaussian ha hn β hρ X hX
  have hQ := integrable_logMeanOverlapNoiseIntegrand_sq_gaussian ha hn β hρ X hX
  simp_rw [sphHess_exponentialLogMeanOverlap c ha]
  exact ((hH.const_mul c).add (hQ.const_mul (c ^ 2))).const_mul _

/-- The original Gaussian exponential test has its true logarithmic
generator and actual logarithmic noise second moment.
Source: arXiv:2604.01978v1, §7, the original exponential maximal argument. -/
theorem sphGenerator_exponentialLogMeanOverlap_gaussian {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialLogMeanOverlap c a) X =
      exponentialLogMeanOverlap c a X *
        (c * sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X +
          c ^ 2 / 2 * ∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ) :=
  sphGenerator_exponentialLogMeanOverlap c ha ρ (bField β ρ) (noiseField β 1 1 ρ) X
    (integrable_sphHess_logMeanOverlap_gaussian ha hn β hρ X hX)
    (integrable_logMeanOverlapNoiseIntegrand_sq_gaussian ha hn β hρ X hX)

/-- **The genuine uniform logarithmic noise bound gives the actual
exponential generator inequality for either sign of the parameter.**
Source: arXiv:2604.01978v1, §7, the exponential maximal argument
after the original Gaussian mean bracket calculation. -/
theorem sphGenerator_exponentialLogMeanOverlap_le {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (hreg : 20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 ≤ a)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialLogMeanOverlap c a) X ≤
      exponentialLogMeanOverlap c a X *
        (c * sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X + c ^ 2 / 2 * (4 / (d : ℝ))) := by
  rw [sphGenerator_exponentialLogMeanOverlap_gaussian c ha hn β hρ X hX]
  apply mul_le_mul_of_nonneg_left _ (exponentialLogMeanOverlap_pos c a X).le
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (integral_logMeanOverlapNoise_sq_le_four_div_dim ha hn β hβ hρ hσ hreg X hX) (by positivity))

/-- The actual exponential logarithmic generator has a finite
uniform bound on unit configurations at the fixed small regularization.
Source: arXiv:2604.01978v1, §7, the bounded generator domination
in the exponential maximal argument. -/
theorem abs_sphGenerator_exponentialLogMeanOverlap_regularization_le {d n : ℕ}
    (hd : 2 ≤ d) (c : ℝ) (hn : 0 < n) (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) (hsmall : meanLogRegularization d β σA ≤ 1)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
        (exponentialLogMeanOverlap c (meanLogRegularization d β σA)) X| ≤
      Real.exp (|c| * (|Real.log (meanLogRegularization d β σA)| + |Real.log (1 + meanLogRegularization d β σA)|)) *
        (9 * |c| + c ^ 2 / 2 * (4 / (d : ℝ))) := by
  let a := meanLogRegularization d β σA
  have ha : 0 < a := meanLogRegularization_pos (by omega) β σA
  have hL := abs_sphGenerator_logMeanOverlap_regularization_le_nine hd hn β hβ hρ hσ hsmall X hX
  have hq := integral_logMeanOverlapNoise_regularization_sq_le (by omega) hn β hβ hρ hσ X hX
  have hq0 : 0 ≤ ∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ := integral_nonneg fun θ => sq_nonneg _
  have hc : 0 ≤ c ^ 2 / 2 := by positivity
  rw [sphGenerator_exponentialLogMeanOverlap_gaussian c ha hn β hρ X hX, abs_mul,
    abs_of_pos (exponentialLogMeanOverlap_pos c a X)]
  have ht : |c * sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X +
      c ^ 2 / 2 * ∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ| ≤
        |c| * 9 + c ^ 2 / 2 * (4 / (d : ℝ)) := by
    apply (abs_add_le _ _).trans
    rw [abs_mul, abs_of_nonneg (mul_nonneg hc hq0)]
    exact add_le_add (mul_le_mul_of_nonneg_left hL (abs_nonneg c)) (mul_le_mul_of_nonneg_left hq hc)
  exact (mul_le_mul (exponentialLogMeanOverlap_le_exp_abs c ha X hX) ht (abs_nonneg _) (Real.exp_nonneg _)).trans_eq (by ring)

/-- The full original adapted continuous sphere martingale problem
supplies the actual selected logarithmic exponential test identity.
Source: arXiv:2604.01978v1, §7 and Appendix, `def:ito.formula`. -/
theorem IsSphereMartingaleSolution.exponentialLogMeanOverlap {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    {ρ : Measure (HeadParam d)} {B : (Idx n → EucSpace d) → Idx n → EucSpace d}
    {G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d}
    {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ m}
    {T : ℝ} {X : ℝ → Ω → (Idx n → EucSpace d)}
    (h : IsSphereMartingaleSolution ρ B G P ℱ T X) :
    IsMartingaleOn T ℱ P (fun t ω => exponentialLogMeanOverlap c a (X t ω) - exponentialLogMeanOverlap c a (X 0 ω) -
      ∫ s in (0 : ℝ)..t, sphGenerator ρ B G (exponentialLogMeanOverlap c a) (X s ω)) :=
  h.2.2.2 _ (contDiff_exponentialLogMeanOverlap c ha)

/-- Positive original Gaussian variances, unit queries and the
nonzero small regularization jointly realize every generator premise. -/
example : ∃ X : Idx 2 → EucSpace 2, (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) (1 / 2) (gaussHeadLaw 2 (stdSigmaV 2) (1 / 2)) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ (1 / 4 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    meanLogRegularization 2 (1 / 4) (1 / 2) ≤ 1 := by
  refine ⟨fun k => EuclideanSpace.single k 1, le_rfl, by decide, fun k => by simp,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, by norm_num, ?_⟩
  norm_num [meanLogRegularization]

/-- The original positive Gaussian head and unit queries also realize
both actual expectation premises of the arbitrary-law chain rule. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℝ) < 1 ∧ (∀ k, ‖X k‖ = 1) ∧
    Integrable (fun θ => sphHess (logMeanOverlap 1) X
      (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) θ X)) (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    Integrable (fun θ => (fderiv ℝ (logMeanOverlap 1) X
      (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) θ X)) ^ 2) (gaussHeadLaw 2 (stdSigmaV 2) 1) := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hX : ∀ k, ‖X k‖ = 1 := fun k => by simp [X]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 1
  exact ⟨X, one_pos, hX, integrable_sphHess_logMeanOverlap_gaussian one_pos (by decide) 0 hρ X hX,
    integrable_logMeanOverlapNoiseIntegrand_sq_gaussian one_pos (by decide) 0 hρ X hX⟩

/-- Positive regularization and an actual full-test sphere solution
jointly realize the exponential martingale selector's hypotheses. -/
example : ∃ ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit), (0 : ℝ) < 1 ∧
    IsSphereMartingaleSolution (Measure.dirac (0 : HeadParam 2))
      (bField 1 (Measure.dirac 0)) (noiseField 1 1 1 (Measure.dirac 0)) (Measure.dirac ()) ℱ 1
      (fun _ _ => (fun i : Idx 2 => EuclideanSpace.single i (1 : ℝ))) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  exact ⟨ℱ, one_pos, isSphereMartingaleSolution_dirac_zero (Measure.dirac ()) ℱ 1 1 1 1 _ (fun i => by simp)⟩

end Transformer.Homogenized
