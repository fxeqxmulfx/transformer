module

public import Transformer.Homogenized.Section7_ExponentialMeanLogGenerator
public import Transformer.Homogenized.Section7_GramFieldContinuity

/-!
# Actual mean-noise covariance and continuity of the logarithmic fields

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the original Gaussian bracket calculation following
`eq:gronwall_e_clean`. The true mean-noise second moment is the
finite covariance contraction of the actual projected centroid
vectors with the original barycenter correlations.

Every scalar cross term is genuinely integrable under the original
Gaussian head law before the finite sums are integrated. The formula
keeps correlations between all original queries and makes no token
or query independence assumption.

The projected centroid vectors depend continuously on the actual
configuration, and the original Gaussian correlations are already
proved continuous on unit configurations, including singular Gram
matrices. The covariance identity therefore gives continuity of the
actual mean-noise second moment. The exact original generator and
positive regularization transfer this to the logarithmic and
exponential-logarithmic generator fields used in the maximal bound.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- **The genuine original mean-noise second moment is its full
projected-centroid Gaussian covariance contraction.**
Source: arXiv:2604.01978v1, §7, the original mean bracket calculation.
Both original query indices remain in every correlation. -/
theorem integral_meanOverlapNoise_sq_eq_centroid_covariance {d n : ℕ} (β : ℝ)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    (∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ) =
      4 * (σV : ℝ) ^ 2 * ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        inner ℝ (proj d (X i) (simplexCentroid X)) (proj d (X j) (simplexCentroid X)) *
          baryCorr β ρ (empMeasure X) (X i) (X j) := by
  let F := fun θ : HeadParam d => fun i : Idx n => inner ℝ (simplexCentroid X) (Gfield β ρ X θ i)
  have hi (i j : Idx n) : Integrable (fun θ => F θ i * F θ j) ρ :=
    integrable_inner_Gfield_scalar_cross_gaussian β hρ X hX i j _ _
  have he (θ : HeadParam d) : meanOverlapNoiseIntegrand β ρ X θ ^ 2 =
      (4 * ((n : ℝ) ^ 2)⁻¹) * ∑ i : Idx n, ∑ j : Idx n, F θ i * F θ j := by
    have hs : (∑ i : Idx n, F θ i) ^ 2 = ∑ i : Idx n, ∑ j : Idx n, F θ i * F θ j := by
      rw [pow_two, Finset.sum_mul_sum]
    rw [meanOverlapNoiseIntegrand_eq_centroid_sum]
    change (2 * (n : ℝ)⁻¹ * ∑ i : Idx n, F θ i) ^ 2 = _
    calc
      _ = 4 * ((n : ℝ)⁻¹) ^ 2 * (∑ i : Idx n, F θ i) ^ 2 := by ring
      _ = _ := by rw [inv_pow, hs]
  have hE (i j : Idx n) : (∫ θ, F θ i * F θ j ∂ρ) =
      (σV : ℝ) ^ 2 * inner ℝ (proj d (X i) (simplexCentroid X))
        (proj d (X j) (simplexCentroid X)) * baryCorr β ρ (empMeasure X) (X i) (X j) :=
    integral_inner_Gfield_scalar_cross_gaussian β hρ X hX i j _ _
  simp_rw [he]
  rw [integral_const_mul, integral_finsetSum _ fun i hi' => integrable_finsetSum _ fun j hj => hi i j]
  simp_rw [integral_finsetSum _ fun j hj => hi _ j, hE]
  simp only [mul_assoc, ← Finset.mul_sum]
  ring

/-- The actual original centroid is continuous on the whole ambient
configuration space. Source: arXiv:2604.01978v1, §7,
the centroid in part (a) of `eq:drift_mismatch_target_clean`. -/
theorem continuous_simplexCentroid {d n : ℕ} :
    Continuous (simplexCentroid (d := d) (n := n)) := by
  unfold simplexCentroid
  fun_prop

/-- The genuine projected-centroid scalar covariance coefficient
is globally continuous. Source: arXiv:2604.01978v1, §7,
the Gaussian projection factors in the original mean bracket. -/
theorem continuous_projected_centroid_inner {d n : ℕ} (i j : Idx n) :
    Continuous (fun X : Idx n → EucSpace d => inner ℝ
      (proj d (X i) (simplexCentroid X)) (proj d (X j) (simplexCentroid X))) := by
  have hp (k : Idx n) : Continuous (fun X : Idx n → EucSpace d => proj d (X k) (simplexCentroid X)) := by
    unfold proj
    exact continuous_simplexCentroid.sub (((continuous_apply k).inner (𝕜 := ℝ) continuous_simplexCentroid).smul (continuous_apply k))
  exact (hp i).inner (𝕜 := ℝ) (hp j)

/-- The original mean-noise second moment is continuous on all
actual unit configurations, including singular Grams.
Source: arXiv:2604.01978v1, §7, the original mean bracket and `lem:Gram_stability`. -/
theorem continuousOn_meanOverlapNoise_sq_integral_gaussian {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) :
    ContinuousOn (fun X : Idx n → EucSpace d => ∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ)
      {X | ∀ k, ‖X k‖ = 1} := by
  have hp : ContinuousOn (fun X : Idx n → EucSpace d =>
      4 * (σV : ℝ) ^ 2 * ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        inner ℝ (proj d (X i) (simplexCentroid X)) (proj d (X j) (simplexCentroid X)) *
          baryCorr β ρ (empMeasure X) (X i) (X j)) {X | ∀ k, ‖X k‖ = 1} :=
    continuousOn_const.mul (continuousOn_finsetSum _ fun i hi => continuousOn_finsetSum _ fun j hj =>
      (continuous_projected_centroid_inner i j).continuousOn.mul
        (continuousOn_baryCorr_empMeasure_gaussian β σV σA ρ hρ i j))
  exact hp.congr fun X hX => integral_meanOverlapNoise_sq_eq_centroid_covariance β hρ X hX

/-- The genuine averaged original Gaussian drift is continuous on
the actual sphere product. Source: arXiv:2604.01978v1, §7,
`eq:F_def_clean`, averaging the original `eq:Dij_explicit_clean`. -/
theorem continuousOn_meanOverlapDrift_gaussian {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) :
    ContinuousOn (meanOverlapDrift (d := d) (n := n) β ρ) {X | ∀ k, ‖X k‖ = 1} := by
  exact continuousOn_const.mul (continuousOn_finsetSum _ fun i hi => continuousOn_finsetSum _ fun j hj =>
    continuousOn_overlapDrift_gaussian hd β σV σA ρ hρ i j)

/-- Positive regularization transfers the genuine covariance
continuity to the actual logarithmic noise second moment.
Source: arXiv:2604.01978v1, §7, the original bracket under its actual logarithmic differential. -/
theorem continuousOn_logMeanOverlapNoise_sq_integral_gaussian {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) :
    ContinuousOn (fun X : Idx n → EucSpace d => ∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ)
      {X | ∀ k, ‖X k‖ = 1} := by
  have hm : Continuous (fun X : Idx n → EucSpace d => meanOverlap X + a) := contDiff_meanOverlap.continuous.add continuous_const
  have hi := hm.inv₀ fun X => (meanOverlap_add_pos ha X).ne'
  have hp := (hi.pow 2).continuousOn.mul (continuousOn_meanOverlapNoise_sq_integral_gaussian β σV σA ρ hρ)
  apply hp.congr
  intro X hX
  change (∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ) =
    ((meanOverlap X + a)⁻¹) ^ 2 * ∫ θ, meanOverlapNoiseIntegrand β ρ X θ ^ 2 ∂ρ
  rw [integral_logMeanOverlapNoiseIntegrand_sq ha, inv_pow]

/-- The actual original Gaussian logarithmic generator is continuous
on the genuine sphere product. Source: arXiv:2604.01978v1,
§7, the original mean generator's time integral in its maximal estimate. -/
theorem continuousOn_sphGenerator_logMeanOverlap_gaussian {d n : ℕ} (hd : 2 ≤ d) {a : ℝ} (ha : 0 < a)
    (hn : 0 < n) (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) :
    ContinuousOn (fun X : Idx n → EucSpace d =>
      sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X) {X | ∀ k, ‖X k‖ = 1} := by
  have hm : Continuous (fun X : Idx n → EucSpace d => meanOverlap X + a) := contDiff_meanOverlap.continuous.add continuous_const
  have hp := ((continuousOn_meanOverlapDrift_gaussian hd β σV σA ρ hρ).div hm.continuousOn
    (fun X hX => (meanOverlap_add_pos ha X).ne')).sub
      ((continuousOn_logMeanOverlapNoise_sq_integral_gaussian ha β σV σA ρ hρ).div_const 2)
  exact hp.congr fun X hX => sphGenerator_logMeanOverlap_gaussian_eq_div ha hn β hρ hσ X hX

/-- The true logarithmic exponential generator is continuous on
the original unit configurations. Source: arXiv:2604.01978v1,
§7, the original integrating-factor maximal estimate. -/
theorem continuousOn_sphGenerator_exponentialLogMeanOverlap_gaussian {d n : ℕ} (hd : 2 ≤ d)
    (c : ℝ) {a : ℝ} (ha : 0 < a) (hn : 0 < n) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) :
    ContinuousOn (fun X : Idx n → EucSpace d =>
      sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialLogMeanOverlap c a) X) {X | ∀ k, ‖X k‖ = 1} := by
  have hD := continuousOn_sphGenerator_logMeanOverlap_gaussian hd ha hn β σV σA ρ hρ hσ
  have hq := continuousOn_logMeanOverlapNoise_sq_integral_gaussian (n := n) ha β σV σA ρ hρ
  have hp : ContinuousOn (fun X : Idx n → EucSpace d => exponentialLogMeanOverlap c a X *
      (c * sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (logMeanOverlap a) X +
        c ^ 2 / 2 * ∫ θ, logMeanOverlapNoiseIntegrand a β ρ X θ ^ 2 ∂ρ)) {X | ∀ k, ‖X k‖ = 1} :=
    (contDiff_exponentialLogMeanOverlap c ha).continuous.continuousOn.mul
      ((continuousOn_const.mul hD).add (continuousOn_const.mul hq))
  exact hp.congr fun X hX => sphGenerator_exponentialLogMeanOverlap_gaussian c ha hn β hρ X hX

/-- Original positive Gaussian variances, positive regularization
and nonempty unit queries jointly realize all continuity hypotheses. -/
example : ∃ X : Idx 2 → EucSpace 2, (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ (∀ k, ‖X k‖ = 1) :=
  ⟨fun k => EuclideanSpace.single k 1, le_rfl, by decide, one_pos,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, fun k => by simp⟩

end Transformer.Homogenized
