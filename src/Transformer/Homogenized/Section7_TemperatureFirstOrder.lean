module

public import Transformer.Homogenized.Section7_TemperatureCorrelation
public import Transformer.Homogenized.Section7_GaussianAttentionSymmetry
public import Transformer.Homogenized.Section7_MeanOverlap

/-!
# Actual Gaussian cancellation of the original temperature linear term

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the first-order cancellation in `eq:s_mu_small_beta_clean`.
At zero temperature the original empirical barycenter is its
centroid. Its genuine derivative is the original finite sum of
centered query scores, with the actual `1/n` normalization retained.

The centered-score functions are genuinely integrable under every
original Gaussian head law. Their original expectations vanish by
the proved attention-matrix sign symmetry. Integrating the actual
finite derivative sum therefore gives zero, without assuming that
the product attention matrix is Gaussian or adding a mean-zero
derivative as an input to the argument.

The actual two-barycenter correlation has the genuine mean overlap
as its zeroth-order term and an integrable, centered actual derivative
as its first-order term. These are the original terms to which the
proved uniform temperature Taylor remainder will be applied.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- At actual temperature zero the original normalized finite sum
is the actual centroid. Source: arXiv:2604.01978v1, §7,
part (a) and the Taylor constant term in part (b). -/
theorem temperatureBarycenter_zero {d n : ℕ} (X : Idx n → EucSpace d) (a : Idx n → ℝ) :
    temperatureBarycenter X a 0 = simplexCentroid X := by
  simp only [temperatureBarycenter, zero_mul, Perspective.softmaxWeight, Real.exp_zero,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one,
    one_div, simplexCentroid, Finset.smul_sum]

/-- The genuine first temperature derivative at zero is the actual
normalized centered-score sum. Source: arXiv:2604.01978v1, §7,
the linearity in `A` used in `eq:s_mu_small_beta_clean`. -/
theorem deriv_temperatureBarycenter_zero {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (a : Idx n → ℝ) :
    deriv (temperatureBarycenter X a) 0 = ∑ k : Idx n,
      ((n : ℝ)⁻¹ * (a k - (n : ℝ)⁻¹ * ∑ l : Idx n, a l)) • X k := by
  rw [(hasDerivAt_temperatureBarycenter hn X a 0).deriv]
  simp_rw [(hasDerivAt_softmaxWeight_temperature hn a _ 0).deriv]
  simp only [zero_mul, Perspective.softmaxWeight, Real.exp_zero, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one, one_div, ← Finset.mul_sum]

/-- **The actual original first barycenter derivative is integrable
and has zero Gaussian expectation.** Source: arXiv:2604.01978v1,
§7, the displayed cancellation preceding `eq:s_mu_small_beta_clean`.
The original query scores, normalization and Gaussian head law are retained. -/
theorem gaussian_temperatureBarycenter_deriv_zero {d n : ℕ} (hn : 0 < n)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (i : Idx n) :
    Integrable (fun θ => deriv (temperatureBarycenter X (fun k => inner ℝ (qkMap θ (X i)) (X k))) 0) ρ ∧
      (∫ θ, deriv (temperatureBarycenter X (fun k => inner ℝ (qkMap θ (X i)) (X k))) 0 ∂ρ) = 0 := by
  let a := fun θ : HeadParam d => fun k : Idx n => inner ℝ (qkMap θ (X i)) (X k)
  have hi (k : Idx n) : Integrable (fun θ => a θ k) ρ := integrable_gaussian_head_score hρ _ _
  have hs : Integrable (fun θ => ∑ k : Idx n, a θ k) ρ :=
    integrable_finsetSum _ fun k hk => hi k
  have hc (k : Idx n) : Integrable
      (fun θ => (n : ℝ)⁻¹ * (a θ k - (n : ℝ)⁻¹ * ∑ l : Idx n, a θ l)) ρ := by
    convert ((hi k).sub (hs.const_mul (n : ℝ)⁻¹)).const_mul (n : ℝ)⁻¹ using 1
  have he (k : Idx n) : (∫ θ, (n : ℝ)⁻¹ * (a θ k - (n : ℝ)⁻¹ * ∑ l : Idx n, a θ l) ∂ρ) = 0 := by
    rw [integral_const_mul, integral_sub (hi k) (hs.const_mul _), integral_const_mul,
      integral_finsetSum _ fun l hl => hi l]
    simp only [a, integral_gaussian_head_score_eq_zero hρ, Finset.sum_const_zero,
      mul_zero, sub_zero]
  simp_rw [deriv_temperatureBarycenter_zero hn]
  constructor
  · exact integrable_finsetSum _ fun k hk => (hc k).smul_const (X k)
  · rw [integral_finsetSum _ fun k hk => (hc k).smul_const (X k)]
    simp only [integral_smul_const, he, zero_smul, Finset.sum_const_zero]

/-- The cancellation holds for the genuine original exponential
barycenter itself, differentiated in its original temperature parameter.
The finite normalized sum is identified pointwise with that same
original function before differentiating or integrating it.
Source: arXiv:2604.01978v1, §7, the displayed identity
`E ∂_β m_{0,A}[μ](x) = 0` in `eq:s_mu_small_beta_clean`.
All original head-law variances, token vectors and query indices
are retained; a unit-token assumption is unnecessary for this linear term. -/
theorem gaussian_softBary_temperature_deriv_zero {d n : ℕ} (hn : 0 < n)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (i : Idx n) :
    Integrable (fun θ => deriv (fun β : ℝ => softBary β θ.2 (empMeasure X) (X i)) 0) ρ ∧
      (∫ θ, deriv (fun β : ℝ => softBary β θ.2 (empMeasure X) (X i)) 0 ∂ρ) = 0 := by
  have he (θ : HeadParam d) : (fun β : ℝ => softBary β θ.2 (empMeasure X) (X i)) =
      temperatureBarycenter X (fun k => inner ℝ (qkMap θ (X i)) (X k)) := by
    funext β
    exact (temperatureBarycenter_eq_softBary X θ.2 i β).symm
  simp_rw [he]
  exact gaussian_temperatureBarycenter_deriv_zero hn hρ X i

/-- The actual correlation's zeroth-order term is exactly the
original empirical mean overlap, with no unit-token restriction.
Source: arXiv:2604.01978v1, §7, part (a) of the small-temperature argument. -/
theorem temperatureBarycenterCorrelation_zero {d n : ℕ}
    (X : Idx n → EucSpace d) (a b : Idx n → ℝ) :
    temperatureBarycenterCorrelation X a b 0 = meanOverlap X := by
  rw [temperatureBarycenterCorrelation, temperatureBarycenter_zero, temperatureBarycenter_zero,
    real_inner_self_eq_norm_sq, meanOverlap_eq_norm_simplexCentroid_sq]

/-- **The genuine correlation's first-order term is integrable and
has zero original Gaussian expectation, including self-pairs.**
Source: arXiv:2604.01978v1, §7, part (b) of the proof of
`eq:s_mu_small_beta_clean`, the first-order terms after expanding
the two original barycenters. No covariance or remainder is assumed. -/
theorem gaussian_temperatureBarycenterCorrelation_deriv_zero {d n : ℕ} (hn : 0 < n)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (i j : Idx n) :
    Integrable (fun θ => deriv (temperatureBarycenterCorrelation X
      (fun k => inner ℝ (qkMap θ (X i)) (X k)) (fun k => inner ℝ (qkMap θ (X j)) (X k))) 0) ρ ∧
    (∫ θ, deriv (temperatureBarycenterCorrelation X
      (fun k => inner ℝ (qkMap θ (X i)) (X k)) (fun k => inner ℝ (qkMap θ (X j)) (X k))) 0 ∂ρ) = 0 := by
  let D := fun k : Idx n => fun θ : HeadParam d =>
    deriv (temperatureBarycenter X (fun l => inner ℝ (qkMap θ (X k)) (X l))) 0
  have hi (k : Idx n) : Integrable (D k) ρ ∧ (∫ θ, D k θ ∂ρ) = 0 :=
    gaussian_temperatureBarycenter_deriv_zero hn hρ X k
  have he (θ : HeadParam d) : deriv (temperatureBarycenterCorrelation X
      (fun k => inner ℝ (qkMap θ (X i)) (X k)) (fun k => inner ℝ (qkMap θ (X j)) (X k))) 0 =
      inner ℝ (simplexCentroid X) (D j θ + D i θ) := by
    rw [(hasDerivAt_temperatureBarycenterCorrelation hn X _ _ 0).deriv,
      temperatureBarycenter_zero, temperatureBarycenter_zero, inner_add_right]
    rw [real_inner_comm (simplexCentroid X) (D i θ)]
  simp_rw [he]
  have hD : Integrable (fun θ => D j θ + D i θ) ρ := by convert (hi j).1.add (hi i).1 using 1
  constructor
  · exact hD.const_inner (simplexCentroid X)
  · rw [integral_inner hD, integral_add (hi j).1 (hi i).1, (hi j).2, (hi i).2,
      add_zero, inner_zero_right]

/-- The original headwise soft-barycenter correlation has the same
integrable, centered genuine first derivative, before its expectation
defines the original `baryCorr`. Source: arXiv:2604.01978v1, §7,
the two original barycenter factors in `eq:s_mu_small_beta_clean`. -/
theorem gaussian_softBary_inner_temperature_deriv_zero {d n : ℕ} (hn : 0 < n)
    {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (i j : Idx n) :
    Integrable (fun θ => deriv (fun β : ℝ => inner ℝ
      (softBary β θ.2 (empMeasure X) (X i)) (softBary β θ.2 (empMeasure X) (X j))) 0) ρ ∧
    (∫ θ, deriv (fun β : ℝ => inner ℝ
      (softBary β θ.2 (empMeasure X) (X i)) (softBary β θ.2 (empMeasure X) (X j))) 0 ∂ρ) = 0 := by
  have he (θ : HeadParam d) : (fun β : ℝ => inner ℝ
      (softBary β θ.2 (empMeasure X) (X i)) (softBary β θ.2 (empMeasure X) (X j))) =
      temperatureBarycenterCorrelation X (fun k => inner ℝ (qkMap θ (X i)) (X k))
        (fun k => inner ℝ (qkMap θ (X j)) (X k)) := by
    funext β
    exact (temperatureBarycenterCorrelation_eq_inner_softBary X θ.2 i j β).symm
  simp_rw [he]
  exact gaussian_temperatureBarycenterCorrelation_deriv_zero hn hρ X i j

/-- Original positive Gaussian variances, nonempty orthogonal unit
queries and a genuine query pair jointly realize all cancellation hypotheses. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧
    IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) ∧
      temperatureBarycenterCorrelation X ![1, 0] ![0, 1] 0 = 1 / 2 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  refine ⟨X, by decide, fun k => by simp [X], isGaussianHeadLaw_gaussHeadLaw 2 1 1, ?_⟩
  rw [temperatureBarycenterCorrelation_zero]
  norm_num [meanOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]

end Transformer.Homogenized
