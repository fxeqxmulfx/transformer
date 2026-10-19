module

public import Transformer.Homogenized.Section7_ExponentialGenerator
public import Transformer.Homogenized.Section7_OverlapDriftGramComparison

/-!
# Uniform bounds for the original exponential generator

ArXiv:2604.01978v1, §7, `eq:Dij_explicit_clean` and the
exponential maximal inequality after `eq:Gronwall_E_selfcontained`.
The original drift has absolute value at most 2 on unit
configurations in dimensions at least two. This follows from
the actual barycenter correlations and the source's two drift
coefficients, independently of a Gaussian head law.

Under the actual Gaussian law with σV²=1/d, the original
exponential generator is uniformly bounded by
exp(|c|)(2|c|+c²(3/d)/2). The same bound controls the actual
compensation coefficient without assuming a drift bound.
The quadratic generator remainder is retained and bounded
between zero and exp(|c|)c²(3/d)/2 for either sign of c.
These are pointwise bounds on the original fields, supplying
the domination needed for the proved integrating-factor argument.
-/

@[expose] public section

open Set MeasureTheory
open scoped NNReal
namespace Transformer.Homogenized

/-- The original drift is bounded by 2 using the original unit-token correlations.
Source: arXiv:2604.01978v1, §7, `eq:Dij_explicit_clean`. -/
theorem abs_overlapDrift_le_two {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (ρ : Measure (HeadParam d))
    [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    |overlapDrift β ρ X i j| ≤ 2 := by
  let R := inner (𝕜 := ℝ) (X i) (X j)
  let S := baryCorr β ρ (empMeasure X) (X i) (X j)
  let P := baryCorr β ρ (empMeasure X) (X i) (X i)
  let Q := baryCorr β ρ (empMeasure X) (X j) (X j)
  have hr : |R| ≤ 1 := by simpa [R, hX i, hX j] using abs_real_inner_le_norm (X i) (X j)
  have hS : |S| ≤ 1 := abs_baryCorr_empMeasure_le_one β ρ X hX i j
  have hP : |P| ≤ 1 := abs_baryCorr_empMeasure_le_one β ρ X hX i i
  have hQ : |Q| ≤ 1 := abs_baryCorr_empMeasure_le_one β ρ X hX j j
  have hdR : (2 : ℝ) ≤ d := by exact_mod_cast hd
  have hp := abs_overlap_gram_coefficient_le_one (d : ℝ) R hdR hr
  have hq := overlap_self_coefficient_mem_Icc (d : ℝ) hdR
  have hfirst : |(1 / (d : ℝ)) * ((d : ℝ) - 2 + R ^ 2) * S| ≤ 1 := by
    rw [abs_mul]
    simpa only [one_mul] using mul_le_mul hp hS (abs_nonneg _) zero_le_one
  have hqr : |(((d : ℝ) - 1) / (2 * d)) * R| ≤ 1 / 2 := by
    rw [abs_mul, abs_of_nonneg hq.1]
    exact (mul_le_mul_of_nonneg_left hr hq.1).trans (by simpa only [mul_one] using hq.2)
  have hpq : |P + Q| ≤ 2 := (abs_add_le _ _).trans (by linarith)
  have hsecond : |(((d : ℝ) - 1) / (2 * d)) * R * (P + Q)| ≤ 1 := by
    rw [abs_mul]
    have h := mul_le_mul hqr hpq (abs_nonneg _) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    norm_num at h ⊢
    exact h
  change |(1 / (d : ℝ)) * ((d : ℝ) - 2 + R ^ 2) * S -
    (((d : ℝ) - 1) / (2 * d)) * R * (P + Q)| ≤ 2
  exact (abs_sub _ _).trans ((add_le_add hfirst hsecond).trans_eq (by norm_num))

/-- The actual linear drift and quadratic noise coefficient obey a uniform bound.
Source: arXiv:2604.01978v1, §7, exponential compensation. -/
theorem abs_exponential_drift_coefficient_le (c D q Q : ℝ) (hD : |D| ≤ 2) (hq : q ∈ Icc (0 : ℝ) Q) :
    |c * D + c ^ 2 / 2 * q| ≤ 2 * |c| + c ^ 2 / 2 * Q := by
  have hc : 0 ≤ c ^ 2 / 2 := by positivity
  calc
    _ ≤ |c * D| + |c ^ 2 / 2 * q| := abs_add_le _ _
    _ = |c| * |D| + c ^ 2 / 2 * q := by rw [abs_mul, abs_of_nonneg (mul_nonneg hc hq.1)]
    _ ≤ |c| * 2 + c ^ 2 / 2 * Q :=
      add_le_add (mul_le_mul_of_nonneg_left hD (abs_nonneg _)) (mul_le_mul_of_nonneg_left hq.2 hc)
    _ = _ := by ring

/-- The uniform coefficient bound is nonnegative at every dimension and parameter.
Source: arXiv:2604.01978v1, §7, the domination in the maximal-inequality argument. -/
theorem exponential_overlap_bound_nonneg (d : ℕ) (c : ℝ) :
    0 ≤ 2 * |c| + c ^ 2 / 2 * (3 / (d : ℝ)) := by positivity

/-- The coefficient of the actual exponential integrating factor has the same bound.
Source: arXiv:2604.01978v1, §7, with the proved noise upper bound 3/d. -/
theorem abs_overlap_compensation_coefficient_le {d n : ℕ} (hd : 2 ≤ d) (c β : ℝ) (ρ : Measure (HeadParam d))
    [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    |c * overlapDrift β ρ X i j + c ^ 2 / 2 * (3 / (d : ℝ))| ≤
      2 * |c| + c ^ 2 / 2 * (3 / (d : ℝ)) := by
  exact abs_exponential_drift_coefficient_le c _ _ _ (abs_overlapDrift_le_two hd β ρ X hX i j)
    ⟨by positivity, le_rfl⟩

/-- The original Gaussian exponential generator has the required actual uniform bound.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`, at σV²=1/d. -/
theorem abs_sphGenerator_exponentialOverlap_le {d n : ℕ} (hd : 2 ≤ d) (c β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    |sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) X| ≤
      Real.exp |c| * (2 * |c| + c ^ 2 / 2 * (3 / (d : ℝ))) := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hD := abs_overlapDrift_le_two hd β ρ X hX i j
  have hq : (∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) ∈ Icc (0 : ℝ) (3 / (d : ℝ)) :=
    ⟨integral_overlapNoiseIntegrand_sq_nonneg β ρ X i j,
      integral_overlapNoiseIntegrand_sq_le_three_div_dim β hρ hσ X hX i j⟩
  rw [sphGenerator_exponentialOverlap_gaussian c β hρ hσ X hX, abs_mul,
    abs_of_pos (exponentialOverlap_pos c i j X)]
  exact mul_le_mul (exponentialOverlap_le_exp_abs c i j X hX)
    (abs_exponential_drift_coefficient_le c _ _ _ hD hq) (abs_nonneg _) (Real.exp_nonneg _)

/-- The original quadratic generator remainder is nonnegative and bounded.
Source: arXiv:2604.01978v1, §7, the original Gaussian overlap-noise calculation.
The actual second moment remains in the identity used by the proof. -/
theorem sphGenerator_exponentialOverlap_remainder_mem_Icc {d n : ℕ} (c β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) X -
      exponentialOverlap c i j X * (c * overlapDrift β ρ X i j) ∈
      Icc (0 : ℝ) (Real.exp |c| * (c ^ 2 / 2 * (3 / (d : ℝ)))) := by
  rw [sphGenerator_exponentialOverlap_gaussian c β hρ hσ X hX]
  have heq : exponentialOverlap c i j X * (c * overlapDrift β ρ X i j +
      c ^ 2 / 2 * ∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) -
      exponentialOverlap c i j X * (c * overlapDrift β ρ X i j) =
      exponentialOverlap c i j X * (c ^ 2 / 2 * ∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) := by ring
  rw [heq]
  have hc : 0 ≤ c ^ 2 / 2 := by positivity
  have hq0 := integral_overlapNoiseIntegrand_sq_nonneg β ρ X i j
  exact ⟨mul_nonneg (exponentialOverlap_pos c i j X).le (mul_nonneg hc hq0),
    mul_le_mul (exponentialOverlap_le_exp_abs c i j X hX)
      (mul_le_mul_of_nonneg_left (integral_overlapNoiseIntegrand_sq_le_three_div_dim β hρ hσ X hX i j) hc)
      (mul_nonneg hc hq0) (Real.exp_nonneg _)⟩


/-- A negative exponential parameter and positive noise density also
realize the scalar hypotheses used for the second side of the maximal bound. -/
example : |(-1 : ℝ) * 1 + (-1 : ℝ) ^ 2 / 2 * 1| ≤
    2 * |(-1 : ℝ)| + (-1 : ℝ) ^ 2 / 2 * 1 := by
  exact abs_exponential_drift_coefficient_le (-1) 1 1 1 (by norm_num) (by norm_num)

/-- A genuine standard positive-value-variance Gaussian law and two
orthonormal tokens realize every field hypothesis. The scalar data
also realize the coefficient bound with nonzero c and q. -/
example : ∃ X : Idx 2 → EucSpace 2,
    (2 : ℕ) ≤ 2 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 0 (gaussHeadLaw 2 (stdSigmaV 2) 0) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧ (∀ k, ‖X k‖ = 1) ∧
    |(1 : ℝ)| ≤ 2 ∧ (1 : ℝ) ∈ Icc (0 : ℝ) (3 / 2) ∧
    |sphGenerator (gaussHeadLaw 2 (stdSigmaV 2) 0) (bField 1 (gaussHeadLaw 2 (stdSigmaV 2) 0))
      (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 0)) (exponentialOverlap 1 0 1) X| ≤
        Real.exp |(1 : ℝ)| * (2 * |(1 : ℝ)| + (1 : ℝ) ^ 2 / 2 * (3 / 2)) := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hX : ∀ k, ‖X k‖ = 1 := fun k => by simp [X]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 0
  refine ⟨X, le_rfl, hρ, stdSigmaV_sq 2, hX, by norm_num, by norm_num, ?_⟩
  exact abs_sphGenerator_exponentialOverlap_le le_rfl 1 1 hρ (stdSigmaV_sq 2) X hX 0 1

end Transformer.Homogenized
