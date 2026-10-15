/-
# Geometry and zero-temperature averages of the original mean overlap

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:def_m_smallbeta` and part (a) of `eq:drift_mismatch_target_clean`.
The original empirical mean overlap is the squared norm of the actual
token centroid, even without a simplex assumption.

For arbitrary unit token configurations it lies in `[0,1]`.
The mean squared Gram entry also lies in `[0,1]`. Empty configurations
are included in these geometric assertions through the original total
inverse convention. A positive common lower overlap bound gives a
positive mean when the original token set is nonempty.

At inverse temperature zero, each actual attention probability is `1/n`.
The original empirical softmax barycenter is the actual centroid, and
its original expected correlation is exactly the original mean overlap
under every probability head law. Neither a simplex condition nor a
Gaussian assumption is needed for these zero-temperature identities.
-/

import Transformer.Homogenized.GaussianInit
import Transformer.Homogenized.Section7_SimplexCentroid
import Transformer.Homogenized.AttnAverage
import Transformer.Homogenized.SimplexBary
import Transformer.Homogenized.GaussianEnsemble

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- `eq:def_m_smallbeta`: the empirical second moment
`m = n^{-2} Σ_i Σ_j ⟨x_i, x_j⟩`.

Source: arXiv:2604.01978v1, `eq:def_m_smallbeta`. -/
noncomputable def meanOverlap {d n : ℕ} (x : Idx n → EucSpace d) : ℝ :=
  ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, inner (𝕜 := ℝ) (x i) (x j)

/-- A configuration of equal unit tokens has second moment `1`.
Source: arXiv:2604.01978v1, §2.3.5, `eq:def_m_smallbeta`. -/
theorem meanOverlap_const {d n : ℕ} (v : EucSpace (d + 1)) (hv : ‖v‖ = 1) :
    meanOverlap (fun _ : Idx (n + 1) => v) = 1 := by
  have hvv : inner (𝕜 := ℝ) v v = 1 := by
    rw [real_inner_self_eq_norm_sq, hv, one_pow]
  have hn : ((n : ℝ) + 1) ≠ 0 := by positivity
  simp only [meanOverlap, hvv, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one, Nat.cast_add, Nat.cast_one]
  field_simp

/-- The original mean squared Gram entry in the high-temperature proof.
Source: arXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`, part (a). -/
noncomputable def meanSquareOverlap {d n : ℕ} (X : Idx n → EucSpace d) : ℝ :=
  ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, inner ℝ (X i) (X j) ^ 2

/-- The actual mean overlap is the squared norm of the actual centroid.
Source: arXiv:2604.01978v1, §7, `eq:def_m_smallbeta`, part (a) of the proof. -/
theorem meanOverlap_eq_norm_simplexCentroid_sq {d n : ℕ} (X : Idx n → EucSpace d) :
    meanOverlap X = ‖simplexCentroid X‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  simp only [meanOverlap, simplexCentroid, real_inner_smul_left, real_inner_smul_right,
    sum_inner, inner_sum, ← Finset.mul_sum]
  rw [← inv_pow, Finset.sum_comm]
  ring

/-- The original mean overlap is nonnegative for every configuration.
Source: arXiv:2604.01978v1, §7, the range of `m(X)` in the high-temperature proof. -/
theorem meanOverlap_nonneg {d n : ℕ} (X : Idx n → EucSpace d) : 0 ≤ meanOverlap X := by
  rw [meanOverlap_eq_norm_simplexCentroid_sq]
  exact sq_nonneg _

/-- The actual centroid of unit original tokens has norm at most one.
Source: arXiv:2604.01978v1, §7, the centroid in the proof at `β=0`. -/
theorem norm_simplexCentroid_le_one {d n : ℕ} (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    ‖simplexCentroid X‖ ≤ 1 := by
  have hs : ‖∑ i : Idx n, X i‖ ≤ (n : ℝ) := by
    simpa only [hX, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] using
      norm_sum_le Finset.univ X
  rw [simplexCentroid, norm_smul_of_nonneg (by positivity : 0 ≤ (n : ℝ)⁻¹)]
  apply (mul_le_mul_of_nonneg_left hs (by positivity)).trans
  by_cases hn : n = 0
  · simp [hn]
  · exact (inv_mul_cancel₀ (by exact_mod_cast hn : (n : ℝ) ≠ 0)).le

/-- The original mean overlap lies in the logistic state interval.
Source: arXiv:2604.01978v1, §7, the bounds on `m(X)` in part (a). -/
theorem meanOverlap_mem_Icc {d n : ℕ} (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    meanOverlap X ∈ Set.Icc (0 : ℝ) 1 := by
  refine ⟨meanOverlap_nonneg X, ?_⟩
  rw [meanOverlap_eq_norm_simplexCentroid_sq]
  simpa only [one_pow] using pow_le_pow_left₀ (norm_nonneg _) (norm_simplexCentroid_le_one X hX) 2

/-- A lower bound on the original off-diagonal overlaps also bounds
their actual mean when the original token set is nonempty.
Source: arXiv:2604.01978v1, §2.3.5 and §7,
the initial hemisphere hypothesis of `thm:clustering_small_beta`. -/
theorem le_meanOverlap_of_pairwise_lower {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) (γ : ℝ) (hγ : γ ≤ 1)
    (hpair : ∀ i j : Idx n, i ≠ j → γ ≤ inner ℝ (X i) (X j)) : γ ≤ meanOverlap X := by
  have he (i j : Idx n) : γ ≤ inner ℝ (X i) (X j) := by
    by_cases hij : i = j
    · subst j
      simpa only [real_inner_self_eq_norm_sq, hX, one_pow] using hγ
    · exact hpair i j hij
  have hs : (n : ℝ) ^ 2 * γ ≤ ∑ i : Idx n, ∑ j : Idx n, inner ℝ (X i) (X j) := by
    calc
      _ = ∑ _i : Idx n, ∑ _j : Idx n, γ := by simp [pow_two, mul_assoc]
      _ ≤ _ := Finset.sum_le_sum fun i hi => Finset.sum_le_sum fun j hj => he i j
  have hnR : (n : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 (by exact_mod_cast hn.ne')
  calc
    γ = ((n : ℝ) ^ 2)⁻¹ * ((n : ℝ) ^ 2 * γ) := by rw [← mul_assoc, inv_mul_cancel₀ hnR, one_mul]
    _ ≤ meanOverlap X := mul_le_mul_of_nonneg_left hs (by positivity)

/-- The mean squared Gram entry is nonnegative for every original configuration.
Source: arXiv:2604.01978v1, §7, the statistic `q(X)` in part (a). -/
theorem meanSquareOverlap_nonneg {d n : ℕ} (X : Idx n → EucSpace d) :
    0 ≤ meanSquareOverlap X := by
  unfold meanSquareOverlap
  exact mul_nonneg (by positivity) (Finset.sum_nonneg fun i hi => Finset.sum_nonneg fun j hj => sq_nonneg _)

/-- The actual squared Gram statistic also lies in the unit interval.
Source: arXiv:2604.01978v1, §7, the range of `q(X)` in the high-temperature proof. -/
theorem meanSquareOverlap_mem_Icc {d n : ℕ} (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) :
    meanSquareOverlap X ∈ Set.Icc (0 : ℝ) 1 := by
  refine ⟨meanSquareOverlap_nonneg X, ?_⟩
  have he (i j : Idx n) : inner ℝ (X i) (X j) ^ 2 ≤ 1 := by
    have ha : |inner ℝ (X i) (X j)| ≤ 1 := by simpa only [hX, mul_one] using abs_real_inner_le_norm (X i) (X j)
    simpa only [sq_abs, one_pow] using pow_le_pow_left₀ (abs_nonneg _) ha 2
  have hs : (∑ i : Idx n, ∑ j : Idx n, inner ℝ (X i) (X j) ^ 2) ≤ (n : ℝ) ^ 2 := by
    calc
      _ ≤ ∑ _i : Idx n, ∑ _j : Idx n, (1 : ℝ) :=
        Finset.sum_le_sum fun i hi => Finset.sum_le_sum fun j hj => he i j
      _ = _ := by simp [pow_two]
  unfold meanSquareOverlap
  apply (mul_le_mul_of_nonneg_left hs (by positivity)).trans
  by_cases hn : n = 0
  · simp [hn]
  · exact (inv_mul_cancel₀ (pow_ne_zero 2 (by exact_mod_cast hn : (n : ℝ) ≠ 0))).le

/-- At actual inverse temperature zero every original attention weight is uniform.
Source: arXiv:2604.01978v1, §7, proof of `eq:s_mu_small_beta_clean`, part (a). -/
theorem attnProb_zero_temperature {d n : ℕ} (A : Matrix (Fin d) (Fin d) ℝ)
    (X : Idx n → EucSpace d) (i k : Idx n) : attnProb 0 A X i k = (n : ℝ)⁻¹ := by
  simp [attnProb, attnWeight]

/-- The actual zero-temperature attention barycenter is the original centroid.
Source: arXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`, part (a). -/
theorem softBary_zero_temperature_empMeasure {d n : ℕ} (A : Matrix (Fin d) (Fin d) ℝ)
    (X : Idx n → EucSpace d) (i : Idx n) :
    softBary 0 A (empMeasure X) (X i) = simplexCentroid X := by
  rw [softBary_empMeasure]
  simp only [attnProb_zero_temperature, ← Finset.smul_sum, simplexCentroid]

/-- **The actual zero-temperature barycenter correlation is the original mean.**
This holds under every probability head law and for every original token pair.
Source: arXiv:2604.01978v1, §7, `s_μ(x_i,x_j)=m(X)` in part (a). -/
theorem baryCorr_zero_temperature_empMeasure {d n : ℕ} (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ]
    (X : Idx n → EucSpace d) (i j : Idx n) :
    baryCorr 0 ρ (empMeasure X) (X i) (X j) = meanOverlap X := by
  simp only [baryCorr, softBary_zero_temperature_empMeasure, real_inner_self_eq_norm_sq]
  simpa only [integral_const, probReal_univ, one_smul] using (meanOverlap_eq_norm_simplexCentroid_sq X).symm

/-- Actual orthogonal unit tokens and a positive-variance original Gaussian
head give a nonzero centroid and a genuinely intermediate mean overlap. -/
example : ∃ X : Idx 2 → EucSpace 2,
    (∀ i, ‖X i‖ = 1) ∧ meanOverlap X = 1 / 2 ∧ meanSquareOverlap X = 1 / 2 ∧
    IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) := by
  let X : Idx 2 → EucSpace 2 := fun i => EuclideanSpace.single i 1
  refine ⟨X, fun i => by simp [X], ?_, ?_, isGaussianHeadLaw_gaussHeadLaw 2 1 1⟩
  · norm_num [meanOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]
  · norm_num [meanSquareOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]

/-- The positive initial-overlap hypotheses hold jointly for an actual
nonempty unit configuration at the common lower bound one half. -/
example : (0 : ℕ) < 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    (∀ i j : Idx 2, i ≠ j → (1 / 2 : ℝ) ≤
      inner ℝ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) (EuclideanSpace.single 0 (1 : ℝ))) := by
  exact ⟨by decide, by norm_num, by simp, fun _ _ _ => by norm_num⟩

end Transformer.Homogenized
