/-
# The actual Gaussian mean drift is uniformly close to logistic drift

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:drift_mismatch_target_clean` and the averaging after
`eq:s_mu_small_beta_clean`. The original Gaussian correlations
have a proved temperature error `20 β² d² σA⁴`. Substituting them
in the genuine overlap drift and averaging gives an original mean
drift error `2/d + 80 β² d² σA⁴` from the logistic vector field.

At the source's adopted value variance `σV² = 1/d`, the actual
original Gaussian sphere generator on the mean therefore differs
from `m(1-m)` by at most `82 max(1/d, β² d² σA⁴)`.
Every correlation estimate and generator identity is proved from
the original head law, rather than assumed as an input to this bound.

The signed bounds also retain the actual zero-temperature correction
`-2m/d` in the lower drift bound and its nonpositivity in the upper
bound. They use arbitrary original unit configurations without a
simplex condition. The source's isolated unbound `r` in `f(u)=r(1-u)`
is replaced by its actual logistic field `f(u)=u(1-u)`.
-/

import Transformer.Homogenized.Section7_SmallTemperatureCorrelation
import Transformer.Homogenized.Section7_OverlapDriftGramComparison

open Real MeasureTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The genuine original pair drift changes quadratically with
temperature, uniformly in token count.
Source: arXiv:2604.01978v1, §7, the substitution in
`eq:Dij_explicit_clean` after `eq:s_mu_small_beta_clean`. -/
theorem abs_overlapDrift_small_sub_zero {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    |overlapDrift β ρ X i j - overlapDrift 0 ρ X i j| ≤
      80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hm := meanOverlap_mem_Icc X hX
  have hr : |inner ℝ (X i) (X j)| ≤ 1 := by simpa only [hX, mul_one] using abs_real_inner_le_norm (X i) (X j)
  have hb (k l : Idx n) := abs_baryCorr_sub_meanOverlap_le_small_temperature hn β hβ hρ X hX k l
  have h := abs_overlapDrift_scalar_sub_le (d : ℝ) (inner ℝ (X i) (X j)) (inner ℝ (X i) (X j))
    (baryCorr β ρ (empMeasure X) (X i) (X j)) (meanOverlap X)
    (baryCorr β ρ (empMeasure X) (X i) (X i)) (baryCorr β ρ (empMeasure X) (X j) (X j))
    (meanOverlap X) (meanOverlap X) 1 (20 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
    (by exact_mod_cast hd) hr hr (by simpa only [abs_of_nonneg hm.1] using hm.2)
    (by rw [abs_of_nonneg (by positivity [hm.1])]; linarith [hm.2]) le_rfl (by positivity)
    (by simp only [sub_self, abs_zero]; positivity)
    (by simpa only [one_mul] using hb i j) (by simpa only [one_mul] using hb i i)
    (by simpa only [one_mul] using hb j j)
  simp only [overlapDrift, baryCorr_zero_temperature_empMeasure]
  convert h using 1
  ring

/-- The actual normalized pair average preserves the original
quadratic temperature error.
Source: arXiv:2604.01978v1, §7, the averaging in `eq:F_def_clean`. -/
theorem abs_meanOverlapDrift_small_sub_zero {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |meanOverlapDrift β ρ X - meanOverlapDrift 0 ρ X| ≤
      80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 := by
  have h := abs_empirical_pair_average_le hn (fun i j => overlapDrift β ρ X i j - overlapDrift 0 ρ X i j)
    _ (fun i j => abs_overlapDrift_small_sub_zero hd hn β hβ hρ X hX i j)
  simpa only [meanOverlapDrift, Finset.sum_sub_distrib, mul_sub] using h

/-- The original zero-temperature lower drift bound retains its
actual mean factor. Source: arXiv:2604.01978v1, §7, the exact
dimension correction in part (a) of `eq:drift_mismatch_target_clean`. -/
theorem meanOverlapDrift_zero_ge_logistic_sub {d n : ℕ} (hd : 0 < d)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) :
    meanOverlap X * (1 - meanOverlap X) - 2 * meanOverlap X / (d : ℝ) ≤ meanOverlapDrift 0 ρ X := by
  have hm := meanOverlap_mem_Icc X hX
  have hq := meanSquareOverlap_mem_Icc X hX
  have he := meanOverlapDrift_zero_sub_logistic hd ρ X
  have hp := mul_le_mul_of_nonneg_left
    (show (-2 : ℝ) ≤ meanSquareOverlap X - 2 + meanOverlap X by linarith [hm.1, hq.1])
    (div_nonneg hm.1 (Nat.cast_nonneg d))
  rw [show meanOverlap X / (d : ℝ) * (-2) = -2 * meanOverlap X / (d : ℝ) by ring, ← he] at hp
  calc
    _ = meanOverlap X * (1 - meanOverlap X) + (-2 * meanOverlap X / (d : ℝ)) := by ring
    _ ≤ meanOverlap X * (1 - meanOverlap X) +
        (meanOverlapDrift 0 ρ X - meanOverlap X * (1 - meanOverlap X)) := add_le_add_right hp _
    _ = _ := by ring

/-- The actual small-temperature mean drift has signed logistic
comparison bounds for every original unit configuration.
Source: arXiv:2604.01978v1, §7, parts (a) and (b) of
`eq:drift_mismatch_target_clean`, retaining the original mean factor. -/
theorem meanOverlapDrift_small_temperature_mem_Icc {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    meanOverlapDrift β ρ X ∈ Icc
      (meanOverlap X * (1 - meanOverlap X) - 2 * meanOverlap X / (d : ℝ) -
        80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
      (meanOverlap X * (1 - meanOverlap X) + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hd0 : 0 < d := by omega
  have h := abs_le.mp (abs_meanOverlapDrift_small_sub_zero hd hn β hβ hρ X hX)
  have hl := meanOverlapDrift_zero_ge_logistic_sub hd0 ρ X hX
  have hu := meanOverlapDrift_zero_le_logistic hd0 ρ X hX
  constructor <;> linarith [h.1, h.2]

/-- The original mean drift has the source's uniform dimension and
temperature mismatch from its logistic field.
Source: arXiv:2604.01978v1, §7, `eq:drift_mismatch_target_clean`. -/
theorem abs_meanOverlapDrift_small_sub_logistic_le {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |meanOverlapDrift β ρ X - meanOverlap X * (1 - meanOverlap X)| ≤
      2 / (d : ℝ) + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4 := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  exact (abs_sub_le _ (meanOverlapDrift 0 ρ X) _).trans
    ((add_le_add (abs_meanOverlapDrift_small_sub_zero hd hn β hβ hρ X hX)
      (abs_meanOverlapDrift_zero_sub_logistic_le (by omega) ρ X hX)).trans_eq (by ring))

/-- **The genuine original Gaussian sphere generator has the full
source logistic mean-drift approximation at small temperature.**
Source: arXiv:2604.01978v1, §7, `eq:drift_mismatch_target_clean`,
with §2.3.3's adopted `σV² = 1/d` made explicit. -/
theorem abs_sphGenerator_meanOverlap_small_sub_logistic_le {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) meanOverlap X -
        meanOverlap X * (1 - meanOverlap X)| ≤
      82 * max (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by
  rw [sphGenerator_meanOverlap_gaussian β hρ hσ X hX]
  apply (abs_meanOverlapDrift_small_sub_logistic_le hd hn β hβ hρ X hX).trans
  have h1 := le_max_left (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
  have h2 := le_max_right (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
  calc
    _ = 2 * (1 / (d : ℝ)) + 80 * (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by ring
    _ ≤ 2 * max (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) +
        80 * max (1 / (d : ℝ)) (β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) :=
      add_le_add (mul_le_mul_of_nonneg_left h1 (by norm_num)) (mul_le_mul_of_nonneg_left h2 (by norm_num))
    _ = _ := by ring

/-- The genuine original Gaussian generator also retains the signed
comparison with the logistic field and its original `2m/d` correction.
Source: arXiv:2604.01978v1, §7, parts (a) and (b) of
`eq:drift_mismatch_target_clean`, at §2.3.3's adopted value variance. -/
theorem sphGenerator_meanOverlap_small_temperature_mem_Icc {d n : ℕ} (hd : 2 ≤ d) (hn : 0 < n)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) meanOverlap X ∈ Icc
      (meanOverlap X * (1 - meanOverlap X) - 2 * meanOverlap X / (d : ℝ) -
        80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4)
      (meanOverlap X * (1 - meanOverlap X) + 80 * β ^ 2 * (d : ℝ) ^ 2 * (σA : ℝ) ^ 4) := by
  rw [sphGenerator_meanOverlap_gaussian β hρ hσ X hX]
  exact meanOverlapDrift_small_temperature_mem_Icc hd hn β hβ hρ X hX

/-- Positive dimension, nonempty orthogonal unit queries, the original
Gaussian head and the source's adopted variance satisfy all bounds jointly. -/
example : ∃ X : Idx 2 → EucSpace 2, (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    ((stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ)) ∧ (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 :=
  ⟨fun k => EuclideanSpace.single k 1, le_rfl, by decide, fun k => by simp,
    isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, by norm_num⟩

end Transformer.Homogenized
