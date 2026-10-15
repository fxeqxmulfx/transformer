/-
# The actual zero-temperature mean-overlap drift

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:F_def_clean` and part (a) of `eq:drift_mismatch_target_clean`.
The actual average of the original Gaussian overlap-drift formula is
computed without any simplex condition on the original tokens.

At inverse temperature zero its exact value is

  `g₀(X) = m(X)/d * (d-2 + q(X) - (d-1)m(X))`.

For original unit tokens the difference from the actual logistic drift
is nonpositive and has absolute value at most `2/d`. This follows from
the proved actual ranges of the original mean and squared Gram entry.
The source's isolated `f(u)=r(1-u)` has an unbound `r`; the logistic
drift here is `m(1-m)`, as in `eq:logistic_u_smallbeta`.

The drift formula uses the source's normalized value variance `1/d`.
The algebraic identities concern its explicit `overlapDrift` formula;
identification with the original Gaussian generator separately requires
the original value-variance hypothesis of `ito_formula_overlap`.
-/

import Transformer.Homogenized.Section7_MeanOverlap
import Transformer.Homogenized.OverlapDrift

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual average of the original overlap drifts.
Source: arXiv:2604.01978v1, §7, `eq:F_def_clean`, at diffusive time scale one. -/
noncomputable def meanOverlapDrift {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : Idx n → EucSpace d) : ℝ :=
  ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, overlapDrift β ρ X i j

/-- **The exact original zero-temperature drift of the mean overlap.**
No simplex assumption or unproved probabilistic input is used.
Source: arXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`, part (a). -/
theorem meanOverlapDrift_zero_temperature {d n : ℕ} (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ]
    (X : Idx n → EucSpace d) :
    meanOverlapDrift 0 ρ X = meanOverlap X / (d : ℝ) *
      ((d : ℝ) - 2 + meanSquareOverlap X - ((d : ℝ) - 1) * meanOverlap X) := by
  by_cases hn : n = 0
  · subst n
    simp [meanOverlapDrift, meanOverlap, meanSquareOverlap]
  have hnR : (n : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 (by exact_mod_cast hn)
  have hRsum : (∑ i : Idx n, ∑ j : Idx n, inner ℝ (X i) (X j)) =
      (n : ℝ) ^ 2 * meanOverlap X := by
    rw [meanOverlap, ← mul_assoc, mul_inv_cancel₀ hnR, one_mul]
  have hQsum : (∑ i : Idx n, ∑ j : Idx n, inner ℝ (X i) (X j) ^ 2) =
      (n : ℝ) ^ 2 * meanSquareOverlap X := by
    rw [meanSquareOverlap, ← mul_assoc, mul_inv_cancel₀ hnR, one_mul]
  have hD (i j : Idx n) : overlapDrift 0 ρ X i j = meanOverlap X / (d : ℝ) *
      (((d : ℝ) - 2) + inner ℝ (X i) (X j) ^ 2 - ((d : ℝ) - 1) * inner ℝ (X i) (X j)) := by
    simp only [overlapDrift, baryCorr_zero_temperature_empMeasure]
    ring
  have hsum : (∑ i : Idx n, ∑ j : Idx n,
      (((d : ℝ) - 2) + inner ℝ (X i) (X j) ^ 2 - ((d : ℝ) - 1) * inner ℝ (X i) (X j))) =
      (n : ℝ) ^ 2 * ((d : ℝ) - 2 + meanSquareOverlap X - ((d : ℝ) - 1) * meanOverlap X) := by
    simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [hRsum, hQsum]
    ring
  unfold meanOverlapDrift
  simp_rw [hD, ← Finset.mul_sum]
  rw [hsum]
  calc
    _ = (((n : ℝ) ^ 2)⁻¹ * (n : ℝ) ^ 2) *
        (meanOverlap X / (d : ℝ) * ((d : ℝ) - 2 + meanSquareOverlap X - ((d : ℝ) - 1) * meanOverlap X)) := by ring
    _ = _ := by rw [inv_mul_cancel₀ hnR, one_mul]

/-- The exact dimension correction to the source's logistic mean drift.
Source: arXiv:2604.01978v1, §7, part (a) of `eq:drift_mismatch_target_clean`,
with `f(m)=m(1-m)` as in the original logistic Cauchy problem. -/
theorem meanOverlapDrift_zero_sub_logistic {d n : ℕ} (hd : 0 < d)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d) :
    meanOverlapDrift 0 ρ X - meanOverlap X * (1 - meanOverlap X) =
      meanOverlap X / (d : ℝ) * (meanSquareOverlap X - 2 + meanOverlap X) := by
  have hdR : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  rw [meanOverlapDrift_zero_temperature]
  field_simp
  ring

/-- The actual zero-temperature finite-dimensional drift is below its
logistic approximation for every original unit configuration.
Source: arXiv:2604.01978v1, §7, the signed correction in part (a). -/
theorem meanOverlapDrift_zero_le_logistic {d n : ℕ} (hd : 0 < d)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d)
    (hX : ∀ i, ‖X i‖ = 1) : meanOverlapDrift 0 ρ X ≤ meanOverlap X * (1 - meanOverlap X) := by
  have hm := meanOverlap_mem_Icc X hX
  have hq := meanSquareOverlap_mem_Icc X hX
  have he := meanOverlapDrift_zero_sub_logistic hd ρ X
  have hsign : meanOverlap X / (d : ℝ) * (meanSquareOverlap X - 2 + meanOverlap X) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos (div_nonneg hm.1 (Nat.cast_nonneg d)) (by linarith [hm.2, hq.2])
  linarith

/-- **Uniform actual zero-temperature drift mismatch, at rate `2/d`.**
Source: arXiv:2604.01978v1, §7, the conclusion of part (a)
in the proof of `eq:drift_mismatch_target_clean`. -/
theorem abs_meanOverlapDrift_zero_sub_logistic_le {d n : ℕ} (hd : 0 < d)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d)
    (hX : ∀ i, ‖X i‖ = 1) :
    |meanOverlapDrift 0 ρ X - meanOverlap X * (1 - meanOverlap X)| ≤ 2 / (d : ℝ) := by
  have hm := meanOverlap_mem_Icc X hX
  have hq := meanSquareOverlap_mem_Icc X hX
  rw [meanOverlapDrift_zero_sub_logistic hd]
  have hsign : meanOverlap X / (d : ℝ) * (meanSquareOverlap X - 2 + meanOverlap X) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos (div_nonneg hm.1 (Nat.cast_nonneg d)) (by linarith [hm.2, hq.2])
  rw [abs_of_nonpos hsign]
  calc
    _ = (meanOverlap X * (2 - meanSquareOverlap X - meanOverlap X)) / (d : ℝ) := by ring
    _ ≤ _ := div_le_div_of_nonneg_right
      (by nlinarith [mul_nonneg hm.1 hq.1, sq_nonneg (meanOverlap X)]) (Nat.cast_nonneg d)

/-- A common bound on actual pair errors bounds their actual normalized
average. Source: arXiv:2604.01978v1, §7, the averaging step from
`eq:Dij_explicit_clean` to `eq:F_def_clean`. -/
theorem abs_empirical_pair_average_le {n : ℕ} (hn : 0 < n)
    (F : Idx n → Idx n → ℝ) (C : ℝ) (hF : ∀ i j, |F i j| ≤ C) :
    |((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, F i j| ≤ C := by
  have hnR : (n : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 (by exact_mod_cast hn.ne')
  have hs : |∑ i : Idx n, ∑ j : Idx n, F i j| ≤ (n : ℝ) ^ 2 * C := by
    calc
      _ ≤ ∑ i : Idx n, |∑ j : Idx n, F i j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i : Idx n, ∑ j : Idx n, |F i j| :=
        Finset.sum_le_sum fun i hi => Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Idx n, ∑ _j : Idx n, C :=
        Finset.sum_le_sum fun i hi => Finset.sum_le_sum fun j hj => hF i j
      _ = _ := by simp [pow_two, mul_assoc]
  rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ ((n : ℝ) ^ 2)⁻¹)]
  exact (mul_le_mul_of_nonneg_left hs (by positivity)).trans_eq
    (by rw [← mul_assoc, inv_mul_cancel₀ hnR, one_mul])

/-- An actual two-token configuration in dimension three has nonzero
original Gaussian drift `1/12`, at the source's positive value variance. -/
example : ∃ X : Idx 2 → EucSpace 3,
    (0 : ℕ) < 3 ∧ (0 : ℕ) < 2 ∧ (∀ i, ‖X i‖ = 1) ∧
    IsGaussianHeadLaw 3 (stdSigmaV 3) 1 (gaussHeadLaw 3 (stdSigmaV 3) 1) ∧
    meanOverlapDrift 0 (gaussHeadLaw 3 (stdSigmaV 3) 1) X = 1 / 12 := by
  let X : Idx 2 → EucSpace 3 :=
    ![EuclideanSpace.single (0 : Fin 3) 1, EuclideanSpace.single (1 : Fin 3) 1]
  have hX : ∀ i, ‖X i‖ = 1 := by intro i; fin_cases i <;> simp [X]
  have hm : meanOverlap X = 1 / 2 := by
    norm_num [meanOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]
  have hq : meanSquareOverlap X = 1 / 2 := by
    norm_num [meanSquareOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]
  refine ⟨X, by decide, by decide, hX, isGaussianHeadLaw_gaussHeadLaw 3 _ _, ?_⟩
  rw [meanOverlapDrift_zero_temperature, hm, hq]
  norm_num

/-- A nonzero actual two-index error array satisfies the common-bound
and nonempty-index hypotheses of the averaging inequality jointly. -/
example : (0 : ℕ) < 2 ∧ ∀ i j : Idx 2, |(i : ℝ) - (j : ℝ)| ≤ 1 := by
  refine ⟨by decide, ?_⟩
  intro i j
  fin_cases i <;> fin_cases j <;> norm_num

end Transformer.Homogenized
