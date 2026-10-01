/-
# Bounded-amplitude second and fourth moment estimates

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The constants depend on the amplitude bound and elapsed time, rather
than the number of Brownian grid intervals.
-/

import Transformer.BatchSize.Section4_ItoFourthStep

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A bounded adapted Brownian sum has second moment at most A^2
times its elapsed time, Section 4.3 (2)--(3). -/
theorem brownianItoSum_secondMoment_bound {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (hmono : Monotone t) (H : ℕ → BrownianSample d → ℝ)
    (hH : ∀ j, StronglyMeasurable[brownianFiltration d (t j)] (H j))
    (A : ℝ≥0) (hbound : ∀ j ω, |H j ω| ≤ A) (n : ℕ) :
    (∫ ω, brownianItoSum k t H n ω ^ 2 ∂brownianNoiseLaw d) ≤
      A ^ 2 * ((t n : ℝ) - t 0) := by
  have hHL2 (j : ℕ) : MemLp (H j) 2 (brownianNoiseLaw d) := MemLp.of_bound
    ((hH j).mono ((brownianFiltration d).le (t j))).aestronglyMeasurable A
    (Eventually.of_forall fun ω => by simpa only [Real.norm_eq_abs] using hbound j ω)
  have hcoef (j : ℕ) : (∫ ω, H j ω ^ 2 ∂brownianNoiseLaw d) ≤ (A : ℝ) ^ 2 := by
    have h := integral_mono (hHL2 j).integrable_sq (integrable_const ((A : ℝ) ^ 2)) fun ω => by
      simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) A.coe_nonneg).mpr (hbound j ω)
    simpa using h
  rw [brownianItoSum_isometry k t hmono H hH hHL2 n]
  calc
    _ ≤ ∑ j ∈ Finset.range n, (A : ℝ) ^ 2 * ((t (j + 1) : ℝ) - t j) := by
      apply Finset.sum_le_sum
      intro j hj
      exact mul_le_mul_of_nonneg_right (hcoef j)
        (sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hmono (Nat.le_succ j))))
    _ = _ := by rw [← Finset.mul_sum, Finset.sum_range_sub (fun j => (t j : ℝ)) n]

/-- A bounded-amplitude fourth-moment update estimate derived from
the exact adapted identity, Section 4.3 (2)--(3). -/
theorem brownianIncrement_fourth_energy_bound {d : ℕ} (k : Fin d)
    (Z H : BrownianSample d → ℝ) (s t : ℝ≥0) (hst : s ≤ t)
    (hZ : StronglyMeasurable[brownianFiltration d s] Z)
    (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hZL4 : MemLp Z 4 (brownianNoiseLaw d)) (A : ℝ≥0) (hbound : ∀ ω, |H ω| ≤ A) :
    (∫ ω, (Z ω + H ω * brownianIncrement k s t ω) ^ 4 ∂brownianNoiseLaw d) ≤
      (∫ ω, Z ω ^ 4 ∂brownianNoiseLaw d) +
        6 * A ^ 2 * ((t : ℝ) - s) * (∫ ω, Z ω ^ 2 ∂brownianNoiseLaw d) +
          3 * A ^ 4 * ((t : ℝ) - s) ^ 2 := by
  have hHm : AEStronglyMeasurable H (brownianNoiseLaw d) :=
    (hH.mono ((brownianFiltration d).le s)).aestronglyMeasurable
  have hHL4 : MemLp H 4 (brownianNoiseLaw d) := MemLp.of_bound hHm A
    (Eventually.of_forall fun ω => by simpa only [Real.norm_eq_abs] using hbound ω)
  have hH2m : AEStronglyMeasurable (fun ω => H ω ^ 2) (brownianNoiseLaw d) :=
    (continuous_pow 2).comp_aestronglyMeasurable hHm
  have hH2bound (ω : BrownianSample d) : H ω ^ 2 ≤ (A : ℝ) ^ 2 := by
    simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) A.coe_nonneg).mpr (hbound ω)
  have hZ2i := memLp_four_integrable_pow (brownianNoiseLaw d) Z hZL4 2 (by norm_num)
  have hCi : Integrable (fun ω => Z ω ^ 2 * H ω ^ 2) (brownianNoiseLaw d) :=
    hZ2i.mul_bdd (c := (A : ℝ) ^ 2) hH2m (Eventually.of_forall fun ω => by
      simpa only [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (H ω))] using hH2bound ω)
  have hCbound : (∫ ω, Z ω ^ 2 * H ω ^ 2 ∂brownianNoiseLaw d) ≤
      A ^ 2 * (∫ ω, Z ω ^ 2 ∂brownianNoiseLaw d) := by
    have h := integral_mono hCi (hZ2i.const_mul ((A : ℝ) ^ 2)) fun ω => by
      exact (mul_le_mul_of_nonneg_left (hH2bound ω) (sq_nonneg (Z ω))).trans_eq (mul_comm _ _)
    simpa only [integral_const_mul] using h
  have hH4bound : (∫ ω, H ω ^ 4 ∂brownianNoiseLaw d) ≤ (A : ℝ) ^ 4 := by
    have hi := memLp_four_integrable_pow (brownianNoiseLaw d) H hHL4 4 le_rfl
    have h := integral_mono hi (integrable_const ((A : ℝ) ^ 4)) fun ω => by
      have hs := pow_le_pow_left₀ (sq_nonneg (H ω)) (hH2bound ω) 2
      simpa only [← pow_mul] using hs
    simpa using h
  rw [brownianIncrement_adapted_fourth_energy k Z H s t hst hZ hH hZL4 A hbound]
  have hδ : 0 ≤ (t : ℝ) - s := sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst)
  calc
    _ ≤ (∫ ω, Z ω ^ 4 ∂brownianNoiseLaw d) +
        6 * ((t : ℝ) - s) * (A ^ 2 * (∫ ω, Z ω ^ 2 ∂brownianNoiseLaw d)) +
          3 * ((t : ℝ) - s) ^ 2 * A ^ 4 := by
      gcongr
    _ = _ := by ring

/-- Joint nonvacuity of bounded moment estimates, Section 4.3:
unit amplitudes on unit times and a nonconstant past Gaussian state. -/
example : Monotone (fun j : ℕ => (j : ℝ≥0)) ∧
    (∀ j : ℕ, StronglyMeasurable[brownianFiltration 1 j]
      ((fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) j)) ∧
    (∀ (j : ℕ) (ω : BrownianSample 1),
      |(fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) j ω| ≤ (1 : ℝ≥0)) ∧
    (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (coordinateBrownian (0 : Fin 1) 1) ∧
    MemLp (coordinateBrownian (0 : Fin 1) 1) 4 (brownianNoiseLaw 1) :=
  ⟨fun i j hij => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast hij,
    fun _ => stronglyMeasurable_const, by simp,
    by norm_num, (coordinateBrownian_filtered (0 : Fin 1)).stronglyAdapted 1,
    (coordinateBrownian_isBrownian (0 : Fin 1)).isGaussianProcess.hasGaussianLaw_eval 1 |>.memLp
      (by norm_num)⟩

end Transformer.BatchSize
