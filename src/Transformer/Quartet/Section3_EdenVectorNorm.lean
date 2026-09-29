/-
# Magnitude and projection of the MS-EDEN counterexample

Source: arXiv:2601.22813v2, §3.2–§3.3 and Algorithm 1.
-/

import Transformer.Quartet.Section3_EdenVectorBias

open MeasureTheory

namespace Transformer.Quartet

variable {k : ℕ}

/-- The squared Euclidean norm of the witness from §3.3. -/
theorem edenPair_sqNorm (k : ℕ) :
    (∑ i : Fin (2 ^ k), ∑ j : Fin 16, (edenPair k i j) ^ 2) = 101 / 100 := by
  rw [Finset.sum_eq_single (0 : Fin (2 ^ k))
    (fun i _ hi => Finset.sum_eq_zero (fun j _ => by simp [edenPair, hi])) (by simp)]
  rw [Finset.sum_eq_add_of_mem (0 : Fin 16) 1 (by simp) (by simp)
    (by decide) (fun j _ hj => by simp [edenPair, hj.1, hj.2])]
  norm_num [edenPair]

/-- The squared Euclidean norm of the bias in the §3.3 witness. -/
theorem edenPair_bias_sqNorm (k : ℕ) :
    (∑ i : Fin (2 ^ k), ∑ j : Fin 16,
      (edenMeanTarget k i j - edenPair k i j) ^ 2) = 101 / 10404 := by
  rw [Finset.sum_eq_single (0 : Fin (2 ^ k))
    (fun i _ hi => Finset.sum_eq_zero (fun j _ => by simp [edenMeanTarget, twoSupport, edenPair, hi]))
    (by simp)]
  rw [Finset.sum_eq_add_of_mem (0 : Fin 16) 1 (by simp) (by simp)
    (by decide) (fun j _ hj => by simp [edenMeanTarget, twoSupport, edenPair, hj.1, hj.2])]
  norm_num [edenMeanTarget, twoSupport, edenPair]

/-- The Euclidean norm of the exact §3.3 mean bias relative to the witness is `5/51`. -/
theorem edenPair_relativeBias (k : ℕ) :
    Real.sqrt (∑ i : Fin (2 ^ k), ∑ j : Fin 16,
      (edenMeanTarget k i j - edenPair k i j) ^ 2) /
      Real.sqrt (∑ i : Fin (2 ^ k), ∑ j : Fin 16, (edenPair k i j) ^ 2) = 5 / 51 := by
  rw [edenPair_bias_sqNorm, edenPair_sqNorm]
  have hd : 0 < Real.sqrt (101 / 100 : ℝ) := Real.sqrt_pos.2 (by norm_num)
  have hn : 0 ≤ Real.sqrt (101 / 10404 : ℝ) / Real.sqrt (101 / 100 : ℝ) :=
    div_nonneg (Real.sqrt_nonneg _) hd.le
  have hsq : (Real.sqrt (101 / 10404 : ℝ) / Real.sqrt (101 / 100 : ℝ)) ^ 2 =
      (5 / 51 : ℝ) ^ 2 := by
    rw [div_pow, Real.sq_sqrt (by norm_num), Real.sq_sqrt (by norm_num)]
    norm_num
  nlinarith

/-- The EDEN scalar correction preserves the witness's projection exactly (§3.3). -/
theorem edenMeanTarget_projection (k : ℕ) :
    (∑ i : Fin (2 ^ k), ∑ j : Fin 16,
      edenMeanTarget k i j * edenPair k i j) = 101 / 100 := by
  rw [Finset.sum_eq_single (0 : Fin (2 ^ k))
    (fun i _ hi => Finset.sum_eq_zero (fun j _ => by simp [edenMeanTarget, twoSupport, edenPair, hi]))
    (by simp)]
  rw [Finset.sum_eq_add_of_mem (0 : Fin 16) 1 (by simp) (by simp)
    (by decide) (fun j _ hj => by simp [edenMeanTarget, twoSupport, edenPair, hj.1, hj.2])]
  norm_num [edenMeanTarget, twoSupport, edenPair]

/-- The §3.3 counterexample's conditional mean preserves the projection on the source vector. -/
theorem integral_edenPair_projection
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) :
    (∑ i : Fin (2 ^ k), ∑ j : Fin 16,
      (∫ u in edenCoinCube k, rhtInv k ε (msEden k s (edenPair k) ε u) i j) *
        edenPair k i j) = 101 / 100 := by
  simp_rw [integral_rhtInv_msEden_edenPair_vector hs₀ hs₁ ε]
  exact edenMeanTarget_projection k

/-- The relative Euclidean bias of the actual §3.3 mean is exactly `5/51`. -/
theorem mean_rhtInv_msEden_edenPair_relativeBias (k : ℕ) {s : ℝ}
    (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93) :
    Real.sqrt (∑ i : Fin (2 ^ k), ∑ j : Fin 16,
      (mean k (fun ε u => rhtInv k ε (msEden k s (edenPair k) ε u) i j) -
        edenPair k i j) ^ 2) /
      Real.sqrt (∑ i : Fin (2 ^ k), ∑ j : Fin 16, (edenPair k i j) ^ 2) =
        5 / 51 := by
  simp_rw [mean_rhtInv_msEden_edenPair_vector hs₀ hs₁]
  exact edenPair_relativeBias k

/-- The paper's `d = 128` and chosen scale satisfy the relative-bias hypotheses. -/
example :
    Real.sqrt (∑ i : Fin (2 ^ 3), ∑ j : Fin 16,
      (mean 3 (fun ε u => rhtInv 3 ε
        (msEden 3 (3200 / 527) (edenPair 3) ε u) i j) - edenPair 3 i j) ^ 2) /
      Real.sqrt (∑ i : Fin (2 ^ 3), ∑ j : Fin 16, (edenPair 3 i j) ^ 2) =
        5 / 51 :=
  mean_rhtInv_msEden_edenPair_relativeBias 3 (by norm_num) (by norm_num)

/-- The projection claim holds at the scale and rotation size used in §3.3. -/
example (ε : Fin (2 ^ 3) → Fin 16 → Bool) :
    (∑ i : Fin (2 ^ 3), ∑ j : Fin 16,
      (∫ u in edenCoinCube 3,
        rhtInv 3 ε (msEden 3 (3200 / 527) (edenPair 3) ε u) i j) *
          edenPair 3 i j) = 101 / 100 :=
  integral_edenPair_projection (by norm_num) (by norm_num) ε

end Transformer.Quartet
