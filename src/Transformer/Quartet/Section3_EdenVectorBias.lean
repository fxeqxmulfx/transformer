/-
# Exact full-vector bias of MS-EDEN on a two-coordinate witness

Source: arXiv:2601.22813v2, §3.2–§3.3 and Algorithm 1.
-/

import Transformer.Quartet.Section3_EdenVectorCore

open MeasureTheory

namespace Transformer.Quartet

variable {k : ℕ}

/-- The two-coordinate mean predicted by the §3.3 witness calculation. -/
noncomputable def edenMeanTarget (k : ℕ) : Fin (2 ^ k) → Fin 16 → ℝ :=
  twoSupport k (101 / 102) (101 / 510)

/-- A sign case of the full-vector mean of §3.3, Algorithm 1 on the two-coordinate witness. -/
theorem integral_edenPair_vector_case
    {ε : Fin (2 ^ k) → Fin 16 → Bool}
    {s P R qP qR σ₀ σ₁ a : ℝ}
    (hy : rht k ε (edenPair k) = twoValued k P R)
    (hσ₀ : (if ε 0 0 then (-1 : ℝ) else 1) = σ₀)
    (hσ₁ : (if ε 0 1 then (-1 : ℝ) else 1) = σ₁)
    (ha : a = 1 / Real.sqrt (2 ^ (k + 4)) / 10)
    (hs : 5 < s) (hs' : s ≤ 61 / 10)
    (hP : |P| ≤ 11 * a) (hR : |R| ≤ 11 * a)
    (h : |P| = 11 * a ∨ |R| = 11 * a)
    (hqP : rtn fp4 (P * s / (11 * a)) = qP)
    (hqR : rtn fp4 (R * s / (11 * a)) = qR)
    (hPR : P ^ 2 + R ^ 2 = 202 * a ^ 2)
    (hD : P * qP + R * qR = 102 * a)
    (hqPval : qP = 5 * σ₀ + σ₁) (hqRval : qR = 5 * σ₀ - σ₁)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, rhtInv k ε (msEden k s (edenPair k) ε u) i j =
      edenMeanTarget k i j := by
  have ha0 : 0 < a := by rw [ha]; positivity
  have hM : 0 < 11 * a := by positivity
  have hs0 : 0 < s := by linarith
  have hD0 : 0 < P * qP + R * qR := by rw [hD]; positivity
  have hz : s * (P ^ 2 + R ^ 2) / (11 * a * (P * qP + R * qR)) * 256 ≤ 448 := by
    rw [hPR, hD, show s * (202 * a ^ 2) / (11 * a * (102 * a)) * 256 =
      s * (202 * 256 / 1122) by field_simp; norm_num]
    linarith
  rw [integral_rhtInv_msEden_twoValued_vector hy hs0 hM hP hR h hqP hqR hD0 hz]
  have hratio : (P ^ 2 + R ^ 2) / (P * qP + R * qR) = 101 / 51 * a := by
    rw [hPR, hD]
    field_simp
    ring
  have hrot : twoValued k
      (qP * ((P ^ 2 + R ^ 2) / (P * qP + R * qR)))
      (qR * ((P ^ 2 + R ^ 2) / (P * qP + R * qR))) = rht k ε (edenMeanTarget k) := by
    funext i' j'
    unfold edenMeanTarget
    rw [rht_twoSupport]
    unfold twoValued
    rw [hσ₀, hσ₁, hratio, hqPval, hqRval]
    by_cases hj : (j' : ℕ) % 2 = 0 <;> simp [hj, ha] <;> field_simp <;> ring
  rw [hrot, rhtInv_rht]

/-- §3.3: every fixed sign seed has the same full-vector coin mean on the witness. -/
theorem integral_rhtInv_msEden_edenPair_vector
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, rhtInv k ε (msEden k s (edenPair k) ε u) i j =
      edenMeanTarget k i j := by
  norm_num at hs₀ hs₁
  have hs5 : 5 < s := by linarith
  have hs61 : s ≤ 61 / 10 := by linarith
  have hy := rht_edenPair (k := k) ε
  obtain ⟨a, ha⟩ : ∃ a : ℝ, a = 1 / Real.sqrt (2 ^ (k + 4)) / 10 := ⟨_, rfl⟩
  have ha0 : 0 < a := by rw [ha]; positivity
  rcases h0 : ε 0 0 with _ | _ <;> rcases h1 : ε 0 1 with _ | _
  · have hy' : rht k ε (edenPair k) = twoValued k (11 * a) (9 * a) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> (simp [hj, ha]; ring)
    exact integral_edenPair_vector_case (qP := 6) (qR := 4) (σ₀ := 1) (σ₁ := 1) hy'
      (by rw [h0]; simp) (by rw [h1]; simp) ha hs5 hs61
      (abs_of_pos (by positivity)).le
      (by rw [abs_of_pos (by positivity)]; linarith)
      (Or.inl (abs_of_pos (by positivity)))
      (by rw [show 11 * a * s / (11 * a) = s by field_simp]; exact rtn_fp4_six hs5)
      (by rw [show 9 * a * s / (11 * a) = 9 / 11 * s by field_simp]
          exact rtn_fp4_four (by linarith) (by linarith))
      (by ring) (by ring) (by norm_num) (by norm_num) i j
  · have hy' : rht k ε (edenPair k) = twoValued k (9 * a) (11 * a) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> (simp [hj, ha]; ring)
    exact integral_edenPair_vector_case (qP := 4) (qR := 6) (σ₀ := 1) (σ₁ := -1) hy'
      (by rw [h0]; simp) (by rw [h1]; simp) ha hs5 hs61
      (by rw [abs_of_pos (by positivity)]; linarith)
      (abs_of_pos (by positivity)).le (Or.inr (abs_of_pos (by positivity)))
      (by rw [show 9 * a * s / (11 * a) = 9 / 11 * s by field_simp]
          exact rtn_fp4_four (by linarith) (by linarith))
      (by rw [show 11 * a * s / (11 * a) = s by field_simp]; exact rtn_fp4_six hs5)
      (by ring) (by ring) (by norm_num) (by norm_num) i j
  · have hy' : rht k ε (edenPair k) = twoValued k (-(9 * a)) (-(11 * a)) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> (simp [hj, ha]; ring)
    exact integral_edenPair_vector_case (qP := -4) (qR := -6) (σ₀ := -1) (σ₁ := 1) hy'
      (by rw [h0]; simp) (by rw [h1]; simp) ha hs5 hs61
      (by rw [abs_neg, abs_of_pos (by positivity)]; linarith)
      (by rw [abs_neg, abs_of_pos (by positivity)])
      (Or.inr (by rw [abs_neg, abs_of_pos (by positivity)]))
      (by rw [show -(9 * a) * s / (11 * a) = -(9 / 11 * s) by field_simp]
          exact rtn_fp4_neg_four (by linarith) (by linarith))
      (by rw [show -(11 * a) * s / (11 * a) = -s by field_simp]; exact rtn_fp4_neg_six hs5)
      (by ring) (by ring) (by norm_num) (by norm_num) i j
  · have hy' : rht k ε (edenPair k) = twoValued k (-(11 * a)) (-(9 * a)) := by
      rw [hy, h0, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> (simp [hj, ha]; ring)
    exact integral_edenPair_vector_case (qP := -6) (qR := -4) (σ₀ := -1) (σ₁ := -1) hy'
      (by rw [h0]; simp) (by rw [h1]; simp) ha hs5 hs61
      (by rw [abs_neg, abs_of_pos (by positivity)])
      (by rw [abs_neg, abs_of_pos (by positivity)]; linarith)
      (Or.inl (by rw [abs_neg, abs_of_pos (by positivity)]))
      (by rw [show -(11 * a) * s / (11 * a) = -s by field_simp]; exact rtn_fp4_neg_six hs5)
      (by rw [show -(9 * a) * s / (11 * a) = -(9 / 11 * s) by field_simp]
          exact rtn_fp4_neg_four (by linarith) (by linarith))
      (by ring) (by ring) (by norm_num) (by norm_num) i j

/-- Averaging over the RHT seeds of §3.2 preserves the §3.3 full-vector conditional mean. -/
theorem mean_rhtInv_msEden_edenPair_vector
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    mean k (fun ε u => rhtInv k ε (msEden k s (edenPair k) ε u) i j) =
      edenMeanTarget k i j := by
  apply mean_eq_of_forall
  intro ε
  exact integral_rhtInv_msEden_edenPair_vector hs₀ hs₁ ε i j

/-- The vector mean is `(101/102) · (e₀ + e₁/5)` in the notation of §3.3. -/
theorem edenMeanTarget_eq_scaledPair (k : ℕ) :
    edenMeanTarget k = fun i j => (101 / 102 : ℝ) * twoSupport k 1 (1 / 5) i j := by
  funext i j
  unfold edenMeanTarget twoSupport
  split_ifs <;> norm_num

/-- The sign-case hypotheses are realized by the all-positive §3.3 seed at `d = 16`, `s = 6`. -/
example :
    ∫ u in edenCoinCube 0,
      rhtInv 0 (fun _ _ => false)
        (msEden 0 6 (edenPair 0) (fun _ _ => false) u) 0 0 =
      edenMeanTarget 0 0 0 := by
  have h4 : Real.sqrt (2 ^ (0 + 4) : ℝ) = 4 := by
    rw [show (2 : ℝ) ^ (0 + 4) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hy : rht 0 (fun _ _ => false) (edenPair 0) = twoValued 0 (11 / 40) (9 / 40) := by
    rw [rht_edenPair, h4]
    funext i j
    unfold twoValued
    norm_num
  have hqP : rtn fp4 ((11 / 40 : ℝ) * 6 / (11 * (1 / 40))) = 6 := by
    rw [show (11 / 40 : ℝ) * 6 / (11 * (1 / 40)) = 6 by norm_num]
    exact rtn_fp4_six (by norm_num)
  have hqR : rtn fp4 ((9 / 40 : ℝ) * 6 / (11 * (1 / 40))) = 4 := by
    rw [show (9 / 40 : ℝ) * 6 / (11 * (1 / 40)) = 54 / 11 by norm_num]
    exact rtn_fp4_four (by norm_num) (by norm_num)
  exact integral_edenPair_vector_case (qP := 6) (qR := 4) (σ₀ := 1) (σ₁ := 1)
    (a := 1 / 40) hy (by norm_num) (by norm_num) (by rw [h4]; norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Or.inl (by norm_num))
    hqP hqR (by norm_num) (by norm_num) (by norm_num) (by norm_num) 0 0

/-- The §3.3 full-vector statement applies to the paper's 128-entry RHT and chosen scale. -/
example (ε : Fin (2 ^ 3) → Fin 16 → Bool) (i : Fin (2 ^ 3)) (j : Fin 16) :
    ∫ u in edenCoinCube 3,
      rhtInv 3 ε (msEden 3 (3200 / 527) (edenPair 3) ε u) i j =
        edenMeanTarget 3 i j :=
  integral_rhtInv_msEden_edenPair_vector (by norm_num) (by norm_num) ε i j

end Transformer.Quartet
