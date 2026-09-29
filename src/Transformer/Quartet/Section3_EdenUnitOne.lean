/-
# MS-EDEN mean on the unit-coordinate operand

Source: arXiv:2601.22813v2, §3.2–§3.3 and Algorithm 1.
-/

import Transformer.Quartet.Section3_EdenVectorBias

open MeasureTheory

namespace Transformer.Quartet

variable {k : ℕ}

/-- The second §3.3 GEMM operand: the unit vector at coordinate one. -/
def edenUnitOne (k : ℕ) : Fin (2 ^ k) → Fin 16 → ℝ := twoSupport k 0 1

/-- For §3.3 scales, Algorithm 1 reproduces `e₁` in conditional expectation for every sign seed. -/
theorem integral_rhtInv_msEden_edenUnitOne
    {s : ℝ} (hs₀ : 6 * (16 / 17) ≤ s) (hs₁ : s ≤ 6 * (16 / 17) / 0.93)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in edenCoinCube k, rhtInv k ε (msEden k s (edenUnitOne k) ε u) i j =
      edenUnitOne k i j := by
  norm_num at hs₀ hs₁
  have hs5 : 5 < s := by linarith
  have hs61 : s ≤ 61 / 10 := by linarith
  obtain ⟨a, ha⟩ : ∃ a : ℝ, a = 1 / Real.sqrt (2 ^ (k + 4)) := ⟨_, rfl⟩
  have ha0 : 0 < a := by rw [ha]; positivity
  have hy := rht_twoSupport (k := k) 0 1 ε
  rcases h1 : ε 0 1 with _ | _
  · have hy' : rht k ε (edenUnitOne k) = twoValued k a (-a) := by
      rw [edenUnitOne, hy, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> simp [hj, ha, div_eq_mul_inv]
    have hqP : rtn fp4 (a * s / a) = 6 := by
      rw [show a * s / a = s by field_simp]; exact rtn_fp4_six hs5
    have hqR : rtn fp4 (-a * s / a) = -6 := by
      rw [show -a * s / a = -s by field_simp]; exact rtn_fp4_neg_six hs5
    have hz : s * (a ^ 2 + (-a) ^ 2) / (a * (a * 6 + (-a) * -6)) * 256 ≤ 448 := by
      rw [show s * (a ^ 2 + (-a) ^ 2) / (a * (a * 6 + (-a) * -6)) * 256 =
        s * (256 / 6) by field_simp]
      linarith
    rw [integral_rhtInv_msEden_twoValued_vector hy' (by linarith) ha0
      (by rw [abs_of_pos ha0]) (by rw [abs_neg, abs_of_pos ha0])
      (Or.inl (abs_of_pos ha0)) hqP hqR (by nlinarith [sq_pos_of_pos ha0]) hz]
    have hrat : (a ^ 2 + (-a) ^ 2) / (a * 6 + (-a) * -6) = a / 6 := by
      field_simp
    rw [hrat, show (6 : ℝ) * (a / 6) = a by ring,
      show (-6 : ℝ) * (a / 6) = -a by ring, ←hy', rhtInv_rht]
  · have hy' : rht k ε (edenUnitOne k) = twoValued k (-a) a := by
      rw [edenUnitOne, hy, h1]; funext i j; unfold twoValued
      by_cases hj : (j : ℕ) % 2 = 0 <;> simp [hj, ha, div_eq_mul_inv]
    have hqP : rtn fp4 (-a * s / a) = -6 := by
      rw [show -a * s / a = -s by field_simp]; exact rtn_fp4_neg_six hs5
    have hqR : rtn fp4 (a * s / a) = 6 := by
      rw [show a * s / a = s by field_simp]; exact rtn_fp4_six hs5
    have hz : s * ((-a) ^ 2 + a ^ 2) / (a * ((-a) * -6 + a * 6)) * 256 ≤ 448 := by
      rw [show s * ((-a) ^ 2 + a ^ 2) / (a * ((-a) * -6 + a * 6)) * 256 =
        s * (256 / 6) by field_simp]
      linarith
    rw [integral_rhtInv_msEden_twoValued_vector hy' (by linarith) ha0
      (by rw [abs_neg, abs_of_pos ha0]) (by rw [abs_of_pos ha0])
      (Or.inr (abs_of_pos ha0)) hqP hqR (by nlinarith [sq_pos_of_pos ha0]) hz]
    have hrat : ((-a) ^ 2 + a ^ 2) / ((-a) * -6 + a * 6) = a / 6 := by
      field_simp
    rw [hrat, show (-6 : ℝ) * (a / 6) = -a by ring,
      show (6 : ℝ) * (a / 6) = a by ring, ←hy', rhtInv_rht]

/-- The §3.3 unit-coordinate result applies at `d = 128` and the chosen scale. -/
example (ε : Fin (2 ^ 3) → Fin 16 → Bool) (i : Fin (2 ^ 3)) (j : Fin 16) :
    ∫ u in edenCoinCube 3,
      rhtInv 3 ε (msEden 3 (3200 / 527) (edenUnitOne 3) ε u) i j =
        edenUnitOne 3 i j :=
  integral_rhtInv_msEden_edenUnitOne (by norm_num) (by norm_num) ε i j

end Transformer.Quartet
