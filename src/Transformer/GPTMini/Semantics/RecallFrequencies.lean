import Transformer.GPTMini.Semantics.RecallCodes
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

/-!
# Actual low RoPE frequencies for compact recall matching

Source: original head_dim=16/theta=10000 rope_tables at f11b6e2.
The compact four-digit symbol code will occupy pairs four through seven.
Their unequal frequencies must all be included: replacing them by zero
or by an assumed position-independent dot product would change GPTMini.
The bounds below are derived from the exact original real power law.

Over sixty-four raw positions their total squared angular displacement
is small enough to preserve a categorical symbol gap. These scalar
bounds are independent of tables and do not assume a correct query route.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- Pair four through seven of the actual small-model RoPE.
Source: the unchanged power-law table, with the selected pair index explicitly in range. -/
noncomputable def recallFrequency (pair : Fin 4) : ℝ :=
  invFreq 16 10000 ⟨pair.val + 4, by have hp := pair.isLt; omega⟩

/-- Every selected frequency is the corresponding inverse square-root-of-ten power.
Source: the exact theta=10^4 exponent, not a numerical approximation to the rotation table. -/
theorem recallFrequency_power (pair : Fin 4) :
    recallFrequency pair = (1 / Real.sqrt 10) ^ (pair.val + 4) := by
  unfold recallFrequency invFreq
  norm_num only [Nat.cast_ofNat, Nat.cast_mul, Nat.cast_add]
  rw [show (10000 : ℝ) = (10 : ℝ) ^ 4 from by norm_num,
    ← Real.rpow_natCast_mul (by norm_num : (0 : ℝ) ≤ 10) 4]
  norm_num only [Nat.cast_ofNat]
  have he : (4 : ℝ) * (-(2 * ((pair.val : ℝ) + 4)) / 16) =
      -(1 / 2) * (pair.val + 4 : ℕ) := by
    push_cast
    ring
  rw [he]
  rw [Real.rpow_mul_natCast (by norm_num) _ (pair.val + 4)]
  rw [Real.rpow_neg (by norm_num), ← Real.sqrt_eq_rpow]
  simp only [one_div]

/-- Every actual selected pair rotates by a positive frequency.
Source: the original positive theta and the exact exponent; positivity is needed for latest-write ordering. -/
theorem recallFrequency_pos (pair : Fin 4) : 0 < recallFrequency pair := by
  unfold recallFrequency invFreq
  exact Real.rpow_pos_of_pos (by norm_num) _

/-- The exact fourth, fifth and sixth square-root powers used by the real table.
Source: sqrt(10)^2=10; these equalities evaluate the actual power-law constants. -/
theorem recall_sqrt_powers :
    Real.sqrt 10 ^ 4 = 100 ∧ Real.sqrt 10 ^ 5 = 100 * Real.sqrt 10 ∧
      Real.sqrt 10 ^ 6 = 1000 := by
  have hs : Real.sqrt 10 ^ 2 = 10 := Real.sq_sqrt (by norm_num)
  have h4 : Real.sqrt 10 ^ 4 = 100 := by
    calc _ = (Real.sqrt 10 ^ 2) ^ 2 := by ring
         _ = 100 := by rw [hs]; norm_num
  have h5 : Real.sqrt 10 ^ 5 = 100 * Real.sqrt 10 := by
    rw [show (5 : ℕ) = 4 + 1 from rfl, pow_add, h4]
    simp
  have h6 : Real.sqrt 10 ^ 6 = 1000 := by
    calc _ = (Real.sqrt 10 ^ 2) ^ 3 := by ring
         _ = 1000 := by rw [hs]; norm_num
  exact ⟨h4, h5, h6⟩

/-- Exact leading frequencies and conservative upper bounds for all four matching pairs.
Source: the actual power-law formula, retaining each pair's distinct frequency. -/
theorem recallFrequency_bounds :
    recallFrequency 0 = 1 / 100 ∧ recallFrequency 1 ≤ 1 / 200 ∧
      recallFrequency 2 = 1 / 1000 ∧ recallFrequency 3 ≤ 1 / 1000 := by
  have hs : (2 : ℝ) ≤ Real.sqrt 10 := by
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 10), Real.sqrt_nonneg 10]
  have hform (p : Fin 4) : recallFrequency p = 1 / Real.sqrt 10 ^ (p.val + 4) := by
    rw [recallFrequency_power, div_pow]
    simp
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hform]
    change 1 / Real.sqrt 10 ^ 4 = 1 / 100
    rw [recall_sqrt_powers.1]
  · rw [hform]
    change 1 / Real.sqrt 10 ^ 5 ≤ 1 / 200
    rw [recall_sqrt_powers.2.1, div_le_iff₀ (by positivity)]
    nlinarith
  · rw [hform]
    change 1 / Real.sqrt 10 ^ 6 = 1 / 1000
    rw [recall_sqrt_powers.2.2]
  · rw [hform]
    change 1 / Real.sqrt 10 ^ 7 ≤ 1 / 1000
    rw [show (7 : ℕ) = 6 + 1 from rfl, pow_add, recall_sqrt_powers.2.2]
    simp only [pow_one]
    rw [div_le_iff₀ (by positivity)]
    nlinarith

/-- No matching pair rotates more than 1/100 per raw position.
Source: the exact four-pair bounds, useful for mismatched categorical axes. -/
theorem recallFrequency_le (pair : Fin 4) : recallFrequency pair ≤ 1 / 100 := by
  have h := recallFrequency_bounds
  fin_cases pair
  · exact le_of_eq h.1
  · exact h.2.1.trans (by norm_num)
  · exact (le_of_eq h.2.2.1).trans (by norm_num)
  · exact h.2.2.2.trans (by norm_num)

/-- The summed squared frequencies are small enough to protect all four simultaneous matching axes.
Source: each actual pair's upper bound, rather than multiplying the fastest frequency by four. -/
theorem recallFrequency_sum_sq : (∑ p : Fin 4, recallFrequency p ^ 2) ≤ 127 / 1000000 := by
  have h := recallFrequency_bounds
  have h1 := (sq_le_sq₀ (recallFrequency_pos 1).le (by norm_num : (0 : ℝ) ≤ 1 / 200)).mpr h.2.1
  have h3 := (sq_le_sq₀ (recallFrequency_pos 3).le (by norm_num : (0 : ℝ) ≤ 1 / 1000)).mpr h.2.2.2
  rw [Fin.sum_univ_four, h.1, h.2.2.1]
  nlinarith

/-- Every actual matching-pair angle stays below 16/25 over a complete recall context.
Source: the unchanged raw context cap 64 and the fastest of the four actual frequencies. -/
theorem recall_angle_bound (pair : Fin 4) (d : ℝ) (hd : |d| ≤ 64) :
    |d * recallFrequency pair| ≤ 16 / 25 := by
  rw [abs_mul, abs_of_pos (recallFrequency_pos pair)]
  have h := mul_le_mul_of_nonneg hd (recallFrequency_le pair) (abs_nonneg d)
    (by norm_num : (0 : ℝ) ≤ 1 / 100)
  norm_num at h
  exact h

example : |(-63 : ℝ)| ≤ 64 := by norm_num

/-- The actual four-pair cosine score of identical symbols remains at least 93/100 at every recall displacement.
Source: the full unequal-frequency sum and the real cosine bound; no zero-RoPE approximation is made. -/
theorem recall_cosine_match_lower (d : ℝ) (hd : |d| ≤ 64) :
    93 / 100 ≤ (∑ p : Fin 4, Real.cos (d * recallFrequency p)) / 4 := by
  have hd2 : d ^ 2 ≤ (4096 : ℝ) := by
    have h := sq_le_sq' (abs_le.mp hd).1 (abs_le.mp hd).2
    norm_num at h
    exact h
  have hproduct := mul_le_mul_of_nonneg hd2 recallFrequency_sum_sq (sq_nonneg d)
    (by norm_num : (0 : ℝ) ≤ 127 / 1000000)
  have hcos : (∑ p : Fin 4, (1 - (d * recallFrequency p) ^ 2 / 2)) ≤
      ∑ p : Fin 4, Real.cos (d * recallFrequency p) := by
    apply Finset.sum_le_sum
    intro p hp
    exact Real.one_sub_sq_div_two_le_cos
  have hid : (∑ p : Fin 4, (1 - (d * recallFrequency p) ^ 2 / 2)) =
      4 - d ^ 2 / 2 * (∑ p : Fin 4, recallFrequency p ^ 2) := by
    rw [Fin.sum_univ_four, Fin.sum_univ_four]
    ring
  rw [hid] at hcos
  nlinarith

example : |(63 : ℝ)| ≤ 64 := by norm_num

end Transformer.GPTMini.Semantics
