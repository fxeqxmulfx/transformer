/-
# The mean of `MS-EDEN` over the coins, on a two-valued rotation

Support for the counterexample of `Transformer.Quartet.Section3_EdenBias` to the Corollary of
§3.3.  On a two-valued rotation (`Section3_EdenTwoValued`) the first entry of the inverse
rotation of `MS-EDEN` is a constant times the sum over the `2^k` groups of the stochastic
rounding of the corrected scale `256 S` (`rhtInv_msEden_twoValued`), and that sum integrates to
`2^k · 256 S` by the unbiasedness of `SR` (`integral_sum_sr`).  What is left is arithmetic in
`P`, `R`, `q_P`, `q_R` (`integral_rhtInv_msEden_twoValued`), and for the rotation of
`e₀ + e₁/10` (`Section3_EdenPair`) it collapses to `101/102`, provided the roundings are those
of `integral_rhtInv_msEden_edenPair_case`.

Source: arXiv:2601.22813v2, §3.1 (`SR` is unbiased), §3.3 and Algorithm 1.
-/

import Transformer.Quartet.Section3_EdenTwoValued
import Transformer.Quartet.Section3_EdenCoins

open MeasureTheory

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- **The mean over the coins of `MS-EDEN` on a two-valued rotation**, at the first
coordinate: with `q_P = RTN_FP4(P s/M)`, `q_R = RTN_FP4(R s/M)`, and the corrected group scale
inside the E4M3 range, the `2^k` stochastic roundings of the scale are unbiased
(`integral_sum_sr`), and what is left is the arithmetic of the two values. -/
theorem integral_rhtInv_msEden_twoValued {x : Fin (2 ^ k) → Fin 16 → ℝ}
    {ε : Fin (2 ^ k) → Fin 16 → Bool} {P R M s qP qR : ℝ} (hy : rht k ε x = twoValued k P R)
    (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M) (hR : |R| ≤ M) (h : |P| = M ∨ |R| = M)
    (hqP : rtn fp4 (P * s / M) = qP) (hqR : rtn fp4 (R * s / M) = qR)
    (hD : 0 < P * qP + R * qR)
    (hz : s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256 ≤ 448) :
    ∫ u in Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1),
        rhtInv k ε (msEden k s x ε u) 0 0 =
      (if ε 0 0 then -1 else 1) * (1 / Real.sqrt (2 ^ (k + 4)) * (8 * (qP + qR)) * 2 ^ k *
        ((P ^ 2 + R ^ 2) / (P * qP + R * qR))) := by
  have hpt := rhtInv_msEden_twoValued hy hs hM hP hR h hqP hqR hD.ne'
  have hfun : (fun u : Fin (2 ^ k) → ℝ => rhtInv k ε (msEden k s x ε u) 0 0) = fun u =>
      ((if ε 0 0 then -1 else 1) *
        (1 / Real.sqrt (2 ^ (k + 4)) * (8 * (qP + qR)) * (M / (s * 256)))) *
        ∑ i' : Fin (2 ^ k),
          sr fp8 (s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256) (u i') := funext hpt
  have hz0 : 0 ≤ s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256 := by positivity
  rw [hfun, integral_const_mul, integral_sum_sr hz0 hz]
  field_simp

/-- `(1/√(2^{k+4}))² · 2^k = 1/16`: the normalization of the Hadamard matrix against the number
of groups. -/
theorem sq_inv_sqrt_mul_pow (k : ℕ) : (1 / Real.sqrt (2 ^ (k + 4))) ^ 2 * 2 ^ k = 1 / 16 := by
  rw [div_pow, Real.sq_sqrt (by positivity), one_pow, pow_add]
  field_simp
  norm_num

/-- **One sign pattern of the counterexample.**  If the rotation of `edenPair` is two-valued with
`P, R` of absolute value at most `11a` and `a = 1/(10 √d)`, the two roundings `q_P`, `q_R` are as
below, and the values satisfy `P² + R² = 202 a²`, `P q_P + R q_R = 102 a` and `± (q_P + q_R) = 10`
for the sign `±` of the seed, then the mean over the coins at `(0, 0)` is `101/102`. -/
theorem integral_rhtInv_msEden_edenPair_case {ε : Fin (2 ^ k) → Fin 16 → Bool}
    {s P R qP qR σ a : ℝ} (hy : rht k ε (edenPair k) = twoValued k P R)
    (hσ : (if ε 0 0 then (-1 : ℝ) else 1) = σ) (ha : a = 1 / Real.sqrt (2 ^ (k + 4)) / 10)
    (hs : 5 < s) (hs' : s ≤ 61 / 10) (hP : |P| ≤ 11 * a) (hR : |R| ≤ 11 * a)
    (h : |P| = 11 * a ∨ |R| = 11 * a) (hqP : rtn fp4 (P * s / (11 * a)) = qP)
    (hqR : rtn fp4 (R * s / (11 * a)) = qR) (hPR : P ^ 2 + R ^ 2 = 202 * a ^ 2)
    (hD : P * qP + R * qR = 102 * a) (hq : σ * (qP + qR) = 10) :
    ∫ u in Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1),
      rhtInv k ε (msEden k s (edenPair k) ε u) 0 0 = 101 / 102 := by
  have ha0 : 0 < a := by rw [ha]; positivity
  have hM : 0 < 11 * a := by positivity
  have hs0 : 0 < s := by linarith
  have hD0 : 0 < P * qP + R * qR := by rw [hD]; positivity
  have hz : s * (P ^ 2 + R ^ 2) / (11 * a * (P * qP + R * qR)) * 256 ≤ 448 := by
    rw [hPR, hD, show s * (202 * a ^ 2) / (11 * a * (102 * a)) * 256 =
      s * (202 * 256 / 1122) by field_simp; norm_num]
    linarith
  rw [integral_rhtInv_msEden_twoValued hy hs0 hM hP hR h hqP hqR hD0 hz, hσ, hPR, hD,
    show 202 * a ^ 2 / (102 * a) = 101 / 51 * a by field_simp; ring]
  set c : ℝ := 1 / Real.sqrt (2 ^ (k + 4)) with hc
  calc σ * (c * (8 * (qP + qR)) * 2 ^ k * (101 / 51 * a))
      = (σ * (qP + qR)) * (c * a) * 2 ^ k * (8 * (101 / 51)) := by ring
    _ = 10 * (c * (c / 10)) * 2 ^ k * (8 * (101 / 51)) := by rw [hq, ha]
    _ = (c ^ 2 * 2 ^ k) * (8 * (101 / 51)) := by ring
    _ = 101 / 102 := by rw [hc, sq_inv_sqrt_mul_pow]; norm_num

/-- The hypotheses of the sign pattern are satisfiable: `k = 0`, the all-`false` seed,
`P = 11a`, `R = 9a`, `q_P = 6`, `q_R = 4`, `s = 6`. -/
example : ∃ (ε : Fin (2 ^ 0) → Fin 16 → Bool) (s P R qP qR σ a : ℝ),
    rht 0 ε (edenPair 0) = twoValued 0 P R ∧ (if ε 0 0 then (-1 : ℝ) else 1) = σ ∧
    a = 1 / Real.sqrt (2 ^ (0 + 4)) / 10 ∧ 5 < s ∧ s ≤ 61 / 10 ∧ |P| ≤ 11 * a ∧ |R| ≤ 11 * a ∧
    (|P| = 11 * a ∨ |R| = 11 * a) ∧ rtn fp4 (P * s / (11 * a)) = qP ∧
    rtn fp4 (R * s / (11 * a)) = qR ∧ P ^ 2 + R ^ 2 = 202 * a ^ 2 ∧ P * qP + R * qR = 102 * a ∧
    σ * (qP + qR) = 10 := by
  have h4 : √((2 : ℝ) ^ (0 + 4)) = 4 := by
    rw [show ((2 : ℝ) ^ (0 + 4)) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  refine ⟨fun _ _ => false, 6, 11 / 40, 9 / 40, 6, 4, 1, 1 / 40, ?_, by norm_num, ?_, by norm_num,
    by norm_num, by norm_num [abs_of_pos], by norm_num [abs_of_pos],
    Or.inl (by norm_num [abs_of_pos]), ?_, ?_, by norm_num, by norm_num, by norm_num⟩
  · rw [rht_edenPair, h4]
    funext i j
    unfold twoValued
    norm_num
  · rw [h4]; norm_num
  · rw [show (11 / 40 : ℝ) * 6 / (11 * (1 / 40)) = 6 by norm_num]
    exact rtn_fp4_six (by norm_num)
  · rw [show (9 / 40 : ℝ) * 6 / (11 * (1 / 40)) = 54 / 11 by norm_num]
    exact rtn_fp4_four (by norm_num) (by norm_num)

end Quartet
end Transformer
