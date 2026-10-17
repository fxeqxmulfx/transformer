/-
# Sums over a single token

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Steps 2 and 4: the sums over a
single token `l` of the covariance, `Σ_l Ĝ^l_i Ĝ^l_j ≲ i⁻¹ e^{√(CT log(i/j))}`
(`eq:est-cov-phi`, `sum_profileEnv_mul_two_le`), and of the diagonal part of the third
cumulant, `Σ_l Ĝ^l_i Ĝ^l_j Ĝ^l_k ≲ (ij)⁻¹ e^{√(CT log(i/k))}`
(`sum_profileEnv_mul_three_le`). Below the earliest token every profile factors
through it (`profileEnv_mul_chainEnv`).
-/

import Transformer.Kinetic.Section3_PairEnvelope

namespace Transformer
namespace Kinetic

/-- For `c < x`, `Ĝ^l_x F(c, l) = τ q_γ(x, c)/(x+1) · q_γ(c, l) F(c, l)`. -/
theorem profileEnv_mul_chainEnv {γ τ : ℝ} {x c : ℕ} (hcx : c < x) (l : ℕ) :
    profileEnv γ τ x l * chainEnv γ c l =
      τ * ratioPow γ x c / ((x : ℝ) + 1) * (ratioPow γ c l * chainEnv γ c l) := by
  rcases lt_or_ge l c with hl | hl
  · rw [profileEnv_of_lt (hl.trans hcx), ← ratioPow_mul γ x c l]
    ring
  · rw [chainEnv, ite_eq_right (by omega)]
    ring

theorem profileEnv_eq (γ τ : ℝ) (x l : ℕ) :
    profileEnv γ τ x l = (if l = x then 1 else 0) + τ * chainEnv γ x l := rfl

/-- **The covariance sum.** For `j < i`, `Σ_l Ĝ^l_i Ĝ^l_j ≤ τ(1 + 2τ) q_γ(i, j)/(i+1)`. -/
theorem sum_profileEnv_mul_two_le {γ τ : ℝ} (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 4) (hτ : 0 ≤ τ)
    (N : ℕ) {i j : ℕ} (hji : j < i) :
    ∑ l ∈ Finset.range N, profileEnv γ τ i l * profileEnv γ τ j l ≤
      τ * (1 + 2 * τ) * (ratioPow γ i j / ((i : ℝ) + 1)) := by
  have hsplit : ∑ l ∈ Finset.range N, profileEnv γ τ i l * profileEnv γ τ j l =
      (∑ l ∈ Finset.range N, if l = j then profileEnv γ τ i l else 0) +
        τ * (τ * ratioPow γ i j / ((i : ℝ) + 1) *
          ∑ l ∈ Finset.range N, ratioPow γ j l ^ 1 * chainEnv γ j l) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [pow_one, ← profileEnv_mul_chainEnv hji, profileEnv_eq γ τ j l]
    split_ifs <;> ring
  rw [hsplit, Finset.sum_ite_eq']
  have h1 : (if j ∈ Finset.range N then profileEnv γ τ i j else 0) ≤
      τ * (ratioPow γ i j / ((i : ℝ) + 1)) := by
    rw [← profileEnv_of_lt hji]
    split_ifs
    · exact le_rfl
    · exact profileEnv_nonneg hτ i j
  have h2 := sum_chainEnv_mul_le hγ0.le (r := 1) (by push_cast; linarith) N j
  have := ratioPow_pos γ i j
  have h3 := mul_le_mul_of_nonneg_left h2
    (by positivity : 0 ≤ τ * ratioPow γ i j / ((i : ℝ) + 1))
  have h4 := mul_le_mul_of_nonneg_left h3 hτ
  have h5 : τ * (τ * ratioPow γ i j / ((i : ℝ) + 1) * 2) =
      2 * τ * (τ * (ratioPow γ i j / ((i : ℝ) + 1))) := by ring
  nlinarith

/-- **The diagonal sum of the third cumulant.** For `k < j < i`,
`Σ_l Ĝ^l_i Ĝ^l_j Ĝ^l_k ≤ τ²(1 + 2τ) q_γ(i, k) q_γ(j, k)/((i+1)(j+1))`. -/
theorem sum_profileEnv_mul_three_le {γ τ : ℝ} (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 6) (hτ : 0 ≤ τ)
    (N : ℕ) {i j k : ℕ} (hkj : k < j) (hji : j < i) :
    ∑ l ∈ Finset.range N, profileEnv γ τ i l * profileEnv γ τ j l * profileEnv γ τ k l ≤
      τ ^ 2 * (1 + 2 * τ) *
        (ratioPow γ i k * ratioPow γ j k / (((i : ℝ) + 1) * ((j : ℝ) + 1))) := by
  obtain ⟨c, hc⟩ : ∃ c : ℝ,
      c = ratioPow γ i k * ratioPow γ j k / (((i : ℝ) + 1) * ((j : ℝ) + 1)) := ⟨_, rfl⟩
  have hc0 : 0 ≤ c := by
    have := ratioPow_pos γ i k
    have := ratioPow_pos γ j k
    rw [hc]; positivity
  have hsplit : ∑ l ∈ Finset.range N,
      profileEnv γ τ i l * profileEnv γ τ j l * profileEnv γ τ k l =
      (∑ l ∈ Finset.range N, if l = k then profileEnv γ τ i l * profileEnv γ τ j l else 0) +
        τ * (τ ^ 2 * c * ∑ l ∈ Finset.range N, ratioPow γ k l ^ 2 * chainEnv γ k l) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun l _ => ?_
    have e : profileEnv γ τ i l * profileEnv γ τ j l * chainEnv γ k l =
        τ ^ 2 * c * (ratioPow γ k l ^ 2 * chainEnv γ k l) := by
      calc profileEnv γ τ i l * profileEnv γ τ j l * chainEnv γ k l
          = profileEnv γ τ i l * (profileEnv γ τ j l * chainEnv γ k l) := by ring
        _ = τ * ratioPow γ j k / ((j : ℝ) + 1) * ratioPow γ k l *
              (profileEnv γ τ i l * chainEnv γ k l) := by
            rw [profileEnv_mul_chainEnv hkj]; ring
        _ = τ ^ 2 * c * (ratioPow γ k l ^ 2 * chainEnv γ k l) := by
            rw [profileEnv_mul_chainEnv (hkj.trans hji), hc]
            simp only [div_eq_mul_inv, mul_inv]
            ring
    rw [← e, profileEnv_eq γ τ k l]
    split_ifs <;> ring
  rw [hsplit, Finset.sum_ite_eq']
  have h1 : (if k ∈ Finset.range N then profileEnv γ τ i k * profileEnv γ τ j k else 0) ≤
      τ ^ 2 * c := by
    have e : profileEnv γ τ i k * profileEnv γ τ j k = τ ^ 2 * c := by
      rw [profileEnv_of_lt (hkj.trans hji), profileEnv_of_lt hkj, hc]
      simp only [div_eq_mul_inv, mul_inv]
      ring
    rw [e]
    split_ifs
    · exact le_rfl
    · positivity
  have h2 := sum_chainEnv_mul_le hγ0.le (r := 2) (by push_cast; linarith) N k
  have h3 := mul_le_mul_of_nonneg_left h2 (by positivity : 0 ≤ τ ^ 2 * c)
  have h4 := mul_le_mul_of_nonneg_left h3 hτ
  have h5 : τ * (τ ^ 2 * c * 2) = 2 * τ * (τ ^ 2 * c) := by ring
  rw [← hc]
  nlinarith

/-- The hypotheses of `profileEnv_mul_chainEnv`, `sum_profileEnv_mul_two_le` and
`sum_profileEnv_mul_three_le` hold at `γ = 1/16` (`γ = 1` for the first), `τ = 1` and the
tokens `0 < 1 < 2`. -/
example := And.intro (profileEnv_mul_chainEnv (γ := 1) (τ := 1) (x := 1) (c := 0) Nat.one_pos
    0) <|
  And.intro (sum_profileEnv_mul_two_le (γ := 1 / 16) (τ := 1) (by norm_num) (by norm_num)
    zero_le_one 1 (i := 1) (j := 0) Nat.one_pos)
  (sum_profileEnv_mul_three_le (γ := 1 / 16) (τ := 1) (by norm_num) (by norm_num) zero_le_one
    1 (i := 2) (j := 1) (k := 0) Nat.one_pos (by norm_num))

end Kinetic
end Transformer
