/-
# The pair sums of the third cumulant, in total

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
off-diagonal part of the third cumulant of the tokens `k < j < i` consists of four
double sums `Σ_l Σ_m Ĝ^l_y Ĝ^m_z (E(x; l, m) + E(x; m, l))`, each the sum of two
`pairSum`s (`sum_sum_pairEnv`). All eight are at most `Λ q_γ(i, k)⁶/((i+1)(j+1))`
(`sum_pairEnv_four_le`).
-/

import Transformer.Kinetic.Section3_PairSumsLower

namespace Transformer
namespace Kinetic

/-- The symmetrized pair envelope sums to two pair sums. -/
theorem sum_sum_pairEnv (γ τ : ℝ) (N x y z : ℕ) :
    ∑ l ∈ Finset.range N, ∑ m ∈ Finset.range N, profileEnv γ τ y l * profileEnv γ τ z m *
      (pairEnv γ x l m + pairEnv γ x m l) = pairSum γ τ N x y z + pairSum γ τ N x z y := by
  have h : pairSum γ τ N x z y = ∑ l ∈ Finset.range N, ∑ m ∈ Finset.range N,
      profileEnv γ τ y l * profileEnv γ τ z m * pairEnv γ x m l := by
    unfold pairSum
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun l _ => Finset.sum_congr rfl fun m _ => by ring
  rw [h]
  unfold pairSum
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun m _ => by ring

/-- **The off-diagonal sums of the third cumulant.** For `k < j < i`, the four double
sums are at most `8Λ q_γ(i, k)⁶/((i+1)(j+1))`. -/
theorem sum_pairEnv_four_le {γ τ : ℝ} (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 12) (hτ : 0 ≤ τ) (N : ℕ)
    {i j k : ℕ} (hkj : k < j) (hji : j < i) :
    ∑ l ∈ Finset.range N, ∑ m ∈ Finset.range N,
      (profileEnv γ τ k l * profileEnv γ τ i m * (pairEnv γ j l m + pairEnv γ j m l) +
        profileEnv γ τ j l * profileEnv γ τ i m * (pairEnv γ k l m + pairEnv γ k m l) +
        profileEnv γ τ k l * profileEnv γ τ j m * (pairEnv γ i l m + pairEnv γ i m l) +
        profileEnv γ τ i l * profileEnv γ τ j m * (pairEnv γ k l m + pairEnv γ k m l)) ≤
      8 * pairConst γ τ * (ratioPow γ i k ^ 6 / (((i : ℝ) + 1) * ((j : ℝ) + 1))) := by
  simp only [Finset.sum_add_distrib, sum_sum_pairEnv]
  have hki := hkj.trans hji
  have hQ1 : 1 ≤ ratioPow γ i k := one_le_ratioPow hγ0.le hki.le
  have hQij : ratioPow γ i j ≤ ratioPow γ i k := ratioPow_le_ratioPow_right hγ0.le i hkj.le
  have hQjk : ratioPow γ j k ≤ ratioPow γ i k := ratioPow_le_ratioPow_left hγ0.le hji.le k
  have hQik : ratioPow γ i k ≤ ratioPow γ i k := le_rfl
  have h1 := pairSum_up_le hγ0 hγ hτ N hji hQ1 hQij hQjk
  have h2 := pairSum_low_le hγ0 hγ hτ N hkj hki hQ1 hQjk hQik
  have h3 := pairSum_first_le hγ0 hγ hτ N hkj hki hQ1 hQjk hQik
  have h4 := pairSum_first_le hγ0 hγ hτ N hki hkj hQ1 hQik hQjk
  have h5 := pairSum_mid_le hγ0 hγ hτ N hji hQ1 hQik hQjk
  have h6 := pairSum_low_le hγ0 hγ hτ N hki hkj hQ1 hQik hQjk
  rw [mul_comm ((j : ℝ) + 1)] at h2 h3
  linarith

/-- The hypotheses of `sum_pairEnv_four_le` are satisfiable: `γ = 1/16`, `τ = 1`,
`k = 0 < j = 1 < i = 2`. -/
example : (0 : ℝ) < 1 / 16 ∧ (1 / 16 : ℝ) ≤ 1 / 12 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 ∧
    (1 : ℕ) < 2 := by norm_num

end Kinetic
end Transformer
