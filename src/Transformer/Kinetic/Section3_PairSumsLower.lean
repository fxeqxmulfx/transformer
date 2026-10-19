module

public import Transformer.Kinetic.Section3_PairSums

/-!
# The sums of a pair below the earliest observable

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
sums `S(x; y, z)` (`pairSum`) whose pair `n < M` lies below the earliest observable
`k`, because the later token is weighted by `k` (`pairSum_low_le`) or the pair
decays at `k` (`pairSum_first_le`). The earlier token `n < k` is then weighted by
`τ q_γ(y, k) F(k, n)`, and `Σ_{n<k} q_γ(k, n)^r F(k, n) ≤ 2` absorbs the sum.
-/

@[expose] public section

namespace Transformer
namespace Kinetic

/-- **`S(x; y, k)`** for `k < x, y`: the later token weighted by the earliest observable. -/
theorem pairSum_low_le {γ τ Q : ℝ} (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 12) (hτ : 0 ≤ τ) (N : ℕ)
    {x y k : ℕ} (hkx : k < x) (hky : k < y) (hQ1 : 1 ≤ Q) (hQx : ratioPow γ x k ≤ Q)
    (hQy : ratioPow γ y k ≤ Q) :
    pairSum γ τ N x y k ≤ pairConst γ τ * (Q ^ 6 / (((x : ℝ) + 1) * ((y : ℝ) + 1))) := by
  have hγ1 : γ ≤ 1 := by linarith
  have hqx := ratioPow_pos γ x k
  have hqy := ratioPow_pos γ y k
  obtain ⟨u, hu⟩ : ∃ u : ℝ, u = 1 / (((x : ℝ) + 1) * ((y : ℝ) + 1)) := ⟨_, rfl⟩
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  refine (pairSum_le hτ (w := fun n => if n < k then
    (1 + τ * ratioPow γ k n * ratioPow γ k n / γ) / ((k : ℝ) + 1) * (1 / ((x : ℝ) + 1)) else 0)
    fun n => sum_profileEnv_pairDecay_down hγ0 hγ1 hτ hkx N n).trans ?_
  have heq : ∑ n ∈ Finset.range N, profileEnv γ τ y n * ratioPow γ x n ^ 3 *
      (if n < k then (1 + τ * ratioPow γ k n * ratioPow γ k n / γ) / ((k : ℝ) + 1) *
        (1 / ((x : ℝ) + 1)) else 0) =
      τ * u * (ratioPow γ y k * ratioPow γ x k ^ 3) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * chainEnv γ k n +
        τ * u * (τ / γ) * (ratioPow γ y k * ratioPow γ x k ^ 3) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * chainEnv γ k n := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun n _ => ?_
    unfold chainEnv
    split_ifs with hn
    · rw [profileEnv_of_lt (hn.trans hky), ← ratioPow_mul γ y k n, ← ratioPow_mul γ x k n, hu]
      simp only [div_eq_mul_inv, mul_inv]
      ring
    · simp
  rw [heq]
  have h3 := sum_chainEnv_mul_le hγ0.le (r := 3) (by push_cast; linarith) N k
  have h5 := sum_chainEnv_mul_le hγ0.le (r := 5) (by push_cast; linarith) N k
  have hs3 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * chainEnv γ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 3)
      (chainEnv_nonneg γ k n)
  have hs5 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * chainEnv γ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 5)
      (chainEnv_nonneg γ k n)
  have hq : ratioPow γ y k * ratioPow γ x k ^ 3 ≤ Q ^ 6 :=
    calc ratioPow γ y k * ratioPow γ x k ^ 3 ≤ Q * Q ^ 3 :=
          mul_le_mul hQy (pow_le_pow_left₀ hqx.le hQx 3) (by positivity) (by positivity)
      _ ≤ Q ^ 6 := by
          rw [← pow_succ']; exact pow_le_pow_right₀ hQ1 (by norm_num)
  have hκ : 0 ≤ τ / γ := div_nonneg hτ hγ0.le
  have hc := (pairConst_ge hγ0 hγ1 hτ).2.1
  calc τ * u * (ratioPow γ y k * ratioPow γ x k ^ 3) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * chainEnv γ k n +
        τ * u * (τ / γ) * (ratioPow γ y k * ratioPow γ x k ^ 3) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * chainEnv γ k n
      ≤ τ * u * Q ^ 6 * 2 + τ * u * (τ / γ) * Q ^ 6 * 2 :=
        add_le_add (mul_le_mul (mul_le_mul_of_nonneg_left hq (by positivity)) h3 hs3
          (by positivity)) (mul_le_mul (mul_le_mul_of_nonneg_left hq (by positivity)) h5 hs5
          (by positivity))
    _ = 2 * τ * (1 + τ / γ) * (Q ^ 6 * u) := by ring
    _ ≤ pairConst γ τ * (Q ^ 6 * u) := mul_le_mul_of_nonneg_right hc (by positivity)
    _ = pairConst γ τ * (Q ^ 6 / (((x : ℝ) + 1) * ((y : ℝ) + 1))) := by rw [hu]; ring

/-- **`S(k; y, z)`** for `k < y, z`: the pair decays at the earliest observable. -/
theorem pairSum_first_le {γ τ Q : ℝ} (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 12) (hτ : 0 ≤ τ) (N : ℕ)
    {y z k : ℕ} (hky : k < y) (hkz : k < z) (hQ1 : 1 ≤ Q) (hQy : ratioPow γ y k ≤ Q)
    (hQz : ratioPow γ z k ≤ Q) :
    pairSum γ τ N k y z ≤ pairConst γ τ * (Q ^ 6 / (((y : ℝ) + 1) * ((z : ℝ) + 1))) := by
  have hγ1 : γ ≤ 1 := by linarith
  have hqy := ratioPow_pos γ y k
  have hqz := ratioPow_pos γ z k
  obtain ⟨u, hu⟩ : ∃ u : ℝ, u = 1 / (((y : ℝ) + 1) * ((z : ℝ) + 1)) := ⟨_, rfl⟩
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  refine (pairSum_le hτ (w := fun n => if n < k then τ * ratioPow γ z k / ((z : ℝ) + 1) *
    ((1 + ratioPow γ k n * ratioPow γ k n / γ) / ((k : ℝ) + 1)) else 0)
    fun n => sum_profileEnv_pairDecay_up hγ0 hγ1 hτ hkz N n).trans ?_
  have heq : ∑ n ∈ Finset.range N, profileEnv γ τ y n * ratioPow γ k n ^ 3 *
      (if n < k then τ * ratioPow γ z k / ((z : ℝ) + 1) *
        ((1 + ratioPow γ k n * ratioPow γ k n / γ) / ((k : ℝ) + 1)) else 0) =
      τ ^ 2 * u * (ratioPow γ y k * ratioPow γ z k) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * chainEnv γ k n +
        τ ^ 2 * u / γ * (ratioPow γ y k * ratioPow γ z k) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * chainEnv γ k n := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun n _ => ?_
    unfold chainEnv
    split_ifs with hn
    · rw [profileEnv_of_lt (hn.trans hky), ← ratioPow_mul γ y k n, hu]
      simp only [div_eq_mul_inv, mul_inv]
      ring
    · simp
  rw [heq]
  have h3 := sum_chainEnv_mul_le hγ0.le (r := 3) (by push_cast; linarith) N k
  have h5 := sum_chainEnv_mul_le hγ0.le (r := 5) (by push_cast; linarith) N k
  have hs3 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * chainEnv γ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 3)
      (chainEnv_nonneg γ k n)
  have hs5 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * chainEnv γ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 5)
      (chainEnv_nonneg γ k n)
  have hq : ratioPow γ y k * ratioPow γ z k ≤ Q ^ 6 :=
    calc ratioPow γ y k * ratioPow γ z k ≤ Q * Q :=
          mul_le_mul hQy hQz hqz.le (by positivity)
      _ ≤ Q ^ 6 := by
          rw [← pow_two]; exact pow_le_pow_right₀ hQ1 (by norm_num)
  have hc := (pairConst_ge hγ0 hγ1 hτ).2.2.1
  calc τ ^ 2 * u * (ratioPow γ y k * ratioPow γ z k) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * chainEnv γ k n +
        τ ^ 2 * u / γ * (ratioPow γ y k * ratioPow γ z k) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * chainEnv γ k n
      ≤ τ ^ 2 * u * Q ^ 6 * 2 + τ ^ 2 * u / γ * Q ^ 6 * 2 :=
        add_le_add (mul_le_mul (mul_le_mul_of_nonneg_left hq (by positivity)) h3 hs3
          (by positivity)) (mul_le_mul (mul_le_mul_of_nonneg_left hq (by positivity)) h5 hs5
          (by positivity))
    _ = 2 * (τ ^ 2 + τ * (τ / γ)) * (Q ^ 6 * u) := by ring
    _ ≤ pairConst γ τ * (Q ^ 6 * u) := mul_le_mul_of_nonneg_right hc (by positivity)
    _ = pairConst γ τ * (Q ^ 6 / (((y : ℝ) + 1) * ((z : ℝ) + 1))) := by rw [hu]; ring

/-- The hypotheses of `pairSum_low_le` and `pairSum_first_le` hold at `γ = 1/16`, `τ = 1`
and the tokens `0 < 1 = 1`. -/
example :=
  have hQ := one_le_ratioPow (γ := 1 / 16) (by norm_num) (Nat.zero_le 1)
  And.intro (pairSum_low_le (γ := 1 / 16) (τ := 1) (by norm_num) (by norm_num) zero_le_one 1
    (x := 1) (y := 1) (k := 0) Nat.one_pos Nat.one_pos hQ le_rfl le_rfl)
  (pairSum_first_le (γ := 1 / 16) (τ := 1) (by norm_num) (by norm_num) zero_le_one 1 (y := 1)
    (z := 1) (k := 0) Nat.one_pos Nat.one_pos hQ le_rfl le_rfl)

end Kinetic
end Transformer
