module

public import Transformer.Kinetic.Section3_InnerSums

/-!
# The sums of a pair

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
sums `S(x; y, z) = Σ_n Σ_M Ĝ^n_y Ĝ^M_z E(x; n, M)` (`pairSum`) of the third cumulant
for the tokens `k < j < i`. Each is at most `Λ Q⁶/((i+1)(j+1))`, with
`Λ = 4(1 + 2τ)(1 + κ)²` (`pairConst`) and `Q ≥ q_γ(i, k)`: the source's
`(ij)⁻¹ e^{√(CT log(i/k))}`. Here the earlier token `n` of the pair is weighted by
the earliest observable, `y = k` (`pairSum_up_le`, `pairSum_mid_le`).
-/

@[expose] public section

namespace Transformer
namespace Kinetic

/-- `S(x; y, z) = Σ_n Σ_M Ĝ^n_y Ĝ^M_z E(x; n, M)`, over `n, M < N`. -/
noncomputable def pairSum (γ τ : ℝ) (N x y z : ℕ) : ℝ :=
  ∑ n ∈ Finset.range N, ∑ M ∈ Finset.range N,
    profileEnv γ τ y n * profileEnv γ τ z M * pairEnv γ x n M

/-- `Λ = 4(1 + 2τ)(1 + τ/γ)²`. -/
noncomputable def pairConst (γ τ : ℝ) : ℝ := 4 * (1 + 2 * τ) * (1 + τ / γ) ^ 2

theorem pairConst_ge {γ τ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hτ : 0 ≤ τ) :
    (1 + 2 * τ) * (τ + τ / γ) ≤ pairConst γ τ ∧ 2 * τ * (1 + τ / γ) ≤ pairConst γ τ ∧
      2 * (τ ^ 2 + τ * (τ / γ)) ≤ pairConst γ τ ∧
      (1 + 2 * τ) * (1 + τ / γ) ≤ pairConst γ τ := by
  have hκ : 0 ≤ τ / γ := div_nonneg hτ hγ0.le
  have hτκ : τ ≤ τ / γ := by rw [le_div_iff₀ hγ0]; nlinarith
  have h1 : 1 + τ / γ ≤ 4 * (1 + τ / γ) ^ 2 := by nlinarith
  have h2 : τ + τ / γ ≤ 4 * (1 + τ / γ) ^ 2 := by nlinarith
  have h12 : 0 ≤ 1 + 2 * τ := by linarith
  unfold pairConst
  refine ⟨by nlinarith [mul_le_mul_of_nonneg_left h2 h12], ?_, ?_,
    by nlinarith [mul_le_mul_of_nonneg_left h1 h12]⟩
  · nlinarith [mul_le_mul_of_nonneg_left hτκ (by linarith : (0 : ℝ) ≤ 1 + τ / γ),
      mul_le_mul_of_nonneg_right h12 (sq_nonneg (1 + τ / γ))]
  · nlinarith [mul_le_mul hτκ hτκ hτ hκ, mul_le_mul_of_nonneg_left hτκ hτ,
      mul_le_mul_of_nonneg_right h12 (sq_nonneg (1 + τ / γ))]

theorem pow_le_pow_of_le {a Q : ℝ} (ha : 0 ≤ a) (h : a ≤ Q) (hQ : 1 ≤ Q) {m n : ℕ}
    (hmn : m ≤ n) : a ^ m ≤ Q ^ n :=
  (pow_le_pow_left₀ ha h m).trans (pow_le_pow_right₀ hQ hmn)

/-- **The outer sum.** Bounds `w(n)` of the inner sums bound `S(x; y, z)` by
`Σ_n Ĝ^n_y q_γ(x, n)³ w(n)`. -/
theorem pairSum_le {γ τ : ℝ} (hτ : 0 ≤ τ) {N x y z : ℕ} {w : ℕ → ℝ}
    (hw : ∀ n, ∑ M ∈ Finset.range N,
      (if n < M then profileEnv γ τ z M * pairDecay x M else 0) ≤ w n) :
    pairSum γ τ N x y z ≤
      ∑ n ∈ Finset.range N, profileEnv γ τ y n * ratioPow γ x n ^ 3 * w n := by
  unfold pairSum
  refine Finset.sum_le_sum fun n _ => ?_
  have heq : ∑ M ∈ Finset.range N, profileEnv γ τ y n * profileEnv γ τ z M * pairEnv γ x n M =
      profileEnv γ τ y n * ratioPow γ x n ^ 3 *
        ∑ M ∈ Finset.range N, (if n < M then profileEnv γ τ z M * pairDecay x M else 0) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun M _ => ?_
    unfold pairEnv
    split_ifs <;> ring
  rw [heq]
  exact mul_le_mul_of_nonneg_left (hw n)
    (mul_nonneg (profileEnv_nonneg hτ y n) (pow_nonneg (ratioPow_pos γ x n).le 3))

/-- **`S(j; k, i)`**, the later token of the pair weighted by the latest observable. -/
theorem pairSum_up_le {γ τ Q : ℝ} (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 12) (hτ : 0 ≤ τ) (N : ℕ)
    {i j k : ℕ} (hji : j < i) (hQ1 : 1 ≤ Q) (hQij : ratioPow γ i j ≤ Q)
    (hQjk : ratioPow γ j k ≤ Q) :
    pairSum γ τ N j k i ≤ pairConst γ τ * (Q ^ 6 / (((i : ℝ) + 1) * ((j : ℝ) + 1))) := by
  have hγ1 : γ ≤ 1 := by linarith
  have hqij := ratioPow_pos γ i j
  have hqjk := ratioPow_pos γ j k
  obtain ⟨u, hu⟩ : ∃ u : ℝ, u = 1 / (((i : ℝ) + 1) * ((j : ℝ) + 1)) := ⟨_, rfl⟩
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  refine (pairSum_le hτ (w := fun n => τ * ratioPow γ i j / ((i : ℝ) + 1) *
    ((1 + ratioPow γ j n * ratioPow γ j n / γ) / ((j : ℝ) + 1))) fun n => ?_).trans ?_
  · refine (sum_profileEnv_pairDecay_up hγ0 hγ1 hτ hji N n).trans ?_
    have := ratioPow_pos γ j n
    split_ifs
    · exact le_rfl
    · positivity
  have heq : ∑ n ∈ Finset.range N, profileEnv γ τ k n * ratioPow γ j n ^ 3 *
      (τ * ratioPow γ i j / ((i : ℝ) + 1) *
        ((1 + ratioPow γ j n * ratioPow γ j n / γ) / ((j : ℝ) + 1))) =
      τ * ratioPow γ i j * u * ratioPow γ j k ^ 3 *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * profileEnv γ τ k n +
        τ * ratioPow γ i j * u / γ * ratioPow γ j k ^ 5 *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * profileEnv γ τ k n := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun n _ => ?_
    rw [← ratioPow_mul γ j k n, hu]
    simp only [div_eq_mul_inv, mul_inv]
    ring
  rw [heq]
  have h3 := sum_profileEnv_mul_le hγ0.le hτ (r := 3) (by push_cast; linarith) N k
  have h5 := sum_profileEnv_mul_le hγ0.le hτ (r := 5) (by push_cast; linarith) N k
  have hs3 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * profileEnv γ τ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 3)
      (profileEnv_nonneg hτ k n)
  have hs5 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * profileEnv γ τ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 5)
      (profileEnv_nonneg hτ k n)
  have hq3 := pow_le_pow_of_le hqjk.le hQjk hQ1 (by norm_num : 3 ≤ 5)
  have hq5 := pow_le_pow_of_le hqjk.le hQjk hQ1 (le_refl 5)
  have hc := (pairConst_ge hγ0 hγ1 hτ).1
  have hQ5 : 0 ≤ Q ^ 5 := by positivity
  calc τ * ratioPow γ i j * u * ratioPow γ j k ^ 3 *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * profileEnv γ τ k n +
        τ * ratioPow γ i j * u / γ * ratioPow γ j k ^ 5 *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * profileEnv γ τ k n
      ≤ τ * ratioPow γ i j * u * Q ^ 5 * (1 + 2 * τ) +
          τ * ratioPow γ i j * u / γ * Q ^ 5 * (1 + 2 * τ) :=
        add_le_add (mul_le_mul (mul_le_mul_of_nonneg_left hq3 (by positivity)) h3 hs3
          (by positivity)) (mul_le_mul (mul_le_mul_of_nonneg_left hq5 (by positivity)) h5 hs5
          (by positivity))
    _ = (1 + 2 * τ) * (τ + τ / γ) * (ratioPow γ i j * Q ^ 5 * u) := by ring
    _ ≤ pairConst γ τ * (Q * Q ^ 5 * u) :=
        mul_le_mul hc (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hQij hQ5) hu0)
          (by positivity) (le_trans (by positivity) hc)
    _ = pairConst γ τ * (Q ^ 6 / (((i : ℝ) + 1) * ((j : ℝ) + 1))) := by rw [hu]; ring

/-- **`S(i; k, j)`**, the later token of the pair weighted by the middle observable. -/
theorem pairSum_mid_le {γ τ Q : ℝ} (hγ0 : 0 < γ) (hγ : γ ≤ 1 / 12) (hτ : 0 ≤ τ) (N : ℕ)
    {i j k : ℕ} (hji : j < i) (hQ1 : 1 ≤ Q) (hQik : ratioPow γ i k ≤ Q)
    (hQjk : ratioPow γ j k ≤ Q) :
    pairSum γ τ N i k j ≤ pairConst γ τ * (Q ^ 6 / (((i : ℝ) + 1) * ((j : ℝ) + 1))) := by
  have hγ1 : γ ≤ 1 := by linarith
  have hqik := ratioPow_pos γ i k
  have hqjk := ratioPow_pos γ j k
  obtain ⟨u, hu⟩ : ∃ u : ℝ, u = 1 / (((i : ℝ) + 1) * ((j : ℝ) + 1)) := ⟨_, rfl⟩
  have hu0 : 0 ≤ u := by rw [hu]; positivity
  refine (pairSum_le hτ (w := fun n => (1 + τ * ratioPow γ j n * ratioPow γ j n / γ) /
    ((j : ℝ) + 1) * (1 / ((i : ℝ) + 1))) fun n => ?_).trans ?_
  · refine (sum_profileEnv_pairDecay_down hγ0 hγ1 hτ hji N n).trans ?_
    have := ratioPow_pos γ j n
    split_ifs
    · exact le_rfl
    · positivity
  have heq : ∑ n ∈ Finset.range N, profileEnv γ τ k n * ratioPow γ i n ^ 3 *
      ((1 + τ * ratioPow γ j n * ratioPow γ j n / γ) / ((j : ℝ) + 1) * (1 / ((i : ℝ) + 1))) =
      u * ratioPow γ i k ^ 3 * ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * profileEnv γ τ k n +
        u * (τ / γ) * (ratioPow γ i k ^ 3 * ratioPow γ j k ^ 2) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * profileEnv γ τ k n := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun n _ => ?_
    rw [← ratioPow_mul γ i k n, ← ratioPow_mul γ j k n, hu]
    simp only [div_eq_mul_inv, mul_inv]
    ring
  rw [heq]
  have h3 := sum_profileEnv_mul_le hγ0.le hτ (r := 3) (by push_cast; linarith) N k
  have h5 := sum_profileEnv_mul_le hγ0.le hτ (r := 5) (by push_cast; linarith) N k
  have hs3 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * profileEnv γ τ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 3)
      (profileEnv_nonneg hτ k n)
  have hs5 : 0 ≤ ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * profileEnv γ τ k n :=
    Finset.sum_nonneg fun n _ => mul_nonneg (pow_nonneg (ratioPow_pos γ k n).le 5)
      (profileEnv_nonneg hτ k n)
  have hq3 := pow_le_pow_of_le hqik.le hQik hQ1 (by norm_num : 3 ≤ 6)
  have hq32 : ratioPow γ i k ^ 3 * ratioPow γ j k ^ 2 ≤ Q ^ 6 := by
    have := pow_le_pow_of_le hqik.le hQik hQ1 (le_refl 3)
    have := pow_le_pow_of_le hqjk.le hQjk hQ1 (le_refl 2)
    calc ratioPow γ i k ^ 3 * ratioPow γ j k ^ 2 ≤ Q ^ 3 * Q ^ 2 :=
          mul_le_mul (by assumption) (by assumption) (by positivity) (by positivity)
      _ ≤ Q ^ 6 := by rw [← pow_add]; exact pow_le_pow_right₀ hQ1 (by norm_num)
  have hκ : 0 ≤ τ / γ := div_nonneg hτ hγ0.le
  have hc := (pairConst_ge hγ0 hγ1 hτ).2.2.2
  calc u * ratioPow γ i k ^ 3 * ∑ n ∈ Finset.range N, ratioPow γ k n ^ 3 * profileEnv γ τ k n +
        u * (τ / γ) * (ratioPow γ i k ^ 3 * ratioPow γ j k ^ 2) *
          ∑ n ∈ Finset.range N, ratioPow γ k n ^ 5 * profileEnv γ τ k n
      ≤ u * Q ^ 6 * (1 + 2 * τ) + u * (τ / γ) * Q ^ 6 * (1 + 2 * τ) :=
        add_le_add (mul_le_mul (mul_le_mul_of_nonneg_left hq3 hu0) h3 hs3 (by positivity))
          (mul_le_mul (mul_le_mul_of_nonneg_left hq32 (by positivity)) h5 hs5 (by positivity))
    _ = (1 + 2 * τ) * (1 + τ / γ) * (Q ^ 6 * u) := by ring
    _ ≤ pairConst γ τ * (Q ^ 6 * u) := mul_le_mul_of_nonneg_right hc (by positivity)
    _ = pairConst γ τ * (Q ^ 6 / (((i : ℝ) + 1) * ((j : ℝ) + 1))) := by rw [hu]; ring

/-- The hypotheses of `pairConst_ge`, `pow_le_pow_of_le` and `pairSum_le` hold at
`γ = τ = 1` and one token. -/
example := And.intro (pairConst_ge (γ := 1) (τ := 1) one_pos le_rfl zero_le_one) <|
  And.intro (pow_le_pow_of_le (a := 1) (Q := 1) zero_le_one le_rfl le_rfl (m := 1) (n := 2)
    (by norm_num))
  (pairSum_le (γ := 1) (τ := 1) zero_le_one (N := 1) (x := 0) (y := 0) (z := 0)
    (w := fun n => ∑ M ∈ Finset.range 1, if n < M then profileEnv 1 1 0 M * pairDecay 0 M
      else 0) (fun _ => le_rfl))

/-- The hypotheses of `pairSum_up_le` and `pairSum_mid_le` hold at `γ = 1/16`, `τ = 1`
and the tokens `0 = 0 < 1`. -/
example :=
  have hQ := one_le_ratioPow (γ := 1 / 16) (by norm_num) (Nat.zero_le 1)
  have hQ0 : ratioPow (1 / 16) 0 0 ≤ ratioPow (1 / 16) 1 0 := by rw [ratioPow_self]; exact hQ
  And.intro (pairSum_up_le (γ := 1 / 16) (τ := 1) (by norm_num) (by norm_num) zero_le_one 1
    (i := 1) (j := 0) (k := 0) Nat.one_pos hQ le_rfl hQ0)
  (pairSum_mid_le (γ := 1 / 16) (τ := 1) (by norm_num) (by norm_num) zero_le_one 1 (i := 1)
    (j := 0) (k := 0) Nat.one_pos hQ le_rfl hQ0)

end Kinetic
end Transformer
