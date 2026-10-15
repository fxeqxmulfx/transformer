import Transformer.Homogenized.Section7_WeightedGridLimit

/-!
# Limits of the actual compensated partition sums

ArXiv:2604.01978v1, §7, the exponential maximal inequality after
`eq:Gronwall_E_selfcontained`, and the SDE identity `def:ito.formula`.
The weighted increments of the compensated observable are written as
an endpoint product minus two ordinary time-integral sums.
The derivative of the integrating factor gives the second sum by
the fundamental theorem of calculus on every actual subinterval.

Both sums converge by continuity on the compact original interval.
A bound independent of the grid size will justify passing this
identity through expectation. These are deterministic statements;
the martingale property is used in the next expectation argument.
-/

open scoped BigOperators
open Set MeasureTheory Filter Topology
namespace Transformer.Homogenized

/-- Exact finite compensation identity for the original sampled paths.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`.
The factor increments are derived from its actual derivative. -/
theorem weighted_compensated_partition_eq (C Y H V : ℝ → ℝ) (θ : ℕ → ℝ) (hθ : Monotone θ) (N : ℕ)
    (hV : ContinuousOn V (Icc (θ 0) (θ N)))
    (hd : ∀ x ∈ Icc (θ 0) (θ N), HasDerivAt C (V x) x) :
    (∑ k ∈ Finset.range N, C (θ k) * (Y (θ (k + 1)) - Y (θ k) -
      ∫ x in θ k..θ (k + 1), H x)) =
      C (θ N) * Y (θ N) - C (θ 0) * Y (θ 0) -
      (∑ k ∈ Finset.range N, C (θ k) * ∫ x in θ k..θ (k + 1), H x) -
      (∑ k ∈ Finset.range N, Y (θ (k + 1)) * ∫ x in θ k..θ (k + 1), V x) := by
  have hsub (k : ℕ) (hk : k < N) : Icc (θ k) (θ (k + 1)) ⊆ Icc (θ 0) (θ N) :=
    fun x hx => ⟨(hθ (Nat.zero_le k)).trans hx.1, hx.2.trans (hθ (Nat.succ_le_of_lt hk))⟩
  have hdiff (k : ℕ) (hk : k < N) : C (θ (k + 1)) - C (θ k) =
      ∫ x in θ k..θ (k + 1), V x := by
    symm
    apply intervalIntegral.integral_eq_sub_of_hasDerivAt
    · intro x hx
      rw [uIcc_of_le (hθ (Nat.le_succ k))] at hx
      exact hd x (hsub k hk hx)
    · apply ContinuousOn.intervalIntegrable
      rw [uIcc_of_le (hθ (Nat.le_succ k))]
      exact hV.mono (hsub k hk)
  have hp := weighted_partition_integration_by_parts C Y θ N
  have hsum : (∑ k ∈ Finset.range N, (C (θ (k + 1)) - C (θ k)) * Y (θ (k + 1))) =
      ∑ k ∈ Finset.range N, Y (θ (k + 1)) * ∫ x in θ k..θ (k + 1), V x := by
    apply Finset.sum_congr rfl
    intro k hk
    rw [hdiff k (Finset.mem_range.mp hk), mul_comm]
  rw [hsum] at hp
  simp only [mul_sub, Finset.sum_sub_distrib] at hp ⊢
  linarith

/-- Actual weighted compensated increments converge to the endpoint
product minus the two drift integrals. Source: arXiv:2604.01978v1, §7,
the integrating-factor argument for the exponential observable. -/
theorem weighted_compensated_intervalGrid_tendsto (T : ℝ) (hT : 0 < T) (C Y H V : ℝ → ℝ)
    (hC : ContinuousOn C (Icc (0 : ℝ) T)) (hY : ContinuousOn Y (Icc (0 : ℝ) T))
    (hH : ContinuousOn H (Icc (0 : ℝ) T)) (hV : ContinuousOn V (Icc (0 : ℝ) T))
    (hd : ∀ x ∈ Icc (0 : ℝ) T, HasDerivAt C (V x) x)
    (BH BV : ℝ) (hBH : 0 ≤ BH) (hBV : 0 ≤ BV)
    (hHb : ∀ x ∈ Icc (0 : ℝ) T, |H x| ≤ BH) (hVb : ∀ x ∈ Icc (0 : ℝ) T, |V x| ≤ BV) :
    Tendsto (fun n => ∑ k ∈ Finset.range (n + 1), C (intervalGridTime T n k) *
      (Y (intervalGridTime T n (k + 1)) - Y (intervalGridTime T n k) -
      ∫ x in intervalGridTime T n k..intervalGridTime T n (k + 1), H x)) atTop
      (𝓝 (C T * Y T - C 0 * Y 0 - (∫ x in (0 : ℝ)..T, C x * H x) -
        ∫ x in (0 : ℝ)..T, Y x * V x)) := by
  have hl := left_weighted_intervalGrid_integrals_tendsto T hT C H hC hH BH hBH hHb
  have hr := right_weighted_intervalGrid_integrals_tendsto T hT Y V hY hV BV hBV hVb
  have heq (n : ℕ) := weighted_compensated_partition_eq C Y H V (intervalGridTime T n)
    (intervalGridTime_monotone T hT.le n) (n + 1)
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hV)
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hd)
  simp only [intervalGridTime_zero, intervalGridTime_last] at heq
  simp_rw [heq]
  exact (tendsto_const_nhds.sub hl).sub hr

/-- A bound uniform in the grid size for the same actual compensated
partition sums. Source: arXiv:2604.01978v1, §7, the passage through
expectation in the exponential maximal-inequality argument. -/
theorem abs_weighted_compensated_intervalGrid_le (T : ℝ) (hT : 0 ≤ T) (C Y H V : ℝ → ℝ)
    (hH : ContinuousOn H (Icc (0 : ℝ) T)) (hV : ContinuousOn V (Icc (0 : ℝ) T))
    (hd : ∀ x ∈ Icc (0 : ℝ) T, HasDerivAt C (V x) x)
    (BC BY BH BV : ℝ) (hBH : 0 ≤ BH) (hBV : 0 ≤ BV)
    (hCb : ∀ x ∈ Icc (0 : ℝ) T, |C x| ≤ BC) (hYb : ∀ x ∈ Icc (0 : ℝ) T, |Y x| ≤ BY)
    (hHb : ∀ x ∈ Icc (0 : ℝ) T, |H x| ≤ BH) (hVb : ∀ x ∈ Icc (0 : ℝ) T, |V x| ≤ BV)
    (n : ℕ) :
    |∑ k ∈ Finset.range (n + 1), C (intervalGridTime T n k) *
      (Y (intervalGridTime T n (k + 1)) - Y (intervalGridTime T n k) -
      ∫ x in intervalGridTime T n k..intervalGridTime T n (k + 1), H x)| ≤
      2 * BC * BY + BC * BH * T + BY * BV * T := by
  have heq := weighted_compensated_partition_eq C Y H V (intervalGridTime T n)
    (intervalGridTime_monotone T hT n) (n + 1)
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hV)
    (by simpa only [intervalGridTime_zero, intervalGridTime_last] using hd)
  simp only [intervalGridTime_zero, intervalGridTime_last] at heq
  rw [heq]
  have hBC : 0 ≤ BC := (abs_nonneg (C 0)).trans (hCb 0 ⟨le_rfl, hT⟩)
  have hprod (x : ℝ) (hx : x ∈ Icc (0 : ℝ) T) : |C x * Y x| ≤ BC * BY := by
    rw [abs_mul]
    exact mul_le_mul (hCb x hx) (hYb x hx) (abs_nonneg _) hBC
  have hend : |C T * Y T - C 0 * Y 0| ≤ 2 * BC * BY := by
    calc
      _ ≤ |C T * Y T| + |C 0 * Y 0| := abs_sub _ _
      _ ≤ BC * BY + BC * BY := add_le_add (hprod T ⟨hT, le_rfl⟩) (hprod 0 ⟨le_rfl, hT⟩)
      _ = _ := by ring
  have hl := abs_weighted_intervalGrid_sum_le T hT C H n (intervalGridTime T n)
    (fun k hk => intervalGridTime_mem T hT n k) BC BH hBH hH hCb hHb
  have hr := abs_weighted_intervalGrid_sum_le T hT Y V n (fun k => intervalGridTime T n (k + 1))
    (fun k hk => intervalGridTime_mem T hT n (k + 1)) BY BV hBV hV hYb hVb
  exact ((abs_sub _ _).trans (add_le_add ((abs_sub _ _).trans (add_le_add hend hl)) hr))

/-- Nonconstant paths satisfy every continuity, derivative and bound
premise, and the resulting compensated endpoint expression is 1/6.
The natural-number clock with one interval also realizes the finite
partition hypotheses of `weighted_compensated_partition_eq`. -/
example : ∃ C Y H V : ℝ → ℝ,
    (0 : ℝ) < 1 ∧ ContinuousOn C (Icc (0 : ℝ) 1) ∧ ContinuousOn Y (Icc (0 : ℝ) 1) ∧
    ContinuousOn H (Icc (0 : ℝ) 1) ∧ ContinuousOn V (Icc (0 : ℝ) 1) ∧
    (∀ x ∈ Icc (0 : ℝ) 1, HasDerivAt C (V x) x) ∧
    (∀ x ∈ Icc (0 : ℝ) 1, |C x| ≤ 2) ∧ (∀ x ∈ Icc (0 : ℝ) 1, |Y x| ≤ 1) ∧
    (∀ x ∈ Icc (0 : ℝ) 1, |H x| ≤ 1) ∧ (∀ x ∈ Icc (0 : ℝ) 1, |V x| ≤ 1) ∧
    C 1 * Y 1 - C 0 * Y 0 - (∫ x in (0 : ℝ)..1, C x * H x) -
      (∫ x in (0 : ℝ)..1, Y x * V x) = 1 / 6 ∧
    Monotone (fun k : ℕ => (k : ℝ)) ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨fun x => 1 + x, fun x => x ^ 2, fun _ => 1, fun _ => 1,
    one_pos, (continuous_const.add continuous_id).continuousOn,
    (continuous_id.pow 2).continuousOn, continuousOn_const, continuousOn_const, ?_, ?_, ?_,
    fun x hx => by norm_num, fun x hx => by norm_num, ?_, ?_, zero_le_one⟩
  · intro x hx
    exact (hasDerivAt_id x).const_add 1
  · intro x hx
    rw [abs_of_nonneg (by linarith [hx.1] : 0 ≤ 1 + x)]
    linarith [hx.2]
  · intro x hx
    rw [abs_of_nonneg (sq_nonneg x)]
    nlinarith [hx.1, hx.2]
  · simp only [mul_one, one_pow, zero_pow (by decide : 2 ≠ 0), mul_zero, sub_zero]
    have hOne : IntervalIntegrable (fun _ : ℝ => (1 : ℝ)) volume 0 1 := intervalIntegrable_const
    have hId : IntervalIntegrable (fun x : ℝ => x) volume 0 1 := continuous_id.intervalIntegrable 0 1
    rw [intervalIntegral.integral_add hOne hId, integral_id, integral_pow]
    norm_num
  · intro i j hij
    change (i : ℝ) ≤ (j : ℝ)
    exact_mod_cast hij

end Transformer.Homogenized
