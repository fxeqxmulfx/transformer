module

public import Transformer.RoPE.LongRoPE2.Section3_Hypothesis

/-!
# LongRoPE2 §3.2, Algorithm 1 and Fig. 4: the candidates for the real critical dimension

Shang et al., *LongRoPE2*, arXiv:2502.20082v1. Algorithm 1, line 2,
`d10_tcd = ⌈(d/2)·log_θbase(L_train/(2π×10))⌉`, "Compute the dim with a theoretical 10 periods";
line 3, "for int `d_rcd = d10_tcd` to `d_tcd`"; §3.2, "candidates ranging from `d10_tcd`, where
the theoretical RoPE period spans 10 periods in the pre-training window, and `d_tcd`"; and
"The practical critical dimensions (`d_rcd`) are shifted earlier to 25 and 30, compared to the
theoretical values `d_tcd` of 31 and 35, respectively" (Phi3-mini and LLaMA3-8B).

* `tenPeriodPair`, `ten_periods_iff`: line 2 has no factor `2`, unlike Eq. (7), so it is a pair
  index: more than ten periods fit into `L_train` exactly at the pairs below it;
* `period_bounds`: a period bracketed by integer powers, for the numerics below;
* `phi3_candidates`: Phi3-mini (`d = 96`, base `10000`, `L_train = 2048`) has `d10_tcd = 19`;
  with the pair index `d_tcd = 31` the candidates `19, …, 31` are 13 pairs and contain 25, while
  Eq. (7)'s `62` lies beyond the last pair `47`;
* `llama3_candidates`: LLaMA3-8B, with the base `500000` and window `8192` of Men et al.
  (arXiv:2405.14591v1, Table 1) and the head dimension `d = 128`, which neither paper prints,
  has `d_tcd = 35` and `d10_tcd = 24`, and 30 lies between them.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

open ScalingLaws

/-- Algorithm 1, line 2: the pair index `d10_tcd = ⌈(d/2)·log_θbase(L_train/(2π×10))⌉`. -/
noncomputable def tenPeriodPair (base : ℝ) (d : ℕ) (train : ℝ) : ℤ :=
  ⌈((d : ℝ) / 2) * Real.logb base (train / (2 * Real.pi * 10))⌉

/-- Algorithm 1, line 2: more than ten periods of the pair `i` fit into `L_train` exactly when
`i < d10_tcd`. -/
theorem ten_periods_iff {base train : ℝ} {d : ℕ} (hbase : 1 < base) (htrain : 0 < train)
    (hd : 0 < d) (i : ℕ) :
    10 * period (frequency base d i) < train ↔ (i : ℤ) < tenPeriodPair base d train := by
  have heq : tenPeriodPair base d train = criticalPair base d (train / 10) := by
    rw [tenPeriodPair, criticalPair, div_div, mul_comm 10 (2 * Real.pi)]
  rw [heq, ← period_lt_window_iff_critical hbase (div_pos htrain (by norm_num)) hd i,
    lt_div_iff₀ (by norm_num), mul_comm 10]

example :
    10 * period (frequency 10000 96 30) < 2048 ↔ ((30 : ℕ) : ℤ) < tenPeriodPair 10000 96 2048 :=
  ten_periods_iff (by norm_num) (by norm_num) (by norm_num) 30

/-- The period of the pair `i` of dimension `2k`, bracketed by integer powers of the base. -/
theorem period_bounds {base lo hi : ℝ} {d k i : ℕ} (hbase : 0 < base) (hd : d = 2 * k)
    (hk : 0 < k) (hlo : 0 ≤ lo) (hhi : 0 ≤ hi) (hl : lo ^ k < base ^ i) (hh : base ^ i < hi ^ k) :
    2 * Real.pi * lo < period (frequency base d i) ∧
      period (frequency base d i) < 2 * Real.pi * hi := by
  subst hd
  rw [frequency_period hbase, Nat.cast_mul, Nat.cast_two,
    mul_div_mul_left _ _ (two_ne_zero : (2 : ℝ) ≠ 0)]
  have h2 : 0 < 2 * Real.pi := by positivity
  exact ⟨mul_lt_mul_of_pos_left (YaRN.lt_rpow_div hbase.le hlo hk hl) h2,
    mul_lt_mul_of_pos_left (YaRN.rpow_div_lt hbase.le hhi hk hh) h2⟩

example : 2 * Real.pi * 0 < period (frequency 10000 96 18) ∧
    period (frequency 10000 96 18) < 2 * Real.pi * 32 :=
  period_bounds (k := 48) (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num)
    (by norm_num) (by norm_num)

/-- §3.2 and Fig. 4: Phi3-mini's candidates for `d_rcd`. -/
theorem phi3_candidates :
    tenPeriodPair 10000 96 2048 = 19 ∧
      (∀ i : ℕ, 10 * period (frequency 10000 96 i) < 2048 ↔ i < 19) ∧
      Finset.Icc (tenPeriodPair 10000 96 2048).toNat (criticalPair 10000 96 2048).toNat =
        Finset.Icc 19 31 ∧
      (Finset.Icc 19 31).card = 13 ∧ 25 ∈ Finset.Icc 19 31 ∧ 25 < 31 ∧
      47 < criticalDimension 10000 96 2048 := by
  have h18 : 10 * period (frequency 10000 96 18) < 2048 := by
    have h := (period_bounds (base := 10000) (d := 96) (k := 48) (i := 18) (lo := 0) (hi := 32)
      (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num)).2
    linarith [Real.pi_lt_d2]
  have h19 : 2048 ≤ 10 * period (frequency 10000 96 19) := by
    have h := (period_bounds (base := 10000) (d := 96) (k := 48) (i := 19) (lo := 35) (hi := 40)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)).1
    linarith [Real.pi_gt_d2]
  have hsplit := fun i => ten_periods_iff (base := 10000) (train := 2048) (d := 96)
    (by norm_num) (by norm_num) (by norm_num) i
  have hc : tenPeriodPair 10000 96 2048 = 19 := by
    have a := (hsplit 18).1 h18
    have b := mt (hsplit 19).2 (not_lt.mpr h19)
    push_cast at a b
    omega
  refine ⟨hc, fun i => by rw [hsplit i, hc]; omega, by rw [hc, phi3_critical.1]; rfl,
    by simp, by simp, by norm_num, by rw [phi3_critical.2.1]; norm_num⟩

/-- §3.2 and Fig. 4: LLaMA3-8B's `d_tcd = 35` and `d10_tcd = 24` for the base `500000`, the
window `8192` and `d = 128`. -/
theorem llama3_candidates :
    criticalPair 500000 128 8192 = 35 ∧ tenPeriodPair 500000 128 8192 = 24 ∧
      (∀ i : ℕ, period (frequency 500000 128 i) < 8192 ↔ i < 35) ∧
      (∀ i : ℕ, 10 * period (frequency 500000 128 i) < 8192 ↔ i < 24) ∧
      30 ∈ Finset.Icc 24 35 ∧ 30 < 35 := by
  have hb := fun (i : ℕ) (lo hi : ℝ) (hlo : 0 ≤ lo) (hhi : 0 ≤ hi)
      (hl : lo ^ 64 < (500000 : ℝ) ^ i) (hh : (500000 : ℝ) ^ i < hi ^ 64) =>
    period_bounds (k := 64) (i := i) (d := 128) (lo := lo) (hi := hi) (by norm_num)
      (by norm_num) (by norm_num) hlo hhi hl hh
  have h34 := (hb 34 0 1100 le_rfl (by norm_num) (by norm_num) (by norm_num)).2
  have h35 := (hb 35 1305 1400 (by norm_num) (by norm_num) (by norm_num) (by norm_num)).1
  have h23 := (hb 23 0 120 le_rfl (by norm_num) (by norm_num) (by norm_num)).2
  have h24 := (hb 24 131 140 (by norm_num) (by norm_num) (by norm_num) (by norm_num)).1
  have hsplit := fun i => period_lt_window_iff_critical (base := 500000) (window := 8192)
    (dimension := 128) (by norm_num) (by norm_num) (by norm_num) i
  have hsplit10 := fun i => ten_periods_iff (base := 500000) (train := 8192) (d := 128)
    (by norm_num) (by norm_num) (by norm_num) i
  have hc : criticalPair 500000 128 8192 = 35 := by
    have a := (hsplit 34).1 (by linarith [Real.pi_lt_d2])
    have b := mt (hsplit 35).2 (not_lt.mpr (by linarith [Real.pi_gt_d4]))
    push_cast at a b
    omega
  have hc10 : tenPeriodPair 500000 128 8192 = 24 := by
    have a := (hsplit10 23).1 (by linarith [Real.pi_lt_d2])
    have b := mt (hsplit10 24).2 (not_lt.mpr (by linarith [Real.pi_gt_d2]))
    push_cast at a b
    omega
  exact ⟨hc, hc10, fun i => by rw [hsplit i, hc]; omega, fun i => by rw [hsplit10 i, hc10]; omega,
    by simp, by norm_num⟩

end Transformer.RoPE.LongRoPE2
