module

public import Transformer.RoPE.LongRoPE2.Section2_Critical
public import Transformer.RoPE.LongRoPE2.Section2_Rescaling
public import Transformer.RoPE.LongRoPE.Section2_Unified

/-!
# LongRoPE2 §§2.3–2.4: NTK and YaRN against Eq. (10)

Shang et al., *LongRoPE2*, arXiv:2502.20082v1. The factor `λ_i` of Eq. (8) divides the pair's
frequency, `θ̂_i = θ_i/λ_i` (`scaledFrequency`); Eq. (10), `L/L_train ≤ λ_i`, is equivalent to the
phase bound of Eq. (9) (`phase_range_iff`).

* `critical_base_factor`: §2.3, the NTK of Liu et al., "`θ_ntk ≥ θ^{log_{L_train/2π}(L/2π)}`,
  yielding `λ_i ≥ L/L_train` (`i > d_tcd`)": every base at least `θ^k`, with
  `k = log_{L_train/2π}(L/2π)`, gives `λ_i = θ_i/θ'_i ≥ L/L_train` at every pair whose period
  is at least `L_train`;
* `ntk_aware_insufficient`, `ntk_aware_last_pair`: §2.3, "(LocalLLaMA, 2023) use
  `λ_i = s^{2i/(d−2)}`, leading to insufficient interpolation": for `s = L/L_train > 1` the phase at
  `L` exceeds the phase at `L_train` at every pair below the last; the last pair's factor is `s`;
* `yarn_factor_eq_iff`, `phi3_yarn`: §2.3, "For high dimensions, YaRN adopt PI and set
  `λ_i = L/L_train`", and §2.4, "Beyond dimension `d_tcd = 31`, YaRN strictly adheres to this
  bound (`L/L_train = 64`)": with YaRN's ramp `α = 1`, `β = 32` the factor is `s` at the pairs whose
  period is at least `L_train` and below `s` at the others; for Phi3-mini from 2k to 128k it is
  `64` exactly at the pairs `i ≥ 31`;
* `yarn_factor_lt_linear`: §2.3, "For dimensions that fall in-between use a linearly increasing
  scale factor": the ramp is linear in the blend of the frequencies; the factor
  `1/((1−γ)/s + γ)` lies strictly below the linear blend `(1−γ)·s + γ` for `0 < γ < 1`, `s ≠ 1`.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

open ScalingLaws

/-- §2.3: every base `θ' ≥ θ^k`, `k = log_{L_train/2π}(L/2π)`, gives `λ_i ≥ L/L_train` at every
pair whose period is at least `L_train`. -/
theorem critical_base_factor {base base' train target : ℝ} {d : ℕ} (hbase : 1 < base)
    (htrain : 2 * Real.pi < train) (htarget : train ≤ target)
    (hbase' : base ^ Real.logb (train / (2 * Real.pi)) (target / (2 * Real.pi)) ≤ base')
    {i : ℕ} (hi : train ≤ period (frequency base d i)) :
    target / train ≤ frequency base d i / frequency base' d i := by
  have hpi : 0 < 2 * Real.pi := by positivity
  have hb0 : 0 < base := by linarith
  have hu : 1 < train / (2 * Real.pi) := (one_lt_div hpi).mpr htrain
  have hv : 0 < target / (2 * Real.pi) := div_pos (by linarith) hpi
  have hk : 1 ≤ Real.logb (train / (2 * Real.pi)) (target / (2 * Real.pi)) := by
    rw [← Real.logb_self_eq_one hu]
    exact Real.logb_le_logb_of_le hu (by linarith) (div_le_div_of_nonneg_right htarget hpi.le)
  have hpow : (train / (2 * Real.pi)) ^ Real.logb (train / (2 * Real.pi)) (target / (2 * Real.pi))
      = target / (2 * Real.pi) := Real.rpow_logb (by linarith) hu.ne' hv
  have hwin : train / (2 * Real.pi) ≤ base ^ (2 * (i : ℝ) / (d : ℝ)) := by
    rw [frequency_period hb0] at hi
    rw [div_le_iff₀ hpi]
    linarith
  generalize Real.logb (train / (2 * Real.pi)) (target / (2 * Real.pi)) = k at hk hpow hbase'
  generalize hp : 2 * (i : ℝ) / (d : ℝ) = p at hwin
  have hp0 : 0 ≤ p := by rw [← hp]; positivity
  have hfreq : ∀ b : ℝ, 0 < b → frequency b d i = (b ^ p)⁻¹ :=
    fun b hb => by rw [frequency, neg_div, hp, Real.rpow_neg hb.le]
  have hbp : 0 < base ^ p := Real.rpow_pos_of_pos hb0 p
  have hb'0 : 0 < base' := (Real.rpow_pos_of_pos hb0 k).trans_le hbase'
  rw [hfreq base hb0, hfreq base' hb'0, inv_div_inv]
  calc target / train = (train / (2 * Real.pi)) ^ (k - 1) := by
        rw [Real.rpow_sub_one (by linarith), hpow, div_div_div_cancel_right₀ hpi.ne']
    _ ≤ (base ^ p) ^ (k - 1) := Real.rpow_le_rpow (by linarith) hwin (by linarith)
    _ = (base ^ k) ^ p / base ^ p := by
        rw [Real.rpow_sub_one hbp.ne', ← Real.rpow_mul hb0.le, ← Real.rpow_mul hb0.le, mul_comm]
    _ ≤ base' ^ p / base ^ p :=
        div_le_div_of_nonneg_right (Real.rpow_le_rpow (Real.rpow_pos_of_pos hb0 k).le hbase' hp0)
          hbp.le

example : (131072 : ℝ) / 2048 ≤ frequency 10000 96 31 /
    frequency (10000 ^ Real.logb (2048 / (2 * Real.pi)) (131072 / (2 * Real.pi))) 96 31 :=
  critical_base_factor (by norm_num) (by linarith [Real.pi_lt_d2]) (by norm_num) le_rfl
    (not_lt.mp fun h => absurd ((phi3_critical.2.2 31).1 h) (by omega))

/-- §2.3: below the last pair, the NTK-aware factor `s^{2i/(d−2)}` is below `s = L/L_train`, so
the phase at `L = s·L_train` exceeds the phase at `L_train`. -/
theorem ntk_aware_insufficient {s train omega : ℝ} {d i : ℕ} (hs : 1 < s) (htrain : 0 < train)
    (homega : 0 < omega) (hi : 2 * i + 2 < d) :
    LongRoPE.ntkFactor s d i < s ∧
      train * omega < s * train * scaledFrequency omega (LongRoPE.ntkFactor s d i) := by
  have hcast : ((2 * i + 2 : ℕ) : ℝ) < d := by exact_mod_cast hi
  push_cast at hcast
  have hi0 : (0 : ℝ) ≤ i := Nat.cast_nonneg i
  have hexp : 2 * (i : ℝ) / ((d : ℝ) - 2) < 1 := by
    rw [div_lt_one (by linarith)]
    linarith
  have hfac : LongRoPE.ntkFactor s d i < s := by
    have h := Real.rpow_lt_rpow_of_exponent_lt hs hexp
    rwa [Real.rpow_one] at h
  have hfac0 : 0 < LongRoPE.ntkFactor s d i := Real.rpow_pos_of_pos (by linarith) _
  refine ⟨hfac, lt_of_not_ge fun h => ?_⟩
  rw [phase_range_iff homega hfac0 htrain, mul_div_assoc, div_self htrain.ne', mul_one] at h
  linarith

example : LongRoPE.ntkFactor 64 96 31 < 64 ∧
    2048 * frequency 10000 96 31 <
      64 * 2048 * scaledFrequency (frequency 10000 96 31) (LongRoPE.ntkFactor 64 96 31) :=
  ntk_aware_insufficient (by norm_num) (by norm_num) (Real.rpow_pos_of_pos (by norm_num) _)
    (by norm_num)

/-- §2.3: the NTK-aware factor of the last pair of `d = 2(p+1)` is `s`. -/
theorem ntk_aware_last_pair (s : ℝ) {p : ℕ} (hp : 0 < p) :
    LongRoPE.ntkFactor s (2 * (p + 1)) p = s := by
  have hp' : (0 : ℝ) < p := by exact_mod_cast hp
  rw [LongRoPE.ntkFactor, show (((2 * (p + 1) : ℕ) : ℝ) - 2) = 2 * (p : ℝ) by push_cast; ring,
    div_self (mul_pos two_pos hp').ne', Real.rpow_one]

example : LongRoPE.ntkFactor 64 (2 * (47 + 1)) 47 = 64 := ntk_aware_last_pair 64 (by norm_num)

/-- §§2.3–2.4: YaRN's factor is `s` at the pairs whose period is at least `L_train` and below
`s` at the pairs with a shorter period. -/
theorem yarn_factor_eq_iff {base s train : ℝ} {d : ℕ} (hbase : 0 < base) (hs : 1 < s)
    (htrain : 0 < train) (i : ℕ) :
    (train ≤ period (frequency base d i) → LongRoPE.yarnFactor base d s train i = s) ∧
      (period (frequency base d i) < train → LongRoPE.yarnFactor base d s train i < s) := by
  have hθ : 0 < frequency base d i := Real.rpow_pos_of_pos hbase _
  have hT : 0 < period (frequency base d i) := div_pos (by positivity) hθ
  have hrot : YaRN.rotations train (frequency base d i) = train / period (frequency base d i) := by
    rw [YaRN.rotations, period]
    field_simp
  constructor
  · intro h
    apply (LongRoPE.yarnFactor_bounds hs.le d train i).2.1
    have hr : YaRN.rotations train (frequency base d i) ≤ 1 := by
      rw [hrot, div_le_one hT]
      exact h
    rw [LongRoPE.yarnWeight, YaRN.ramp]
    exact max_eq_left
      ((min_le_right _ _).trans (div_nonpos_of_nonpos_of_nonneg (by linarith) (by norm_num)))
  · intro h
    have hr : 1 < YaRN.rotations train (frequency base d i) := by
      rw [hrot, one_lt_div hT]
      exact h
    have hγ : 0 < LongRoPE.yarnWeight base d train i := by
      rw [LongRoPE.yarnWeight, YaRN.ramp]
      exact lt_max_of_lt_right (lt_min one_pos (div_pos (by linarith) (by norm_num)))
    have hγ1 : LongRoPE.yarnWeight base d train i ≤ 1 := (YaRN.ramp_bounds _ _ _).2
    have hs0 : 0 < s := by linarith
    have hD : 0 < (1 - LongRoPE.yarnWeight base d train i) / s +
        LongRoPE.yarnWeight base d train i := by
      have := div_nonneg (sub_nonneg.mpr hγ1) hs0.le
      linarith
    rw [LongRoPE.yarnFactor, div_lt_iff₀ hD]
    have key : s * ((1 - LongRoPE.yarnWeight base d train i) / s +
        LongRoPE.yarnWeight base d train i) =
          1 + LongRoPE.yarnWeight base d train i * (s - 1) := by
      field_simp
      ring
    rw [key]
    nlinarith [mul_pos hγ (sub_pos.mpr hs)]

example : (2048 ≤ period (frequency 10000 96 40) → LongRoPE.yarnFactor 10000 96 64 2048 40 = 64) ∧
    (period (frequency 10000 96 40) < 2048 → LongRoPE.yarnFactor 10000 96 64 2048 40 < 64) :=
  yarn_factor_eq_iff (by norm_num) (by norm_num) (by norm_num) 40

/-- §2.4: Phi3-mini from `2048` to `131072` tokens has `L/L_train = 64`, and YaRN's factor is
`64` exactly at the pairs `i ≥ 31`. -/
theorem phi3_yarn :
    YaRN.contextScale 2048 131072 = 64 ∧
      ∀ i : ℕ, (31 ≤ i → LongRoPE.yarnFactor 10000 96 64 2048 i = 64) ∧
        (i < 31 → LongRoPE.yarnFactor 10000 96 64 2048 i < 64) := by
  refine ⟨by rw [YaRN.contextScale, max_eq_right (by norm_num)]; norm_num, fun i => ?_⟩
  have h := yarn_factor_eq_iff (base := 10000) (s := 64) (train := 2048) (d := 96)
    (by norm_num) (by norm_num) (by norm_num) i
  exact ⟨fun hi => h.1 (not_lt.mp fun h' => absurd ((phi3_critical.2.2 i).1 h') (by omega)),
    fun hi => h.2 ((phi3_critical.2.2 i).2 hi)⟩

/-- §2.3: YaRN's middle-band factor lies strictly below the linear blend of `1` and `s`. -/
theorem yarn_factor_lt_linear {s γ : ℝ} (hs : 0 < s) (hs1 : s ≠ 1) (hγ0 : 0 < γ)
    (hγ1 : γ < 1) : 1 / ((1 - γ) / s + γ) < (1 - γ) * s + γ := by
  have hD : 0 < (1 - γ) / s + γ := by
    have := div_pos (sub_pos.mpr hγ1) hs
    linarith
  rw [div_lt_iff₀ hD]
  have hsq : 0 < (s - 1) ^ 2 :=
    lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 (sub_ne_zero.mpr hs1)))
  have key : ((1 - γ) * s + γ) * ((1 - γ) / s + γ) = 1 + γ * (1 - γ) * (s - 1) ^ 2 / s := by
    field_simp
    ring
  rw [key]
  have := div_pos (mul_pos (mul_pos hγ0 (sub_pos.mpr hγ1)) hsq) hs
  linarith

example : 1 / ((1 - 1 / 2) / 64 + 1 / 2) < (1 - 1 / 2) * 64 + (1 / 2 : ℝ) :=
  yarn_factor_lt_linear (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.RoPE.LongRoPE2
