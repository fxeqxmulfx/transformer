module

public import Transformer.RoPE.ScalingLaws.Section3_Periods
public import Transformer.RoPE.YaRN.Section3_Pressure
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# LongRoPE2 §§2.1–2.2: periods, the critical dimension and RoPE OOD

Shang et al., *LongRoPE2*, arXiv:2502.20082v1. Eq. (5) is `frequency`, Eq. (6) is `period`, and
the split behind Eq. (7) is `ScalingLaws.period_lt_window_iff_critical`. Phi3-mini has `d = 96`,
`θ_base = 10000` and `L_train = 2048`; pairs are indexed `i = 0, …, d/2 − 1`, as in Eq. (5).

* `period_lt_iff_two_mul_lt`: Eq. (7) as printed, `d_tcd = 2⌈(d/2)·log_θbase(L_train/2π)⌉`,
  counts real coordinates: the period `T_i` lies below `L_train` exactly when `2i < d_tcd`, so
  §2.2's split "`T_i < L_train`, `i < d_tcd`" holds for the pair index `d_tcd/2`;
* `phi3_critical`: §2.2, "the critical dimension is 62, corresponding to the 31st cosine
  dimension": Eq. (7) gives `62`, and `T_i < 2048` holds exactly for the pairs `i < 31`, so the
  8th and 16th dimensions (`i = 7, 15`) of Fig. 2(a) complete full periods within `2048` tokens
  and the 40th (`i = 39`) does not;
* `phi3_periods`: §2.1, "the period length of the highest dimension (i.e., the 48th cosine
  dimension) is 51861, covering less than 4% of a full period": `51861 < T_47 < 51862` and
  `2048 < 0.04·T_47`; §3.1, "the 8th dimension has a short period of 24": `24 < T_7 < 25`, and a
  2048-token sample holds `85` full periods of it, not `86`;
* `cos_beyond_window`, `phi3_ood`: §2.2 and Fig. 2(a), "When directly extrapolated to 4k, the
  cosine values between 2k and 4k fall outside the pre-trained range": when `L·θ ≤ π`, the cosine
  at every position in `(L_train, L]` lies below the cosine at every position in `[0, L_train]`;
  the 40th cosine dimension of Phi3-mini (`i = 39`) has `4096·θ_39 ≤ π`.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

open ScalingLaws

/-- Eq. (7) as printed: `d_tcd = 2⌈(d/2)·log_θbase(L_train/2π)⌉`. -/
noncomputable def criticalDimension (base : ℝ) (d : ℕ) (train : ℝ) : ℤ :=
  2 * ⌈((d : ℝ) / 2) * Real.logb base (train / (2 * Real.pi))⌉

/-- Eq. (7) counts real coordinates: the period of the pair `i` lies below `L_train` exactly
when `2i < d_tcd`. -/
theorem period_lt_iff_two_mul_lt {base train : ℝ} {d : ℕ} (hbase : 1 < base)
    (htrain : 0 < train) (hd : 0 < d) (i : ℕ) :
    period (frequency base d i) < train ↔ 2 * (i : ℤ) < criticalDimension base d train := by
  rw [period_lt_window_iff_critical hbase htrain hd]
  show _ ↔ 2 * (i : ℤ) < 2 * criticalPair base d train
  omega

example : period (frequency 10000 96 30) < 2048 ↔
    2 * ((30 : ℕ) : ℤ) < criticalDimension 10000 96 2048 :=
  period_lt_iff_two_mul_lt (by norm_num) (by norm_num) (by norm_num) 30

/-- Phi3-mini's periods, `T_i = 2π·10000^{i/48}`. -/
theorem phi3_period (i : ℕ) :
    period (frequency 10000 96 i) = 2 * Real.pi * 10000 ^ ((i : ℝ) / ((48 : ℕ) : ℝ)) := by
  rw [frequency_period (by norm_num)]
  congr 2
  push_cast
  ring

/-- Bounds on Phi3-mini's period `T_i` from integer powers: `lo^48 < 10000^i < hi^48` gives
`2π·lo < T_i < 2π·hi`. -/
theorem phi3_period_bounds {i : ℕ} {lo hi : ℝ} (hlo : 0 ≤ lo) (hhi : 0 ≤ hi)
    (hl : lo ^ 48 < (10000 : ℝ) ^ i) (hh : (10000 : ℝ) ^ i < hi ^ 48) :
    2 * Real.pi * lo < period (frequency 10000 96 i) ∧
      period (frequency 10000 96 i) < 2 * Real.pi * hi := by
  rw [phi3_period]
  have h2 : 0 < 2 * Real.pi := by positivity
  exact ⟨mul_lt_mul_of_pos_left (YaRN.lt_rpow_div (by norm_num) hlo (by norm_num) hl) h2,
    mul_lt_mul_of_pos_left (YaRN.rpow_div_lt (by norm_num) hhi (by norm_num) hh) h2⟩

example : 2 * Real.pi * 0 < period (frequency 10000 96 30) ∧
    period (frequency 10000 96 30) < 2 * Real.pi * 320 :=
  phi3_period_bounds le_rfl (by norm_num) (by norm_num) (by norm_num)

/-- §2.2: for Phi3-mini, Eq. (7) gives `d_tcd = 62`, the pair index `31`, and the periods below
`L_train = 2048` are exactly those of the pairs `i < 31`. -/
theorem phi3_critical :
    criticalPair 10000 96 2048 = 31 ∧ criticalDimension 10000 96 2048 = 62 ∧
      ∀ i : ℕ, period (frequency 10000 96 i) < 2048 ↔ i < 31 := by
  have h30 : period (frequency 10000 96 30) < 2048 := by
    have h := (phi3_period_bounds (i := 30) (lo := 0) (hi := 320) le_rfl (by norm_num) (by norm_num)
      (by norm_num)).2
    linarith [Real.pi_lt_d2]
  have h31 : 2048 ≤ period (frequency 10000 96 31) := by
    have h := (phi3_period_bounds (i := 31) (lo := 350) (hi := 400) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)).1
    linarith [Real.pi_gt_d2]
  have hsplit := fun i => period_lt_window_iff_critical (base := 10000) (window := 2048)
    (dimension := 96) (by norm_num) (by norm_num) (by norm_num) i
  have hc : criticalPair 10000 96 2048 = 31 := by
    have h₁ := (hsplit 30).1 h30
    have h₂ := mt (hsplit 31).2 (not_lt.mpr h31)
    push_cast at h₁ h₂
    omega
  refine ⟨hc, ?_, fun i => ?_⟩
  · show 2 * criticalPair 10000 96 2048 = 62
    rw [hc]
    norm_num
  · rw [hsplit i, hc]
    omega

/-- §2.1 and §3.1: Phi3-mini's highest pair has `51861 < T_47 < 51862`, and `L_train = 2048`
covers less than `4%` of it; the 8th pair has `24 < T_7 < 25`, `85` of whose periods fit into
`2048` tokens and `86` do not. -/
theorem phi3_periods :
    51861 < period (frequency 10000 96 47) ∧ period (frequency 10000 96 47) < 51862 ∧
      2048 < 0.04 * period (frequency 10000 96 47) ∧
      24 < period (frequency 10000 96 7) ∧ period (frequency 10000 96 7) < 25 ∧
      85 * period (frequency 10000 96 7) < 2048 ∧ 2048 < 86 * period (frequency 10000 96 7) := by
  have h47 := phi3_period_bounds (i := 47) (lo := 8254) (hi := 8254.05) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
  have h7 := phi3_period_bounds (i := 7) (lo := 3.83) (hi := 3.834) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    linarith [Real.pi_gt_d6, Real.pi_lt_d6, h47.1, h47.2, h7.1, h7.2]

/-- §2.2: when `L·θ ≤ π`, the cosine at a position `n ∈ (L_train, L]` lies below the cosine at
every position `m ∈ [0, L_train]`, outside the range the window `[0, L_train]` produces. -/
theorem cos_beyond_window {θ train target m n : ℝ} (hθ : 0 < θ) (hm0 : 0 ≤ m)
    (hm : m ≤ train) (hn : train < n) (hnt : n ≤ target) (hpi : target * θ ≤ Real.pi) :
    Real.cos (n * θ) < Real.cos (m * θ) :=
  Real.cos_lt_cos_of_nonneg_of_le_pi (mul_nonneg hm0 hθ.le)
    ((mul_le_mul_of_nonneg_right hnt hθ.le).trans hpi)
    (mul_lt_mul_of_pos_right (hm.trans_lt hn) hθ)

/-- Fig. 2(a): the 40th cosine dimension of Phi3-mini, `i = 39`, has `4096·θ_39 ≤ π`. -/
theorem phi3_ood : 4096 * frequency 10000 96 39 ≤ Real.pi := by
  have h := (phi3_period_bounds (i := 39) (lo := 1500) (hi := 2000) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)).1
  have hθ : 0 < frequency 10000 96 39 := Real.rpow_pos_of_pos (by norm_num) _
  have hT : 8192 ≤ period (frequency 10000 96 39) := by linarith [Real.pi_gt_d2]
  rw [period, le_div_iff₀ hθ] at hT
  linarith

example : Real.cos (3000 * frequency 10000 96 39) < Real.cos (2048 * frequency 10000 96 39) :=
  cos_beyond_window (Real.rpow_pos_of_pos (by norm_num) _) (by norm_num) le_rfl (by norm_num)
    (by norm_num) phi3_ood

end Transformer.RoPE.LongRoPE2
