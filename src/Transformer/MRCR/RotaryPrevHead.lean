module

public import Transformer.MRCR.PhraseKeyRotary

/-!
# Full RoPE weighs keys arbitrarily far

In no paper. `Transformer.MRCR.PhraseKeyRotary` rules out a RoPE head that reads the token before
with a margin, a logit at most `−1` at every far key. At precision `s = 0` no margin is needed:
the weight `round(exp z)` of a key, its term in the denominator of the rounded attention, is `0`
as soon as `z < 0` (`Transformer.CRASP.Grams.val_round_exp_neg`). Full RoPE fails even so.

Full RoPE rotates every plane, at frequencies outside `2πℤ`, `sin(θ_c / 2) ≠ 0`. For a query and a
key vector the logit at the distance `t` is a RoPE score (`Transformer.MRCR.ropeScore`,
arXiv:2309.00071v3, §2.1, Eq. (5)) without a constant term, and its partial sums are bounded:
`2 sin(θ/2)` times the sum of a plane telescopes (`two_sin_mul_sum_range`,
`neg_le_sum_ropeScore`). A score below `0` from `N` on would be at most `−δ/2` at the linearly
many returns of `exists_returns`, and its partial sums would fall without bound. So the score is
at least `0` at distances as large as wanted (`exists_nonneg_ropeScore`), and there the weight is
at least `1` (`exists_one_le_weight_rope`), with RoPE's grid `base^{−2m/d}`
(`exists_one_le_weight_grid`) and with YaRN's frequencies and temperature (arXiv:2309.00071v3,
Definition 2, `exists_one_le_weight_yarn`) too. So a head of the first layer, whose vectors depend
on the token alone, weighs a key of any token placed that far beside the token before. A deeper
head, whose vectors depend on the context, and a reading through mixtures are not settled here.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.CRASP Transformer.CRASP.Grams

variable {K : ℕ}

/-- **One plane telescopes**: `2 sin(θ/2)` times the sum of `a cos θ(x + t) + b sin θ(x + t)` over
`t < T` is a difference of two values. -/
theorem two_sin_mul_sum_range (θ a b x : ℝ) (T : ℕ) :
    2 * Real.sin (θ / 2) *
        ∑ t ∈ range T, (a * Real.cos (θ * (x + t)) + b * Real.sin (θ * (x + t))) =
      (a * Real.sin (θ * (x + T) - θ / 2) - b * Real.cos (θ * (x + T) - θ / 2)) -
        (a * Real.sin (θ * x - θ / 2) - b * Real.cos (θ * x - θ / 2)) := by
  rw [mul_sum]
  have h : ∀ t : ℕ,
      2 * Real.sin (θ / 2) * (a * Real.cos (θ * (x + t)) + b * Real.sin (θ * (x + t))) =
        (a * Real.sin (θ * (x + (t + 1 : ℕ)) - θ / 2) -
            b * Real.cos (θ * (x + (t + 1 : ℕ)) - θ / 2)) -
          (a * Real.sin (θ * (x + t) - θ / 2) - b * Real.cos (θ * (x + t) - θ / 2)) := fun t => by
    push_cast
    rw [show θ * (x + (t + 1)) - θ / 2 = θ * (x + t) + θ / 2 by ring, Real.sin_add,
      Real.cos_add, Real.sin_sub, Real.cos_sub]
    ring
  rw [sum_congr rfl fun t _ => h t, sum_range_sub (fun t : ℕ =>
    a * Real.sin (θ * (x + t) - θ / 2) - b * Real.cos (θ * (x + t) - θ / 2))]
  simp only [Nat.cast_zero, add_zero]

/-- **A plane's partial sums are bounded** when `θ ∉ 2πℤ`, that is `sin(θ/2) ≠ 0`. -/
theorem abs_sum_range_le (θ a b x : ℝ) (hθ : Real.sin (θ / 2) ≠ 0) (T : ℕ) :
    |∑ t ∈ range T, (a * Real.cos (θ * (x + t)) + b * Real.sin (θ * (x + t)))| ≤
      (|a| + |b|) / |Real.sin (θ / 2)| := by
  have h := two_sin_mul_sum_range θ a b x T
  rw [le_div_iff₀ (abs_pos.2 hθ)]
  have bound : ∀ u y : ℝ, |u| ≤ 1 → -|y| ≤ y * u ∧ y * u ≤ |y| := fun u y hu =>
    abs_le.1 (abs_mul y u ▸ mul_le_of_le_one_right (abs_nonneg y) hu)
  generalize ∑ t ∈ range T, (a * Real.cos (θ * (x + t)) + b * Real.sin (θ * (x + t))) = S at h ⊢
  have e1 := bound _ a (Real.abs_sin_le_one (θ * (x + T) - θ / 2))
  have e2 := bound _ b (Real.abs_cos_le_one (θ * (x + T) - θ / 2))
  have e3 := bound _ a (Real.abs_sin_le_one (θ * x - θ / 2))
  have e4 := bound _ b (Real.abs_cos_le_one (θ * x - θ / 2))
  have key : |2 * Real.sin (θ / 2) * S| ≤ 2 * (|a| + |b|) := by
    rw [h, abs_le]
    constructor <;> linarith [e1.1, e1.2, e2.1, e2.2, e3.1, e3.2, e4.1, e4.2]
  rw [abs_mul, abs_mul, abs_two] at key
  linarith

example := abs_sum_range_le Real.pi 1 0 0 (by rw [Real.sin_pi_div_two]; norm_num) 10

/-- **The partial sums of a RoPE score are bounded** when no frequency is in `2πℤ`. -/
theorem neg_le_sum_ropeScore (θ a b : Fin K → ℝ) (hθ : ∀ c, Real.sin (θ c / 2) ≠ 0) (N T : ℕ) :
    -(∑ c, (|a c| + |b c|) / |Real.sin (θ c / 2)|) ≤
      ∑ t ∈ range T, ropeScore θ a b ((N + t : ℕ) : ℝ) := by
  simp only [ropeScore]
  rw [sum_comm, ← sum_neg_distrib]
  refine sum_le_sum fun c _ => ?_
  push_cast
  exact (abs_le.1 (abs_sum_range_le (θ c) (a c) (b c) N (hθ c) T)).1

example := neg_le_sum_ropeScore (fun _ : Fin 1 => Real.pi) 1 0
  (fun _ => by rw [Real.sin_pi_div_two]; norm_num) 0 10

/-- **Full RoPE comes back to `0`**: when no frequency is in `2πℤ`, the score is at least `0` at
distances as large as wanted, whatever the coefficients. -/
theorem exists_nonneg_ropeScore (θ a b : Fin K → ℝ) (hθ : ∀ c, Real.sin (θ c / 2) ≠ 0) (N : ℕ) :
    ∃ t : ℕ, N ≤ t ∧ 0 ≤ ropeScore θ a b t := by
  by_contra hcon
  push Not at hcon
  have hB : 0 ≤ ∑ c, (|a c| + |b c|) / |Real.sin (θ c / 2)| := sum_nonneg fun c _ => by positivity
  generalize hBdef : ∑ c, (|a c| + |b c|) / |Real.sin (θ c / 2)| = B at hB
  have hδ : 0 < -ropeScore θ a b N := by linarith [hcon N le_rfl]
  generalize hδdef : -ropeScore θ a b N = δ at hδ
  have hn := ropeNorm_nonneg a b
  obtain ⟨M, hM⟩ : ∃ M : ℕ, 2 * Real.pi / ((M : ℝ) + 1) * ropeNorm a b ≤ δ / 2 := by
    refine ⟨⌈4 * Real.pi * ropeNorm a b / δ⌉₊, ?_⟩
    have h1 := Nat.le_ceil (4 * Real.pi * ropeNorm a b / δ)
    rw [div_le_iff₀ hδ] at h1
    rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
    nlinarith [Real.pi_pos]
  obtain ⟨k, hk, hk0⟩ : ∃ k : ℕ, 2 * B / δ + 2 ≤ k ∧ 0 < k :=
    ⟨⌈2 * B / δ⌉₊ + 2, by push_cast; linarith [Nat.le_ceil (2 * B / δ)], by omega⟩
  have hP : (0 : ℝ) < ((M : ℝ) + 1) ^ K := by positivity
  obtain ⟨R, hcard, hR⟩ :=
    exists_returns θ M (H := k * (M + 1) ^ K) (Nat.mul_pos hk0 (by positivity))
  rw [show ((k * (M + 1) ^ K : ℕ) : ℝ) / ((M : ℝ) + 1) ^ K - 1 = k - 1 by
    push_cast; rw [mul_div_cancel_right₀ _ hP.ne']] at hcard
  have hret : ∀ d ∈ R, ropeScore θ a b ((N + d : ℕ) : ℝ) ≤ -(δ / 2) := fun d hd => by
    have h1 := abs_ropeScore_add_sub_le (a := a) (b := b) (hR d hd).2.2 N
    have h2 := (abs_le.1 (h1.trans hM)).2
    push_cast
    linarith
  have hsplit := sum_sdiff (f := fun t => ropeScore θ a b ((N + t : ℕ) : ℝ))
    (fun d hd => mem_range.2 (hR d hd).2.1 : R ⊆ range (k * (M + 1) ^ K))
  have hrest : ∑ t ∈ range (k * (M + 1) ^ K) \ R, ropeScore θ a b ((N + t : ℕ) : ℝ) ≤ 0 :=
    sum_nonpos fun t _ => (hcon (N + t) (by omega)).le
  have hRsum : ∑ t ∈ R, ropeScore θ a b ((N + t : ℕ) : ℝ) ≤ R.card * -(δ / 2) :=
    (sum_le_sum hret).trans (by rw [sum_const, nsmul_eq_mul])
  have hlow := neg_le_sum_ropeScore θ a b hθ N (k * (M + 1) ^ K)
  rw [hBdef] at hlow
  have h3 : (2 * B / δ + 1) * δ ≤ R.card * δ :=
    mul_le_mul_of_nonneg_right (by linarith) hδ.le
  rw [add_mul, div_mul_cancel₀ _ hδ.ne'] at h3
  linarith

example := exists_nonneg_ropeScore (fun _ : Fin 1 => 1) 1 0
  (fun _ => (Real.sin_pos_of_pos_of_lt_pi (by norm_num) (by linarith [Real.pi_gt_three])).ne') 7

/-- **A logit at least `0` gets a weight at least `1`** at the precision `p ≥ 2`, `s = 0`. -/
theorem one_le_val_round_exp {p : ℕ} (hp : 2 ≤ p) {z : ℝ} (hz : 0 ≤ z) :
    1 ≤ (Fx.round p 0 (Real.exp z)).val := by
  have hf : 1 ≤ ⌊Real.exp z⌋ := Int.le_floor.2 (by exact_mod_cast Real.one_le_exp hz)
  have hp' : (2 : ℤ) ≤ 2 ^ (p - 1) := by
    simpa using pow_le_pow_right₀ (a := (2 : ℤ)) (by norm_num) (show 1 ≤ p - 1 by omega)
  have hc : (1 : ℤ) ≤ Fx.clamp p ⌊Real.exp z⌋ := le_max_of_le_right (le_min (by linarith) hf)
  rw [Fx.val, Fx.m_round, pow_zero, mul_one, div_one]
  exact_mod_cast hc

example := one_le_val_round_exp (p := 2) le_rfl le_rfl

/-- **No exact previous-token head with full RoPE**: when no frequency is in `2πℤ`, a positional
head gives a weight at least `1` to keys at distances as large as wanted, at precision `p ≥ 2`. -/
theorem exists_one_le_weight_rope {p : ℕ} (hp : 2 ≤ p) (θ a b : Fin K → ℝ)
    (hθ : ∀ c, Real.sin (θ c / 2) ≠ 0) (N : ℕ) :
    ∃ t : ℕ, N ≤ t ∧ 1 ≤ (Fx.round p 0 (Real.exp (ropeScore θ a b t))).val := by
  obtain ⟨t, ht, h0⟩ := exists_nonneg_ropeScore θ a b hθ N
  exact ⟨t, ht, one_le_val_round_exp hp h0⟩

/-- A frequency in `(0, 2π)` is not in `2πℤ`. -/
theorem sin_half_ne_zero {θ : ℝ} (h0 : 0 < θ) (h1 : θ < 2 * Real.pi) : Real.sin (θ / 2) ≠ 0 :=
  (Real.sin_pos_of_pos_of_lt_pi (by linarith) (by linarith)).ne'

example := exists_one_le_weight_rope le_rfl (fun _ : Fin 1 => 1) 1 0
  (fun _ => sin_half_ne_zero one_pos (by linarith [Real.pi_gt_three])) 7

/-- RoPE's grid `base^{−2m/d}` for `base ≥ 1` (arXiv:2309.00071v3, §2.1) lies in `(0, 1]`. -/
theorem frequency_pos_le_one {base : ℝ} (hbase : 1 ≤ base) (dim m : ℕ) :
    0 < RoPE.frequency base dim m ∧ RoPE.frequency base dim m ≤ 1 := by
  unfold RoPE.frequency
  exact ⟨Real.rpow_pos_of_pos (by linarith) _, Real.rpow_le_one_of_one_le_of_nonpos hbase
    (div_nonpos_of_nonpos_of_nonneg (neg_nonpos.2 (by positivity)) (Nat.cast_nonneg _))⟩

example := frequency_pos_le_one (base := 10000) (by norm_num) 128 63

/-- **Nor with RoPE's grid**: keys arbitrarily far get a weight at least `1`. -/
theorem exists_one_le_weight_grid {p : ℕ} (hp : 2 ≤ p) {base : ℝ} (hbase : 1 ≤ base) (dim : ℕ)
    (a b : Fin K → ℝ) (N : ℕ) : ∃ t : ℕ, N ≤ t ∧
      1 ≤ (Fx.round p 0 (Real.exp (ropeScore (fun m => RoPE.frequency base dim m) a b t))).val :=
  exists_one_le_weight_rope hp _ a b (fun m => sin_half_ne_zero (frequency_pos_le_one hbase dim m).1
    ((frequency_pos_le_one hbase dim m).2.trans_lt (by linarith [Real.pi_gt_three]))) N

example := exists_one_le_weight_grid (K := 64) le_rfl (base := 10000) (by norm_num) 128 1 0 4096

/-- **Nor with YaRN** (arXiv:2309.00071v3, Definition 2): its frequencies of a grid in `(0, 1]`
stay in `(0, 1]`, and its temperature keeps the sign of the score. -/
theorem exists_one_le_weight_yarn {p : ℕ} (hp : 2 ≤ p) (a b omega : Fin K → ℝ)
    (homega : ∀ c, 0 < omega c ∧ omega c ≤ 1) (original target : ℝ) (N : ℕ) :
    ∃ t : ℕ, N ≤ t ∧
      1 ≤ (Fx.round p 0 (Real.exp (RoPE.YaRN.fullScore a b omega original target t))).val := by
  have hs := RoPE.YaRN.context_scale_ge_one original target
  obtain ⟨t, ht, h0⟩ := exists_nonneg_ropeScore (fun m => RoPE.YaRN.yarnFrequency (omega m)
    (RoPE.YaRN.contextScale original target) original) a b (fun m => sin_half_ne_zero
      (RoPE.YaRN.yarn_frequency_positive (homega m).1 hs original)
      (((RoPE.YaRN.yarn_frequency_bounds (homega m).1 hs original).2.trans (homega m).2).trans_lt
        (by linarith [Real.pi_gt_three]))) N
  refine ⟨t, ht, one_le_val_round_exp hp ?_⟩
  have hg := RoPE.YaRN.logit_gain_ge_one hs
  simp only [RoPE.YaRN.fullScore, RoPE.Diagnostics.headScore, RoPE.score]
  exact mul_nonneg (by linarith) h0

example := exists_one_le_weight_yarn le_rfl (fun _ : Fin 1 => 1) 0 (fun _ => 1)
  (fun _ => ⟨one_pos, le_rfl⟩) 4096 16384 7

end Transformer.MRCR
