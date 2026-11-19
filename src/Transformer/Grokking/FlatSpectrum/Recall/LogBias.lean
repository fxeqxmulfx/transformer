module

public import Mathlib.NumberTheory.Harmonic.Bounds
public import Transformer.Grokking.FlatSpectrum.Recall.FittedSlope

/-!
# The bias `−log(1 + r)` reaches the threshold at every distance at once

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB5,
for the ALiBi bias of arXiv:2505.22842v4, §2.2, Lemma 2 (`Transformer.BAM.alibi`), and the scale
`s log n` of SSMax, arXiv:2501.19399v1, §2.

A bias `b` weighs the distances against a key at the distance `R` by
`W_R = ∑_r e^{b_r − b_R}`, and `1 / W_R` is the prior `e^{b_R} / ∑_r e^{b_r}` of `R`, so the
`1 / W_R` sum to `1`. Hence every bias has a distance `R` with `W_R ≥ (R + 1) H_n`, where
`H_n = ∑_{r<n} 1/(r + 1)` (`exists_mul_harmonic_le_sum_exp`), while the bias `−log(1 + r)`, the
prior `1/(r + 1)`, has `W_R = (R + 1) H_n` at every distance (`sum_exp_logBias`): no bias has a
smaller `max_R W_R / (R + 1)`. With `H_n ≤ 1 + log n` it answers every key at a distance `R` with
a raw lead `g` within `2 C (R + 1) (1 + log n) n^{−sg}` (`abs_sub_logBias_le`). So this one bias
recalls every key at a distance `R` with `R + 1 ≤ n^ρ` when `s g > ρ`, for every `ρ` at once
(`exists_forall_logBias_recall`), and loses the keys beyond `n^ρ` when `2 s B < ρ`, as every
decreasing bias does (`exists_lt_logBias_error`, `Recall.NearerTokens`). For one-hot raw scores
`±B` it has the threshold `2 s B = ρ` at every distance `n^ρ`, which a fitted ALiBi slope has
only on its span (`Recall.FittedSlope`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **Every bias weighs some distance by the harmonic number** (`review/flat_spectrum.md`, PB5):
for every bias `b` of `n ≥ 1` distances there is a distance `R` against which the distances weigh
`∑_r e^{b_r − b_R} ≥ (R + 1) H_n`, where `H_n = ∑_{r<n} 1/(r + 1)`. -/
theorem exists_mul_harmonic_le_sum_exp {n : ℕ} (hn : 0 < n) (b : Fin n → ℝ) :
    ∃ R : Fin n, ((R : ℝ) + 1) * ∑ r : Fin n, 1 / ((r : ℝ) + 1) ≤
      ∑ r : Fin n, Real.exp (b r - b R) := by
  by_contra h
  push Not at h
  set H := ∑ r : Fin n, 1 / ((r : ℝ) + 1) with hH
  set Z := ∑ r : Fin n, Real.exp (b r) with hZ
  have h1 : ∀ R : Fin n, Z * (1 / ((R : ℝ) + 1)) < H * Real.exp (b R) := by
    intro R
    have hR := h R
    have hs : ∑ r : Fin n, Real.exp (b r - b R) = Z * Real.exp (-b R) := by
      rw [hZ, sum_mul]
      exact sum_congr rfl fun r _ => by rw [sub_eq_add_neg, Real.exp_add]
    rw [hs] at hR
    have e1 : Real.exp (-b R) * Real.exp (b R) = 1 := by
      rw [← Real.exp_add]
      simp
    rw [mul_one_div, div_lt_iff₀ (by positivity)]
    calc Z = Z * Real.exp (-b R) * Real.exp (b R) := by rw [mul_assoc, e1, mul_one]
      _ < ((R : ℝ) + 1) * H * Real.exp (b R) := mul_lt_mul_of_pos_right hR (Real.exp_pos _)
      _ = H * Real.exp (b R) * ((R : ℝ) + 1) := by ring
  have h2 := sum_lt_sum_of_nonempty ⟨⟨0, hn⟩, mem_univ _⟩ fun R _ => h1 R
  rw [← mul_sum, ← mul_sum, ← hH, ← hZ] at h2
  linarith [mul_comm Z H]

/-- The ALiBi bias of the slope `1` over three distances. -/
example := exists_mul_harmonic_le_sum_exp (n := 3) (by norm_num) fun r => -(r : ℝ)

/-- **The bias `−log(1 + r)` weighs every distance by the harmonic number**
(`review/flat_spectrum.md`, PB5): against the key at every distance `R`, the distances weigh
`∑_r e^{b_r − b_R} = (R + 1) H_n`. -/
theorem sum_exp_logBias {n : ℕ} (R : Fin n) :
    ∑ r : Fin n, Real.exp (-Real.log ((r : ℝ) + 1) - -Real.log ((R : ℝ) + 1)) =
      ((R : ℝ) + 1) * ∑ r : Fin n, 1 / ((r : ℝ) + 1) := by
  rw [mul_sum]
  refine sum_congr rfl fun r _ => ?_
  rw [neg_sub_neg, Real.exp_sub, Real.exp_log (by positivity), Real.exp_log (by positivity)]
  ring

/-- The harmonic number `H_n = ∑_{r<n} 1/(r + 1)` is at most `1 + log n`. -/
lemma sum_one_div_succ_le (n : ℕ) : ∑ r : Fin n, 1 / ((r : ℝ) + 1) ≤ 1 + Real.log n := by
  refine le_trans (le_of_eq ?_) (harmonic_le_one_add_log n)
  rw [Fin.sum_univ_eq_sum_range (fun r : ℕ => 1 / ((r : ℝ) + 1)) n]
  simp [harmonic]

/-- **The bias `−log(1 + r)` at the scale `s log n`** (`review/flat_spectrum.md`, PB5): over `n`
tokens at the distances `r` with the bias `−log(1 + r)`, if the raw score of the token at the
distance `R` leads every other by `g` and the values are bounded by `C`, attention at the scale
`s log n`, `s ≥ 0`, answers its value within `2 C (R + 1) (1 + log n) n^{−sg}`. -/
theorem abs_sub_logBias_le {s g C : ℝ} (hs : 0 ≤ s) {n : ℕ} {z v : Fin n → ℝ} {R : Fin n}
    (hz : ∀ r, r ≠ R → z r + g ≤ z R) (hv : ∀ r, |v r| ≤ C) :
    |v R - biasedLogScaleAttention s (fun r : Fin n => -Real.log ((r : ℝ) + 1)) z v| ≤
      2 * C * (((R : ℝ) + 1) * (1 + Real.log n) * (n : ℝ) ^ (-(s * g))) := by
  have h := abs_sub_biasedLogScaleAttention_le hs (fun r : Fin n => -Real.log ((r : ℝ) + 1)) hz hv
  rw [Fintype.card_fin] at h
  refine h.trans (mul_le_mul_of_nonneg_left ?_ (by linarith [(abs_nonneg _).trans (hv R)]))
  have hsum : ∑ r ∈ univ.erase R, Real.exp (-Real.log ((r : ℝ) + 1) - -Real.log ((R : ℝ) + 1)) ≤
      ((R : ℝ) + 1) * (1 + Real.log n) := by
    calc _ ≤ ∑ r : Fin n, Real.exp (-Real.log ((r : ℝ) + 1) - -Real.log ((R : ℝ) + 1)) :=
          sum_le_sum_of_subset_of_nonneg (erase_subset _ _) fun _ _ _ => (Real.exp_pos _).le
      _ = ((R : ℝ) + 1) * ∑ r : Fin n, 1 / ((r : ℝ) + 1) := sum_exp_logBias R
      _ ≤ ((R : ℝ) + 1) * (1 + Real.log n) :=
          mul_le_mul_of_nonneg_left (sum_one_div_succ_le n) (by positivity)
  rw [mul_comm ((n : ℝ) ^ (-(s * g)))]
  exact mul_le_mul_of_nonneg_right hsum (by positivity)

/-- Three tokens, the query's own token leading by `1`. -/
example := abs_sub_logBias_le (s := 1) (g := 1) (C := 1) zero_le_one (n := 3)
  (z := ![1, 0, 0]) (v := ![1, -1, -1]) (R := 0) (fun r hr => by fin_cases r <;> simp_all)
  fun r => by fin_cases r <;> simp

/-- **One bias recalls at every distance scale** (`review/flat_spectrum.md`, PB5): with the bias
`−log(1 + r)` at the scale `s log n`, `s ≥ 0`, if `ρ < s g` then for every `ε > 0`, from some
length on, every key at a distance `R` with `R + 1 ≤ n^ρ` whose raw score leads every other by
`g` is answered within `ε`, for values bounded by `C`. -/
theorem exists_forall_logBias_recall {s g ρ C ε : ℝ} (hs : 0 ≤ s) (hρ : ρ < s * g)
    (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z v : Fin n → ℝ) (R : Fin n), (R : ℝ) + 1 ≤ (n : ℝ) ^ ρ →
      (∀ r, r ≠ R → z r + g ≤ z R) → (∀ r, |v r| ≤ C) →
        |v R - biasedLogScaleAttention s (fun r : Fin n => -Real.log ((r : ℝ) + 1)) z v| ≤ ε := by
  set a := s * g - ρ with ha
  have ha0 : 0 < a := by linarith
  have hlim : Tendsto (fun n : ℕ => 2 * C * (n : ℝ) ^ (-a) +
      2 * C / (a / 2) * (n : ℝ) ^ (-(a / 2))) atTop (𝓝 0) := by
    have h1 := ((tendsto_rpow_neg_atTop ha0).comp tendsto_natCast_atTop_atTop).const_mul (2 * C)
    have h2 := ((tendsto_rpow_neg_atTop (half_pos ha0)).comp
      tendsto_natCast_atTop_atTop).const_mul (2 * C / (a / 2))
    have h := h1.add h2
    rw [mul_zero, mul_zero, add_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hlim.eventually (ge_mem_nhds hε))
  refine ⟨N + 1, fun n hn z v R hR hz hv =>
    (abs_sub_logBias_le hs hz hv).trans (le_trans ?_ (hN n (by omega)))⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv R)
  have hlog : Real.log n ≤ (n : ℝ) ^ (a / 2) / (a / 2) :=
    Real.log_le_rpow_div hn0.le (half_pos ha0)
  have hlog0 : 0 ≤ 1 + Real.log n := by
    linarith [Real.log_nonneg (by exact_mod_cast (by omega : 1 ≤ n) : (1 : ℝ) ≤ n)]
  have e1 : (n : ℝ) ^ ρ * (n : ℝ) ^ (-(s * g)) = (n : ℝ) ^ (-a) := by
    rw [← Real.rpow_add hn0]
    congr 1
    rw [ha]
    ring
  have e2 : (n : ℝ) ^ ρ * (n : ℝ) ^ (a / 2) * (n : ℝ) ^ (-(s * g)) = (n : ℝ) ^ (-(a / 2)) := by
    rw [← Real.rpow_add hn0, ← Real.rpow_add hn0]
    congr 1
    rw [ha]
    ring
  calc 2 * C * (((R : ℝ) + 1) * (1 + Real.log n) * (n : ℝ) ^ (-(s * g)))
      ≤ 2 * C * ((n : ℝ) ^ ρ * (1 + (n : ℝ) ^ (a / 2) / (a / 2)) * (n : ℝ) ^ (-(s * g))) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right
          (mul_le_mul hR (by linarith) hlog0 (by positivity)) (by positivity)) (by linarith)
    _ = 2 * C * (n : ℝ) ^ (-a) + 2 * C / (a / 2) * (n : ℝ) ^ (-(a / 2)) := by
        rw [← e1, ← e2]
        ring

/-- At `s = g = 1`, the keys within `√n`. -/
example := exists_forall_logBias_recall (s := 1) (g := 1) (ρ := 1 / 2) (C := 1) (ε := 1 / 10)
  zero_le_one (by norm_num) (by norm_num)

/-- **The bias `−log(1 + r)` loses the keys beyond `n^ρ` below `2 s B = ρ`**
(`review/flat_spectrum.md`, PB5): with raw scores bounded by `B` and `2 s B < ρ`, `s ≥ 0`, for
every `ε < 2`, from some length on, the key at every distance `R ≥ n^ρ` has its indicator
dictionary answered with an error above `ε`. -/
theorem exists_lt_logBias_error {s B ρ ε : ℝ} (hs : 0 ≤ s) (hρ : 2 * s * B < ρ) (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ ≤ R → (∀ r, |z r| ≤ B) →
      ε < 1 - biasedLogScaleAttention s (fun r : Fin n => -Real.log ((r : ℝ) + 1)) z
        (fun r => if r = R then 1 else -1) := by
  obtain ⟨N, hN⟩ := exists_lt_error_of_nearer hs hρ hε
  refine ⟨N, fun n hn z R hR hz => hN n hn _ z R hR (fun r hr => ?_) hz⟩
  have : (r : ℝ) ≤ R := by exact_mod_cast hr.le
  exact neg_le_neg (Real.log_le_log (by positivity) (by linarith))

/-- At `s = 1` and `B = 1/8`, the keys beyond `√n`. -/
example := exists_lt_logBias_error (s := 1) (B := 1 / 8) (ρ := 1 / 2) (ε := 1) zero_le_one
  (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
