module

public import Transformer.Grokking.FlatSpectrum.Recall.PowerBias

/-!
# The threshold of the bias `−α log(1 + r)`

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB6,
for the ALiBi bias of arXiv:2505.22842v4, §2.2, Lemma 2 (`Transformer.BAM.alibi`), and the scale
`s log n` of SSMax, arXiv:2501.19399v1, §2.

At the scale `s log n` with the bias `−α log(1 + r)`, `α ≥ 0`, the distance `n^ρ` has the
threshold `τ = α ρ + max(0, 1 − α)`: the keys within it are recalled when their raw scores lead
by `g` with `s g > τ` (`exists_forall_powerBias_recall`), and the keys beyond it are lost when
the raw scores are bounded by `B` with `2 s B < τ` (`exists_lt_powerBias_error`). Above `α = 1`
a far key weighs too little against the nearest token; below it the rivals of every key weigh
`n^{1−α}`. For `0 < ρ < 1`, `τ` exceeds the threshold `ρ` of `α = 1` (`LogBias`) by `(α − 1) ρ`
above `α = 1` and by `(1 − α)(1 − ρ)` below it, and every bias that decreases with the distance
loses the keys beyond `n^ρ` below `2 s B = ρ` (`NearerTokens`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **A power prior of polynomial mass recalls below its threshold** (`review/flat_spectrum.md`,
PB6): if the weights `(1 + r)^{−α}`, `α ≥ 0`, of the distances `r < n` sum to at most `K n^β` at
every length and `α ρ + β < s g`, `s ≥ 0`, then for every `ε > 0`, from some length on, under the
bias `−α log(1 + r)` every key at a distance `R` with `R + 1 ≤ n^ρ` whose raw score leads every
other by `g` is answered within `ε`, for values bounded by `C`. -/
theorem exists_forall_powerBias_recall_of_sum_le {s g ρ C ε α K β : ℝ} (hs : 0 ≤ s)
    (hα : 0 ≤ α) (hK : ∀ n : ℕ, ∑ r : Fin n, ((r : ℝ) + 1) ^ (-α) ≤ K * (n : ℝ) ^ β)
    (hρ : α * ρ + β < s * g) (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z v : Fin n → ℝ) (R : Fin n), (R : ℝ) + 1 ≤ (n : ℝ) ^ ρ →
      (∀ r, r ≠ R → z r + g ≤ z R) → (∀ r, |v r| ≤ C) →
        |v R - biasedLogScaleAttention s (fun r : Fin n => -(α * Real.log ((r : ℝ) + 1))) z v|
          ≤ ε := by
  set a := s * g - (α * ρ + β) with ha
  have hlim : Tendsto (fun n : ℕ => 2 * C * K * (n : ℝ) ^ (-a)) atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith : 0 < a)).comp
      tendsto_natCast_atTop_atTop).const_mul (2 * C * K)
    rw [mul_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hlim.eventually (ge_mem_nhds hε))
  refine ⟨N + 1, fun n hn z v R hR hz hv =>
    (abs_sub_powerBias_le hs hz hv).trans (le_trans ?_ (hN n (by omega)))⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv R)
  have hRα : ((R : ℝ) + 1) ^ α ≤ (n : ℝ) ^ (α * ρ) := by
    rw [show α * ρ = ρ * α by ring, Real.rpow_mul hn0.le]
    exact Real.rpow_le_rpow (by positivity) hR hα
  have e : (n : ℝ) ^ (α * ρ) * (n : ℝ) ^ (-(s * g)) * (n : ℝ) ^ β = (n : ℝ) ^ (-a) := by
    rw [← Real.rpow_add hn0, ← Real.rpow_add hn0]
    congr 1
    rw [ha]
    ring
  calc 2 * C * (((R : ℝ) + 1) ^ α * (n : ℝ) ^ (-(s * g)) * ∑ r : Fin n, ((r : ℝ) + 1) ^ (-α))
      ≤ 2 * C * ((n : ℝ) ^ (α * ρ) * (n : ℝ) ^ (-(s * g)) * (K * (n : ℝ) ^ β)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul (mul_le_mul_of_nonneg_right hRα (by positivity))
          (hK n) (sum_nonneg fun _ _ => by positivity) (by positivity)) (by linarith)
    _ = 2 * C * K * (n : ℝ) ^ (-a) := by
        rw [← e]
        ring

/-- At `α = 2`, whose weights sum to at most `2`, and `s = g = 1`: the keys within `n^{1/4}`. -/
example := exists_forall_powerBias_recall_of_sum_le (s := 1) (g := 1) (ρ := 1 / 4) (C := 1)
  (ε := 1 / 10) (α := 2) (K := 2) (β := 0) zero_le_one (by norm_num)
  (fun n => (sum_rpow_neg_le_of_one_lt (by norm_num) n).trans_eq (by norm_num)) (by norm_num)
  (by norm_num)

/-- **The recall threshold of the bias `−α log(1 + r)`** (`review/flat_spectrum.md`, PB6): at
the scale `s log n`, `s ≥ 0`, with the bias `−α log(1 + r)`, `α ≥ 0`, if
`α ρ + max(0, 1 − α) < s g` then for every `ε > 0`, from some length on, every key at a distance
`R` with `R + 1 ≤ n^ρ` whose raw score leads every other by `g` is answered within `ε`, for
values bounded by `C`. -/
theorem exists_forall_powerBias_recall {s g ρ C ε α : ℝ} (hs : 0 ≤ s) (hα : 0 ≤ α)
    (hρ : α * ρ + max 0 (1 - α) < s * g) (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z v : Fin n → ℝ) (R : Fin n), (R : ℝ) + 1 ≤ (n : ℝ) ^ ρ →
      (∀ r, r ≠ R → z r + g ≤ z R) → (∀ r, |v r| ≤ C) →
        |v R - biasedLogScaleAttention s (fun r : Fin n => -(α * Real.log ((r : ℝ) + 1))) z v|
          ≤ ε := by
  rcases lt_trichotomy α 1 with h1 | rfl | h1
  · rw [max_eq_right (by linarith : (0 : ℝ) ≤ 1 - α)] at hρ
    exact exists_forall_powerBias_recall_of_sum_le hs hα (K := 1 / (1 - α)) (β := 1 - α)
      (fun n => (sum_rpow_neg_le_of_lt_one hα h1 n).trans_eq (by ring)) hρ hε
  · simp only [one_mul]
    exact exists_forall_logBias_recall hs (by simpa using hρ) hε
  · rw [max_eq_left (by linarith : 1 - α ≤ 0)] at hρ
    exact exists_forall_powerBias_recall_of_sum_le hs hα (K := α / (α - 1)) (β := 0)
      (fun n => (sum_rpow_neg_le_of_one_lt h1 n).trans_eq (by rw [Real.rpow_zero, mul_one]))
      hρ hε

/-- At `α = 1/2` and `s = g = 1`: the keys within `√n`, below the threshold `3/4`. -/
example := exists_forall_powerBias_recall (s := 1) (g := 1) (ρ := 1 / 2) (C := 1) (ε := 1 / 10)
  (α := 1 / 2) zero_le_one (by norm_num) (by norm_num [max_def]) (by norm_num)

/-- **A power prior whose rivals weigh `n^γ` loses the far keys** (`review/flat_spectrum.md`,
PB6): under the bias `−α log(1 + r)`, `α ≥ 0`, if at every length `n ≥ 2` the distances other
than any `R ≥ 1` weigh at least `n^γ / 2` in the prior `(1 + r)^{−α}`, then for raw scores
bounded by `B` with `2 s B < α ρ + γ`, `s ≥ 0`, for every `ε < 2`, from some length on, the key
at every distance `R ≥ n^ρ` has its indicator dictionary answered with an error above `ε`. -/
theorem exists_lt_powerBias_error_of_le_sum {s B ρ ε α γ : ℝ} (hs : 0 ≤ s) (hα : 0 ≤ α)
    (hW : ∀ n : ℕ, 2 ≤ n → ∀ R : Fin n, (R : ℕ) ≠ 0 →
      (n : ℝ) ^ γ ≤ 2 * ∑ r ∈ univ.erase R, ((r : ℝ) + 1) ^ (-α))
    (hρ : 2 * s * B < α * ρ + γ) (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ ≤ R → (∀ r, |z r| ≤ B) →
      ε < 1 - biasedLogScaleAttention s (fun r : Fin n => -(α * Real.log ((r : ℝ) + 1))) z
        (fun r => if r = R then 1 else -1) := by
  set a := α * ρ + γ - 2 * s * B with ha
  have hlim : Tendsto (fun n : ℕ => 2 * (n : ℝ) ^ (-a)) atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith : 0 < a)).comp
      tendsto_natCast_atTop_atTop).const_mul 2
    rw [mul_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1
    (hlim.eventually (gt_mem_nhds (by linarith : (0 : ℝ) < (2 - ε) / 2)))
  refine ⟨N + 2, fun n hn z R hR hz => ?_⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hR0 : (0 : ℝ) < R := (Real.rpow_pos_of_pos hn0 ρ).trans_le hR
  have hR1 : (R : ℕ) ≠ 0 := fun h => by simp [h] at hR0
  have h := le_one_sub_biasedLogScaleAttention_of_weight
    (b := fun r : Fin n => -(α * Real.log ((r : ℝ) + 1))) hs hz (notMem_erase R univ)
  simp only [Fintype.card_fin] at h
  refine lt_of_lt_of_le ?_ h
  have hexp : ∀ r : Fin n, Real.exp (-(α * Real.log ((r : ℝ) + 1))) = ((r : ℝ) + 1) ^ (-α) := by
    intro r
    rw [Real.rpow_def_of_pos (by positivity)]
    ring_nf
  simp only [hexp]
  have hWn := hW n (by omega) R hR1
  set W := ∑ r ∈ univ.erase R, ((r : ℝ) + 1) ^ (-α) with hWdef
  have hWpos : 0 < W := by linarith [Real.rpow_pos_of_pos hn0 γ]
  have hK : (n : ℝ) ^ (2 * s * B) * ((R : ℝ) + 1) ^ (-α) ≤ 2 * (n : ℝ) ^ (-a) * W := by
    have h1 : ((R : ℝ) + 1) ^ (-α) ≤ (n : ℝ) ^ (ρ * -α) := by
      rw [Real.rpow_mul hn0.le]
      exact Real.rpow_le_rpow_of_nonpos (Real.rpow_pos_of_pos hn0 ρ) (by linarith) (by linarith)
    have e : (n : ℝ) ^ (2 * s * B) * (n : ℝ) ^ (ρ * -α) = (n : ℝ) ^ (-a) * (n : ℝ) ^ γ := by
      rw [← Real.rpow_add hn0, ← Real.rpow_add hn0]
      congr 1
      rw [ha]
      ring
    calc (n : ℝ) ^ (2 * s * B) * ((R : ℝ) + 1) ^ (-α)
        ≤ (n : ℝ) ^ (2 * s * B) * (n : ℝ) ^ (ρ * -α) :=
          mul_le_mul_of_nonneg_left h1 (by positivity)
      _ = (n : ℝ) ^ (-a) * (n : ℝ) ^ γ := e
      _ ≤ (n : ℝ) ^ (-a) * (2 * W) := mul_le_mul_of_nonneg_left hWn (by positivity)
      _ = 2 * (n : ℝ) ^ (-a) * W := by ring
  have hx := hN n (by omega)
  have hK0 : 0 ≤ (n : ℝ) ^ (2 * s * B) * ((R : ℝ) + 1) ^ (-α) := by positivity
  generalize 2 * (n : ℝ) ^ (-a) = x at hK hx
  generalize (n : ℝ) ^ (2 * s * B) * ((R : ℝ) + 1) ^ (-α) = K at hK hK0 ⊢
  rw [lt_div_iff₀ (by linarith)]
  rcases le_or_gt ε 0 with h0 | h0
  · nlinarith
  · nlinarith [mul_le_mul_of_nonneg_left hK h0.le,
      mul_lt_mul_of_pos_left (mul_lt_mul_of_pos_right hx hWpos) h0,
      mul_nonneg (sq_nonneg (ε - 2)) hWpos.le]

/-- At `α = 1`, `s = 1` and `B = 1/8`: the keys beyond `√n`, the nearest token weighing `1`. -/
example := exists_lt_powerBias_error_of_le_sum (s := 1) (B := 1 / 8) (ρ := 1 / 2) (ε := 1)
  (α := 1) (γ := 0) zero_le_one zero_le_one
  (fun n _ R hR => by rw [Real.rpow_zero]; linarith [one_le_sum_erase_rpow_neg 1 hR])
  (by norm_num) (by norm_num)

/-- **The fading threshold of the bias `−α log(1 + r)`** (`review/flat_spectrum.md`, PB6): at
the scale `s log n`, `s ≥ 0`, with the bias `−α log(1 + r)`, `α ≥ 0`, and raw scores bounded by
`B` with `2 s B < α ρ + max(0, 1 − α)`, for every `ε < 2`, from some length on, the key at every
distance `R ≥ n^ρ` has its indicator dictionary answered with an error above `ε`. -/
theorem exists_lt_powerBias_error {s B ρ ε α : ℝ} (hs : 0 ≤ s) (hα : 0 ≤ α)
    (hρ : 2 * s * B < α * ρ + max 0 (1 - α)) (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ ≤ R → (∀ r, |z r| ≤ B) →
      ε < 1 - biasedLogScaleAttention s (fun r : Fin n => -(α * Real.log ((r : ℝ) + 1))) z
        (fun r => if r = R then 1 else -1) := by
  rcases le_or_gt 1 α with h1 | h1
  · rw [max_eq_left (by linarith : 1 - α ≤ 0)] at hρ
    refine exists_lt_powerBias_error_of_le_sum hs hα (fun n _ R hR => ?_) hρ hε
    rw [Real.rpow_zero]
    linarith [one_le_sum_erase_rpow_neg α hR]
  · rw [max_eq_right (by linarith : (0 : ℝ) ≤ 1 - α)] at hρ
    exact exists_lt_powerBias_error_of_le_sum hs hα
      (fun n hn R _ => rpow_one_sub_le_two_mul_sum_erase hα hn R) hρ hε

/-- At `α = 2`, `s = 1` and `B = 1/8`: the keys beyond `√n`, below the threshold `1`. -/
example := exists_lt_powerBias_error (s := 1) (B := 1 / 8) (ρ := 1 / 2) (ε := 1) (α := 2)
  zero_le_one (by norm_num) (by norm_num [max_def]) (by norm_num)

/-- At `α = 1/2`, `s = 1` and `B = 1/8`: every key but the nearest, below the threshold `1/2`. -/
example := exists_lt_powerBias_error (s := 1) (B := 1 / 8) (ρ := 0) (ε := 1) (α := 1 / 2)
  zero_le_one (by norm_num) (by norm_num [max_def]) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
