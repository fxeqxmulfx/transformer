module

public import Transformer.Grokking.FlatSpectrum.Recall.NearerTokens

/-!
# The ALiBi slope fitted to a span reaches the threshold `2 s B = ρ`

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB4,
for the ALiBi bias of arXiv:2505.22842v4, §2.2, Lemma 2 (`Transformer.BAM.alibi`), and the scale
`s log n` of SSMax, arXiv:2501.19399v1, §2.

The slope `m = log(1 + 1/D)` is the one at which the geometric prior `(1 − e^{−m}) e^{−mr}` of
ALiBi has the mean `D`. Against a key at a distance `R ≤ D` the other distances then weigh at most
`e^{mR} / (1 − e^{−m}) ≤ e (D + 1)`, so every such key with a raw lead `g` is answered within
`2 C e (D + 1) n^{−sg}` (`abs_sub_fittedSlope_le`). So the slope fitted to a span
`D ≤ n^ρ` recalls every key of the span when `s g > ρ` (`exists_forall_fittedSlope_recall`):
for one-hot raw scores `±B` this is the threshold `2 s B = ρ` below which no decreasing bias
recalls a key at the distance `n^ρ` (`Recall.NearerTokens`). Beyond the span the slope is too
steep: the prior falls by `e^{−R/(D+1)}` at least, and the key misses by at least
`2 / (1 + n^{2sB} e^{−R/(D+1)})` (`le_one_sub_fittedSlope`), so the keys beyond
`κ (D + 1) log n` with `2 s B < κ` are lost (`exists_lt_fittedSlope_error`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **Inside the span** (`review/flat_spectrum.md`, PB4): over `n` tokens at the distances `r`
with the ALiBi bias of the slope `log(1 + 1/D)`, `D ≥ 1`, if the raw score of the token at a
distance `R ≤ D` leads every other by `g` and the values are bounded by `C`, attention at the
scale `s log n`, `s ≥ 0`, answers its value within `2 C e (D + 1) n^{−sg}`. -/
theorem abs_sub_fittedSlope_le {s g C : ℝ} (hs : 0 ≤ s) {n D : ℕ} (hD : 1 ≤ D)
    {z v : Fin n → ℝ} {R : Fin n} (hR : (R : ℕ) ≤ D) (hz : ∀ r, r ≠ R → z r + g ≤ z R)
    (hv : ∀ r, |v r| ≤ C) :
    |v R - biasedLogScaleAttention s (fun r : Fin n => -(Real.log (1 + 1 / D) * r)) z v| ≤
      2 * C * Real.exp 1 * (D + 1) * (n : ℝ) ^ (-(s * g)) := by
  have hD0 : (0 : ℝ) < D := by exact_mod_cast hD
  have hm : 0 < Real.log (1 + 1 / D) := Real.log_pos (by linarith [one_div_pos.2 hD0])
  refine (abs_sub_alibi_le hs hm hz hv).trans ?_
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv R)
  have hD1 : (D : ℝ) + 1 ≠ 0 := by positivity
  have hq : 1 - Real.exp (-Real.log (1 + 1 / D)) = 1 / (D + 1) := by
    rw [Real.exp_neg, Real.exp_log (by positivity)]
    field_simp
    ring
  have hmR : Real.log (1 + 1 / D) * R ≤ 1 := by
    have h1 : Real.log (1 + 1 / D) ≤ 1 / D := by
      linarith [Real.log_le_sub_one_of_pos (by positivity : (0 : ℝ) < 1 + 1 / D)]
    have hRD : (R : ℝ) ≤ D := by exact_mod_cast hR
    calc Real.log (1 + 1 / D) * R ≤ 1 / D * D :=
          mul_le_mul h1 hRD (by positivity) (by positivity)
      _ = 1 := by field_simp
  rw [hq, div_div_eq_mul_div, div_one]
  calc 2 * C * ((n : ℝ) ^ (-(s * g)) * Real.exp (Real.log (1 + 1 / D) * R) * (D + 1))
      = 2 * C * (D + 1) * (n : ℝ) ^ (-(s * g)) * Real.exp (Real.log (1 + 1 / D) * R) := by ring
    _ ≤ 2 * C * (D + 1) * (n : ℝ) ^ (-(s * g)) * Real.exp 1 :=
        mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 hmR) (by positivity)
    _ = 2 * C * Real.exp 1 * (D + 1) * (n : ℝ) ^ (-(s * g)) := by ring

/-- Three tokens, the query's own token leading by `1`, at the slope fitted to the span `2`. -/
example := abs_sub_fittedSlope_le (s := 1) (g := 1) (C := 1) zero_le_one (n := 3) (D := 2)
  (by norm_num) (z := ![1, 0, 0]) (v := ![1, -1, -1]) (R := 0) (by simp)
  (fun r hr => by fin_cases r <;> simp_all) fun r => by fin_cases r <;> simp

/-- **The fitted slope recalls the span** (`review/flat_spectrum.md`, PB4): at the scale
`s log n`, `s ≥ 0`, if `ρ < s g` then for every `ε > 0`, from some length on, at the slope
`log(1 + 1/D)` fitted to any span `1 ≤ D ≤ n^ρ`, every key within the span whose raw score leads
every other by `g` is answered within `ε`, for values bounded by `C`. -/
theorem exists_forall_fittedSlope_recall {s g ρ C ε : ℝ} (hs : 0 ≤ s) (hρ : ρ < s * g)
    (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ D : ℕ, 1 ≤ D → (D : ℝ) ≤ (n : ℝ) ^ ρ →
      ∀ (z v : Fin n → ℝ) (R : Fin n), (R : ℕ) ≤ D → (∀ r, r ≠ R → z r + g ≤ z R) →
        (∀ r, |v r| ≤ C) →
          |v R - biasedLogScaleAttention s (fun r : Fin n => -(Real.log (1 + 1 / D) * r)) z v| ≤
            ε := by
  have hlim : Tendsto (fun n : ℕ => 4 * C * Real.exp 1 * (n : ℝ) ^ (-(s * g - ρ))) atTop
      (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (y := s * g - ρ) (by linarith)).comp
      tendsto_natCast_atTop_atTop).const_mul (4 * C * Real.exp 1)
    rw [mul_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hlim.eventually (ge_mem_nhds hε))
  refine ⟨N + 1, fun n hn D hD hDn z v R hR hz hv =>
    (abs_sub_fittedSlope_le hs hD hR hz hv).trans (le_trans ?_ (hN n (by omega)))⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv R)
  have hD1 : (1 : ℝ) ≤ D := by exact_mod_cast hD
  have hsplit : (n : ℝ) ^ (-(s * g - ρ)) = (n : ℝ) ^ (-(s * g)) * (n : ℝ) ^ ρ := by
    rw [← Real.rpow_add hn0]
    ring_nf
  have hc : 0 ≤ 2 * C * Real.exp 1 := by positivity
  calc 2 * C * Real.exp 1 * (D + 1) * (n : ℝ) ^ (-(s * g))
      ≤ 2 * C * Real.exp 1 * (2 * (n : ℝ) ^ ρ) * (n : ℝ) ^ (-(s * g)) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (by linarith) hc) (by positivity)
    _ = 4 * C * Real.exp 1 * (n : ℝ) ^ (-(s * g - ρ)) := by
        rw [hsplit]
        ring

/-- At `s = g = 1`, the spans up to `√n`. -/
example := exists_forall_fittedSlope_recall (s := 1) (g := 1) (ρ := 1 / 2) (C := 1)
  (ε := 1 / 10) zero_le_one (by norm_num) (by norm_num)

/-- **Beyond the span** (`review/flat_spectrum.md`, PB4): over `n` tokens at the distances `r`
with the ALiBi bias of the slope `log(1 + 1/D)`, `D ≥ 1`, and raw scores bounded by `B`, attention
at the scale `s log n`, `s ≥ 0`, answers the indicator dictionary of a key at a distance `R ≥ 1`
with an error of at least `2 / (1 + n^{2sB} e^{−R/(D+1)})`. -/
theorem le_one_sub_fittedSlope {s B : ℝ} (hs : 0 ≤ s) {n D : ℕ} (hD : 1 ≤ D) {z : Fin n → ℝ}
    (hz : ∀ r, |z r| ≤ B) {R : Fin n} (hR : (R : ℕ) ≠ 0) :
    2 / (1 + (n : ℝ) ^ (2 * s * B) * Real.exp (-(R / (D + 1)))) ≤
      1 - biasedLogScaleAttention s (fun r : Fin n => -(Real.log (1 + 1 / D) * r)) z
        (fun r => if r = R then 1 else -1) := by
  refine le_trans ?_ (le_one_sub_alibi hs _ hz hR)
  have hD0 : (0 : ℝ) < D + 1 := by positivity
  have hm : 1 / ((D : ℝ) + 1) ≤ Real.log (1 + 1 / D) := by
    have hD1 : (0 : ℝ) < D := by exact_mod_cast hD
    refine le_trans (le_of_eq ?_) (Real.one_sub_inv_le_log_of_pos (by positivity))
    field_simp
    ring
  have hx : Real.exp (-(Real.log (1 + 1 / D) * R)) ≤ Real.exp (-(R / (D + 1))) := by
    refine Real.exp_le_exp.2 (neg_le_neg ?_)
    rw [div_eq_mul_one_div, mul_comm]
    exact mul_le_mul_of_nonneg_right hm (by positivity)
  have hK := Real.rpow_nonneg (Nat.cast_nonneg n) (2 * s * B)
  have h0 := Real.exp_pos (-(Real.log (1 + 1 / D) * R))
  exact div_le_div_of_nonneg_left (by norm_num) (by positivity)
    (by nlinarith [mul_le_mul_of_nonneg_left hx hK])

/-- Three tokens at the slope fitted to the span `1`, the key at the distance `2`. -/
example := le_one_sub_fittedSlope (s := 1) (B := 1) zero_le_one (D := 1) le_rfl
  (z := ![1, -1, 0]) (fun r => by fin_cases r <;> simp) (R := 2) (by decide)

/-- **The fitted slope loses the keys beyond its span** (`review/flat_spectrum.md`, PB4): with
raw scores bounded by `B` and `2 s B < κ`, `s ≥ 0`, for every `ε < 2`, from some length on, at
the slope `log(1 + 1/D)` fitted to any span `D ≥ 1`, every key beyond the distance
`κ (D + 1) log n` has its indicator dictionary answered with an error above `ε`. -/
theorem exists_lt_fittedSlope_error {s B κ ε : ℝ} (hs : 0 ≤ s) (hκ : 2 * s * B < κ)
    (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ D : ℕ, 1 ≤ D → ∀ (z : Fin n → ℝ) (R : Fin n),
      κ * (D + 1) * Real.log n ≤ R → (∀ r, |z r| ≤ B) →
        ε < 1 - biasedLogScaleAttention s (fun r : Fin n => -(Real.log (1 + 1 / D) * r)) z
          (fun r => if r = R then 1 else -1) := by
  have hlim : Tendsto (fun n : ℕ => (n : ℝ) ^ (-(κ - 2 * s * B))) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop (by linarith)).comp tendsto_natCast_atTop_atTop
  obtain ⟨N, hN⟩ := eventually_atTop.1
    (hlim.eventually (gt_mem_nhds (by linarith : (0 : ℝ) < (2 - ε) / 2)))
  refine ⟨N + 2, fun n hn D hD z R hR hz => ?_⟩
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast (by omega : 2 ≤ n)
  have hlog : 0 < Real.log n := Real.log_pos (by linarith)
  have hD0 : (0 : ℝ) < D + 1 := by positivity
  have hB : 0 ≤ B := (abs_nonneg _).trans (hz R)
  have hκ0 : 0 < κ := by nlinarith [mul_nonneg hs hB]
  have hR0 : (R : ℕ) ≠ 0 := fun h0 => by
    rw [h0, Nat.cast_zero] at hR
    nlinarith [mul_pos (mul_pos hκ0 hD0) hlog]
  refine lt_of_lt_of_le ?_ (le_one_sub_fittedSlope hs hD hz hR0)
  have hx : (n : ℝ) ^ (2 * s * B) * Real.exp (-(R / (D + 1))) ≤
      (n : ℝ) ^ (-(κ - 2 * s * B)) := by
    rw [Real.rpow_def_of_pos (by linarith), Real.rpow_def_of_pos (by linarith), ← Real.exp_add]
    refine Real.exp_le_exp.2 ?_
    have : κ * Real.log n ≤ R / (D + 1) := by
      rw [le_div_iff₀ hD0]
      linarith
    nlinarith
  have hX := hN n (by omega)
  have hx0 : 0 ≤ (n : ℝ) ^ (2 * s * B) * Real.exp (-(R / (D + 1))) := by positivity
  generalize (n : ℝ) ^ (2 * s * B) * Real.exp (-(R / (D + 1))) = x at hx hx0 ⊢
  rw [lt_div_iff₀ (by linarith)]
  rcases le_or_gt ε 0 with h0 | h0
  · nlinarith
  · nlinarith [mul_lt_mul_of_pos_left (hx.trans_lt hX) h0, sq_nonneg (ε - 2)]

/-- At `s = 1` and `B = 1/4`, the keys beyond `(D + 1) log n`. -/
example := exists_lt_fittedSlope_error (s := 1) (B := 1 / 4) (κ := 1) (ε := 1) zero_le_one
  (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
