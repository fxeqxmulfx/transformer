module

public import Mathlib.Algebra.Order.Field.GeomSum
public import Transformer.Grokking.FlatSpectrum.Recall.PositionBias

/-!
# A fixed ALiBi slope at the scale `s log n`: a window of radius `(2sB/m) log n`

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB2,
for the ALiBi bias of arXiv:2505.22842v4, §2.2, Lemma 2 (`Transformer.BAM.alibi`), and the scale
`s log n` of SSMax, arXiv:2501.19399v1, §2.

The query reads `n` tokens at the distances `r = 0, …, n − 1`, the token `r` with the ALiBi bias
`−m r` of a slope `m > 0`. The prior weights `e^{−m r}` sum to at most `1 / (1 − e^{−m})` at every
length (`sum_exp_neg_mul_le`). So a key at the distance `R` whose raw score leads every other by
`g` is answered within `2 C n^{−sg} e^{mR} / (1 − e^{−m})` (`abs_sub_alibi_le`), and with raw
scores bounded by `B` the indicator dictionary of a key at a distance `R ≥ 1` misses by at least
`2 / (1 + n^{2sB} e^{−mR})`, against the query's own token (`le_one_sub_alibi`).

At the scale `s log n` the slope leaves a window of radius about `(s g / m) log n`: keys within
`D + ρ log n` of the query with `m ρ < s g` are recalled from some length on
(`exists_forall_alibi_recall`), and keys beyond `ρ log n` with `m ρ > 2 s B` miss by an error
that tends to `2`, the opposite value (`exists_lt_alibi_error`). For one-hot raw scores `±B` the
lead is `g = 2B` and the two radii meet at `(2 s B / m) log n`, a vanishing fraction of the
context. Inside the window the threshold `2 s B = 1` of the scale `s log n` without a bias
(`Recall.Threshold`) is gone: at `ρ = 0` a key at a bounded distance is recalled at every
positive lead, since the bias bounds the weight of all the other tokens by a constant that
`n^{sg}` outgrows. At a fixed scale the window is fixed instead (`Transformer.BAM.alibi_local`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **The ALiBi prior is bounded** (`review/flat_spectrum.md`, PB2): the weights `e^{−m r}` of
the distances `r < n` sum to at most `1 / (1 − e^{−m})` at every length `n`, for `m > 0`. -/
lemma sum_exp_neg_mul_le {m : ℝ} (hm : 0 < m) (n : ℕ) :
    ∑ r : Fin n, Real.exp (-(m * r)) ≤ 1 / (1 - Real.exp (-m)) := by
  have h1 : Real.exp (-m) < 1 := Real.exp_lt_one_iff.2 (by linarith)
  have h := geom_sum_Ico_le_of_lt_one (m := 0) (n := n) (Real.exp_pos (-m)).le h1
  rw [pow_zero, ← Finset.range_eq_Ico] at h
  rw [Fin.sum_univ_eq_sum_range (fun r : ℕ => Real.exp (-(m * r))) n]
  refine le_trans (le_of_eq (sum_congr rfl fun r _ => ?_)) h
  rw [← Real.exp_nat_mul]
  congr 1
  ring

/-- At `m = 1` and five tokens. -/
example := sum_exp_neg_mul_le one_pos 5

/-- **Inside the window** (`review/flat_spectrum.md`, PB2): over `n` tokens at the distances `r`
with the ALiBi bias `−m r`, `m > 0`, if the raw score of the token at the distance `R` leads every
other by `g` and the values are bounded by `C`, attention at the scale `s log n`, `s ≥ 0`,
answers its value within `2 C n^{−sg} e^{mR} / (1 − e^{−m})`. -/
theorem abs_sub_alibi_le {s g m C : ℝ} (hs : 0 ≤ s) (hm : 0 < m) {n : ℕ} {z v : Fin n → ℝ}
    {R : Fin n} (hz : ∀ r, r ≠ R → z r + g ≤ z R) (hv : ∀ r, |v r| ≤ C) :
    |v R - biasedLogScaleAttention s (fun r : Fin n => -(m * r)) z v| ≤
      2 * C * ((n : ℝ) ^ (-(s * g)) * Real.exp (m * R) / (1 - Real.exp (-m))) := by
  have h := abs_sub_biasedLogScaleAttention_le hs (fun r : Fin n => -(m * r)) hz hv
  rw [Fintype.card_fin] at h
  refine h.trans (mul_le_mul_of_nonneg_left ?_ (by linarith [(abs_nonneg _).trans (hv R)]))
  rw [mul_div_assoc]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  calc ∑ r ∈ univ.erase R, Real.exp (-(m * r) - -(m * R))
      ≤ ∑ r : Fin n, Real.exp (-(m * r) - -(m * R)) :=
        sum_le_sum_of_subset_of_nonneg (erase_subset _ _) fun _ _ _ => (Real.exp_pos _).le
    _ = Real.exp (m * R) * ∑ r : Fin n, Real.exp (-(m * r)) := by
        rw [mul_sum]
        exact sum_congr rfl fun r _ => by rw [← Real.exp_add]; congr 1; ring
    _ ≤ Real.exp (m * R) * (1 / (1 - Real.exp (-m))) :=
        mul_le_mul_of_nonneg_left (sum_exp_neg_mul_le hm n) (Real.exp_pos _).le
    _ = Real.exp (m * R) / (1 - Real.exp (-m)) := by ring

/-- Three tokens, the query's own token leading by `1`, at the slope `1/2`. -/
example := abs_sub_alibi_le (s := 1) (g := 1) (m := 1 / 2) (C := 1) zero_le_one (by norm_num)
  (z := ![1, 0, 0]) (v := ![1, -1, -1]) (R := 0) (fun r hr => by fin_cases r <;> simp_all)
  fun r => by fin_cases r <;> simp

/-- **Outside the window** (`review/flat_spectrum.md`, PB2): over `n` tokens at the distances `r`
with the ALiBi bias `−m r` and raw scores bounded by `B`, attention at the scale `s log n`,
`s ≥ 0`, answers the value `1` at the distance `R ≥ 1` and `−1` elsewhere with an error of at
least `2 / (1 + n^{2sB} e^{−mR})`. -/
theorem le_one_sub_alibi {s B : ℝ} (hs : 0 ≤ s) (m : ℝ) {n : ℕ} {z : Fin n → ℝ}
    (hz : ∀ r, |z r| ≤ B) {R : Fin n} (hR : (R : ℕ) ≠ 0) :
    2 / (1 + (n : ℝ) ^ (2 * s * B) * Real.exp (-(m * R))) ≤
      1 - biasedLogScaleAttention s (fun r : Fin n => -(m * r)) z
        (fun r => if r = R then 1 else -1) := by
  have h := le_one_sub_biasedLogScaleAttention hs (fun r : Fin n => -(m * r)) hz (j₀ := R)
    (j₁ := ⟨0, R.pos⟩) fun h => hR (congrArg Fin.val h).symm
  rw [Fintype.card_fin] at h
  rwa [show -(m * (R : ℝ)) - -(m * (((⟨0, R.pos⟩ : Fin n) : ℕ) : ℝ)) = -(m * R) by simp] at h

/-- Three tokens with raw scores `1`, `−1` and `0`, the key at the distance `2`. -/
example := le_one_sub_alibi (s := 1) (B := 1) zero_le_one (1 / 2) (z := ![1, -1, 0])
  (fun r => by fin_cases r <;> simp) (R := 2) (by decide)

/-- **Keys inside the window are recalled** (`review/flat_spectrum.md`, PB2): with the ALiBi bias
`−m r`, `m > 0`, at the scale `s log n`, `s ≥ 0`, if `m ρ < s g` then for every `D` and `ε > 0`,
from some length on, every key within the distance `D + ρ log n` whose raw score leads every other
by `g` is answered within `ε`, for values bounded by `C`. At `ρ = 0` this needs only `s g > 0`. -/
theorem exists_forall_alibi_recall {s g m D ρ C ε : ℝ} (hs : 0 ≤ s) (hm : 0 < m)
    (hρ : m * ρ < s * g) (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z v : Fin n → ℝ) (R : Fin n), (R : ℝ) ≤ D + ρ * Real.log n →
      (∀ r, r ≠ R → z r + g ≤ z R) → (∀ r, |v r| ≤ C) →
        |v R - biasedLogScaleAttention s (fun r : Fin n => -(m * r)) z v| ≤ ε := by
  have hm1 : 0 < 1 - Real.exp (-m) := by linarith [Real.exp_lt_one_iff.2 (by linarith : -m < 0)]
  have hlim : Tendsto (fun n : ℕ => 2 * C * Real.exp (m * D) / (1 - Real.exp (-m)) *
      (n : ℝ) ^ (-(s * g - m * ρ))) atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (y := s * g - m * ρ) (by linarith)).comp
      tendsto_natCast_atTop_atTop).const_mul (2 * C * Real.exp (m * D) / (1 - Real.exp (-m)))
    rw [mul_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hlim.eventually (ge_mem_nhds hε))
  refine ⟨N + 1, fun n hn z v R hR hz hv => (abs_sub_alibi_le hs hm hz hv).trans
    (le_trans ?_ (hN n (by omega)))⟩
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (by omega : 1 ≤ n)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv R)
  have heR : Real.exp (m * R) ≤ Real.exp (m * D) * (n : ℝ) ^ (m * ρ) := by
    rw [Real.rpow_def_of_pos (by linarith), ← Real.exp_add]
    exact Real.exp_le_exp.2 (by nlinarith [mul_le_mul_of_nonneg_left hR hm.le])
  have hsplit : (n : ℝ) ^ (-(s * g - m * ρ)) = (n : ℝ) ^ (-(s * g)) * (n : ℝ) ^ (m * ρ) := by
    rw [← Real.rpow_add (by linarith)]
    ring_nf
  calc 2 * C * ((n : ℝ) ^ (-(s * g)) * Real.exp (m * R) / (1 - Real.exp (-m)))
      = 2 * C / (1 - Real.exp (-m)) * (n : ℝ) ^ (-(s * g)) * Real.exp (m * R) := by ring
    _ ≤ 2 * C / (1 - Real.exp (-m)) * (n : ℝ) ^ (-(s * g)) *
        (Real.exp (m * D) * (n : ℝ) ^ (m * ρ)) := mul_le_mul_of_nonneg_left heR
          (mul_nonneg (div_nonneg (by linarith) hm1.le) (by positivity))
    _ = 2 * C * Real.exp (m * D) / (1 - Real.exp (-m)) * (n : ℝ) ^ (-(s * g - m * ρ)) := by
        rw [hsplit]
        ring

/-- At `s = g = m = 1`, the keys within `(log n) / 2`. -/
example := exists_forall_alibi_recall (s := 1) (g := 1) (m := 1) (D := 0) (ρ := 1 / 2) (C := 1)
  (ε := 1 / 10) zero_le_one one_pos (by norm_num) (by norm_num)

/-- A lead of `1/10` at `s = 1`, below the threshold `s g = 1` without a bias, at the keys within
the distance `5`. -/
example := exists_forall_alibi_recall (s := 1) (g := 1 / 10) (m := 1) (D := 5) (ρ := 0) (C := 1)
  (ε := 1 / 10) zero_le_one one_pos (by norm_num) (by norm_num)

/-- **Keys outside the window are lost** (`review/flat_spectrum.md`, PB2): with the ALiBi bias
`−m r`, `m > 0`, at the scale `s log n`, `s ≥ 0`, and raw scores bounded by `B`, if `2 s B < m ρ`
then for every `ε < 2`, from some length on, every key beyond the distance `ρ log n` has its
indicator dictionary answered with an error above `ε`. -/
theorem exists_lt_alibi_error {s B m ρ ε : ℝ} (hs : 0 ≤ s) (hm : 0 < m) (hρ : 2 * s * B < m * ρ)
    (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), ρ * Real.log n ≤ R → (∀ r, |z r| ≤ B) →
      ε < 1 - biasedLogScaleAttention s (fun r : Fin n => -(m * r)) z
        (fun r => if r = R then 1 else -1) := by
  have hlim : Tendsto (fun n : ℕ => (n : ℝ) ^ (-(m * ρ - 2 * s * B))) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop (by linarith)).comp tendsto_natCast_atTop_atTop
  obtain ⟨N, hN⟩ := eventually_atTop.1
    (hlim.eventually (gt_mem_nhds (by linarith : (0 : ℝ) < (2 - ε) / 2)))
  refine ⟨N + 2, fun n hn z R hR hz => ?_⟩
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast (by omega : 2 ≤ n)
  have hlog : 0 < Real.log n := Real.log_pos (by linarith)
  have hB : 0 ≤ B := (abs_nonneg _).trans (hz R)
  have hρ0 : 0 < ρ := by nlinarith [mul_nonneg hs hB]
  have hR0 : (R : ℕ) ≠ 0 := fun h0 => by
    rw [h0, Nat.cast_zero] at hR
    nlinarith [mul_pos hρ0 hlog]
  refine lt_of_lt_of_le ?_ (le_one_sub_alibi hs m hz hR0)
  have hx : (n : ℝ) ^ (2 * s * B) * Real.exp (-(m * R)) ≤ (n : ℝ) ^ (-(m * ρ - 2 * s * B)) := by
    rw [Real.rpow_def_of_pos (by linarith), Real.rpow_def_of_pos (by linarith), ← Real.exp_add]
    exact Real.exp_le_exp.2 (by nlinarith [mul_le_mul_of_nonneg_left hR hm.le])
  have hX := hN n (by omega)
  have hx0 : 0 ≤ (n : ℝ) ^ (2 * s * B) * Real.exp (-(m * R)) := by positivity
  generalize (n : ℝ) ^ (2 * s * B) * Real.exp (-(m * R)) = x at hx hx0 ⊢
  rw [lt_div_iff₀ (by linarith)]
  rcases le_or_gt ε 0 with h0 | h0
  · nlinarith
  · nlinarith [mul_lt_mul_of_pos_left (hx.trans_lt hX) h0, sq_nonneg (ε - 2)]

/-- At `s = m = ρ = 1` and `B = 1/4`, the keys beyond `log n`. -/
example := exists_lt_alibi_error (s := 1) (B := 1 / 4) (m := 1) (ρ := 1) (ε := 1) zero_le_one
  one_pos (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
