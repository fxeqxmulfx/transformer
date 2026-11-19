module

public import Mathlib.Order.Interval.Finset.Fin
public import Transformer.Grokking.FlatSpectrum.Recall.ALiBiWindow

/-!
# The nearer tokens bound every bias that decreases with the distance

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB3,
for the ALiBi bias of arXiv:2505.22842v4, §2.2, Lemma 2 (`Transformer.BAM.alibi`), and the scale
`s log n` of SSMax, arXiv:2501.19399v1, §2.

With raw scores bounded by `B`, attention at the scale `s log n` misses the indicator dictionary
of a token `j₀` by at least `2k / (k + n^{2sB})`, where `k` counts tokens whose bias is at least
that of `j₀` (`le_one_sub_biasedLogScaleAttention_of_subset`). A bias that does not increase with
the distance, as ALiBi's `−m r` at every slope `m ≥ 0` or `−α log(1 + r)` at every `α ≥ 0`, makes
all the `R` nearer tokens such rivals of the key at the distance `R`, so that key misses by at
least `2R / (R + n^{2sB})` (`le_one_sub_biasedLogScaleAttention_of_nearer`): the error of
`Recall.Threshold` with `R` keys in place of `n − 1`. So under every such bias, which may change
with the length, the keys at the distances `R ≥ n^ρ` with `2 s B < ρ` are lost
(`exists_lt_error_of_nearer`); for ALiBi, at every slope (`exists_lt_alibi_error_of_rpow_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **The tokens of no less bias bound the miss** (`review/flat_spectrum.md`, PB3): with raw
scores bounded by `B`, attention at the scale `s log n`, `s ≥ 0`, with a bias `b` answers the
value `1` at `j₀` and `−1` elsewhere with an error of at least `2k / (k + n^{2sB})`, where `k` is
the size of a set `S ∌ j₀` of tokens whose bias is at least that of `j₀`. -/
theorem le_one_sub_biasedLogScaleAttention_of_subset {ι : Type*} [Fintype ι] [DecidableEq ι]
    {s B : ℝ} (hs : 0 ≤ s) {b z : ι → ℝ} (hz : ∀ j, |z j| ≤ B) {j₀ : ι} {S : Finset ι}
    (hS : j₀ ∉ S) (hb : ∀ j ∈ S, b j₀ ≤ b j) :
    2 * (S.card : ℝ) / (S.card + (Fintype.card ι : ℝ) ^ (2 * s * B)) ≤
      1 - biasedLogScaleAttention s b z (fun j => if j = j₀ then 1 else -1) := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩
  have hlog : 0 ≤ s * Real.log (Fintype.card ι) :=
    mul_nonneg hs (Real.log_nonneg (by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩))
  have hng : (Fintype.card ι : ℝ) ^ (2 * s * B) =
      Real.exp (2 * (s * Real.log (Fintype.card ι)) * B) := by
    rw [Real.rpow_def_of_pos hn]
    ring_nf
  rw [biasedLogScaleAttention, one_sub_softmaxAttention_indicator,
    ← add_sum_erase _ _ (mem_univ j₀), hng]
  set c := s * Real.log (Fintype.card ι) with hc
  have hX : (S.card : ℝ) * Real.exp (b j₀ - c * B) ≤
      ∑ j ∈ univ.erase j₀, Real.exp (c * z j + b j) := by
    calc (S.card : ℝ) * Real.exp (b j₀ - c * B) = ∑ j ∈ S, Real.exp (b j₀ - c * B) := by
          rw [sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ S, Real.exp (c * z j + b j) := sum_le_sum fun j hj => Real.exp_le_exp.2 (by
          have := mul_le_mul_of_nonneg_left (neg_le_of_abs_le (hz j)) hlog
          linarith [hb j hj])
      _ ≤ ∑ j ∈ univ.erase j₀, Real.exp (c * z j + b j) :=
          sum_le_sum_of_subset_of_nonneg
            (fun j hj => mem_erase.2 ⟨ne_of_mem_of_not_mem hj hS, mem_univ _⟩)
            fun _ _ _ => (Real.exp_pos _).le
  have hA : Real.exp (c * z j₀ + b j₀) ≤ Real.exp (2 * c * B) * Real.exp (b j₀ - c * B) := by
    rw [← Real.exp_add]
    refine Real.exp_le_exp.2 ?_
    have := mul_le_mul_of_nonneg_left (le_of_abs_le (hz j₀)) hlog
    linarith
  have hk : (0 : ℝ) ≤ S.card := Nat.cast_nonneg _
  have hK := Real.exp_pos (2 * c * B)
  have hA0 := Real.exp_pos (c * z j₀ + b j₀)
  have hX0 : 0 ≤ ∑ j ∈ univ.erase j₀, Real.exp (c * z j + b j) :=
    sum_nonneg fun _ _ => (Real.exp_pos _).le
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_left hA hk, mul_le_mul_of_nonneg_left hX hK.le]

/-- Three tokens, the first of raw score `1` and a rival of equal bias. -/
example := le_one_sub_biasedLogScaleAttention_of_subset (s := 1) (B := 1) zero_le_one
  (b := ![0, 0, -1]) (z := ![1, 0, 0]) (fun j => by fin_cases j <;> simp) (j₀ := 0) (S := {1})
  (by decide) fun j hj => by rw [mem_singleton.1 hj]; simp

/-- **The nearer tokens bound the miss** (`review/flat_spectrum.md`, PB3): over `n` tokens at the
distances `r` with a bias `b` that is at the distance `R` at most its value at every nearer
distance, and raw scores bounded by `B`, attention at the scale `s log n`, `s ≥ 0`, answers the
indicator dictionary of the key at the distance `R` with an error of at least
`2R / (R + n^{2sB})`. -/
theorem le_one_sub_biasedLogScaleAttention_of_nearer {s B : ℝ} (hs : 0 ≤ s) {n : ℕ}
    {b z : Fin n → ℝ} (hz : ∀ r, |z r| ≤ B) {R : Fin n} (hb : ∀ r, r < R → b R ≤ b r) :
    2 * (R : ℝ) / (R + (n : ℝ) ^ (2 * s * B)) ≤
      1 - biasedLogScaleAttention s b z (fun r => if r = R then 1 else -1) := by
  have h := le_one_sub_biasedLogScaleAttention_of_subset hs hz (j₀ := R) (S := Iio R) (by simp)
    fun r hr => hb r (mem_Iio.1 hr)
  rwa [Fin.card_Iio, Fintype.card_fin] at h

/-- The ALiBi bias of the slope `1/2` over three tokens, the key at the distance `2`. -/
example := le_one_sub_biasedLogScaleAttention_of_nearer (s := 1) (B := 1) zero_le_one
  (b := fun r : Fin 3 => -(1 / 2 * r)) (z := ![1, -1, 0]) (fun r => by fin_cases r <;> simp)
  (R := 2) fun r hr => neg_le_neg (mul_le_mul_of_nonneg_left (by exact_mod_cast hr.le)
    (by norm_num))

/-- **Every decreasing bias loses the keys beyond `n^ρ`** (`review/flat_spectrum.md`, PB3): with
raw scores bounded by `B` and `2 s B < ρ`, `s ≥ 0`, for every `ε < 2`, from some length on, every
bias that is at a distance `R ≥ n^ρ` at most its value at every nearer distance answers the
indicator dictionary of the key at `R` with an error above `ε`. -/
theorem exists_lt_error_of_nearer {s B ρ ε : ℝ} (hs : 0 ≤ s) (hρ : 2 * s * B < ρ)
    (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (b z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ ≤ R →
      (∀ r, r < R → b R ≤ b r) → (∀ r, |z r| ≤ B) →
        ε < 1 - biasedLogScaleAttention s b z (fun r => if r = R then 1 else -1) := by
  have hlim : Tendsto (fun n : ℕ => (n : ℝ) ^ (-(ρ - 2 * s * B))) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop (by linarith)).comp tendsto_natCast_atTop_atTop
  obtain ⟨N, hN⟩ := eventually_atTop.1
    (hlim.eventually (gt_mem_nhds (by linarith : (0 : ℝ) < (2 - ε) / 2)))
  refine ⟨N + 1, fun n hn b z R hR hb hz => ?_⟩
  have hn1 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hR0 : (0 : ℝ) < R := (Real.rpow_pos_of_pos hn1 ρ).trans_le hR
  refine lt_of_lt_of_le ?_ (le_one_sub_biasedLogScaleAttention_of_nearer hs hz hb)
  have hK := Real.rpow_pos_of_pos hn1 (2 * s * B)
  have hx : (n : ℝ) ^ (2 * s * B) ≤ (n : ℝ) ^ (-(ρ - 2 * s * B)) * R := by
    calc (n : ℝ) ^ (2 * s * B) = (n : ℝ) ^ (-(ρ - 2 * s * B)) * (n : ℝ) ^ ρ := by
          rw [← Real.rpow_add hn1]
          ring_nf
      _ ≤ _ := mul_le_mul_of_nonneg_left hR (by positivity)
  have hX := hN n (by omega)
  generalize (n : ℝ) ^ (-(ρ - 2 * s * B)) = x at hx hX
  rw [lt_div_iff₀ (by linarith)]
  rcases le_or_gt ε 0 with h0 | h0
  · nlinarith
  · nlinarith [mul_le_mul_of_nonneg_left hx h0.le,
      mul_lt_mul_of_pos_left (mul_lt_mul_of_pos_right hX hR0) h0,
      mul_nonneg (sq_nonneg (ε - 2)) hR0.le]

/-- At `s = 1` and `B = 1/8`, the keys beyond `√n`. -/
example := exists_lt_error_of_nearer (s := 1) (B := 1 / 8) (ρ := 1 / 2) (ε := 1) zero_le_one
  (by norm_num) (by norm_num)

/-- **No ALiBi slope recalls beyond `n^ρ` below `2 s B = ρ`** (`review/flat_spectrum.md`, PB3):
with raw scores bounded by `B` and `2 s B < ρ`, `s ≥ 0`, for every `ε < 2`, from some length on,
at every slope `m ≥ 0`, the key at a distance `R ≥ n^ρ` has its indicator dictionary answered
with an error above `ε`. -/
theorem exists_lt_alibi_error_of_rpow_le {s B ρ ε : ℝ} (hs : 0 ≤ s) (hρ : 2 * s * B < ρ)
    (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ m : ℝ, 0 ≤ m → ∀ (z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ ≤ R →
      (∀ r, |z r| ≤ B) → ε < 1 - biasedLogScaleAttention s (fun r : Fin n => -(m * r)) z
        (fun r => if r = R then 1 else -1) := by
  obtain ⟨N, hN⟩ := exists_lt_error_of_nearer hs hρ hε
  exact ⟨N, fun n hn m hm z R hR hz => hN n hn _ z R hR
    (fun r hr => neg_le_neg (mul_le_mul_of_nonneg_left (by exact_mod_cast hr.le) hm)) hz⟩

/-- At `s = 1` and `B = 1/8`, the keys beyond `√n`. -/
example := exists_lt_alibi_error_of_rpow_le (s := 1) (B := 1 / 8) (ρ := 1 / 2) (ε := 1)
  zero_le_one (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
