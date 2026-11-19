module

public import Transformer.Grokking.FlatSpectrum.Recall.PowerThreshold

/-!
# The prior is a budget

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB7,
for the positional priors of arXiv:2505.22842v4, §2.1–§2.3, and the scale `s log n` of SSMax,
arXiv:2501.19399v1, §2.

A bias `b` of the tokens is the prior `p_j = e^{b_j} / ∑_k e^{b_k}`. With raw scores bounded by
`B`, a token `j₀` whose indicator dictionary attention at the scale `s log n` answers within
`ε < 2` has the prior `p_{j₀} ≥ (2 − ε) / (2 − ε + ε n^{2sB})` (`mul_sum_exp_le_of_one_sub_le`),
by the weighted bound of `NearerTokens`. The priors sum to `1`, so under every bias at most
`1 + ε n^{2sB} / (2 − ε)` tokens are answered within `ε` (`card_le_of_one_sub_le`), and below
`2 s B = 1`, for every `δ > 0`, from some length on at most `δ n` of the `n` distances, under every
bias of each length (`exists_card_filter_le_mul`). The bias chooses which keys the scale reaches,
not how many. Conversely, a key whose rivals outweigh it by `w n^γ` in the prior is lost when
`2 s B < γ` (`exists_lt_one_sub_of_le_sum_erase`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **A recalled key needs the prior `n^{−2sB}`** (`review/flat_spectrum.md`, PB7): with raw
scores bounded by `B`, if attention at the scale `s log n`, `s ≥ 0`, with the bias `b` answers
the indicator dictionary of `j₀` within `ε`, then
`(2 − ε) ∑_j e^{b_j} ≤ (2 − ε + ε n^{2sB}) e^{b_{j₀}}`: for `ε < 2` the prior of `j₀` is at least
`(2 − ε) / (2 − ε + ε n^{2sB})`. -/
theorem mul_sum_exp_le_of_one_sub_le {ι : Type*} [Fintype ι] [DecidableEq ι] {s B ε : ℝ}
    (hs : 0 ≤ s) {b z : ι → ℝ} (hz : ∀ j, |z j| ≤ B) {j₀ : ι}
    (h : 1 - biasedLogScaleAttention s b z (fun j => if j = j₀ then 1 else -1) ≤ ε) :
    (2 - ε) * ∑ j, Real.exp (b j) ≤
      (2 - ε + ε * (Fintype.card ι : ℝ) ^ (2 * s * B)) * Real.exp (b j₀) := by
  have hW := (le_one_sub_biasedLogScaleAttention_of_weight hs hz (notMem_erase j₀ univ)).trans h
  rw [sum_erase_eq_sub (mem_univ j₀)] at hW
  have hK : 0 < (Fintype.card ι : ℝ) ^ (2 * s * B) :=
    Real.rpow_pos_of_pos (by exact_mod_cast Fintype.card_pos_iff.2 ⟨j₀⟩) _
  have he := Real.exp_pos (b j₀)
  have hP : Real.exp (b j₀) ≤ ∑ j, Real.exp (b j) :=
    single_le_sum (fun j _ => (Real.exp_pos (b j)).le) (mem_univ j₀)
  rw [div_le_iff₀ (by nlinarith)] at hW
  nlinarith

/-- The miss of an indicator dictionary is at most `2`. -/
lemma one_sub_biasedLogScaleAttention_le_two {ι : Type*} [Fintype ι] [DecidableEq ι] (s : ℝ)
    (b z : ι → ℝ) (j₀ : ι) :
    1 - biasedLogScaleAttention s b z (fun j => if j = j₀ then 1 else -1) ≤ 2 := by
  rw [biasedLogScaleAttention, one_sub_softmaxAttention_indicator]
  have hZ : 0 < ∑ j, Real.exp (s * Real.log (Fintype.card ι) * z j + b j) :=
    sum_pos (fun j _ => Real.exp_pos _) ⟨j₀, mem_univ _⟩
  have := sum_le_sum_of_subset_of_nonneg (erase_subset j₀ univ)
    fun j _ _ => (Real.exp_pos (s * Real.log (Fintype.card ι) * z j + b j)).le
  rw [div_le_iff₀ hZ]
  linarith

/-- Three tokens, the first answered within `2`. -/
example := mul_sum_exp_le_of_one_sub_le (s := 1) (B := 1) zero_le_one (b := ![0, 0, -1])
  (z := ![1, 0, 0]) (fun j => by fin_cases j <;> simp)
  (one_sub_biasedLogScaleAttention_le_two 1 _ _ (0 : Fin 3))

/-- **The prior is a budget** (`review/flat_spectrum.md`, PB7): with raw scores bounded by `B`,
attention at the scale `s log n`, `s ≥ 0`, under any bias `b` answers the indicator dictionaries of
at most `1 + ε n^{2sB} / (2 − ε)` tokens within `ε`, `0 ≤ ε < 2`. The raw scores `z j₀` may depend
on the token `j₀` that holds the key; the bias does not. -/
theorem card_le_of_one_sub_le {ι : Type*} [Fintype ι] [DecidableEq ι] {s B ε : ℝ} (hs : 0 ≤ s)
    (hε0 : 0 ≤ ε) (hε : ε < 2) (b : ι → ℝ) {z : ι → ι → ℝ} (hz : ∀ j₀ j, |z j₀ j| ≤ B)
    {T : Finset ι} (hT : ∀ j₀ ∈ T,
      1 - biasedLogScaleAttention s b (z j₀) (fun j => if j = j₀ then 1 else -1) ≤ ε) :
    (T.card : ℝ) ≤ 1 + ε / (2 - ε) * (Fintype.card ι : ℝ) ^ (2 * s * B) := by
  set K := (Fintype.card ι : ℝ) ^ (2 * s * B) with hKdef
  have hK : 0 ≤ K := Real.rpow_nonneg (Nat.cast_nonneg _) _
  have h2 : (0 : ℝ) < 2 - ε := by linarith
  suffices hc : (T.card : ℝ) * (2 - ε) ≤ 2 - ε + ε * K by
    rw [div_mul_eq_mul_div, ← sub_le_iff_le_add', le_div_iff₀ h2]
    linarith
  rcases T.eq_empty_or_nonempty with rfl | ⟨j₁, -⟩
  · simp only [card_empty, Nat.cast_zero, zero_mul]
    nlinarith
  have hP : 0 < ∑ j, Real.exp (b j) := sum_pos (fun j _ => Real.exp_pos (b j)) ⟨j₁, mem_univ _⟩
  have h1 := card_nsmul_le_sum T (fun j₀ => (2 - ε + ε * K) * Real.exp (b j₀))
    ((2 - ε) * ∑ j, Real.exp (b j)) fun j₀ hj₀ =>
      mul_sum_exp_le_of_one_sub_le hs (hz j₀) (hT j₀ hj₀)
  rw [nsmul_eq_mul, ← mul_sum] at h1
  have h3 : ∑ j₀ ∈ T, Real.exp (b j₀) ≤ ∑ j, Real.exp (b j) :=
    sum_le_sum_of_subset_of_nonneg (subset_univ T) fun j _ _ => (Real.exp_pos (b j)).le
  have h4 := h1.trans (mul_le_mul_of_nonneg_left h3 (by nlinarith))
  rw [← mul_assoc] at h4
  exact le_of_mul_le_mul_right h4 hP

/-- One token, answered exactly, within `1`. -/
example := card_le_of_one_sub_le (s := 1) (B := 1) (ε := 1) zero_le_one zero_le_one one_lt_two
  ![0] (z := fun _ => ![1]) (fun _ j => by fin_cases j; simp) (T := univ) fun j₀ _ => by
    rw [biasedLogScaleAttention, one_sub_softmaxAttention_indicator]
    fin_cases j₀
    simp

/-- **Below `2 s B = 1` almost every distance is lost under every bias**
(`review/flat_spectrum.md`, PB7): with raw scores bounded by `B`, `2 s B < 1`, `s ≥ 0`, for every
`0 ≤ ε < 2` and `δ > 0`, from some length on, under every bias of the `n` distances, attention at
the scale `s log n` answers the indicator dictionaries of at most `δ n` distances within `ε`. -/
theorem exists_card_filter_le_mul {s B ε δ : ℝ} (hs : 0 ≤ s) (hB : 2 * s * B < 1) (hε0 : 0 ≤ ε)
    (hε : ε < 2) (hδ : 0 < δ) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (b : Fin n → ℝ) (z : Fin n → Fin n → ℝ), (∀ R r, |z R r| ≤ B) →
      ((univ.filter fun R : Fin n => 1 - biasedLogScaleAttention s b (z R)
        (fun r => if r = R then 1 else -1) ≤ ε).card : ℝ) ≤ δ * n := by
  set c := ε / (2 - ε)
  have hlim : Tendsto (fun n : ℕ => (n : ℝ) ^ (-(1 : ℝ)) + c * (n : ℝ) ^ (-(1 - 2 * s * B)))
      atTop (𝓝 0) := by
    have h1 := (tendsto_rpow_neg_atTop zero_lt_one).comp tendsto_natCast_atTop_atTop
    have h2 := ((tendsto_rpow_neg_atTop (by linarith : 0 < 1 - 2 * s * B)).comp
      tendsto_natCast_atTop_atTop).const_mul c
    have h := h1.add h2
    rw [mul_zero, add_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hlim.eventually (ge_mem_nhds hδ))
  refine ⟨N + 1, fun n hn b z hz => ?_⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  refine (card_le_of_one_sub_le hs hε0 hε b hz fun R hR => (mem_filter.1 hR).2).trans ?_
  rw [Fintype.card_fin]
  have e : (n : ℝ) ^ (2 * s * B) = n * (n : ℝ) ^ (-(1 - 2 * s * B)) := by
    conv_lhs => rw [show 2 * s * B = 1 + -(1 - 2 * s * B) by ring]
    rw [Real.rpow_add hn0, Real.rpow_one]
  have e1 : (n : ℝ) * (n : ℝ) ^ (-(1 : ℝ)) = 1 := by
    rw [Real.rpow_neg_one, mul_inv_cancel₀ hn0.ne']
  calc 1 + c * (n : ℝ) ^ (2 * s * B)
      = n * ((n : ℝ) ^ (-(1 : ℝ)) + c * (n : ℝ) ^ (-(1 - 2 * s * B))) := by
        rw [e, mul_add, e1]
        ring
    _ ≤ n * δ := mul_le_mul_of_nonneg_left (hN n (by omega)) hn0.le
    _ = δ * n := mul_comm _ _

/-- At `s = 1/4`, `B = 1` and `ε = δ = 1`. -/
example := exists_card_filter_le_mul (s := 1 / 4) (B := 1) (ε := 1) (δ := 1) (by norm_num)
  (by norm_num) zero_le_one (by norm_num) one_pos

/-- **Rivals that outweigh the key by `w n^γ` lose it** (`review/flat_spectrum.md`, PB7): for
`w > 0` and raw scores bounded by `B` with `2 s B < γ`, `s ≥ 0`, for every `ε < 2`, from some
length on, under every bias `b` a key at `R` whose rivals weigh `∑_{r ≠ R} e^{b_r}` at least
`w n^γ e^{b_R}` has its indicator dictionary answered with an error above `ε`. -/
theorem exists_lt_one_sub_of_le_sum_erase {s B γ w ε : ℝ} (hs : 0 ≤ s) (hw : 0 < w)
    (hγ : 2 * s * B < γ) (hε : ε < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (b z : Fin n → ℝ) (R : Fin n),
      w * (n : ℝ) ^ γ * Real.exp (b R) ≤ ∑ r ∈ univ.erase R, Real.exp (b r) →
      (∀ r, |z r| ≤ B) →
        ε < 1 - biasedLogScaleAttention s b z (fun r => if r = R then 1 else -1) := by
  set a := γ - 2 * s * B with ha
  have hlim : Tendsto (fun n : ℕ => (n : ℝ) ^ (-a)) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop (by linarith)).comp tendsto_natCast_atTop_atTop
  obtain ⟨N, hN⟩ := eventually_atTop.1
    (hlim.eventually (gt_mem_nhds (by nlinarith : (0 : ℝ) < (2 - ε) * w / 2)))
  refine ⟨N + 1, fun n hn b z R hW hz => ?_⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have h := le_one_sub_biasedLogScaleAttention_of_weight (b := b) hs hz (notMem_erase R univ)
  rw [Fintype.card_fin] at h
  refine lt_of_lt_of_le ?_ h
  have e1 : (n : ℝ) ^ (2 * s * B) = (n : ℝ) ^ (-a) * (n : ℝ) ^ γ := by
    rw [← Real.rpow_add hn0]
    congr 1
    rw [ha]
    ring
  rw [e1]
  have hx := hN n (by omega)
  have hG := Real.rpow_pos_of_pos hn0 γ
  have hX := Real.rpow_pos_of_pos hn0 (-a)
  have he := Real.exp_pos (b R)
  generalize ∑ r ∈ univ.erase R, Real.exp (b r) = W at hW ⊢
  generalize (n : ℝ) ^ γ = G at hG hW ⊢
  generalize (n : ℝ) ^ (-a) = X at hX hx ⊢
  generalize Real.exp (b R) = e at he hW ⊢
  have hW0 : 0 < W := lt_of_lt_of_le (by positivity) hW
  rw [lt_div_iff₀ (by positivity)]
  rcases le_or_gt ε 0 with h0 | h0
  · nlinarith [mul_pos (mul_pos hX hG) he]
  · nlinarith [mul_lt_mul_of_pos_right hx (by positivity : 0 < ε * G * e),
      mul_le_mul_of_nonneg_left hW (by nlinarith : (0 : ℝ) ≤ ε * (2 - ε) / 2),
      mul_nonneg (sq_nonneg (ε - 2)) hW0.le]

/-- At `w = γ = 1`, `s = 1` and `B = 1/4`: rivals of weight `n` lose the key. -/
example := exists_lt_one_sub_of_le_sum_erase (s := 1) (B := 1 / 4) (γ := 1) (w := 1) (ε := 1)
  zero_le_one one_pos (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
