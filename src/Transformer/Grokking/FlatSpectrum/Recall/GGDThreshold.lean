module

public import Transformer.Grokking.FlatSpectrum.Recall.GGDWindow

/-!
# The global heads of GGD-BAM: the threshold of NoPE

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB8,
for the bias that GGD-BAM learns, arXiv:2505.22842v4, Appendix C (`Transformer.BAM.implBias`),
with `θ_μ = 0` as in its experiments (§3), and the scale `s log n` of SSMax, arXiv:2501.19399v1,
§2. The bias is added after the scale, as in `PositionBias`.

For `θ_β ≤ 0` and `ε ≥ 0` the bias `−e^{θ_α} (r + ε)^{θ_β}` lies in `[−e^{θ_α}, 0]` at every
distance `r ≥ 1` (`neg_exp_le_implBias`, `implBias_neg_nonpos`): the prior is flat up to the factor
`e^{e^{θ_α}}`, and the head has the threshold of NoPE at every distance. Every key whose raw score
leads by `g` is recalled once `s g > 1` (`exists_forall_implBias_recall_of_nonpos`), and with raw
scores bounded by `B` every key is lost once `2 s B < 1` (`exists_lt_implBias_error_of_nonpos`).
The query's own token, at the distance `0`, has the bias `−e^{θ_α} ε^{θ_β}`, which for `θ_β < 0` is
far below the others (`ε = 10^{−5}` in BAM's implementation, Appendix C) but fixed in `n`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **A global head's bias is bounded below** (`review/flat_spectrum.md`, PB8): for `θ_β ≤ 0` and
`ε ≥ 0`, BAM's bias at a distance `r ≥ 1` is at least `−e^{θ_α}`. -/
theorem neg_exp_le_implBias (θα : ℝ) {θβ ε r : ℝ} (hβ : θβ ≤ 0) (hε : 0 ≤ ε) (hr : 1 ≤ r) :
    -Real.exp θα ≤ BAM.implBias 0 θα θβ ε (-r) := by
  rw [implBias_zero_neg θα θβ ε (by linarith), neg_le_neg_iff]
  exact mul_le_of_le_one_right (Real.exp_pos θα).le
    (Real.rpow_le_one_of_one_le_of_nonpos (by linarith) hβ)

/-- At `θ_α = 0`, `θ_β = −1`, `ε = 10^{−5}` and the distance `2`. -/
example := neg_exp_le_implBias 0 (θβ := -1) (ε := 1 / 100000) (r := 2) (by norm_num) (by norm_num)
  (by norm_num)

/-- BAM's bias is at most `0` at every distance `r ≥ 0`, for `ε ≥ 0`. -/
theorem implBias_neg_nonpos (θα θβ : ℝ) {ε r : ℝ} (hε : 0 ≤ ε) (hr : 0 ≤ r) :
    BAM.implBias 0 θα θβ ε (-r) ≤ 0 := by
  rw [implBias_zero_neg θα θβ ε hr, neg_nonpos]
  exact mul_nonneg (Real.exp_pos θα).le (Real.rpow_nonneg (by linarith) θβ)

/-- At `θ_α = θ_β = 0`, `ε = 10^{−5}` and the distance `0`. -/
example := implBias_neg_nonpos 0 0 (ε := 1 / 100000) (r := 0) (by norm_num) le_rfl

/-- **A global head recalls every key once `s g > 1`** (`review/flat_spectrum.md`, PB8): for
`θ_β ≤ 0`, `ε ≥ 0` and `s g > 1`, `s ≥ 0`, for every `ε' > 0`, from some length on, under BAM's
bias every key whose raw score leads every other by `g` is answered within `ε'`, for values bounded
by `C`: the threshold of NoPE, at every distance. -/
theorem exists_forall_implBias_recall_of_nonpos {s g C ε' θα θβ ε : ℝ} (hs : 0 ≤ s)
    (hβ : θβ ≤ 0) (hε : 0 ≤ ε) (hg : 1 < s * g) (hε' : 0 < ε') :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z v : Fin n → ℝ) (R : Fin n),
      (∀ r, r ≠ R → z r + g ≤ z R) → (∀ r, |v r| ≤ C) →
        |v R - biasedLogScaleAttention s (fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ)))
          z v| ≤ ε' := by
  set D := Real.exp θα * max 1 (ε ^ θβ) with hD
  have hlim : Tendsto (fun n : ℕ => 2 * C * Real.exp D * (n : ℝ) ^ (-(s * g - 1))) atTop
      (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith : 0 < s * g - 1)).comp
      tendsto_natCast_atTop_atTop).const_mul (2 * C * Real.exp D)
    rw [mul_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hlim.eventually (ge_mem_nhds hε'))
  refine ⟨N + 1, fun n hn z v R hz hv => ?_⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv R)
  set b := fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ)) with hb
  have h := abs_sub_biasedLogScaleAttention_le hs b hz hv
  rw [Fintype.card_fin] at h
  refine h.trans (le_trans ?_ (hN n (by omega)))
  have hb0 (r : Fin n) : b r ≤ 0 := implBias_neg_nonpos θα θβ hε (Nat.cast_nonneg _)
  have hbR : -b R ≤ D := by
    change -BAM.implBias 0 θα θβ ε (-((R : ℕ) : ℝ)) ≤ D
    rw [implBias_zero_neg θα θβ ε (Nat.cast_nonneg _), neg_neg, hD]
    refine mul_le_mul_of_nonneg_left ?_ (Real.exp_pos _).le
    rcases Nat.eq_zero_or_pos (R : ℕ) with h0 | h0
    · rw [h0, Nat.cast_zero, zero_add]
      exact le_max_right _ _
    · have h1 : (1 : ℝ) ≤ (R : ℕ) := by exact_mod_cast h0
      exact (Real.rpow_le_one_of_one_le_of_nonpos (by linarith) hβ).trans (le_max_left _ _)
  have hsum : ∑ r ∈ univ.erase R, Real.exp (b r - b R) ≤ n * Real.exp D := by
    refine (sum_le_card_nsmul _ _ (Real.exp D)
      fun r _ => Real.exp_le_exp.2 (by linarith [hb0 r])).trans ?_
    rw [nsmul_eq_mul]
    refine mul_le_mul_of_nonneg_right ?_ (Real.exp_pos _).le
    exact_mod_cast card_erase_le.trans (card_fin n).le
  calc 2 * C * ((n : ℝ) ^ (-(s * g)) * ∑ r ∈ univ.erase R, Real.exp (b r - b R))
      ≤ 2 * C * ((n : ℝ) ^ (-(s * g)) * (n * Real.exp D)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hsum (by positivity)) (by linarith)
    _ = 2 * C * Real.exp D * (n : ℝ) ^ (-(s * g - 1)) := by
        rw [show -(s * g - 1) = -(s * g) + 1 by ring, Real.rpow_add hn0, Real.rpow_one]
        ring

/-- At `θ_α = 0`, `θ_β = −1`, `ε = 10^{−5}`, `s = 1` and `g = 2`. -/
example := exists_forall_implBias_recall_of_nonpos (s := 1) (g := 2) (C := 1) (ε' := 1 / 10)
  (θα := 0) (θβ := -1) (ε := 1 / 100000) zero_le_one (by norm_num) (by norm_num) (by norm_num)
  (by norm_num)

/-- **A global head loses every key once `2 s B < 1`** (`review/flat_spectrum.md`, PB8): for
`θ_β ≤ 0`, `ε ≥ 0` and raw scores bounded by `B` with `2 s B < 1`, `s ≥ 0`, for every `ε' < 2`,
from some length on, under BAM's bias every key has its indicator dictionary answered with an error
above `ε'`. -/
theorem exists_lt_implBias_error_of_nonpos {s B ε' θα θβ ε : ℝ} (hs : 0 ≤ s) (hβ : θβ ≤ 0)
    (hε : 0 ≤ ε) (hB : 2 * s * B < 1) (hε' : ε' < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), (∀ r, |z r| ≤ B) →
      ε' < 1 - biasedLogScaleAttention s (fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ))) z
        (fun r => if r = R then 1 else -1) := by
  set E := Real.exp (-Real.exp θα) with hE
  have hE0 : 0 < E := Real.exp_pos _
  obtain ⟨N, hN⟩ := exists_lt_one_sub_of_le_sum_erase hs (by positivity : 0 < E / 3) hB hε'
  refine ⟨max N 3, fun n hn z R hz => hN n (le_of_max_le_left hn) _ z R ?_ hz⟩
  have hn3 : 3 ≤ n := le_of_max_le_right hn
  have hn3' : (3 : ℝ) ≤ n := by exact_mod_cast hn3
  set S := (univ.erase R).erase (⟨0, by omega⟩ : Fin n) with hS
  have hcard : (n : ℝ) - 2 ≤ S.card := by
    have h1 : (univ.erase R).card - 1 ≤ S.card := pred_card_le_card_erase
    have h2 : (univ : Finset (Fin n)).card - 1 ≤ (univ.erase R).card := pred_card_le_card_erase
    rw [card_fin] at h2
    have h3 : ((n - 2 : ℕ) : ℝ) ≤ S.card := by exact_mod_cast (by omega : n - 2 ≤ S.card)
    rwa [Nat.cast_sub (by omega), Nat.cast_ofNat] at h3
  have hR : Real.exp (BAM.implBias 0 θα θβ ε (-(R : ℝ))) ≤ 1 :=
    Real.exp_le_one_iff.2 (implBias_neg_nonpos θα θβ hε (Nat.cast_nonneg _))
  have hr (r : Fin n) (hr : r ∈ S) : E ≤ Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ))) := by
    refine Real.exp_le_exp.2 (neg_exp_le_implBias θα hβ hε ?_)
    have h0 : (r : ℕ) ≠ 0 := fun h => (mem_erase.1 hr).1 (Fin.ext h)
    exact_mod_cast Nat.one_le_iff_ne_zero.2 h0
  calc E / 3 * (n : ℝ) ^ (1 : ℝ) * Real.exp (BAM.implBias 0 θα θβ ε (-(R : ℝ)))
      ≤ E * S.card := by
        rw [Real.rpow_one]
        nlinarith [mul_le_mul_of_nonneg_left hR (by positivity : (0 : ℝ) ≤ E * n / 3),
          mul_le_mul_of_nonneg_left hcard hE0.le]
    _ = ∑ r ∈ S, E := by rw [sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ r ∈ S, Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ))) := sum_le_sum hr
    _ ≤ ∑ r ∈ univ.erase R, Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ))) :=
        sum_le_sum_of_subset_of_nonneg (erase_subset _ _) fun _ _ _ => (Real.exp_pos _).le

/-- At `θ_α = 0`, `θ_β = −1`, `ε = 10^{−5}`, `s = 1` and `B = 1/4`. -/
example := exists_lt_implBias_error_of_nonpos (s := 1) (B := 1 / 4) (ε' := 1) (θα := 0)
  (θβ := -1) (ε := 1 / 100000) zero_le_one (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
