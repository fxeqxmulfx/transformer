module

public import Transformer.Grokking.FlatSpectrum.Recall.GGDThreshold
public import Transformer.Grokking.FlatSpectrum.Recall.LogBias

/-!
# GGD-BAM loses the far keys below `2 s B = 1`

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB8,
for the bias that GGD-BAM learns, arXiv:2505.22842v4, Appendix C (`Transformer.BAM.implBias`),
with `θ_μ = 0` as in its experiments (§3), and the scale `s log n` of SSMax, arXiv:2501.19399v1,
§2. The bias is added after the scale, as in `PositionBias`.

For `θ_β > 0` the keys at the distances `R ≥ n^ρ`, `ρ > 0`, leave the window of `GGDWindow` from
some length on, at every scale, since `log n` is `o(n^{ρ θ_β})` (`exists_lt_implBias_error_of_pos`).
For `θ_β ≤ 0` every key is lost once `2 s B < 1` (`GGDThreshold`). So below `2 s B = 1` every head
of GGD-BAM loses every key at a distance `R ≥ n^ρ`, whatever its parameters
(`exists_lt_implBias_error`). The bias `−log(1 + r)` of `LogBias` recalls the keys within `n^ρ`
once `s g > ρ`: at the distances from `n^{ρ'}` to `n^ρ − 1` it recalls what every head of GGD-BAM
loses (`exists_logBias_recall_and_implBias_error`). For one-hot raw scores `±B` the lead is
`g = 2B`, and both hold at every `ρ < 2 s B < 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **A local head loses the keys beyond `n^ρ`** (`review/flat_spectrum.md`, PB8): for `θ_β > 0`,
`ε ≥ 0` and `ρ > 0`, with raw scores bounded by `B`, `s ≥ 0`, for every `ε' < 2`, from some length
on, under BAM's bias every key at a distance `R ≥ n^ρ` has its indicator dictionary answered with
an error above `ε'`, at every scale `s`. -/
theorem exists_lt_implBias_error_of_pos {s B ρ ε' θα θβ ε : ℝ} (hs : 0 ≤ s) (hβ : 0 < θβ)
    (hε : 0 ≤ ε) (hρ : 0 < ρ) (hε' : ε' < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ ≤ R → (∀ r, |z r| ≤ B) →
      ε' < 1 - biasedLogScaleAttention s (fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ))) z
        (fun r => if r = R then 1 else -1) := by
  set κ := max (2 * s * B + 1) 1 with hκ
  have hκ1 : 1 ≤ κ := le_max_right _ _
  obtain ⟨N₁, hN₁⟩ := exists_lt_implBias_error_of_log_le (θα := θα) (θβ := θβ) (ε := ε) hs
    ((lt_add_one _).trans_le (le_max_left _ _) : 2 * s * B < κ) hε'
  set a := ρ * θβ / 2 with ha
  have ha0 : 0 < a := by rw [ha]; positivity
  have hE := Real.exp_pos θα
  obtain ⟨N₂, hN₂⟩ := eventually_atTop.1 (((tendsto_rpow_atTop ha0).comp
    tendsto_natCast_atTop_atTop).eventually (eventually_ge_atTop (κ / (a * Real.exp θα))))
  refine ⟨max N₁ N₂, fun n hn z R hR hz => hN₁ n (le_of_max_le_left hn) z R ?_ hz⟩
  have hx : κ / (a * Real.exp θα) ≤ (n : ℝ) ^ a := hN₂ n (le_of_max_le_right hn)
  rw [div_le_iff₀ (mul_pos ha0 hE)] at hx
  have hna : 0 ≤ (n : ℝ) ^ a := Real.rpow_nonneg (Nat.cast_nonneg n) a
  have h2a : (n : ℝ) ^ a * (n : ℝ) ^ a = ((n : ℝ) ^ ρ) ^ θβ := by
    rw [← Real.rpow_add' (Nat.cast_nonneg n) (add_pos ha0 ha0).ne',
      ← Real.rpow_mul (Nat.cast_nonneg n), ha]
    congr 1
    ring
  calc κ * Real.log n ≤ κ * ((n : ℝ) ^ a / a) :=
        mul_le_mul_of_nonneg_left (Real.log_le_rpow_div (Nat.cast_nonneg n) ha0) (by linarith)
    _ ≤ Real.exp θα * ((n : ℝ) ^ a * (n : ℝ) ^ a) := by
        rw [mul_div_assoc', div_le_iff₀ ha0]
        nlinarith [mul_le_mul_of_nonneg_left hx hna]
    _ = Real.exp θα * ((n : ℝ) ^ ρ) ^ θβ := by rw [h2a]
    _ ≤ Real.exp θα * ((R : ℝ) + ε) ^ θβ :=
        mul_le_mul_of_nonneg_left (Real.rpow_le_rpow (Real.rpow_nonneg (Nat.cast_nonneg n) ρ)
          (by linarith) hβ.le) hE.le

/-- At `θ_α = 0`, `θ_β = 2`, `ε = 10^{−5}`, `s = 1` and `ρ = 1/2`. -/
example := exists_lt_implBias_error_of_pos (s := 1) (B := 1) (ρ := 1 / 2) (ε' := 1) (θα := 0)
  (θβ := 2) (ε := 1 / 100000) zero_le_one two_pos (by norm_num) (by norm_num) (by norm_num)

/-- **Below `2 s B = 1` every head of GGD-BAM loses the far keys** (`review/flat_spectrum.md`,
PB8): with raw scores bounded by `B` and `2 s B < 1`, `s ≥ 0`, for every `ρ > 0`, `ε ≥ 0` and
`ε' < 2`, at every `θ_α` and `θ_β`, from some length on, under BAM's bias every key at a distance
`R ≥ n^ρ` has its indicator dictionary answered with an error above `ε'`. -/
theorem exists_lt_implBias_error {s B ρ ε' : ℝ} (θα θβ : ℝ) {ε : ℝ} (hs : 0 ≤ s) (hε : 0 ≤ ε)
    (hB : 2 * s * B < 1) (hρ : 0 < ρ) (hε' : ε' < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ ≤ R → (∀ r, |z r| ≤ B) →
      ε' < 1 - biasedLogScaleAttention s (fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ))) z
        (fun r => if r = R then 1 else -1) := by
  rcases lt_or_ge 0 θβ with hβ | hβ
  · exact exists_lt_implBias_error_of_pos hs hβ hε hρ hε'
  · obtain ⟨N, hN⟩ := exists_lt_implBias_error_of_nonpos (θα := θα) hs hβ hε hB hε'
    exact ⟨N, fun n hn z R _ hz => hN n hn z R hz⟩

/-- At `θ_α = 0`, `θ_β = −1`, `ε = 10^{−5}`, `s = 1/4`, `B = 1` and `ρ = 1/2`. -/
example := exists_lt_implBias_error 0 (-1) (s := 1 / 4) (B := 1) (ρ := 1 / 2) (ε' := 1)
  (ε := 1 / 100000) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- **The harmonic bias recalls what GGD-BAM loses** (`review/flat_spectrum.md`, PB8): with a raw
lead `g` and raw scores bounded by `B`, `2 s B < 1`, `s ≥ 0`, for `ρ' > 0` and `ρ < s g`, for every
`ε' > 0` and `ε'' < 2`, at every `θ_α`, `θ_β` and `ε ≥ 0`, from some length on, every key at a
distance `R` with `n^{ρ'} ≤ R` and `R + 1 ≤ n^ρ` has its indicator dictionary answered within `ε'`
under the bias `−log(1 + r)` and with an error above `ε''` under BAM's bias. -/
theorem exists_logBias_recall_and_implBias_error {s g B ρ' ρ ε' ε'' : ℝ} (θα θβ : ℝ) {ε : ℝ}
    (hs : 0 ≤ s) (hε : 0 ≤ ε) (hB : 2 * s * B < 1) (hρ' : 0 < ρ') (hρ : ρ < s * g)
    (hε' : 0 < ε') (hε'' : ε'' < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n), (n : ℝ) ^ ρ' ≤ R →
      (R : ℝ) + 1 ≤ (n : ℝ) ^ ρ → (∀ r, r ≠ R → z r + g ≤ z R) → (∀ r, |z r| ≤ B) →
        1 - biasedLogScaleAttention s (fun r : Fin n => -Real.log ((r : ℝ) + 1)) z
          (fun r => if r = R then 1 else -1) ≤ ε' ∧
        ε'' < 1 - biasedLogScaleAttention s (fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ)))
          z (fun r => if r = R then 1 else -1) := by
  obtain ⟨N₁, hN₁⟩ := exists_forall_logBias_recall (C := 1) hs hρ hε'
  obtain ⟨N₂, hN₂⟩ := exists_lt_implBias_error θα θβ hs hε hB hρ' hε''
  refine ⟨max N₁ N₂, fun n hn z R h₁ h₂ hg hz => ⟨?_, hN₂ n (le_of_max_le_right hn) z R h₁ hz⟩⟩
  have h := hN₁ n (le_of_max_le_left hn) z (fun r => if r = R then 1 else -1) R h₂ hg
    fun r => by by_cases hr : r = R <;> simp [hr]
  exact (le_abs_self _).trans (by simpa using h)

/-- One-hot raw scores `±1/2`, with the lead `g = 1`, at `s = 1/2`, `ρ' = 1/8` and `ρ = 1/4`. -/
example := exists_logBias_recall_and_implBias_error 0 (-1) (s := 1 / 2) (g := 1) (B := 1 / 2)
  (ρ' := 1 / 8) (ρ := 1 / 4) (ε' := 1 / 10) (ε'' := 1) (ε := 1 / 100000) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
