module

public import Transformer.BAM.SectionC_Parametrization
public import Transformer.Grokking.FlatSpectrum.Recall.PriorBudget

/-!
# The local heads of GGD-BAM: a window polylogarithmic in the length

Source: `review/flat_spectrum.md`, *A positional bias at the scale `s log n`* (added), item PB8,
for the bias that GGD-BAM learns, arXiv:2505.22842v4, Appendix C (`Transformer.BAM.implBias`),
with `θ_μ = 0` as in its experiments (§3: "we train only `θ_α` and `θ_β`, fixing `θ_μ = 0`"), and
the scale `s log n` of SSMax, arXiv:2501.19399v1, §2. The bias is added after the scale, as in
`PositionBias`.

At the distance `r` before the query the bias is `−e^{θ_α} (r + ε)^{θ_β}` (`implBias_zero_neg`).
For `θ_β > 0` its prior has a mass bounded at every length (`exists_sum_exp_implBias_le`): by
`y^K / K! ≤ e^y` with `θ_β K ≥ 2` its weights lie below a multiple of the power prior
`(1 + r)^{−2}` of `PowerBias`. So with the lead `g` the keys with `e^{θ_α} (R + ε)^{θ_β} ≤ κ log n`,
`κ < s g`, are recalled (`exists_forall_implBias_recall`), and with raw scores bounded by `B` the
keys with `e^{θ_α} (R + ε)^{θ_β} ≥ κ log n`, `κ > 2 s B`, are lost
(`exists_lt_implBias_error_of_log_le`). So the radius of the window lies between
`α (s g log n)^{1/θ_β}` and `α (2 s B log n)^{1/θ_β}`, up to `ε`, where `α = e^{−θ_α/θ_β}` is the
scale of the GGD (`BAM.implParams_iff`): it grows polylogarithmically with the length, and at
`θ_β = 1` it is the window of ALiBi with the slope `e^{θ_α}` (`ALiBiWindow`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- **BAM's bias at the distance `r`** (`review/flat_spectrum.md`, PB8): with `θ_μ = 0`, at the
relative position `x = −r` of a key `r ≥ 0` tokens before the query, the bias of the
implementation is `−e^{θ_α} (r + ε)^{θ_β}`. -/
theorem implBias_zero_neg (θα θβ ε : ℝ) {r : ℝ} (hr : 0 ≤ r) :
    BAM.implBias 0 θα θβ ε (-r) = -(Real.exp θα * (r + ε) ^ θβ) := by
  simp [BAM.implBias, abs_of_nonneg hr]

/-- At `θ_α = 0`, `θ_β = 1`, `ε = 10^{−5}` and the distance `3`. -/
example := implBias_zero_neg 0 1 (1 / 100000) (r := 3) (by norm_num)

/-- **A local head has a prior of bounded mass** (`review/flat_spectrum.md`, PB8): for
`θ_β > 0` and `ε ≥ 0` the weights `e^{−e^{θ_α} (r + ε)^{θ_β}}` of the distances `r < n` sum to at
most a bound that does not depend on `n`. -/
theorem exists_sum_exp_implBias_le (θα : ℝ) {θβ ε : ℝ} (hβ : 0 < θβ) (hε : 0 ≤ ε) :
    ∃ M : ℝ, ∀ n : ℕ, ∑ r : Fin n, Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ))) ≤ M := by
  obtain ⟨K, hK⟩ : ∃ K : ℕ, 2 ≤ θβ * K := ⟨⌈2 / θβ⌉₊, by
    have h := Nat.le_ceil (2 / θβ)
    rw [div_le_iff₀ hβ] at h
    linarith⟩
  have hc := Real.exp_pos θα
  set A := max 1 (4 * K.factorial / Real.exp θα ^ K) with hA
  have hterm (r : ℕ) : Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ))) ≤
      A * ((r : ℝ) + 1) ^ (-(2 : ℝ)) := by
    rw [implBias_zero_neg θα θβ ε (Nat.cast_nonneg r)]
    rcases Nat.eq_zero_or_pos r with rfl | hr
    · simp only [Nat.cast_zero, zero_add, Real.one_rpow, mul_one]
      refine (Real.exp_le_one_iff.2 (neg_nonpos.2 ?_)).trans (le_max_left _ _)
      exact mul_nonneg hc.le (Real.rpow_nonneg hε _)
    have hr1 : (1 : ℝ) ≤ r := by exact_mod_cast hr
    set x := ((r : ℝ) + 1) / 2 with hx
    have hx1 : 1 ≤ x := by rw [hx]; linarith
    set y := Real.exp θα * x ^ θβ with hy
    have hy0 : 0 < y := mul_pos hc (Real.rpow_pos_of_pos (by linarith) _)
    have hyr : y ≤ Real.exp θα * ((r : ℝ) + ε) ^ θβ :=
      mul_le_mul_of_nonneg_left (Real.rpow_le_rpow (by linarith) (by rw [hx]; linarith) hβ.le)
        hc.le
    have hfac := Real.pow_div_factorial_le_exp y hy0.le K
    have hpow : Real.exp θα ^ K * x ^ (2 : ℝ) ≤ y ^ K := by
      rw [hy, mul_pow, ← Real.rpow_natCast (x ^ θβ) K, ← Real.rpow_mul (by linarith)]
      exact mul_le_mul_of_nonneg_left (Real.rpow_le_rpow_of_exponent_le hx1 hK) (by positivity)
    have hx2 : x ^ (2 : ℝ) = ((r : ℝ) + 1) ^ (2 : ℝ) / 4 := by
      rw [hx, Real.div_rpow (by positivity) (by norm_num)]
      norm_num
    have hQ : 0 < ((r : ℝ) + 1) ^ (2 : ℝ) := Real.rpow_pos_of_pos (by positivity) _
    have hF : (0 : ℝ) < K.factorial := by exact_mod_cast K.factorial_pos
    have hE : Real.exp θα ^ K * ((r : ℝ) + 1) ^ (2 : ℝ) ≤ 4 * K.factorial * Real.exp y := by
      rw [div_le_iff₀ hF] at hfac
      rw [hx2] at hpow
      nlinarith
    calc Real.exp (-(Real.exp θα * ((r : ℝ) + ε) ^ θβ)) ≤ Real.exp (-y) :=
          Real.exp_le_exp.2 (neg_le_neg hyr)
      _ = 1 / Real.exp y := by rw [Real.exp_neg, one_div]
      _ ≤ 4 * K.factorial / (Real.exp θα ^ K * ((r : ℝ) + 1) ^ (2 : ℝ)) := by
          rw [div_le_div_iff₀ (Real.exp_pos y) (by positivity)]
          linarith
      _ = 4 * K.factorial / Real.exp θα ^ K * ((r : ℝ) + 1) ^ (-(2 : ℝ)) := by
          rw [Real.rpow_neg (by positivity), div_mul_eq_div_div, div_eq_mul_inv _ (_ ^ (2 : ℝ))]
      _ ≤ A * ((r : ℝ) + 1) ^ (-(2 : ℝ)) :=
          mul_le_mul_of_nonneg_right (le_max_right _ _) (by positivity)
  refine ⟨A * 2, fun n => ?_⟩
  calc ∑ r : Fin n, Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ)))
      ≤ ∑ r : Fin n, A * ((r : ℝ) + 1) ^ (-(2 : ℝ)) := sum_le_sum fun r _ => hterm r
    _ = A * ∑ r : Fin n, ((r : ℝ) + 1) ^ (-(2 : ℝ)) := (mul_sum _ _ _).symm
    _ ≤ A * 2 := mul_le_mul_of_nonneg_left
        ((sum_rpow_neg_le_of_one_lt one_lt_two n).trans_eq (by norm_num))
        ((zero_le_one.trans (le_max_left _ _)))

/-- At `θ_α = 0`, `θ_β = 1/2` and `ε = 10^{−5}`. -/
example := exists_sum_exp_implBias_le 0 (θβ := 1 / 2) (ε := 1 / 100000) (by norm_num) (by norm_num)

/-- **A local head recalls within its window** (`review/flat_spectrum.md`, PB8): for `θ_β > 0`,
`ε ≥ 0` and `κ < s g`, `s ≥ 0`, for every `ε' > 0`, from some length on, under BAM's bias every key
at a distance `R` with `e^{θ_α} (R + ε)^{θ_β} ≤ κ log n` whose raw score leads every other by `g`
is answered within `ε'`, for values bounded by `C`. -/
theorem exists_forall_implBias_recall {s g C κ ε' θα θβ ε : ℝ} (hs : 0 ≤ s) (hβ : 0 < θβ)
    (hε : 0 ≤ ε) (hκ : κ < s * g) (hε' : 0 < ε') :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z v : Fin n → ℝ) (R : Fin n),
      Real.exp θα * ((R : ℝ) + ε) ^ θβ ≤ κ * Real.log n →
      (∀ r, r ≠ R → z r + g ≤ z R) → (∀ r, |v r| ≤ C) →
        |v R - biasedLogScaleAttention s (fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ)))
          z v| ≤ ε' := by
  obtain ⟨M, hM⟩ := exists_sum_exp_implBias_le θα hβ hε
  set a := s * g - κ with ha
  have hlim : Tendsto (fun n : ℕ => 2 * C * M * (n : ℝ) ^ (-a)) atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith : 0 < a)).comp
      tendsto_natCast_atTop_atTop).const_mul (2 * C * M)
    rw [mul_zero] at h
    exact h
  obtain ⟨N, hN⟩ := eventually_atTop.1 (hlim.eventually (ge_mem_nhds hε'))
  refine ⟨N + 1, fun n hn z v R hR hz hv => ?_⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv R)
  set b := fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ)) with hb
  have h := abs_sub_biasedLogScaleAttention_le hs b hz hv
  rw [Fintype.card_fin] at h
  refine h.trans (le_trans ?_ (hN n (by omega)))
  have hbR : -b R ≤ Real.log n * κ := by
    rw [hb]
    dsimp only
    rw [implBias_zero_neg θα θβ ε (Nat.cast_nonneg _), neg_neg, mul_comm (Real.log n)]
    exact hR
  have hsum : ∑ r ∈ univ.erase R, Real.exp (b r - b R) ≤ (n : ℝ) ^ κ * M := by
    calc ∑ r ∈ univ.erase R, Real.exp (b r - b R)
        = Real.exp (-b R) * ∑ r ∈ univ.erase R, Real.exp (b r) := by
          rw [mul_sum]
          exact sum_congr rfl fun r _ => by rw [← Real.exp_add]; ring_nf
      _ ≤ Real.exp (-b R) * ∑ r, Real.exp (b r) :=
          mul_le_mul_of_nonneg_left (sum_le_sum_of_subset_of_nonneg (erase_subset _ _)
            fun _ _ _ => (Real.exp_pos _).le) (Real.exp_pos _).le
      _ ≤ (n : ℝ) ^ κ * M := by
          refine mul_le_mul ?_ (hM n) (sum_nonneg fun _ _ => (Real.exp_pos _).le) (by positivity)
          rw [Real.rpow_def_of_pos hn0]
          exact Real.exp_le_exp.2 hbR
  calc 2 * C * ((n : ℝ) ^ (-(s * g)) * ∑ r ∈ univ.erase R, Real.exp (b r - b R))
      ≤ 2 * C * ((n : ℝ) ^ (-(s * g)) * ((n : ℝ) ^ κ * M)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hsum (by positivity)) (by linarith)
    _ = 2 * C * M * (n : ℝ) ^ (-a) := by
        rw [ha, show -(s * g - κ) = -(s * g) + κ by ring, Real.rpow_add hn0]
        ring

/-- At `θ_α = 0`, `θ_β = 2`, `ε = 10^{−5}` and `s = g = 1`: the keys with
`(R + ε)^2 ≤ (log n) / 2`. -/
example := exists_forall_implBias_recall (s := 1) (g := 1) (C := 1) (κ := 1 / 2) (ε' := 1 / 10)
  (θα := 0) (θβ := 2) (ε := 1 / 100000) zero_le_one two_pos (by norm_num) (by norm_num)
  (by norm_num)

/-- **A local head loses the keys outside its window** (`review/flat_spectrum.md`, PB8): with raw
scores bounded by `B` and `κ > 2 s B`, `s ≥ 0`, for every `ε' < 2`, from some length on, under
BAM's bias every key at a distance `R` with `e^{θ_α} (R + ε)^{θ_β} ≥ κ log n` has its indicator
dictionary answered with an error above `ε'`, at every `θ_α`, `θ_β` and `ε`. -/
theorem exists_lt_implBias_error_of_log_le {s B κ ε' θα θβ ε : ℝ} (hs : 0 ≤ s)
    (hκ : 2 * s * B < κ) (hε' : ε' < 2) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (R : Fin n),
      κ * Real.log n ≤ Real.exp θα * ((R : ℝ) + ε) ^ θβ → (∀ r, |z r| ≤ B) →
        ε' < 1 - biasedLogScaleAttention s (fun r : Fin n => BAM.implBias 0 θα θβ ε (-(r : ℝ))) z
          (fun r => if r = R then 1 else -1) := by
  set D := Real.exp θα * max (ε ^ θβ) ((1 + ε) ^ θβ) with hD
  obtain ⟨N, hN⟩ := exists_lt_one_sub_of_le_sum_erase hs (Real.exp_pos (-D)) hκ hε'
  refine ⟨max N 2, fun n hn z R hR hz => hN n (le_of_max_le_left hn) _ z R ?_ hz⟩
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hkey : (n : ℝ) ^ κ * Real.exp (BAM.implBias 0 θα θβ ε (-(R : ℝ))) ≤ 1 := by
    rw [implBias_zero_neg θα θβ ε (Nat.cast_nonneg _), Real.rpow_def_of_pos hn0, ← Real.exp_add]
    exact Real.exp_le_one_iff.2 (by linarith [mul_comm (Real.log n) κ])
  have h1 (j : Fin n) (hj : (j : ℕ) ≤ 1) : -D ≤ BAM.implBias 0 θα θβ ε (-(j : ℝ)) := by
    rw [implBias_zero_neg θα θβ ε (Nat.cast_nonneg _), neg_le_neg_iff, hD]
    refine mul_le_mul_of_nonneg_left ?_ (Real.exp_pos _).le
    rcases Nat.le_one_iff_eq_zero_or_eq_one.1 hj with h | h <;> simp [h, add_comm]
  obtain ⟨j₁, hj₁, hb₁⟩ : ∃ j₁ : Fin n, j₁ ≠ R ∧ -D ≤ BAM.implBias 0 θα θβ ε (-(j₁ : ℝ)) := by
    by_cases hR0 : (R : ℕ) = 0
    · exact ⟨⟨1, by omega⟩, fun h => by simp [← h] at hR0, h1 _ le_rfl⟩
    · exact ⟨⟨0, by omega⟩, fun h => hR0 (by rw [← h]), h1 _ zero_le_one⟩
  calc Real.exp (-D) * (n : ℝ) ^ κ * Real.exp (BAM.implBias 0 θα θβ ε (-(R : ℝ)))
      ≤ Real.exp (-D) := by
        rw [mul_assoc]
        exact mul_le_of_le_one_right (Real.exp_pos _).le hkey
    _ ≤ Real.exp (BAM.implBias 0 θα θβ ε (-(j₁ : ℝ))) := Real.exp_le_exp.2 hb₁
    _ ≤ ∑ r ∈ univ.erase R, Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ))) :=
        single_le_sum (f := fun r : Fin n => Real.exp (BAM.implBias 0 θα θβ ε (-(r : ℝ))))
          (fun r _ => (Real.exp_pos _).le) (mem_erase.2 ⟨hj₁, mem_univ _⟩)

/-- At `θ_α = 0`, `θ_β = 2`, `ε = 10^{−5}`, `s = 1` and `B = 1/4`: the keys with
`(R + ε)^2 ≥ log n`. -/
example := exists_lt_implBias_error_of_log_le (s := 1) (B := 1 / 4) (κ := 1) (ε' := 1) (θα := 0)
  (θβ := 2) (ε := 1 / 100000) zero_le_one (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
