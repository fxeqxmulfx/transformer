module

public import Transformer.MRCR.PhraseKeyRotary

/-!
# RoPE with the distance clamped at a window answers spelled asks

In no paper. ReRoPE (Su, 2023), which arXiv:2309.00071v3, §2.3, cites without its formula,
extends RoPE by modifying the attention: it clamps the relative position at a window.
`reropeLogit` is our reading of it: the dot product divided by `c` plus a RoPE score of the
distance `min (i − j) W`. From the window on the logit no longer depends on the distance
(`reropeLogit_of_le`), so it converges, which no RoPE score does unless it is constant
(`Transformer.MRCR.ropeScore_eq_of_tendsto`).

That is enough to read the token before. Take one frequency `θ` with `0 < θ` and
`θ (W + 2) ≤ π`, and the sine coefficient `−β / sin θ` for `β = b / c ≥ 3`. The score at the
clamped distance `t` is `−β sin(θ t) / sin θ`: `0` at `0` and `−β` at `1`. From `2` to `W` it is at
most `−2β cos θ ≤ −4β/3`, since `sin(θ t) ≥ sin(2θ)` there (`sin_two_mul_le_sin`) and
`cos θ ≥ 2/3` (`two_thirds_le_cos`). So `ReadsPrev` holds at every window `W ≥ 2`
(`readsPrev_rerope`), and five layers answer spelled asks at every length, the logit divided by
`1`, `√d` or `d` (`exists_rerope_phraseAnswer`), under SSMax with `s log 2 ≥ 1` as well
(`exists_ssmax_rerope_phraseAnswer`).

The frequency is one of the head's own, the others get the coefficient `0`. A grid of
frequencies serves a window up to `π / θ_min − 2`; for RoPE's grid `10000^{−2m/d}` with `d = 128`
the slowest frequency is `10000^{−126/128} ≈ 1.2 · 10^{−4}` (a hand computation), about
`2.7 · 10^4` tokens. As for ALiBi's slope, the coefficients are parameters of the logit; that
the constant pairs of a rounded query and key carry them is not modelled.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP Transformer.CRASP.Grams

variable {K F N V R : ℕ}

/-- **ReRoPE's logit**, as read here: the dot product `x` divided by `c` plus the RoPE score of the
distance `i − j` clamped at the window `W`. -/
noncomputable def reropeLogit (c : ℝ) (θ a b : Fin K → ℝ) (W i j : ℕ) (x : ℝ) : ℝ :=
  x / c + ropeScore θ a b (min (i - j) W : ℕ)

/-- From the window on, the logit does not depend on the distance. -/
theorem reropeLogit_of_le (c : ℝ) (θ a b : Fin K → ℝ) {W i j : ℕ} (h : W ≤ i - j) (x : ℝ) :
    reropeLogit c θ a b W i j x = x / c + ropeScore θ a b W := by
  rw [reropeLogit, min_eq_right h]

/-- The hypothesis of `reropeLogit_of_le` holds at the distance `5` for the window `2`. -/
example := reropeLogit_of_le 1 (fun _ : Fin 1 => 1) 0 0 (W := 2) (i := 7) (j := 2) (by norm_num) 0

/-- The score of one sine at the frequency `θ m`. -/
theorem ropeScore_single (θ : Fin K → ℝ) (m : Fin K) (B t : ℝ) :
    ropeScore θ 0 (Pi.single m B) t = B * Real.sin (θ m * t) := by
  simp [ropeScore, Pi.single_apply]

/-- From `2` to `W`, `sin(θ t)` stays at least `sin(2θ)` when `θ (W + 2) ≤ π`. -/
theorem sin_two_mul_le_sin {θ : ℝ} (hθ : 0 < θ) {W t : ℕ} (hθW : θ * (W + 2) ≤ Real.pi)
    (h2 : 2 ≤ t) (hW : t ≤ W) : Real.sin (2 * θ) ≤ Real.sin (θ * t) := by
  have ht : (2 : ℝ) ≤ t := by exact_mod_cast h2
  have htW : (t : ℝ) ≤ W := by exact_mod_cast hW
  rcases le_total (θ * t) (Real.pi / 2) with h | h
  · exact Real.sin_le_sin_of_le_of_le_pi_div_two (by nlinarith [Real.pi_pos]) h (by nlinarith)
  · rw [← Real.sin_pi_sub (θ * t)]
    exact Real.sin_le_sin_of_le_of_le_pi_div_two (by nlinarith [Real.pi_pos]) (by linarith)
      (by nlinarith)

/-- The hypotheses of `sin_two_mul_le_sin` hold for `θ = π/4` and `t = W = 2`. -/
example := sin_two_mul_le_sin (θ := Real.pi / 4) (by positivity) (W := 2) (t := 2)
  (by push_cast; linarith) le_rfl le_rfl

/-- Below `π/4` the cosine is at least `2/3`. -/
theorem two_thirds_le_cos {θ : ℝ} (hθ : 0 ≤ θ) (h : θ ≤ Real.pi / 4) : 2 / 3 ≤ Real.cos θ := by
  have h1 := Real.one_sub_sq_div_two_le_cos (x := θ)
  have h2 := Real.pi_lt_d2
  nlinarith

/-- The hypotheses of `two_thirds_le_cos` hold at `0`. -/
example := two_thirds_le_cos le_rfl (by positivity)

/-- **A clamped RoPE head reads the token before**: the window `W ≥ 2`, one frequency `θ m` with
`0 < θ m` and `θ m (W + 2) ≤ π`, its sine coefficient `−(b / c) / sin(θ m)` and the dot product
`b ≥ 3c`. -/
theorem readsPrev_rerope {c : ℝ} (hc : 0 < c) {b : ℕ} (hb : 3 * c ≤ b) {θ : Fin K → ℝ}
    {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 2 ≤ W) (hθW : θ m * (W + 2) ≤ Real.pi) :
    ReadsPrev (reropeLogit c θ 0 (Pi.single m (-(b / c) / Real.sin (θ m))) W) b := by
  have hW' : (2 : ℝ) ≤ W := by exact_mod_cast hW
  have hπ : θ m ≤ Real.pi / 4 := by nlinarith
  have hS : 0 < Real.sin (θ m) := Real.sin_pos_of_pos_of_lt_pi hθ (by linarith [Real.pi_pos])
  have hβ : 3 ≤ (b : ℝ) / c := by rw [le_div_iff₀ hc]; linarith
  have hcos := two_thirds_le_cos hθ.le hπ
  have hB : -((b : ℝ) / c) / Real.sin (θ m) ≤ 0 :=
    div_nonpos_of_nonpos_of_nonneg (by linarith) hS.le
  refine ⟨fun i => ?_, fun i hi => ?_, fun i hi => ?_, fun i j hij => ?_,
    fun i => Or.inl fun j _ => ?_⟩
  · simp [reropeLogit, ropeScore_single]
  · have h1 : min (i - (i - 1)) W = 1 := by omega
    simp only [reropeLogit, h1, ropeScore_single, Nat.cast_one, mul_one,
      div_mul_cancel₀ _ hS.ne']
    ring
  · have h1 : min (i - (i - 1)) W = 1 := by omega
    simp only [reropeLogit, h1, ropeScore_single, Nat.cast_one, mul_one,
      div_mul_cancel₀ _ hS.ne', zero_div, zero_add]
    linarith
  · have hs := sin_two_mul_le_sin hθ hθW (by omega : 2 ≤ min (i - j) W) (min_le_right _ _)
    rw [Real.sin_two_mul] at hs
    have key : -((b : ℝ) / c) / Real.sin (θ m) * Real.sin (θ m * (min (i - j) W : ℕ)) ≤
        -(2 * (b / c) * Real.cos (θ m)) :=
      calc _ ≤ -((b : ℝ) / c) / Real.sin (θ m) * (2 * Real.sin (θ m) * Real.cos (θ m)) :=
            mul_le_mul_of_nonpos_left hs hB
        _ = _ := by rw [div_mul_eq_mul_div, div_eq_iff hS.ne']; ring
    simp only [reropeLogit, ropeScore_single, zero_div, zero_add]
    constructor <;> nlinarith
  · have hle : ((min (i - j) W : ℕ) : ℝ) ≤ W := by exact_mod_cast min_le_right _ _
    have hsin : 0 ≤ Real.sin (θ m * (min (i - j) W : ℕ)) :=
      Real.sin_nonneg_of_nonneg_of_le_pi (by positivity) (by nlinarith)
    have h1 : -1 / c < 0 := div_neg_of_neg_of_pos (by norm_num) hc
    simp only [reropeLogit, ropeScore_single]
    nlinarith [mul_nonpos_of_nonpos_of_nonneg hB hsin]

/-- The hypotheses of `readsPrev_rerope` hold for `θ = π/4`, the window `2` and the dot product
`3`. -/
example := readsPrev_rerope (c := 1) one_pos (b := 3) (by norm_num)
  (θ := fun _ : Fin 1 => Real.pi / 4) (m := 0) (by positivity) (W := 2) le_rfl
  (by push_cast; linarith)

/-- **Some transformer with a clamped RoPE head answers spelled asks at every length**: five
layers, at every window `W ≥ 2` and every frequency `θ m` with `0 < θ m` and
`θ m (W + 2) ≤ π`, the logit divided by `c d`, with some sine coefficients. -/
theorem exists_rerope_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) {θ : Fin K → ℝ}
    {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 2 ≤ W) (hθW : θ m * (W + 2) ≤ Real.pi)
    (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5),
      T.logit = reropeLogit (c d) θ 0 b W ∧ T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => reropeLogit (c d) θ 0
      (Pi.single m (-(((3 * ⌈c d⌉₊ : ℕ) : ℝ) / c d) / Real.sin (θ m))) W)
    (fun d => 3 * ⌈c d⌉₊) (fun d hd => readsPrev_rerope (hc d hd)
      (by push_cast; linarith [Nat.le_ceil (c d)]) hθ hW hθW) v
  exact ⟨p, d, _, T, hT, hR⟩

/-- The hypotheses of `exists_rerope_phraseAnswer` hold for the divisor `√d`, `θ = π/4` and the
window `2`. -/
example := exists_rerope_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) (θ := fun _ : Fin 1 => Real.pi / 4)
  (m := 0) (by positivity) (W := 2) le_rfl (by push_cast; linarith) (0 : Fin 1)

/-- **SSMax and clamped RoPE**: SSMax with `s log 2 ≥ 1` (arXiv:2501.19399v1, `eq:ssmax`) of the
same logit, scaling the whole logit. -/
theorem exists_ssmax_rerope_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) {s : ℝ}
    (hs : 1 ≤ s * Real.log 2) {θ : Fin K → ℝ} {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 2 ≤ W)
    (hθW : θ m * (W + 2) ≤ Real.pi) (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5),
      T.logit = ssmaxLogit s (reropeLogit (c d) θ 0 b W) ∧
      T.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, T, hd, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d => ssmaxLogit s (reropeLogit (c d) θ 0
      (Pi.single m (-(((3 * ⌈c d⌉₊ : ℕ) : ℝ) / c d) / Real.sin (θ m))) W))
    (fun d => 3 * ⌈c d⌉₊) (fun d hd => readsPrev_ssmax hs (readsPrev_rerope (hc d hd)
      (by push_cast; linarith [Nat.le_ceil (c d)]) hθ hW hθW)) v
  exact ⟨p, d, _, T, hT, hR⟩

/-- The hypotheses of `exists_ssmax_rerope_phraseAnswer` hold for `s = 2`, the divisor `d`,
`θ = π/4` and the window `2`. -/
example := exists_ssmax_rerope_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => (d : ℝ))
  (fun _ hd => by exact_mod_cast hd) (s := 2) (by have := Real.log_two_gt_d9; linarith)
  (θ := fun _ : Fin 1 => Real.pi / 4) (m := 0) (by positivity) (W := 2) le_rfl
  (by push_cast; linarith) (0 : Fin 1)

end Transformer.MRCR
