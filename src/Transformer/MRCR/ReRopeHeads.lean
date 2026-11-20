module

public import Transformer.MRCR.PhraseKeyReRope
public import Transformer.RoPE.ReRoPE.Section4_Experiments
public import Transformer.Grokking.FlatSpectrum.Recall.LogScale

/-!
# Clamped RoPE heads under the real softmax: SSMax reads inside the window

In no paper. The rounded transformers of `Transformer.MRCR.PhraseKeyReRope` drop every key
whose logit is at most `−1`, at every length, so there the softmax and SSMax behave alike. With
the real softmax they do not (`Transformer.MRCR.ReRopeFading` has the plain softmax). A head at
the query position `i` scores the `n = i + 1` keys `j ≤ i` by `ropeScore θ a b (min (i − j) W)`,
the positional part of `reropeLogit`.

* `ropeScore_peak`: one frequency with the coefficients `A cos(θ_m k)`, `A sin(θ_m k)` scores the
  distance `t` by `A cos(θ_m (t − k))`.
* `reropePeak_gap`: for `k < W` and `θ_m W ≤ π` every other distance, the clamped ones too,
  trails by `g = A (1 − cos θ_m)`.
* `abs_logScaleAttention_rerope_sub_le`: at the scale `s log n` of SSMax the head returns the
  value at the distance `k` within `2 C i (i + 1)^{−s g}`, so for `s g > 1` at every length to
  every accuracy (`exists_forall_rerope_recall`).
* `softmaxAttention_logScale_eq`: from the training length `T` on, Su's scale `max(1, log_T n)`
  (kexue.fm/archives/9708, `Transformer.RoPE.ReRoPE.logScale`) is SSMax with `s = 1 / log T`. So
  ReRoPE with Su's scale reads inside the window at every length when `g > log T`
  (`exists_forall_logScale_rerope_recall`).
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Filter Topology Transformer.Grokking.FlatSpectrum.Recall
open Transformer.RoPE.ReRoPE (logScale)

variable {K : ℕ}

/-- The coefficients `A cos(θ_m k)`, `A sin(θ_m k)` of one frequency put its peak at `k`:
the score is `A cos(θ_m (t − k))`. -/
theorem ropeScore_peak (θ : Fin K → ℝ) (m : Fin K) (A k t : ℝ) :
    ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k))) t =
      A * Real.cos (θ m * (t - k)) := by
  rw [ropeScore, sum_eq_single m (fun l _ hl => by simp [hl]) (by simp)]
  simp only [Pi.single_eq_same]
  rw [mul_sub, Real.cos_sub]
  ring

/-- At least `1` and at most `π / θ` away from the peak, the cosine is at most `cos θ`. -/
theorem cos_mul_le_cos {θ d : ℝ} (hθ : 0 ≤ θ) (h1 : 1 ≤ |d|) (hπ : θ * |d| ≤ Real.pi) :
    Real.cos (θ * d) ≤ Real.cos θ := by
  rw [← Real.cos_abs (θ * d), abs_mul, abs_of_nonneg hθ]
  exact Real.cos_le_cos_of_nonneg_of_le_pi hθ hπ (le_mul_of_one_le_right hθ h1)

/-- The hypotheses of `cos_mul_le_cos` hold for `θ = 1` and `d = 2`. -/
example : Real.cos (1 * 2) ≤ Real.cos 1 :=
  cos_mul_le_cos zero_le_one (by norm_num) (by norm_num; linarith [Real.pi_gt_three])

/-- **A clamped RoPE head peaks inside the window**: with the coefficients of `ropeScore_peak`
at a distance `k < W`, `0 < θ_m` and `θ_m W ≤ π`, every other distance `t`, clamped at `W`,
scores at least `A (1 − cos θ_m)` below `k`. -/
theorem reropePeak_gap (θ : Fin K → ℝ) (m : Fin K) {A : ℝ} (hA : 0 ≤ A) (hθ : 0 < θ m)
    {k W : ℕ} (hk : k < W) (hθW : θ m * W ≤ Real.pi) {t : ℕ} (ht : t ≠ k) :
    ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
        (min t W : ℕ) + A * (1 - Real.cos (θ m)) ≤
      ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
        (min k W : ℕ) := by
  rw [ropeScore_peak, ropeScore_peak, min_eq_left hk.le, sub_self, mul_zero, Real.cos_zero]
  have hne : min t W ≠ k := by omega
  have hmW : ((min t W : ℕ) : ℝ) ≤ W := by exact_mod_cast min_le_right t W
  have hkW : (k : ℝ) ≤ W := by exact_mod_cast hk.le
  have h0 : (0 : ℝ) ≤ (min t W : ℕ) := Nat.cast_nonneg _
  have h1 : (1 : ℝ) ≤ |((min t W : ℕ) : ℝ) - k| := by
    rcases Nat.lt_or_gt_of_ne hne with h | h
    · have : ((min t W : ℕ) : ℝ) + 1 ≤ k := by exact_mod_cast h
      rw [abs_sub_comm, abs_of_nonneg (by linarith)]
      linarith
    · have : (k : ℝ) + 1 ≤ (min t W : ℕ) := by exact_mod_cast h
      rw [abs_of_nonneg (by linarith)]
      linarith
  have h2 : |((min t W : ℕ) : ℝ) - k| ≤ W := abs_le.2 ⟨by linarith, by linarith⟩
  have hcos := cos_mul_le_cos hθ.le h1 ((mul_le_mul_of_nonneg_left h2 hθ.le).trans hθW)
  nlinarith [mul_le_mul_of_nonneg_left hcos hA]

/-- The hypotheses of `reropePeak_gap` hold for `θ = π/4`, `A = 1`, the window `4`, the peak `1`
and the distance `3`. -/
example := reropePeak_gap (fun _ : Fin 1 => Real.pi / 4) 0 zero_le_one (by positivity)
  (k := 1) (W := 4) (by norm_num) (by push_cast; linarith) (t := 3) (by norm_num)

/-- **SSMax reads inside the window**: at the query position `i ≥ k`, the head of
`reropePeak_gap` at the scale `s log n` over the `n = i + 1` keys `j ≤ i` returns the value at the
distance `k < W` within `2 C i (i + 1)^{−s A (1 − cos θ_m)}`, for values bounded by `C`. -/
theorem abs_logScaleAttention_rerope_sub_le (θ : Fin K → ℝ) (m : Fin K) {A s C : ℝ}
    (hA : 0 ≤ A) (hs : 0 ≤ s) (hθ : 0 < θ m) {k W : ℕ} (hk : k < W)
    (hθW : θ m * W ≤ Real.pi) {i : ℕ} (hki : k ≤ i) {v : Fin (i + 1) → ℝ}
    (hv : ∀ j, |v j| ≤ C) :
    |logScaleAttention s (fun j : Fin (i + 1) => ropeScore θ
        (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
        (min (i - j) W : ℕ)) v - v ⟨i - k, by omega⟩| ≤
      2 * C * i * ((i : ℝ) + 1) ^ (-(s * (A * (1 - Real.cos (θ m))))) := by
  have hgap : ∀ j : Fin (i + 1), j ≠ ⟨i - k, by omega⟩ →
      ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
          (min (i - j) W : ℕ) + A * (1 - Real.cos (θ m)) ≤
        ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
          (min (i - (⟨i - k, by omega⟩ : Fin (i + 1))) W : ℕ) := fun j hj => by
    have hik : i - (i - k) = k := Nat.sub_sub_self hki
    have hjk : i - (j : ℕ) ≠ k := fun h' => hj (Fin.ext (show (j : ℕ) = i - k by omega))
    simpa only [Fin.val_mk, hik] using reropePeak_gap θ m hA hθ hk hθW hjk
  simpa using abs_logScaleAttention_sub_le hs hgap hv

/-- The hypotheses of `abs_logScaleAttention_rerope_sub_le` hold at `s = 2`, the window `4`, the
peak `1` and the query position `3`. -/
example := abs_logScaleAttention_rerope_sub_le (fun _ : Fin 1 => Real.pi / 4) 0 (A := 1) (s := 2)
  (C := 1) zero_le_one zero_le_two (by positivity) (k := 1) (W := 4) (by norm_num)
  (by push_cast; linarith) (i := 3) (by norm_num) (v := fun _ => 1) fun _ => by norm_num

/-- **SSMax reads inside the window at every length**: for `s A (1 − cos θ_m) > 1` the head of
`abs_logScaleAttention_rerope_sub_le` returns the value at the distance `k < W`, for values
bounded by `C`, to every accuracy `ε > 0` from some query position on. -/
theorem exists_forall_rerope_recall (θ : Fin K → ℝ) (m : Fin K) {A s C ε : ℝ} (hA : 0 ≤ A)
    (hs : 0 ≤ s) (hθ : 0 < θ m) {k W : ℕ} (hk : k < W) (hθW : θ m * W ≤ Real.pi)
    (hsg : 1 < s * (A * (1 - Real.cos (θ m)))) (hε : 0 < ε) :
    ∃ M : ℕ, ∀ i ≥ M, ∀ v : Fin (i + 1) → ℝ, (∀ j, |v j| ≤ C) →
      |logScaleAttention s (fun j : Fin (i + 1) => ropeScore θ
          (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
          (min (i - j) W : ℕ)) v - v ⟨i - k, by omega⟩| ≤ ε := by
  have ht : Tendsto (fun i : ℕ => 2 * C * ((i : ℝ) + 1) ^ (1 - s * (A * (1 - Real.cos (θ m)))))
      atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith :
      0 < s * (A * (1 - Real.cos (θ m))) - 1)).comp
      (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)).const_mul (2 * C)
    rw [mul_zero] at h
    exact Tendsto.congr (fun i => by simp only [Function.comp_apply, neg_sub]) h
  obtain ⟨M, hM⟩ := eventually_atTop.1 (ht.eventually (ge_mem_nhds hε))
  refine ⟨max M k, fun i hi v hv => ?_⟩
  have hx : (0 : ℝ) < (i : ℝ) + 1 := by positivity
  have hC : 0 ≤ C := (abs_nonneg _).trans (hv 0)
  calc _ ≤ 2 * C * i * ((i : ℝ) + 1) ^ (-(s * (A * (1 - Real.cos (θ m))))) :=
        abs_logScaleAttention_rerope_sub_le θ m hA hs hθ hk hθW (le_of_max_le_right hi) hv
    _ ≤ 2 * C * ((i : ℝ) + 1) * ((i : ℝ) + 1) ^ (-(s * (A * (1 - Real.cos (θ m))))) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (by linarith) (by linarith))
          (Real.rpow_nonneg hx.le _)
    _ = 2 * C * ((i : ℝ) + 1) ^ (1 - s * (A * (1 - Real.cos (θ m)))) := by
        rw [Real.rpow_sub hx, Real.rpow_one, Real.rpow_neg hx.le]
        ring
    _ ≤ ε := hM i (le_of_max_le_left hi)

/-- The hypotheses of `exists_forall_rerope_recall` hold at `s = 2`, `θ = π/2`, `A = 1`, the window
`2` and the peak `1`. -/
example := exists_forall_rerope_recall (fun _ : Fin 1 => Real.pi / 2) 0 (A := 1) (s := 2) (C := 1)
  (ε := 1 / 10) zero_le_one zero_le_two (by positivity) (k := 1) (W := 2) (by norm_num)
  (by push_cast; linarith) (by norm_num) (by norm_num)

/-- **Su's scale is SSMax from the training length on**: at the query position `i` with
`1 < T ≤ i + 1`, attention with the logit multiplied by `max(1, log_T (i + 1))` is attention at
the scale `s log n` of SSMax with `s = 1 / log T`. -/
theorem softmaxAttention_logScale_eq {T : ℝ} (hT : 1 < T) {i : ℕ} (hTi : T ≤ (i : ℝ) + 1)
    (z v : Fin (i + 1) → ℝ) :
    softmaxAttention (fun j => logScale T ((i : ℝ) + 1) * z j) v =
      logScaleAttention (1 / Real.log T) z v := by
  have h1 : 1 ≤ Real.logb T ((i : ℝ) + 1) := by
    rw [Real.logb, one_le_div (Real.log_pos hT)]
    exact Real.log_le_log (by linarith) hTi
  rw [logScaleAttention, logScale, max_eq_right h1, Real.logb, Fintype.card_fin]
  push_cast
  exact congrArg (softmaxAttention · v) (funext fun j => by ring)

/-- The hypotheses of `softmaxAttention_logScale_eq` hold for Su's training length `4096` at the
query position `4095`. -/
example := softmaxAttention_logScale_eq (T := 4096) (by norm_num) (i := 4095) (by norm_num)
  (fun _ => 0) fun _ => 0

/-- **ReRoPE with Su's scale reads inside the window at every length** when the gap
`A (1 − cos θ_m)` exceeds `log T`: the head of `reropePeak_gap`, its logit multiplied by
`max(1, log_T (i + 1))` with `T > 1`, returns the value at the distance `k < W`, for values
bounded by `C`, to every accuracy `ε > 0` from some query position on. -/
theorem exists_forall_logScale_rerope_recall (θ : Fin K → ℝ) (m : Fin K) {A T C ε : ℝ}
    (hA : 0 ≤ A) (hT : 1 < T) (hθ : 0 < θ m) {k W : ℕ} (hk : k < W)
    (hθW : θ m * W ≤ Real.pi) (hg : Real.log T < A * (1 - Real.cos (θ m))) (hε : 0 < ε) :
    ∃ M : ℕ, ∀ i ≥ M, ∀ v : Fin (i + 1) → ℝ, (∀ j, |v j| ≤ C) →
      |softmaxAttention (fun j : Fin (i + 1) => logScale T ((i : ℝ) + 1) * ropeScore θ
          (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
          (min (i - j) W : ℕ)) v - v ⟨i - k, by omega⟩| ≤ ε := by
  have hlogT := Real.log_pos hT
  obtain ⟨M, hM⟩ := exists_forall_rerope_recall θ m (C := C) hA (one_div_pos.2 hlogT).le hθ hk
    hθW (by rwa [one_div, inv_mul_eq_div, one_lt_div hlogT]) hε
  refine ⟨max M ⌈T⌉₊, fun i hi v hv => ?_⟩
  have hTi : T ≤ (i : ℝ) + 1 := by
    have : ((⌈T⌉₊ : ℕ) : ℝ) ≤ i := by exact_mod_cast le_of_max_le_right hi
    linarith [Nat.le_ceil T]
  have h := hM i (le_of_max_le_left hi) v hv
  rwa [← softmaxAttention_logScale_eq hT hTi] at h

/-- The hypotheses of `exists_forall_logScale_rerope_recall` hold for `T = 2`, `θ = π/2`, `A = 2`,
the window `2` and the peak `1`. -/
example := exists_forall_logScale_rerope_recall (fun _ : Fin 1 => Real.pi / 2) 0 (A := 2) (T := 2)
  (C := 1) (ε := 1 / 10) zero_le_two one_lt_two (by positivity) (k := 1) (W := 2) (by norm_num)
  (by push_cast; linarith)
  (by norm_num; linarith [Real.log_le_sub_one_of_pos (zero_lt_two : (0 : ℝ) < 2)]) (by norm_num)

end Transformer.MRCR
