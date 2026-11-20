module

public import Transformer.MRCR.ReRopeHeads

/-!
# ReRoPE slots under SSMax: one layer reads the last tokens exactly, at every length

In no paper. `Transformer.MRCR.ReRopeHeads` reads the token at a distance `k < W` with the real
softmax at the scale `s log n` of SSMax (arXiv:2501.19399v1, §2.2), from the query position `k`
on. Before that position the peak lies beyond the first token, and the head reads the first token
instead: towards the peak the scores rise by at least `g = A (1 − cos θ_m)` per step
(`cos_add_add_one_le`, `reropeEarly_gap`). So at every position `x` the head returns the value at
the position `x - k`, read as `0` while `x < k`, within `2 C x (x + 1)^{−s g}`
(`abs_slotHead_sub_le`).

For values in `{0, 1}` and `s g > 4` that error is below `1/4` at every position, and the
rounding `clampUnit`, a ReLU layer of width two (`clampUnit_eq_relu`), makes the reading exact:
the slot of the distance `k` at the position `x` is `1` iff the token at `x - k` is `a`
(`clampUnit_slotHead`). `Transformer.MRCR.ReRopeStyleKeys` reads MRCR-V2's keys off four such
slots.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall

variable {K : ℕ}

/-- `cos (x + y) + 1 ≤ cos x + cos y` for `x, y ≥ 0` with `x + y ≤ π`: a step of length `y` from
`x` lowers the cosine at least as much as the step from `0`. -/
theorem cos_add_add_one_le {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ Real.pi) :
    Real.cos (x + y) + 1 ≤ Real.cos x + Real.cos y := by
  obtain ⟨a, rfl⟩ : ∃ a, x = 2 * a := ⟨x / 2, by ring⟩
  obtain ⟨b, rfl⟩ : ∃ b, y = 2 * b := ⟨y / 2, by ring⟩
  have hsa : 0 ≤ Real.sin a :=
    Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith [Real.pi_pos])
  have hsb : 0 ≤ Real.sin b :=
    Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith [Real.pi_pos])
  have hc : 0 ≤ Real.cos a * Real.cos b - Real.sin a * Real.sin b := Real.cos_add a b ▸
    Real.cos_nonneg_of_mem_Icc (x := a + b) ⟨by linarith [Real.pi_pos], by linarith⟩
  rw [show 2 * a + 2 * b = 2 * (a + b) by ring, Real.cos_two_mul, Real.cos_two_mul,
    Real.cos_two_mul, Real.cos_add]
  have key : 2 * Real.cos a ^ 2 - 1 + (2 * Real.cos b ^ 2 - 1) -
      (2 * (Real.cos a * Real.cos b - Real.sin a * Real.sin b) ^ 2 - 1 + 1) =
      4 * (Real.sin a * Real.sin b * (Real.cos a * Real.cos b - Real.sin a * Real.sin b)) := by
    linear_combination (2 * Real.sin b ^ 2) * Real.sin_sq_add_cos_sq a +
      (2 * (1 - Real.cos a ^ 2)) * Real.sin_sq_add_cos_sq b
  nlinarith [mul_nonneg (mul_nonneg hsa hsb) hc]

/-- The hypotheses of `cos_add_add_one_le` hold for `x = y = 1`. -/
example : Real.cos (1 + 1) + 1 ≤ Real.cos 1 + Real.cos 1 :=
  cos_add_add_one_le zero_le_one zero_le_one (by linarith [Real.pi_gt_three])

/-- **Below the peak the scores rise**: at the query position `i < k`, with the coefficients of
`ropeScore_peak` at a distance `k < W` and `θ_m W ≤ π`, the distance `i` of the first token scores
at least `A (1 − cos θ_m)` above every distance `d < i`. -/
theorem reropeEarly_gap (θ : Fin K → ℝ) (m : Fin K) {A : ℝ} (hA : 0 ≤ A) (hθ : 0 < θ m)
    {k W : ℕ} (hk : k < W) (hθW : θ m * W ≤ Real.pi) {i d : ℕ} (hik : i < k) (hd : d < i) :
    ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
        (min d W : ℕ) + A * (1 - Real.cos (θ m)) ≤
      ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
        (min i W : ℕ) := by
  rw [ropeScore_peak, ropeScore_peak, min_eq_left (by omega : d ≤ W),
    min_eq_left (by omega : i ≤ W)]
  have hdi : (d : ℝ) + 1 ≤ i := by exact_mod_cast hd
  have hik' : (i : ℝ) + 1 ≤ k := by exact_mod_cast hik
  have hkW : (k : ℝ) ≤ W := by exact_mod_cast hk.le
  have hd0 : (0 : ℝ) ≤ d := d.cast_nonneg
  have h1 := cos_add_add_one_le (x := θ m * (k - i)) (y := θ m) (by nlinarith) hθ.le
    (by nlinarith)
  have h2 : Real.cos (θ m * (d - k)) ≤ Real.cos (θ m * (k - i) + θ m) := by
    rw [← Real.cos_neg, show -(θ m * (d - k)) = θ m * (k - d) by ring]
    exact Real.cos_le_cos_of_nonneg_of_le_pi (by nlinarith) (by nlinarith) (by nlinarith)
  rw [show θ m * ((i : ℝ) - k) = -(θ m * (k - i)) by ring, Real.cos_neg]
  nlinarith [mul_le_mul_of_nonneg_left h2 hA, mul_le_mul_of_nonneg_left h1 hA]

/-- The hypotheses of `reropeEarly_gap` hold for `θ = π/4`, `A = 1`, the window `4`, the peak `3`,
the query position `2` and the distance `1`. -/
example := reropeEarly_gap (fun _ : Fin 1 => Real.pi / 4) 0 zero_le_one (by positivity) (k := 3)
  (W := 4) (by norm_num) (by push_cast; linarith) (i := 2) (d := 1) (by norm_num) one_lt_two

/-- **Before the peak the head reads the first token**: at the query position `i < k` the head of
`reropeEarly_gap` at the scale `s log n` returns the value at the position `0` within
`2 C i (i + 1)^{−s A (1 − cos θ_m)}`, for values bounded by `C`. -/
theorem abs_logScaleAttention_rerope_early_sub_le (θ : Fin K → ℝ) (m : Fin K) {A s C : ℝ}
    (hA : 0 ≤ A) (hs : 0 ≤ s) (hθ : 0 < θ m) {k W : ℕ} (hk : k < W)
    (hθW : θ m * W ≤ Real.pi) {i : ℕ} (hik : i < k) {v : Fin (i + 1) → ℝ}
    (hv : ∀ j, |v j| ≤ C) :
    |logScaleAttention s (fun j : Fin (i + 1) => ropeScore θ
        (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
        (min (i - j) W : ℕ)) v - v 0| ≤
      2 * C * i * ((i : ℝ) + 1) ^ (-(s * (A * (1 - Real.cos (θ m))))) := by
  have hgap : ∀ j : Fin (i + 1), j ≠ 0 →
      ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
          (min (i - j) W : ℕ) + A * (1 - Real.cos (θ m)) ≤
        ropeScore θ (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
          (min (i - ((0 : Fin (i + 1)) : ℕ)) W : ℕ) := fun j hj => by
    have hj0 : 0 < (j : ℕ) := Nat.pos_of_ne_zero fun h => hj (Fin.ext (by simp [h]))
    simpa only [Fin.val_zero, Nat.sub_zero] using
      reropeEarly_gap θ m hA hθ hk hθW hik (show i - j < i by omega)
  simpa using abs_logScaleAttention_sub_le hs hgap hv

/-- The hypotheses of `abs_logScaleAttention_rerope_early_sub_le` hold at `s = 2`, the window `4`,
the peak `3` and the query position `2`. -/
example := abs_logScaleAttention_rerope_early_sub_le (fun _ : Fin 1 => Real.pi / 4) 0 (A := 1)
  (s := 2) (C := 1) zero_le_one zero_le_two (by positivity) (k := 3) (W := 4) (by norm_num)
  (by push_cast; linarith) (i := 2) (by norm_num) (v := fun _ => 1) fun _ => by norm_num

/-- **The slot head of the distance `k`**: at the position `x` of `w`, attention at the scale
`s log n` over the positions `j ≤ x`, scored by ReRoPE's clamped score `min (x − j) W` with its
peak at `k` (`ropeScore_peak`), reading `1` at the positions that hold `a` and `0` elsewhere. -/
noncomputable def slotHead (s A : ℝ) (θ : Fin K → ℝ) (m : Fin K) (W : ℕ) {α : Type*}
    [DecidableEq α] (w : List α) (x k : ℕ) (a : α) : ℝ :=
  logScaleAttention s (fun j : Fin (x + 1) => ropeScore θ
      (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
      (min (x - j) W : ℕ)) fun j => if w[(j : ℕ)]? = some a then 1 else 0

/-- **The slot head reads the position `x - k`** at every position `x`, the first token while
`x < k`: within `2 x (x + 1)^{−s A (1 − cos θ_m)}`. -/
theorem abs_slotHead_sub_le {s A : ℝ} (hA : 0 ≤ A) (hs : 0 ≤ s) {θ : Fin K → ℝ} {m : Fin K}
    (hθ : 0 < θ m) {k W : ℕ} (hk : k < W) (hθW : θ m * W ≤ Real.pi) {α : Type*} [DecidableEq α]
    (w : List α) (x : ℕ) (a : α) :
    |slotHead s A θ m W w x k a - (if w[x - k]? = some a then 1 else 0)| ≤
      2 * x * ((x : ℝ) + 1) ^ (-(s * (A * (1 - Real.cos (θ m))))) := by
  have hv : ∀ j : Fin (x + 1), |(if w[(j : ℕ)]? = some a then 1 else 0 : ℝ)| ≤ 1 := fun j => by
    split_ifs <;> simp
  rw [slotHead]
  rcases le_or_gt k x with hkx | hxk
  · simpa using abs_logScaleAttention_rerope_sub_le θ m hA hs hθ hk hθW hkx hv
  · have h := abs_logScaleAttention_rerope_early_sub_le θ m hA hs hθ hk hθW hxk hv
    rw [mul_one] at h
    rwa [show x - k = ((0 : Fin (x + 1)) : ℕ) by rw [Fin.val_zero]; omega]

/-- The hypotheses of `abs_slotHead_sub_le` hold at `s = 2`, the window `4` and the distance `3`. -/
example (w : List Bool) (x : ℕ) (a : Bool) := abs_slotHead_sub_le (s := 2) (A := 1) zero_le_one
  zero_le_two (θ := fun _ : Fin 1 => Real.pi / 4) (m := 0) (by positivity) (k := 3) (W := 4)
  (by norm_num) (by push_cast; linarith) w x a

/-- The rounding to `{0, 1}`: `0` up to `1/4`, `1` from `3/4` on, linear between. -/
noncomputable def clampUnit (x : ℝ) : ℝ := max 0 (min 1 (2 * x - 1 / 2))

/-- **`clampUnit` is a ReLU layer of width two.** -/
theorem clampUnit_eq_relu (x : ℝ) :
    clampUnit x = max 0 (2 * x - 1 / 2) - max 0 (2 * x - 3 / 2) := by
  unfold clampUnit
  simp only [max_def, min_def]
  split_ifs <;> linarith

/-- **Rounding is exact near a bit**: within `1/4` of `[p]`, `clampUnit` returns `[p]`. -/
theorem clampUnit_ite {p : Prop} [Decidable p] {x : ℝ}
    (h : |x - (if p then 1 else 0)| < 1 / 4) : clampUnit x = if p then 1 else 0 := by
  rw [abs_lt] at h
  unfold clampUnit
  split_ifs at h ⊢
  · rw [min_eq_left (by linarith), max_eq_right zero_le_one]
  · rw [min_eq_right (by linarith), max_eq_left (by linarith)]

/-- The hypothesis of `clampUnit_ite` holds at `x = 0.9` for a true bit. -/
example : clampUnit 0.9 = if True then 1 else 0 :=
  clampUnit_ite (by norm_num [abs_lt])

/-- **Rounded slots are exact at every length**: for `s A (1 − cos θ_m) > 4` the rounded slot of
the distance `k < W` at the position `x` is `1` iff the token at the position `x - k` is `a` (the
first token while `x < k`), and `0` otherwise. -/
theorem clampUnit_slotHead {s A : ℝ} (hA : 0 ≤ A) (hs : 0 ≤ s) {θ : Fin K → ℝ} {m : Fin K}
    (hθ : 0 < θ m) {k W : ℕ} (hk : k < W) (hθW : θ m * W ≤ Real.pi)
    (hsg : 4 < s * (A * (1 - Real.cos (θ m)))) {α : Type*} [DecidableEq α] (w : List α)
    (x : ℕ) (a : α) :
    clampUnit (slotHead s A θ m W w x k a) = if w[x - k]? = some a then 1 else 0 := by
  refine clampUnit_ite ((abs_slotHead_sub_le hA hs hθ hk hθW w x a).trans_lt ?_)
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · norm_num
  have hx1 : (2 : ℝ) ≤ (x : ℝ) + 1 := by linarith [(Nat.one_le_cast.2 hx : (1 : ℝ) ≤ x)]
  have hpos : (0 : ℝ) < (x : ℝ) + 1 := by linarith
  have hp := Real.rpow_nonneg hpos.le (-(s * (A * (1 - Real.cos (θ m)))))
  calc 2 * (x : ℝ) * ((x : ℝ) + 1) ^ (-(s * (A * (1 - Real.cos (θ m)))))
      ≤ 2 * (((x : ℝ) + 1) * ((x : ℝ) + 1) ^ (-(s * (A * (1 - Real.cos (θ m)))))) := by
        rw [mul_assoc]
        exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right (by linarith) hp) zero_le_two
    _ = 2 * ((x : ℝ) + 1) ^ (1 - s * (A * (1 - Real.cos (θ m)))) := by
        rw [show (1 : ℝ) - s * (A * (1 - Real.cos (θ m))) = 1 + -(s * (A * (1 - Real.cos (θ m))))
          by ring, Real.rpow_add hpos, Real.rpow_one]
    _ ≤ 2 * 2 ^ (1 - s * (A * (1 - Real.cos (θ m)))) :=
        mul_le_mul_of_nonneg_left (Real.rpow_le_rpow_of_nonpos (by norm_num) hx1 (by linarith))
          zero_le_two
    _ < 2 * 2 ^ (-3 : ℝ) :=
        mul_lt_mul_of_pos_left (Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by linarith))
          zero_lt_two
    _ = 1 / 4 := by rw [Real.rpow_neg (by norm_num), Real.rpow_ofNat]; norm_num

/-- The hypotheses of `clampUnit_slotHead` hold at `s = 5`, `A = 1`, `θ = π/2`, the window `2`
and the distance `1`. -/
example (w : List Bool) (x : ℕ) (a : Bool) :=
  clampUnit_slotHead (s := 5) (A := 1) zero_le_one (by norm_num)
    (θ := fun _ : Fin 1 => Real.pi / 2) (m := 0) (by positivity) (k := 1) (W := 2) (by norm_num)
    (by push_cast; linarith) (by rw [Real.cos_pi_div_two]; norm_num) w x a

end Transformer.MRCR
