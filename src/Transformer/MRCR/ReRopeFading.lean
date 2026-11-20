module

public import Transformer.MRCR.PhraseKeyReRope
public import Transformer.Grokking.FlatSpectrum.Recall.LogScale

/-!
# Clamped RoPE heads under the real softmax: softmax fades, far keys tie

In no paper. The other side of `Transformer.MRCR.ReRopeHeads`: heads with ReRoPE's logit
`reropeLogit` under the plain softmax, under SSMax below its threshold, and beyond the window.

* `abs_ropeScore_le`: a RoPE score is bounded by `‖(a, b)‖ = ∑ (|a_m| + |b_m|)`.
* `two_mul_le_one_sub_softmax_rerope`: so at any frequencies, coefficients and window, and any
  content bounded by `X`, the plain softmax answers the dictionary of `±1` that marks one key
  with an error of at least `2 i / (e^{2B} + i)`, `B = X + ‖(a, b)‖`, at the query position `i`.
  That tends to `2` (`exists_lt_softmax_rerope_error`).
* `exists_lt_logScale_rerope_error`: below its threshold, `2 s B < 1`, SSMax fades too.
* `two_mul_le_one_sub_softmaxAttention_of_tie`: tied keys share the weight.
  `two_mul_le_one_sub_softmax_rerope_far`: the keys at distances `≥ W` tie, so a positional head,
  whatever factor multiplies its logit (`1` for softmax, `s log n` for SSMax, Su's
  `max(1, log_T n)`), answers the dictionary that marks a key at a distance `≥ W` with an error of
  at least `2 (i − W) / (i − W + 1)`. Copying an answer by its position fails there at every
  scale.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall

variable {K : ℕ}

/-- A RoPE score is bounded by the size of its coefficients. -/
theorem abs_ropeScore_le (θ a b : Fin K → ℝ) (t : ℝ) : |ropeScore θ a b t| ≤ ropeNorm a b := by
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun l _ => (abs_add_le _ _).trans ?_)
  rw [abs_mul, abs_mul]
  exact add_le_add (mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _))
    (mul_le_of_le_one_right (abs_nonneg _) (Real.abs_sin_le_one _))

/-- ReRoPE's logit with content bounded by `X` is bounded by `X + ‖(a, b)‖`. -/
theorem abs_reropeLogit_le {c X : ℝ} (θ a b : Fin K → ℝ) (W i j : ℕ) {x : ℝ} (hx : |x / c| ≤ X) :
    |reropeLogit c θ a b W i j x| ≤ X + ropeNorm a b :=
  (abs_add_le _ _).trans (add_le_add hx (abs_ropeScore_le _ _ _ _))

/-- The hypothesis of `abs_reropeLogit_le` holds for the content `0`. -/
example := abs_reropeLogit_le (c := 1) (X := 0) (fun _ : Fin 1 => Real.pi / 1024) 0 0 512 7 2
  (x := 0) (by simp)

/-- **The plain softmax fades**: at the query position `i`, ReRoPE's logit over the keys `j ≤ i`,
at any frequencies, coefficients and window, with content `x j` bounded by `X` after the division
by `c`, answers the dictionary of `±1` that marks the key `j₀` with an error of at least
`2 i / (e^{2B} + i)`, `B = X + ‖(a, b)‖`. -/
theorem two_mul_le_one_sub_softmax_rerope (c : ℝ) (θ a b : Fin K → ℝ) (W i : ℕ)
    {x : Fin (i + 1) → ℝ} {X : ℝ} (hx : ∀ j, |x j / c| ≤ X) (j₀ : Fin (i + 1)) :
    2 * (i / (Real.exp (2 * (X + ropeNorm a b)) + i)) ≤
      1 - softmaxAttention (fun j : Fin (i + 1) => reropeLogit c θ a b W i j (x j))
        (fun j => if j = j₀ then 1 else -1) := by
  have h := two_mul_le_one_sub_softmaxAttention_indicator
    (fun j : Fin (i + 1) => abs_reropeLogit_le θ a b W i j (hx j)) j₀
  have hn : (Fintype.card (Fin (i + 1)) : ℝ) = (i : ℝ) + 1 := by simp
  rw [hn] at h
  convert h using 3 <;> ring

/-- The hypotheses of `two_mul_le_one_sub_softmax_rerope` hold for the content `0`. -/
example := two_mul_le_one_sub_softmax_rerope 1 (fun _ : Fin 1 => Real.pi / 1024) 0 0 512 7
  (x := fun _ => 0) (X := 0) (fun _ => by simp) 3

/-- **The error tends to `2`**: for every `ε < 2`, from some query position on, ReRoPE's logit
with content bounded by `X` answers the dictionary that marks any key with an error above `ε`. -/
theorem exists_lt_softmax_rerope_error (c : ℝ) (θ a b : Fin K → ℝ) (W : ℕ) (X : ℝ) {ε : ℝ}
    (hε : ε < 2) :
    ∃ M : ℕ, ∀ i ≥ M, ∀ x : Fin (i + 1) → ℝ, (∀ j, |x j / c| ≤ X) → ∀ j₀ : Fin (i + 1),
      ε < 1 - softmaxAttention (fun j : Fin (i + 1) => reropeLogit c θ a b W i j (x j))
        (fun j => if j = j₀ then 1 else -1) := by
  obtain ⟨E, hE⟩ : ∃ E, Real.exp (2 * (X + ropeNorm a b)) = E := ⟨_, rfl⟩
  have hE0 : 0 < E := hE ▸ Real.exp_pos _
  refine ⟨⌈ε * E / (2 - ε)⌉₊ + 1, fun i hi x hx j₀ => ?_⟩
  refine lt_of_lt_of_le ?_ (two_mul_le_one_sub_softmax_rerope c θ a b W i hx j₀)
  rw [hE, ← mul_div_assoc, lt_div_iff₀ (add_pos_of_pos_of_nonneg hE0 i.cast_nonneg)]
  have hi' : ε * E / (2 - ε) < i := by
    have : ((⌈ε * E / (2 - ε)⌉₊ + 1 : ℕ) : ℝ) ≤ i := by exact_mod_cast hi
    push_cast at this
    linarith [Nat.le_ceil (ε * E / (2 - ε))]
  rw [div_lt_iff₀ (by linarith)] at hi'
  linarith

/-- The hypothesis of `exists_lt_softmax_rerope_error` holds for `ε = 1`. -/
example := exists_lt_softmax_rerope_error 1 (fun _ : Fin 1 => Real.pi / 1024) 0 0 512 0 (ε := 1)
  (by norm_num)

/-- **Below its threshold SSMax fades too**: with content `x β` for the value `β`, bounded by `X`
after the division by `c`, and `2 s B < 1`, `B = X + ‖(a, b)‖`, ReRoPE's logit at the scale
`s log n`, for every accuracy `ε < 2`, from some query position on answers some dictionary of
values `±1` at every key with an error above `ε` (`exists_lt_logScaleAttention_recall_error`). -/
theorem exists_lt_logScale_rerope_error (c : ℝ) (θ a b : Fin K → ℝ) (W : ℕ) {s X ε : ℝ}
    (hs : 0 ≤ s) {x : Bool → ℝ} (hx : ∀ β, |x β / c| ≤ X)
    (hsB : 2 * s * (X + ropeNorm a b) < 1) (hε : ε < 2) :
    ∃ M : ℕ, ∀ i ≥ M, ∀ q : Fin (i + 1), ∃ π : Fin (i + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) - logScaleAttention s
        (fun j : Fin (i + 1) => reropeLogit c θ a b W i j (x (π j)))
        fun j => if π j then (1 : ℝ) else -1| :=
  exists_lt_logScaleAttention_recall_error hs hsB hε
    (fun i _ j β => reropeLogit c θ a b W i j (x β)) fun i _ j β =>
      abs_reropeLogit_le θ a b W i j (hx β)

/-- The hypotheses of `exists_lt_logScale_rerope_error` hold at `s = 1/4` for the content `0`. -/
example := exists_lt_logScale_rerope_error 1 (fun _ : Fin 1 => Real.pi / 1024) 0 0 512
  (s := 1 / 4) (X := 0) (ε := 1) (by norm_num) (x := fun _ => 0) (fun _ => by simp)
  (by norm_num [ropeNorm]) (by norm_num)

/-- **Tied keys share the weight**: if the scores agree on a set `T` of keys that holds `j₀`, the
dictionary of `±1` that marks `j₀` is answered with an error of at least `2 (|T| − 1) / |T|`. -/
theorem two_mul_le_one_sub_softmaxAttention_of_tie {ι : Type*} [Fintype ι] [DecidableEq ι]
    (s : ι → ℝ) {T : Finset ι} {j₀ : ι} (hj₀ : j₀ ∈ T) (hT : ∀ j ∈ T, s j = s j₀) :
    2 * (((#T : ℝ) - 1) / #T) ≤ 1 - softmaxAttention s (fun j => if j = j₀ then 1 else -1) := by
  have hT1 : 1 ≤ #T := card_pos.2 ⟨j₀, hj₀⟩
  have hpos : 0 < ∑ j, Real.exp (s j) := sum_pos (fun j _ => Real.exp_pos _) ⟨j₀, mem_univ _⟩
  have hle : ((#T : ℝ) - 1) * Real.exp (s j₀) ≤ ∑ j ∈ univ.erase j₀, Real.exp (s j) :=
    calc ((#T : ℝ) - 1) * Real.exp (s j₀) = ∑ j ∈ T.erase j₀, Real.exp (s j₀) := by
          rw [sum_const, card_erase_of_mem hj₀, nsmul_eq_mul, Nat.cast_sub hT1, Nat.cast_one]
      _ = ∑ j ∈ T.erase j₀, Real.exp (s j) :=
          sum_congr rfl fun j hj => by rw [hT j (mem_of_mem_erase hj)]
      _ ≤ ∑ j ∈ univ.erase j₀, Real.exp (s j) :=
          sum_le_sum_of_subset_of_nonneg (erase_subset_erase _ (subset_univ _))
            fun j _ _ => (Real.exp_pos _).le
  have hZ : ∑ j, Real.exp (s j) = Real.exp (s j₀) + ∑ j ∈ univ.erase j₀, Real.exp (s j) :=
    (add_sum_erase _ _ (mem_univ j₀)).symm
  have hTpos : (0 : ℝ) < #T := by exact_mod_cast hT1
  rw [one_sub_softmaxAttention_indicator, mul_div_assoc]
  refine mul_le_mul_of_nonneg_left ?_ zero_le_two
  rw [div_le_div_iff₀ hTpos hpos, hZ]
  linarith

/-- The hypotheses of `two_mul_le_one_sub_softmaxAttention_of_tie` hold for three keys tied at
`0`. -/
example := two_mul_le_one_sub_softmaxAttention_of_tie (fun _ : Fin 3 => (0 : ℝ)) (T := univ)
  (j₀ := 0) (mem_univ _) fun _ _ => rfl

/-- **Beyond the window the keys tie**: a head whose score at the query position `i ≥ W` depends
on the key `j ≤ i` only through the clamped distance `min (i − j) W` answers the dictionary of
`±1` that marks a key at a distance `≥ W` with an error of at least `2 (i − W) / (i − W + 1)`. -/
theorem two_mul_le_one_sub_softmaxAttention_clamped (F : ℕ → ℝ) {W i : ℕ} (hWi : W ≤ i)
    {j₀ : Fin (i + 1)} (hj₀ : W ≤ i - j₀) :
    2 * (((i : ℝ) - W) / ((i : ℝ) - W + 1)) ≤
      1 - softmaxAttention (fun j : Fin (i + 1) => F (min (i - j) W))
        (fun j => if j = j₀ then 1 else -1) := by
  have hj₀i := j₀.isLt
  have h := two_mul_le_one_sub_softmaxAttention_of_tie (fun j : Fin (i + 1) => F (min (i - j) W))
    (T := Iic ⟨i - W, by omega⟩) (j₀ := j₀)
    (mem_Iic.2 (Fin.le_def.2 (show (j₀ : ℕ) ≤ i - W by omega))) fun j hj => by
      have hj' : (j : ℕ) ≤ i - W := Fin.le_def.1 (mem_Iic.1 hj)
      simp only [min_eq_right (show W ≤ i - j by omega), min_eq_right hj₀]
  have hc : ((#(Iic (⟨i - W, by omega⟩ : Fin (i + 1))) : ℕ) : ℝ) = (i : ℝ) - W + 1 := by
    rw [Fin.card_Iic, Fin.val_mk, Nat.cast_add, Nat.cast_sub hWi, Nat.cast_one]
  rwa [hc, add_sub_cancel_right] at h

/-- The hypotheses of `two_mul_le_one_sub_softmaxAttention_clamped` hold at the query position `5`
with the window `2` for the first key. -/
example := two_mul_le_one_sub_softmaxAttention_clamped (fun _ => 0) (W := 2) (i := 5)
  (by norm_num) (j₀ := 0) (by norm_num)

/-- **Copying by position fails beyond the window at every scale**: ReRoPE's logit with the same
content `x` at every key, multiplied by any factor `g` (`1` for softmax, `s log (i + 1)` for
SSMax, `max(1, log_T (i + 1))` for Su's scale), answers the dictionary of `±1` that marks a key
at a distance `≥ W` from the query position `i ≥ W` with an error of at least
`2 (i − W) / (i − W + 1)`. -/
theorem two_mul_le_one_sub_softmax_rerope_far (g c : ℝ) (θ a b : Fin K → ℝ) {W i : ℕ}
    (hWi : W ≤ i) (x : ℝ) {j₀ : Fin (i + 1)} (hj₀ : W ≤ i - j₀) :
    2 * (((i : ℝ) - W) / ((i : ℝ) - W + 1)) ≤
      1 - softmaxAttention (fun j : Fin (i + 1) => g * reropeLogit c θ a b W i j x)
        (fun j => if j = j₀ then 1 else -1) :=
  two_mul_le_one_sub_softmaxAttention_clamped (fun d => g * (x / c + ropeScore θ a b d)) hWi hj₀

/-- The hypotheses of `two_mul_le_one_sub_softmax_rerope_far` hold for Su's window `512` at the
query position `4096`, for the first key. -/
example := two_mul_le_one_sub_softmax_rerope_far 2 1 (fun _ : Fin 1 => Real.pi / 1024) 0 0
  (W := 512) (i := 4096) (by norm_num) 0 (j₀ := 0) (by norm_num)

end Transformer.MRCR
