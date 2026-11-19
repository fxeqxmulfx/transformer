module

public import Mathlib.Analysis.SpecialFunctions.Log.Base
public import Transformer.RoPE.ReRoPE.Section3_Computation

/-!
# ReRoPE, 实验 (Experiments): the `log n` scale and the reference code

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 实验
(Experiments), and the code it shares, github.com/bojone/rerope at commit acfe788
(`rerope_patch.py`, `leaky_rerope_patch.py`, patches of the Hugging Face LLaMA attention).
The accuracy and loss tables are measurements and are not formalized.

* `logScale T n = max(1, log_T n)`: in the `log n†` runs "at test time each `q_n` is multiplied by
  `max(1, log_maxlen n)`"; the code multiplies the query at position `i` by
  `(log(i + 1)/log(training_length)).clip(1)`, `n = i + 1`. `logScale_eq_one`: up to the training
  length the scale is `1`, so the scaled model is the unscaled one there; `one_lt_logScale`:
  beyond it the queries are scaled up;
* `code_leaky_score`: Leaky ReRoPE's code rotates the second queries at `i + offset` with the table
  of the positions divided by `k = scaling_factor`, `offset = w(k − 1)`, and the second keys at
  `j`; `(i + w(k − 1))/k = (i − w)/k + w` is the source's query position;
* `codePosition`, `codePosition_eq`, `code_decoding_leaky`: the decoding table
  `cat(cos[:w], cos2[w + offset:])` holds at distance `d` the position `d` for `d < w` and
  `(d + offset)/k` beyond, Leaky ReRoPE's position; the key is rotated by
  `apply_rotary_pos_emb(…, cos, −sin, …)`, which is `R^{−t}` (`code_key_rotation`), so decoding
  gives the merged score for every `k ≠ 0`, where the max form `merge_leaky_eq_max` needs `k ≥ 1`;
* `code_decoding_rectified`: ReRoPE's decoding rotates the key back by the distance clipped at the
  window, `(i − j).clip(max=w)`, and gives the merged score. The code's mask `|i − j| < w` is the
  source's `i − j < w` on the causal pairs `j ≤ i`, the only ones the softmax keeps.
-/

@[expose] public section

open Matrix

namespace Transformer.RoPE.ReRoPE

open RoFormer

/-- Experiments: the query scale `max(1, log_T n)` of the `log n` runs, `T` the training length. -/
noncomputable def logScale (T n : ℝ) : ℝ :=
  max 1 (Real.logb T n)

/-- Experiments: up to the training length `T > 1` the scale is `1`. -/
theorem logScale_eq_one {T n : ℝ} (hT : 1 < T) (hn : 0 < n) (hnT : n ≤ T) :
    logScale T n = 1 := by
  rw [logScale, max_eq_left]
  calc Real.logb T n ≤ Real.logb T T := Real.logb_le_logb_of_le hT hn hnT
    _ = 1 := Real.logb_self_eq_one hT

example (n : ℝ) (h1 : 1 ≤ n) (h2 : n ≤ 4096) : logScale 4096 n = 1 :=
  logScale_eq_one (by norm_num) (by linarith) h2

/-- Experiments: beyond the training length `T > 1` the queries are scaled up. -/
theorem one_lt_logScale {T n : ℝ} (hT : 1 < T) (hTn : T < n) : 1 < logScale T n := by
  have h : 1 < Real.logb T n := by
    rw [← Real.logb_self_eq_one hT]
    exact Real.logb_lt_logb hT (by linarith) hTn
  exact lt_of_lt_of_le h (le_max_right _ _)

example : 1 < logScale 4096 16384 :=
  one_lt_logScale (by norm_num) (by norm_num)

variable {p : ℕ}

/-- `leaky_rerope_patch.py`: the second queries rotated at `(i + offset)/k`, `offset = w(k − 1)`,
and the second keys at `j/k` give the source's `a⁽²⁾_{i,j}`, with the query at `(i − w)/k + w`. -/
theorem code_leaky_score (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w i j : ℝ) {k : ℝ}
    (hk : k ≠ 0) :
    (rotary θ ((i + w * (k - 1)) / k) *ᵥ q) ⬝ᵥ (rotary θ (j / k) *ᵥ key) =
      (rotary θ ((i - w) / k + w) *ᵥ q) ⬝ᵥ (rotary θ (j / k) *ᵥ key) := by
  rw [show (i + w * (k - 1)) / k = (i - w) / k + w by field_simp; ring]

example (θ : Fin 64 → ℝ) (q key : Fin 2 × Fin 64 → ℝ) (i j : ℝ) :
    (rotary θ ((i + 512 * (16 - 1)) / 16) *ᵥ q) ⬝ᵥ (rotary θ (j / 16) *ᵥ key) =
      (rotary θ ((i - 512) / 16 + 512) *ᵥ q) ⬝ᵥ (rotary θ (j / 16) *ᵥ key) :=
  code_leaky_score θ q key 512 i j (by norm_num)

/-- `leaky_rerope_patch.py`, decoding: the position at distance `d` of the table
`cat(cos[:w], cos2[w + offset:])`, `d` for `d < w` and `(d + offset)/k`, `offset = w(k − 1)`,
beyond. -/
noncomputable def codePosition (w k d : ℝ) : ℝ :=
  if d < w then d else (d + w * (k - 1)) / k

/-- The decoding table holds Leaky ReRoPE's positions. -/
theorem codePosition_eq (w d : ℝ) {k : ℝ} (hk : k ≠ 0) :
    codePosition w k d = leakyPosition w k d := by
  unfold codePosition leakyPosition
  split_ifs
  · rfl
  · field_simp
    ring

example (d : ℝ) : codePosition 512 16 d = leakyPosition 512 16 d :=
  codePosition_eq 512 d (by norm_num)

/-- The code's key rotation `apply_rotary_pos_emb(None, k, cos, −sin, t)`,
`k cos tθ + rotate_half(k)(−sin tθ)`, is `R^{−t} k`. -/
theorem code_key_rotation (θ : Fin p → ℝ) (t : ℝ) (x : Fin 2 × Fin p → ℝ) :
    (fun ri => x ri * Real.cos (t * θ ri.2) + rotateHalf x ri * -Real.sin (t * θ ri.2)) =
      rotary θ (-t) *ᵥ x := by
  rw [rotary_mulVec]
  ext ri
  rw [neg_mul, Real.cos_neg, Real.sin_neg]

/-- `leaky_rerope_patch.py`, decoding (`q_len == 1`): the keys rotated back by the table position
at the distance `i − j` give the merged Leaky ReRoPE score, for every `k ≠ 0`. -/
theorem code_decoding_leaky (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w i j : ℝ) {k : ℝ}
    (hk : k ≠ 0) :
    q ⬝ᵥ (fun ri => key ri * Real.cos (codePosition w k (i - j) * θ ri.2) +
        rotateHalf key ri * -Real.sin (codePosition w k (i - j) * θ ri.2)) =
      if i - j < w then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key)
      else (rotary θ ((i - w) / k + w) *ᵥ q) ⬝ᵥ (rotary θ (j / k) *ᵥ key) := by
  rw [code_key_rotation, merge_leaky, codePosition_eq w _ hk]

example (θ : Fin 64 → ℝ) (q key : Fin 2 × Fin 64 → ℝ) (i j : ℝ) :
    q ⬝ᵥ (fun ri => key ri * Real.cos (codePosition 512 16 (i - j) * θ ri.2) +
        rotateHalf key ri * -Real.sin (codePosition 512 16 (i - j) * θ ri.2)) =
      if i - j < 512 then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key)
      else (rotary θ ((i - 512) / 16 + 512) *ᵥ q) ⬝ᵥ (rotary θ (j / 16) *ᵥ key) :=
  code_decoding_leaky θ q key 512 i j (by norm_num)

/-- `rerope_patch.py`, decoding (`q_len == 1`): the keys rotated back by the distance clipped at
the window, `(i − j).clip(max=w)`, give the merged ReRoPE score. -/
theorem code_decoding_rectified (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (w i j : ℝ) :
    q ⬝ᵥ (fun ri => key ri * Real.cos (min (i - j) w * θ ri.2) +
        rotateHalf key ri * -Real.sin (min (i - j) w * θ ri.2)) =
      if i - j < w then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key)
      else (rotary θ w *ᵥ q) ⬝ᵥ key := by
  rw [code_key_rotation, merge_rectified, rectifiedPosition]

end Transformer.RoPE.ReRoPE
