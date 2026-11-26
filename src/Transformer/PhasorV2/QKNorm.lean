module

public import Transformer.PhasorV2.SSMax
public import Transformer.PhasorV2.LeRoPE

/-!
# Phasor v2: Q/K RMS

The fourth element of Phasor v2 normalizes queries and keys. Source: DiffGPT
(`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`, *Architecture*:
`q_i = LeRoPE_i(R_d(W_Q z_i))` and `k_i = LeRoPE_i(R_d(W_K z_i))` per head, "Q/K RMS uses the
head width", every `R` is parameter-free, and the SSMax gain is `ln N / d`. A head vector here is
`p` complex bands, so its real width is `d = 2p` and its squared length is `∑_m |q_m|²` (`sqLen`).
As for the streams (`Transformer.PhasorV2.PeriNorm`), Phasor v2 takes the mean over the complex
bands, `R(q) = q / √(mean_m |q_m|² + ε)` (`headRMS`), and with it the gain `ln N / p`.

* A normalized head vector has squared length at most `p` (`sqLen_headRMS_le`), and LeRoPE keeps
  the length (`sqLen_turn`).
* Cauchy–Schwarz, `|score q k| ≤ √(sqLen q) √(sqLen k)` (`abs_score_le_sqrt`), bounds the score of a
  normalized query and key by `p` at all positions and frequencies (`abs_score_headRMS_le`). Under
  the gain `s = 1/p` the scaled score is at most one (`abs_mul_score_le`), so every logit at the
  length `N` is at most `ln N` (`abs_logit_le`), for all weights.
* Then no key outweighs another by more than the factor `N²` (`ssmaxWeight_le_sq_mul`): at a fixed
  length no key is dropped and none takes all the weight, while a lead of more than `p` in score
  still focuses at every length (`exists_forall_focus`).
-/

@[expose] public section

namespace Transformer.PhasorV2

variable {p : ℕ} {ι : Type*} [Fintype ι]

/-- The squared length `∑_m |q_m|²` of a head vector, that of its `2p` real coordinates. -/
noncomputable def sqLen (q : Fin p → ℂ) : ℝ := ∑ m, ‖q m‖ ^ 2

/-- **Q/K RMS** (DiffGPT's main model, *Architecture*, over complex bands):
`R(q) = √p q / √(sqLen q + p ε)`, that is `q / √(mean_m |q_m|² + ε)` over the `p` complex bands of
the head vector `q`. -/
noncomputable def headRMS (eps : ℝ) (q : Fin p → ℂ) : Fin p → ℂ :=
  fun m => ((Real.sqrt p / Real.sqrt (sqLen q + p * eps) : ℝ) : ℂ) * q m

theorem sqLen_nonneg (q : Fin p → ℂ) : 0 ≤ sqLen q :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- Scaling every band by a real `c` scales the squared length by `c²`. -/
theorem sqLen_ofReal_mul (c : ℝ) (q : Fin p → ℂ) :
    sqLen (fun m => (c : ℂ) * q m) = c ^ 2 * sqLen q := by
  simp only [sqLen, norm_mul, Complex.norm_real, Real.norm_eq_abs, mul_pow, sq_abs,
    Finset.mul_sum]

/-- **LeRoPE keeps the length.** -/
theorem sqLen_turn (θ : Fin p → ℝ) (t : ℝ) (q : Fin p → ℂ) : sqLen (turn θ t q) = sqLen q := by
  simp only [sqLen, norm_turn]

/-- **A normalized head vector has squared length at most `p`.** -/
theorem sqLen_headRMS_le {eps : ℝ} (heps : 0 < eps) (q : Fin p → ℂ) :
    sqLen (headRMS eps q) ≤ p := by
  have hden : 0 ≤ sqLen q + p * eps := add_nonneg (sqLen_nonneg q) (by positivity)
  unfold headRMS
  rw [sqLen_ofReal_mul, div_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt hden,
    div_mul_eq_mul_div, mul_div_assoc]
  exact mul_le_of_le_one_right (by positivity)
    (div_le_one_of_le₀ (le_add_of_nonneg_right (by positivity)) hden)

/-- Sixteen bands at the main model's `ε = 2^{−23}`. -/
example (q : Fin 16 → ℂ) := sqLen_headRMS_le (eps := 1 / 2 ^ 23) (by norm_num) q

/-- **Cauchy–Schwarz**: `|score q k| ≤ √(sqLen q) √(sqLen k)`. -/
theorem abs_score_le_sqrt (q k : Fin p → ℂ) :
    |score q k| ≤ Real.sqrt (sqLen q) * Real.sqrt (sqLen k) :=
  (abs_score_le q k).trans (Real.sum_mul_le_sqrt_mul_sqrt _ _ _)

/-- **A normalized query and key score at most `p`** in absolute value, at all positions and
frequencies. -/
theorem abs_score_headRMS_le {eps : ℝ} (heps : 0 < eps) (θ : Fin p → ℝ) (i j : ℝ)
    (q k : Fin p → ℂ) :
    |score (turn θ i (headRMS eps q)) (turn θ j (headRMS eps k))| ≤ p := by
  refine (abs_score_le_sqrt _ _).trans ?_
  rw [sqLen_turn, sqLen_turn]
  calc Real.sqrt (sqLen (headRMS eps q)) * Real.sqrt (sqLen (headRMS eps k))
      ≤ Real.sqrt p * Real.sqrt p :=
        mul_le_mul (Real.sqrt_le_sqrt (sqLen_headRMS_le heps q))
          (Real.sqrt_le_sqrt (sqLen_headRMS_le heps k)) (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
    _ = p := Real.mul_self_sqrt (Nat.cast_nonneg p)

/-- The query at the position `7` against the key at `3`: at most `16`. -/
example (θ : Fin 16 → ℝ) (q k : Fin 16 → ℂ) :=
  abs_score_headRMS_le (eps := 1 / 2 ^ 23) (by norm_num) θ 7 3 q k

/-- **Under the gain `1/p` the scaled score is at most one.** -/
theorem abs_mul_score_le (hp : 0 < p) {eps : ℝ} (heps : 0 < eps) (θ : Fin p → ℝ) (i j : ℝ)
    (q k : Fin p → ℂ) :
    |1 / p * score (turn θ i (headRMS eps q)) (turn θ j (headRMS eps k))| ≤ 1 := by
  have hp' : (0 : ℝ) < p := Nat.cast_pos.2 hp
  rw [abs_mul, abs_of_pos (one_div_pos.2 hp'), one_div_mul_eq_div, div_le_one hp']
  exact abs_score_headRMS_le heps θ i j q k

/-- Sixteen bands. -/
example (θ : Fin 16 → ℝ) (q k : Fin 16 → ℂ) :=
  abs_mul_score_le (p := 16) (by norm_num) (eps := 1 / 2 ^ 23) (by norm_num) θ 7 3 q k

/-- **Every logit is at most `ln N`**: the SSMax logit `s ln N z` of `ssmaxWeight` at the length
`N ≥ 1`, when the scaled score `s z` is at most one. -/
theorem abs_logit_le {s z N : ℝ} (hz : |s * z| ≤ 1) (hN : 1 ≤ N) :
    |s * Real.log N * z| ≤ Real.log N := by
  have hlog := Real.log_nonneg hN
  rw [mul_right_comm, abs_mul, abs_of_nonneg hlog]
  exact mul_le_of_le_one_left hlog hz

/-- A normalized query and key at the length `8`: the logit is at most `ln 8`. -/
example (θ : Fin 16 → ℝ) (q k : Fin 16 → ℂ) :=
  abs_logit_le (abs_mul_score_le (p := 16) (by norm_num) (eps := 1 / 2 ^ 23) (by norm_num) θ 7 3
    q k) (by norm_num : (1 : ℝ) ≤ 8)

/-- **No key outweighs another by more than the factor `N²`**, `N` keys, when every scaled score
`s z_k` is at most one. -/
theorem ssmaxWeight_le_sq_mul [Nonempty ι] {s : ℝ} {z : ι → ℝ} (hz : ∀ k, |s * z k| ≤ 1)
    (j k : ι) : ssmaxWeight s z j ≤ (Fintype.card ι : ℝ) ^ 2 * ssmaxWeight s z k := by
  have hN : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hN0 : (0 : ℝ) < Fintype.card ι := by linarith
  have hS : 0 < ∑ k, (Fintype.card ι : ℝ) ^ (s * z k) :=
    Finset.sum_pos (fun k _ => Real.rpow_pos_of_pos hN0 _) Finset.univ_nonempty
  rw [ssmaxWeight_eq_rpow, ssmaxWeight_eq_rpow, ← mul_div_assoc]
  refine div_le_div_of_nonneg_right ?_ hS.le
  rw [← Real.rpow_two, ← Real.rpow_add hN0]
  refine Real.rpow_le_rpow_of_exponent_le hN ?_
  linarith [(abs_le.1 (hz j)).2, (abs_le.1 (hz k)).1]

/-- **A head with Q/K RMS**: the query at the position `7` against eight keys, sixteen bands, the
gain `1/16`; any two keys' weights differ by at most the factor `64`. -/
example (θ : Fin 16 → ℝ) (q : Fin 16 → ℂ) (key : Fin 8 → Fin 16 → ℂ) (j k : Fin 8) :=
  ssmaxWeight_le_sq_mul (s := 1 / (16 : ℕ))
    (z := fun l : Fin 8 => score (turn θ 7 (headRMS (1 / 2 ^ 23) q))
      (turn θ (l : ℕ) (headRMS (1 / 2 ^ 23) (key l))))
    (fun l : Fin 8 => abs_mul_score_le (by norm_num) (by norm_num) θ 7 (l : ℕ) q (key l)) j k

end Transformer.PhasorV2
