module

public import Transformer.RoPE.Scores
public import Transformer.RoPE.YaRN.Section3_Ramp
public import Transformer.Precision.Basic

/-!
# YaRN's empirical temperature prescription

Peng et al., arXiv:2309.00071v3, §3.3, Eqs. (14)–(15). The equation
`sqrt(1/t) = 1 + 0.1 log(s)` is a fitted recommendation for LLaMA models.
We prove its algebraic effects, not optimality of the fitted coefficient,
perplexity improvement, or successful retrieval for arbitrary learned weights.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- Eq. (15): the fitted gain applied to each of Q and K. -/
noncomputable def lengthGain (scale : ℝ) : ℝ := 1 + Real.log scale / 10

/-- Eq. (14): scaling both Q and K multiplies logits by the square. -/
noncomputable def logitGain (scale : ℝ) : ℝ := lengthGain scale ^ 2

/-- The temperature `t` in Eq. (14), recovered from the prescribed gain. -/
noncomputable def temperature (scale : ℝ) : ℝ := (logitGain scale)⁻¹

/-- Extending context cannot reduce the prescribed Q/K length gain. -/
theorem length_gain_ge_one {scale : ℝ} (hs : 1 ≤ scale) : 1 ≤ lengthGain scale := by
  have h := Real.log_nonneg hs
  unfold lengthGain
  linarith

example : 1 ≤ lengthGain 4 := length_gain_ge_one (by norm_num)

/-- The empirical rule sharpens logits by a positive factor at least one. -/
theorem logit_gain_ge_one {scale : ℝ} (hs : 1 ≤ scale) : 1 ≤ logitGain scale := by
  have h := length_gain_ge_one hs
  unfold logitGain
  nlinarith [sq_nonneg (lengthGain scale - 1)]

example : 1 ≤ logitGain 4 := logit_gain_ge_one (by norm_num)

/-- The temperature is well-defined and lies in `(0,1]` for extensions. -/
theorem temperature_bounds {scale : ℝ} (hs : 1 ≤ scale) :
    0 < temperature scale ∧ temperature scale ≤ 1 := by
  have h := logit_gain_ge_one hs
  unfold temperature
  constructor
  · positivity
  · exact (inv_le_one₀ (by linarith : 0 < logitGain scale)).2 h

example : 0 < temperature 4 ∧ temperature 4 ≤ 1 := temperature_bounds (by norm_num)

/-- Eqs. (14)–(15) require the squared Q/K gain, not a single factor. -/
theorem inverse_temperature (scale : ℝ) : 1 / temperature scale = lengthGain scale ^ 2 := by
  simp [temperature, logitGain]

/-- The actual two-dimensional Q/K dot product scales quadratically. -/
theorem scaled_pair_score (query key : ℝ × ℝ) (omega distance gain : ℝ) :
    pairScore (gain * query.1, gain * query.2) (gain * key.1, gain * key.2) omega distance =
      gain ^ 2 * pairScore query key omega distance := by
  unfold pairScore
  ring

/-- Positive uniform temperature preserves all strict logit rankings.
This does not say that the winning token is the desired token. -/
theorem temperature_preserves_order {scale : ℝ} (hs : 1 ≤ scale) (a b : ℝ) :
    logitGain scale * a < logitGain scale * b ↔ a < b := by
  have hg : 0 < logitGain scale := lt_of_lt_of_le (by norm_num) (logit_gain_ge_one hs)
  constructor <;> intro h <;> nlinarith

example : logitGain 4 * 1 < logitGain 4 * 2 ↔ (1 : ℝ) < 2 :=
  temperature_preserves_order (by norm_num) _ _

/-- Eq. (14): the softmax odds depend exponentially on the scaled margin.
The denominator is a sum over the actual visible keys. -/
theorem softmax_odds {n : ℕ} (scores : Fin n → ℝ) (i j : Fin n) (scale : ℝ) :
    Precision.softmax (fun k => logitGain scale * scores k) i /
        Precision.softmax (fun k => logitGain scale * scores k) j =
      Real.exp (logitGain scale * (scores i - scores j)) := by
  have hz := Precision.softmax_denom_pos (fun k => logitGain scale * scores k) i
  unfold Precision.softmax
  rw [div_div_div_cancel_right₀ (ne_of_gt hz), ← Real.exp_sub, ← mul_sub]

/-- With a nonnegative margin the fitted temperature increases its odds.
With a wrong winner it instead reinforces the wrong order. -/
theorem softmax_odds_increase {n : ℕ} (scores : Fin n → ℝ) (i j : Fin n)
    {scale : ℝ} (hs : 1 ≤ scale) (hm : scores j ≤ scores i) :
    Precision.softmax scores i / Precision.softmax scores j ≤
      Precision.softmax (fun k => logitGain scale * scores k) i /
        Precision.softmax (fun k => logitGain scale * scores k) j := by
  have hz := Precision.softmax_denom_pos scores i
  rw [softmax_odds]
  unfold Precision.softmax
  rw [div_div_div_cancel_right₀ (ne_of_gt hz), ← Real.exp_sub]
  apply Real.exp_le_exp.2
  have hg := logit_gain_ge_one hs
  nlinarith

example : Precision.softmax (fun i : Fin 2 => (i.val : ℝ)) 1 /
      Precision.softmax (fun i : Fin 2 => (i.val : ℝ)) 0 ≤
    Precision.softmax (fun i : Fin 2 => logitGain 4 * (i.val : ℝ)) 1 /
      Precision.softmax (fun i : Fin 2 => logitGain 4 * (i.val : ℝ)) 0 :=
  softmax_odds_increase _ _ _ (by norm_num) (by norm_num)

/-- Without context extension, neither Q/K length nor logits are rescaled. -/
theorem gains_one : lengthGain 1 = 1 ∧ logitGain 1 = 1 := by
  norm_num [lengthGain, logitGain]

end Transformer.RoPE.YaRN
