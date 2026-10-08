import Transformer.Grokking.AdamW.MomentFeedback

/-!
# Native moment limits from convergent generated inputs

Source: PyTorch 2.14.1 AdamW at lab commit 6badf0c, retained
exponential buffers, completed-clock corrections and epsilon floor.
Prove that convergent inputs force their actual first and second
moments to converge; do not assume that buffers already match the
current gradient. Arbitrary finite initial buffers are permitted.

The tail estimate keeps earlier history explicitly. The limiting
direction is the actual corrected history, not a moment reset or a
gradient-descent surrogate. Input convergence is a hypothesis here;
the closed CE application must derive it from parameter convergence.
No convergence of GPTMini parameters or stochastic minibatches is
claimed. In particular, bounded clipped inputs need not converge.
-/

namespace Transformer.Grokking.AdamW

open Filter

/-- A controlled input tail gives a controlled moment tail plus
the decaying retained past. Source: native AdamW at 6badf0c;
the entire earlier history survives in the actual buffer at start. -/
theorem moment_tail_limit_bound (beta initial value error : ℝ) (input : ℕ → ℝ)
    (start k : ℕ) (hb : 0 ≤ beta) (h1 : beta ≤ 1) (he : 0 ≤ error)
    (hg : ∀ j : ℕ, start ≤ j → |input j - value| ≤ error) :
    |momentAt beta input initial (start + k) - value| ≤
      beta ^ k * |momentAt beta input initial start - value| + error := by
  have hm := moment_different_prefix_bound beta (momentAt beta input initial start) value error
    (fun j => input (start + j)) (fun _ => value) k hb h1
    (fun j _ => hg (start + j) (by omega))
  have hc : momentAt beta (fun _ => value) value k = value := by
    rw [moment_constant]
    ring
  rw [hc, ← moment_continue] at hm
  have hp := mul_nonneg (pow_nonneg hb k) he
  linarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ j : ℕ, 1 ≤ j → |(if j = 0 then (10 : ℝ) else 2) - 2| ≤ 0) := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro j hj
  rw [ite_eq_right (by omega)]
  norm_num

/-- Convergent inputs force actual retained exponential buffers to
the same limit. Source: native AdamW at 6badf0c; the tail argument
derives convergence without assuming buffer/input agreement. -/
theorem moment_tendsto_of_input_tendsto (beta initial value : ℝ) (input : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hg : Tendsto input atTop (nhds value)) :
    Tendsto (momentAt beta input initial) atTop (nhds value) := by
  apply Metric.tendsto_atTop.mpr
  intro epsilon he
  have hh : 0 < epsilon / 2 := by linarith
  obtain ⟨start, hs⟩ := Metric.tendsto_atTop.mp hg (epsilon / 2) hh
  have hp : Tendsto (fun k : ℕ => beta ^ k * |momentAt beta input initial start - value|)
      atTop (nhds 0) := by
    simpa only [zero_mul] using
      (tendsto_pow_atTop_nhds_zero_of_lt_one hb h1).mul_const |momentAt beta input initial start - value|
  obtain ⟨cutoff, hc⟩ := eventually_atTop.mp (Tendsto.eventually_lt_const hh hp)
  refine ⟨start + cutoff, ?_⟩
  intro n hn
  let k := n - start
  have hk : cutoff ≤ k := by dsimp [k]; omega
  have hn' : n = start + k := by dsimp [k]; omega
  have ht := moment_tail_limit_bound beta initial value (epsilon / 2) input start k hb
    (le_of_lt h1) (le_of_lt hh) (fun j hj => le_of_lt (by simpa only [Real.dist_eq] using hs j hj))
  rw [Real.dist_eq, hn']
  have hsmall := hc k hk
  linarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  exact ⟨by norm_num, by norm_num, tendsto_const_nhds⟩

/-- The actual zero-initialized first moment inherits the gradient
limit. Source: native AdamW at 6badf0c, unsquared input recurrence. -/
theorem first_moment_tendsto (beta value : ℝ) (gradient : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hg : Tendsto gradient atTop (nhds value)) :
    Tendsto (firstMomentAt beta gradient) atTop (nhds value) := by
  exact moment_tendsto_of_input_tendsto beta 0 value gradient hb h1 hg

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  exact ⟨by norm_num, by norm_num, tendsto_const_nhds⟩

/-- Squared actual gradients force the second moment to the squared
gradient limit. Source: native non-AMSGrad AdamW at 6badf0c. -/
theorem second_moment_tendsto (beta value : ℝ) (gradient : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hg : Tendsto gradient atTop (nhds value)) :
    Tendsto (secondMomentAt beta gradient) atTop (nhds (value ^ 2)) := by
  exact moment_tendsto_of_input_tendsto beta 0 (value ^ 2) (fun k => gradient k ^ 2) hb h1 (hg.pow 2)

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  exact ⟨by norm_num, by norm_num, tendsto_const_nhds⟩

/-- Any retained clock tending to infinity supplies the actual
correction limit. Source: native AdamW at 6badf0c; no finite clock
limit or restarted optimizer is substituted for the completed count. -/
theorem corrected_buffer_clock_tendsto (beta value : ℝ) (buffer : ℕ → ℝ) (clock : ℕ → ℕ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hm : Tendsto buffer atTop (nhds value))
    (hc : Tendsto clock atTop atTop) :
    Tendsto (fun n => buffer n / (1 - beta ^ clock n)) atTop (nhds value) := by
  have hp := (tendsto_pow_atTop_nhds_zero_of_lt_one hb h1).comp hc
  have hd : Tendsto (fun n => 1 - beta ^ clock n) atTop (nhds 1) := by
    simpa only [Function.comp_apply, sub_zero] using tendsto_const_nhds.sub hp
  simpa only [Pi.div_def, div_one] using hm.div hd (by norm_num)

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) ∧
    Tendsto (fun n : ℕ => n + 1) atTop atTop := by
  exact ⟨by norm_num, by norm_num, tendsto_const_nhds, tendsto_add_atTop_nat 1⟩

/-- The genuine completed-clock correction tends to one and the
corrected first moment to the gradient. Source: native AdamW at
6badf0c; the exceptional zero clock cannot affect a tail limit. -/
theorem corrected_first_moment_tendsto (beta value : ℝ) (gradient : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hg : Tendsto gradient atTop (nhds value)) :
    Tendsto (fun n => firstMomentAt beta gradient n / (1 - beta ^ n)) atTop (nhds value) := by
  have hc : Tendsto (fun n : ℕ => 1 - beta ^ n) atTop (nhds 1) := by
    simpa only [sub_zero] using tendsto_const_nhds.sub (tendsto_pow_atTop_nhds_zero_of_lt_one hb h1)
  simpa only [Pi.div_def, div_one] using
    (first_moment_tendsto beta value gradient hb h1 hg).div hc (by norm_num)

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  exact ⟨by norm_num, by norm_num, tendsto_const_nhds⟩

/-- Physical limiting corrected variance is the squared gradient.
Source: native AdamW at 6badf0c, actual second-moment history and
clock correction, with no instantaneous-variance substitution. -/
theorem corrected_second_moment_tendsto (beta value : ℝ) (gradient : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hg : Tendsto gradient atTop (nhds value)) :
    Tendsto (fun n => secondMomentAt beta gradient n / (1 - beta ^ n)) atTop (nhds (value ^ 2)) := by
  have hc : Tendsto (fun n : ℕ => 1 - beta ^ n) atTop (nhds 1) := by
    simpa only [sub_zero] using tendsto_const_nhds.sub (tendsto_pow_atTop_nhds_zero_of_lt_one hb h1)
  simpa only [Pi.div_def, div_one] using
    (second_moment_tendsto beta value gradient hb h1 hg).div hc (by norm_num)

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  exact ⟨by norm_num, by norm_num, tendsto_const_nhds⟩

/-- The actual corrected-history direction tends to the normalized
gradient with its real epsilon floor. Source: native AdamW at 6badf0c;
this derived limit belongs to the retained optimizer, not a reset. -/
theorem history_direction_tendsto (b1 b2 eps value : ℝ) (gradient : ℕ → ℝ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hg : Tendsto gradient atTop (nhds value)) :
    Tendsto (historyDirection b1 b2 eps gradient) atTop (nhds (value / (|value| + eps))) := by
  change Tendsto (fun n => historyDirection b1 b2 eps gradient n) atTop
    (nhds (value / (|value| + eps)))
  have hm := corrected_first_moment_tendsto b1 value gradient hb1 h1 hg
  have hv := corrected_second_moment_tendsto b2 value gradient hb2 h2 hg
  have hd : Real.sqrt (value ^ 2) + eps ≠ 0 := by
    rw [Real.sqrt_sq_eq_abs]
    have ha := abs_nonneg value
    linarith
  simpa only [Pi.div_def, historyDirection, Real.sqrt_sq_eq_abs] using hm.div (hv.sqrt.add_const eps) hd

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    Tendsto (fun _ : ℕ => (-1 : ℝ)) atTop (nhds (-1)) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, tendsto_const_nhds⟩

end Transformer.Grokking.AdamW
