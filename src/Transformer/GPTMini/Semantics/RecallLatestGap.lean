import Transformer.GPTMini.Semantics.RecallMatching

/-!
# A uniform finite-temperature gap for the latest equal-key record

Source: the original head_dim=16/theta=10000 RoPE at f11b6e2
and raw integer positions in MQAR at cbafbe9. Strict preference
alone does not give a uniform finite-softmax copy bound. The actual
first compact matching frequency is 1/100. Two records separated
by at least one raw position have a cosine-score difference at
least 1-cos(1/100), uniformly over the complete recall context.

The other three actual matching pairs preserve nonnegative recency
differences. Averaging all four therefore supplies a positive
normalized gap. This gap is also below the proven different-key
content gap, so one constant handles both kinds of competitor.
Copied-key errors and genuine second-block projections remain to
be connected; no already-correct query route is assumed here.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The original fastest compact matching pair supplies a fixed positive normalized recency margin.
Source: its real frequency 1/100 and the actual four-pair normalized score. -/
noncomputable def recallLatestMargin : ℝ := (1 - Real.cos (1 / 100)) / 4

/-- The stated actual rotary margin is strictly positive.
Source: strict cosine decrease between zero and the original positive matching frequency. -/
theorem recallLatestMargin_pos : 0 < recallLatestMargin := by
  have hc := Real.cos_lt_cos_of_nonneg_of_le_pi (x := 0) (y := 1 / 100)
    (by norm_num) (by linarith [Real.pi_gt_three]) (by norm_num)
  rw [Real.cos_zero] at hc
  unfold recallLatestMargin
  linarith

/-- The normalized recency margin fits inside the independently proved categorical content margin.
Source: the exact frequency and the real Taylor lower bound for its cosine. -/
theorem recallLatestMargin_le_content : recallLatestMargin ≤ 1 / 50 := by
  have hc := Real.one_sub_sq_div_two_le_cos (x := 1 / 100)
  unfold recallLatestMargin
  nlinarith

/-- At raw position separation at least one, the actual cosine difference has a uniform boundary-step lower bound.
Source: the two-sine cosine difference formula and sine monotonicity on the proved small-angle interval. -/
theorem recall_cosine_step_gap (a b : ℝ) (ha : 0 ≤ a) (hsep : a + 1 ≤ b) (hb : b ≤ 64) :
    1 - Real.cos (1 / 100) ≤ Real.cos (a / 100) - Real.cos (b / 100) := by
  have hs : 0 < Real.sin (1 / 200) := Real.sin_pos_of_pos_of_lt_pi
    (by norm_num) (by linarith [Real.pi_gt_three])
  have hl := Real.sin_le_sin_of_le_of_le_pi_div_two (x := 1 / 200) (y := (a + b) / 200)
    (by linarith [Real.pi_gt_three]) (by linarith [Real.pi_gt_three]) (by linarith)
  have hr := Real.sin_le_sin_of_le_of_le_pi_div_two (x := 1 / 200) (y := (b - a) / 200)
    (by linarith [Real.pi_gt_three]) (by linarith [Real.pi_gt_three]) (by linarith)
  have hp := mul_le_mul hl hr hs.le (by linarith : 0 ≤ Real.sin ((a + b) / 200))
  have hbase : 1 - Real.cos (1 / 100) = 2 * Real.sin (1 / 200) ^ 2 := by
    have hc := Real.cos_sub_cos (x := 0) (y := 1 / 100)
    norm_num at hc
    nlinarith
  have hd : Real.cos (a / 100) - Real.cos (b / 100) =
      2 * Real.sin ((a + b) / 200) * Real.sin ((b - a) / 200) := by
    rw [Real.cos_sub_cos]
    rw [show (a / 100 + b / 100) / 2 = (a + b) / 200 from by ring,
      show (a / 100 - b / 100) / 2 = -((b - a) / 200) from by ring, Real.sin_neg]
    ring
  rw [hbase, hd]
  nlinarith

example : (0 : ℝ) ≤ 0 ∧ (0 + 1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 64 := by norm_num

/-- Every other actual compact matching pair weakly favors the later visible record.
Source: its derived positive original frequency and the complete sixty-four-position small-angle interval. -/
theorem recall_pair_latest_le (p : Fin 4) (query earlier later : ℝ)
    (horder : earlier ≤ later) (hvisible : later ≤ query) (hcontext : query - earlier ≤ 64) :
    Real.cos ((earlier - query) * recallFrequency p) ≤ Real.cos ((later - query) * recallFrequency p) := by
  have hf := recallFrequency_pos p
  have hh : (query - earlier) * recallFrequency p ≤ 64 * (1 / 100 : ℝ) :=
    mul_le_mul_of_nonneg hcontext (recallFrequency_le p) (by linarith) (by norm_num)
  rw [show (earlier - query) * recallFrequency p = -((query - earlier) * recallFrequency p) from by ring,
    show (later - query) * recallFrequency p = -((query - later) * recallFrequency p) from by ring,
    Real.cos_neg, Real.cos_neg]
  apply Real.cos_le_cos_of_nonneg_of_le_pi
  · nlinarith
  · linarith [Real.pi_gt_three]
  · nlinarith

example : (2 : ℝ) ≤ 32 ∧ (32 : ℝ) ≤ 63 ∧ (63 - 2 : ℝ) ≤ 64 := by norm_num

/-- All actual QKNorm/RoPE matching pairs give a uniform finite-temperature gap against earlier same-key records.
Source: the real four-pair score, with raw records at least one integer position apart and no assumed score gap. -/
theorem recall_latest_gap (alpha eps : ℝ) (heps : eps ≤ 2) (a : Fin 4 → Fin 4)
    (query earlier later : ℝ) (hsep : earlier + 1 ≤ later) (hvisible : later ≤ query)
    (hcontext : query - earlier ≤ 64) :
    score alpha eps (applyRope 16 10000 query (recallRotaryCode a))
        (applyRope 16 10000 earlier (recallRotaryCode a)) ≤
      score alpha eps (applyRope 16 10000 query (recallRotaryCode a))
        (applyRope 16 10000 later (recallRotaryCode a)) - Real.exp alpha * recallLatestMargin := by
  have horder : earlier ≤ later := by linarith
  have hz := recall_cosine_step_gap (query - later) (query - earlier) (by linarith) (by linarith) hcontext
  have he : Real.cos ((earlier - query) * recallFrequency 0) = Real.cos ((query - earlier) / 100) := by
    rw [recallFrequency_bounds.1,
      show (earlier - query) * (1 / 100) = -((query - earlier) / 100) from by ring, Real.cos_neg]
  have hl : Real.cos ((later - query) * recallFrequency 0) = Real.cos ((query - later) / 100) := by
    rw [recallFrequency_bounds.1,
      show (later - query) * (1 / 100) = -((query - later) / 100) from by ring, Real.cos_neg]
  have hs : (∑ p : Fin 4, Real.cos ((earlier - query) * recallFrequency p)) ≤
      (∑ p : Fin 4, Real.cos ((later - query) * recallFrequency p)) - 4 * recallLatestMargin := by
    rw [Fin.sum_univ_four, Fin.sum_univ_four, he, hl]
    unfold recallLatestMargin
    linarith [recall_pair_latest_le 1 query earlier later horder hvisible hcontext,
      recall_pair_latest_le 2 query earlier later horder hvisible hcontext,
      recall_pair_latest_le 3 query earlier later horder hvisible hcontext]
  rw [recallRotaryCode_score _ _ _ _ heps, recallRotaryCode_score _ _ _ _ heps]
  simp only [quarterRotaryInner_self]
  have hm := mul_le_mul_of_nonneg_left hs (Real.exp_pos alpha).le
  linarith

example : (1 / 100000 : ℝ) ≤ 2 ∧ (2 + 1 : ℝ) ≤ 32 ∧ (32 : ℝ) ≤ 63 ∧ (63 - 2 : ℝ) ≤ 64 := by
  norm_num

/-- The same fixed positive recency margin also separates every different-key competitor.
Source: the actual categorical content gap is larger than the derived boundary-step rotary margin. -/
theorem recall_different_latest_gap (alpha eps : ℝ) (heps : eps ≤ 2) (query other : Fin 256)
    (hne : query ≠ other) (s selected competitor : ℝ)
    (hselected : |selected - s| ≤ 64) (hcompetitor : |competitor - s| ≤ 64) :
    score alpha eps (applyRope 16 10000 s (recallRotaryCode (recallDigit query)))
        (applyRope 16 10000 competitor (recallRotaryCode (recallDigit other))) ≤
      score alpha eps (applyRope 16 10000 s (recallRotaryCode (recallDigit query)))
        (applyRope 16 10000 selected (recallRotaryCode (recallDigit query))) - Real.exp alpha * recallLatestMargin := by
  have hg := recall_content_gap alpha eps heps query other hne s selected competitor hselected hcompetitor
  have hm := mul_le_mul_of_nonneg_left recallLatestMargin_le_content (Real.exp_pos alpha).le
  linarith

example : (1 / 100000 : ℝ) ≤ 2 ∧ (0 : Fin 256) ≠ 1 ∧
    |(0 : ℝ) - 63| ≤ 64 ∧ |(62 : ℝ) - 63| ≤ 64 := by norm_num

/-- An excluded zero-key position is also below a matching record by the same fixed gap.
Source: the actual QKNorm zero-key score and the independently derived positive matching-code bound. -/
theorem recall_zero_latest_gap (alpha eps : ℝ) (heps : eps ≤ 2) (a : Fin 4 → Fin 4)
    (query selected : ℝ) (hselected : |selected - query| ≤ 64) :
    score alpha eps (applyRope 16 10000 query (recallRotaryCode a)) 0 ≤
      score alpha eps (applyRope 16 10000 query (recallRotaryCode a))
        (applyRope 16 10000 selected (recallRotaryCode a)) - Real.exp alpha * recallLatestMargin := by
  have hz : score alpha eps (applyRope 16 10000 query (recallRotaryCode a)) 0 = 0 := by
    simp only [score, normL2, smul_zero, inner_zero_right, mul_zero]
  rw [hz, recallRotaryCode_score _ _ _ _ heps]
  have hlo := mul_le_mul_of_nonneg_left (recall_matching_score_lower a _ hselected) (Real.exp_pos alpha).le
  have hm := mul_le_mul_of_nonneg_left recallLatestMargin_le_content (Real.exp_pos alpha).le
  linarith [Real.exp_pos alpha]

example : (1 / 100000 : ℝ) ≤ 2 ∧ |(0 : ℝ) - 63| ≤ 64 := by norm_num

end Transformer.GPTMini.Semantics
