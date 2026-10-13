import Transformer.GPTMini.Semantics.RecallRotaryCode

/-!
# Compact query-key code retains a strict gap under original RoPE

Source: the original softmax/QKNorm/RoPE head at f11b6e2 and the 256
raw MQAR symbols at cbafbe9. The explicit four categorical pairs occupy
the actual slow RoPE coordinates. Identical symbols score at least .93
after normalization at any displacement within the recall context;
different symbols score at most .91. Both inequalities evaluate the
actual rotated vectors. No prepared ideal-score-gap premise is used.

This verifies geometry for a given compact code. Producing each code
from the raw simultaneous embedding/QKV, gating the table, selecting the
latest equal-key write and proving the full model answer still remain.
The construction is an ordinary softmax baseline component, not a
claim that this trainable parameterization is convex.
The positive content gap and strict same-key recency preference coexist
in one actual sixteen-coordinate query/key representation.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- Identical categorical axes contribute the exact relative-position cosine.
Source: the actual two-dimensional rotation formula for each of the four signed axes. -/
theorem quarterRotaryInner_self (a : Fin 4) (angle : ℝ) :
    quarterRotaryInner a a angle = Real.cos angle := by
  fin_cases a <;> norm_num [quarterRotaryInner, quarterX, quarterY]

/-- Every normalized categorical pair score is at most one, at any actual angle.
Source: the sixteen signed-axis comparisons, retaining both cosine and sine branches. -/
theorem quarterRotaryInner_le_one (a b : Fin 4) (angle : ℝ) :
    quarterRotaryInner a b angle ≤ 1 := by
  fin_cases a <;> fin_cases b <;> norm_num [quarterRotaryInner, quarterX, quarterY] <;>
    linarith [Real.cos_le_one (x := angle), Real.sin_le_one angle,
      Real.neg_one_le_cos angle, Real.neg_one_le_sin angle]

/-- A differing pair contributes at most .64 at every displacement admitted by the actual recall frequencies.
Source: the real rotation, |sin(angle)|≤|angle| and positive cos(angle) on this proved interval. -/
theorem quarterRotaryInner_mismatch (a b : Fin 4) (hne : a ≠ b) (angle : ℝ)
    (hangle : |angle| ≤ 16 / 25) : quarterRotaryInner a b angle ≤ 16 / 25 := by
  have hsq := sq_le_sq' (abs_le.mp hangle).1 (abs_le.mp hangle).2
  have hcos : 0 ≤ Real.cos angle := by
    nlinarith [Real.one_sub_sq_div_two_le_cos (x := angle)]
  have hsin : |Real.sin angle| ≤ 16 / 25 := Real.abs_sin_le_abs.trans hangle
  have hs := abs_le.mp hsin
  fin_cases a <;> fin_cases b
  all_goals norm_num at hne
  all_goals norm_num [quarterRotaryInner, quarterX, quarterY]
  all_goals linarith

example : (0 : Fin 4) ≠ 1 ∧ |(1 / 100 : ℝ)| ≤ 16 / 25 := by norm_num

/-- All four pairs simultaneously retain an identical-symbol normalized score at least .93.
Source: the actual unequal-frequency cosine sum, rather than treating the symbols as position-independent. -/
theorem recall_matching_score_lower (a : Fin 4 → Fin 4) (d : ℝ) (hd : |d| ≤ 64) :
    93 / 100 ≤ (∑ p : Fin 4, quarterRotaryInner (a p) (a p) (d * recallFrequency p)) / 4 := by
  simp only [quarterRotaryInner_self]
  exact recall_cosine_match_lower d hd

example : |(-63 : ℝ)| ≤ 64 := by norm_num

/-- A differing compact symbol has normalized score at most .91 after all four actual rotations.
Source: one differing pair and the remaining three bounded pair scores, with the real context cap. -/
theorem recall_mismatch_score_upper (a b : Fin 4 → Fin 4) (hne : a ≠ b) (d : ℝ)
    (hd : |d| ≤ 64) :
    (∑ p : Fin 4, quarterRotaryInner (a p) (b p) (d * recallFrequency p)) / 4 ≤ 91 / 100 := by
  classical
  obtain ⟨p, hp⟩ : ∃ p, a p ≠ b p := by
    by_contra hn
    apply hne
    funext p
    by_contra h
    exact hn ⟨p, h⟩
  have hs : (∑ r : Fin 4, quarterRotaryInner (a r) (b r) (d * recallFrequency r)) ≤
      ∑ r : Fin 4, if r = p then (16 / 25 : ℝ) else 1 := by
    apply Finset.sum_le_sum
    intro r hr
    by_cases he : r = p
    · subst r
      rw [ite_eq_left (rfl : p = p)]
      exact quarterRotaryInner_mismatch _ _ hp _ (recall_angle_bound p d hd)
    · rw [ite_eq_right he]
      exact quarterRotaryInner_le_one _ _ _
  have hc : (∑ r : Fin 4, if r = p then (16 / 25 : ℝ) else 1) = 91 / 25 := by
    fin_cases p <;> norm_num [Fin.sum_univ_four]
  rw [hc] at hs
  linarith

example : (fun _ : Fin 4 => (0 : Fin 4)) ≠ (fun _ : Fin 4 => (1 : Fin 4)) ∧
    |(63 : ℝ)| ≤ 64 := by
  refine ⟨?_, by norm_num⟩
  intro h
  have he := congrArg (fun digits : Fin 4 → Fin 4 => digits 0) h
  contradiction

/-- The actual QKNorm/RoPE score separates any different raw symbol from any matching raw symbol.
Source: the full original score formula and proved compact-code margins, without an assumed score gap. -/
theorem recall_content_gap (alpha eps : ℝ) (heps : eps ≤ 2) (query other : Fin 256)
    (hne : query ≠ other) (s selected competitor : ℝ)
    (hselected : |selected - s| ≤ 64) (hcompetitor : |competitor - s| ≤ 64) :
    score alpha eps (applyRope 16 10000 s (recallRotaryCode (recallDigit query)))
        (applyRope 16 10000 competitor (recallRotaryCode (recallDigit other))) ≤
      score alpha eps (applyRope 16 10000 s (recallRotaryCode (recallDigit query)))
        (applyRope 16 10000 selected (recallRotaryCode (recallDigit query))) - Real.exp alpha / 50 := by
  rw [recallRotaryCode_score _ _ _ _ heps, recallRotaryCode_score _ _ _ _ heps]
  have hd : recallDigit query ≠ recallDigit other := fun he => hne (recallDigit_injective he)
  have hlo := mul_le_mul_of_nonneg_left
    (recall_matching_score_lower (recallDigit query) _ hselected) (Real.exp_pos alpha).le
  have hhi := mul_le_mul_of_nonneg_left
    (recall_mismatch_score_upper _ _ hd _ hcompetitor) (Real.exp_pos alpha).le
  linarith

example : (1 / 100000 : ℝ) ≤ 2 ∧ (0 : Fin 256) ≠ 1 ∧
    |(0 : ℝ) - 63| ≤ 64 ∧ |(62 : ℝ) - 63| ≤ 64 := by norm_num

/-- Among equal-key records, the actual compact RoPE score strictly prefers the later visible write.
Source: strict cosine monotonicity at every actual positive slow frequency, with the complete recall context. -/
theorem recall_matching_latest (alpha eps : ℝ) (heps : eps ≤ 2) (a : Fin 4 → Fin 4)
    (query earlier later : ℝ) (horder : earlier < later) (hvisible : later ≤ query)
    (hcontext : query - earlier ≤ 64) :
    score alpha eps (applyRope 16 10000 query (recallRotaryCode a))
        (applyRope 16 10000 earlier (recallRotaryCode a)) <
      score alpha eps (applyRope 16 10000 query (recallRotaryCode a))
        (applyRope 16 10000 later (recallRotaryCode a)) := by
  have hc (p : Fin 4) : Real.cos ((earlier - query) * recallFrequency p) <
      Real.cos ((later - query) * recallFrequency p) := by
    have hf := recallFrequency_pos p
    have hh : (query - earlier) * recallFrequency p ≤ 64 * (1 / 100 : ℝ) :=
      mul_le_mul_of_nonneg hcontext (recallFrequency_le p) (by linarith) (by norm_num)
    rw [show (earlier - query) * recallFrequency p = -((query - earlier) * recallFrequency p) from by ring,
      show (later - query) * recallFrequency p = -((query - later) * recallFrequency p) from by ring,
      Real.cos_neg, Real.cos_neg]
    apply Real.cos_lt_cos_of_nonneg_of_le_pi
    · nlinarith
    · nlinarith [Real.pi_gt_three]
    · nlinarith
  have hs : (∑ p : Fin 4, Real.cos ((earlier - query) * recallFrequency p)) <
      ∑ p : Fin 4, Real.cos ((later - query) * recallFrequency p) :=
    Finset.sum_lt_sum (fun p hp => (hc p).le) ⟨0, Finset.mem_univ _, hc 0⟩
  rw [recallRotaryCode_score _ _ _ _ heps, recallRotaryCode_score _ _ _ _ heps]
  simp only [quarterRotaryInner_self]
  exact mul_lt_mul_of_pos_left (div_lt_div_of_pos_right hs (by norm_num)) (Real.exp_pos alpha)

example : (1 / 100000 : ℝ) ≤ 2 ∧ (2 : ℝ) < 32 ∧ (32 : ℝ) ≤ 63 ∧
    (63 - 2 : ℝ) ≤ 64 := by norm_num

end Transformer.GPTMini.Semantics
