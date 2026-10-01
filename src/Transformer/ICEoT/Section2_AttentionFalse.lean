/-
# IC-EoT: ordinary dot-product attention need not be input convex

arXiv:2603.22095v2, §2.1 and §2.2.4. A one-dimensional head with
`Q=K=V=1` and input tokens `(t,-t)` has logits `(t²,-t²)` at its first
query. Its actual softmax output fails Jensen on the negative half-line.
This supplies a concrete witness for the source's architectural motivation.
-/

import Transformer.ICEoT.Section3_Closure
import Mathlib.Analysis.SpecialFunctions.Exp

noncomputable section

namespace Transformer.ICEoT

/-- The normalized difference of two softmax weights with logits `s,-s`;
§2.1's ordinary scaled dot-product attention, scalar head dimension one. -/
def softmaxDifference (s : ℝ) : ℝ :=
  (Real.exp s - Real.exp (-s)) / (Real.exp s + Real.exp (-s))

/-- The first-token attention output on `(t,-t)`, with unit query/key/value
matrices and scale `sqrt(d_h)=1`; §2.1 and §2.2.4. -/
def conventionalAttentionSlice (t : ℝ) : ℝ :=
  (Real.exp (t ^ 2) * t + Real.exp (-(t ^ 2)) * (-t)) /
    (Real.exp (t ^ 2) + Real.exp (-(t ^ 2)))

/-- Exact simplification of the actual weighted-value output, §2.1. -/
theorem conventionalAttentionSlice_eq (t : ℝ) :
    conventionalAttentionSlice t = t * softmaxDifference (t ^ 2) := by
  unfold conventionalAttentionSlice softmaxDifference
  ring

/-- A strict comparison needed for the Jensen witness; §2.2.4's
non-convex-attention obstruction. Both denominators are positive. -/
theorem softmaxDifference_one_lt_four : softmaxDifference 1 < softmaxDifference 4 := by
  unfold softmaxDifference
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  have he : Real.exp (1 : ℝ) * Real.exp (-4) < Real.exp 4 * Real.exp (-1) := by
    rw [← Real.exp_add, ← Real.exp_add]
    exact Real.exp_lt_exp.mpr (by norm_num)
  nlinarith

/-- Ordinary self-attention can be non-convex even with non-negative unit
Q/K/V weights. Jensen fails at tokens `(0,0)`, `(-2,2)` and their midpoint
`(-1,1)`: the scalar outputs are `0`, `-2*difference(4)`, and
`-difference(1)`. Source: §2.1 and §2.2.4. -/
theorem conventionalAttention_not_convex :
    ¬ ConvexOn ℝ Set.univ conventionalAttentionSlice := by
  intro h
  have hj := h.2 (Set.mem_univ (0 : ℝ)) (Set.mem_univ (-2 : ℝ))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  norm_num only [smul_eq_mul] at hj
  simp only [conventionalAttentionSlice_eq] at hj
  norm_num at hj
  linarith [softmaxDifference_one_lt_four]

end Transformer.ICEoT
