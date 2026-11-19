module

public import Transformer.Grokking.Convex.ConvexChannel.HiddenPath.Basic

/-!
# Answer-only CE is nonconvex for the two learned hidden stages

Source: `review/convex/moments.md`, CM74. At parameters ±(log 3, log 3), the
true next-object probability is 5/8. At their midpoint it is 1/2. Ordinary
answer CE is strictly larger at the midpoint, although every complete-path
likelihood is convex. No observed internal state is available to the answer
objective. This is a counterexample for this architecture, not for every
nonlinear convex learner or every future sequence parameterization.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.HiddenPath

open Transformer.GPTMini.Convex.Likelihood (routeSigmoid)

/-- Two trained stages give the stated exact marginal probability, CM74. -/
theorem positive_answer :
    answerProbability (Real.log 3, Real.log 3) true = 5 / 8 := by
  norm_num [answerProbability, pathProbability, bernoulli, signed,
    Fintype.sum_bool, routeSigmoid, Real.exp_neg,
    Real.exp_log (by norm_num : (0 : ℝ) < 3)]

/-- Reversing both stages preserves this answer probability, CM74. -/
theorem negative_answer :
    answerProbability (-Real.log 3, -Real.log 3) true = 5 / 8 := by
  norm_num [answerProbability, pathProbability, bernoulli, signed,
    Fintype.sum_bool, routeSigmoid, Real.exp_neg,
    Real.exp_log (by norm_num : (0 : ℝ) < 3)]

/-- The true midpoint has strictly less mass on the same output, CM74. -/
theorem midpoint_answer : answerProbability (0, 0) true = 1 / 2 := by
  norm_num [answerProbability, pathProbability, bernoulli, signed,
    Fintype.sum_bool, routeSigmoid]

/-- Ordinary next-object CE is not jointly convex in the separately learned
encoder and decoder parameters, CM74. Complete-path supervision cannot be
silently replaced by answer-only supervision in the earlier stack theorem. -/
theorem answerLoss_not_convex : ¬ ConvexOn ℝ Set.univ (answerLoss true) := by
  intro h
  have hj := h.2 (Set.mem_univ (Real.log 3, Real.log 3))
    (Set.mem_univ (-Real.log 3, -Real.log 3))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  have hm : (1 / 2 : ℝ) • ((Real.log 3, Real.log 3) : ℝ × ℝ) +
      (1 / 2 : ℝ) • ((-Real.log 3, -Real.log 3) : ℝ × ℝ) = (0, 0) := by
    apply Prod.ext <;> change (1 / 2 : ℝ) * Real.log 3 + 1 / 2 * (-Real.log 3) = 0 <;> ring
  rw [hm, answerLoss_eq, answerLoss_eq, answerLoss_eq,
    positive_answer, negative_answer, midpoint_answer] at hj
  have hl := Real.log_lt_log (by norm_num : (0 : ℝ) < 1 / 2)
    (by norm_num : (1 / 2 : ℝ) < 5 / 8)
  simp only [smul_eq_mul] at hj
  linarith

end Transformer.Grokking.Convex.ConvexChannel.HiddenPath
