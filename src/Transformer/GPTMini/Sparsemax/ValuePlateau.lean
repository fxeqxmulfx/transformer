import Transformer.GPTMini.Sparsemax.ActiveDirection

/-!
# A task-loss plateau that survives a sub-unit score range

Counterexample to the proposed inference that excluding singleton
sparsemax rows removes all flat outer losses. This is a derived example,
not a claim refuted from arXiv:1602.02068v2. Its §2.2 closed form and §2.5
Jacobian explain the plateau of the actual causal variational projection.

Freeze scalar values at `(0, 0, 1)` in the linear attention readout
`Attention.forward` at commit `73f8a0b`, and use an ordinary output target
`1/2`. The two active values coincide. The squared output error is `1/4`
on an open neighborhood even though a zero-error score vector exists
under the same absolute score bound `2/5`. The score-to-weight map itself
has a nonzero active direction. This is a frozen row counterexample, not
a stationary point of the entire trainable transformer.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open Filter
open scoped BigOperators Topology

/-- The normalized threshold candidate for the first two visible slots.
Source: arXiv:1602.02068v2, §2.2, `threshold_closedform`, conditional
on those two slots being the support; the condition is proved below. -/
def leaderThreshold (scores : Fin 3 → ℝ) : ℝ :=
  (scores 0 + scores 1 - 1) / 2

/-- With two scores above their candidate threshold and the third below
it, the actual third probability is zero. Source: arXiv:1602.02068v2,
§2.2, `sparsemax_closedform`, including the inactive boundary. -/
theorem sparseWeights_third_zero_of_two_leaders (scores : Fin 3 → ℝ)
    (h0 : leaderThreshold scores < scores 0)
    (h1 : leaderThreshold scores < scores 1)
    (h2 : scores 2 ≤ leaderThreshold scores) : sparseWeights scores 2 2 = 0 := by
  have hs0 : 0 ≤ scores 0 - leaderThreshold scores := by linarith
  have hs1 : 0 ≤ scores 1 - leaderThreshold scores := by linarith
  have hs2 : scores 2 - leaderThreshold scores ≤ 0 := by linarith
  have hsum : ∑ n : Fin 3, thresholdWeights scores 2 (leaderThreshold scores) n = 1 := by
    norm_num [Fin.sum_univ_three, thresholdWeights]
    rw [max_eq_left hs0, max_eq_left hs1, max_eq_right hs2]
    unfold leaderThreshold
    ring
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hsum]
  norm_num [thresholdWeights, max_eq_right hs2]

/-- All support premises hold strictly at the bounded sparse example.
Source context: arXiv:1602.02068v2, §2.2, derived two-leader example. -/
example : leaderThreshold twoActiveScores < twoActiveScores 0 ∧
    leaderThreshold twoActiveScores < twoActiveScores 1 ∧
    twoActiveScores 2 ≤ leaderThreshold twoActiveScores := by
  norm_num [leaderThreshold, twoActiveScores]

/-- Ordinary scalar-output squared loss with frozen values `(0, 0, 1)`.
Source: linear value aggregation in `Attention.forward` at `73f8a0b`;
the illustrative output target is not an attention-position label. -/
def unseenValueLoss (scores : Fin 3 → ℝ) : ℝ :=
  (sparseWeights scores 2 2 - 1 / 2) ^ 2

/-- The output error stays positive throughout the two-leader region.
Source: arXiv:1602.02068v2, §2.2, composed with the frozen scalar
value readout of `Attention.forward` at `73f8a0b`. -/
theorem unseenValueLoss_eq_quarter (scores : Fin 3 → ℝ)
    (h0 : leaderThreshold scores < scores 0)
    (h1 : leaderThreshold scores < scores 1)
    (h2 : scores 2 ≤ leaderThreshold scores) : unseenValueLoss scores = 1 / 4 := by
  unfold unseenValueLoss
  rw [sparseWeights_third_zero_of_two_leaders scores h0 h1 h2]
  norm_num

/-- The positive-loss region has a concrete inhabited instance.
Source context: §2.2's closed form and the frozen scalar value readout. -/
example : unseenValueLoss twoActiveScores = 1 / 4 :=
  unseenValueLoss_eq_quarter _ (by norm_num [leaderThreshold, twoActiveScores])
    (by norm_num [leaderThreshold, twoActiveScores])
    (by norm_num [leaderThreshold, twoActiveScores])

/-- Strict support inequalities persist on a full score neighborhood.
Source: arXiv:1602.02068v2, §2.2 and §2.5, the open region between
support-splitting points; all three score coordinates may vary. -/
theorem unseenValueLoss_eventually_quarter :
    ∀ᶠ scores in 𝓝 twoActiveScores, unseenValueLoss scores = 1 / 4 := by
  have hc : Continuous leaderThreshold :=
    ((continuous_apply 0).add (continuous_apply 1)).sub continuous_const |>.div_const 2
  have h0 : ∀ᶠ scores in 𝓝 twoActiveScores, leaderThreshold scores < scores 0 :=
    hc.continuousAt.eventually_lt (continuous_apply 0).continuousAt
      (by norm_num [leaderThreshold, twoActiveScores])
  have h1 : ∀ᶠ scores in 𝓝 twoActiveScores, leaderThreshold scores < scores 1 :=
    hc.continuousAt.eventually_lt (continuous_apply 1).continuousAt
      (by norm_num [leaderThreshold, twoActiveScores])
  have h2 : ∀ᶠ scores in 𝓝 twoActiveScores, scores 2 < leaderThreshold scores :=
    (continuous_apply 2).continuousAt.eventually_lt hc.continuousAt
      (by norm_num [leaderThreshold, twoActiveScores])
  exact ((h0.and h1).and h2).mono fun scores h =>
    unseenValueLoss_eq_quarter scores h.1.1 h.1.2 (le_of_lt h.2)

/-- The ordinary output loss has a zero full score derivative on this
positive-error plateau. Source: the frozen attention readout at `73f8a0b`
and §2.5 of arXiv:1602.02068v2; no custom backward rule is used. -/
theorem unseenValueLoss_hasFDerivAt_zero :
    HasFDerivAt (𝕜 := ℝ) unseenValueLoss 0 twoActiveScores :=
  (hasFDerivAt_const (1 / 4 : ℝ) twoActiveScores).congr_of_eventuallyEq
    unseenValueLoss_eventually_quarter

/-- Alternative scores whose scalar readout matches the same target.
Source context: arXiv:1602.02068v2, §2.2, derived closed-form example. -/
def recoveredValueScores : Fin 3 → ℝ :=
  fun n => if n = 2 then 1 / 6 else -(1 / 12)

/-- The alternative actual projection has third probability one half.
Source: arXiv:1602.02068v2, §2.2, `sparsemax_closedform`, evaluated
with threshold -1/3; all three probabilities are positive. -/
theorem recoveredValueScores_projection : sparseWeights recoveredValueScores 2 =
    fun n => if n = 2 then (1 / 2 : ℝ) else 1 / 4 := by
  have hsum : ∑ n : Fin 3, thresholdWeights recoveredValueScores 2 (-(1 / 3)) n = 1 := by
    norm_num [Fin.sum_univ_three, thresholdWeights, recoveredValueScores]
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hsum]
  funext n
  fin_cases n <;> norm_num [thresholdWeights, recoveredValueScores]

/-- Both the plateau and the zero-error alternative obey the same
absolute bound below one half. Source context: §2.2's derived bounded
score restriction, applied to the frozen scalar attention example. -/
theorem valuePlateau_scores_bounded :
    (∀ n, |twoActiveScores n| ≤ (2 / 5 : ℝ)) ∧
    (∀ n, |recoveredValueScores n| ≤ (2 / 5 : ℝ)) := by
  constructor <;> intro n <;> fin_cases n <;>
    norm_num [twoActiveScores, recoveredValueScores]

/-- Counterexample to the proposed sufficiency of a sub-unit score
range for removing every bad outer-loss plateau. Source context:
arXiv:1602.02068v2, §2.2 and §2.5, and the frozen value readout at
`73f8a0b`; the active score-to-weight direction is still nonzero. -/
theorem bounded_scores_positive_stationary_output_loss :
    unseenValueLoss twoActiveScores = 1 / 4 ∧
    HasFDerivAt (𝕜 := ℝ) unseenValueLoss 0 twoActiveScores ∧
    (¬ HasFDerivAt (𝕜 := ℝ) (fun z : Fin 3 → ℝ => sparseWeights z 2) 0 twoActiveScores) ∧
    unseenValueLoss recoveredValueScores = 0 := by
  refine ⟨?_, unseenValueLoss_hasFDerivAt_zero, ?_, ?_⟩
  · norm_num [unseenValueLoss, twoActiveScores_projection]
  · exact sparseWeights_not_hasFDerivAt_zero_of_two_active _ 2 0 1 (by decide)
      (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
  · norm_num [unseenValueLoss, recoveredValueScores_projection]

end Transformer.GPTMini.Sparsemax
