import Transformer.GPTMini.Sparsemax.ActiveDirection

/-!
# Restrictions on the outer loss and value readout

Derived from arXiv:1602.02068v2, §2.5, `sparsemax_gradient` and the
active-pair curve of the actual projection. A nonzero score-to-weight
direction produces a corrective loss derivative only if the outer
derivative distinguishes those two active coordinates.

For a frozen attention value readout at commit `73f8a0b`, ordinary
backpropagation assigns the probability-coordinate derivative
`output_gradient(value_j)`. The relevant difference is therefore
`output_gradient(value_j - value_k)`, not merely a distance between the
values. These are conditional row results, not a global convergence claim
or a new training loss needing a target route.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- An arbitrary differentiable outer loss has the actual active-pair
directional derivative. Source: arXiv:1602.02068v2, §2.5,
`sparsemax_gradient`, composed with the outer derivative.
No assumption on differentiability at inactive support boundaries is used. -/
theorem outerLoss_active_pair_hasDerivAt {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (loss : (Fin T → ℝ) → ℝ)
    (gradient : (Fin T → ℝ) →L[ℝ] ℝ) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient (sparseWeights scores i)) :
    HasDerivAt (fun t => loss (sparseWeights (transferScores scores j k t) i))
      (gradient (basis j - basis k)) (0 : ℝ) := by
  have hpoint : transferScores scores j k 0 = scores := by
    funext n
    simp [transferScores]
  have hl' : HasFDerivAt (𝕜 := ℝ) loss gradient
      (sparseWeights (transferScores scores j k 0) i) := by rw [hpoint]; exact hl
  simpa [Function.comp_def] using hl'.comp_hasDerivAt (0 : ℝ)
    (sparseWeights_active_pair_hasDerivAt_row scores i j k hne hj hk)

/-- An outer coordinate readout inhabits the derivative hypotheses.
Source context: arXiv:1602.02068v2, §2.5, derived outer-loss condition. -/
example : HasDerivAt
    (fun t => sparseWeights (transferScores twoActiveScores 0 1 t) 2 0) 1 0 := by
  simpa [basis] using outerLoss_active_pair_hasDerivAt twoActiveScores 2 0 1
    (fun p => p 0) (ContinuousLinearMap.proj 0) (by decide)
    (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
    (ContinuousLinearMap.proj 0 : (Fin 3 → ℝ) →L[ℝ] ℝ).hasFDerivAt

/-- Separation of the outer derivatives on two active coordinates rules
out a zero full score derivative of the composed loss. Source: §2.5 of
arXiv:1602.02068v2, with ordinary outer differentiation. Unlike a supplied
route, this condition concerns the current task loss's own derivative. -/
theorem outerLoss_not_hasFDerivAt_zero {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (loss : (Fin T → ℝ) → ℝ)
    (gradient : (Fin T → ℝ) →L[ℝ] ℝ) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient (sparseWeights scores i))
    (hsep : gradient (basis j - basis k) ≠ 0) :
    ¬ HasFDerivAt (𝕜 := ℝ) (fun z => loss (sparseWeights z i)) 0 scores := by
  intro hz
  have hpoint : transferScores scores j k 0 = scores := by
    funext n
    simp [transferScores]
  have hz' : HasFDerivAt (𝕜 := ℝ) (fun z => loss (sparseWeights z i)) 0
      (transferScores scores j k 0) := by rw [hpoint]; exact hz
  have hzero : HasDerivAt (fun t => loss (sparseWeights (transferScores scores j k t) i))
      0 (0 : ℝ) := by
    simpa [Function.comp_def] using hz'.comp_hasDerivAt (0 : ℝ)
      (transferScores_hasDerivAt scores j k)
  exact hsep ((outerLoss_active_pair_hasDerivAt scores i j k loss gradient hne hj hk hl).unique hzero)

/-- The no-zero-loss-derivative hypotheses hold on a sparse row and a
coordinate-sensitive outer readout. Source context: arXiv:1602.02068v2, §2.5. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (fun z : Fin 3 → ℝ => sparseWeights z 2 0) 0
    twoActiveScores :=
  outerLoss_not_hasFDerivAt_zero _ 2 0 1 (fun p => p 0) (ContinuousLinearMap.proj 0)
    (by decide) (by norm_num [twoActiveScores_projection])
    (by norm_num [twoActiveScores_projection])
    (ContinuousLinearMap.proj 0 : (Fin 3 → ℝ) →L[ℝ] ℝ).hasFDerivAt (by norm_num [basis])

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The actual frozen linear value readout. Source: `Attention.forward`
at `73f8a0b`, one row of `weights @ values`, before XSA and output maps. -/
def frozenValueReadout {T : ℕ} (values : Fin T → E) : (Fin T → ℝ) →L[ℝ] E :=
  ∑ n, (ContinuousLinearMap.proj n).smulRight (values n)

/-- Pull back an output derivative through the frozen linear value
readout of attention. Source: `Attention` at commit `73f8a0b`, the sum
of attention weights times value vectors; §2.5 supplies the score path. -/
def pullbackValueGradient {T : ℕ} (gradient : E →L[ℝ] ℝ)
    (values : Fin T → E) : (Fin T → ℝ) →L[ℝ] ℝ :=
  gradient.comp (frozenValueReadout values)

/-- The readout pullback evaluates the probability perturbation against
the output sensitivities of all values. Source: `Attention` at `73f8a0b`,
linear value aggregation followed by ordinary differentiation. -/
theorem pullbackValueGradient_apply {T : ℕ} (gradient : E →L[ℝ] ℝ)
    (values : Fin T → E) (direction : Fin T → ℝ) :
    pullbackValueGradient gradient values direction = ∑ n, direction n * gradient (values n) := by
  simp [pullbackValueGradient, frozenValueReadout, sum_apply, map_sum, smul_eq_mul]

/-- The pullback is the derivative of an actual task loss composed with
the frozen value sum. Source: `Attention.forward` at `73f8a0b`, followed
by the usual chain rule; no attention-position targets are used. -/
theorem valueReadoutLoss_hasFDerivAt {T : ℕ} (values : Fin T → E)
    (loss : E → ℝ) (gradient : E →L[ℝ] ℝ) (p : Fin T → ℝ)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient (frozenValueReadout values p)) :
    HasFDerivAt (𝕜 := ℝ) (fun row => loss (frozenValueReadout values row))
      (pullbackValueGradient gradient values) p := by
  simpa only [Function.comp_def, pullbackValueGradient] using
    hl.comp p (frozenValueReadout values).hasFDerivAt

/-- A scalar readout task inhabits the chain-rule premise.
Source context: `Attention.forward` at `73f8a0b`, frozen value aggregation. -/
example : HasFDerivAt (𝕜 := ℝ)
    (fun p : Fin 3 → ℝ => (ContinuousLinearMap.id ℝ ℝ) (frozenValueReadout (basis 0) p))
    (pullbackValueGradient (ContinuousLinearMap.id ℝ ℝ) (basis 0))
    (sparseWeights twoActiveScores 2) :=
  valueReadoutLoss_hasFDerivAt _ _ _ _ (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

/-- Two active coordinates are distinguished exactly by the output
gradient applied to their value difference. Source: `Attention` at
`73f8a0b` and arXiv:1602.02068v2, §2.5, active-pair score differentiation. -/
theorem pullbackValueGradient_active_pair {T : ℕ} (gradient : E →L[ℝ] ℝ)
    (values : Fin T → E) (j k : Fin T) :
    pullbackValueGradient gradient values (basis j - basis k) = gradient (values j - values k) := by
  classical
  have hb (n : Fin T) : pullbackValueGradient gradient values (basis n) = gradient (values n) := by
    rw [pullbackValueGradient_apply]
    simp [basis]
  rw [(pullbackValueGradient gradient values).map_sub, hb j, hb k, gradient.map_sub]

/-- An ordinary task-gradient separation of active values excludes a
zero score derivative, without prescribing an attention route. Source:
`Attention` at `73f8a0b`, combined with §2.5 of arXiv:1602.02068v2.
The derivative hypothesis states how the frozen value readout enters the loss. -/
theorem value_sensitivity_not_hasFDerivAt_zero {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (loss : (Fin T → ℝ) → ℝ) (gradient : E →L[ℝ] ℝ)
    (values : Fin T → E) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k)
    (hl : HasFDerivAt (𝕜 := ℝ) loss (pullbackValueGradient gradient values) (sparseWeights scores i))
    (hsep : gradient (values j - values k) ≠ 0) :
    ¬ HasFDerivAt (𝕜 := ℝ) (fun z => loss (sparseWeights z i)) 0 scores := by
  apply outerLoss_not_hasFDerivAt_zero scores i j k loss _ hne hj hk hl
  simpa only [pullbackValueGradient_active_pair] using hsep

/-- Scalar output gradients can distinguish two active values.
Source context: `Attention` at `73f8a0b`, derived value-sensitivity condition. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun z : Fin 3 → ℝ => pullbackValueGradient (ContinuousLinearMap.id ℝ ℝ)
      (basis 0) (sparseWeights z 2)) 0 twoActiveScores :=
  value_sensitivity_not_hasFDerivAt_zero twoActiveScores 2 0 1
    (pullbackValueGradient (ContinuousLinearMap.id ℝ ℝ) (basis 0))
    (ContinuousLinearMap.id ℝ ℝ) (basis 0) (by decide)
    (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
    (pullbackValueGradient (ContinuousLinearMap.id ℝ ℝ) (basis 0)).hasFDerivAt
    (by norm_num [basis])

/-- Equal active values cancel the transfer signal for every output
gradient. Source: `Attention` at `73f8a0b`, linear value readout, combined
with arXiv:1602.02068v2, §2.5. This explains a limitation of score-only bounds. -/
theorem equal_active_values_cancel_transfer {T : ℕ} (gradient : E →L[ℝ] ℝ)
    (values : Fin T → E) (j k : Fin T) (heq : values j = values k) :
    pullbackValueGradient gradient values (basis j - basis k) = 0 := by
  rw [pullbackValueGradient_active_pair, heq, sub_self, map_zero]

/-- An inactive nonzero value can coexist with equal active values.
Source context: `Attention` at `73f8a0b`, derived cancellation example. -/
example : (fun n : Fin 3 => if n = 2 then (1 : ℝ) else 0) 0 =
    (fun n : Fin 3 => if n = 2 then (1 : ℝ) else 0) 1 := by norm_num

/-- Distinct values alone do not guarantee sensitivity: a nonzero value
difference can lie in the output derivative's kernel. Source context:
`Attention` at `73f8a0b`, the linear readout in the condition above. -/
example : (basis (1 : Fin 2) : Fin 2 → ℝ) ≠ 0 ∧
    (ContinuousLinearMap.proj 0 : (Fin 2 → ℝ) →L[ℝ] ℝ) (basis 1) = 0 := by
  constructor
  · intro h
    have he := congrFun h 1
    norm_num [basis] at he
  · norm_num [basis]

end Transformer.GPTMini.Sparsemax
