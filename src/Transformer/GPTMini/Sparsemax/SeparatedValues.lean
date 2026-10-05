import Transformer.GPTMini.Sparsemax.ValueSpan
import Mathlib.Analysis.Calculus.Deriv.Pow

/-!
# Pure sparsemax with separated active values and an ordinary output loss

Derived from arXiv:1602.02068v2, §2.2 and §2.5, and the linear value
readout in `Attention.forward` at commit `73f8a0b`. A scalar squared
output error has a nonzero active-pair score derivative whenever its
output is wrong and the two active values differ. The supplied target
is a scalar task output, not an attention-position label.

The frozen values `(0, 0, 1)` in the earlier plateau violate this value
restriction. A constructed separated assignment `(-4, 4, 1)` keeps the
same initial output, ordinary output target and sparse weights, but gives
a corrective derivative. An explicit bounded score step reaches zero
error while leaving the third weight exactly zero. The values are changed
in this example; no rule enforcing their separation in every trainable
head, or global convergence of the model, is claimed.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Ordinary scalar task loss after the actual sparse value sum.
Source: `Attention.forward` at `73f8a0b`, before XSA; the illustrative
quadratic output objective is a derived example, not benchmark CE. -/
def scalarValueSquaredLoss {T : ℕ} (values : Fin T → ℝ) (target : ℝ)
    (i : Fin T) (scores : Fin T → ℝ) : ℝ :=
  (frozenValueReadout values (sparseWeights scores i) - target) ^ 2

/-- The scalar value sum has the actual value-difference tangent along
an active score transfer. Source: §2.5 of arXiv:1602.02068v2,
composed with the linear value readout at `73f8a0b`. -/
theorem scalarValueReadout_active_pair_hasDerivAt {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (values : Fin T → ℝ) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k) :
    HasDerivAt (fun t => frozenValueReadout values
      (sparseWeights (transferScores scores j k t) i)) (values j - values k) 0 := by
  have hvalue : frozenValueReadout values (basis j - basis k) = values j - values k := by
    simpa [pullbackValueGradient] using
      pullbackValueGradient_active_pair (ContinuousLinearMap.id ℝ ℝ) values j k
  simpa [Function.comp_def, hvalue] using
    (frozenValueReadout values).hasFDerivAt.comp_hasDerivAt (0 : ℝ)
      (sparseWeights_active_pair_hasDerivAt_row scores i j k hne hj hk)

/-- Active readout hypotheses are inhabited with a visible zero slot.
Source context: §2.5 and the frozen scalar value sum at `73f8a0b`. -/
example : HasDerivAt (fun t => frozenValueReadout (basis 0)
    (sparseWeights (transferScores twoActiveScores 0 1 t) 2)) 1 0 := by
  simpa [basis] using scalarValueReadout_active_pair_hasDerivAt twoActiveScores 2 0 1
    (basis 0) (by decide) (by norm_num [twoActiveScores_projection])
    (by norm_num [twoActiveScores_projection])

/-- The ordinary squared task loss has a computed corrective derivative,
not an assumed output-gradient separation. Source: arXiv:1602.02068v2,
§2.5, composed with the scalar readout at `73f8a0b` and ordinary calculus. -/
theorem scalarValueSquaredLoss_active_pair_hasDerivAt {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (values : Fin T → ℝ) (target : ℝ) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k) :
    HasDerivAt (fun t => scalarValueSquaredLoss values target i (transferScores scores j k t))
      (2 * (frozenValueReadout values (sparseWeights scores i) - target) *
        (values j - values k)) 0 := by
  have hpoint : transferScores scores j k 0 = scores := by
    funext n
    simp [transferScores]
  simpa [scalarValueSquaredLoss, hpoint, Pi.pow_def] using
    ((scalarValueReadout_active_pair_hasDerivAt scores i j k values hne hj hk).sub_const target).pow 2

/-- A sparse scalar task inhabits all derivative premises.
Source context: the ordinary squared output objective above. -/
example : HasDerivAt (fun t => scalarValueSquaredLoss (basis 0) 0 2
    (transferScores twoActiveScores 0 1 t)) 1 0 := by
  simpa [frozenValueReadout, sum_apply, Fin.sum_univ_three, basis,
    twoActiveScores_projection, smul_eq_mul] using
    scalarValueSquaredLoss_active_pair_hasDerivAt twoActiveScores 2 0 1 (basis 0) 0
      (by decide) (by norm_num [twoActiveScores_projection])
      (by norm_num [twoActiveScores_projection])

/-- A wrong scalar output and distinct active values cannot have zero
full score derivative. Source: arXiv:1602.02068v2, §2.5, with the actual
scalar squared task loss; no attention target or dense branch is used. -/
theorem scalarValueSquaredLoss_no_zero_score_derivative {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (values : Fin T → ℝ) (target : ℝ) (hne : j ≠ k)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k)
    (hv : values j ≠ values k)
    (hbad : frozenValueReadout values (sparseWeights scores i) ≠ target) :
    ¬ HasFDerivAt (𝕜 := ℝ) (scalarValueSquaredLoss values target i) 0 scores := by
  intro hz
  have hpoint : transferScores scores j k 0 = scores := by
    funext n
    simp [transferScores]
  have hz' : HasFDerivAt (𝕜 := ℝ) (scalarValueSquaredLoss values target i) 0
      (transferScores scores j k 0) := by rw [hpoint]; exact hz
  have hzero : HasDerivAt
      (fun t => scalarValueSquaredLoss values target i (transferScores scores j k t)) 0 0 := by
    simpa [Function.comp_def] using hz'.comp_hasDerivAt (0 : ℝ)
      (transferScores_hasDerivAt scores j k)
  have hn := mul_ne_zero (mul_ne_zero (by norm_num : (2 : ℝ) ≠ 0)
    (sub_ne_zero.mpr hbad)) (sub_ne_zero.mpr hv)
  exact hn ((scalarValueSquaredLoss_active_pair_hasDerivAt scores i j k values target
    hne hj hk).unique hzero)

/-- Constructed separated scalar values with the same initial output
as the collapsed example. Source context: `Attention.forward` at
`73f8a0b`; changing the values is the explicit derived restriction. -/
def separatedActiveValues : Fin 3 → ℝ :=
  fun n => if n = 0 then -4 else if n = 1 then 4 else 1

/-- The no-zero-derivative hypotheses hold with the original scalar
target and unchanged initial sparse row. Source context: the derived
value restriction on §2.5's score-gradient path. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (scalarValueSquaredLoss separatedActiveValues (1 / 2) 2)
    0 twoActiveScores := by
  apply scalarValueSquaredLoss_no_zero_score_derivative twoActiveScores 2 1 0
    separatedActiveValues (1 / 2) (by decide)
    (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
  · norm_num [separatedActiveValues]
  · norm_num [frozenValueReadout, sum_apply, Fin.sum_univ_three, separatedActiveValues,
      twoActiveScores_projection, smul_eq_mul]

/-- A small transfer for the separated readout, with no change to the
third raw score. Source: §2.5's active-pair direction, step `1/16`. -/
def valueRepairScores : Fin 3 → ℝ := transferScores twoActiveScores 1 0 (1 / 16)

/-- The actual repaired row remains sparse. Source: arXiv:1602.02068v2,
§2.2, via the proved unchanged-threshold active-pair transfer. -/
theorem valueRepairScores_projection : sparseWeights valueRepairScores 2 =
    fun n => if n = 2 then (0 : ℝ) else if n = 1 then 9 / 16 else 7 / 16 := by
  have h := sparseWeights_transfer_active_pair twoActiveScores 2 1 0 (1 / 16)
    (by decide) (by norm_num [twoActiveScores_projection])
    (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
    (by norm_num [twoActiveScores_projection])
  rw [valueRepairScores, h]
  funext n
  fin_cases n <;> norm_num [twoActiveScores_projection, basis]

/-- Both endpoints obey the same absolute cap below one half. Source
context: §2.2's derived score restriction, retained by this value repair. -/
theorem valueRepairScores_bounded :
    (∀ n, |twoActiveScores n| < (12 / 25 : ℝ)) ∧
    (∀ n, |valueRepairScores n| < (12 / 25 : ℝ)) := by
  constructor <;> intro n <;> fin_cases n <;>
    norm_num [twoActiveScores, valueRepairScores, transferScores, basis]

/-- A concrete pure-sparse correction after imposing value separation:
same positive initial output error, a negative actual derivative, then
zero error with the third probability still zero. Source: §2.2 and §2.5
of arXiv:1602.02068v2 and the scalar value readout at `73f8a0b`. -/
theorem separated_values_correct_output_with_sparse_step :
    scalarValueSquaredLoss separatedActiveValues (1 / 2) 2 twoActiveScores = 1 / 4 ∧
    HasDerivAt (fun t => scalarValueSquaredLoss separatedActiveValues (1 / 2) 2
      (transferScores twoActiveScores 1 0 t)) (-8) 0 ∧
    scalarValueSquaredLoss separatedActiveValues (1 / 2) 2 valueRepairScores = 0 ∧
    sparseWeights valueRepairScores 2 2 = 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · norm_num [scalarValueSquaredLoss, frozenValueReadout, sum_apply, Fin.sum_univ_three,
      separatedActiveValues, twoActiveScores_projection, smul_eq_mul]
  · have h := scalarValueSquaredLoss_active_pair_hasDerivAt twoActiveScores 2 1 0
      separatedActiveValues (1 / 2) (by decide) (by norm_num [twoActiveScores_projection])
      (by norm_num [twoActiveScores_projection])
    norm_num [frozenValueReadout, sum_apply, Fin.sum_univ_three, separatedActiveValues,
      twoActiveScores_projection, smul_eq_mul] at h
    exact h
  · norm_num [scalarValueSquaredLoss, frozenValueReadout, sum_apply, Fin.sum_univ_three,
      separatedActiveValues, valueRepairScores_projection, smul_eq_mul]
  · norm_num [valueRepairScores_projection]

end Transformer.GPTMini.Sparsemax
