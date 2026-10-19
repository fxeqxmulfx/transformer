module

public import Transformer.Landscape.Neural.MLP.Model
public import Transformer.Landscape.SaddleAvoidance.LocalInequality

/-!
# The actual perceptron's regression and classification objectives

Sources: M9.2c and M9.3 of `landscape_grokking_plan.md`;
`python/src/lab/infrastructure/benchmarks/paired.py`, `row_losses`.

Half squared error sums the output coordinates and averages the equations.
A singleton answer index is the scalar objective; one-hot vector targets
give the vector control. Classification retains every competing answer.
Both objectives are analytic in all actual MLP parameters and satisfy the
proved local Lojasiewicz inequality. These properties also hold for the
analytic Rotate90 models, so they alone cannot explain the training gap.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Landscape.Neural.MLP

variable {D H O X : Type*} [Fintype D] [Fintype H] [Fintype O]

/-- M9.3's exact squared objective, with no division by answer dimension.
Empty splits use real division by zero; actual experiments are nonempty. -/
noncomputable def halfSquaredLoss (rows : Finset X) (input : X → D → ℝ)
    (target : X → O → ℝ) (w : Weights D H O) : ℝ :=
  (∑ r ∈ rows, ∑ q, (prediction w (input r) q - target r q) ^ 2) / (2 * rows.card)

/-- M9.3's full-class cross-entropy on the same complete affine outputs. -/
noncomputable def classificationLoss (rows : Finset X) (input : X → D → ℝ)
    (target : X → O) (w : Weights D H O) : ℝ :=
  meanCrossEntropy rows (fun r => prediction w (input r)) target

/-- The singleton output recovers the scalar half-MSE, with both input
operands and every trainable parameter retained (M9.3b, number readout). -/
theorem halfSquaredLoss_singleton (rows : Finset X) (input : X → D → ℝ)
    (target : X → Fin 1 → ℝ) (w : Weights D H (Fin 1)) :
    halfSquaredLoss rows input target w =
      meanSquaredError rows (fun r => prediction w (input r) 0) (fun r => target r 0) / 2 := by
  simp only [halfSquaredLoss, Fin.sum_univ_one, meanSquaredError]
  ring

/-- Exact conversion to the scalar observer's unrounded residue-value MSE
at modulus 97. Decoding multiplies normalized errors by 48, and half-MSE
therefore differs by a factor of 4608 (M9.3b, numeric-error units). This
concerns continuous predictions, before rounding or accuracy evaluation. -/
theorem scalar_numeric_mse_mod97 (rows : Finset X) (input : X → D → ℝ)
    (target : X → Fin 1 → ℝ) (w : Weights D H (Fin 1)) :
    meanSquaredError rows (fun r => (prediction w (input r) 0 + 1) * 48)
      (fun r => (target r 0 + 1) * 48) = 4608 * halfSquaredLoss rows input target w := by
  have hrow (r : X) :
      ((prediction w (input r) 0 + 1) * 48 - (target r 0 + 1) * 48) ^ 2 =
        2304 * (prediction w (input r) 0 - target r 0) ^ 2 := by ring
  simp only [meanSquaredError, halfSquaredLoss, Fin.sum_univ_one]
  simp_rw [hrow]
  rw [← Finset.mul_sum]
  ring

/-- Squared loss is bounded below in the actual weights, without a fit or
bounded-parameter hypothesis (M9.2c). -/
theorem halfSquaredLoss_nonneg (rows : Finset X) (input : X → D → ℝ)
    (target : X → O → ℝ) (w : Weights D H O) : 0 ≤ halfSquaredLoss rows input target w := by
  exact div_nonneg (Finset.sum_nonneg fun r _ =>
    Finset.sum_nonneg fun q _ => sq_nonneg (prediction w (input r) q - target r q))
    (mul_nonneg (by norm_num) (Nat.cast_nonneg _))

/-- Cross-entropy has the same lower bound; it keeps a separate free output
for every answer (M9.2c and M9.3b, vector CE). -/
theorem classificationLoss_nonneg (rows : Finset X) (input : X → D → ℝ)
    (target : X → O) (w : Weights D H O) : 0 ≤ classificationLoss rows input target w := by
  exact meanCrossEntropy_nonneg rows _ target

/-- Actual scalar and vector squared losses are analytic everywhere, with
arbitrary fixed input codes, targets and train/held-out row sets (M9.2c). -/
theorem analyticOnNhd_halfSquaredLoss (rows : Finset X) (input : X → D → ℝ)
    (target : X → O → ℝ) :
    AnalyticOnNhd ℝ (halfSquaredLoss (H := H) rows input target) Set.univ := by
  intro x hx
  exact (Finset.analyticAt_fun_sum rows fun r _ =>
    Finset.analyticAt_fun_sum _ fun q _ =>
      ((analyticAt_prediction x (input r) q).fun_sub analyticAt_const).fun_pow 2).div_const

/-- Actual classification loss is analytic in the complete MLP parameter
space; there is no assumed analytic-network hypothesis (M9.2c). -/
theorem analyticOnNhd_classificationLoss (rows : Finset X) (input : X → D → ℝ)
    (target : X → O) :
    AnalyticOnNhd ℝ (classificationLoss (H := H) rows input target) Set.univ := by
  intro x hx
  exact analyticAt_meanCrossEntropy rows (fun r _ q => analyticAt_prediction x (input r) q) target

/-- The local inequality holds at every scalar/vector MSE parameter vector.
It does not assert that gradient training reaches an interpolating solution
(M9.2c and the research task's representability/access distinction). -/
theorem exists_isLojasiewiczAt_halfSquaredLoss (rows : Finset X) (input : X → D → ℝ)
    (target : X → O → ℝ) (x : Weights D H O) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧
      SaddleAvoidance.IsLojasiewiczAt (halfSquaredLoss rows input target) x a := by
  exact SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt
    (analyticOnNhd_halfSquaredLoss rows input target x (Set.mem_univ x))

/-- The same local inequality holds for the untied MLP classifier, without
implying that an empirical minimum generalizes (M9.2c). -/
theorem exists_isLojasiewiczAt_classificationLoss (rows : Finset X) (input : X → D → ℝ)
    (target : X → O) (x : Weights D H O) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧
      SaddleAvoidance.IsLojasiewiczAt (classificationLoss rows input target) x a := by
  exact SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt
    (analyticOnNhd_classificationLoss rows input target x (Set.mem_univ x))

-- Nonempty splits and nonconstant inputs/targets instantiate both losses.
example : AnalyticOnNhd ℝ
    (halfSquaredLoss (H := Fin 3) (Finset.univ : Finset (Fin 2))
      (fun r i : Fin 2 => (r.val : ℝ) + i.val)
      (fun (r : Fin 2) (q : Fin 1) => (r.val : ℝ) + q.val)) Set.univ := by
  exact analyticOnNhd_halfSquaredLoss _ _ _

example : AnalyticOnNhd ℝ
    (classificationLoss (H := Fin 3) (Finset.univ : Finset (Fin 3 × Fin 3))
      (fun r => oneHotInput r.1 r.2) (fun r => r.1)) Set.univ := by
  exact analyticOnNhd_classificationLoss _ _ _

example : ∃ a, 0 ≤ a ∧ a < 1 ∧ SaddleAvoidance.IsLojasiewiczAt
    (halfSquaredLoss (H := Fin 3) (Finset.univ : Finset (Fin 2))
      (fun r i : Fin 2 => (r.val : ℝ) + i.val)
      (fun (r : Fin 2) (q : Fin 1) => (r.val : ℝ) + q.val)) 0 a := by
  exact exists_isLojasiewiczAt_halfSquaredLoss _ _ _ 0

example : ∃ a, 0 ≤ a ∧ a < 1 ∧ SaddleAvoidance.IsLojasiewiczAt
    (classificationLoss (H := Fin 3) (Finset.univ : Finset (Fin 3 × Fin 3))
      (fun r => oneHotInput r.1 r.2) (fun r => r.1)) 0 a := by
  exact exists_isLojasiewiczAt_classificationLoss _ _ _ 0

-- Scalar targets may vary across rows; the conversion keeps every row.
example (w : Weights (Fin 2) (Fin 3) (Fin 1)) :
    meanSquaredError (Finset.univ : Finset (Fin 2))
      (fun r => (prediction w (fun i : Fin 2 => (r.val : ℝ) + i.val) 0 + 1) * 48)
      (fun r => ((r.val : ℝ) + 1) * 48) =
        4608 * halfSquaredLoss (Finset.univ : Finset (Fin 2))
          (fun r i : Fin 2 => (r.val : ℝ) + i.val)
          (fun (r : Fin 2) (q : Fin 1) => (r.val : ℝ) + q.val) w := by
  simpa using scalar_numeric_mse_mod97 (Finset.univ : Finset (Fin 2))
    (fun r i : Fin 2 => (r.val : ℝ) + i.val)
    (fun (r : Fin 2) (q : Fin 1) => (r.val : ℝ) + q.val) w

end Transformer.Landscape.Neural.MLP
