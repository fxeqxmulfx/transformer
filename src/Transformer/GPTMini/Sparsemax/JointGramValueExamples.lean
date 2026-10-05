import Transformer.GPTMini.Sparsemax.GramValueExampleGrams

/-!
# A concrete joint embedding, attention and value update

Derived finite prototype for sparsemax arXiv:1602.02068v2, Eq. (1), and
the original `attn @ v` at `73f8a0b`. Both Q/K embedding families change,
the second attention row acquires an earlier active position, and a common
learned value table changes from `(1, 0)` to `(0, 2)`. Outputs move from
`(1, 0)` to `(0, 1)` in the exact convex Gram/output coordinates.

At their midpoint, actual attention is `[[1, 0], [1/4, 3/4]]`, decoded
values are `(1/2, 1/2)`, and outputs are `(1/2, 1/2)`. The literal mean
of the original value tables instead gives second output `7/8`. Thus the
coordinate change is real and nonlinear, while output training is affine.
One value table is shared across both causal rows; no private row values,
task loss, FFN, training experiment or arbitrary-context claim is used.
All outputs precede XSA and the output projection. Original penalties on
decoded values and a fixed embedding width are not preserved. The two token
IDs index one learned value table; outputs on a new context are not covered
by this finite-context certificate.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Initial output table, also the initial values because attention is identity.
Source: the derived two-token joint coordinate witness for `attn @ v`. -/
def valueExampleStartOutputs : Matrix (Fin 2) (Fin 1) ℝ :=
  fun i _ => if i = 0 then 1 else 0

/-- Changed output table, with a different value at each causal query.
Source: the derived two-token joint coordinate witness. -/
def valueExampleStopOutputs : Matrix (Fin 2) (Fin 1) ℝ :=
  fun i _ => if i = 0 then 0 else 1

/-- One common changed value table, decoded from the changed attention and outputs.
Source: the exact inverse value coordinates for `attn @ v` at `73f8a0b`. -/
def valueExampleStopValues : Matrix (Fin 2) (Fin 1) ℝ :=
  fun i _ => if i = 0 then 0 else 2

/-- The true midpoint in the convex learned Gram/output parameter domain.
Source: the derived exact value-coordinate architecture. -/
def valueExampleMidPoint : EmbeddingGram 2 × Matrix (Fin 2) (Fin 1) ℝ :=
  (1 / 2 : ℝ) • (valueExampleStartGram, valueExampleStartOutputs) +
    (1 / 2 : ℝ) • (valueExampleStopGram, valueExampleStopOutputs)

/-- Initial original values give the stated actual causal outputs.
Source: the ordinary `attn @ v`, using the proved identity sparsemax matrix. -/
theorem valueExampleStart_actual_output :
    causalGramValueOutput valueExampleStartGram valueExampleStartOutputs = valueExampleStartOutputs := by
  unfold causalGramValueOutput
  rw [valueExampleStartGram_attention, Matrix.one_mul]

/-- Changed original values give the changed actual causal outputs.
Source: the ordinary shared-table `attn @ v`, without private row values. -/
theorem valueExampleStop_actual_output :
    causalGramValueOutput valueExampleStopGram valueExampleStopValues = valueExampleStopOutputs := by
  unfold causalGramValueOutput
  rw [valueExampleStopGram_attention]
  ext i d
  fin_cases i <;> norm_num [Matrix.mul_apply, Fin.sum_univ_two,
    valueExampleStopValues, valueExampleStopOutputs]

/-- The initial decoded values are exactly the initial original table.
Source: the proved bijective coordinate recovery for actual `attn @ v`. -/
theorem valueExampleStart_recovered :
    recoverGramValues valueExampleStartGram valueExampleStartOutputs = valueExampleStartOutputs := by
  exact (recoverGramValues_unique 4 (1 / 2) _ _ _ (by norm_num)
    valueExampleStartGram_mem_domain valueExampleStart_actual_output).symm

/-- The changed decoded values are exactly `(0, 2)`, not a frozen table.
Source: the same exact coordinate recovery at the changed embedding Gram. -/
theorem valueExampleStop_recovered :
    recoverGramValues valueExampleStopGram valueExampleStopOutputs = valueExampleStopValues := by
  exact (recoverGramValues_unique 4 (1 / 2) _ _ _ (by norm_num)
    valueExampleStopGram_mem_domain valueExampleStop_actual_output).symm

/-- The original learned values really change as both embedding families change.
Source: the actual shared value tables in the derived joint prototype. -/
theorem valueExample_values_change : valueExampleStartOutputs ≠ valueExampleStopValues := by
  intro h
  have hi := congrArg (fun V : Matrix (Fin 2) (Fin 1) ℝ => V 1 0) h
  norm_num [valueExampleStartOutputs, valueExampleStopValues] at hi

/-- The entire joint midpoint remains inside the structural convex domain.
Source: normalized learned Grams and the positive self-weight floor. -/
theorem valueExampleMidPoint_mem_domain :
    valueExampleMidPoint ∈ jointGramValueDomain 2 1 4 (1 / 2) := by
  exact jointGramValueDomain_convex 2 1 4 (1 / 2)
    valueExampleStartGram_mem_domain valueExampleStopGram_mem_domain
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)

/-- The true midpoint sparsemax matrix is the mean endpoint attention.
Source: actual Eq. (1) on the normalized learned Gram segment. -/
theorem valueExampleMidPoint_attention : causalGramAttention valueExampleMidPoint.1 =
    Matrix.of (fun i j => if i = 0 then (if j = 0 then 1 else 0)
      else (if j = 0 then 1 / 4 else 3 / 4)) := by
  rw [causalGramAttention_normalized 4 _ valueExampleMidPoint_mem_domain.1]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [valueExampleMidPoint, valueExampleStartGram, valueExampleStopGram,
      featureGram_apply, valueExampleStartFeatures, valueExampleStopFeatures, Fin.sum_univ_two]

/-- Actual jointly decoded outputs equal the mean output table.
Source: the proved exact convex coordinate change, not an assumed output variable. -/
theorem valueExampleMidPoint_outputs : jointGramValueForward valueExampleMidPoint =
    Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (1 / 2 : ℝ)) := by
  rw [jointGramValueForward_eq_outputs 4 (1 / 2) _ (by norm_num) valueExampleMidPoint_mem_domain]
  ext i d
  fin_cases i <;> norm_num [valueExampleMidPoint, valueExampleStartOutputs, valueExampleStopOutputs]

/-- A common midpoint value table realizes the actual midpoint output.
Source: the ordinary attention/value sum, using the proved midpoint sparsemax. -/
theorem valueExampleMidPoint_actual_output : causalGramValueOutput valueExampleMidPoint.1
    (Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (1 / 2 : ℝ))) = valueExampleMidPoint.2 := by
  unfold causalGramValueOutput
  rw [valueExampleMidPoint_attention]
  ext i d
  fin_cases i <;> norm_num [Matrix.mul_apply, Fin.sum_univ_two,
    valueExampleMidPoint, valueExampleStartOutputs, valueExampleStopOutputs]

/-- The true recovered midpoint values are `(1/2, 1/2)`.
Source: unique exact inverse recovery at the feasible midpoint Gram. -/
theorem valueExampleMidPoint_recovered : recoverGramValues valueExampleMidPoint.1
    valueExampleMidPoint.2 = Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (1 / 2 : ℝ)) := by
  exact (recoverGramValues_unique 4 (1 / 2) _ _ _ (by norm_num)
    valueExampleMidPoint_mem_domain valueExampleMidPoint_actual_output).symm

/-- Literal affine interpolation in original values gives a different output.
Source: the actual `attn @ v`; this distinguishes the new convex coordinates
from an incorrect claim of convexity in the original value parameters. -/
theorem valueExample_literal_value_midpoint :
    causalGramValueOutput valueExampleMidPoint.1
      ((1 / 2 : ℝ) • valueExampleStartOutputs + (1 / 2 : ℝ) • valueExampleStopValues) 1 0 = 7 / 8 := by
  unfold causalGramValueOutput
  rw [valueExampleMidPoint_attention]
  norm_num [Matrix.mul_apply, Fin.sum_univ_two, valueExampleStartOutputs, valueExampleStopValues]

/-- The midpoint inverse also has a concrete nonzero determinant certificate.
Source: the structural recovery guarantee, evaluated on the actual middle
sparsemax matrix while both embeddings and values change. -/
theorem valueExampleMidPoint_det : (causalGramAttention valueExampleMidPoint.1).det = 3 / 4 := by
  rw [Matrix.det_fin_two, valueExampleMidPoint_attention]
  norm_num

end Transformer.GPTMini.Sparsemax
