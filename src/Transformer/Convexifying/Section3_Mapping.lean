/-
# Recovering heads from convex rows

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.2, Proposition 1
(`prop:mapping`) and Appendix A.2.

The displayed formula applies to each *nonzero* row of `Z`.  The paper's
premise says only that there are `h` nonzero rows, not that they are the first
`h` rows; the printed `e_j` must therefore be replaced by `e_{k_j}` for an
enumeration of their actual token indices.  The one-row identity below is
the algebraic core of that corrected mapping.
-/

import Transformer.Convexifying.Section3_Scaling
import Transformer.Convexifying.Section3_ScalarCounterexample

open scoped BigOperators

namespace Transformer.Convexifying

/-- The value vector in Proposition 1, defined for a nonzero row `z`. -/
noncomputable def mappedValue {d : ℕ} (z : Vec d) : Vec d :=
  fun q => z q / Real.sqrt (norm₂ z)

/-- The output weight in Proposition 1. -/
noncomputable def mappedOutput {d : ℕ} (z : Vec d) : ℝ :=
  Real.sqrt (norm₂ z)

/-- A nonzero row has a nonzero denominator in the paper's mapping. -/
theorem mappedOutput_pos {d : ℕ} (z : Vec d) (hz : norm₂ z ≠ 0) :
    0 < mappedOutput z := by
  unfold mappedOutput
  exact Real.sqrt_pos.2 (lt_of_le_of_ne (norm₂_nonneg z) (Ne.symm hz))

/-- The corrected per-row reconstruction formula.  Its attention vector is
the basis vector at the row's *actual* token index `k`.
Source: arXiv:2211.11052v1, §3.2, `prop:mapping`. -/
theorem mapped_row_prediction {n d : ℕ} (X : Fin n → Vec d)
    (z : Vec d) (k : Fin n) (hz : norm₂ z ≠ 0) :
    scalarHead X (fun j => if j = k then 1 else 0)
      (mappedValue z) (mappedOutput z) =
    ∑ q, X k q * z q := by
  have ht : mappedOutput z ≠ 0 := (mappedOutput_pos z hz).ne'
  simp only [scalarHead, mappedValue, mappedOutput]
  simp
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro q _
  field_simp [show Real.sqrt (norm₂ z) ≠ 0 from ht]

/-- The hypothesis of `mapped_row_prediction` is satisfiable: a unit row
has nonzero Euclidean norm. -/
example : norm₂ (fun _ : Fin 1 => (1 : ℝ)) ≠ 0 := by
  norm_num [norm₂, normSq, Fin.sum_univ_one]

/-- The matrix whose only nonzero row is the *second* token. -/
def secondRowOnly : Fin 2 → Vec 1 :=
  fun k _ => if k = 1 then 1 else 0

/-- A datum reading only the second token. -/
def secondTokenDatum : Fin 2 → Vec 1 :=
  fun k _ => if k = 1 then 1 else 0

/-- The printed map with `h = 1` reads the first row and therefore gives
zero, although the convex matrix predicts one.  This demonstrates why an
enumeration of the support is required in Proposition 1.
Source: arXiv:2211.11052v1, §3.2, `prop:mapping`, displayed recovery map. -/
theorem printed_row_index_fails :
    scalarHead secondTokenDatum (fun j : Fin 2 => if j = 0 then 1 else 0)
      (mappedValue (secondRowOnly 0)) (mappedOutput (secondRowOnly 0)) = 0 ∧
    convexPrediction secondTokenDatum secondRowOnly = 1 := by
  constructor
  · norm_num [scalarHead, secondTokenDatum, secondRowOnly, mappedValue,
      mappedOutput, norm₂, normSq, Fin.sum_univ_two, Fin.sum_univ_one]
  · norm_num [convexPrediction, secondTokenDatum, secondRowOnly,
      Fin.sum_univ_two, Fin.sum_univ_one]

/-- A one-sample instance in which only token one is visible. -/
def secondTokenData : Data 1 2 1 := fun _ => secondTokenDatum

/-- Unit target for the support-indexing example. -/
def secondTokenTarget : Vec 1 := fun _ => 1

/-- The printed convex objective is minimized at `Z₁ = 7/8` and
`Z₀ = 0` for squared loss and `β = 1/8`. -/
noncomputable def secondTokenOptimalWeights : Fin 2 → Vec 1 :=
  fun k _ => if k = 1 then 7 / 8 else 0

private theorem secondToken_objective_formula (Z : Fin 2 → Vec 1) :
    printedConvexObjective secondTokenData secondTokenTarget squareLoss
      (1 / 8) Z =
    (1 / 2) * (Z 1 0 - 1) ^ 2 +
      (1 / 8) * (|Z 0 0| + |Z 1 0|) := by
  simp [printedConvexObjective, convexPrediction, secondTokenData,
    secondTokenDatum, secondTokenTarget, squareLoss, norm₂, normSq,
    Fin.sum_univ_two, Real.sqrt_sq_eq_abs]

/-- The value attained by the sparse convex optimum. -/
theorem secondToken_optimal_value :
    printedConvexObjective secondTokenData secondTokenTarget squareLoss
      (1 / 8) secondTokenOptimalWeights = 15 / 128 := by
  rw [secondToken_objective_formula]
  norm_num [secondTokenOptimalWeights]

/-- The displayed sparse matrix is a global minimizer of the corrected
convex problem. -/
theorem secondToken_optimal (Z : Fin 2 → Vec 1) :
    printedConvexObjective secondTokenData secondTokenTarget squareLoss
      (1 / 8) secondTokenOptimalWeights ≤
    printedConvexObjective secondTokenData secondTokenTarget squareLoss
      (1 / 8) Z := by
  rw [secondToken_optimal_value, secondToken_objective_formula]
  have hzero : 0 ≤ |Z 0 0| := abs_nonneg _
  rcases le_total 0 (Z 1 0) with hz | hz
  · rw [abs_of_nonneg hz]
    nlinarith [sq_nonneg (Z 1 0 - 7 / 8)]
  · rw [abs_of_nonpos hz]
    nlinarith [sq_nonneg (Z 1 0)]

/-- **Counterexample to the literal index choice in Proposition 1 at a
global convex optimum.**  The optimal matrix has exactly one nonzero row,
but that row is token one, not the first row.  The printed `j = 0` recovery
formula predicts zero where the optimum predicts `7/8`.  Replacing `e_j`
by the basis vector at the actual support index repairs the formula.
Source: arXiv:2211.11052v1, §3.2, `prop:mapping`, Appendix A.2. -/
theorem printed_mapping_fails_at_optimum :
    (∀ Z : Fin 2 → Vec 1,
      printedConvexObjective secondTokenData secondTokenTarget squareLoss
        (1 / 8) secondTokenOptimalWeights ≤
      printedConvexObjective secondTokenData secondTokenTarget squareLoss
        (1 / 8) Z) ∧
    (secondTokenOptimalWeights 0 0 = 0 ∧
      secondTokenOptimalWeights 1 0 ≠ 0) ∧
    scalarHead secondTokenDatum (fun j : Fin 2 => if j = 0 then 1 else 0)
      (mappedValue (secondTokenOptimalWeights 0))
      (mappedOutput (secondTokenOptimalWeights 0)) ≠
    convexPrediction secondTokenDatum secondTokenOptimalWeights := by
  constructor
  · exact secondToken_optimal
  constructor
  · norm_num [secondTokenOptimalWeights]
  · norm_num [scalarHead, secondTokenDatum, secondTokenOptimalWeights,
      mappedValue, mappedOutput, convexPrediction, norm₂, normSq,
      Fin.sum_univ_two, Fin.sum_univ_one]

end Transformer.Convexifying
