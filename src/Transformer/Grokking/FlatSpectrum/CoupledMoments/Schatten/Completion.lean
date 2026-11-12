module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Intermediate
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Sharing

/-!
# An exact CE optimum can leave missing answers uniform

Source: `review/flat_spectrum.md`, CM10. Retain columns zero and two of the CM9
one-hot table and put uniform probabilities in columns one and three. The resulting
moment has smaller operator and nuclear norms, yet fits all eight observations.
It globally minimizes ordinary CE while missing target mass remains one half.
A fixed, explicitly correct sharing map repairs this witness without changing CE.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Matrix Finset
open scoped Matrix.Norms.L2Operator

/-- Eight observations, one representative from each of two column pairs (CM10). -/
def representativeCells (s : Fin 4 × Bool) : Fin 4 × Fin 4 :=
  (s.1, if s.2 then 2 else 0)

/-- The observed labels are restrictions of the CM9 table (CM10). -/
def representativeLabels (s : Fin 4 × Bool) : Bool :=
  intermediateLabel s.1 (representativeCells s).2

/-- Observed columns are one-hot; wholly unobserved columns are uniform (CM10). -/
noncomputable def maskedTable : Table (Fin 4) (Fin 4) Bool := fun i j c =>
  if j = 0 ∨ j = 2 then intermediateTable i j c else 1 / 2

/-- The corresponding moment has zeros in the two unobserved columns (CM10). -/
def maskedMatrix : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 1, 0; 1, 0, 1, 0; 1, 0, -1, 0; 1, 0, -1, 0]

/-- Uniform completion is still a probability table (CM10). -/
theorem masked_rows : Rows maskedTable := by
  constructor
  · intro i j c
    by_cases h : j = 0 ∨ j = 2
    · simpa [maskedTable, intermediateTable, h] using
        (rows_pointMass intermediateLabel).1 i j c
    · norm_num [maskedTable, h]
  · intro i j
    by_cases h : j = 0 ∨ j = 2
    · simpa [maskedTable, intermediateTable, h] using
        (rows_pointMass intermediateLabel).2 i j
    · norm_num [maskedTable, h]

/-- The mask is applied to the shared class probabilities themselves (CM10). -/
theorem masked_moment : moment (intermediateCode 0) maskedTable = maskedMatrix := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [moment, intermediateCode, maskedTable, intermediateTable,
      pointMass, intermediateLabel, maskedMatrix]

/-- The masked moment has nonzero squared singular value four (CM10). -/
theorem masked_gram_square :
    maskedMatrixᴴ * maskedMatrix * (maskedMatrixᴴ * maskedMatrix) =
      (4 : ℂ) • (maskedMatrixᴴ * maskedMatrix) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [maskedMatrix, Matrix.mul_apply, Fin.sum_univ_succ]

/-- Uniform missing rows reduce the two norms to two and four (CM10). -/
theorem masked_norms : ‖maskedMatrix‖ = 2 ∧ nuclearNorm maskedMatrix = 4 := by
  have ho := operator_eq_sqrt_of_gram_square (show (0 : ℝ) < 4 by norm_num)
    (X := maskedMatrix) (by
      intro h
      have he := congrFun (congrFun h 0) 0
      norm_num [maskedMatrix] at he) masked_gram_square
  have he : frobeniusSq maskedMatrix = 8 := by
    norm_num [frobeniusSq, maskedMatrix, Fin.sum_univ_succ]
  have hr := frobeniusSq_eq_rank_mul masked_gram_square
  have hn := nuclearNorm_eq_rank_mul_sqrt masked_gram_square
  rw [he] at hr
  norm_num at ho hr hn
  exact ⟨ho, by linarith⟩

/-- The uncertain completion lies inside the new intermediate budget (CM10). -/
theorem masked_budget : BudgetFeasible intermediateCode
    (fun _ => √8) (fun _ => 2 * √8) maskedTable := by
  have hs : (2 : ℝ) ≤ √8 := by
    nlinarith [Real.sqrt_nonneg (8 : ℝ), Real.sq_sqrt (show (0 : ℝ) ≤ 8 by norm_num)]
  refine ⟨⟨masked_rows, ?_⟩, ?_⟩
  · intro f
    fin_cases f
    change ‖moment (intermediateCode 0) maskedTable‖ ≤ √8
    rw [masked_moment, masked_norms.1]
    exact hs
  · intro f
    fin_cases f
    change nuclearNorm (moment (intermediateCode 0) maskedTable) ≤ 2 * √8
    rw [masked_moment, masked_norms.2]
    linarith

/-- Every observed answer is fitted exactly (CM10). -/
theorem masked_fit (s : Fin 4 × Bool) :
    maskedTable (representativeCells s).1 (representativeCells s).2 =
      ConvexRecall.targetLogits (representativeLabels s) := by
  rcases s with ⟨i, b⟩
  funext c
  cases b <;> simp [maskedTable, representativeCells, representativeLabels,
    intermediateTable, pointMass, ConvexRecall.targetLogits]

/-- A globally optimal ordinary-CE model can remain uncertain on an unseen answer,
even with the intermediate spectral budgets (CM10). -/
theorem masked_optimal_but_uncertain :
    BudgetFeasible intermediateCode (fun _ => √8) (fun _ => 2 * √8) maskedTable ∧
    IsMinOn (loss representativeCells representativeLabels)
      {P | BudgetFeasible intermediateCode (fun _ => √8) (fun _ => 2 * √8) P}
      maskedTable ∧ maskedTable 2 3 false = 1 / 2 ∧ intermediateTable 2 3 false = 1 := by
  refine ⟨masked_budget, (budget_fit_minimum _ _ _ representativeCells representativeLabels
    masked_budget masked_fit).2, ?_⟩
  norm_num [maskedTable, intermediateTable, pointMass, intermediateLabel]

/-- The proposed column sharing is fixed and correct for this witness (CM10). -/
def pairedRepresentative (j : Fin 4) : Fin 4 := if j.val < 2 then 0 else 2

/-- The full target satisfies the additional affine sharing constraint (CM10). -/
theorem intermediate_shared : ColumnSharing pairedRepresentative intermediateTable := by
  intro i j
  funext c
  fin_cases j <;>
    simp [pairedRepresentative, intermediateTable, pointMass, intermediateLabel]

/-- With correct fixed sharing, actual observed CE optimality recovers every entry
of the witness. The hypothesis supplies the representation that the budgets lacked (CM10). -/
theorem paired_recovery {P : Table (Fin 4) (Fin 4) Bool} (hP : Rows P)
    (hshare : ColumnSharing pairedRepresentative P)
    (hL : loss representativeCells representativeLabels P ≤ idealLoss representativeLabels) :
    P = intermediateTable := by
  funext i j c
  by_cases hj : j.val < 2
  · have he := shared_target_of_loss_le pairedRepresentative representativeCells
      representativeLabels hP hshare hL (s := (i, false)) (i := i) (j := j)
      (by simp [representativeCells, pairedRepresentative, hj])
    have hn : ¬ 2 ≤ j.val := by omega
    simpa [representativeLabels, representativeCells, intermediateTable, pointMass,
      ConvexRecall.targetLogits, intermediateLabel, hn] using congrFun he c
  · have he := shared_target_of_loss_le pairedRepresentative representativeCells
      representativeLabels hP hshare hL (s := (i, true)) (i := i) (j := j)
      (by simp [representativeCells, pairedRepresentative, hj])
    have hn : 2 ≤ j.val := by omega
    simpa [representativeLabels, representativeCells, intermediateTable, pointMass,
      ConvexRecall.targetLogits, intermediateLabel, hn] using congrFun he c

example := paired_recovery (rows_pointMass intermediateLabel) intermediate_shared
  (show loss representativeCells representativeLabels intermediateTable ≤
      idealLoss representativeLabels from (loss_eq_ideal_of_fit _ _ (by
        intro s
        rfl)).le)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
