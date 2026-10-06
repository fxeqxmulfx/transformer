import Transformer.GPTMini.Sparsemax.EnergyCoupledMemory
import Mathlib.Data.Matrix.ColumnRowPartitioned

/-!
# Nonidentity coupled-energy witnesses with all common values learned

New finite examples for the restriction after arXiv:1602.02068v2, Eq. (1).
There are three memory slots and two possible path edges. The aggregate
edge budget is 1/8, the self-weight floor is 3/4, and the value energy is 6.
An edge connects two slots with answer 1; the remaining answer is -2.

Both possible connections are feasible in the same convex domain. PSD is
proved by a diagonal-plus-outer-product decomposition and a genuine matrix
congruence, not by assuming a factorization certificate. The actual original
common value table is recovered, and happens to equal the answers in these
witnesses. Subsequent necessity results show the labels selecting the edge.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The two finite attention choices retain independently allowed Q/K norm coordinates.
Source: new energy-coupled witnesses after arXiv:1602.02068v2, Eq. (1). -/
def taskEnergyParameters (edge : Fin 2) : LocalMemoryParameters 2 :=
  ((fun e => if e = edge then 1 / 8 else 0), (fun _ => 1))

/-- Ordinary observed answers: equal on the chosen adjacent pair, different elsewhere.
Source: regression witnesses for the new restriction after arXiv:1602.02068v2, Eq. (1).
These are output targets, not supervised attention routes. -/
def taskEnergyTarget (edge : Fin 2) : Matrix (Fin 3) (Fin 1) ℝ :=
  Matrix.of (fun i => fun _ => if i = edge.castSucc ∨ i = edge.succ then 1 else -2)

/-- Three-slot actual embedding scores, including both independently learned edges.
Source: the path permutation mixture before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryThree_scores (p : LocalMemoryParameters 2) :
    memoryGramScores (localMemoryGram p) =
      !![1 - p.1 0, p.1 0, 0;
         p.1 0, 1 - p.1 0 - p.1 1, p.1 1;
         0, p.1 1, 1 - p.1 1] := by
  rw [localMemoryGram_scores]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [localMemoryCore_scores_apply, Fintype.sum_option, localMemoryWeights,
      localMemoryPermutation, Fin.sum_univ_two, Equiv.swap_apply_def] <;> ring

/-- Either choice is a genuine feasible embedding with a positive learned edge.
Source: the local budget restriction for arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyParameters_mem (edge : Fin 2) :
    taskEnergyParameters edge ∈ incidentMemoryParameterDomain 2 4 (3 / 4) := by
  constructor
  · constructor
    · intro e
      fin_cases edge <;> fin_cases e <;> norm_num [taskEnergyParameters]
    · intro i
      fin_cases edge <;> fin_cases i <;>
        norm_num [taskEnergyParameters, localIncidentWeight, Fin.sum_univ_two]
  · intro x
    norm_num [taskEnergyParameters]

/-- Both witnesses use exactly the same aggregate edge budget.
Source: the new fixed-total restriction after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyParameters_budget (edge : Fin 2) :
    (∑ e, (taskEnergyParameters edge).1 e) = 1 / 8 := by
  fin_cases edge <;> norm_num [taskEnergyParameters, Fin.sum_univ_two]

/-- The cross scores themselves are PSD, proved independently of the full Q/K Gram.
Source: the new value-energy witnesses after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyAttention_posSemidef (edge : Fin 2) :
    (memoryGramScores (localMemoryGram (taskEnergyParameters edge))).PosSemidef := by
  let pair : Fin 3 → ℝ := fun i => if i = edge.castSucc ∨ i = edge.succ then 1 else 0
  let diagonal : Fin 3 → ℝ := fun i => if i = edge.castSucc ∨ i = edge.succ then 3 / 4 else 1
  have hd : (Matrix.diagonal diagonal).PosSemidef := by
    apply Matrix.PosSemidef.diagonal
    intro i
    fin_cases edge <;> fin_cases i <;> norm_num [diagonal]
  have hp : (Matrix.vecMulVec pair pair).PosSemidef := by
    simpa using Matrix.posSemidef_vecMulVec_self_star pair
  have he : memoryGramScores (localMemoryGram (taskEnergyParameters edge)) =
      Matrix.diagonal diagonal + (1 / 8 : ℝ) • Matrix.vecMulVec pair pair := by
    rw [localMemoryThree_scores]
    ext i j
    fin_cases edge <;> fin_cases i <;> fin_cases j <;>
      norm_num [taskEnergyParameters, pair, diagonal, Matrix.diagonal_apply,
        Matrix.vecMulVec_apply]
  rw [he]
  exact hd.add (hp.smul (by norm_num : (0 : ℝ) ≤ 1 / 8))

/-- Exact targets satisfy the entire block PSD energy coupling.
Source: a proved congruence certificate for the new restriction after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyMatrix_posSemidef (edge : Fin 2) :
    (energyCoupledMemoryMatrix 6 (taskEnergyParameters edge, taskEnergyTarget edge)).PosSemidef := by
  let C := Matrix.fromRows (1 : Matrix (Fin 3) (Fin 3) ℝ) (taskEnergyTarget edge).transpose
  have he : energyCoupledMemoryMatrix 6 (taskEnergyParameters edge, taskEnergyTarget edge) =
      C * memoryGramScores (localMemoryGram (taskEnergyParameters edge)) * C.conjTranspose := by
    simp only [energyCoupledMemoryMatrix, localMemoryThree_scores]
    ext i j
    cases i with
    | inl i =>
      cases j with
      | inl j =>
        fin_cases edge <;> fin_cases i <;> fin_cases j <;>
          norm_num [C, Matrix.fromRows, Matrix.mul_apply, Matrix.conjTranspose_apply,
            Fin.sum_univ_succ, Matrix.of_apply, Sum.elim, Matrix.transpose_apply, Matrix.one_apply,
            taskEnergyTarget, taskEnergyParameters]
      | inr j =>
        fin_cases edge <;> fin_cases i <;> fin_cases j <;>
          norm_num [C, Matrix.fromRows, Matrix.mul_apply, Matrix.conjTranspose_apply,
            Fin.sum_univ_succ, Matrix.of_apply, Sum.elim, Matrix.transpose_apply, Matrix.one_apply,
            taskEnergyTarget, taskEnergyParameters]
    | inr i =>
      cases j with
      | inl j =>
        fin_cases edge <;> fin_cases i <;> fin_cases j <;>
          norm_num [C, Matrix.fromRows, Matrix.mul_apply, Matrix.conjTranspose_apply,
            Fin.sum_univ_succ, Matrix.of_apply, Sum.elim, Matrix.transpose_apply, Matrix.one_apply,
            taskEnergyTarget, taskEnergyParameters]
      | inr j =>
        fin_cases edge <;> fin_cases i <;> fin_cases j <;>
          norm_num [C, Matrix.fromRows, Matrix.mul_apply, Matrix.conjTranspose_apply,
            Fin.sum_univ_succ, Matrix.of_apply, Sum.elim, Matrix.transpose_apply, Matrix.one_apply,
            taskEnergyTarget, taskEnergyParameters]
  rw [he]
  exact (taskEnergyAttention_posSemidef edge).mul_mul_conjTranspose_same C

/-- Both answer patterns fit in one common joint convex domain with nonidentity attention.
Source: a complete energy-coupled instance after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyPair_mem (edge : Fin 2) :
    (taskEnergyParameters edge, taskEnergyTarget edge) ∈
      energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6 :=
  ⟨taskEnergyParameters_mem edge, taskEnergyParameters_budget edge, taskEnergyMatrix_posSemidef edge⟩

/-- Actual sparsemax outputs fit the ordinary answers with one learned common value table.
Source: the genuine shared-value decoder after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyForward_exact (edge : Fin 2) :
    localJointMemoryForward (oneHotContextCodes (fun j : Fin 3 => j))
      (taskEnergyParameters edge, taskEnergyTarget edge) = taskEnergyTarget edge := by
  rw [energyCoupledMemoryForward_eq 4 (3 / 4) (1 / 8) 6 _ _ (by norm_num)
    (oneHotContextCodes_mem _) (taskEnergyPair_mem edge), oneHotContextCodes_mul]
  rfl

/-- The recovered original shared values are nonconstant and fit both answer patterns.
Source: the genuine invertible value chart after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyValues_recovered (edge : Fin 2) :
    recoverMemoryValues (localMemoryGram (taskEnergyParameters edge)) (taskEnergyTarget edge) =
      taskEnergyTarget edge := by
  apply (recoverMemoryValues_unique 4 (3 / 4) _ _ _ (by norm_num)
    (incidentMemoryGram_mem 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem edge)) ?_).symm
  unfold memoryValueOutput
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num)
    (taskEnergyParameters_mem edge), ← localMemoryGram_scores, localMemoryThree_scores]
  ext i d
  fin_cases edge <;> fin_cases i <;> fin_cases d <;>
    norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.of_apply,
      taskEnergyTarget, taskEnergyParameters]

end Transformer.GPTMini.Sparsemax
