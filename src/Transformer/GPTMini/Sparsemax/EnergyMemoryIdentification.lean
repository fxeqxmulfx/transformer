import Transformer.GPTMini.Sparsemax.EnergyMemoryExamples

/-!
# Ordinary answers identify actual attention under the energy coupling

Derived regression result after arXiv:1602.02068v2, Eq. (1). A test vector
in the block PSD inequality limits how well the output aligns with either
answer pattern. Exact output equality and energy 6 force the edge joining
different answers to vanish. The prescribed total weight 1/8 then forces
the other edge to carry all weight. Q/K norms are not identified.

No criterion comparing embeddings or target routes is used. The outputs
come from actual sparsemax attention and one global learned value table.
The two nonconstant answer patterns choose opposite nonidentity supports.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The other of the two available path edges in the finite regression example.
Source: the new three-slot witness after arXiv:1602.02068v2, Eq. (1). -/
def otherTaskEnergyEdge (edge : Fin 2) : Fin 2 := if edge = 0 then 1 else 0

/-- The PSD constraint bounds ordinary target/output alignment by the discordant edge.
Source: a quadratic test of the new block constraint after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyAlignment_bound (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6) :
    2 * (∑ i, taskEnergyTarget edge i 0 * x.2 i 0) ≤
      12 - 9 * x.1.1 (otherTaskEnergyEdge edge) := by
  have h := hx.2.2.dotProduct_mulVec_nonneg
    (Sum.elim (fun i : Fin 3 => taskEnergyTarget edge i 0) (fun _ : Fin 1 => (-1 : ℝ)))
  simp only [energyCoupledMemoryMatrix, localMemoryThree_scores] at h
  fin_cases edge <;>
    norm_num [dotProduct, Matrix.mulVec, Fintype.sum_sum_type, Fin.sum_univ_succ,
      Matrix.fromBlocks, Matrix.of_apply, taskEnergyTarget, Matrix.transpose_apply,
      otherTaskEnergyEdge] at h ⊢ <;> nlinarith

/-- A nonconstant exactly fitted target inhabits every alignment premise. -/
example : 2 * (∑ i, taskEnergyTarget 0 i 0 * taskEnergyTarget 0 i 0) ≤
    12 - 9 * (taskEnergyParameters 0).1 (otherTaskEnergyEdge 0) :=
  taskEnergyAlignment_bound 0 _ (taskEnergyPair_mem 0)

/-- Exact ordinary outputs force the discordant edge to vanish and determine both weights.
Source: the new active-energy regression guarantee after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergyExact_identifies_edges (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
    (hz : x.2 = taskEnergyTarget edge) : x.1.1 = (taskEnergyParameters edge).1 := by
  have ha := taskEnergyAlignment_bound edge x hx
  rw [hz] at ha
  have h0 := hx.1.1.1 0
  have h1 := hx.1.1.1 1
  have ht := hx.2.1
  rw [Fin.sum_univ_two] at ht
  ext e
  fin_cases edge <;> fin_cases e <;>
    norm_num [taskEnergyTarget, taskEnergyParameters, otherTaskEnergyEdge,
      Fin.sum_univ_succ, Matrix.of_apply] at ha ⊢ <;> linarith

/-- The nonidentity exact-fit witness satisfies the complete identification premises. -/
example : (taskEnergyParameters 0).1 = (taskEnergyParameters 0).1 :=
  taskEnergyExact_identifies_edges 0 _ (taskEnergyPair_mem 0) rfl

/-- Exact genuine shared-memory outputs identify the actual sparsemax attention matrix.
Source: the new task-dependent restriction after arXiv:1602.02068v2, Eq. (1).
This conclusion identifies the cross block, not the independent Q/K squared norms. -/
theorem taskEnergyExact_identifies_attention (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
    (ho : localJointMemoryForward (oneHotContextCodes (fun j : Fin 3 => j)) x =
      taskEnergyTarget edge) :
    memoryGramAttention (localMemoryGram x.1) =
      memoryGramAttention (localMemoryGram (taskEnergyParameters edge)) := by
  have hz : x.2 = taskEnergyTarget edge := by
    rw [energyCoupledMemoryForward_eq 4 (3 / 4) (1 / 8) 6 _ _ (by norm_num)
      (oneHotContextCodes_mem _) hx, oneHotContextCodes_mul] at ho
    exact ho
  have he := taskEnergyExact_identifies_edges edge x hx hz
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) hx.1,
    incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem edge)]
  exact congrArg (fun t : Fin 2 → ℝ => memoryGramScores (localMemoryCore t)) he

/-- Nonconstant fitted outputs inhabit the actual attention identification premises. -/
example : memoryGramAttention (localMemoryGram (taskEnergyParameters 1)) =
    memoryGramAttention (localMemoryGram (taskEnergyParameters 1)) :=
  taskEnergyExact_identifies_attention 1 (taskEnergyParameters 1, taskEnergyTarget 1)
    (taskEnergyPair_mem 1) (taskEnergyForward_exact 1)

/-- The two output label patterns select genuinely different actual attention matrices.
Source: new finite task-dependent support witnesses after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySelected_attention_ne :
    memoryGramAttention (localMemoryGram (taskEnergyParameters 0)) ≠
      memoryGramAttention (localMemoryGram (taskEnergyParameters 1)) := by
  intro h
  have he := congrFun (congrFun h 0) 1
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem 0),
    incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem 1),
    ← localMemoryGram_scores, ← localMemoryGram_scores, localMemoryThree_scores,
    localMemoryThree_scores] at he
  norm_num [taskEnergyParameters] at he

/-- Both selected attention matrices are nonidentity, so this is genuine learned mixing.
Source: new nontrivial task-selection witnesses after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySelected_attention_ne_one (edge : Fin 2) :
    memoryGramAttention (localMemoryGram (taskEnergyParameters edge)) ≠ 1 := by
  intro h
  have he := congrFun (congrFun h edge.castSucc) edge.succ
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem edge),
    ← localMemoryGram_scores, localMemoryThree_scores] at he
  fin_cases edge <;> norm_num [taskEnergyParameters, Matrix.one_apply] at he

/-- Changing Q/K norm coordinates at fixed edges leaves the entire energy coupling unchanged.
Source: the remaining norm freedom in the new construction after arXiv:1602.02068v2, Eq. (1).
The ordinary task guarantees identify attention, not all embedding geometry. -/
theorem energyCoupledMemoryMatrix_edges_invariant {N D : ℕ} (energy : ℝ)
    (p q : LocalMemoryParameters N) (Z : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (he : p.1 = q.1) :
    energyCoupledMemoryMatrix energy (p, Z) = energyCoupledMemoryMatrix energy (q, Z) := by
  unfold energyCoupledMemoryMatrix
  simp only [localMemoryGram_scores]
  rw [he]

/-- Distinct norm coordinates satisfy the remaining-gauge premise with a nonzero edge. -/
example : energyCoupledMemoryMatrix 6
    (((taskEnergyParameters 0).1, (0 : Sum (Fin 3) (Fin 3) → ℝ)), taskEnergyTarget 0) =
      energyCoupledMemoryMatrix 6 (taskEnergyParameters 0, taskEnergyTarget 0) :=
  energyCoupledMemoryMatrix_edges_invariant 6 _ _ _ rfl

/-- Any feasible norm change preserves coupled feasibility at the same outputs and learned edges.
Source: the explicit remaining norm gauge after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryDomain_change_norms {N D : ℕ} (cap floor budget energy : ℝ)
    (p q : LocalMemoryParameters N) (Z : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hq : q ∈ incidentMemoryParameterDomain N cap floor) (he : p.1 = q.1)
    (hx : (p, Z) ∈ energyCoupledMemoryDomain N D cap floor budget energy) :
    (q, Z) ∈ energyCoupledMemoryDomain N D cap floor budget energy := by
  refine ⟨hq, ?_, ?_⟩
  · change (∑ e, q.1 e) = budget
    rw [← he]
    exact hx.2.1
  · rw [← energyCoupledMemoryMatrix_edges_invariant energy p q Z he]
    exact hx.2.2

/-- A nonzero budget and nonconstant target inhabit every norm-change premise. -/
example : (taskEnergyParameters 1, taskEnergyTarget 1) ∈
    energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6 :=
  energyCoupledMemoryDomain_change_norms 4 (3 / 4) (1 / 8) 6 _ _ _
    (taskEnergyParameters_mem 1) rfl (taskEnergyPair_mem 1)

end Transformer.GPTMini.Sparsemax
