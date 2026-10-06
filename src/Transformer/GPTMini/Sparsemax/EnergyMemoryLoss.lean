import Transformer.GPTMini.Sparsemax.EnergyMemoryIdentification

/-!
# Convex ordinary output error selects nontrivial attention

New regression guarantee after arXiv:1602.02068v2, Eq. (1). The criterion
is only squared error of actual shared-memory predictions against observed
answers. It contains no embedding criterion. Under the common PSD energy
and total edge budgets it is jointly convex and has attained value zero.
Every zero-error point selects the corresponding actual attention matrix.

An explicit lower bound holds after all values are relearned at the opposite
edge allocation. Thus geometry affects attainable ordinary output error,
even though the chart's unconstrained partial formula remains MZ.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Squared ordinary output error of the actual sparsemax/common-value forward.
Source: the new energy-coupled regression experiment after arXiv:1602.02068v2, Eq. (1). -/
def taskEnergySquaredError (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ) : ℝ :=
  ∑ i, (localJointMemoryForward (oneHotContextCodes (fun j : Fin 3 => j)) x i 0 -
    taskEnergyTarget edge i 0) ^ 2

/-- The ordinary criterion is nonnegative for every actual prediction, without feasibility assumptions.
Source: squared regression error after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySquaredError_nonneg (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ) :
    0 ≤ taskEnergySquaredError edge x :=
  Finset.sum_nonneg (fun _ _ => sq_nonneg _)

/-- Feasible actual output error equals ordinary error of the learned output coordinates.
Source: the exact shared-value chart after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySquaredError_eq (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6) :
    taskEnergySquaredError edge x = ∑ i, (x.2 i 0 - taskEnergyTarget edge i 0) ^ 2 := by
  unfold taskEnergySquaredError
  rw [energyCoupledMemoryForward_eq 4 (3 / 4) (1 / 8) 6 _ _ (by norm_num)
    (oneHotContextCodes_mem _) hx, oneHotContextCodes_mul]
  rfl

/-- The nonconstant target and nonidentity actual attention inhabit the reduction premises. -/
example : taskEnergySquaredError 0 (taskEnergyParameters 0, taskEnergyTarget 0) =
    ∑ i, (taskEnergyTarget 0 i 0 - taskEnergyTarget 0 i 0) ^ 2 :=
  taskEnergySquaredError_eq 0 _ (taskEnergyPair_mem 0)

/-- Ordinary output error is jointly convex with every learned embedding and common value coordinate.
Source: the new convex domain and actual forward after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySquaredError_convex (edge : Fin 2) :
    ConvexOn ℝ (energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
      (taskEnergySquaredError edge) := by
  apply energyCoupledMemoryObjective_convex 4 (3 / 4) (1 / 8) 6
    (oneHotContextCodes (fun j : Fin 3 => j))
    (fun Y => ∑ i, (Y i 0 - taskEnergyTarget edge i 0) ^ 2)
    (by norm_num) (oneHotContextCodes_mem _)
  have hs : ConvexOn ℝ Set.univ (fun z : ℝ => z ^ 2) := (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro A _ B _ a b ha hb hab
  change (∑ i, (a * A i 0 + b * B i 0 - taskEnergyTarget edge i 0) ^ 2) ≤ _
  have h := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin 3))) (fun i _ =>
    hs.2 (Set.mem_univ (A i 0 - taskEnergyTarget edge i 0))
      (Set.mem_univ (B i 0 - taskEnergyTarget edge i 0)) ha hb hab)
  have he (i : Fin 3) : a * A i 0 + b * B i 0 - taskEnergyTarget edge i 0 =
      a * (A i 0 - taskEnergyTarget edge i 0) + b * (B i 0 - taskEnergyTarget edge i 0) := by
    have ht := congrArg (fun r : ℝ => r * taskEnergyTarget edge i 0) hab
    nlinarith
  simp_rw [he]
  simpa only [smul_eq_mul, Finset.sum_add_distrib, ← Finset.mul_sum] using h

/-- The correct learned edge allocation attains zero ordinary output error.
Source: a nonconstant actual forward witness after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySquaredError_exact (edge : Fin 2) :
    taskEnergySquaredError edge (taskEnergyParameters edge, taskEnergyTarget edge) = 0 := by
  rw [taskEnergySquaredError_eq edge _ (taskEnergyPair_mem edge)]
  simp

/-- Zero ordinary error on the coupled domain identifies actual attention.
Source: the new task-based identification guarantee after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySquaredError_zero_identifies_attention (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
    (hl : taskEnergySquaredError edge x = 0) :
    memoryGramAttention (localMemoryGram x.1) =
      memoryGramAttention (localMemoryGram (taskEnergyParameters edge)) := by
  have ho : localJointMemoryForward (oneHotContextCodes (fun j : Fin 3 => j)) x =
      taskEnergyTarget edge := by
    have hz := (Finset.sum_eq_zero_iff_of_nonneg
      (fun i (_ : i ∈ (Finset.univ : Finset (Fin 3))) =>
        sq_nonneg (localJointMemoryForward (oneHotContextCodes (fun j : Fin 3 => j)) x i 0 -
          taskEnergyTarget edge i 0))).1 hl
    ext i d
    fin_cases d
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp (hz i (Finset.mem_univ i)))
  exact taskEnergyExact_identifies_attention edge x hx ho

/-- The zero-loss identification assumptions are inhabited by learned nonidentity attention. -/
example : memoryGramAttention (localMemoryGram (taskEnergyParameters 0)) =
    memoryGramAttention (localMemoryGram (taskEnergyParameters 0)) :=
  taskEnergySquaredError_zero_identifies_attention 0 (taskEnergyParameters 0, taskEnergyTarget 0)
    (taskEnergyPair_mem 0)
    (taskEnergySquaredError_exact 0)

/-- At the opposite allocation every relearned common value table leaves positive ordinary error.
Source: a quantitative PSD-test separation after arXiv:1602.02068v2, Eq. (1).
The bound is on sum squared error, without a mean or a factor of one half. -/
theorem taskEnergySquaredError_wrong_edge_lower (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
    (he : x.1.1 (otherTaskEnergyEdge edge) = 1 / 8) :
    27 / 512 ≤ taskEnergySquaredError edge x := by
  have ha := taskEnergyAlignment_bound edge x hx
  rw [he] at ha
  rw [taskEnergySquaredError_eq edge x hx]
  fin_cases edge <;>
    norm_num [taskEnergyTarget, Matrix.of_apply, Fin.sum_univ_succ] at ha ⊢
  · have h0 := sq_nonneg (x.2 0 0 - x.2 1 0)
    have h1 := sq_nonneg (2 * (x.2 0 0 - 1) + (x.2 2 0 + 2))
    have h2 := sq_nonneg (2 * (x.2 1 0 - 1) + (x.2 2 0 + 2))
    have hd : 9 / 16 ≤ 6 - (x.2 0 0 + x.2 1 0 - 2 * x.2 2 0) := by linarith
    nlinarith [sq_nonneg (6 - (x.2 0 0 + x.2 1 0 - 2 * x.2 2 0) - 9 / 16)]
  · have h0 := sq_nonneg (x.2 1 0 - x.2 2 0)
    have h1 := sq_nonneg (2 * (x.2 1 0 - 1) + (x.2 0 0 + 2))
    have h2 := sq_nonneg (2 * (x.2 2 0 - 1) + (x.2 0 0 + 2))
    have hd : 9 / 16 ≤ 6 - (-2 * x.2 0 0 + x.2 1 0 + x.2 2 0) := by linarith
    nlinarith [sq_nonneg (6 - (-2 * x.2 0 0 + x.2 1 0 + x.2 2 0) - 9 / 16)]

/-- Opposite ordinary labels supply a feasible wrong-edge instance; the lower-bound premise is nonempty. -/
example : 27 / 512 ≤ taskEnergySquaredError 0 (taskEnergyParameters 1, taskEnergyTarget 1) := by
  apply taskEnergySquaredError_wrong_edge_lower 0 _ (taskEnergyPair_mem 1)
  norm_num [otherTaskEnergyEdge, taskEnergyParameters]

/-- The exact-fit nonidentity witness is an attained minimum of ordinary output error.
Source: the new jointly convex regression domain after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySquaredError_isMinOn (edge : Fin 2) :
    IsMinOn (taskEnergySquaredError edge)
      (energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
      (taskEnergyParameters edge, taskEnergyTarget edge) := by
  intro x hx
  change taskEnergySquaredError edge (taskEnergyParameters edge, taskEnergyTarget edge) ≤
    taskEnergySquaredError edge x
  rw [taskEnergySquaredError_exact, taskEnergySquaredError_eq edge x hx]
  exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)

/-- Every attained ordinary output-error minimum selects the corresponding actual attention.
Source: the new output-only task-selection guarantee after arXiv:1602.02068v2, Eq. (1). -/
theorem taskEnergySquaredError_minimum_identifies_attention (edge : Fin 2)
    (x : LocalMemoryParameters 2 × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
    (hm : IsMinOn (taskEnergySquaredError edge)
      (energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6) x) :
    memoryGramAttention (localMemoryGram x.1) =
      memoryGramAttention (localMemoryGram (taskEnergyParameters edge)) := by
  have hl := hm (taskEnergyPair_mem edge)
  change taskEnergySquaredError edge x ≤
    taskEnergySquaredError edge (taskEnergyParameters edge, taskEnergyTarget edge) at hl
  rw [taskEnergySquaredError_exact] at hl
  exact taskEnergySquaredError_zero_identifies_attention edge x hx
    (le_antisymm hl (taskEnergySquaredError_nonneg edge x))

/-- Genuine mixed attention and nonconstant values inhabit every minimizer-identification premise. -/
example : memoryGramAttention (localMemoryGram (taskEnergyParameters 1)) =
    memoryGramAttention (localMemoryGram (taskEnergyParameters 1)) :=
  taskEnergySquaredError_minimum_identifies_attention 1 (taskEnergyParameters 1, taskEnergyTarget 1)
    (taskEnergyPair_mem 1) (taskEnergySquaredError_isMinOn 1)

end Transformer.GPTMini.Sparsemax
