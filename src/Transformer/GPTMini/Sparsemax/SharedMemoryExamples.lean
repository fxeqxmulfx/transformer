import Transformer.GPTMini.Sparsemax.MemoryExampleGrams

/-!
# Three different repeated-token contexts with jointly learned common values

Derived witness for sparsemax arXiv:1602.02068v2, Eq. (1), and the
shared-memory `attn @ v` at `73f8a0b`. Actual data prefixes `(1,1,1)`,
`(0,1,1)` and `(0,0,1)` produce different causal query codes. One learned
memory and one value table serve all three contexts. Both Q/K families,
the memory values and the actual sparse support change together.

Original values change from `(1,0)` to `(-1/2,3/2)`; global output
coordinates change from `(1,0)` to `(0,1)`. Every context output at the
true joint midpoint is one half, realized by the same decoded values
`(1/2,1/2)`. Literal mean original values give first-context output
`11/16` instead. The positive result uses fixed causal data codes and a
learned parameter dictionary, not ordinary token self-attention. No task
loss, FFN, floating-point execution or new training result is asserted.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Initial global output coordinates, also the initial original values.
Source: the identity memory in the derived shared-context witness. -/
def memoryExampleStartOutputs : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 1 else 0

/-- Changed global output coordinates shared by all context codes.
Source: the derived common-memory witness, not independent row outputs. -/
def memoryExampleStopOutputs : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 0 else 1

/-- One changed original value table used by every different data context.
Source: exact memory inverse recovery after the actual sparsemax projection. -/
def memoryExampleStopValues : Matrix (Fin 2) (Fin 1) ℝ :=
  fun i _ => if i = 0 then -(1 / 2) else 3 / 2

/-- The true midpoint of the common learned Gram/output parameter pair.
Source: the derived convex shared-memory coordinate domain. -/
def memoryExampleMidPoint : EmbeddingGram 2 × Matrix (Fin 2) (Fin 1) ℝ :=
  (1 / 2 : ℝ) • (memoryExampleStartGram, memoryExampleStartOutputs) +
    (1 / 2 : ℝ) • (memoryExampleStopGram, memoryExampleStopOutputs)

/-- Initial actual memory values give the stated global outputs.
Source: the ordinary value sum with the proved identity sparsemax memory. -/
theorem memoryExampleStart_actual : memoryValueOutput memoryExampleStartGram
    memoryExampleStartOutputs = memoryExampleStartOutputs := by
  unfold memoryValueOutput
  rw [memoryExampleStartGram_attention, Matrix.one_mul]

/-- Changed actual common values realize the changed global outputs.
Source: the ordinary memory value sum after Eq. (1), without private values. -/
theorem memoryExampleStop_actual : memoryValueOutput memoryExampleStopGram
    memoryExampleStopValues = memoryExampleStopOutputs := by
  unfold memoryValueOutput
  rw [memoryExampleStopGram_attention]
  ext i d
  rw [Matrix.mul_apply, Fin.sum_univ_two]
  fin_cases i <;> norm_num [memoryExampleStopValues, memoryExampleStopOutputs]

/-- The global decoder recovers exactly the changed original common values.
Source: uniqueness of the actual dictionary value realization. -/
theorem memoryExampleStop_recovered : recoverMemoryValues memoryExampleStopGram
    memoryExampleStopOutputs = memoryExampleStopValues := by
  exact (recoverMemoryValues_unique 4 (3 / 4) _ _ _ (by norm_num)
    memoryExampleStopGram_mem memoryExampleStop_actual).symm

/-- Original values genuinely change, together with both learned Q/K families.
Source: the explicit common tables in the shared-context witness. -/
theorem memoryExample_values_change : memoryExampleStartOutputs ≠ memoryExampleStopValues := by
  intro h
  have hi := congrArg (fun V : Matrix (Fin 2) (Fin 1) ℝ => V 0 0) h
  norm_num [memoryExampleStartOutputs, memoryExampleStopValues] at hi

/-- Actual initial outputs differ across the three repeated-token contexts.
Source: actual variational sparsemax with the one recovered global value table. -/
theorem memoryExampleStart_context_outputs : jointContextMemoryForward memoryExampleCodes
    (memoryExampleStartGram, memoryExampleStartOutputs) =
    Matrix.of (fun r _ => if r = 0 then 0 else if r = 1 then 1 / 3 else 2 / 3) := by
  rw [jointContextMemoryForward_eq 4 (3 / 4) _ _ (by norm_num)
    memoryExampleCodes_mem memoryExampleStartGram_mem]
  ext r d
  fin_cases r <;> norm_num [Matrix.mul_apply, Fin.sum_univ_two,
    memoryExampleCodes_apply, memoryExampleStartOutputs]

/-- Actual changed outputs use one common learned table for all contexts.
Source: the actual common-value inverse and the same causal data codes. -/
theorem memoryExampleStop_context_outputs : jointContextMemoryForward memoryExampleCodes
    (memoryExampleStopGram, memoryExampleStopOutputs) =
    Matrix.of (fun r _ => if r = 0 then 1 else if r = 1 then 2 / 3 else 1 / 3) := by
  rw [jointContextMemoryForward_eq 4 (3 / 4) _ _ (by norm_num)
    memoryExampleCodes_mem memoryExampleStopGram_mem]
  ext r d
  fin_cases r <;> norm_num [Matrix.mul_apply, Fin.sum_univ_two,
    memoryExampleCodes_apply, memoryExampleStopOutputs]

/-- The actual first context acquires another positive memory route.
Source: Eq. (1) on genuine changed Q/K mixture scores computed from data. -/
theorem memoryExample_context_support_changes :
    contextMemoryAttention memoryExampleStartGram memoryExampleCodes 0 0 = 0 ∧
    0 < contextMemoryAttention memoryExampleStopGram memoryExampleCodes 0 0 := by
  rw [contextMemoryAttention_eq_product 4 (3 / 4) _ _ memoryExampleStartGram_mem memoryExampleCodes_mem,
    contextMemoryAttention_eq_product 4 (3 / 4) _ _ memoryExampleStopGram_mem memoryExampleCodes_mem,
    memoryExampleStartGram_attention, memoryExampleStopGram_attention]
  norm_num [Matrix.mul_apply, Fin.sum_univ_two, memoryExampleCodes_apply]

/-- The true midpoint remains in the convex domain with one shared memory.
Source: the structural PSD, probability and strict diagonal floor constraints. -/
theorem memoryExampleMidPoint_mem : memoryExampleMidPoint ∈ jointMemoryDomain 1 1 4 (3 / 4) :=
  jointMemoryDomain_convex 1 1 4 (3 / 4) memoryExampleStartGram_mem memoryExampleStopGram_mem
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)

/-- Actual midpoint sparsemax memory is the mean endpoint attention.
Source: the proved affine normalized-Gram projection, with changing supports. -/
theorem memoryExampleMidPoint_attention : memoryGramAttention memoryExampleMidPoint.1 =
    Matrix.of (fun i j => if i = j then 7 / 8 else 1 / 8) := by
  change memoryGramAttention ((1 / 2 : ℝ) • memoryExampleStartGram +
    (1 / 2 : ℝ) • memoryExampleStopGram) = _
  rw [memoryGramAttention_affine 4 (3 / 4) _ _ memoryExampleStartGram_mem memoryExampleStopGram_mem
    (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num),
    memoryExampleStartGram_attention, memoryExampleStopGram_attention]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num

/-- Every actual midpoint context output equals one half, with shared values.
Source: the exact global coordinate change applied to all three causal data codes. -/
theorem memoryExampleMidPoint_outputs : jointContextMemoryForward memoryExampleCodes memoryExampleMidPoint =
    Matrix.of (fun _ : Fin 3 => fun _ : Fin 1 => (1 / 2 : ℝ)) := by
  rw [jointContextMemoryForward_eq 4 (3 / 4) _ _ (by norm_num)
    memoryExampleCodes_mem memoryExampleMidPoint_mem]
  ext r d
  fin_cases r <;> norm_num [Matrix.mul_apply, Fin.sum_univ_two, memoryExampleCodes_apply,
    memoryExampleMidPoint, memoryExampleStartOutputs, memoryExampleStopOutputs]

/-- The one shared decoded midpoint table is exactly `(1/2,1/2)`.
Source: unique global inverse recovery, witnessed by an actual attention product. -/
theorem memoryExampleMidPoint_recovered : recoverMemoryValues memoryExampleMidPoint.1 memoryExampleMidPoint.2 =
    Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (1 / 2 : ℝ)) := by
  apply Eq.symm
  apply recoverMemoryValues_unique 4 (3 / 4) _ _ _ (by norm_num) memoryExampleMidPoint_mem
  unfold memoryValueOutput
  rw [memoryExampleMidPoint_attention]
  ext i d
  rw [Matrix.mul_apply, Fin.sum_univ_two]
  fin_cases i <;> norm_num [memoryExampleMidPoint,
    memoryExampleStartOutputs, memoryExampleStopOutputs]

/-- Literal mean original values give a different first-context output.
Source: the actual attention/value product; convexity uses the exact new coordinates. -/
theorem memoryExample_literal_values_midpoint : contextMemoryValueOutput memoryExampleMidPoint.1 memoryExampleCodes
    ((1 / 2 : ℝ) • memoryExampleStartOutputs + (1 / 2 : ℝ) • memoryExampleStopValues) 0 0 = 11 / 16 := by
  rw [contextMemoryValueOutput_factor 4 (3 / 4) _ _ _ memoryExampleMidPoint_mem memoryExampleCodes_mem]
  unfold memoryValueOutput
  rw [memoryExampleMidPoint_attention]
  rw [Matrix.mul_apply, Fin.sum_univ_two]
  norm_num [memoryExampleCodes_apply]
  rw [Matrix.mul_apply, Fin.sum_univ_two]
  norm_num [memoryExampleStartOutputs, memoryExampleStopValues]

/-- All actual outputs obey the common-table linear coupling, `Y2 = 2 Y1 - Y0`.
Source: the exact prediction class of the derived three-prefix memory witness. -/
theorem memoryExample_context_output_relation
    (p : EmbeddingGram 2 × Matrix (Fin 2) (Fin 1) ℝ) (hp : p ∈ jointMemoryDomain 1 1 4 (3 / 4)) :
    jointContextMemoryForward memoryExampleCodes p 2 0 =
      2 * jointContextMemoryForward memoryExampleCodes p 1 0 -
        jointContextMemoryForward memoryExampleCodes p 0 0 := by
  rw [jointContextMemoryForward_eq 4 (3 / 4) _ _ (by norm_num) memoryExampleCodes_mem hp]
  norm_num [Matrix.mul_apply, Fin.sum_univ_two, memoryExampleCodes_apply]
  ring

/-- The coupling hypothesis has an actual nonzero, nonconstant output witness. -/
example : jointContextMemoryForward memoryExampleCodes (memoryExampleStartGram, memoryExampleStartOutputs) 2 0 =
    2 * jointContextMemoryForward memoryExampleCodes (memoryExampleStartGram, memoryExampleStartOutputs) 1 0 -
      jointContextMemoryForward memoryExampleCodes (memoryExampleStartGram, memoryExampleStartOutputs) 0 0 :=
  memoryExample_context_output_relation _ memoryExampleStartGram_mem

/-- Three arbitrary targets need not be jointly attainable by common values.
Source: the new memory architecture's prediction class, not a claim about
sparsemax Eq. (1) or an unrestricted transformer. Feasible Grams exist above. -/
theorem memoryExample_target_triple_unattainable : ¬ ∃ G ∈ memoryGramDomain 1 4 (3 / 4),
    ∃ values : Matrix (Fin 2) (Fin 1) ℝ,
      contextMemoryValueOutput G memoryExampleCodes values 0 0 = 0 ∧
      contextMemoryValueOutput G memoryExampleCodes values 1 0 = 0 ∧
      contextMemoryValueOutput G memoryExampleCodes values 2 0 = 1 := by
  rintro ⟨G, hG, values, ha, hb, hc⟩
  have he := memoryExample_context_output_relation (G, memoryValueOutput G values) hG
  rw [jointContextMemoryForward_original 4 (3 / 4) _ _ _ (by norm_num) hG] at he
  rw [ha, hb, hc] at he
  norm_num at he

end Transformer.GPTMini.Sparsemax
