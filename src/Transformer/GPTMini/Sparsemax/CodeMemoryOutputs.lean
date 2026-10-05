import Transformer.GPTMini.Sparsemax.MemoryExampleGrams

/-!
# Exact attainable outputs for categorical data codes

Derived shared-memory architecture before sparsemax arXiv:1602.02068v2,
Eq. (1), and `attn @ v` at `73f8a0b`. A categorical code selects one
learned query dictionary entry. Its actual attention is still learned:
the selected row of the normalized memory Gram, followed by one common
value table. The code is an observation feature, not a target route.

Unlike a frequency encoder, a code that distinguishes observations has
no extra linear restriction on their outputs. The exact attainable class
consists of tables constant on equal-code fibers. A single global output
table extends every such target table, and the memory inverse recovers
one common original value table for every feasible learned Gram.

Targets here characterize expressivity; they are not supplied to the data
encoder. Duplicate codes must share predictions. Dictionary size and feature
width are unrestricted, and no efficient representation is asserted.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- A fixed categorical observation code as a probability row.
Source: the derived memory input architecture before sparsemax Eq. (1). -/
def oneHotContextCodes {R N : ℕ} (code : Fin R → Fin (N + 1)) :
    Matrix (Fin R) (Fin (N + 1)) ℝ := Matrix.of (fun r k => if code r = k then 1 else 0)

/-- Every categorical data encoder satisfies the required probability domain.
Source: unit mass at its encoded observation, independently of targets.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem oneHotContextCodes_mem {R N : ℕ} (code : Fin R → Fin (N + 1)) :
    oneHotContextCodes code ∈ contextCodeDomain R N := by
  intro r
  refine ⟨?_, ?_, fun k hk => False.elim (hk (Set.mem_univ k))⟩
  · intro k
    change 0 ≤ if code r = k then (1 : ℝ) else 0
    split_ifs <;> norm_num
  · change (∑ k, if code r = k then (1 : ℝ) else 0) = 1
    exact Fintype.sum_ite_eq (code r) (fun _ => (1 : ℝ))

/-- Multiplication by categorical codes evaluates one common output table.
Source: the actual shared-memory factorization, with a categorical input.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem oneHotContextCodes_mul {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) :
    oneHotContextCodes code * Z = Matrix.of (fun r => Z (code r)) := by
  ext r d
  rw [Matrix.mul_apply]
  simp only [oneHotContextCodes, Matrix.of_apply, ite_mul, one_mul, zero_mul]
  exact Fintype.sum_ite_eq (code r) (fun k => Z k d)

/-- Compatibility is a predicate on an encoder and its desired output table.
Source: deterministic prediction from a common observation code. -/
def codeCompatibleTargets {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (Y : Matrix (Fin R) (Fin D) ℝ) : Prop := ∀ r s, code r = code s → Y r = Y s

/-- Extend target rows to dictionary slots; unobserved slots receive zero.
Source: the derived expressivity construction. This is one global table,
and its encoder argument is fixed from data before any target is supplied. -/
def extendCodeOutputs {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (Y : Matrix (Fin R) (Fin D) ℝ) : Matrix (Fin (N + 1)) (Fin D) ℝ :=
  Matrix.of (fun k => if h : ∃ r, code r = k then Y (Classical.choose h) else 0)

/-- Consistent targets are recovered at every observed dictionary slot.
Source: the global fiber extension for the derived categorical memory.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem extendCodeOutputs_at_code {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (Y : Matrix (Fin R) (Fin D) ℝ) (hY : codeCompatibleTargets code Y) (r : Fin R) :
    extendCodeOutputs code Y (code r) = Y r := by
  classical
  have hw : ∃ s, code s = code r := ⟨r, rfl⟩
  change (if h : ∃ s, code s = code r then Y (Classical.choose h) else 0) = Y r
  rw [dite_eq_left hw]
  exact hY _ r (Classical.choose_spec hw)

/-- Two distinct data codes and nonconstant targets inhabit fiber consistency. -/
example : extendCodeOutputs (fun r : Fin 2 => r)
    (Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ))) 1 =
    (fun _ : Fin 1 => (2 : ℝ)) := by
  have hc : codeCompatibleTargets (fun r : Fin 2 => r)
      (Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ))) := by
    intro r s h
    exact congrArg (fun i : Fin 2 => fun _ : Fin 1 => (i.val + 1 : ℝ)) h
  convert extendCodeOutputs_at_code _ _ hc (1 : Fin 2) using 1
  funext d
  norm_num [Matrix.of_apply]

/-- The linear output range is exactly consistency on equal observation codes.
Source: the derived extension and evaluation identities, not a rank assumption.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem oneHotContextCodes_range {R N D : ℕ} (code : Fin R → Fin (N + 1)) :
    Set.range (fun Z : Matrix (Fin (N + 1)) (Fin D) ℝ => oneHotContextCodes code * Z) =
      {Y | codeCompatibleTargets code Y} := by
  ext Y
  constructor
  · rintro ⟨Z, hz⟩
    change oneHotContextCodes code * Z = Y at hz
    rw [← hz, oneHotContextCodes_mul]
    intro r s h
    change Z (code r) = Z (code s)
    rw [h]
  · intro hY
    refine ⟨extendCodeOutputs code Y, ?_⟩
    change oneHotContextCodes code * extendCodeOutputs code Y = Y
    rw [oneHotContextCodes_mul]
    ext r d
    exact congrFun (extendCodeOutputs_at_code code Y hY r) d

/-- Any feasible learned memory realizes every consistent categorical target.
Source: actual sparsemax Eq. (1), the common-value inverse and fiber extension.
The same statement also proves necessity; contradictory duplicate codes cannot fit.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextMemory_categorical_attainable_iff {R N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (code : Fin R → Fin (N + 1))
    (Y : Matrix (Fin R) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) :
    (∃ values : Matrix (Fin (N + 1)) (Fin D) ℝ,
      contextMemoryValueOutput G (oneHotContextCodes code) values = Y) ↔
      codeCompatibleTargets code Y := by
  constructor
  · rintro ⟨values, hy⟩
    have hr : Y ∈ Set.range (fun Z : Matrix (Fin (N + 1)) (Fin D) ℝ =>
        oneHotContextCodes code * Z) := by
      refine ⟨memoryValueOutput G values, ?_⟩
      rw [← hy, contextMemoryValueOutput_factor cap floor G _ _ hG
        (oneHotContextCodes_mem code)]
    rwa [oneHotContextCodes_range] at hr
  · intro hY
    refine ⟨recoverMemoryValues G (extendCodeOutputs code Y), ?_⟩
    rw [contextMemoryValueOutput_factor cap floor G _ _ hG (oneHotContextCodes_mem code),
      recoverMemoryValues_exact cap floor G _ hf hG, oneHotContextCodes_mul]
    ext r d
    exact congrFun (extendCodeOutputs_at_code code Y hY r) d

/-- A genuine feasible Gram and a nonzero output satisfy the inverse premises. -/
example : (∃ values : Matrix (Fin 1) (Fin 1) ℝ,
    contextMemoryValueOutput normalizedGramUnit (oneHotContextCodes (fun _ : Fin 1 => 0))
      values = Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) ↔
    codeCompatibleTargets (fun _ : Fin 1 => (0 : Fin 1))
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) :=
  contextMemory_categorical_attainable_iff 1 _ _ _ _ (by norm_num)
    normalizedGramUnit_mem_memory

/-- The actual categorical prediction class is exactly fiber consistency.
Source: the derived shared-memory prediction class, with an inhabited Gram domain.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem sharedMemoryPredictionSet_categorical {R N D : ℕ} (cap floor : ℝ)
    (code : Fin R → Fin (N + 1)) (hf : 1 / 2 < floor)
    (hn : (memoryGramDomain N cap floor).Nonempty) :
    sharedMemoryPredictionSet D cap floor (oneHotContextCodes code) =
      {Y | codeCompatibleTargets code Y} := by
  rw [sharedMemoryPredictionSet_eq_range cap floor _ hf (oneHotContextCodes_mem code) hn,
    oneHotContextCodes_range]

/-- Multiple observations and all memory-domain premises are genuinely inhabited. -/
example : sharedMemoryPredictionSet 1 1 (3 / 4) (oneHotContextCodes (fun r : Fin 2 => r)) =
    {Y | codeCompatibleTargets (fun r : Fin 2 => r) Y} :=
  sharedMemoryPredictionSet_categorical 1 _ _ (by norm_num) (memoryGramDomain_nonempty 1)

end Transformer.GPTMini.Sparsemax
