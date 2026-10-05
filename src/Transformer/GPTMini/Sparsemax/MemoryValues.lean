import Transformer.GPTMini.Sparsemax.MemoryGram

/-!
# One learned value table for a shared invertible dictionary

Derived memory version of `attn @ v` at `73f8a0b`, following actual
sparsemax arXiv:1602.02068v2, Eq. (1). A diagonal floor above one half
makes the learned dictionary attention invertible without triangular
supports. Its output coordinates decode exactly one shared value table.
The decoder has no context index. Later context queries all use this table.

Both query/key families and values may change. The original values remain
unrestricted; their usual penalties or norm bounds are not preserved by
this nonlinear coordinate change. Fixed small feature width, task loss,
FFN and ordinary token self-attention are not claimed by this architecture.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Actual dictionary attention multiplied by its common learned values.
Source: the memory specialization of `attn @ v` at `73f8a0b`. -/
def memoryValueOutput {N D : ℕ} (G : EmbeddingGram (N + 1))
    (values : Matrix (Fin (N + 1)) (Fin D) ℝ) : Matrix (Fin (N + 1)) (Fin D) ℝ :=
  memoryGramAttention G * values

/-- Decode one global table from the learned memory and output coordinates.
Source: the exact invertible coordinate change for the memory value sum. -/
def recoverMemoryValues {N D : ℕ} (G : EmbeddingGram (N + 1))
    (outputs : Matrix (Fin (N + 1)) (Fin D) ℝ) : Matrix (Fin (N + 1)) (Fin D) ℝ :=
  (memoryGramAttention G)⁻¹ * outputs

/-- Every dictionary output has an exact common value realization.
Source: the derived value coordinates after sparsemax Eq. (1), using
the proved strict-dominance inverse guarantee rather than a fixed matrix. -/
theorem recoverMemoryValues_exact {N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (outputs : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor) :
    memoryValueOutput G (recoverMemoryValues G outputs) = outputs := by
  exact Matrix.mul_nonsing_inv_cancel_left (memoryGramAttention G) outputs
    (memoryGramAttention_det_unit cap floor G hf hG)

/-- A nonzero output table inhabits every recovery premise. -/
example : memoryValueOutput normalizedGramUnit
    (recoverMemoryValues normalizedGramUnit (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ)))) =
      Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ)) :=
  recoverMemoryValues_exact 1 _ _ _ (by norm_num) normalizedGramUnit_mem_memory

/-- Encoding then decoding any original global value table is exact.
Source: the inverse coordinate change for the ordinary memory value product. -/
theorem recoverMemoryValues_original {N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (values : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor) :
    recoverMemoryValues G (memoryValueOutput G values) = values := by
  exact Matrix.nonsing_inv_mul_cancel_left (memoryGramAttention G) values
    (memoryGramAttention_det_unit cap floor G hf hG)

/-- Original nonzero learned values satisfy both inverse hypotheses. -/
example : recoverMemoryValues normalizedGramUnit (memoryValueOutput normalizedGramUnit
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)))) =
      Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) :=
  recoverMemoryValues_original 1 _ _ _ (by norm_num) normalizedGramUnit_mem_memory

/-- The global value realization is unique, with no relaxed private tables.
Source: the derived bijection for the actual shared dictionary value sum. -/
theorem recoverMemoryValues_unique {N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (values outputs : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor)
    (ho : memoryValueOutput G values = outputs) : values = recoverMemoryValues G outputs := by
  rw [← ho]
  exact (recoverMemoryValues_original cap floor G values hf hG).symm

/-- An actual nonzero product inhabits the uniqueness premises. -/
example : (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
    recoverMemoryValues normalizedGramUnit (memoryValueOutput normalizedGramUnit
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)))) :=
  recoverMemoryValues_unique 1 _ _ _ _ (by norm_num) normalizedGramUnit_mem_memory rfl

/-- The common value table can realize every dictionary output table.
Source: the exact inverse of `attn @ v` on the structural memory domain. -/
theorem memoryValueOutput_surjective {N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) :
    Function.Surjective (memoryValueOutput G :
      Matrix (Fin (N + 1)) (Fin D) ℝ → Matrix (Fin (N + 1)) (Fin D) ℝ) := by
  intro outputs
  exact ⟨recoverMemoryValues G outputs, recoverMemoryValues_exact cap floor G outputs hf hG⟩

/-- Actual nonzero embeddings inhabit the surjectivity hypotheses. -/
example : Function.Surjective (memoryValueOutput normalizedGramUnit :
    Matrix (Fin 1) (Fin 1) ℝ → Matrix (Fin 1) (Fin 1) ℝ) :=
  memoryValueOutput_surjective 1 _ _ (by norm_num) normalizedGramUnit_mem_memory

/-- Joint learned embedding/output coordinates for one common memory.
Source: the derived exact Gram and value-coordinate architecture. -/
def jointMemoryDomain (N D : ℕ) (cap floor : ℝ) :
    Set (EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ) :=
  {p | p.1 ∈ memoryGramDomain N cap floor}

/-- Evaluate the original value operation with its actual recovered table.
Source: `attn @ v` after variational sparsemax Eq. (1), in memory coordinates. -/
def jointMemoryForward {N D : ℕ}
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ) :=
  memoryValueOutput p.1 (recoverMemoryValues p.1 p.2)

/-- Joint memory embeddings and value-output coordinates form a convex domain.
Source: the structural dictionary constraints and free output coordinates. -/
theorem jointMemoryDomain_convex (N D : ℕ) (cap floor : ℝ) :
    Convex ℝ (jointMemoryDomain N D cap floor) := by
  intro p hp q hq a b ha hb hab
  exact memoryGramDomain_convex N cap floor hp hq ha hb hab

/-- Actual forward computation recovers the learned output table.
Source: the derived exact memory coordinates, with the inverse computed explicitly. -/
theorem jointMemoryForward_eq_outputs {N D : ℕ} (cap floor : ℝ)
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hp : p ∈ jointMemoryDomain N D cap floor) :
    jointMemoryForward p = p.2 := by
  exact recoverMemoryValues_exact cap floor p.1 p.2 hf hp

/-- A concrete nonzero learned output satisfies the joint-forward premises. -/
example : jointMemoryForward (normalizedGramUnit,
    Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
      Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) :=
  jointMemoryForward_eq_outputs 1 _ _ (by norm_num) normalizedGramUnit_mem_memory

/-- Actual dictionary outputs are affine in the joint learned coordinates.
Source: the exact memory inverse, while both embeddings and values change. -/
theorem jointMemoryForward_affine {N D : ℕ} (cap floor : ℝ)
    (p q : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hp : p ∈ jointMemoryDomain N D cap floor)
    (hq : q ∈ jointMemoryDomain N D cap floor) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    jointMemoryForward (a • p + b • q) = a • jointMemoryForward p + b • jointMemoryForward q := by
  have hm := jointMemoryDomain_convex N D cap floor hp hq ha hb hab
  rw [jointMemoryForward_eq_outputs cap floor _ hf hm,
    jointMemoryForward_eq_outputs cap floor _ hf hp,
    jointMemoryForward_eq_outputs cap floor _ hf hq]
  simp only [Prod.snd_add, Prod.smul_snd]

/-- Distinct output tables inhabit every joint-affinity hypothesis. -/
example : jointMemoryForward
    ((1 / 2 : ℝ) • (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ))) +
      (1 / 2 : ℝ) • (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (4 : ℝ)))) =
    (1 / 2 : ℝ) • jointMemoryForward (normalizedGramUnit,
      Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ))) +
    (1 / 2 : ℝ) • jointMemoryForward (normalizedGramUnit,
      Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (4 : ℝ))) :=
  jointMemoryForward_affine 1 _ _ _ (by norm_num)
    normalizedGramUnit_mem_memory normalizedGramUnit_mem_memory _ _
    (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
