import Transformer.GPTMini.Sparsemax.PrefixMemoryCodes

/-!
# Exact common values for arbitrarily many context queries

Derived shared-memory architecture for sparsemax arXiv:1602.02068v2,
Eq. (1), followed by `attn @ v` at `73f8a0b`. All queries use the same
learned dictionary, the same learned Gram and one recovered value table.
The ordinary forward operation equals M times the global output coordinates,
proved from actual sparsemax, matrix associativity and the structural inverse.

Different contexts and repeated observations need no private value tables.
Both Q/K families and values change jointly in a convex parameter domain.
Data codes are fixed probability mixtures of learned queries. Causal prefix
codes are insensitive to future observations. The architecture attends to
parameter memory, not input occurrences, and its expressivity is limited
by the fixed code family. No task loss, FFN or generalization claim is made.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Actual context attention multiplied by one common learned memory value table.
Source: the derived memory version of `attn @ v` at `73f8a0b`. -/
def contextMemoryValueOutput {R N D : ℕ} (G : EmbeddingGram (N + 1))
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (values : Matrix (Fin (N + 1)) (Fin D) ℝ) :
    Matrix (Fin R) (Fin D) ℝ := contextMemoryAttention G M * values

/-- A common constant value vector gives that vector in every actual context.
Source: the normalized feasible rows of Eq. (1), followed by `attn @ v`. -/
theorem contextMemoryValueOutput_const {R N D : ℕ} (G : EmbeddingGram (N + 1))
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (value : Fin D → ℝ) :
    contextMemoryValueOutput G M (Matrix.of (fun _ => value)) = Matrix.of (fun _ => value) := by
  ext r d
  unfold contextMemoryValueOutput
  rw [Matrix.mul_apply]
  simp only [Matrix.of_apply]
  rw [← Finset.sum_mul, (contextMemoryAttention_simplex G M r).2.1, one_mul]

/-- Actual context outputs factor through the shared dictionary output.
Source: the proved sparsemax factorization and the ordinary value product. -/
theorem contextMemoryValueOutput_factor {R N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (values : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hG : G ∈ memoryGramDomain N cap floor) (hM : M ∈ contextCodeDomain R N) :
    contextMemoryValueOutput G M values = M * memoryValueOutput G values := by
  unfold contextMemoryValueOutput memoryValueOutput
  rw [contextMemoryAttention_eq_product cap floor G M hG hM, Matrix.mul_assoc]

/-- A real common value table inhabits both factorization premises. -/
example : contextMemoryValueOutput normalizedGramUnit
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) * memoryValueOutput normalizedGramUnit
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) :=
  contextMemoryValueOutput_factor 1 _ _ _ _ normalizedGramUnit_mem_memory unitContextCode_mem

/-- Evaluate every context with the single global decoded value table.
Source: the exact shared memory coordinates; the decoder has no context index. -/
def jointContextMemoryForward {R N D : ℕ} (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ) :
    Matrix (Fin R) (Fin D) ℝ := contextMemoryValueOutput p.1 M (recoverMemoryValues p.1 p.2)

/-- The actual jointly decoded output of every context is M times Z.
Source: the actual Eq. (1) factorization and exact global value recovery. -/
theorem jointContextMemoryForward_eq {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hp : p ∈ jointMemoryDomain N D cap floor) : jointContextMemoryForward M p = M * p.2 := by
  unfold jointContextMemoryForward
  rw [contextMemoryValueOutput_factor cap floor p.1 M _ hp hM,
    recoverMemoryValues_exact cap floor p.1 p.2 hf hp]

/-- Nonzero global outputs inhabit the shared-context recovery hypotheses. -/
example : jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
    (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) *
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) :=
  jointContextMemoryForward_eq 1 _ _ _ (by norm_num) unitContextCode_mem normalizedGramUnit_mem_memory

/-- Original global values are recovered for all contexts simultaneously.
Source: the bijective memory coordinates; no row-dependent recovery occurs. -/
theorem jointContextMemoryForward_original {R N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (values : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor) :
    jointContextMemoryForward M (G, memoryValueOutput G values) =
      contextMemoryValueOutput G M values := by
  unfold jointContextMemoryForward
  rw [recoverMemoryValues_original cap floor G values hf hG]

/-- A concrete common value table satisfies the original-coordinate premises. -/
example : jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
    (normalizedGramUnit, memoryValueOutput normalizedGramUnit
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)))) =
    contextMemoryValueOutput normalizedGramUnit
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) :=
  jointContextMemoryForward_original 1 _ _ _ _ (by norm_num) normalizedGramUnit_mem_memory

/-- Actual outputs across all contexts are affine in joint learned coordinates.
Source: the derived global memory chart, with fixed data codes and common values. -/
theorem jointContextMemoryForward_affine {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (p q : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hp : p ∈ jointMemoryDomain N D cap floor) (hq : q ∈ jointMemoryDomain N D cap floor)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    jointContextMemoryForward M (a • p + b • q) =
      a • jointContextMemoryForward M p + b • jointContextMemoryForward M q := by
  have hm := jointMemoryDomain_convex N D cap floor hp hq ha hb hab
  rw [jointContextMemoryForward_eq cap floor _ _ hf hM hm,
    jointContextMemoryForward_eq cap floor _ _ hf hM hp,
    jointContextMemoryForward_eq cap floor _ _ hf hM hq]
  simp only [Prod.snd_add, Prod.smul_snd, Matrix.mul_add, Matrix.mul_smul]

/-- Distinct learned outputs satisfy every shared-context affinity premise. -/
example : jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
    ((1 / 2 : ℝ) • (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ))) +
      (1 / 2 : ℝ) • (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (4 : ℝ)))) =
    (1 / 2 : ℝ) • jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
      (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ))) +
    (1 / 2 : ℝ) • jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
      (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (4 : ℝ))) :=
  jointContextMemoryForward_affine 1 _ _ _ _ (by norm_num) unitContextCode_mem
    normalizedGramUnit_mem_memory normalizedGramUnit_mem_memory _ _
    (by norm_num) (by norm_num) (by norm_num)

/-- Changing future observations does not change any actual decoded context output.
Source: the derived causal prefix encoder; this holds even outside the Gram domain. -/
theorem jointContextMemoryForward_causal {R T N D : ℕ}
    (tokens other : Fin R → Fin T → Fin (N + 1)) (rows : Fin R → Fin T)
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (ht : ∀ r j, j ≤ rows r → tokens r j = other r j) :
    jointContextMemoryForward (contextPrefixCode tokens rows) p =
      jointContextMemoryForward (contextPrefixCode other rows) p := by
  rw [contextPrefixCode_eq_of_visible tokens other rows ht]

/-- Different future token identities inhabit the actual-forward causality premise. -/
example : jointContextMemoryForward
    (contextPrefixCode (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0))
    (0, Matrix.of (fun i : Fin 2 => fun _ : Fin 1 => (i.val : ℝ))) =
    jointContextMemoryForward (contextPrefixCode
      (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2)) (fun _ => 0))
      (0, Matrix.of (fun i : Fin 2 => fun _ : Fin 1 => (i.val : ℝ))) := by
  apply jointContextMemoryForward_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

end Transformer.GPTMini.Sparsemax
