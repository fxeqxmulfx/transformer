import Transformer.GPTMini.Sparsemax.SharedMemoryValues

/-!
# Convex training geometry and the exact shared-memory prediction class

Derived memory architecture for sparsemax arXiv:1602.02068v2, Eq. (1),
and `attn @ v` at `73f8a0b`. The joint Gram/output domain is convex and
every actual context output is M times Z. Thus any future convex objective
of the entire context output table is convex in the learned coordinates.
No particular task loss is selected; its convexity is an explicit premise.

The attainable output class is exactly the range of multiplication by the
fixed data-code matrix M. It is convex and expresses the coupling imposed
by a common value table. Arbitrary context targets are not automatically
attainable. Gram changes can be compensated by one global value decoder;
an output-only objective does not identify the dictionary attention.
This price is recorded alongside the positive convexity result.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Any convex criterion on all actual context outputs stays jointly convex.
Source: the derived exact shared-memory chart; FFN and task-loss choice remain open. -/
theorem jointContextMemoryObjective_convex {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (jointMemoryDomain N D cap floor)
      (fun p => objective (jointContextMemoryForward M p)) := by
  refine ⟨jointMemoryDomain_convex N D cap floor, ?_⟩
  intro p hp q hq a b ha hb hab
  change objective (jointContextMemoryForward M (a • p + b • q)) ≤
    a • objective (jointContextMemoryForward M p) + b • objective (jointContextMemoryForward M q)
  rw [jointContextMemoryForward_affine cap floor M p q hf hM hp hq a b ha hb hab]
  exact hl.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- A concrete nonconstant convex criterion inhabits the abstract premise.
This is a mathematical witness, not the choice of a downstream task loss. -/
example : ConvexOn ℝ (jointMemoryDomain 0 1 1 (3 / 4))
    (fun p => (jointContextMemoryForward
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) p 0 0) ^ 2) := by
  apply jointContextMemoryObjective_convex 1 (3 / 4) _ (fun Y => (Y 0 0) ^ 2)
    (by norm_num) unitContextCode_mem
  have hs : ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) :=
    (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro X _ Y _ a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (X 0 0)) (Set.mem_univ (Y 0 0)) ha hb hab

/-- Outputs attainable by actual context attention and one original value table.
Source: the prediction class of the derived shared-memory `attn @ v`. -/
def sharedMemoryPredictionSet {R N : ℕ} (D : ℕ) (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) : Set (Matrix (Fin R) (Fin D) ℝ) :=
  {Y | ∃ G ∈ memoryGramDomain N cap floor,
    ∃ values : Matrix (Fin (N + 1)) (Fin D) ℝ, contextMemoryValueOutput G M values = Y}

/-- The exact prediction class is a fixed linear image, not free context outputs.
Source: the derived actual sparsemax factorization and common-value bijection. -/
theorem sharedMemoryPredictionSet_eq_range {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hn : (memoryGramDomain N cap floor).Nonempty) :
    sharedMemoryPredictionSet D cap floor M =
      Set.range (fun Z : Matrix (Fin (N + 1)) (Fin D) ℝ => M * Z) := by
  ext Y
  constructor
  · rintro ⟨G, hG, values, hy⟩
    refine ⟨memoryValueOutput G values, ?_⟩
    rw [← hy, contextMemoryValueOutput_factor cap floor G M values hG hM]
  · rintro ⟨Z, hz⟩
    obtain ⟨G, hG⟩ := hn
    refine ⟨G, hG, recoverMemoryValues G Z, ?_⟩
    rw [contextMemoryValueOutput_factor cap floor G M _ hG hM,
      recoverMemoryValues_exact cap floor G Z hf hG]
    exact hz

/-- A real dictionary and data code inhabit the exact-class hypotheses. -/
example : sharedMemoryPredictionSet 1 1 (3 / 4)
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) =
    Set.range (fun Z : Matrix (Fin 1) (Fin 1) ℝ =>
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) * Z) :=
  sharedMemoryPredictionSet_eq_range 1 _ _ (by norm_num) unitContextCode_mem
    ⟨normalizedGramUnit, normalizedGramUnit_mem_memory⟩

/-- Actual jointly attainable outputs across all contexts form a convex set.
Source: the proved exact linear-image prediction class of the shared memory. -/
theorem sharedMemoryPredictionSet_convex {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hn : (memoryGramDomain N cap floor).Nonempty) :
    Convex ℝ (sharedMemoryPredictionSet D cap floor M) := by
  rw [sharedMemoryPredictionSet_eq_range cap floor M hf hM hn]
  intro Y hY W hW a b _ _ _
  obtain ⟨Z, hz⟩ := hY
  obtain ⟨X, hx⟩ := hW
  refine ⟨a • Z + b • X, ?_⟩
  change M * (a • Z + b • X) = a • Y + b • W
  change M * Z = Y at hz
  change M * X = W at hx
  rw [Matrix.mul_add, Matrix.mul_smul, Matrix.mul_smul, hz, hx]

/-- The actual prediction class contains zero whenever a memory Gram is feasible.
Source: `attn @ v` with one common zero value table, rather than private row values. -/
theorem zero_mem_sharedMemoryPredictionSet {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (hn : (memoryGramDomain N cap floor).Nonempty) :
    (0 : Matrix (Fin R) (Fin D) ℝ) ∈ sharedMemoryPredictionSet D cap floor M := by
  obtain ⟨G, hG⟩ := hn
  refine ⟨G, hG, 0, ?_⟩
  exact Matrix.mul_zero _

/-- A real feasible dictionary inhabits the zero-prediction premise. -/
example : (0 : Matrix (Fin 1) (Fin 1) ℝ) ∈ sharedMemoryPredictionSet 1 1 (3 / 4)
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) :=
  zero_mem_sharedMemoryPredictionSet 1 _ _ ⟨normalizedGramUnit, normalizedGramUnit_mem_memory⟩

/-- Concrete feasible embeddings and data inhabit every convex-class premise. -/
example : Convex ℝ (sharedMemoryPredictionSet 1 1 (3 / 4)
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))) :=
  sharedMemoryPredictionSet_convex 1 _ _ (by norm_num) unitContextCode_mem
    ⟨normalizedGramUnit, normalizedGramUnit_mem_memory⟩

/-- One global value decoder compensates a feasible Gram change for every context.
Source: the actual shared-memory chart; output-only training leaves this freedom. -/
theorem jointContextMemoryForward_gram_invariant {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (G H : EmbeddingGram (N + 1))
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hG : G ∈ memoryGramDomain N cap floor)
    (hH : H ∈ memoryGramDomain N cap floor) :
    jointContextMemoryForward M (G, Z) = jointContextMemoryForward M (H, Z) := by
  rw [jointContextMemoryForward_eq cap floor _ _ hf hM hG,
    jointContextMemoryForward_eq cap floor _ _ hf hM hH]

/-- The compensating freedom has a concrete nonzero-output instance. -/
example : jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
    (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
    jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
      (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) :=
  jointContextMemoryForward_gram_invariant 1 _ _ _ _ _ (by norm_num) unitContextCode_mem
    normalizedGramUnit_mem_memory normalizedGramUnit_mem_memory

/-- Equal data-code rows necessarily receive equal actual decoded outputs.
Source: the exact prediction class; shared-target attainability is not automatic. -/
theorem jointContextMemoryForward_eq_of_code_rows {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hp : p ∈ jointMemoryDomain N D cap floor) (r s : Fin R) (he : M r = M s) :
    jointContextMemoryForward M p r = jointContextMemoryForward M p s := by
  rw [jointContextMemoryForward_eq cap floor _ _ hf hM hp]
  funext d
  simp only [Matrix.mul_apply]
  rw [he]

/-- The row-equality and structural premises are jointly inhabited. -/
example : jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
    (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) 0 =
    jointContextMemoryForward (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)))
      (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) 0 :=
  jointContextMemoryForward_eq_of_code_rows 1 _ _ _ (by norm_num) unitContextCode_mem
    normalizedGramUnit_mem_memory _ _ rfl

end Transformer.GPTMini.Sparsemax
