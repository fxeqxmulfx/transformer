import Transformer.GPTMini.Sparsemax.PrototypeKernelTraining
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Convex geometry, evaluation and the finite-sample memory lower bound

Derived compact architecture for sparsemax arXiv:1602.02068v2, Eq. (1),
and `attn @ v` at `73f8a0b`. Kernel codes give a compact affine actual
forward in the global Gram/output chart. Every future convex output criterion
is convex on the same joint domain; no criterion is chosen for the model.

Finite-sample universality in this chart requires at least as many memory
slots as independently specifiable context rows. This is a matrix-rank
statement about the proved actual output M times Z, not an impossibility
for every transformer architecture. The prototype construction attains this
slot bound with one slot per distinct registered observation.

Dense positive kernel codes also force dense actual query routes under the
structural diagonal floor. This is a proved architectural cost. Memory Gram
supports can still vary, but sparsity of query attention is not retained.
The input features are fixed from data and output-only Gram freedom remains.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A data-code right inverse requires at least one memory slot per independent output row.
Source: the exact M-times-Z chart derived from arXiv:1602.02068v2, Eq. (1),
using finite matrix rank rather than a premise about target impossibility. -/
theorem finiteCode_slots_lowerBound {R W : ℕ} (M : Matrix (Fin R) (Fin W) ℝ)
    (C : Matrix (Fin W) (Fin R) ℝ) (hc : M * C = 1) : R ≤ W := by
  have hr := Matrix.rank_mul_le_left M C
  rw [hc, Matrix.rank_one, Fintype.card_fin] at hr
  exact hr.trans (Matrix.rank_le_width M)

/-- Two real shared slots attain the lower bound for two independent contexts. -/
example : (2 : ℕ) ≤ 2 := finiteCode_slots_lowerBound
  (1 : Matrix (Fin 2) (Fin 2) ℝ) 1 (Matrix.one_mul _)

/-- An exact all-vector-target prediction class imposes the same slot lower bound.
Source: arXiv:1602.02068v2, Eq. (1) factorization and finite rank of the common-value chart. -/
theorem sharedMemory_universal_slots {R N : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (hM : M ∈ contextCodeDomain R N)
    (hu : sharedMemoryPredictionSet R cap floor M = Set.univ) : R ≤ N + 1 := by
  have hi : (1 : Matrix (Fin R) (Fin R) ℝ) ∈ sharedMemoryPredictionSet R cap floor M := by
    rw [hu]
    exact Set.mem_univ _
  obtain ⟨G, hG, values, hy⟩ := hi
  rw [contextMemoryValueOutput_factor cap floor G M values hG hM] at hy
  exact finiteCode_slots_lowerBound M (memoryValueOutput G values) hy

/-- A real kernel code satisfies probability and exact universality at the slot bound. -/
example : (2 : ℕ) ≤ 1 + 1 := sharedMemory_universal_slots 1 (3 / 4)
  (prototypeTrainingCodes (fun r : Fin 2 => r) prototypeExampleFeatures)
  (prototypeKernelCodes_mem _ _ _ _)
  (sharedMemoryPredictionSet_prototype _ _ (by intro r s h; exact h))

/-- The actual learned-memory forward is one compact weighted kernel evaluation.
Source: arXiv:1602.02068v2, Eq. (1), the global inverse chart and normalized prototype codes. -/
theorem jointPrototypeForward_eq {Key : Type*} [DecidableEq Key] {R N F D : ℕ}
    (cap floor : ℝ) (observations : Fin R → Key) (prototypes : Fin (N + 1) → Key)
    (queries : Matrix (Fin R) (Fin F) ℝ) (features : Matrix (Fin (N + 1)) (Fin F) ℝ)
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hp : p ∈ jointMemoryDomain N D cap floor) :
    jointContextMemoryForward (prototypeKernelCodes observations prototypes queries features) p =
      Matrix.of (fun r d =>
        (∑ j, prototypeKernelScores observations prototypes queries features r j * p.2 j d) /
          prototypeKernelMass observations prototypes queries features r) := by
  rw [jointContextMemoryForward_eq cap floor _ _ hf (prototypeKernelCodes_mem _ _ _ _) hp]
  exact prototypeKernelCodes_mul _ _ _ _ _

/-- Two observations and a nonzero common table inhabit the actual evaluation premises. -/
example : jointContextMemoryForward
    (prototypeKernelCodes (fun r : Fin 2 => r) (fun r : Fin 2 => r)
      prototypeExampleFeatures prototypeExampleFeatures)
    (memoryIdentityGram 1, Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val : ℝ))) =
    Matrix.of (fun r _ =>
      (∑ j, prototypeKernelScores (fun r : Fin 2 => r) (fun r : Fin 2 => r)
        prototypeExampleFeatures prototypeExampleFeatures r j * (j.val : ℝ)) /
        prototypeKernelMass (fun r : Fin 2 => r) (fun r : Fin 2 => r)
          prototypeExampleFeatures prototypeExampleFeatures r) :=
  jointPrototypeForward_eq 1 (3 / 4) _ _ _ _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- Every convex output criterion retains convexity in compact joint learned coordinates.
Source: the affine arXiv:1602.02068v2, Eq. (1) memory chart with fixed kernel data codes. -/
theorem jointPrototypeObjective_convex {Key : Type*} [DecidableEq Key] {N F D : ℕ}
    (cap floor : ℝ) (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ)
    (objective : Matrix (Fin (N + 1)) (Fin D) ℝ → ℝ) (hf : 1 / 2 < floor)
    (hl : ConvexOn ℝ Set.univ objective) : ConvexOn ℝ (jointMemoryDomain N D cap floor)
      (fun p => objective (jointContextMemoryForward (prototypeTrainingCodes prototypes features) p)) :=
  jointContextMemoryObjective_convex cap floor _ objective hf (prototypeKernelCodes_mem _ _ _ _) hl

/-- A nonconstant convex criterion inhabits the abstract compact-model premise.
This mathematical witness does not select a downstream training objective. -/
example : ConvexOn ℝ (jointMemoryDomain 1 1 1 (3 / 4)) (fun p =>
    (jointContextMemoryForward (prototypeTrainingCodes (fun r : Fin 2 => r)
      prototypeExampleFeatures) p 0 0) ^ 2) := by
  apply jointPrototypeObjective_convex 1 (3 / 4) _ _ (fun Y => (Y 0 0) ^ 2) (by norm_num)
  have hs : ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) :=
    (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro X hx Y hy a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (X 0 0)) (Set.mem_univ (Y 0 0)) ha hb hab

/-- Positive input profiles force dense actual query attention under the diagonal floor.
Source: arXiv:1602.02068v2, Eq. (1) factorization in the compact architecture.
Memory support variation does not imply sparse query routes in this variant. -/
theorem contextPrototypeAttention_pos {Key : Type*} [DecidableEq Key] {R N F : ℕ}
    (cap floor : ℝ) (G : EmbeddingGram (N + 1)) (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) (r : Fin R) (j : Fin (N + 1)) :
    0 < contextMemoryAttention G (prototypeKernelCodes observations prototypes queries features) r j := by
  let M := prototypeKernelCodes observations prototypes queries features
  have hm := prototypeKernelCodes_mem observations prototypes queries features
  have hp := prototypeKernelCodes_pos observations prototypes queries features r j
  have hd : 0 < memoryGramAttention G j j := by
    have hg := memoryGramAttention_diag_floor cap floor G hG j
    linarith
  rw [contextMemoryAttention_eq_product cap floor G _ hG hm, Matrix.mul_apply]
  have hs := Finset.single_le_sum (s := Finset.univ)
    (f := fun k => M r k * memoryGramAttention G k j)
    (fun k hk => mul_nonneg ((hm r).1 k) ((memoryGramAttention_simplex G k).1 j))
    (Finset.mem_univ j)
  exact lt_of_lt_of_le (mul_pos hp hd) hs

/-- A genuine two-slot memory inhabits all dense-attention premises. -/
example : 0 < contextMemoryAttention (memoryIdentityGram 1)
    (prototypeTrainingCodes (fun r : Fin 2 => r) prototypeExampleFeatures) 0 1 :=
  contextPrototypeAttention_pos 1 (3 / 4) _ _ _ _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num)) _ _

/-- The global decoder compensates any feasible learned Gram change on all kernel queries.
Source: the actual convex shared-memory chart for arXiv:1602.02068v2, Eq. (1).
Compactness does not identify attention from an output-only criterion. -/
theorem jointPrototypeForward_gram_invariant {Key : Type*} [DecidableEq Key] {R N F D : ℕ}
    (cap floor : ℝ) (observations : Fin R → Key) (prototypes : Fin (N + 1) → Key)
    (queries : Matrix (Fin R) (Fin F) ℝ) (features : Matrix (Fin (N + 1)) (Fin F) ℝ)
    (G H : EmbeddingGram (N + 1)) (Z : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor)
    (hH : H ∈ memoryGramDomain N cap floor) :
    jointContextMemoryForward (prototypeKernelCodes observations prototypes queries features) (G, Z) =
      jointContextMemoryForward (prototypeKernelCodes observations prototypes queries features) (H, Z) :=
  jointContextMemoryForward_gram_invariant cap floor _ G H Z hf (prototypeKernelCodes_mem _ _ _ _)
    hG hH

/-- Different genuine Q/K families and changing memory supports inhabit the freedom's premises. -/
example : jointContextMemoryForward (prototypeTrainingCodes (fun r : Fin 2 => r)
    prototypeExampleFeatures) (memoryExampleStartGram, memoryExampleStartOutputs) =
    jointContextMemoryForward (prototypeTrainingCodes (fun r : Fin 2 => r)
      prototypeExampleFeatures) (memoryExampleStopGram, memoryExampleStartOutputs) :=
  jointPrototypeForward_gram_invariant 4 (3 / 4) _ _ _ _ _ _ _ (by norm_num)
    memoryExampleStartGram_mem memoryExampleStopGram_mem

end Transformer.GPTMini.Sparsemax
