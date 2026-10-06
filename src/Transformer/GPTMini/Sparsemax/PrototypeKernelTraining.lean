import Transformer.GPTMini.Sparsemax.PrototypeKernelFeatures

/-!
# Exact compact interpolation through one common learned memory

Derived kernel memory architecture for sparsemax arXiv:1602.02068v2,
Eq. (1), followed by `attn @ v` at `73f8a0b`. The memory has one slot per
distinct registered observation. Its input profiles are the row-normalized
prototype kernel. Positive definiteness proves a fixed right inverse from
data alone, without assuming that desired target rows are attainable.

Every finite vector target table on these prototypes is therefore attainable
for every feasible learned memory Gram. One global inverse decodes original
values, and the actual joint forward remains affine on the convex domain.
The learned Gram is not fixed to the data kernel: the kernel supplies input
codes, while the Gram trains dictionary Q/K embeddings as before.

This is finite-sample compactness. Dense Gram parameters still grow
quadratically with prototype count, and arbitrary unseen outputs or efficient
generalization are not guaranteed. No FFN or task loss is selected.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

variable {Key : Type*} [DecidableEq Key]

/-- Normalized kernel profiles at the registered prototype observations.
Source: the derived compact memory input for arXiv:1602.02068v2, Eq. (1). -/
def prototypeTrainingCodes {N F : ℕ} (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) :
    Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  prototypeKernelCodes prototypes prototypes features features

/-- Normalization is multiplication by one positive inverse-mass diagonal.
Source: the actual compact kernel code preceding arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeTrainingCodes_eq_diagonal {N F : ℕ} (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) :
    prototypeTrainingCodes prototypes features =
      Matrix.diagonal (fun r => 1 / prototypeKernelMass prototypes prototypes features features r) *
        prototypeTrainingKernel prototypes features := by
  ext r j
  rw [Matrix.diagonal_mul]
  change prototypeKernelScores prototypes prototypes features features r j /
      prototypeKernelMass prototypes prototypes features features r =
    1 / prototypeKernelMass prototypes prototypes features features r *
      prototypeKernelScores prototypes prototypes features features r j
  ring

/-- A fixed right inverse, derived from the data kernel and its positive row masses.
Source: the compact input chart for arXiv:1602.02068v2, Eq. (1). -/
def prototypeTrainingRightInverse {N F : ℕ} (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) :
    Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  (prototypeTrainingKernel prototypes features)⁻¹ *
    Matrix.diagonal (prototypeKernelMass prototypes prototypes features features)

/-- The normalized code matrix has its actual derived right inverse on distinct observations.
Source: positive definite training kernel and positive masses before arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeTrainingCodes_rightInverse {N F : ℕ} (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (hs : Function.Injective prototypes) :
    prototypeTrainingCodes prototypes features * prototypeTrainingRightInverse prototypes features =
      1 := by
  rw [prototypeTrainingCodes_eq_diagonal, prototypeTrainingRightInverse, Matrix.mul_assoc,
    ← Matrix.mul_assoc (prototypeTrainingKernel prototypes features)
      (prototypeTrainingKernel prototypes features)⁻¹ _,
    Matrix.mul_nonsing_inv _ (prototypeTrainingKernel_det_unit prototypes features hs),
    Matrix.one_mul, Matrix.diagonal_mul_diagonal, ← Matrix.diagonal_one]
  congr 1
  funext r
  exact one_div_mul_cancel (ne_of_gt (prototypeKernelMass_pos _ _ _ _ r))

/-- Nonconstant data and distinct registered keys satisfy every inverse premise. -/
example : prototypeTrainingCodes (fun r : Fin 2 => r) prototypeExampleFeatures *
    prototypeTrainingRightInverse (fun r : Fin 2 => r) prototypeExampleFeatures = 1 :=
  prototypeTrainingCodes_rightInverse _ _ (by intro r s h; exact h)

/-- One compact global output table produces arbitrary vector targets on all prototypes.
Source: the data-derived right inverse for the arXiv:1602.02068v2, Eq. (1) input chart. -/
theorem prototypeTrainingCodes_target {N F D : ℕ} (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (Y : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hs : Function.Injective prototypes) : prototypeTrainingCodes prototypes features *
      (prototypeTrainingRightInverse prototypes features * Y) = Y := by
  rw [← Matrix.mul_assoc, prototypeTrainingCodes_rightInverse prototypes features hs, Matrix.one_mul]

/-- Distinct observations and a nonconstant target table inhabit the premise. -/
example : prototypeTrainingCodes (fun r : Fin 2 => r) prototypeExampleFeatures *
    (prototypeTrainingRightInverse (fun r : Fin 2 => r) prototypeExampleFeatures *
      Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val : ℝ))) =
    Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val : ℝ)) :=
  prototypeTrainingCodes_target _ _ _ (by intro r s h; exact h)

/-- Any feasible learned embedding Gram fits every target with one common original value table.
Source: arXiv:1602.02068v2, Eq. (1), the compact data inverse and the common `attn @ v` decoder. -/
theorem contextMemory_prototype_target {N F D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (Y : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hs : Function.Injective prototypes) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) :
    contextMemoryValueOutput G (prototypeTrainingCodes prototypes features)
      (recoverMemoryValues G (prototypeTrainingRightInverse prototypes features * Y)) = Y := by
  rw [contextMemoryValueOutput_factor cap floor G (prototypeTrainingCodes prototypes features) _ hG
      (prototypeKernelCodes_mem _ _ _ _),
    recoverMemoryValues_exact cap floor G _ hf hG]
  exact prototypeTrainingCodes_target prototypes features Y hs

/-- A genuine learned-memory Gram and nonconstant targets satisfy all premises. -/
example : contextMemoryValueOutput (memoryIdentityGram 1)
    (prototypeTrainingCodes (fun r : Fin 2 => r) prototypeExampleFeatures)
    (recoverMemoryValues (memoryIdentityGram 1)
      (prototypeTrainingRightInverse (fun r : Fin 2 => r) prototypeExampleFeatures *
        Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val : ℝ)))) =
    Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val : ℝ)) :=
  contextMemory_prototype_target 1 (3 / 4) _ _ _ _ (by intro r s h; exact h) (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- The same exact target recovery holds in the compact convex joint coordinates.
Source: the actual common-value chart for sparsemax arXiv:1602.02068v2, Eq. (1). -/
theorem jointPrototypeForward_target {N F D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (Y : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hs : Function.Injective prototypes) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) :
    jointContextMemoryForward (prototypeTrainingCodes prototypes features)
      (G, prototypeTrainingRightInverse prototypes features * Y) = Y :=
  contextMemory_prototype_target cap floor G prototypes features Y hs hf hG

/-- A real compact memory and nonconstant targets inhabit the joint-coordinate premises. -/
example : jointContextMemoryForward
    (prototypeTrainingCodes (fun r : Fin 2 => r) prototypeExampleFeatures)
    (memoryIdentityGram 1, prototypeTrainingRightInverse (fun r : Fin 2 => r) prototypeExampleFeatures *
      Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val : ℝ))) =
    Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val : ℝ)) :=
  jointPrototypeForward_target 1 (3 / 4) _ _ _ _ (by intro r s h; exact h) (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- Every current feasible compact point admits any simultaneous prototype-output correction.
Source: the actual affine arXiv:1602.02068v2, Eq. (1) chart and the data-derived kernel right inverse. -/
theorem jointPrototypeForward_step {N F D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (Z delta : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hs : Function.Injective prototypes) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) :
    jointContextMemoryForward (prototypeTrainingCodes prototypes features)
      (G, Z + prototypeTrainingRightInverse prototypes features * delta) =
      jointContextMemoryForward (prototypeTrainingCodes prototypes features) (G, Z) + delta := by
  rw [jointContextMemoryForward_eq cap floor (prototypeTrainingCodes prototypes features) _ hf
      (prototypeKernelCodes_mem _ _ _ _) hG,
    jointContextMemoryForward_eq cap floor (prototypeTrainingCodes prototypes features) _ hf
      (prototypeKernelCodes_mem _ _ _ _) hG, Matrix.mul_add,
    prototypeTrainingCodes_target prototypes features delta hs]

/-- A nonzero output correction inhabits all data and learned-memory premises. -/
example : jointContextMemoryForward (prototypeTrainingCodes (fun r : Fin 2 => r)
    prototypeExampleFeatures) (memoryIdentityGram 1,
      (0 : Matrix (Fin 2) (Fin 1) ℝ) +
      prototypeTrainingRightInverse (fun r : Fin 2 => r) prototypeExampleFeatures *
        Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ))) =
    jointContextMemoryForward (prototypeTrainingCodes (fun r : Fin 2 => r)
      prototypeExampleFeatures) (memoryIdentityGram 1, 0) +
      Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ)) :=
  jointPrototypeForward_step 1 (3 / 4) _ _ _ _ _ (by intro r s h; exact h) (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- The compact exact prediction class on distinct prototypes is the entire target space.
Source: the proved data inverse and actual arXiv:1602.02068v2, Eq. (1) memory prediction class. -/
theorem sharedMemoryPredictionSet_prototype {N F D : ℕ} (prototypes : Fin (N + 1) → Key)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (hs : Function.Injective prototypes) :
    sharedMemoryPredictionSet D 1 (3 / 4) (prototypeTrainingCodes prototypes features) = Set.univ := by
  rw [sharedMemoryPredictionSet_eq_range 1 _ (prototypeTrainingCodes prototypes features)
    (by norm_num) (prototypeKernelCodes_mem _ _ _ _)
    (memoryGramDomain_nonempty N)]
  ext Y
  constructor
  · intro h
    exact Set.mem_univ Y
  · intro h
    exact ⟨prototypeTrainingRightInverse prototypes features * Y,
      prototypeTrainingCodes_target prototypes features Y hs⟩

/-- The compact all-targets theorem has an actual distinct-prototype instance. -/
example : sharedMemoryPredictionSet 1 1 (3 / 4)
    (prototypeTrainingCodes (fun r : Fin 2 => r) prototypeExampleFeatures) = Set.univ :=
  sharedMemoryPredictionSet_prototype _ _ (by intro r s h; exact h)

end Transformer.GPTMini.Sparsemax
