import Transformer.GPTMini.Sparsemax.PrefixFeatureCodes

/-!
# Arbitrary three targets from six shared position/token memory slots

Derived architecture for sparsemax arXiv:1602.02068v2, Eq. (1), and
`attn @ v` at `73f8a0b`. The earlier frequency encoder forces a linear
relation between the outputs of `(1,1,1)`, `(0,1,1)` and `(0,0,1)`.
Position/token codes of these same observations have a fixed right inverse.
They therefore realize any three vector targets with one common value table.

For scalar targets `(a,b,c)`, slots 1, 2 and 5 of the global output table
are `3*(a-b)`, `3*(c-b)` and `3*b`, with all other slots zero. Averaging
the slots of each observed prefix gives exactly `a`, `b` and `c`.
The recovered original values also compensate any feasible Gram change.

The proof constructs global output coordinates from the targets and decodes
original values for every feasible learned Gram. It requires neither routing
targets nor fixed Q/K embeddings, values or sparse supports. The six slots
are shared features of positions and token types, not context identifiers.

This removes the specific three-target obstruction at modest dictionary cost.
Position-wise features do not represent every nonlinear causal task; complete
prefix keys provide that separate, exponentially larger construction. No
downstream loss, FFN or empirical language-model result is selected here.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- An explicit right inverse of the positional observation matrix.
Source: the derived three-prefix witness. Slots 1, 2 and 5 respectively
encode `(position 0, token 1)`, `(position 1, token 0)` and `(position 2, token 1)`. -/
def positionExampleRightInverse : Matrix (Fin 6) (Fin 3) ℝ := Matrix.of (fun k r =>
  if k = 1 then (if r = 0 then 3 else if r = 1 then -3 else 0)
  else if k = 2 then (if r = 1 then -3 else if r = 2 then 3 else 0)
  else if k = 5 ∧ r = 1 then 3 else 0)

/-- The actual data-derived positional codes have full row rank, constructively.
Source: the six-slot feature encoder; computing its fixed right inverse
removes the rank-two restriction of the earlier token-frequency matrix.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExampleCodes_rightInverse :
    positionExampleCodes * positionExampleRightInverse = 1 := by
  ext r s
  rw [Matrix.mul_apply, Fin.sum_univ_six]
  fin_cases r <;> fin_cases s <;>
    norm_num [positionExampleCodes_apply, positionExampleRightInverse, Matrix.one_apply]

/-- One global output table realizes any three vector outputs under the positional codes.
Source: the explicit right inverse; output dimension is unrestricted and finite.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExampleCodes_target {D : ℕ} (Y : Matrix (Fin 3) (Fin D) ℝ) :
    positionExampleCodes * (positionExampleRightInverse * Y) = Y := by
  rw [← Matrix.mul_assoc, positionExampleCodes_rightInverse, Matrix.one_mul]

/-- The whole vector-valued target space is attained, not just one scalar witness.
Source: the derived position/token right inverse for the same three observations.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExampleCodes_surjective (D : ℕ) :
    Function.Surjective (fun Z : Matrix (Fin 6) (Fin D) ℝ => positionExampleCodes * Z) := by
  intro Y
  exact ⟨positionExampleRightInverse * Y, positionExampleCodes_target Y⟩

/-- Decode one common original value table from arbitrary three target rows.
Source: the global memory inverse, with fixed position/token data codes. -/
def positionExampleValues {D : ℕ} (G : EmbeddingGram 6) (Y : Matrix (Fin 3) (Fin D) ℝ) :
    Matrix (Fin 6) (Fin D) ℝ := recoverMemoryValues G (positionExampleRightInverse * Y)

/-- Every feasible learned Gram admits one common value table giving all three targets.
Source: actual sparsemax Eq. (1), shared original `attn @ v` and the fixed right inverse.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextMemory_positionExample_targets {D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram 6) (Y : Matrix (Fin 3) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain 5 cap floor) :
    contextMemoryValueOutput G positionExampleCodes (positionExampleValues G Y) = Y := by
  unfold positionExampleValues
  rw [contextMemoryValueOutput_factor cap floor G _ _ hG positionExampleCodes_mem,
    recoverMemoryValues_exact cap floor G _ hf hG]
  exact positionExampleCodes_target Y

/-- Real embeddings and nonconstant vector outputs satisfy both structural premises. -/
example : contextMemoryValueOutput (memoryIdentityGram 5) positionExampleCodes
    (positionExampleValues (memoryIdentityGram 5)
      (Matrix.of (fun r : Fin 3 => fun d : Fin 2 => (r.val + d.val : ℝ)))) =
    Matrix.of (fun r : Fin 3 => fun d : Fin 2 => (r.val + d.val : ℝ)) :=
  contextMemory_positionExample_targets 1 (3 / 4) _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- The same attainment occurs in the exact convex joint Gram/output coordinates.
Source: the proved actual common-value decoder, with no private row parameters.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem jointPositionExampleForward_targets {D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram 6) (Y : Matrix (Fin 3) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain 5 cap floor) :
    jointContextMemoryForward positionExampleCodes (G, positionExampleRightInverse * Y) = Y := by
  exact contextMemory_positionExample_targets cap floor G Y hf hG

/-- A genuine shared memory and different targets inhabit the joint-chart premises. -/
example : jointContextMemoryForward positionExampleCodes (memoryIdentityGram 5,
    positionExampleRightInverse * Matrix.of (fun r : Fin 3 => fun _ : Fin 1 => (r.val : ℝ))) =
    Matrix.of (fun r : Fin 3 => fun _ : Fin 1 => (r.val : ℝ)) :=
  jointPositionExampleForward_targets 1 (3 / 4) _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- Every current feasible point admits any desired three-output correction.
Source: the exact affine memory chart and the positional right inverse.
The update is in one common output table, with no attention target or task loss.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem jointPositionExampleForward_step {D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram 6) (Z : Matrix (Fin 6) (Fin D) ℝ)
    (delta : Matrix (Fin 3) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain 5 cap floor) :
    jointContextMemoryForward positionExampleCodes (G, Z + positionExampleRightInverse * delta) =
      jointContextMemoryForward positionExampleCodes (G, Z) + delta := by
  rw [jointContextMemoryForward_eq cap floor _ _ hf positionExampleCodes_mem hG,
    jointContextMemoryForward_eq cap floor _ _ hf positionExampleCodes_mem hG,
    Matrix.mul_add, positionExampleCodes_target]

/-- A nonzero correction and genuine Gram inhabit the pointwise update premises. -/
example : jointContextMemoryForward positionExampleCodes (memoryIdentityGram 5,
    (0 : Matrix (Fin 6) (Fin 1) ℝ) + positionExampleRightInverse *
      Matrix.of (fun r : Fin 3 => fun _ : Fin 1 => (r.val + 1 : ℝ))) =
    jointContextMemoryForward positionExampleCodes (memoryIdentityGram 5, 0) +
      Matrix.of (fun r : Fin 3 => fun _ : Fin 1 => (r.val + 1 : ℝ)) :=
  jointPositionExampleForward_step 1 (3 / 4) _ _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- The exact prediction class for the three original prefixes is now every target table.
Source: actual sparsemax/value factorization and the constructive positional right inverse.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem sharedMemoryPredictionSet_positionExample (D : ℕ) :
    sharedMemoryPredictionSet D 1 (3 / 4) positionExampleCodes = Set.univ := by
  rw [sharedMemoryPredictionSet_eq_range 1 _ _ (by norm_num) positionExampleCodes_mem
    (memoryGramDomain_nonempty 5)]
  ext Y
  constructor
  · intro h
    exact Set.mem_univ Y
  · intro h
    exact positionExampleCodes_surjective D Y

/-- The formerly impossible frequency-code target triple, as an ordinary output table.
Source: `memoryExample_target_triple_unattainable`; only the data encoder is enriched. -/
def positionExampleCounterTarget : Matrix (Fin 3) (Fin 1) ℝ :=
  Matrix.of (fun r _ => if r = 2 then 1 else 0)

/-- Only the shared `(position 1, token 0)` slot has nonzero value three.
Source: an explicit original-value witness for the formerly excluded triple. -/
def positionExampleCounterValues : Matrix (Fin 6) (Fin 1) ℝ :=
  Matrix.of (fun k _ => if k = 2 then 3 else 0)

/-- The concrete original values give `(0,0,1)` through actual sparsemax and value mixing.
Source: the explicit position/token witness, with a genuine identity feature Gram.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExample_counterTarget_values :
    contextMemoryValueOutput (memoryIdentityGram 5) positionExampleCodes
      positionExampleCounterValues = positionExampleCounterTarget := by
  have hg := memoryIdentityGram_mem 5 1 (3 / 4) (by norm_num) (by norm_num)
  rw [contextMemoryValueOutput_factor 1 _ _ _ _ hg positionExampleCodes_mem,
    memoryValueOutput, memoryGramAttention_normalized 1 _ _ hg,
    memoryIdentityGram_scores, Matrix.one_mul]
  ext r d
  rw [Matrix.mul_apply, Fin.sum_univ_six]
  fin_cases r <;> fin_cases d <;>
    norm_num [positionExampleCodes_apply, positionExampleCounterValues, positionExampleCounterTarget]

/-- The target `(0,0,1)` is exactly attainable by one actual shared-memory value table.
Source: the constructive position/token encoder, refuting a universal three-target obstruction.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExample_counterTarget_attainable :
    ∃ G ∈ memoryGramDomain 5 1 (3 / 4), ∃ values : Matrix (Fin 6) (Fin 1) ℝ,
      contextMemoryValueOutput G positionExampleCodes values 0 0 = 0 ∧
      contextMemoryValueOutput G positionExampleCodes values 1 0 = 0 ∧
      contextMemoryValueOutput G positionExampleCodes values 2 0 = 1 := by
  have hg := memoryIdentityGram_mem 5 1 (3 / 4) (by norm_num) (by norm_num)
  refine ⟨memoryIdentityGram 5, hg, positionExampleCounterValues, ?_⟩
  rw [positionExample_counterTarget_values]
  norm_num [positionExampleCounterTarget]

end Transformer.GPTMini.Sparsemax
