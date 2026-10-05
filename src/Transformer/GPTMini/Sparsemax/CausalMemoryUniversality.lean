import Transformer.GPTMini.Sparsemax.CausalPrefixKeys

/-!
# All consistent finite causal outputs in the convex shared-memory chart

Derived architecture for sparsemax arXiv:1602.02068v2, Eq. (1), followed
by `attn @ v` at `73f8a0b`. Complete prefix codes depend only on observed
tokens and query positions. Their shared learned memory realizes exactly
all deterministic output tables consistent with those observations.

The sufficiency theorem holds for every feasible learned Gram, not merely
identity attention. The original common values are recovered through the
same global inverse. Q/K families and sparse supports can therefore change
while the joint chart remains convex and its actual output map affine.

The theorem removes the rank loss of the earlier frequency encoder. Its
cost is the exponential dictionary already counted in `CausalPrefixKeys`.
It addresses a bounded finite window and the stated memory architecture;
it does not prove efficient ordinary self-attention or language generalization.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Desired outputs are consistent on identical observed prefixes.
Source: the derived deterministic causal memory model; hidden future tokens
cannot distinguish targets, while different visible prefixes are unconstrained. -/
def causalCompatibleTargets {R T V D : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (Y : Matrix (Fin R) (Fin D) ℝ) : Prop :=
  ∀ r s, rows r = rows s → (∀ k, k ≤ rows r → tokens r k = tokens s k) → Y r = Y s

/-- Equal-code consistency is precisely consistency on observed causal data.
Source: the proved complete prefix-key characterization, including query position.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem codeCompatibleTargets_prefix_iff {R T V D : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (Y : Matrix (Fin R) (Fin D) ℝ) :
    codeCompatibleTargets (fun r => causalPrefixKey (tokens r) (rows r)) Y ↔
      causalCompatibleTargets tokens rows Y := by
  constructor
  · intro h r s hp ht
    exact h r s ((causalPrefixKey_eq_iff _ _ _ _).mpr ⟨hp, ht⟩)
  · intro h r s he
    obtain ⟨hp, ht⟩ := (causalPrefixKey_eq_iff _ _ _ _).mp he
    exact h r s hp ht

/-- A single original value table fits exactly the observation-consistent targets.
Source: actual sparsemax Eq. (1), complete causal codes and the shared-value inverse.
This holds for every feasible Gram; it never freezes attention to identity.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextMemory_fullPrefix_attainable_iff {R T V D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (prefixMemoryN T V + 1)) (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (Y : Matrix (Fin R) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain (prefixMemoryN T V) cap floor) :
    (∃ values : Matrix (Fin (prefixMemoryN T V + 1)) (Fin D) ℝ,
      contextMemoryValueOutput G (contextFullPrefixCodes tokens rows) values = Y) ↔
      causalCompatibleTargets tokens rows Y := by
  rw [← codeCompatibleTargets_prefix_iff]
  exact contextMemory_categorical_attainable_iff cap floor G _ Y hf hG

/-- Repeated observations and a nonzero target inhabit the structural premises. -/
example : (∃ values : Matrix (Fin (prefixMemoryN 2 2 + 1)) (Fin 1) ℝ,
    contextMemoryValueOutput (memoryIdentityGram (prefixMemoryN 2 2))
      (contextFullPrefixCodes (fun _ : Fin 2 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 1))
      values = Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (3 : ℝ))) ↔
    causalCompatibleTargets (fun _ : Fin 2 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 1)
      (Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (3 : ℝ))) :=
  contextMemory_fullPrefix_attainable_iff 1 (3 / 4) _ _ _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- Distinct observed prefixes permit arbitrary outputs, in any finite output dimension.
Source: the complete causal encoder and its exact shared-memory attainable class.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextMemory_fullPrefix_distinct_targets {R T V D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (prefixMemoryN T V + 1)) (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (Y : Matrix (Fin R) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain (prefixMemoryN T V) cap floor)
    (hsep : ∀ r s, rows r = rows s →
      (∀ k, k ≤ rows r → tokens r k = tokens s k) → r = s) :
    ∃ values : Matrix (Fin (prefixMemoryN T V + 1)) (Fin D) ℝ,
      contextMemoryValueOutput G (contextFullPrefixCodes tokens rows) values = Y := by
  apply (contextMemory_fullPrefix_attainable_iff cap floor G tokens rows Y hf hG).mpr
  intro r s hp ht
  rw [hsep r s hp ht]

/-- Two different observed prefixes and different targets inhabit all premises. -/
example : ∃ values : Matrix (Fin (prefixMemoryN 2 2 + 1)) (Fin 1) ℝ,
    contextMemoryValueOutput (memoryIdentityGram (prefixMemoryN 2 2))
      (contextFullPrefixCodes (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 1)) values =
      Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ)) := by
  apply contextMemory_fullPrefix_distinct_targets 1 (3 / 4) _ _ _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))
  intro r s hp ht
  exact ht 0 (by norm_num)

/-- The explicit convex coordinates produce every consistent causal target table.
Source: the complete observation extension followed by actual shared-memory decoding.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem jointFullPrefixForward_target {R T V D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (prefixMemoryN T V + 1)) (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (Y : Matrix (Fin R) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain (prefixMemoryN T V) cap floor)
    (hY : causalCompatibleTargets tokens rows Y) :
    jointContextMemoryForward (contextFullPrefixCodes tokens rows)
      (G, extendCodeOutputs (fun r => causalPrefixKey (tokens r) (rows r)) Y) = Y := by
  rw [jointContextMemoryForward_eq cap floor _ _ hf (contextFullPrefixCodes_mem _ _) hG]
  change oneHotContextCodes _ * extendCodeOutputs _ Y = Y
  rw [oneHotContextCodes_mul]
  have hc := (codeCompatibleTargets_prefix_iff tokens rows Y).mpr hY
  ext r d
  exact congrFun (extendCodeOutputs_at_code _ Y hc r) d

/-- Concrete repeated prefixes satisfy all premises for exact target recovery. -/
example : jointContextMemoryForward
    (contextFullPrefixCodes (fun _ : Fin 2 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 1))
    (memoryIdentityGram (prefixMemoryN 2 2), extendCodeOutputs
      (fun _ : Fin 2 => causalPrefixKey (fun _ : Fin 2 => (1 : Fin 2)) 1)
      (Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (3 : ℝ)))) =
    Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (3 : ℝ)) :=
  jointFullPrefixForward_target 1 (3 / 4) _ _ _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num)) (by intro r s hp ht; rfl)

/-- The exact causal prediction class has no restrictions beyond observed-data consistency.
Source: the derived complete encoder and a genuinely inhabited memory domain.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem sharedMemoryPredictionSet_fullPrefix {R T V D : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    sharedMemoryPredictionSet D 1 (3 / 4) (contextFullPrefixCodes tokens rows) =
      {Y | causalCompatibleTargets tokens rows Y} := by
  rw [contextFullPrefixCodes, sharedMemoryPredictionSet_categorical 1 _ _ (by norm_num)
    (memoryGramDomain_nonempty _)]
  ext Y
  exact codeCompatibleTargets_prefix_iff tokens rows Y

/-- This full causal output class is convex despite changing learned embeddings and supports.
Source: the exact shared-memory affine chart for complete prefix codes.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem sharedMemoryPredictionSet_fullPrefix_convex {R T V D : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    Convex ℝ (sharedMemoryPredictionSet D 1 (3 / 4) (contextFullPrefixCodes tokens rows)) :=
  sharedMemoryPredictionSet_convex 1 _ _ (by norm_num) (contextFullPrefixCodes_mem _ _)
    (memoryGramDomain_nonempty _)

/-- Different future tokens leave the actual joint forward unchanged for every parameter.
Source: complete prefix-code causality, independent of the inverse-domain restriction.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem jointFullPrefixForward_causal {R T V D : ℕ}
    (tokens other : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (p : EmbeddingGram (prefixMemoryN T V + 1) ×
      Matrix (Fin (prefixMemoryN T V + 1)) (Fin D) ℝ)
    (h : ∀ r k, k ≤ rows r → tokens r k = other r k) :
    jointContextMemoryForward (contextFullPrefixCodes tokens rows) p =
      jointContextMemoryForward (contextFullPrefixCodes other rows) p := by
  rw [contextFullPrefixCodes_eq_of_visible tokens other rows h]

/-- A future change and a nonzero output table inhabit the actual causality premise. -/
example : jointContextMemoryForward
    (contextFullPrefixCodes (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0))
    (0, Matrix.of (fun k : Fin (prefixMemoryN 2 2 + 1) => fun _ : Fin 1 => (k.val : ℝ))) =
    jointContextMemoryForward (contextFullPrefixCodes
      (fun _ : Fin 1 => fun k : Fin 2 => if k = 0 then 0 else (1 : Fin 2)) (fun _ => 0))
      (0, Matrix.of (fun k : Fin (prefixMemoryN 2 2 + 1) => fun _ : Fin 1 => (k.val : ℝ))) := by
  apply jointFullPrefixForward_causal
  intro r k hk
  fin_cases k
  · rfl
  · norm_num at hk

end Transformer.GPTMini.Sparsemax
