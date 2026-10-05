import Transformer.GPTMini.Sparsemax.SharedMemoryExamples
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Causal prefix features that retain positions

Derived data encoder before sparsemax arXiv:1602.02068v2, Eq. (1).
Push the zero-score causal probability through any fixed position/token
feature map. The resulting codes mix learned query dictionary embeddings;
they never depend on desired outputs. The frequency encoder is the special
case in which this feature map ignores position.

Position/token pairs retain more information with a modest dictionary.
The three earlier binary prefixes now use six common dictionary slots.
Their codes are computed from the same observations and distinguish all
three contexts without assigning a private learned table to any context.
This feature family need not separate every possible nonlinear prefix task;
complete signatures supply the more expensive general construction.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

/-- Encode visible position/token features under uniform causal probabilities.
Source: the derived fixed input map before Eq. (1), using the actual
zero-score sparsemax probabilities rather than teacher routes. -/
def contextEncodedPrefixCodes {R T V N : ℕ} (key : Fin T → Fin V → Fin (N + 1))
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    Matrix (Fin R) (Fin (N + 1)) ℝ :=
  contextPrefixCode (fun r j => key j (tokens r j)) rows

/-- Every feature map preserves the probability-code requirement.
Source: the causal pushforward construction, with arbitrary repeated features.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextEncodedPrefixCodes_mem {R T V N : ℕ}
    (key : Fin T → Fin V → Fin (N + 1)) (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) : contextEncodedPrefixCodes key tokens rows ∈
      contextCodeDomain R N := contextPrefixCode_mem _ _

/-- All position/token feature codes are insensitive to future observations.
Source: causal sparsemax zeros, with no hypothesis on the feature map.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextEncodedPrefixCodes_eq_of_visible {R T V N : ℕ}
    (key : Fin T → Fin V → Fin (N + 1)) (tokens other : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (h : ∀ r j, j ≤ rows r → tokens r j = other r j) :
    contextEncodedPrefixCodes key tokens rows = contextEncodedPrefixCodes key other rows := by
  apply contextPrefixCode_eq_of_visible
  intro r j hj
  exact congrArg (key j) (h r j hj)

/-- Future changes inhabit the feature-code causality premise. -/
example : contextEncodedPrefixCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
    (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0) =
    contextEncodedPrefixCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
      (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2)) (fun _ => 0) := by
  apply contextEncodedPrefixCodes_eq_of_visible
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- Code multiplication averages the shared output table at observed feature slots.
Source: the derived probability pushforward and finite sum interchange.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextEncodedPrefixCodes_mul {R T V N D : ℕ}
    (key : Fin T → Fin V → Fin (N + 1)) (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) :
    contextEncodedPrefixCodes key tokens rows * Z = Matrix.of (fun r d =>
      ∑ j, sparseWeights (fun _ => 0) (rows r) j * Z (key j (tokens r j)) d) := by
  ext r d
  rw [Matrix.mul_apply]
  simp only [contextEncodedPrefixCodes, contextPrefixCode, Matrix.of_apply,
    Finset.sum_mul, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  simp only [Fintype.sum_ite_eq]

/-- Actual joint attention/value outputs use precisely this feature average.
Source: sparsemax Eq. (1), the exact global inverse chart and the input pushforward.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem jointEncodedPrefixForward_eq {R T V N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (key : Fin T → Fin V → Fin (N + 1))
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) :
    jointContextMemoryForward (contextEncodedPrefixCodes key tokens rows) (G, Z) =
      Matrix.of (fun r d => ∑ j,
        sparseWeights (fun _ => 0) (rows r) j * Z (key j (tokens r j)) d) := by
  rw [jointContextMemoryForward_eq cap floor _ _ hf (contextEncodedPrefixCodes_mem _ _ _) hG]
  exact contextEncodedPrefixCodes_mul _ _ _ _

/-- A nonzero shared table and genuine Gram inhabit the forward identity premises. -/
example : jointContextMemoryForward (contextEncodedPrefixCodes
    (fun _ : Fin 2 => fun _ : Fin 2 => (0 : Fin 1))
    (fun _ : Fin 1 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 0))
    (normalizedGramUnit, Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
    Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => ∑ j : Fin 2,
      sparseWeights (fun _ => 0) (0 : Fin 2) j * (3 : ℝ)) :=
  jointEncodedPrefixForward_eq 1 (3 / 4) _ _ _ _ _ (by norm_num)
    normalizedGramUnit_mem_memory

/-- Future changes preserve the actual joint forward under every feature encoder.
Source: causal probability pushforward before the common learned memory.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem jointEncodedPrefixForward_causal {R T V N D : ℕ}
    (key : Fin T → Fin V → Fin (N + 1)) (tokens other : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (h : ∀ r j, j ≤ rows r → tokens r j = other r j) :
    jointContextMemoryForward (contextEncodedPrefixCodes key tokens rows) p =
      jointContextMemoryForward (contextEncodedPrefixCodes key other rows) p := by
  rw [contextEncodedPrefixCodes_eq_of_visible key tokens other rows h]

/-- A nonzero shared table and a changed future observation satisfy the premise. -/
example : jointContextMemoryForward (contextEncodedPrefixCodes
    (fun _ : Fin 2 => fun k : Fin 2 => k)
    (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0))
    (0, Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ))) =
    jointContextMemoryForward (contextEncodedPrefixCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
      (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2)) (fun _ => 0))
      (0, Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ))) := by
  apply jointEncodedPrefixForward_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- Six dictionary identities are the three positions times the two token types.
Source: the derived positional encoder for the earlier repeated-prefix witness. -/
def positionExampleKey (j : Fin 3) (token : Fin 2) : Fin 6 := finProdFinEquiv (j, token)

/-- Position and token jointly determine distinct dictionary identities.
Source: the actual finite product enumeration, not context-specific slots.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExampleKey_injective :
    Function.Injective (fun p : Fin 3 × Fin 2 => positionExampleKey p.1 p.2) :=
  finProdFinEquiv.injective

/-- The same observed prefixes encoded with position/token pairs.
Source: the fixed data feature map; all rows still share one learned memory. -/
def positionExampleCodes : Matrix (Fin 3) (Fin 6) ℝ :=
  contextEncodedPrefixCodes positionExampleKey memoryExampleTokens (fun _ => 2)

/-- The positional witness satisfies the convex memory input domain.
Source: the generic causal feature pushforward, with real repeated observations.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExampleCodes_mem : positionExampleCodes ∈ contextCodeDomain 3 5 :=
  contextEncodedPrefixCodes_mem _ _ _

/-- Actual positional codes occupy slots `(1,3,5)`, `(0,3,5)` and `(0,2,5)`.
Source: zero-score sparsemax Proposition 1 on the same three visible tokens.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem positionExampleCodes_apply (r : Fin 3) (k : Fin 6) :
    positionExampleCodes r k = if r = 0 then
      (if k = 1 ∨ k = 3 ∨ k = 5 then 1 / 3 else 0)
    else if r = 1 then (if k = 0 ∨ k = 3 ∨ k = 5 then 1 / 3 else 0)
    else (if k = 0 ∨ k = 2 ∨ k = 5 then 1 / 3 else 0) := by
  unfold positionExampleCodes contextEncodedPrefixCodes contextPrefixCode
  simp only [Matrix.of_apply, sparseWeights_three_equal]
  fin_cases r <;> fin_cases k <;>
    norm_num [Fin.sum_univ_three, memoryExampleTokens, positionExampleKey, finProdFinEquiv]

end Transformer.GPTMini.Sparsemax
