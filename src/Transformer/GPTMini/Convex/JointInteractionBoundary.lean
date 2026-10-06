import Transformer.GPTMini.Convex.JointInteractionFactorization

/-!
# Capacity and compactness boundaries of the direct joint-coefficient route

Derived from the new linear replacement of arXiv:2211.11052v1, §3's
formula `eq:attention_only`, following the user's drop-in/unchanged-optimizer
requirements on 2026-10-06. The full tensor preserves neighboring key/value
binding and can distinguish the previously indistinguishable swapped tables.
It is a candidate token-to-stream prefix, not an implemented Python block.

Restricting the direct tensor to one original head, even with arbitrarily
large Q/K width, imposes a rank-one relation between query/key and value
coordinates. Two realizable tensors have a nonrealizable midpoint. Thus
this particular compression loses the convex free-parameter domain.
This is not an impossibility theorem for every compact reparameterization,
every changed attention operator, or multiple heads. No AdamW convergence
or normalized-softmax equivalence is asserted.
-/

noncomputable section

namespace Transformer.GPTMini.Convex

open scoped BigOperators

/-- A query/value minor at the fixed predecessor zero and output channel zero.
Source: the new direct tensor's single-head factorization boundary. -/
def interactionMinor (C : InteractionTensor 2 1) : ℝ :=
  C 0 0 0 0 * C 1 0 1 0 - C 0 0 1 0 * C 1 0 0 0

/-- Every original single head has a zero minor, regardless of its Q/K width.
Source: §3's score times original value, after replacing normalization
by the declared linear operator. Independent original values are retained. -/
theorem headInteractionMinor_zero {W : ℕ} (h : LinearTokenHead 2 W 1) :
    interactionMinor (headInteractionTensor h) = 0 := by
  unfold interactionMinor headInteractionTensor
  ring

/-- The exact direct-tensor image of one physical head of arbitrary finite width.
Source: the new original-head compression predicate; membership requires actual tables. -/
def singleHeadInteractionDomain : Set (InteractionTensor 2 1) :=
  {C | ∃ W : ℕ, ∃ h : LinearTokenHead 2 W 1, C = headInteractionTensor h}

/-- One isolated query/predecessor/value interaction, with both index choices used.
Source: explicit endpoint tensors for the original single-head compression. -/
def interactionDiagonal (a : Fin 2) : InteractionTensor 2 1 :=
  fun q k v c => if q = a ∧ k = 0 ∧ v = a ∧ c = 0 then 1 else 0

/-- Each isolated interaction has genuine independent Q/K and original-value tables.
Source: the new selector construction, evaluated on every tensor coordinate. -/
theorem interactionDiagonal_mem (a : Fin 2) : interactionDiagonal a ∈ singleHeadInteractionDomain := by
  refine ⟨1, interactionBasisHead (interactionDiagonal a) (a, 0), ?_⟩
  funext q k v c
  rw [interactionBasisHead_tensor]
  fin_cases a <;> fin_cases q <;> fin_cases k <;> fin_cases v <;> fin_cases c <;>
    norm_num [interactionDiagonal]

/-- The average of the two attainable tensors has a nonzero minor.
Source: an exact rational counterexample for single-head tensor compression. -/
theorem interactionDiagonal_midpoint_minor :
    interactionMinor ((1 / 2 : ℝ) • interactionDiagonal 0 +
      (1 / 2 : ℝ) • interactionDiagonal 1) = 1 / 4 := by
  norm_num [interactionMinor, interactionDiagonal]

/-- Even unlimited Q/K width does not make one-head interaction tensors a convex domain.
Source: the new rank-one value-coupling counterexample for the §3 linear product.
The statement concerns this compression, not all possible convex replacements. -/
theorem singleHeadInteractionDomain_not_convex : ¬ Convex ℝ singleHeadInteractionDomain := by
  intro hc
  have hm := hc (interactionDiagonal_mem 0) (interactionDiagonal_mem 1)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  obtain ⟨W, h, he⟩ := hm
  have hz := headInteractionMinor_zero h
  rw [← he, interactionDiagonal_midpoint_minor] at hz
  norm_num at hz

/-- The same two swapped binding contexts used at cbafbe9, written independently of sparsemax.
Source: `pairedBindingTokens` and the original recall binding counterexample. -/
def interactionBindingTokens (r : Fin 2) (j : Fin 6) : Fin 5 :=
  if j = 0 then 0 else if j = 1 then 1 else
    if j = 2 then (if r = 0 then 3 else 4) else if j = 3 then 2 else
      if j = 4 then (if r = 0 then 4 else 3) else 1

/-- A learned joint coefficient distinguishes the value immediately following the queried key.
Source: the new full-tensor binding witness. This uses neither route labels
nor an externally supplied attention distribution in its actual forward. -/
def interactionBindingParameters : JointInteraction 5 1 :=
  (0, fun q k v c => if q = 1 ∧ k = 1 ∧ v = 4 ∧ c = 0 then 1 else 0)

/-- The actual causal tensor forward gives different answers to the swapped tables.
Source: direct evaluation of the new learned-interaction binding witness. -/
theorem interactionBinding_forward (r : Fin 2) :
    jointInteractionForward (interactionBindingTokens r) interactionBindingParameters 5 0 =
      if r = 0 then 0 else 1 := by
  fin_cases r <;>
    norm_num [jointInteractionForward, interactionBindingParameters,
      interactionBindingTokens, interactionPrevious, Fin.sum_univ_six]

/-- Ordinary answer error on the two binding observations of the full tensor prefix.
Source: the answer-only target pair at cbafbe9; the forward is the new operator. -/
def interactionBindingError (p : JointInteraction 5 1) : ℝ :=
  (jointInteractionForward (interactionBindingTokens 0) p 5 0) ^ 2 +
    (jointInteractionForward (interactionBindingTokens 1) p 5 0 - 1) ^ 2

/-- The constructed joint tensor fits both binding answers exactly.
Source: actual output evaluation, without assuming a learned optimizer found the tensor. -/
theorem interactionBindingError_zero : interactionBindingError interactionBindingParameters = 0 := by
  unfold interactionBindingError
  rw [interactionBinding_forward, interactionBinding_forward]
  norm_num

/-- Ordinary answer error is nonnegative at every jointly trained coefficient assignment.
Source: the new explicit answer-only criterion; no feasibility constraint is used. -/
theorem interactionBindingError_nonneg (p : JointInteraction 5 1) : 0 ≤ interactionBindingError p := by
  unfold interactionBindingError
  positivity

/-- Ordinary answer error has its exact squared-distance gap in the complete joint parameter space.
Source: the new forward's actual joint linearity; no independent output table is substituted. -/
theorem interactionBindingError_gap (p q : JointInteraction 5 1) (a b : ℝ) (hab : a + b = 1) :
    interactionBindingError (a • p + b • q) =
      a * interactionBindingError p + b * interactionBindingError q - a * b *
        ((jointInteractionForward (interactionBindingTokens 0) p 5 0 -
          jointInteractionForward (interactionBindingTokens 0) q 5 0) ^ 2 +
         (jointInteractionForward (interactionBindingTokens 1) p 5 0 -
          jointInteractionForward (interactionBindingTokens 1) q 5 0) ^ 2) := by
  unfold interactionBindingError
  rw [jointInteractionForward_add, jointInteractionForward_add,
    jointInteractionForward_smul, jointInteractionForward_smul,
    jointInteractionForward_smul, jointInteractionForward_smul]
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  have hb : b = 1 - a := by linarith
  rw [hb]
  ring

/-- Distinct fitting and nonfitting joint coefficients inhabit the exact-gap hypothesis. -/
example : interactionBindingError ((1 / 2 : ℝ) • (0 : JointInteraction 5 1) +
    (1 / 2 : ℝ) • interactionBindingParameters) = 1 / 4 := by
  rw [interactionBindingError_gap _ _ _ _ (by norm_num)]
  norm_num [interactionBindingError, jointInteractionForward, interactionBindingParameters,
    interactionBindingTokens, interactionPrevious, Fin.sum_univ_six]

/-- The ordinary answer loss is jointly convex in the embedding and all interaction coordinates.
Source: exact affine-gap calculation for the true candidate block, over all free parameters. -/
theorem interactionBindingError_convex : ConvexOn ℝ Set.univ interactionBindingError := by
  refine ⟨convex_univ, ?_⟩
  intro p hp q hq a b ha hb hab
  rw [interactionBindingError_gap p q a b hab]
  have hs : 0 ≤
      (jointInteractionForward (interactionBindingTokens 0) p 5 0 -
        jointInteractionForward (interactionBindingTokens 0) q 5 0) ^ 2 +
      (jointInteractionForward (interactionBindingTokens 1) p 5 0 -
        jointInteractionForward (interactionBindingTokens 1) q 5 0) ^ 2 := by positivity
  have hm := mul_nonneg (mul_nonneg ha hb) hs
  change _ ≤ a * interactionBindingError p + b * interactionBindingError q
  linarith

/-- The binding witness is a global optimum on the entire free joint parameter space.
Source: the new exact fit and nonnegativity of ordinary answer error.
This is a capacity certificate, not a statement about AdamW convergence. -/
theorem interactionBinding_global_min (p : JointInteraction 5 1) :
    interactionBindingError interactionBindingParameters ≤ interactionBindingError p := by
  rw [interactionBindingError_zero]
  exact interactionBindingError_nonneg p

/-- The Basis-sized full tensor already has billions of independent coordinates at width 64.
Source: the explicit cubic tensor storage; at cbafbe9, synthetic/retrieval.py
uses 256 keys, 256 values and vocabulary.py's 36 reserved token IDs. -/
theorem interactionTensor_basis_coordinate_count :
    Fintype.card (Fin 548 × Fin 548 × Fin 548 × Fin 64) = 10532261888 := by
  rw [interactionTensor_coordinate_count]
  norm_num

end Transformer.GPTMini.Convex
