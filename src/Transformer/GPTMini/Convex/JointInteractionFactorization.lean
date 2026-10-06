import Transformer.GPTMini.Convex.JointInteraction

/-!
# Exact linear-head representation of the unfactorized joint operator

Derived from arXiv:2211.11052v1, §3's formula `eq:attention_only`,
and GPTMini's embedding/projection assembly at cbafbe9. Replace softmax by
unnormalized dot-product weights, use the predecessor token as the key,
and absorb the embedding and Q/K/value products into interaction coefficients.
This is a changed operator, not an equivalence to normalized softmax.

Every such physical linear head maps to a tensor. Conversely every tensor
is exactly a sum of V² width-one linear heads with original-token values.
This recovery is an optional algebraic representation after direct tensor
training; the optimizer need not search for or learn these selector heads.
Neither the tensor's cubic storage nor the recovered head count is compact.
The direct forward already has the token-to-stream prefix's output shape.
-/

noncomputable section

namespace Transformer.GPTMini.Convex

open scoped BigOperators

/-- Independent effective Q/K/value tables of an unnormalized contextual head.
Source: §3's dot products and original values, with the declared operator changes. -/
abbrev LinearTokenHead (V W D : ℕ) :=
  ((Fin V → Fin W → ℝ) × (Fin V → Fin W → ℝ)) × (Fin V → Fin D → ℝ)

/-- Full query/key/value-token coefficients of one genuine linear head.
Source: expansion of the §3 attention/value product without softmax. -/
def headInteractionTensor {V W D : ℕ} (h : LinearTokenHead V W D) : InteractionTensor V D :=
  fun q k v c => (∑ d, h.1.1 q d * h.1.2 k d) * h.2 v c

/-- The physical causal dot-product head reads the original value after the predecessor key.
Source: §3 with signed unnormalized weights and the binding encoder at cbafbe9. -/
def linearTokenHeadOutput {V W D T : ℕ} (h : LinearTokenHead V W D)
    (tokens : Fin T → Fin V) (row : Fin T) (c : Fin D) : ℝ :=
  ∑ j, if j ≤ row then (∑ d, h.1.1 (tokens row) d *
    h.1.2 (tokens (interactionPrevious j)) d) * h.2 (tokens j) c else 0

/-- Every original linear-head output equals the forward in its collected interaction coefficients.
Source: expansion of the actual learned Q/K/value products, not a convexity
claim in those original factored coordinates. -/
theorem linearTokenHeadOutput_eq_tensor {V W D T : ℕ} (h : LinearTokenHead V W D)
    (tokens : Fin T → Fin V) (row : Fin T) (c : Fin D) :
    linearTokenHeadOutput h tokens row c =
      jointInteractionForward tokens (0, headInteractionTensor h) row c := by
  unfold linearTokenHeadOutput jointInteractionForward
  simp only [Pi.zero_apply, zero_add]
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hv : j ≤ row
  · simp only [hv, ite_true, headInteractionTensor]
  · simp only [hv, ite_false]

/-- An ordinary learned token embedding followed by an ordinary trainable linear map.
Source: GPTMini's `embed` followed by Q/K/value projections at cbafbe9. -/
def tokenProjection {V E D : ℕ} (embedding : Fin V → Fin E → ℝ)
    (projection : Fin E → Fin D → ℝ) : Fin V → Fin D → ℝ :=
  fun token d => ∑ e, embedding token e * projection e d

/-- Effective tables retain all three products of the jointly trainable embedding and projections.
Source: GPTMini's embedding and Q/K/value projections at cbafbe9. A final
linear output projection can be absorbed into the value map before this step. -/
def projectedLinearHead {V E W D : ℕ} (embedding : Fin V → Fin E → ℝ)
    (query key : Fin E → Fin W → ℝ) (value : Fin E → Fin D → ℝ) : LinearTokenHead V W D :=
  ((tokenProjection embedding query, tokenProjection embedding key), tokenProjection embedding value)

/-- All learned embedding/projection/value products enter the direct tensor together.
Source: the new unfactorized replacement of the §3 product. Recovery is
exact for this linear operator; original low-width and penalty constraints
have not been transferred to a convex tensor domain. -/
theorem learnedEmbedding_linearHead_tensor {V E W D T : ℕ}
    (embedding : Fin V → Fin E → ℝ) (query key : Fin E → Fin W → ℝ)
    (value : Fin E → Fin D → ℝ) (tokens : Fin T → Fin V) (row : Fin T) (c : Fin D) :
    linearTokenHeadOutput (projectedLinearHead embedding query key value) tokens row c =
      jointInteractionForward tokens (0,
        headInteractionTensor (projectedLinearHead embedding query key value)) row c :=
  linearTokenHeadOutput_eq_tensor _ _ _ _

/-- A full token-code embedding recovers any effective projection table.
Source: the new algebraic shared-embedding realization. Its width is V,
so this construction makes no fixed-small-embedding-width claim. -/
theorem tokenProjection_oneHot {V D : ℕ} (P : Fin V → Fin D → ℝ) :
    tokenProjection (fun q e : Fin V => if q = e then 1 else 0) P = P := by
  funext q d
  unfold tokenProjection
  have hs : (∑ e : Fin V, (if q = e then (1 : ℝ) else 0) * P e d) =
      ∑ e : Fin V, if q = e then P e d else 0 := by
    apply Finset.sum_congr rfl
    intro e he
    by_cases h : q = e <;> simp only [h, ite_true, ite_false, one_mul, zero_mul]
  rw [hs]
  exact Fintype.sum_ite_eq q _

/-- All three effective tables come from one actual shared embedding and linear maps.
Source: the full token-code recovery above; width grows with vocabulary.
This supplies a physical embedding/attention representation after direct training. -/
theorem projectedLinearHead_oneHot {V W D : ℕ} (h : LinearTokenHead V W D) :
    projectedLinearHead (fun q e : Fin V => if q = e then 1 else 0)
      h.1.1 h.1.2 h.2 = h := by
  unfold projectedLinearHead
  rw [tokenProjection_oneHot, tokenProjection_oneHot, tokenProjection_oneHot]

/-- A selector head for a query/predecessor pair, carrying its freely learned original-token values.
Source: constructive algebraic recovery of the new direct tensor, without a pricing oracle. -/
def interactionBasisHead {V D : ℕ} (C : InteractionTensor V D)
    (h : Fin V × Fin V) : LinearTokenHead V 1 D :=
  (((fun q d => if q = h.1 ∧ d = 0 then 1 else 0),
    (fun k d => if k = h.2 ∧ d = 0 then 1 else 0)), C h.1 h.2)

/-- Each recovered physical head contributes exactly its designated tensor slice.
Source: explicit width-one recovery of the new operator; values are indexed by original tokens. -/
theorem interactionBasisHead_tensor {V D : ℕ} (C : InteractionTensor V D)
    (h : Fin V × Fin V) (q k v : Fin V) (c : Fin D) :
    headInteractionTensor (interactionBasisHead C h) q k v c =
      if q = h.1 ∧ k = h.2 then C h.1 h.2 v c else 0 := by
  unfold headInteractionTensor interactionBasisHead
  rw [Fin.sum_univ_one]
  by_cases hq : q = h.1 <;> by_cases hk : k = h.2 <;> simp [hq, hk]

/-- Every direct tensor, with all interactions free, has an exact finite original-head representation.
Source: the new recovery construction; no rank-one certificate or assumed
global head-search optimum is needed. The head count is V². -/
theorem interactionTensor_recovery {V D : ℕ} (C : InteractionTensor V D)
    (q k v : Fin V) (c : Fin D) :
    (∑ h : Fin V × Fin V, headInteractionTensor (interactionBasisHead C h) q k v c) = C q k v c := by
  rw [Fintype.sum_prod_type]
  simp_rw [interactionBasisHead_tensor]
  calc
    (∑ a : Fin V, ∑ b : Fin V, if q = a ∧ k = b then C a b v c else 0) =
        ∑ a : Fin V, if q = a then C a k v c else 0 := by
      apply Finset.sum_congr rfl
      intro a ha
      by_cases hq : q = a
      · simp only [hq, true_and, ite_true]
        exact Fintype.sum_ite_eq k _
      · simp only [hq, false_and, ite_false, Finset.sum_const_zero]
    _ = C q k v c := Fintype.sum_ite_eq q _

/-- The recovered physical head sum equals the full candidate attention output at every token row.
Source: the new exact finite recovery, including its actual causal mask and values. -/
theorem interactionTensor_recovery_forward {V D T : ℕ} (C : InteractionTensor V D)
    (tokens : Fin T → Fin V) (row : Fin T) (c : Fin D) :
    (∑ h : Fin V × Fin V, linearTokenHeadOutput (interactionBasisHead C h) tokens row c) =
      jointInteractionForward tokens (0, C) row c := by
  unfold linearTokenHeadOutput jointInteractionForward
  simp only [Pi.zero_apply, zero_add]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hv : j ≤ row
  · simp only [hv, ite_true]
    exact interactionTensor_recovery C _ _ _ _
  · simp only [hv, ite_false, Finset.sum_const_zero]

/-- Exact head recovery has a vocabulary-dependent quadratic number of heads.
Source: the explicit pair-indexed recovery; this is not a fixed-small-width guarantee. -/
theorem interactionTensor_recovery_head_count (V : ℕ) :
    Fintype.card (Fin V × Fin V) = V ^ 2 := by
  rw [Fintype.card_prod, Fintype.card_fin]
  ring

end Transformer.GPTMini.Convex
