import Transformer.GPTMini.Convex.JointInteractionFactorization

/-!
# The complete interaction tensor is observable in contexts of length two and three

Derived from the direct operator at f9749c7, following arXiv:2211.11052v1,
§3's attention-only product with the declared unnormalized predecessor-key
replacement. This addresses exact compression of that operator's function
class, rather than counting a particular implementation's stored tensor.

The outputs below are actual final-row evaluations on all two- and three-
token contexts. A fixed linear combination recovers every tensor entry
and every residual embedding entry. Thus no hidden gauge in this short-
context forward can make the complete tensor redundant.
This uses all possible short contexts, not just a given training sample.
It does not establish a parameter bound for normalized softmax or for an
arbitrary nonlinear replacement. No FFN, task loss, or optimizer is added.
-/

noncomputable section

namespace Transformer.GPTMini.Convex

/-- The context consisting of one token followed by a query token.
Source: the new direct operator's length-two observation construction. -/
def interactionPairTokens {V : ℕ} (q k : Fin V) (j : Fin 2) : Fin V :=
  if j = 0 then k else q

/-- A key/value pair followed by a query token.
Source: the new direct operator's length-three observation construction. -/
def interactionTripleTokens {V : ℕ} (q k v : Fin V) (j : Fin 3) : Fin V :=
  if j = 0 then k else if j = 1 then v else q

/-- All actual last-row outputs on the two short-context families.
Source: the new direct operator at f9749c7, with its causal boundary convention. -/
abbrev InteractionObservations (V D : ℕ) :=
  (Fin V → Fin V → Fin D → ℝ) × InteractionTensor V D

/-- The length-two actual output expands to an embedding and two interaction terms.
Source: direct evaluation of `jointInteractionForward`, including the repeated left boundary. -/
theorem interactionPair_forward {V D : ℕ} (p : JointInteraction V D)
    (q k : Fin V) (c : Fin D) :
    jointInteractionForward (interactionPairTokens q k) p 1 c =
      p.1 q c + p.2 q k k c + p.2 q k q c := by
  norm_num [jointInteractionForward, interactionPairTokens,
    interactionPrevious, Fin.sum_univ_two]
  ring

/-- The length-three actual output expands to the three consecutive pair contributions.
Source: direct evaluation of `jointInteractionForward`, rather than an independent output table. -/
theorem interactionTriple_forward {V D : ℕ} (p : JointInteraction V D)
    (q k v : Fin V) (c : Fin D) :
    jointInteractionForward (interactionTripleTokens q k v) p 2 c =
      p.1 q c + p.2 q k k c + p.2 q k v c + p.2 q v q c := by
  norm_num [jointInteractionForward, interactionTripleTokens,
    interactionPrevious, Fin.sum_univ_three]
  ring

/-- The observation map is linear because it evaluates the actual jointly linear forward.
Source: the new exact short-context observation construction. -/
def interactionObservationsMap (V D : ℕ) :
    JointInteraction V D →ₗ[ℝ] InteractionObservations V D where
  toFun p := (fun q k c => jointInteractionForward (interactionPairTokens q k) p 1 c,
    fun q k v c => jointInteractionForward (interactionTripleTokens q k v) p 2 c)
  map_add' := by
    intro p r
    apply Prod.ext
    · funext q k c
      exact congrFun (congrFun (jointInteractionForward_add (interactionPairTokens q k) p r) 1) c
    · funext q k v c
      exact congrFun (congrFun (jointInteractionForward_add (interactionTripleTokens q k v) p r) 2) c
  map_smul' := by
    intro a p
    apply Prod.ext
    · funext q k c
      exact congrFun (congrFun (jointInteractionForward_smul (interactionPairTokens q k) a p) 1) c
    · funext q k v c
      exact congrFun (congrFun (jointInteractionForward_smul (interactionTripleTokens q k v) a p) 2) c

/-- Five short-context evaluations recover an interaction coordinate by linear combination.
Source: the new observable-tensor identity, derived from the two forward expansions above. -/
def recoverInteractionMap (V D : ℕ) :
    InteractionObservations V D →ₗ[ℝ] InteractionTensor V D where
  toFun o q k v c := o.2 q k v c - o.2 q k k c + o.1 q k c - 2 * o.1 q v c + o.2 q v v c
  map_add' := by
    intro p r
    funext q k v c
    change (p.2 q k v c + r.2 q k v c) - (p.2 q k k c + r.2 q k k c) +
      (p.1 q k c + r.1 q k c) - 2 * (p.1 q v c + r.1 q v c) +
      (p.2 q v v c + r.2 q v v c) = _
    change _ = (p.2 q k v c - p.2 q k k c + p.1 q k c - 2 * p.1 q v c + p.2 q v v c) +
      (r.2 q k v c - r.2 q k k c + r.1 q k c - 2 * r.1 q v c + r.2 q v v c)
    ring
  map_smul' := by
    intro a p
    funext q k v c
    change a * p.2 q k v c - a * p.2 q k k c + a * p.1 q k c -
      2 * (a * p.1 q v c) + a * p.2 q v v c =
        a * (p.2 q k v c - p.2 q k k c + p.1 q k c - 2 * p.1 q v c + p.2 q v v c)
    ring

/-- Every tensor coordinate is recovered from the actual outputs, even with a free residual embedding.
Source: the new observable-tensor identity. It is uniform over all learned coefficients. -/
theorem recoverInteraction_observations {V D : ℕ} (p : JointInteraction V D) :
    recoverInteractionMap V D (interactionObservationsMap V D p) = p.2 := by
  funext q k v c
  change jointInteractionForward (interactionTripleTokens q k v) p 2 c -
      jointInteractionForward (interactionTripleTokens q k k) p 2 c +
      jointInteractionForward (interactionPairTokens q k) p 1 c -
      2 * jointInteractionForward (interactionPairTokens q v) p 1 c +
      jointInteractionForward (interactionTripleTokens q v v) p 2 c = _
  rw [interactionTriple_forward, interactionTriple_forward,
    interactionPair_forward, interactionPair_forward, interactionTriple_forward]
  ring

/-- Two repeated-token observations also recover each residual embedding coordinate.
Source: the new short-context reconstruction; no output-identification hypothesis is assumed. -/
def recoverInteractionEmbedding {V D : ℕ} (o : InteractionObservations V D) : Fin V → Fin D → ℝ :=
  fun q c => 3 * o.1 q q c - 2 * o.2 q q q c

/-- The recovered residual is the actual freely trained embedding table.
Source: the new short-context embedding reconstruction, cancelling all tensor contributions. -/
theorem recoverInteractionEmbedding_observations {V D : ℕ} (p : JointInteraction V D) :
    recoverInteractionEmbedding (interactionObservationsMap V D p) = p.1 := by
  funext q c
  change 3 * jointInteractionForward (interactionPairTokens q q) p 1 c -
    2 * jointInteractionForward (interactionTripleTokens q q q) p 2 c = _
  rw [interactionPair_forward, interactionTriple_forward]
  ring

/-- The entire joint parameter assignment is identifiable from these actual short-context outputs.
Source: the new embedding and tensor recovery identities, not a property of a chosen loss. -/
theorem interactionObservations_injective (V D : ℕ) :
    Function.Injective (interactionObservationsMap V D) := by
  intro p r he
  apply Prod.ext
  · have h := congrArg recoverInteractionEmbedding he
    simpa only [recoverInteractionEmbedding_observations] using h
  · have h := congrArg (recoverInteractionMap V D) he
    simpa only [recoverInteraction_observations] using h

/-- Pure tensor outputs form a genuine linear observation map without residual parameters.
Source: the direct operator restricted to a zero residual embedding, used as a coverage witness. -/
def tensorObservationsMap (V D : ℕ) : InteractionTensor V D →ₗ[ℝ] InteractionObservations V D :=
  (interactionObservationsMap V D).comp
    (LinearMap.inr ℝ (Fin V → Fin D → ℝ) (InteractionTensor V D))

/-- Pure tensor observations retain every interaction coordinate.
Source: the actual short-context recovery above, with the residual specialized to zero. -/
theorem recoverInteraction_tensorObservations {V D : ℕ} (C : InteractionTensor V D) :
    recoverInteractionMap V D (tensorObservationsMap V D C) = C := by
  change recoverInteractionMap V D (interactionObservationsMap V D (0, C)) = _
  exact recoverInteraction_observations (0, C)

end Transformer.GPTMini.Convex
