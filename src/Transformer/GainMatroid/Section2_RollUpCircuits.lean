module

public import Transformer.GainMatroid.Section2_RollUpBackward
public import Transformer.GainMatroid.Section2_RollUpForward

/-!
# The roll-ups and the unrolling keep the frame matroid

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "It is straightforward to check by comparing circuits that `F(Ĝ, B̂) = F(G, B)`."

> "For each unbalancing class `U` of `Σ_Ĝ(u)` there is a biased graph `(G_U, B_U)` for which
> `F(G_U, B_U) = F(G, B) = F(Ĝ, B̂)`, obtained from `(Ĝ, B̂)` by replacing each link `e = uv ∈ U`
> with a joint incident to `v` (this fact is straightforward to check by comparing circuits; it
> appears in [AlmostBalancedReps, Proposition 2.2])."

Rolling up a set `U` with the properties collected in `IsRollUpClass` keeps the frame circuits
(`IsRollUpClass.isFrameCircuit_rollUp_iff`), by `Section2_RollUpForward` and
`Section2_RollUpBackward`. Each unbalancing class of the unrolling has these properties, which
gives `F(G_U, B_U) = F(Ĝ, B̂)` (`IsUnrolling.isFrameCircuit_rollUp`). So has the set `J` of joints
off `u`, also when it is not a class, and its roll-up is `(G, B)`
(`IsUnrolling.graph_rollUp_offJoints`, `IsUnrolling.balanced_rollUp_offJoints`), which gives
`F(Ĝ, B̂) = F(G, B)` for the corrected `B̂` of `unrollBalanced` (`IsUnrolling.isFrameCircuit_iff`).
The matroids are compared by their circuits, which determine them (`Matroid.ext_isCircuit`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V}

/-- §2.5: rolling up a set with the properties of `IsRollUpClass` keeps the frame circuits. -/
theorem IsRollUpClass.isFrameCircuit_rollUp_iff {U : Set E} (hU : Ω.IsRollUpClass u U)
    (X : Set E) : (Ω.rollUp u U).IsFrameCircuit X ↔ Ω.IsFrameCircuit X :=
  ⟨hU.isFrameCircuit_of_rollUp, hU.isFrameCircuit_rollUp⟩

namespace IsUnrolling

/-- §2.5: "`F(G_U, B_U) = F(Ĝ, B̂)`", citing [AlmostBalancedReps, Proposition 2.2], compared by
circuits: an unbalancing class of the unrolling rolls up (`IsUnrolling.isRollUpClass`). That
`F(Ĝ, B̂) = F(G, B)` is `IsUnrolling.isFrameCircuit_iff`. -/
theorem isFrameCircuit_rollUp (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u) {U : Set E}
    (hU : U ∈ Ω'.unbalancingClasses u) (X : Set E) :
    (Ω'.rollUp u U).IsFrameCircuit X ↔ Ω'.IsFrameCircuit X :=
  (h.isRollUpClass hΩ hU).isFrameCircuit_rollUp_iff X

/-- §2.5: "`F(Ĝ, B̂) = F(G, B)`", compared by circuits as in the source, for the corrected `B̂` of
`unrollBalanced`: `(G, B)` is the roll-up of the unrolling along the joints `J` off `u`
(`IsUnrolling.isRollUpClass_offJoints`). -/
theorem isFrameCircuit_iff (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u) (X : Set E) :
    Ω'.IsFrameCircuit X ↔ Ω.IsFrameCircuit X := by
  have h' := (h.isRollUpClass_offJoints hΩ).isFrameCircuit_rollUp_iff X
  unfold IsFrameCircuit at h' ⊢
  rw [h.graph_rollUp_offJoints, h.balanced_rollUp_offJoints hΩ] at h'
  exact h'.symm

end IsUnrolling

end BiasedGraph

open BiasedGraph

example := pumpkin_isRollUpClass.isFrameCircuit_rollUp_iff {0}

example := twoJoints_isUnrolling.isFrameCircuit_rollUp twoJoints_isRollable twoJointsHat_class_mem

example := twoJoints_isUnrolling.isFrameCircuit_iff twoJoints_isRollable {4}

example := centreJoint_isUnrolling.isFrameCircuit_iff centreJoint_isRollable {4}

end Transformer.GainMatroid
