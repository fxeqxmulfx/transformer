module

public import Transformer.GainMatroid.Section5_FanRollUp

/-!
# Theorems 5.5, 5.6 and 5.9: every representation is canonical

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §5.2. A representation `A` of a matroid represented
by a properly unbalanced biased graph is projectively equivalent to a canonical lift matrix or to a
canonical frame matrix particular to it, but not both (Theorem 5.6, and Theorem 5.5 for 3-connected
matroids of rank greater than two). For an almost-balanced biased graph, `A` is particular as a
lift matrix to `(Ĝ, B̂)` and, over a field other than `GF(2)`, as a frame matrix to each roll-up of
`(Ĝ, B̂)` (Theorems 5.5 and 5.9).

`(Ĝ, B̂)` is the unrolling to the unique balancing vertex after deleting joints (`IsRollable`).
When the biased graph is balanced after deleting its joints, the source (§2.5) unrolls them to a new
vertex instead, giving the balanced biased graph `(H, 𝒞(H)) = unrollNew` whose only roll-up is the
biased graph itself, with the new vertex isolated. The almost-balanced conclusions therefore name
one of the two cases.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {R F : Type*} [Field F]

namespace OrientedGraph

/-- The biased graph on `G` in which every cycle is balanced. -/
def balancedBiased (G : OrientedGraph V E) : BiasedGraph V E where
  graph := G
  balanced := {C | G.IsCycle C}
  isCycle_of_mem _ h := h
  theta h _ _ := h.isCycle.2.2

/-- `φ` is switching equivalent to a gain function vanishing exactly on the links not at `u` and on
the loops of `M` (Theorem 5.9). -/
def IsDerivedZeroPattern (G : OrientedGraph V E) (u : V) (M : Matroid E) (φ : E → F) : Prop :=
  ∃ φ', AddSwitchEquiv G φ φ' ∧
    ∀ e, φ' e = 0 ↔ (G.IsLink e ∧ G.tail e ≠ u ∧ G.head e ≠ u) ∨ M.IsLoop e

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

open Classical in
/-- `H` (§2.5): the joints of `Ω` become links to a new vertex `none`. -/
noncomputable def unrollNewGraph (Ω : BiasedGraph V E) : OrientedGraph (Option V) E :=
  ⟨fun e => some (Ω.graph.tail e), fun e => if Ω.IsJoint e then none else some (Ω.graph.head e)⟩

/-- `(H, 𝒞(H))` (§2.5): for `Ω` balanced after deleting its joints, the joints unrolled to a new
vertex, with every cycle balanced. -/
noncomputable def unrollNew (Ω : BiasedGraph V E) : BiasedGraph (Option V) E :=
  Ω.unrollNewGraph.balancedBiased

variable [Finite V] [Finite E] [DecidableEq V] {Ω : BiasedGraph V E}

/-- **Theorem 5.5.** "Let `M` be a 3-connected matroid of rank greater than two, and let `𝔽` be a
field. Let `A` be a matrix over `𝔽` representing `M` and let `(G, B)` be a biased graph representing
`M`. If `(G, B)` is properly unbalanced then exactly one of the following holds. (i) `A` is
projectively equivalent to a canonical lift matrix particular to `(G, B)`, or (ii) `A` is
projectively equivalent to a canonical frame matrix particular to `(G, B)`. If `(G, B)` is
almost-balanced then each of the following hold, unless `𝔽` is `GF(2)`, in which case (i) holds.
(i) `A` is projectively equivalent to a canonical lift matrix particular to `(Ĝ, B̂)`, and (ii) `A`
is projectively equivalent to a canonical frame matrix particular to each roll-up of `(Ĝ, B̂)`."

The almost-balanced part assumes in addition that `M = F(G, B)`. The source treats an
almost-balanced biased graph as having `F(G, B) = L(G, B)`, which fails when a joint avoids an
unbalanced cycle; and `(Ĝ, B̂)` and its roll-ups represent `F(G, B)` (§2.5). Take `K₄` on
`u, a, b, c` with only the triangle `abc` balanced and a joint at `a`: `M = L(G, B)` is 3-connected
of rank four, while every canonical matrix particular to `(Ĝ, B̂)` represents `F(G, B) ≠ M`. -/
theorem theorem5_5 {M : Matroid E} (hM : IsNConnected M 3) (hr : 2 < M.eRank)
    {A : Matrix R E F} (hA : IsMatrixRepresentation M A) (hΩ : Ω.Represents M) :
    (Ω.IsProperlyUnbalanced → Xor (Ω.IsLiftParticular A) (Ω.IsFrameParticular A)) ∧
    (Ω.IsAlmostBalanced → Ω.IsFrameMatroid M →
      (∃ u, Ω.IsRollable u ∧ ∀ Ω', Ω.IsUnrolling u Ω' → Ω'.IsLiftParticular A ∧
        ((∃ a : Fˣ, a ≠ 1) → ∀ U ∈ Ω'.unbalancingClasses u, (Ω'.rollUp u U).IsFrameParticular A)) ∨
      ((∀ C, Ω.IsUnbalancedCycle C → C.encard = 1) ∧ Ω.unrollNew.IsLiftParticular A ∧
        ((∃ a : Fˣ, a ≠ 1) → Ω.IsFrameParticular A))) := by
  sorry

/-- **Theorem 5.6.** "Let `M` be a matroid represented by a 2-connected, properly unbalanced biased
graph `(G, B)`. Let `𝔽` be a field and let `A` be a matrix over `𝔽` representing `M`. Exactly one of
the following holds: (i) `A` is projectively equivalent to a canonical lift matrix particular to
`(G, B)`, or (ii) `A` is projectively equivalent to a canonical frame matrix particular to
`(G, B)`." -/
theorem IsProperlyUnbalanced.xor_particular (h2 : Ω.graph.IsKConnected 2)
    (hΩ : Ω.IsProperlyUnbalanced) {M : Matroid E} (hM : Ω.Represents M) {A : Matrix R E F}
    (hA : IsMatrixRepresentation M A) : Xor (Ω.IsLiftParticular A) (Ω.IsFrameParticular A) := by
  sorry

/-- **Theorem 5.9.** "Let `M` be a matroid of rank greater than two represented by a 2-connected
almost-balanced biased graph `(G, B)` having no 2-separation with one side balanced. Let `𝔽` be a
field and let `A` be an `𝔽`-matrix representing `M`. Then (i) `A` is projectively equivalent to a
canonical lift matrix particular to `(Ĝ, B̂)`, and (ii) `A` is projectively equivalent to a
canonical frame matrix particular to each roll-up of `(Ĝ, B̂)`, unless `𝔽` is `GF(2)`. Furthermore,
`A` is projectively equivalent to a canonical frame matrix particular to `(Ĝ, B̂)` if and only if
whenever `φ` is an `𝔽⁺`-gain function for which `A` and `A_L(Ĝ, φ)` are projectively equivalent, `φ`
is switching equivalent to a gain function assigning `0` to `e` if and only if `e` is a link not
incident to `u` or `e` is a loop of `M`."

`M` is assumed to be the frame matroid `F(G, B)`, not only represented by `(G, B)`: the example
of Theorem 5.5 (`K₄` on `u, a, b, c` with only `abc` balanced and a joint at `a`) meets the
source's hypotheses with `M = L(G, B) ≠ F(G, B)`, while canonical matrices particular to `(Ĝ, B̂)`
represent `F(G, B)`. -/
theorem IsAlmostBalanced.particular {M : Matroid E} (hr : 2 < M.eRank)
    (h2 : Ω.graph.IsKConnected 2) (hΩ : Ω.IsAlmostBalanced)
    (hsep : ∀ A B, Ω.graph.IsSeparation 2 A B → ¬Ω.IsBalancedSet A) (hM : Ω.IsFrameMatroid M)
    {A : Matrix R E F} (hA : IsMatrixRepresentation M A) :
    (∃ u, Ω.IsRollable u ∧ ∀ Ω', Ω.IsUnrolling u Ω' → Ω'.IsLiftParticular A ∧
      ((∃ a : Fˣ, a ≠ 1) → ∀ U ∈ Ω'.unbalancingClasses u, (Ω'.rollUp u U).IsFrameParticular A) ∧
      (Ω'.IsFrameParticular A ↔ ∀ φ : E → F, ProjEquiv A (liftMatrix Ω'.graph φ) →
        Ω'.graph.IsDerivedZeroPattern u M φ)) ∨
    ((∀ C, Ω.IsUnbalancedCycle C → C.encard = 1) ∧ Ω.unrollNew.IsLiftParticular A ∧
      ((∃ a : Fˣ, a ≠ 1) → Ω.IsFrameParticular A) ∧
      (Ω.unrollNew.IsFrameParticular A ↔ ∀ φ : E → F, ProjEquiv A (liftMatrix Ω.unrollNewGraph φ) →
        Ω.unrollNewGraph.IsDerivedZeroPattern none M φ)) := by
  sorry

end BiasedGraph

open BiasedGraph

local instance instFactPrimeSevenMainCases : Fact (Nat.Prime 7) := ⟨by decide⟩

/-- The frame matrix of `fanGain₁` represents `U_{3,5}`. -/
theorem fan_isMatrixRepresentation :
    IsMatrixRepresentation (uniformMatroid (Fin 5) 3) (frameMatrix fanGraph fanGain₁) := by
  obtain ⟨M, hM, hA⟩ := Realizes.exists_isFrameMatroid_frameMatrix fanGain₁_realizes
  rwa [hM.unique fan_isFrameMatroid] at hA

/-- The hypotheses of Theorems 5.5 and 5.9 hold for `U_{3,5}`, the contrabalanced `fanGraph` and
the frame matrix of `fanGain₁`. -/
example : IsNConnected (uniformMatroid (Fin 5) 3) 3 ∧ 2 < (uniformMatroid (Fin 5) 3).eRank ∧
    fanGraph.contrabalanced.graph.IsKConnected 2 ∧ fanGraph.contrabalanced.IsAlmostBalanced ∧
    (∀ A B, fanGraph.contrabalanced.graph.IsSeparation 2 A B →
      ¬fanGraph.contrabalanced.IsBalancedSet A) ∧
    fanGraph.contrabalanced.Represents (uniformMatroid (Fin 5) 3) ∧
    fanGraph.contrabalanced.IsFrameMatroid (uniformMatroid (Fin 5) 3) ∧
    IsMatrixRepresentation (uniformMatroid (Fin 5) 3) (frameMatrix fanGraph fanGain₁) :=
  ⟨uniformMatroid_three_five_isNConnected,
    by rw [uniformMatroid_three_five_eRank]; exact_mod_cast (by omega : (2 : ℕ) < 3),
    fanGraph_isKConnected, fan_isAlmostBalanced, fun _ _ h => fan_not_isBalancedSet h,
    Or.inl fan_isFrameMatroid, fan_isFrameMatroid, fan_isMatrixRepresentation⟩

/-- The hypotheses of Theorem 5.6 hold for `2C₃` with the gains `twoC3Gain`. -/
example : (twoC3.gainBiased twoC3Gain).graph.IsKConnected 2 ∧
    (twoC3.gainBiased twoC3Gain).IsProperlyUnbalanced ∧
    ∃ M, (twoC3.gainBiased twoC3Gain).Represents M ∧
      IsMatrixRepresentation M (frameMatrix twoC3 twoC3Gain) := by
  obtain ⟨M, hM, hA⟩ := Realizes.exists_isFrameMatroid_frameMatrix
    (Ω := twoC3.gainBiased twoC3Gain) (φ := twoC3Gain) rfl
  exact ⟨twoC3_isKConnected, twoC3Gain_isProperlyUnbalanced, M, Or.inl hM, hA⟩

end Transformer.GainMatroid
