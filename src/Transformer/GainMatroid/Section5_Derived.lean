module

public import Transformer.GainMatroid.Section5_FanRollUp

/-!
# Derived gain functions and Lemma 5.2

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §5.1. Let `u` be a balancing vertex of `(Ĝ, B̂)`
and `(G_U, B_U)` the roll-up of a set `U` of links at `u`. An `𝔽ˣ`-gain function `φ` on `G_U` that
is the identity on every link not at `u` gives the `𝔽⁺`-gain function `φ⁺ = derivedAddGain` on `Ĝ`,
and an `𝔽⁺`-gain function `ψ` on `Ĝ` that is zero on every link not at `u` gives, for a unit
`a ≠ 1`, the `𝔽ˣ`-gain function `ψ^× = derivedMulGain` on `G_U`, where `U` is the set of links at
`u` on which `ψ` vanishes. Lemma 5.2 says that the canonical lift matrix with the row of `u`
deleted, `A_L^{-u}(Ĝ, φ⁺)`, and the canonical frame matrix `A_F(G_U, φ)` differ by scaling columns,
once the row of `u` of the frame matrix is read as the gains row `v₀ = none`; likewise for `ψ`.

The source fixes an orientation with every edge at `u` directed into `u` (for `φ⁺`) or out of `u`
(for `ψ^×`). Here the derived functions read the orientation of each edge instead: a link with head
`u` gets `-φ e` and `-ψ e`, a link with tail `u` gets `(φ e)⁻¹` and `(ψ e)⁻¹`, which is what
reversing the link to the source's orientation and back gives.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {F Γ : Type*}

namespace OrientedGraph

/-- `γ` is the identity on every link of `G` not incident to `u` (§5.1). A realization of a biased
graph with balancing vertex `u` that is normalized on a spanning tree of `G − u` has this form. -/
@[to_additive IsAddNormalizedOff
  /-- `γ` is zero on every link of `G` not incident to `u` (§5.1). -/]
def IsNormalizedOff [One Γ] (G : OrientedGraph V E) (u : V) (γ : E → Γ) : Prop :=
  ∀ e, G.tail e ≠ G.head e → G.tail e ≠ u → G.head e ≠ u → γ e = 1

variable [DecidableEq V] [Field F]

open Classical in
/-- `φ⁺` (§5.1): the `𝔽⁺`-gain function **derived from** an `𝔽ˣ`-gain function `φ` on the graph `G`
of a roll-up from `u`. It is `0` on each link not at `u`, on each loop away from `u` (the rolled-up
joints and the balanced loops) and on each balanced loop, `-1` on each joint at `u`, `-φ e` on a
link with head `u` and `(φ e)⁻¹` on a link with tail `u`. -/
noncomputable def derivedAddGain (G : OrientedGraph V E) (u : V) (φ : E → Fˣ) : E → F := fun e =>
  if G.tail e = G.head e then (if G.tail e = u ∧ φ e ≠ 1 then -1 else 0)
  else if G.head e = u then -(φ e : F)
  else if G.tail e = u then ((φ e)⁻¹ : Fˣ)
  else 0

open Classical in
/-- `ψ^×` (§5.1): the `𝔽ˣ`-gain function **derived from** an `𝔽⁺`-gain function `ψ` on `G = Ĝ`, for
the roll-up of `U` from `u` and a unit `a ≠ 1`. It is `a` on each edge of `U` and on each joint,
`1` on each balanced loop and each link not at `u`, `-ψ e` on a link with head `u` and `(ψ e)⁻¹` on
a link with tail `u`. -/
noncomputable def derivedMulGain (G : OrientedGraph V E) (u : V) (U : Set E) (ψ : E → F) (a : Fˣ) :
    E → Fˣ := fun e =>
  if e ∈ U then a
  else if G.tail e = G.head e then (if ψ e = 0 then 1 else a)
  else if h : ψ e = 0 then 1
  else if G.head e = u then Units.mk0 (-ψ e) (neg_ne_zero.2 h)
  else if G.tail e = u then (Units.mk0 (ψ e) h)⁻¹
  else 1

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable [DecidableEq V] [Field F] {Ω' : BiasedGraph V E} {u : V}

/-- **Lemma 5.2 (i).** "Let `φ` be a `T`-normalized `𝔽ˣ`-gain function realizing a roll-up
`(G_U, B_U)` of `(Ĝ, B̂)`. Then `A_L^{-u}(Ĝ, φ⁺)` is obtained from `A_F(G_U, φ)` by scaling
columns." Row `w ≠ u` of `A_L^{-u}` is its vertex row, and row `u` is the gains row `v₀`.

The source assumes that `(Ĝ, B̂)` is the unrolling of a 2-connected almost-balanced biased graph,
that `U` is an unbalancing class and that `φ` is normalized on a spanning tree `T` of `G − u`. Here
`u` is any balancing vertex of `Ω' = (Ĝ, B̂)`, `U` any set of links at `u`, and `φ` the identity on
the links not at `u`, which a `T`-normalized realization is, as every cycle of `G − u` is
balanced. -/
theorem IsBalancingVertex.liftMatrix_derivedAddGain (hu : Ω'.IsBalancingVertex u) {U : Set E}
    (hU : U ⊆ Ω'.linkStar u) {φ : E → Fˣ} (hφ : (Ω'.rollUp u U).Realizes φ)
    (hn : (Ω'.rollUp u U).graph.IsNormalizedOff u φ) :
    ∃ s : E → Fˣ, ∀ w e, liftMatrix Ω'.graph (derivedAddGain (Ω'.rollUp u U).graph u φ)
      (if w = u then none else some w) e = s e * frameMatrix (Ω'.rollUp u U).graph φ w e := by
  sorry

/-- **Lemma 5.2 (ii).** "Let `ψ` be a `T`-normalized `𝔽⁺`-gain function realizing `(Ĝ, B̂)`. Let `U`
be the (possibly empty) unbalancing class of `Σ(u)` for which `ψ(U) = {0}`. Then `A_F(G_U, ψ^×)` is
obtained from `A_L^{-u}(Ĝ, ψ)` by scaling columns." Here `U` is the set of links at `u` on which `ψ`
vanishes, and `a ≠ 1` is the unit that `ψ^×` puts on the joints.

As in (i), `u` is any balancing vertex of `Ω' = (Ĝ, B̂)` and `ψ` vanishes on the links not at
`u`. -/
theorem IsBalancingVertex.frameMatrix_derivedMulGain (hu : Ω'.IsBalancingVertex u) {ψ : E → F}
    (hψ : Ω'.AddRealizes ψ) (hn : Ω'.graph.IsAddNormalizedOff u ψ) {a : Fˣ} (ha : a ≠ 1) :
    ∃ s : E → Fˣ, ∀ w e, frameMatrix (Ω'.rollUp u {e | e ∈ Ω'.linkStar u ∧ ψ e = 0}).graph
      (derivedMulGain Ω'.graph u {e | e ∈ Ω'.linkStar u ∧ ψ e = 0} ψ a) w e =
        s e * liftMatrix Ω'.graph ψ (if w = u then none else some w) e := by
  sorry

end BiasedGraph

open BiasedGraph

local instance instFactPrimeSevenDerived : Fact (Nat.Prime 7) := ⟨by decide⟩

/-- The hypotheses of Lemma 5.2 (i) hold for the roll-up of the contrabalanced `fanGraph` on `{1}`
with the gains `fanRollUpGain`. -/
example : fanGraph.contrabalanced.IsBalancingVertex 2 ∧
    ({1} : Set (Fin 5)) ⊆ fanGraph.contrabalanced.linkStar 2 ∧
    (fanGraph.contrabalanced.rollUp 2 {1}).Realizes fanRollUpGain ∧
    (fanGraph.contrabalanced.rollUp 2 {1}).graph.IsNormalizedOff 2 fanRollUpGain := by
  refine ⟨fan_isBalancingVertex, ?_, fanRollUpGain_realizes, ?_⟩
  · rintro e rfl
    exact fan_mem_sigmaStar (by decide) |>.elim id fun h => absurd h.1 (fan_not_isJoint _)
  · show (fanGraph.rollUpGraph 2 {1}).IsNormalizedOff 2 fanRollUpGain
    rw [fan_rollUpGraph]
    simp only [IsNormalizedOff, fanRollUpGraph, fanRollUpGain]
    decide

/-- The hypotheses of Lemma 5.2 (ii) hold for the contrabalanced `fanGraph` with the gains
`fanDerived₁` and the unit `2` of `𝔽₇`. -/
example : fanGraph.contrabalanced.IsBalancingVertex 2 ∧
    fanGraph.contrabalanced.AddRealizes fanDerived₁ ∧
    fanGraph.IsAddNormalizedOff 2 fanDerived₁ ∧ fanTwo ≠ 1 := by
  refine ⟨fan_isBalancingVertex, fanDerived₁_addRealizes, ?_, by decide⟩
  simp only [IsAddNormalizedOff, fanGraph, fanDerived₁]
  decide

end Transformer.GainMatroid
