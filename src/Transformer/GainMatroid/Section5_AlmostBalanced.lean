module

public import Transformer.GainMatroid.Section5_DerivedCorollary

/-!
# Theorem 5.4: almost-balanced biased graphs

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §5.1. Let `(G, B)` be 2-connected and almost
balanced, with unique balancing vertex `u` after deleting its joints, no joint at `u`, and no
vertical 2-separation with one side balanced; let `(Ĝ, B̂)` be its unrolling to `u`. Canonical lift
matrices of two `𝔽⁺`-realizations of `(Ĝ, B̂)` are projectively equivalent exactly when the gain
functions are switching-and-scaling equivalent (i). Canonical frame matrices of realizations of
roll-ups, and a canonical lift matrix against a canonical frame matrix, are compared through derived
gain functions (ii)–(iv).

A derived gain function is taken from a switching that is the identity on the links not at `u`;
the statements hold for every such switching. Without a joint at `u`, the derived functions of two
such switchings are switching-and-scaling equivalent, which the source uses implicitly.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {F : Type*}

open OrientedGraph

namespace BiasedGraph

variable [Finite V] [Finite E] [DecidableEq V] [Field F] {Ω Ω' : BiasedGraph V E} {u : V}

/-- **Theorem 5.4 (i).** "The canonical lift matrices given by two `𝔽⁺`-gain functions `φ` and `ψ`
realizing `(Ĝ, B̂)` are projectively equivalent if and only if `φ` and `ψ` are switching-and-scaling
equivalent." -/
theorem IsUnrolling.lift_projEquiv_iff (h2 : Ω.graph.IsKConnected 2) (hu : Ω.IsRollable u)
    (hJ : ∀ e, Ω.IsJoint e → Ω.graph.tail e ≠ u)
    (hsep : ∀ A B, Ω.graph.IsVerticalSeparation 2 A B → ¬Ω.IsBalancedSet A)
    (hΩ' : Ω.IsUnrolling u Ω') {φ ψ : E → F} (hφ : Ω'.AddRealizes φ) (hψ : Ω'.AddRealizes ψ) :
    ProjEquiv (liftMatrix Ω'.graph φ) (liftMatrix Ω'.graph ψ) ↔ SwitchScaleEquiv Ω'.graph φ ψ := by
  sorry

/-- **Theorem 5.4 (ii).** "Let `U` and `W` be unbalancing classes of `Σ(u)`. Let `φ` and `ψ` be
`𝔽ˣ`-gain functions realizing `(G_U, B_U)` and `(G_W, B_W)`, respectively. The canonical frame
matrices given by `φ` and `ψ` are projectively equivalent if and only if their derived gain
functions `φ⁺` and `ψ⁺` are switching-and-scaling equivalent." -/
theorem IsUnrolling.frame_projEquiv_iff (h2 : Ω.graph.IsKConnected 2) (hu : Ω.IsRollable u)
    (hJ : ∀ e, Ω.IsJoint e → Ω.graph.tail e ≠ u)
    (hsep : ∀ A B, Ω.graph.IsVerticalSeparation 2 A B → ¬Ω.IsBalancedSet A)
    (hΩ' : Ω.IsUnrolling u Ω') {U W : Set E} (hU : U ∈ Ω'.unbalancingClasses u)
    (hW : W ∈ Ω'.unbalancingClasses u) {φ ψ : E → Fˣ} (hφ : (Ω'.rollUp u U).Realizes φ)
    (hψ : (Ω'.rollUp u W).Realizes ψ) {η η' : V → Fˣ}
    (hη : (Ω'.rollUp u U).graph.IsNormalizedOff u (switch (Ω'.rollUp u U).graph η φ))
    (hη' : (Ω'.rollUp u W).graph.IsNormalizedOff u (switch (Ω'.rollUp u W).graph η' ψ)) :
    ProjEquiv (frameMatrix (Ω'.rollUp u U).graph φ) (frameMatrix (Ω'.rollUp u W).graph ψ) ↔
      SwitchScaleEquiv Ω'.graph
        (derivedAddGain (Ω'.rollUp u U).graph u (switch (Ω'.rollUp u U).graph η φ))
        (derivedAddGain (Ω'.rollUp u W).graph u (switch (Ω'.rollUp u W).graph η' ψ)) := by
  sorry

/-- **Theorem 5.4 (iii).** "Let `U` be an unbalancing class of `Σ(u)`. The canonical lift matrix
given by an `𝔽⁺`-gain function `φ` realizing `(Ĝ, B̂)` and the canonical frame matrix given by an
`𝔽ˣ`-gain function `ψ` realizing `(G_U, B_U)` are projectively equivalent if and only if `φ` and the
derived gain function `ψ⁺` are switching-and-scaling equivalent." -/
theorem IsUnrolling.lift_frame_projEquiv_iff (h2 : Ω.graph.IsKConnected 2) (hu : Ω.IsRollable u)
    (hJ : ∀ e, Ω.IsJoint e → Ω.graph.tail e ≠ u)
    (hsep : ∀ A B, Ω.graph.IsVerticalSeparation 2 A B → ¬Ω.IsBalancedSet A)
    (hΩ' : Ω.IsUnrolling u Ω') {U : Set E} (hU : U ∈ Ω'.unbalancingClasses u) {φ : E → F}
    {ψ : E → Fˣ} (hφ : Ω'.AddRealizes φ) (hψ : (Ω'.rollUp u U).Realizes ψ) {η : V → Fˣ}
    (hη : (Ω'.rollUp u U).graph.IsNormalizedOff u (switch (Ω'.rollUp u U).graph η ψ)) :
    ProjEquiv (liftMatrix Ω'.graph φ) (frameMatrix (Ω'.rollUp u U).graph ψ) ↔
      SwitchScaleEquiv Ω'.graph φ
        (derivedAddGain (Ω'.rollUp u U).graph u (switch (Ω'.rollUp u U).graph η ψ)) := by
  sorry

/-- **Theorem 5.4 (iv).** "Let `φ` and `ψ` be `𝔽⁺`- and `𝔽ˣ`-gain functions, respectively, realizing
`(Ĝ, B̂)`. The canonical lift matrix given by `φ` and the canonical frame matrix given by `ψ` are
projectively equivalent if and only if `φ` and the derived gain function `ψ⁺` are
switching-and-scaling equivalent." -/
theorem IsUnrolling.lift_frame_projEquiv_iff_self (h2 : Ω.graph.IsKConnected 2)
    (hu : Ω.IsRollable u) (hJ : ∀ e, Ω.IsJoint e → Ω.graph.tail e ≠ u)
    (hsep : ∀ A B, Ω.graph.IsVerticalSeparation 2 A B → ¬Ω.IsBalancedSet A)
    (hΩ' : Ω.IsUnrolling u Ω') {φ : E → F} {ψ : E → Fˣ} (hφ : Ω'.AddRealizes φ)
    (hψ : Ω'.Realizes ψ) {η : V → Fˣ} (hη : Ω'.graph.IsNormalizedOff u (switch Ω'.graph η ψ)) :
    ProjEquiv (liftMatrix Ω'.graph φ) (frameMatrix Ω'.graph ψ) ↔
      SwitchScaleEquiv Ω'.graph φ (derivedAddGain Ω'.graph u (switch Ω'.graph η ψ)) := by
  sorry

end BiasedGraph

open BiasedGraph

local instance instFactPrimeSevenAlmostBalanced : Fact (Nat.Prime 7) := ⟨by decide⟩

/-- The common hypotheses of Theorem 5.4 hold for the contrabalanced `fanGraph` and `u = 2`: it has
no joint, and on three vertices no vertical 2-separation. -/
theorem fan_theorem5_4_hypotheses :
    fanGraph.contrabalanced.graph.IsKConnected 2 ∧ fanGraph.contrabalanced.IsRollable 2 ∧
      (∀ e, fanGraph.contrabalanced.IsJoint e → fanGraph.contrabalanced.graph.tail e ≠ 2) ∧
      (∀ A B, fanGraph.contrabalanced.graph.IsVerticalSeparation 2 A B →
        ¬fanGraph.contrabalanced.IsBalancedSet A) ∧
      fanGraph.contrabalanced.IsUnrolling 2 fanGraph.contrabalanced :=
  ⟨fanGraph_isKConnected, fan_isRollable, fun e h => absurd h (fan_not_isJoint e),
    fun A B h => absurd h (not_isVerticalSeparation_two A B), fan_isUnrolling⟩

/-- The further hypotheses of Theorem 5.4 (i)–(iii) hold with the gains `fanDerived₁` on `fanGraph`
and `fanRollUpGain` on its roll-up on `{1}`, normalized by the trivial switching. -/
example : fanGraph.contrabalanced.AddRealizes fanDerived₁ ∧
    ({1} : Set (Fin 5)) ∈ fanGraph.contrabalanced.unbalancingClasses 2 ∧
    (fanGraph.contrabalanced.rollUp 2 {1}).Realizes fanRollUpGain ∧
    (fanGraph.contrabalanced.rollUp 2 {1}).graph.IsNormalizedOff 2
      (switch (fanGraph.contrabalanced.rollUp 2 {1}).graph 1 fanRollUpGain) := by
  refine ⟨fanDerived₁_addRealizes, fan_mem_unbalancingClasses, fanRollUpGain_realizes, ?_⟩
  rw [switch_one]
  show (fanGraph.rollUpGraph 2 {1}).IsNormalizedOff 2 fanRollUpGain
  rw [fan_rollUpGraph]
  simp only [IsNormalizedOff, fanRollUpGraph, fanRollUpGain]
  decide

/-- The further hypotheses of Theorem 5.4 (iv) hold with the gains `fanDerived₁` and `fanGain₁` on
`fanGraph`, the latter normalized by the trivial switching. -/
example : fanGraph.contrabalanced.AddRealizes fanDerived₁ ∧
    fanGraph.contrabalanced.Realizes fanGain₁ ∧
    fanGraph.contrabalanced.graph.IsNormalizedOff 2 (switch fanGraph 1 fanGain₁) := by
  refine ⟨fanDerived₁_addRealizes, fanGain₁_realizes, ?_⟩
  rw [switch_one]
  simp only [contrabalanced, IsNormalizedOff, fanGraph, fanGain₁]
  decide

end Transformer.GainMatroid
