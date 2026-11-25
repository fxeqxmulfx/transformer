module

public import Transformer.GainMatroid.Section5_D10
public import Transformer.GainMatroid.Section5_Derived

/-!
# Corollary 5.3

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §5.1. Let `(G, B)` be 2-connected and almost
balanced with unique balancing vertex `u` after deleting its joints, `(Ĝ, B̂)` its unrolling to `u`,
and `𝔽` a field other than `GF(2)`. A switching of an `𝔽⁺`-realization of `(Ĝ, B̂)` derives an
`𝔽ˣ`-realization of each roll-up with projectively equivalent canonical matrix (i), and a switching
of an `𝔽ˣ`-realization of a roll-up derives an `𝔽⁺`-realization of `(Ĝ, B̂)` (ii).

Part (iii) fails as printed (`corollary5_3_iii_fails`): it asks for an `𝔽⁺`-realization nonzero on
`Σ(u)`, and switching at a vertex other than `u` makes such a realization of `d10` over `𝔽₃`, which
has three unbalancing classes and so no `𝔽₃ˣ`-realization. The source derives (iii) from a
realization that vanishes on the links not at `u`, and with that requirement it holds
(`IsUnrolling.exists_realizes_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {F : Type*}

open OrientedGraph

namespace BiasedGraph

variable [Finite V] [Finite E] [DecidableEq V] [Field F] {Ω Ω' : BiasedGraph V E} {u : V}

/-- **Corollary 5.3 (i).** "For every `𝔽⁺`-gain function `φ` realizing `(Ĝ, B̂)`, and every
unbalancing class `U ⊆ Σ(u)`, there is a derived `𝔽ˣ`-gain function `φ^×` realizing the roll-up
`(G_U, B_U)` of `(Ĝ, B̂)` such that `A_L(Ĝ, φ)` and `A_F(G_U, φ^×)` are projectively equivalent."
The derived function is taken from a switching of `φ` vanishing on the links not at `u` and on `U`,
for any unit `a ≠ 1`. -/
theorem IsUnrolling.exists_derivedMulGain (h2 : Ω.graph.IsKConnected 2) (hu : Ω.IsRollable u)
    (hΩ' : Ω.IsUnrolling u Ω') {φ : E → F} (hφ : Ω'.AddRealizes φ) {U : Set E}
    (hU : U ∈ Ω'.unbalancingClasses u) {a : Fˣ} (ha : a ≠ 1) :
    ∃ η : V → F, Ω'.graph.IsAddNormalizedOff u (addSwitch Ω'.graph η φ) ∧
      (∀ e ∈ U, addSwitch Ω'.graph η φ e = 0) ∧
      (Ω'.rollUp u U).Realizes (derivedMulGain Ω'.graph u U (addSwitch Ω'.graph η φ) a) ∧
      ProjEquiv (liftMatrix Ω'.graph φ) (frameMatrix (Ω'.rollUp u U).graph
        (derivedMulGain Ω'.graph u U (addSwitch Ω'.graph η φ) a)) := by
  sorry

/-- **Corollary 5.3 (ii).** "For every `𝔽ˣ`-gain function `ψ` realizing a roll-up `(G_U, B_U)` of
`(Ĝ, B̂)`, there is a derived `𝔽⁺`-gain function `ψ⁺` realizing `(Ĝ, B̂)` for which `A_F(G_U, ψ)`
and `A_L(Ĝ, ψ⁺)` are projectively equivalent." The derived function is taken from a switching of `ψ`
that is the identity on the links not at `u`. -/
theorem IsUnrolling.exists_derivedAddGain (h2 : Ω.graph.IsKConnected 2) (hu : Ω.IsRollable u)
    (hΩ' : Ω.IsUnrolling u Ω') {U : Set E} (hU : U ∈ Ω'.unbalancingClasses u) {ψ : E → Fˣ}
    (hψ : (Ω'.rollUp u U).Realizes ψ) :
    ∃ η : V → Fˣ, (Ω'.rollUp u U).graph.IsNormalizedOff u (switch (Ω'.rollUp u U).graph η ψ) ∧
      Ω'.AddRealizes (derivedAddGain (Ω'.rollUp u U).graph u (switch (Ω'.rollUp u U).graph η ψ)) ∧
      ProjEquiv (frameMatrix (Ω'.rollUp u U).graph ψ)
        (liftMatrix Ω'.graph (derivedAddGain (Ω'.rollUp u U).graph u
          (switch (Ω'.rollUp u U).graph η ψ))) := by
  sorry

/-- **Corollary 5.3 (iii)**, corrected. The source: "There is an `𝔽ˣ`-gain function realizing
`(Ĝ, B̂)` if and only if there is an `𝔽⁺`-gain function `φ` realizing `(Ĝ, B̂)` for which `φ(e) ≠ 0`
for each edge in `Σ(u)`." That fails (`corollary5_3_iii_fails`); here `φ` also vanishes on each
link not at `u`, as the derived gain functions of the source's proof do. -/
theorem IsUnrolling.exists_realizes_iff (h2 : Ω.graph.IsKConnected 2) (hu : Ω.IsRollable u)
    (hΩ' : Ω.IsUnrolling u Ω') (hF : ∃ a : Fˣ, a ≠ 1) :
    (∃ ψ : E → Fˣ, Ω'.Realizes ψ) ↔ ∃ φ : E → F, Ω'.AddRealizes φ ∧
      Ω'.graph.IsAddNormalizedOff u φ ∧ ∀ e ∈ Ω'.sigmaStar u, φ e ≠ 0 := by
  sorry

end BiasedGraph

open BiasedGraph

/-- **Corollary 5.3 (iii) fails as printed.** For `(G, B) = d10`, which is 2-connected, almost
balanced with unique balancing vertex `0` and its own unrolling, and `𝔽 = 𝔽₃`, some
`𝔽₃⁺`-realization is nonzero on `Σ(0)`, while no `𝔽₃ˣ`-realization exists: the three edges at `0`
lie in three unbalancing classes, which need three distinct gains, and `𝔽₃ˣ` has two elements. -/
theorem corollary5_3_iii_fails :
    d10.graph.IsKConnected 2 ∧ d10.IsRollable 0 ∧ d10.IsUnrolling 0 d10 ∧
      ¬((∃ ψ : Fin 6 → (ZMod 3)ˣ, d10.Realizes ψ) ↔
        ∃ φ : Fin 6 → ZMod 3, d10.AddRealizes φ ∧ ∀ e ∈ d10.sigmaStar 0, φ e ≠ 0) := by
  sorry

local instance instFactPrimeSevenDerivedCorollary : Fact (Nat.Prime 7) := ⟨by decide⟩

/-- The hypotheses of Corollary 5.3 (i) hold for the contrabalanced `fanGraph`, its derived gains
`fanDerived₁`, the class `{1}` and the unit `2` of `𝔽₇`. -/
example : fanGraph.contrabalanced.graph.IsKConnected 2 ∧ fanGraph.contrabalanced.IsRollable 2 ∧
    fanGraph.contrabalanced.IsUnrolling 2 fanGraph.contrabalanced ∧
    fanGraph.contrabalanced.AddRealizes fanDerived₁ ∧
    ({1} : Set (Fin 5)) ∈ fanGraph.contrabalanced.unbalancingClasses 2 ∧ fanTwo ≠ 1 :=
  ⟨fanGraph_isKConnected, fan_isRollable, fan_isUnrolling, fanDerived₁_addRealizes,
    fan_mem_unbalancingClasses, by decide⟩

/-- The hypotheses of Corollary 5.3 (ii) and (iii) hold for the contrabalanced `fanGraph`, the
class `{1}`, the gains `fanRollUpGain` of the roll-up and the field `𝔽₇`. -/
example : (fanGraph.contrabalanced.rollUp 2 {1}).Realizes fanRollUpGain ∧
    ∃ a : (ZMod 7)ˣ, a ≠ 1 :=
  ⟨fanRollUpGain_realizes, fanTwo, by decide⟩

end Transformer.GainMatroid
