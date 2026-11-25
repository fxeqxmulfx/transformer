module

public import Transformer.GainMatroid.Section3_Unavoidable
public import Transformer.GainMatroid.Section4_U24

/-!
# Confining inequivalence to a small minor

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2 and §3.3. Two realizations of a 2-connected,
loopless, properly unbalanced biased graph over an abelian group that are not switching equivalent
remain inequivalent on a small minor (Theorem 6): either on a link minor in `𝒢₀`
(`IsG0Localized`), or on the 2-cycle of a minor `𝖴₂` and on the theta of a link minor `𝖴₃`
(`IsU2U3Localized`). The minor `𝖴₂` may need "a single frame-type contraction of a joint", so it is
a `FrameMinorMap`. Over the additive group of a field the source assumes and concludes that the
gain functions are not switching-and-scaling equivalent (`addInequivalence_localized`). For a
tangled biased graph the first outcome holds (Corollary 3.6).

`𝖴₂` and `𝖴₃` are `U2` and `U3` with every cycle unbalanced; the 2-cycle of `U2` is `{2, 3}`
and the theta of `U3` is `{1, 2, 3}`. A gain function induced on a minor is defined up to
switching, so the outcomes ask for some induced gain functions.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

universe uV uE

variable {V : Type uV} {E : Type uE} {Ω : BiasedGraph V E} {Γ F : Type*}

/-- Outcome (i) of Theorem 6: gain functions induced by `φ` and `ψ` on a link minor in `𝒢₀` that
are not switching equivalent. -/
def IsG0Localized [Group Γ] (Ω : BiasedGraph V E) (φ ψ : E → Γ) : Prop :=
  (∃ S, IsG0TwoC3 S ∧ ∃ (m : Ω.LinkMinorMap S) (ι κ : Fin 6 → Γ), m.IsInducedGain φ ι ∧
      m.IsInducedGain ψ κ ∧ ¬SwitchEquiv S.graph ι κ) ∨
    (∃ S, IsG0Tube S ∧ ∃ (m : Ω.LinkMinorMap S) (ι κ : Fin 6 → Γ), m.IsInducedGain φ ι ∧
      m.IsInducedGain ψ κ ∧ ¬SwitchEquiv S.graph ι κ) ∨
    ∃ S, IsG0K4 S ∧ ∃ (m : Ω.LinkMinorMap S) (ι κ : Fin 6 → Γ), m.IsInducedGain φ ι ∧
      m.IsInducedGain ψ κ ∧ ¬SwitchEquiv S.graph ι κ

/-- Outcome (ii) of Theorem 6: gain functions induced on a minor `𝖴₂` that are not switching
equivalent on its 2-cycle, and on a link minor `𝖴₃` that are not on its theta. -/
def IsU2U3Localized [Group Γ] (Ω : BiasedGraph V E) (φ ψ : E → Γ) : Prop :=
  (∃ (m : Ω.FrameMinorMap U2.contrabalanced) (ι κ : Fin 4 → Γ), m.IsInducedGain φ ι ∧
      m.IsInducedGain ψ κ ∧ ¬SwitchEquiv (U2.delete {0, 1}) (fun e => ι e) (fun e => κ e)) ∧
    ∃ (m : Ω.LinkMinorMap U3.contrabalanced) (ι κ : Fin 4 → Γ), m.IsInducedGain φ ι ∧
      m.IsInducedGain ψ κ ∧ ¬SwitchEquiv (U3.delete {0}) (fun e => ι e) (fun e => κ e)

/-- Outcome (i) of Theorem 6 over `F⁺`: not switching-and-scaling equivalent. -/
def IsG0AddLocalized [Field F] (Ω : BiasedGraph V E) (φ ψ : E → F) : Prop :=
  (∃ S, IsG0TwoC3 S ∧ ∃ (m : Ω.LinkMinorMap S) (ι κ : Fin 6 → F), m.IsAddInducedGain φ ι ∧
      m.IsAddInducedGain ψ κ ∧ ¬SwitchScaleEquiv S.graph ι κ) ∨
    (∃ S, IsG0Tube S ∧ ∃ (m : Ω.LinkMinorMap S) (ι κ : Fin 6 → F), m.IsAddInducedGain φ ι ∧
      m.IsAddInducedGain ψ κ ∧ ¬SwitchScaleEquiv S.graph ι κ) ∨
    ∃ S, IsG0K4 S ∧ ∃ (m : Ω.LinkMinorMap S) (ι κ : Fin 6 → F), m.IsAddInducedGain φ ι ∧
      m.IsAddInducedGain ψ κ ∧ ¬SwitchScaleEquiv S.graph ι κ

/-- Outcome (ii) of Theorem 6 over `F⁺`: not switching-and-scaling equivalent. -/
def IsU2U3AddLocalized [Field F] (Ω : BiasedGraph V E) (φ ψ : E → F) : Prop :=
  (∃ (m : Ω.FrameMinorMap U2.contrabalanced) (ι κ : Fin 4 → F), m.IsAddInducedGain φ ι ∧
      m.IsAddInducedGain ψ κ ∧
      ¬SwitchScaleEquiv (U2.delete {0, 1}) (fun e => ι e) (fun e => κ e)) ∧
    ∃ (m : Ω.LinkMinorMap U3.contrabalanced) (ι κ : Fin 4 → F), m.IsAddInducedGain φ ι ∧
      m.IsAddInducedGain ψ κ ∧ ¬SwitchScaleEquiv (U3.delete {0}) (fun e => ι e) (fun e => κ e)

variable [Finite V] [Finite E] [DecidableEq V]

/-- **Theorem 6**, over an abelian group. "Let `(G, B)` be a 2-connected, loopless, and properly
unbalanced biased graph. Let `Γ` be an abelian group and let `φ` and `ψ` be `Γ`-realizations of
`(G, B)`. Assume that `φ` and `ψ` are not switching equivalent. Then either (i) … or (ii) …". -/
theorem inequivalence_localized [CommGroup Γ] (h2 : Ω.graph.IsKConnected 2)
    (hG : Ω.graph.Loopless) (hΩ : Ω.IsProperlyUnbalanced) {φ ψ : E → Γ} (hφ : Ω.Realizes φ)
    (hψ : Ω.Realizes ψ) (hne : ¬SwitchEquiv Ω.graph φ ψ) :
    Ω.IsG0Localized φ ψ ∨ Ω.IsU2U3Localized φ ψ := by
  sorry

/-- **Theorem 6**, over the additive group of a field: "In the case that `Γ` is the additive group
of a field, assume that `φ` and `ψ` are not switching-and-scaling equivalent", and conclude so on
the minors. -/
theorem addInequivalence_localized [Field F] (h2 : Ω.graph.IsKConnected 2)
    (hG : Ω.graph.Loopless) (hΩ : Ω.IsProperlyUnbalanced) {φ ψ : E → F} (hφ : Ω.AddRealizes φ)
    (hψ : Ω.AddRealizes ψ) (hne : ¬SwitchScaleEquiv Ω.graph φ ψ) :
    Ω.IsG0AddLocalized φ ψ ∨ Ω.IsU2U3AddLocalized φ ψ := by
  sorry

/-- **Corollary 3.6.** "If `(G, B)` is tangled, then the first outcome of Theorem 6 holds." -/
theorem IsTangled.isG0Localized [CommGroup Γ] (hT : Ω.IsTangled) (h2 : Ω.graph.IsKConnected 2)
    (hG : Ω.graph.Loopless) {φ ψ : E → Γ} (hφ : Ω.Realizes φ) (hψ : Ω.Realizes ψ)
    (hne : ¬SwitchEquiv Ω.graph φ ψ) : Ω.IsG0Localized φ ψ := by
  sorry

/-- **Corollary 3.6** over the additive group of a field. -/
theorem IsTangled.isG0AddLocalized [Field F] (hT : Ω.IsTangled) (h2 : Ω.graph.IsKConnected 2)
    (hG : Ω.graph.Loopless) {φ ψ : E → F} (hφ : Ω.AddRealizes φ) (hψ : Ω.AddRealizes ψ)
    (hne : ¬SwitchScaleEquiv Ω.graph φ ψ) : Ω.IsG0AddLocalized φ ψ := by
  sorry

end BiasedGraph

open BiasedGraph

local instance : Fact (Nat.Prime 7) := ⟨by decide⟩

/-- The contrabalanced `2C₃` is tangled. -/
theorem twoC3_contrabalanced_isTangled : twoC3.contrabalanced.IsTangled :=
  ⟨(isProperlyUnbalanced_iff_of_twoC3 (IsReorientationOf.refl _)).2 fun _ h => absurd h
      (Set.notMem_empty _),
    fun _ _ hC hD => not_disjoint_verts (by simp [Set.encard_univ]) twoC3_loopless hC.1 hD.1⟩

/-- The hypotheses of Theorem 6 and Corollary 3.6 hold on the contrabalanced `2C₃` with the gains
`twoC3T0Gain₁` and `twoC3T0Gain₂` over `𝔽₇ˣ`. -/
example : twoC3.contrabalanced.IsTangled ∧ twoC3.IsKConnected 2 ∧ twoC3.Loopless ∧
    twoC3.contrabalanced.IsProperlyUnbalanced ∧ twoC3.contrabalanced.Realizes twoC3T0Gain₁ ∧
    twoC3.contrabalanced.Realizes twoC3T0Gain₂ ∧ ¬SwitchEquiv twoC3 twoC3T0Gain₁ twoC3T0Gain₂ :=
  ⟨twoC3_contrabalanced_isTangled, twoC3_isKConnected, twoC3_loopless,
    twoC3_contrabalanced_isTangled.1, twoC3T0Gain₁_realizes, twoC3T0Gain₂_realizes,
    twoC3T0Gain_not_switchEquiv⟩

/-- The hypotheses of the additive Theorem 6 and Corollary 3.6 hold on the contrabalanced `2C₃`
with the gains `twoC3T0AddGain₁` and `twoC3T0AddGain₂` over `𝔽₇⁺`. -/
example : twoC3.contrabalanced.AddRealizes twoC3T0AddGain₁ ∧
    twoC3.contrabalanced.AddRealizes twoC3T0AddGain₂ ∧
    ¬SwitchScaleEquiv twoC3 twoC3T0AddGain₁ twoC3T0AddGain₂ :=
  ⟨twoC3T0AddGain₁_addRealizes, twoC3T0AddGain₂_addRealizes,
    twoC3T0AddGain_not_switchScaleEquiv⟩

end Transformer.GainMatroid
