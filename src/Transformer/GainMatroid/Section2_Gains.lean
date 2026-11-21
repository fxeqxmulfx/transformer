module

public import Transformer.GainMatroid.Section1_Canonical

/-!
# Switching and scaling of gain functions

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, paragraph *Switching and scaling*. A
`Γ`-gain function is a function `E → Γ` read in the reference orientation; the source's condition
`γ(e⁻¹) = γ(e)⁻¹` on links is what fixes the gain of the reverse traversal.

`switch G η γ` is `γ^η` for a multiplicative group and `addSwitch G η γ` for an additive one.
`switch_switch` is the source's identity `(γ^η₁)^η₂ = γ^(η₁η₂)`, and `switchEquiv_equivalence` and
`switchScaleEquiv_equivalence` prove that its switching classes and switching-and-scaling classes
are the classes of equivalence relations.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E Γ F : Type*}

/-- The **switching** `γ^η` (§2.3): `γ^η(e) = η(tail e)⁻¹ · γ(e) · η(head e)`. -/
@[to_additive addSwitch
  /-- The additive **switching** `γ^η` (§2.3): `γ^η(e) = -η(tail e) + γ(e) + η(head e)`. -/]
def switch [Group Γ] (G : OrientedGraph V E) (η : V → Γ) (γ : E → Γ) : E → Γ :=
  fun e => (η (G.tail e))⁻¹ * γ e * η (G.head e)

@[to_additive addSwitch_apply]
theorem switch_apply [Group Γ] (G : OrientedGraph V E) (η : V → Γ) (γ : E → Γ) (e : E) :
    switch G η γ e = (η (G.tail e))⁻¹ * γ e * η (G.head e) := rfl

@[to_additive addSwitch_zero]
theorem switch_one [Group Γ] (G : OrientedGraph V E) (γ : E → Γ) : switch G 1 γ = γ := by
  funext e
  simp [switch]

/-- The source's identity `(γ^η₁)^η₂ = γ^(η₁η₂)` (§2.3). -/
@[to_additive addSwitch_addSwitch
  /-- The source's identity `(γ^η₁)^η₂ = γ^(η₁+η₂)` (§2.3). -/]
theorem switch_switch [Group Γ] (G : OrientedGraph V E) (η₁ η₂ : V → Γ) (γ : E → Γ) :
    switch G η₂ (switch G η₁ γ) = switch G (η₁ * η₂) γ := by
  funext e
  simp [switch, mul_assoc]

/-- `φ` and `ψ` are **switching equivalent** (§2.3): `φ^η = ψ` for a switching function `η`. -/
@[to_additive AddSwitchEquiv
  /-- `φ` and `ψ` are additively **switching equivalent** (§2.3). -/]
def SwitchEquiv [Group Γ] (G : OrientedGraph V E) (φ ψ : E → Γ) : Prop :=
  ∃ η : V → Γ, switch G η φ = ψ

/-- Switching equivalence is an equivalence relation, so it has switching classes (§2.3). -/
@[to_additive addSwitchEquiv_equivalence
  /-- Additive switching equivalence is an equivalence relation (§2.3). -/]
theorem switchEquiv_equivalence [Group Γ] (G : OrientedGraph V E) :
    Equivalence (SwitchEquiv (Γ := Γ) G) where
  refl γ := ⟨1, switch_one G γ⟩
  symm := by
    rintro φ ψ ⟨η, rfl⟩
    exact ⟨η⁻¹, by rw [switch_switch, mul_inv_cancel, switch_one]⟩
  trans := by
    rintro φ ψ χ ⟨η₁, rfl⟩ ⟨η₂, rfl⟩
    exact ⟨η₁ * η₂, (switch_switch G η₁ η₂ φ).symm⟩

/-- `φ` and `ψ` are **switching-and-scaling equivalent** (§2.3): `a · φ^η = ψ` for a switching
function `η` and a nonzero scalar `a`. -/
def SwitchScaleEquiv [Field F] (G : OrientedGraph V E) (φ ψ : E → F) : Prop :=
  ∃ (η : V → F) (a : Fˣ), (a : F) • addSwitch G η φ = ψ

/-- Switching-and-scaling equivalence is an equivalence relation (§2.3). -/
theorem switchScaleEquiv_equivalence [Field F] (G : OrientedGraph V E) :
    Equivalence (SwitchScaleEquiv (F := F) G) where
  refl γ := ⟨0, 1, by rw [addSwitch_zero, Units.val_one, one_smul]⟩
  symm := by
    rintro φ ψ ⟨η, a, rfl⟩
    refine ⟨fun v => -((a : F) * η v), a⁻¹, ?_⟩
    funext e
    have ha : (a : F) ≠ 0 := a.ne_zero
    simp only [Pi.smul_apply, addSwitch_apply, smul_eq_mul, Units.val_inv_eq_inv_val]
    field_simp
    ring
  trans := by
    rintro φ ψ χ ⟨η, a, rfl⟩ ⟨ζ, b, rfl⟩
    refine ⟨fun v => η v + ((a⁻¹ : Fˣ) : F) * ζ v, b * a, ?_⟩
    funext e
    have ha : (a : F) ≠ 0 := a.ne_zero
    simp only [Pi.smul_apply, addSwitch_apply, smul_eq_mul, Units.val_mul,
      Units.val_inv_eq_inv_val]
    field_simp
    ring

end Transformer.GainMatroid
