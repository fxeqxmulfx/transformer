module

public import Transformer.GainMatroid.Section2_ExchangeEquivalence
public import Transformer.GainMatroid.Section2_ExchangeRealizes

/-!
# Realization classes across an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3 and §2.4. The `Γ`-realizations of a biased
graph `Ω` fall into switching classes (`RealizationClass`), and its `𝔽⁺`-realizations into
switching-and-scaling classes (`AddRealizationClass`).

Proposition 2.10 (ii) for an exchange pair `T`, `S` on `X = {a, b, c}` (`IsExchange`), with `X`
balanced in `Ω` and `Ω'` the biased graph on `S` with the balanced cycles `Δ_X B`: the classes of
`Ω` and of `Ω'` correspond, and the correspondence fixes the class of every gain function with
identity gains on `X` (`IsExchange.exists_realizationClass_equiv`,
`IsExchange.exists_addRealizationClass_equiv`). The source's proof takes such gain functions as
representatives:

* every realization of `Ω` switches to one with identity gains on `X`, as `X` is balanced, and every
  realization of `Ω'` does too (`IsExchange.exists_switch_eq_one`);
* such a gain function realizes `Ω` exactly when it realizes `Ω'` (`IsExchange.realizes_iff`);
* two of them are equivalent in `T` exactly when they are in `S` (`IsExchange.switchEquiv_iff`,
  `IsExchange.switchScaleEquiv_iff`).

A relation between two setoids with these properties gives a bijection of the quotients
(`exists_quotient_equiv`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

/-- A relation `R` between two setoids gives a bijection of their quotients that matches the classes
of related elements, when every class on either side has an element related to one on the other,
and related pairs are equivalent on one side exactly when they are on the other. -/
theorem exists_quotient_equiv {α β : Type*} (s : Setoid α) (t : Setoid β) (R : α → β → Prop)
    (hα : ∀ p, ∃ p' q, s.r p p' ∧ R p' q) (hβ : ∀ q, ∃ p q', t.r q q' ∧ R p q')
    (hR : ∀ {p₁ p₂ q₁ q₂}, R p₁ q₁ → R p₂ q₂ → (s.r p₁ p₂ ↔ t.r q₁ q₂)) :
    ∃ e : Quotient s ≃ Quotient t, ∀ p q, R p q → e (Quotient.mk s p) = Quotient.mk t q := by
  choose f' f hf' hf using hα
  choose g g' hg' hg using hβ
  have hfR : ∀ p q, R p q → t.r (f p) q := fun p _ hpq =>
    (hR (hf p) hpq).1 (s.iseqv.symm (hf' p))
  have hgR : ∀ p q, R p q → s.r (g q) p := fun _ q hpq =>
    (hR (hg q) hpq).2 (t.iseqv.symm (hg' q))
  refine ⟨⟨Quotient.lift (fun p => Quotient.mk t (f p)) fun p₁ p₂ hp => Quotient.sound ?_,
    Quotient.lift (fun q => Quotient.mk s (g q)) fun q₁ q₂ hq => Quotient.sound ?_,
    fun x => Quotient.inductionOn x fun p => Quotient.sound ?_,
    fun x => Quotient.inductionOn x fun q => Quotient.sound ?_⟩,
    fun p q hpq => Quotient.sound (hfR p q hpq)⟩
  · exact (hR (hf p₁) (hf p₂)).1
      (s.iseqv.trans (s.iseqv.trans (s.iseqv.symm (hf' p₁)) hp) (hf' p₂))
  · exact (hR (hg q₁) (hg q₂)).2
      (t.iseqv.trans (t.iseqv.trans (t.iseqv.symm (hg' q₁)) hq) (hg' q₂))
  · exact s.iseqv.trans (hgR (f' p) (f p) (hf p)) (s.iseqv.symm (hf' p))
  · exact t.iseqv.trans (hfR (g q) (g' q) (hg q)) (t.iseqv.symm (hg' q))

variable {V W E : Type*} [DecidableEq V]

/-- Switching equivalence on the `Γ`-realizations of `Ω` (§2.3). -/
def realizationSetoid (Ω : BiasedGraph V E) (Γ : Type*) [Group Γ] :
    Setoid {γ : E → Γ // Ω.Realizes γ} where
  r φ ψ := SwitchEquiv Ω.graph φ.1 ψ.1
  iseqv := ⟨fun φ => (switchEquiv_equivalence Ω.graph).refl φ.1,
    fun h => (switchEquiv_equivalence Ω.graph).symm h,
    fun h h' => (switchEquiv_equivalence Ω.graph).trans h h'⟩

/-- The switching classes of `Γ`-realizations of `Ω` (§2.3). -/
def RealizationClass (Ω : BiasedGraph V E) (Γ : Type*) [Group Γ] : Type _ :=
  Quotient (realizationSetoid Ω Γ)

/-- Switching-and-scaling equivalence on the `𝔽⁺`-realizations of `Ω` (§2.3). -/
def addRealizationSetoid (Ω : BiasedGraph V E) (F : Type*) [Field F] :
    Setoid {ψ : E → F // Ω.AddRealizes ψ} where
  r φ ψ := SwitchScaleEquiv Ω.graph φ.1 ψ.1
  iseqv := ⟨fun φ => (switchScaleEquiv_equivalence Ω.graph).refl φ.1,
    fun h => (switchScaleEquiv_equivalence Ω.graph).symm h,
    fun h h' => (switchScaleEquiv_equivalence Ω.graph).trans h h'⟩

/-- The switching-and-scaling classes of `𝔽⁺`-realizations of `Ω` (§2.3). -/
def AddRealizationClass (Ω : BiasedGraph V E) (F : Type*) [Field F] : Type _ :=
  Quotient (addRealizationSetoid Ω F)

namespace OrientedGraph.IsExchange

variable [DecidableEq W] {ι : V → W} {v : W} {a b c : E} {x y z : V} {Ω : BiasedGraph V E}
  {Ω' : BiasedGraph W E}

/-- §2.4, Proposition 2.10 (ii) for an exchange pair, multiplicative: if `X` is balanced in `Ω` and
`Ω'` has the balanced cycles `Δ_X B`, a bijection of switching classes fixes the class of every gain
function that is the identity on `X` and realizes both biased graphs. -/
theorem exists_realizationClass_equiv {Γ : Type*} [Group Γ]
    (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    (hΩ' : Ω'.balanced = deltaBalanced Ω.balanced {a, b, c}) :
    ∃ e : RealizationClass Ω Γ ≃ RealizationClass Ω' Γ,
      ∀ (φ : E → Γ) (hφ : Ω.Realizes φ) (hφ' : Ω'.Realizes φ), φ a = 1 → φ b = 1 → φ c = 1 →
        e (Quotient.mk _ ⟨φ, hφ⟩) = Quotient.mk _ ⟨φ, hφ'⟩ := by
  obtain ⟨e, he⟩ := exists_quotient_equiv (realizationSetoid Ω Γ) (realizationSetoid Ω' Γ)
    (fun φ ψ => φ.1 = ψ.1 ∧ ∀ g ∈ ({a, b, c} : Set E), φ.1 g = 1)
    (fun φ => by
      have hφ : Ω.graph.balancedCycles φ.1 = Ω.balanced := φ.2
      obtain ⟨-, η, hη⟩ := mem_balancedCycles_iff.1
        (by rw [hφ]; exact hbal : ({a, b, c} : Set E) ∈ Ω.graph.balancedCycles φ.1)
      exact ⟨⟨_, φ.2.switch η⟩, ⟨_, (h.realizes_iff hbal hΩ' hη).1 (φ.2.switch η)⟩, ⟨η, rfl⟩,
        rfl, hη⟩)
    (fun ψ => by
      obtain ⟨η, hη⟩ := h.exists_switch_eq_one ψ.1
      exact ⟨⟨_, (h.realizes_iff hbal hΩ' hη).2 (ψ.2.switch η)⟩, ⟨_, ψ.2.switch η⟩, ⟨η, rfl⟩,
        rfl, hη⟩)
    (fun {φ₁ φ₂ ψ₁ ψ₂} ⟨e₁, h₁⟩ ⟨e₂, h₂⟩ => by
      show SwitchEquiv Ω.graph φ₁.1 φ₂.1 ↔ SwitchEquiv Ω'.graph ψ₁.1 ψ₂.1
      rw [← e₁, ← e₂]
      exact h.switchEquiv_iff h₁ h₂)
  refine ⟨e, fun φ hφ hφ' ha hb hc => he _ _ ⟨rfl, ?_⟩⟩
  rintro g (rfl | rfl | rfl) <;> assumption

/-- §2.4, Proposition 2.10 (ii) for an exchange pair, additive: if `X` is balanced in `Ω` and `Ω'`
has the balanced cycles `Δ_X B`, a bijection of switching-and-scaling classes fixes the class of
every gain function that is `0` on `X` and realizes both biased graphs. -/
theorem exists_addRealizationClass_equiv {F : Type*} [Field F]
    (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    (hΩ' : Ω'.balanced = deltaBalanced Ω.balanced {a, b, c}) :
    ∃ e : AddRealizationClass Ω F ≃ AddRealizationClass Ω' F,
      ∀ (ψ : E → F) (hψ : Ω.AddRealizes ψ) (hψ' : Ω'.AddRealizes ψ), ψ a = 0 → ψ b = 0 →
        ψ c = 0 → e (Quotient.mk _ ⟨ψ, hψ⟩) = Quotient.mk _ ⟨ψ, hψ'⟩ := by
  obtain ⟨e, he⟩ := exists_quotient_equiv (addRealizationSetoid Ω F) (addRealizationSetoid Ω' F)
    (fun φ ψ => φ.1 = ψ.1 ∧ ∀ g ∈ ({a, b, c} : Set E), φ.1 g = 0)
    (fun φ => by
      have hφ : Ω.graph.addBalancedCycles φ.1 = Ω.balanced := φ.2
      obtain ⟨-, η, hη⟩ := mem_addBalancedCycles_iff.1
        (by rw [hφ]; exact hbal : ({a, b, c} : Set E) ∈ Ω.graph.addBalancedCycles φ.1)
      exact ⟨⟨_, φ.2.addSwitch η⟩, ⟨_, (h.addRealizes_iff hbal hΩ' hη).1 (φ.2.addSwitch η)⟩,
        ⟨η, 1, by rw [Units.val_one, one_smul]⟩, rfl, hη⟩)
    (fun ψ => by
      obtain ⟨η, hη⟩ := h.exists_addSwitch_eq_zero ψ.1
      exact ⟨⟨_, (h.addRealizes_iff hbal hΩ' hη).2 (ψ.2.addSwitch η)⟩, ⟨_, ψ.2.addSwitch η⟩,
        ⟨η, 1, by rw [Units.val_one, one_smul]⟩, rfl, hη⟩)
    (fun {φ₁ φ₂ ψ₁ ψ₂} ⟨e₁, h₁⟩ ⟨e₂, h₂⟩ => by
      show SwitchScaleEquiv Ω.graph φ₁.1 φ₂.1 ↔ SwitchScaleEquiv Ω'.graph ψ₁.1 ψ₂.1
      rw [← e₁, ← e₂]
      exact h.switchScaleEquiv_iff h₁ h₂)
  refine ⟨e, fun ψ hψ hψ' ha hb hc => he _ _ ⟨rfl, ?_⟩⟩
  rintro g (rfl | rfl | rfl) <;> assumption

end OrientedGraph.IsExchange

example := exists_quotient_equiv ⟨(· = ·), eq_equivalence⟩ ⟨(· = ·), eq_equivalence⟩ (α := ℕ)
  (β := ℕ) (· = ·) (fun p => ⟨p, p, rfl, rfl⟩) (fun q => ⟨q, q, rfl, rfl⟩)
  fun h₁ h₂ => h₁ ▸ h₂ ▸ Iff.rfl

example := triangle_isDeltaExchange.isExchange.exists_realizationClass_equiv (Γ := ℚˣ)
  triangle_isDeltaExchange.balanced triangle_isDeltaExchange.balanced_eq

example := triangle_isDeltaExchange.isExchange.exists_addRealizationClass_equiv (F := ℚ)
  triangle_isDeltaExchange.balanced triangle_isDeltaExchange.balanced_eq

end Transformer.GainMatroid
