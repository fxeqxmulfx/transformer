module

public import Transformer.GainMatroid.Section2_ExchangeMatroids

/-!
# Gain functions under a Δ-Y exchange: Proposition 2.10

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4.

**Proposition 2.10.** "Let `X` be a balanced triangle of a biased graph `(G, B)`. Let `Γ` be the
multiplicative (resp., additive) group of a field. (i) `φ` is a `Γ`-realization of `(G, B)` if and
only if `φ` is a `Γ`-realization of `Δ_X (G, B)`. (ii) The switching (resp., switching-and-scaling)
classes of `Γ`-realizations of `(G, B)` are in one-to-one correspondence with the switching (resp.,
switching-and-scaling) classes of `Γ`-realizations of `Δ_X (G, B)`."

Part (i) is false as printed (`proposition2_10_fails`, `proposition2_10_add_fails`). If `G` is a
triangle whose cycle `X` is balanced, then `Δ_X G` is a star, which has no cycles, so every gain
function realizes `Δ_X (G, B)`. A gain function with non-identity value around `X` does not realize
`(G, B)`.

The source's proof compares the realizations "assigning identity gains to each edge in `X`". The
corrected part (i) (`IsDeltaExchange.realizes_iff`, `IsDeltaExchange.addRealizes_iff`) takes that
hypothesis. Part (ii) is the source's "canonical bijection": it fixes the class of each gain
function that is the identity on `X` and realizes both biased graphs.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E : Type*}

/-- In `Δ_X G`, the edges of the triangle form a star at the new vertex (§2.4). -/
theorem OrientedGraph.IsLabelledTriangle.deltaGraph_isLabelledClaw {G : OrientedGraph V E}
    {a b c : E} {x y z : V} (h : G.IsLabelledTriangle a b c x y z) :
    (G.deltaGraph a b c x y z).IsLabelledClaw none a b c (some x) (some y) (some z) := by
  classical
  refine ⟨Or.inr ⟨?_, ?_⟩, Or.inr ⟨?_, ?_⟩, Or.inr ⟨?_, ?_⟩, h.ne_ab, h.ne_ac, h.ne_bc,
    by simpa using h.ne_xy, by simpa using h.ne_xz, by simpa using h.ne_yz, by simp, by simp,
    by simp, ?_⟩
  · simp [deltaGraph]
  · simp [deltaGraph]
  · simp [deltaGraph, h.ne_ab.symm]
  · simp [deltaGraph]
  · simp [deltaGraph, h.ne_ac.symm, h.ne_bc.symm]
  · simp [deltaGraph]
  · intro e he
    by_contra hne
    push Not at hne
    rcases he with he | he
    · have : (G.deltaGraph a b c x y z).tail e = some (G.tail e) := by
        simp [deltaGraph, hne.1, hne.2.1, hne.2.2]
      simp [this] at he
    · have : (G.deltaGraph a b c x y z).head e = some (G.head e) := by
        simp [deltaGraph, hne.1, hne.2.1, hne.2.2]
      simp [this] at he

variable [DecidableEq V]

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

namespace BiasedGraph.IsDeltaExchange

variable {Ω : BiasedGraph V E} {Ω' : BiasedGraph (Option V) E} {a b c : E} {x y z : V}

/-- **Proposition 2.10 (i)**, multiplicative case, corrected: for a gain function that is the
identity on `X`, realizing `(G, B)` and realizing `Δ_X (G, B)` are equivalent. As printed the
statement has no hypothesis on `φ` and is false (`proposition2_10_fails`). -/
theorem realizes_iff {Γ : Type*} [Group Γ] (h : Ω.IsDeltaExchange a b c x y z Ω') {φ : E → Γ}
    (ha : φ a = 1) (hb : φ b = 1) (hc : φ c = 1) : Ω.Realizes φ ↔ Ω'.Realizes φ := by
  sorry

/-- **Proposition 2.10 (i)**, additive case, corrected as in `realizes_iff`. -/
theorem addRealizes_iff {F : Type*} [Field F] (h : Ω.IsDeltaExchange a b c x y z Ω') {ψ : E → F}
    (ha : ψ a = 0) (hb : ψ b = 0) (hc : ψ c = 0) : Ω.AddRealizes ψ ↔ Ω'.AddRealizes ψ := by
  sorry

/-- **Proposition 2.10 (ii)**, multiplicative case: a bijection of switching classes that fixes the
class of every gain function that is the identity on `X` and realizes both biased graphs. -/
theorem exists_realizationClass_equiv {Γ : Type*} [Group Γ]
    (h : Ω.IsDeltaExchange a b c x y z Ω') :
    ∃ e : RealizationClass Ω Γ ≃ RealizationClass Ω' Γ,
      ∀ (φ : E → Γ) (hφ : Ω.Realizes φ) (hφ' : Ω'.Realizes φ), φ a = 1 → φ b = 1 → φ c = 1 →
        e (Quotient.mk _ ⟨φ, hφ⟩) = Quotient.mk _ ⟨φ, hφ'⟩ := by
  sorry

/-- **Proposition 2.10 (ii)**, additive case: a bijection of switching-and-scaling classes that
fixes the class of every gain function that is `0` on `X` and realizes both biased graphs. -/
theorem exists_addRealizationClass_equiv {F : Type*} [Field F]
    (h : Ω.IsDeltaExchange a b c x y z Ω') :
    ∃ e : AddRealizationClass Ω F ≃ AddRealizationClass Ω' F,
      ∀ (ψ : E → F) (hψ : Ω.AddRealizes ψ) (hψ' : Ω'.AddRealizes ψ), ψ a = 0 → ψ b = 0 →
        ψ c = 0 → e (Quotient.mk _ ⟨ψ, hψ⟩) = Quotient.mk _ ⟨ψ, hψ'⟩ := by
  sorry

end BiasedGraph.IsDeltaExchange

/-- The gain `2` on every edge of `triangle` has product `8` around it, so no switching makes it
the identity. -/
private theorem not_balanced_ratTwo (η : Fin 3 → ℚˣ)
    (h : ∀ e ∈ ({1, 2, 0} : Set (Fin 3)), switch triangle η (fun _ => ratTwo) e = 1) : False := by
  have h0 : (η 0)⁻¹ * ratTwo * η 1 = 1 := h 0 (by simp)
  have h1 : (η 1)⁻¹ * ratTwo * η 2 = 1 := h 1 (by simp)
  have h2 : (η 2)⁻¹ * ratTwo * η 0 = 1 := h 2 (by simp)
  rw [mul_assoc, inv_mul_eq_one] at h0 h1 h2
  have e0 : (η 0 : ℚ) = 2 * (η 1 : ℚ) := by simpa [ratTwo] using congrArg Units.val h0
  have e1 : (η 1 : ℚ) = 2 * (η 2 : ℚ) := by simpa [ratTwo] using congrArg Units.val h1
  have e2 : (η 2 : ℚ) = 2 * (η 0 : ℚ) := by simpa [ratTwo] using congrArg Units.val h2
  exact (η 0).ne_zero (by linarith)

/-- The gain `1` on every edge of `triangle` has sum `3` around it. -/
private theorem not_balanced_one (η : Fin 3 → ℚ)
    (h : ∀ e ∈ ({1, 2, 0} : Set (Fin 3)), addSwitch triangle η (fun _ => (1 : ℚ)) e = 0) :
    False := by
  have h0 : -η 0 + 1 + η 1 = 0 := h 0 (by simp)
  have h1 : -η 1 + 1 + η 2 = 0 := h 1 (by simp)
  have h2 : -η 2 + 1 + η 0 = 0 := h 2 (by simp)
  linarith

/-- **Proposition 2.10 (i) is false as printed**, multiplicative case: on the triangle with its
cycle balanced, the constant gain `2` realizes `Δ_X (G, B)` and does not realize `(G, B)`. -/
theorem proposition2_10_fails :
    ∃ (Ω : BiasedGraph (Fin 3) (Fin 3)) (Ω' : BiasedGraph (Option (Fin 3)) (Fin 3))
      (φ : Fin 3 → ℚˣ), Ω.IsDeltaExchange 1 2 0 0 1 2 Ω' ∧ Ω'.Realizes φ ∧ ¬Ω.Realizes φ := by
  refine ⟨triangleBiased, _, fun _ => ratTwo, triangle_isDeltaExchange, ?_, fun hr => ?_⟩
  · refine Set.eq_empty_of_forall_notMem fun C hC => ?_
    exact triangleStar_not_isCycle C (triangle_deltaGraph ▸ (mem_balancedCycles_iff.1 hC).1)
  · have hX : ({1, 2, 0} : Set (Fin 3)) ∈ triangle.balancedCycles (fun _ : Fin 3 => ratTwo) :=
      (Set.ext_iff.1 hr _).2 rfl
    obtain ⟨-, η, hη⟩ := mem_balancedCycles_iff.1 hX
    exact not_balanced_ratTwo η hη

/-- **Proposition 2.10 (i) is false as printed**, additive case: the constant gain `1`. -/
theorem proposition2_10_add_fails :
    ∃ (Ω : BiasedGraph (Fin 3) (Fin 3)) (Ω' : BiasedGraph (Option (Fin 3)) (Fin 3))
      (ψ : Fin 3 → ℚ), Ω.IsDeltaExchange 1 2 0 0 1 2 Ω' ∧ Ω'.AddRealizes ψ ∧ ¬Ω.AddRealizes ψ := by
  refine ⟨triangleBiased, _, fun _ => 1, triangle_isDeltaExchange, ?_, fun hr => ?_⟩
  · refine Set.eq_empty_of_forall_notMem fun C hC => ?_
    exact triangleStar_not_isCycle C (triangle_deltaGraph ▸ (mem_addBalancedCycles_iff.1 hC).1)
  · have hX : ({1, 2, 0} : Set (Fin 3)) ∈ triangle.addBalancedCycles (fun _ : Fin 3 => (1 : ℚ)) :=
      (Set.ext_iff.1 hr _).2 rfl
    obtain ⟨-, η, hη⟩ := mem_addBalancedCycles_iff.1 hX
    exact not_balanced_one η hη

example := triangle_isLabelledTriangle.deltaGraph_isLabelledClaw

example := triangle_isDeltaExchange.realizes_iff (φ := (1 : Fin 3 → ℚˣ)) rfl rfl rfl

example := triangle_isDeltaExchange.addRealizes_iff (F := ℚ) (ψ := 0) rfl rfl rfl

example := triangle_isDeltaExchange.exists_realizationClass_equiv (Γ := ℚˣ)

example := triangle_isDeltaExchange.exists_addRealizationClass_equiv (F := ℚ)

end Transformer.GainMatroid
