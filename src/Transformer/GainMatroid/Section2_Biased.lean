module

public import Mathlib.Tactic.Ring
public import Transformer.GainMatroid.Section2_Balance

/-!
# Biased graphs and the balanced cycles of a gain function

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1 and §2.3. A biased graph is a graph with a
collection of cycles such that no theta contains exactly two of them (`BiasedGraph`,
`BiasedGraph.theta_property`). A gain function `γ` determines the collection `B_γ` of its balanced
cycles (`balancedCycles`). That collection has the theta property (`IsTheta.balanced_of_two`), so
`(G, B_γ)` is a biased graph (`gainBiased`). `balancedCycles_switch` is the source's
`B_γ = B_{γ^η}` and `addBalancedCycles_smul` its `B_{aφ} = B_φ`.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {Γ : Type*}

namespace OrientedGraph

open Walk

/-- Permuting the first two paths of a theta gives a theta. -/
theorem IsTheta.swap₁₂ {G : OrientedGraph V E} {x y : V} {P Q R : G.Walk x y}
    (h : IsTheta P Q R) : IsTheta Q P R :=
  ⟨h.ne, h.pathQ, h.pathP, h.pathR, h.openPQ.symm, h.openQR, h.openPR⟩

/-- Permuting the last two paths of a theta gives a theta. -/
theorem IsTheta.swap₂₃ {G : OrientedGraph V E} {x y : V} {P Q R : G.Walk x y}
    (h : IsTheta P Q R) : IsTheta P R Q :=
  ⟨h.ne, h.pathP, h.pathR, h.pathQ, h.openPR, h.openPQ, h.openQR.symm⟩

variable [DecidableEq V] {G : OrientedGraph V E}

/-- The collection `B_γ` of cycles balanced with respect to `γ` (§2.3): the edge sets of closed
walks around a cycle with gain one. -/
@[to_additive addBalancedCycles
  /-- The collection `B_γ` of cycles balanced with respect to an additive gain `γ` (§2.3). -/]
def balancedCycles [Group Γ] (G : OrientedGraph V E) (γ : E → Γ) : Set (Set E) :=
  {C | ∃ (u : V) (W : G.Walk u u), W.IsCycleWalk ∧ W.edgeSet = C ∧ W.gain γ = 1}

/-- A cycle is balanced exactly when some switching function trivializes all of its edges. -/
@[to_additive mem_addBalancedCycles_iff
  /-- A cycle is balanced exactly when some additive switching trivializes all of its edges. -/]
theorem mem_balancedCycles_iff [Group Γ] {γ : E → Γ} {C : Set E} :
    C ∈ G.balancedCycles γ ↔ G.IsCycle C ∧ ∃ η : V → Γ, ∀ e ∈ C, switch G η γ e = 1 := by
  constructor
  · rintro ⟨u, W, hW, rfl, hg⟩
    obtain ⟨η, hη⟩ := (hW.gain_eq_one_iff γ).1 hg
    exact ⟨⟨u, W, hW, rfl⟩, η, hη⟩
  · rintro ⟨⟨u, W, hW, rfl⟩, η, hη⟩
    exact ⟨u, W, hW, rfl, (hW.gain_eq_one_iff γ).2 ⟨η, hη⟩⟩

/-- Any one closed walk around a cycle decides whether the cycle is balanced (§2.3). -/
@[to_additive mem_addBalancedCycles_iff
  /-- Any one closed walk around a cycle decides whether it is balanced (§2.3). -/]
theorem Walk.IsCycleWalk.mem_balancedCycles_iff [Group Γ] (γ : E → Γ) {u : V} {W : G.Walk u u}
    (hW : W.IsCycleWalk) : W.edgeSet ∈ G.balancedCycles γ ↔ W.gain γ = 1 := by
  constructor
  · rintro ⟨v, W', hW', h, hg⟩
    exact (gain_eq_one_iff_of_edgeSet_eq γ hW' hW h).1 hg
  · exact fun hg => ⟨u, W, hW, rfl, hg⟩

/-- The source's `B_γ = B_{γ^η}` (§2.3). -/
@[to_additive addBalancedCycles_addSwitch
  /-- The source's `B_γ = B_{γ^η}` for additive gains (§2.3). -/]
theorem balancedCycles_switch [Group Γ] (G : OrientedGraph V E) (ζ : V → Γ) (γ : E → Γ) :
    G.balancedCycles (switch G ζ γ) = G.balancedCycles γ := by
  ext C
  simp only [mem_balancedCycles_iff]
  refine and_congr_right fun _ => ⟨fun ⟨η, h⟩ => ⟨ζ * η, fun e he => ?_⟩,
    fun ⟨η, h⟩ => ⟨ζ⁻¹ * η, fun e he => ?_⟩⟩
  · rw [← switch_switch]
    exact h e he
  · rw [switch_switch, mul_inv_cancel_left]
    exact h e he

omit [DecidableEq V] in
/-- Scaling a gain function and its switching function together scales the result. -/
theorem addSwitch_smul {F : Type*} [Field F] (G : OrientedGraph V E) (a : F) (η : V → F)
    (φ : E → F) : addSwitch G (a • η) (a • φ) = a • addSwitch G η φ := by
  funext e
  simp only [addSwitch_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- The source's `B_{aφ} = B_φ` for a nonzero scalar `a` (§2.3). -/
theorem addBalancedCycles_smul {F : Type*} [Field F] (G : OrientedGraph V E) {a : F} (ha : a ≠ 0)
    (φ : E → F) : G.addBalancedCycles (a • φ) = G.addBalancedCycles φ := by
  ext C
  simp only [mem_addBalancedCycles_iff]
  refine and_congr_right fun _ => ⟨fun ⟨η, h⟩ => ⟨a⁻¹ • η, fun e he => ?_⟩,
    fun ⟨η, h⟩ => ⟨a • η, fun e he => ?_⟩⟩
  · have key : addSwitch G η (a • φ) = a • addSwitch G (a⁻¹ • η) φ := by
      rw [← addSwitch_smul, smul_smul, mul_inv_cancel₀ ha, one_smul]
    have h₀ := h e he
    rw [key, Pi.smul_apply, smul_eq_mul] at h₀
    exact (mul_eq_zero.1 h₀).resolve_left ha
  · rw [addSwitch_smul, Pi.smul_apply, h e he, smul_zero]

/-- Two openly disjoint paths form a balanced cycle exactly when they have equal gains. -/
@[to_additive mem_addBalancedCycles_iff
  /-- Two openly disjoint paths form a balanced cycle exactly when their additive gains agree. -/]
theorem Walk.Openly.mem_balancedCycles_iff [Group Γ] (γ : E → Γ) {x y : V} {P Q : G.Walk x y}
    (hxy : x ≠ y) (hP : P.IsPath) (hQ : Q.IsPath) (h : Openly P Q) :
    P.edgeSet ∪ Q.edgeSet ∈ G.balancedCycles γ ↔ P.gain γ = Q.gain γ := by
  rw [← edgeSet_append_reverse, (h.isCycleWalk hxy hP hQ).mem_balancedCycles_iff, gain_append,
    gain_reverse γ hQ, mul_inv_eq_one]

/-- `B_γ` has the theta property (§2.3): two balanced cycles of a theta force the third. -/
@[to_additive addBalanced_of_two
  /-- The balanced cycles of an additive gain have the theta property (§2.3). -/]
theorem IsTheta.balanced_of_two [Group Γ] (γ : E → Γ) {x y : V} {P Q R : G.Walk x y}
    (h : IsTheta P Q R) (hPQ : P.edgeSet ∪ Q.edgeSet ∈ G.balancedCycles γ)
    (hPR : P.edgeSet ∪ R.edgeSet ∈ G.balancedCycles γ) :
    Q.edgeSet ∪ R.edgeSet ∈ G.balancedCycles γ := by
  rw [h.openPQ.mem_balancedCycles_iff γ h.ne h.pathP h.pathQ] at hPQ
  rw [h.openPR.mem_balancedCycles_iff γ h.ne h.pathP h.pathR] at hPR
  rw [h.openQR.mem_balancedCycles_iff γ h.ne h.pathQ h.pathR]
  exact hPQ.symm.trans hPR

end OrientedGraph

/-- A **biased graph** `(G, B)` (§2.1): a graph with a collection of cycles such that if two cycles
of a theta are in the collection, then so is the third. `BiasedGraph.theta_property` restates this
in the source's words. -/
structure BiasedGraph (V : Type uV) (E : Type uE) where
  /-- The underlying graph, in a reference orientation that the cycles ignore. -/
  graph : OrientedGraph V E
  /-- The balanced cycles. -/
  balanced : Set (Set E)
  /-- Members of the collection are cycles. -/
  isCycle_of_mem : ∀ C ∈ balanced, graph.IsCycle C
  /-- The theta property. -/
  theta : ∀ {x y : V} {P Q R : graph.Walk x y}, OrientedGraph.IsTheta P Q R →
    P.edgeSet ∪ Q.edgeSet ∈ balanced → P.edgeSet ∪ R.edgeSet ∈ balanced →
      Q.edgeSet ∪ R.edgeSet ∈ balanced

/-- The theta property in the source's words (§2.1): no theta contains exactly two cycles of `B`. -/
theorem BiasedGraph.theta_property (Ω : BiasedGraph V E) {x y : V} {P Q R : Ω.graph.Walk x y}
    (h : OrientedGraph.IsTheta P Q R) :
    ¬(P.edgeSet ∪ Q.edgeSet ∈ Ω.balanced ∧ P.edgeSet ∪ R.edgeSet ∈ Ω.balanced ∧
        Q.edgeSet ∪ R.edgeSet ∉ Ω.balanced) ∧
      ¬(P.edgeSet ∪ Q.edgeSet ∈ Ω.balanced ∧ Q.edgeSet ∪ R.edgeSet ∈ Ω.balanced ∧
        P.edgeSet ∪ R.edgeSet ∉ Ω.balanced) ∧
      ¬(P.edgeSet ∪ R.edgeSet ∈ Ω.balanced ∧ Q.edgeSet ∪ R.edgeSet ∈ Ω.balanced ∧
        P.edgeSet ∪ Q.edgeSet ∉ Ω.balanced) := by
  refine ⟨fun ⟨h₁, h₂, h₃⟩ => h₃ (Ω.theta h h₁ h₂), fun ⟨h₁, h₂, h₃⟩ => h₃ ?_,
    fun ⟨h₁, h₂, h₃⟩ => h₃ ?_⟩
  · exact Ω.theta h.swap₁₂ (by rwa [Set.union_comm]) h₂
  · exact Ω.theta h.swap₂₃.swap₁₂ (by rwa [Set.union_comm]) (by rwa [Set.union_comm])

namespace OrientedGraph

variable [DecidableEq V]

/-- The biased graph `(G, B_γ)` of a gain graph (§2.3). -/
@[to_additive addGainBiased /-- The biased graph `(G, B_γ)` of an additive gain graph (§2.3). -/]
def gainBiased [Group Γ] (G : OrientedGraph V E) (γ : E → Γ) : BiasedGraph V E where
  graph := G
  balanced := G.balancedCycles γ
  isCycle_of_mem _ hC := (mem_balancedCycles_iff.1 hC).1
  theta h h₁ h₂ := h.balanced_of_two γ h₁ h₂

end OrientedGraph

/-- `γ` **realizes** the biased graph `Ω` (§2.3): `B_γ` is the collection of `Ω`. -/
@[to_additive AddRealizes /-- The additive gain `γ` **realizes** `Ω` (§2.3). -/]
def BiasedGraph.Realizes [DecidableEq V] [Group Γ] (Ω : BiasedGraph V E) (γ : E → Γ) : Prop :=
  Ω.graph.balancedCycles γ = Ω.balanced

/-- Switching preserves realizations (§2.3). -/
@[to_additive AddRealizes.addSwitch /-- Additive switching preserves realizations (§2.3). -/]
theorem BiasedGraph.Realizes.switch [DecidableEq V] [Group Γ] {Ω : BiasedGraph V E} {γ : E → Γ}
    (h : Ω.Realizes γ) (η : V → Γ) : Ω.Realizes (Transformer.GainMatroid.switch Ω.graph η γ) := by
  rw [BiasedGraph.Realizes, OrientedGraph.balancedCycles_switch]
  exact h

end Transformer.GainMatroid
