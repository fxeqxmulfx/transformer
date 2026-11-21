module

public import Transformer.GainMatroid.Section2_Connectivity

/-!
# Balanced, almost-balanced, properly unbalanced and tangled biased graphs

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2 and §2.1. A biased graph is **balanced** when
all of its cycles are balanced, **almost-balanced** when it is not balanced but some vertex lies on
every unbalanced cycle of length at least two, and **properly unbalanced** otherwise. A properly
unbalanced biased graph with no two vertex-disjoint unbalanced cycles is **tangled**. A **joint** is
an unbalanced loop, and a **balancing vertex** lies on every unbalanced cycle.

For a loop, balance under a gain function means that the gain is the identity
(`singleton_mem_balancedCycles_iff`), so a joint of a realized biased graph is a loop with
non-identity gain (`BiasedGraph.Realizes.isJoint_iff`). The source's remark in §1.2 that every gain
function realizing a balanced biased graph is switching equivalent to the identity gain is
`BiasedGraph.Realizes.switchEquiv_one`.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- The closed walk around a loop. -/
def loopWalk (G : OrientedGraph V E) (e : E) (h : G.tail e = G.head e) :
    G.Walk (G.tail e) (G.tail e) :=
  .cons e (show G.Joins e (G.tail e) (G.tail e) from Or.inl ⟨rfl, h.symm⟩) (.nil _)

theorem loopWalk_isCycleWalk {e : E} (h : G.tail e = G.head e) : (G.loopWalk e h).IsCycleWalk :=
  ⟨by simp [loopWalk, Walk.edges], by simp [loopWalk, Walk.edges],
    by simp [loopWalk, Walk.support]⟩

theorem edgeSet_loopWalk {e : E} (h : G.tail e = G.head e) : (G.loopWalk e h).edgeSet = {e} := by
  ext f
  simp [loopWalk, Walk.edgeSet, Walk.edges]

/-- A loop alone is a cycle. -/
theorem isCycle_singleton_of_loop {e : E} (h : G.tail e = G.head e) : G.IsCycle {e} :=
  ⟨_, G.loopWalk e h, loopWalk_isCycleWalk h, edgeSet_loopWalk h⟩

variable [DecidableEq V]

/-- A loop is balanced exactly when its gain is the identity (§2.3). -/
@[to_additive singleton_mem_addBalancedCycles_iff
  /-- A loop is balanced exactly when its additive gain is zero (§2.3). -/]
theorem singleton_mem_balancedCycles_iff [Group Γ] (γ : E → Γ) {e : E} (h : G.tail e = G.head e) :
    {e} ∈ G.balancedCycles γ ↔ γ e = 1 := by
  have hW := (loopWalk_isCycleWalk h).mem_balancedCycles_iff γ
  rw [edgeSet_loopWalk h] at hW
  rw [hW, loopWalk, Walk.gain_cons, dart_of_tail γ rfl, Walk.gain_nil, mul_one]

/-- A gain function under which every cycle is balanced is switching equivalent to the identity
gain (§1.2). The proof normalizes on a maximal forest. Each other edge closes a balanced cycle with
the forest, so its normalized gain is the identity. -/
@[to_additive addSwitchEquiv_zero_of_balanced
  /-- An additive gain function under which every cycle is balanced is switching equivalent to the
  zero gain (§1.2). -/]
theorem switchEquiv_one_of_balanced [Group Γ] {γ : E → Γ}
    (hB : ∀ C, G.IsCycle C → C ∈ G.balancedCycles γ) : SwitchEquiv G γ 1 := by
  obtain ⟨F, -, hF⟩ := exists_maximal_isForest (G := G) isForest_empty
  obtain ⟨η, hη⟩ := exists_switch_eq_one_of_isForest hF.1 γ
  refine ⟨η, funext fun e => ?_⟩
  by_cases he : e ∈ F
  · exact hη e he
  obtain ⟨u, W, hW, heW, hsub⟩ := exists_cycleWalk_of_maximal hF he
  obtain ⟨-, ξ, hξ⟩ := mem_balancedCycles_iff.1 (hB _ ⟨u, W, hW, rfl⟩)
  have hsw : ∀ f, switch G (η⁻¹ * ξ) (switch G η γ) f = switch G ξ γ f := fun f => by
    rw [switch_switch, mul_inv_cancel_left]
  have hconst : ∀ f ∈ W.edges, f ≠ e → (η⁻¹ * ξ) (G.tail f) = (η⁻¹ * ξ) (G.head f) := by
    intro f hf hfe
    have h₀ := hsw f
    rw [hξ f hf, switch_apply, hη f (hsub f hf hfe), mul_one, inv_mul_eq_one] at h₀
    exact h₀
  have h₀ := hsw e
  rw [hξ e heW, switch_apply, hW.eq_of_forall_ne heW (η⁻¹ * ξ) hconst] at h₀
  have h₁ := congrArg (fun z => (η⁻¹ * ξ) (G.head e) * z * ((η⁻¹ * ξ) (G.head e))⁻¹) h₀
  simpa [mul_assoc] using h₁

end OrientedGraph

namespace BiasedGraph

variable (Ω : BiasedGraph V E)

/-- `C` is an **unbalanced cycle** of `Ω` (§2.1). -/
def IsUnbalancedCycle (C : Set E) : Prop := Ω.graph.IsCycle C ∧ C ∉ Ω.balanced

/-- `Ω` is **balanced** (§2.1): all of its cycles are balanced. -/
def IsBalanced : Prop := ∀ C, Ω.graph.IsCycle C → C ∈ Ω.balanced

/-- `e` is a **joint** of `Ω` (§1.2): an unbalanced loop. -/
def IsJoint (e : E) : Prop := Ω.graph.tail e = Ω.graph.head e ∧ {e} ∉ Ω.balanced

/-- `v` is a **balancing vertex** of `Ω` (§2.1): every unbalanced cycle contains `v`. -/
def IsBalancingVertex (v : V) : Prop := ∀ C, Ω.IsUnbalancedCycle C → v ∈ Ω.graph.verts C

/-- `Ω` is **almost-balanced** (§1.2): it is not balanced, and some vertex lies on every unbalanced
cycle of length at least two. -/
def IsAlmostBalanced : Prop :=
  ¬Ω.IsBalanced ∧ ∃ v, ∀ C, Ω.IsUnbalancedCycle C → 2 ≤ C.encard → v ∈ Ω.graph.verts C

/-- `Ω` is **properly unbalanced** (§1.2): neither balanced nor almost-balanced. -/
def IsProperlyUnbalanced : Prop := ¬Ω.IsBalanced ∧ ¬Ω.IsAlmostBalanced

/-- `Ω` is **tangled** (§1.2): properly unbalanced, with no two vertex-disjoint unbalanced
cycles. -/
def IsTangled : Prop :=
  Ω.IsProperlyUnbalanced ∧ ∀ C D, Ω.IsUnbalancedCycle C → Ω.IsUnbalancedCycle D →
    ¬Disjoint (Ω.graph.verts C) (Ω.graph.verts D)

/-- The three classes of §1.2 cover all biased graphs. -/
theorem balanced_or_almost_or_proper :
    Ω.IsBalanced ∨ Ω.IsAlmostBalanced ∨ Ω.IsProperlyUnbalanced := by
  by_cases h₁ : Ω.IsBalanced
  · exact Or.inl h₁
  by_cases h₂ : Ω.IsAlmostBalanced
  · exact Or.inr (Or.inl h₂)
  · exact Or.inr (Or.inr ⟨h₁, h₂⟩)

variable {Ω} [DecidableEq V]

/-- A loop of a realized biased graph is a joint exactly when its gain is not the identity. -/
@[to_additive
  /-- A loop of a realized biased graph is a joint exactly when its additive gain is not zero. -/]
theorem Realizes.isJoint_iff [Group Γ] {γ : E → Γ} (h : Ω.Realizes γ) {e : E}
    (hloop : Ω.graph.tail e = Ω.graph.head e) : Ω.IsJoint e ↔ γ e ≠ 1 := by
  have hmem := OrientedGraph.singleton_mem_balancedCycles_iff γ hloop
  rw [show Ω.graph.balancedCycles γ = Ω.balanced from h] at hmem
  simp [IsJoint, hloop, hmem]

/-- Every gain function realizing a balanced biased graph is switching equivalent to the gain
function assigning the identity to every edge (§1.2). -/
@[to_additive AddRealizes.addSwitchEquiv_zero
  /-- Every additive gain function realizing a balanced biased graph is switching equivalent to the
  zero gain function (§1.2). -/]
theorem Realizes.switchEquiv_one [Group Γ] {γ : E → Γ} (h : Ω.Realizes γ) (hB : Ω.IsBalanced) :
    SwitchEquiv Ω.graph γ 1 :=
  OrientedGraph.switchEquiv_one_of_balanced fun C hC => by
    rw [show Ω.graph.balancedCycles γ = Ω.balanced from h]
    exact hB C hC

end BiasedGraph

open OrientedGraph

/-- The identity gain function makes every cycle balanced. -/
@[to_additive /-- The zero gain function makes every cycle balanced. -/]
theorem gainBiased_one_isBalanced [DecidableEq V] [Group Γ] (G : OrientedGraph V E) :
    (G.gainBiased (1 : E → Γ)).IsBalanced :=
  fun _ hC => mem_balancedCycles_iff.2 ⟨hC, 1, fun e _ => by simp [switch]⟩

example : loopGraph.IsCycle {()} := isCycle_singleton_of_loop rfl

example [Group Γ] (γ : Unit → Γ) : {()} ∈ loopGraph.balancedCycles γ ↔ γ () = 1 :=
  singleton_mem_balancedCycles_iff γ rfl

example [Group Γ] : SwitchEquiv pumpkin (1 : Fin 3 → Γ) 1 :=
  switchEquiv_one_of_balanced (gainBiased_one_isBalanced pumpkin)

example (c : ℚ) : (![c, c] : Fin 2 → ℚ) (pumpkin.tail 0) = ![c, c] (pumpkin.head 0) :=
  (pumpkin_cycleWalk (i := 0) (j := 1) (by decide)).eq_of_forall_ne
    (by simp [Walk.edges_append, Walk.edges_reverse, pumpkinWalk, Walk.edges])
    _ (fun _ _ _ => by simp [pumpkin])

example [Group Γ] : SwitchEquiv pumpkin (1 : Fin 3 → Γ) 1 :=
  BiasedGraph.Realizes.switchEquiv_one (Ω := pumpkin.gainBiased 1) rfl
    (gainBiased_one_isBalanced pumpkin)

example : AddSwitchEquiv pumpkin (0 : Fin 3 → ℚ) 0 :=
  BiasedGraph.AddRealizes.addSwitchEquiv_zero (Ω := pumpkin.addGainBiased 0) rfl
    (gainBiased_zero_isBalanced pumpkin)

example [Group Γ] (g : Γ) (hg : g ≠ 1) :
    (loopGraph.gainBiased fun _ => g).IsJoint () :=
  (BiasedGraph.Realizes.isJoint_iff (Ω := loopGraph.gainBiased fun _ => g) rfl rfl).2 hg

example : ¬(pumpkin.gainBiased (1 : Fin 3 → Multiplicative ℚ)).IsAlmostBalanced := fun h =>
  h.1 (gainBiased_one_isBalanced pumpkin)

end Transformer.GainMatroid
