module

public import Transformer.GainMatroid.Section2_Classes

/-!
# Unbalanced closed walks off a vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4. Let `T A S = B` for canonical matrices `A`, `B`
of one graph, with `T` nonsingular and `S` diagonal. In the row of a vertex `w`, the matrix `B`
vanishes on every edge away from `w`, so what constrains that row of `T` is the part of the graph
off `w`. The source works this out separately on each small graph in Lemmas 4.3, 4.4, 4.9 and 4.10.

This module names the two properties those computations use. `ConnectedOff` says that deleting a
vertex keeps the non-isolated vertices connected, and `HasNoBalancingVertex` that it leaves a closed
walk of non-identity gain. Isolated vertices are set aside, as the source deletes them without
comment (§2.1). The closed walks used on the small graphs are two parallel links (`pairWalk`) and a
triangle (`triangleWalk`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

theorem Joins.ne_of_loopless {e : E} {u v : V} (hG : G.Loopless) (h : G.Joins e u v) : u ≠ v := by
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · exact hG e
  · exact (hG e).symm

theorem Joins.left_mem_verts {e : E} {u v : V} (h : G.Joins e u v) : u ∈ G.verts Set.univ := by
  rcases h with ⟨h₁, -⟩ | ⟨-, h₂⟩
  · exact mem_verts.2 ⟨e, trivial, Or.inl h₁⟩
  · exact mem_verts.2 ⟨e, trivial, Or.inr h₂⟩

theorem Joins.right_mem_verts {e : E} {u v : V} (h : G.Joins e u v) : v ∈ G.verts Set.univ :=
  h.symm.left_mem_verts

theorem tail_mem_verts (e : E) : G.tail e ∈ G.verts Set.univ :=
  mem_verts.2 ⟨e, trivial, Or.inl rfl⟩

theorem head_mem_verts (e : E) : G.head e ∈ G.verts Set.univ :=
  mem_verts.2 ⟨e, trivial, Or.inr rfl⟩

/-- Deleting any one vertex keeps the non-isolated vertices connected. -/
def ConnectedOff (G : OrientedGraph V E) : Prop :=
  ∀ w x y, x ∈ G.verts Set.univ → y ∈ G.verts Set.univ → x ≠ w → y ≠ w →
    ∃ p : G.Walk x y, w ∉ p.support

/-- A loopless graph that stays connected off each vertex, with an edge missing each vertex, is
connected on its non-isolated vertices. -/
theorem ConnectedOff.nonempty_walk (hc : G.ConnectedOff) (hG : G.Loopless)
    (hoff : ∀ w, ∃ e, G.tail e ≠ w ∧ G.head e ≠ w) {x y : V}
    (hx : x ∈ G.verts Set.univ) (hy : y ∈ G.verts Set.univ) : Nonempty (G.Walk x y) := by
  by_cases hxy : x = y
  · subst hxy
    exact ⟨.nil x⟩
  obtain ⟨e, h₁, h₂⟩ := hoff x
  by_cases hty : G.tail e = y
  · have hhy : G.head e ≠ y := fun h => hG e (hty.trans h.symm)
    obtain ⟨p, -⟩ := hc (G.head e) x y hx hy h₂.symm hhy.symm
    exact ⟨p⟩
  · obtain ⟨p, -⟩ := hc (G.tail e) x y hx hy h₁.symm (Ne.symm hty)
    exact ⟨p⟩

/-- A cycle walk has as many distinct edges as steps. -/
theorem Walk.IsCycleWalk.encard_edgeSet {u : V} {W : G.Walk u u} (hW : W.IsCycleWalk) :
    W.edgeSet.encard = W.edges.length := by
  classical
  have h : W.edgeSet = ↑W.edges.toFinset := by
    ext e
    simp [Walk.edgeSet]
  rw [h, Set.encard_coe_eq_coe_finsetCard, List.toFinset_card_of_nodup hW.edges_nodup]

/-- The closed walk around two parallel links. -/
def pairWalk {e f : E} {u v : V} (he : G.Joins e u v) (hf : G.Joins f u v) : G.Walk u u :=
  .cons e he (.cons f hf.symm (.nil u))

theorem pairWalk_isCycleWalk {e f : E} {u v : V} (he : G.Joins e u v) (hf : G.Joins f u v)
    (hne : e ≠ f) (huv : u ≠ v) : (pairWalk he hf).IsCycleWalk :=
  ⟨by simp [pairWalk, Walk.edges], by simp [pairWalk, Walk.edges, hne],
    by simp [pairWalk, Walk.support, huv.symm]⟩

/-- The closed walk around a triangle. -/
def triangleWalk {a b c : E} {x y z : V} (ha : G.Joins a x y) (hb : G.Joins b y z)
    (hc : G.Joins c z x) : G.Walk x x :=
  .cons a ha (.cons b hb (.cons c hc (.nil x)))

theorem triangleWalk_isCycleWalk {a b c : E} {x y z : V} (ha : G.Joins a x y) (hb : G.Joins b y z)
    (hc : G.Joins c z x) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hxy : x ≠ y) (hxz : x ≠ z)
    (hyz : y ≠ z) : (triangleWalk ha hb hc).IsCycleWalk :=
  ⟨by simp [triangleWalk, Walk.edges], by simp [triangleWalk, Walk.edges, hab, hac, hbc],
    by simp [triangleWalk, Walk.support, hyz, hxy.symm, hxz.symm]⟩

section Gain

variable [DecidableEq V]

/-- Every vertex is avoided by a closed walk of non-identity gain. -/
@[to_additive HasNoAddBalancingVertex
  /-- Every vertex is avoided by a closed walk of nonzero additive gain. -/]
def HasNoBalancingVertex [Group Γ] (G : OrientedGraph V E) (γ : E → Γ) : Prop :=
  ∀ w, ∃ (u : V) (W : G.Walk u u), w ∉ W.support ∧ W.gain γ ≠ 1

/-- `x` is joined to a closed walk of non-identity gain, both avoiding `w`. -/
def IsUnbalancedOff [Group Γ] (G : OrientedGraph V E) (γ : E → Γ) (w x : V) : Prop :=
  ∃ (u : V) (W : G.Walk u u) (p : G.Walk u x), w ∉ W.support ∧ w ∉ p.support ∧ W.gain γ ≠ 1

/-- A closed walk of non-identity gain has an edge. -/
@[to_additive exists_mem_edges_of_addGain_ne_zero
  /-- A closed walk of nonzero additive gain has an edge. -/]
theorem Walk.exists_mem_edges_of_gain_ne_one [Group Γ] {γ : E → Γ} {u : V} (W : G.Walk u u)
    (h : W.gain γ ≠ 1) : ∃ e, e ∈ W.edges := by
  cases W with
  | nil => exact absurd (Walk.gain_nil γ u) h
  | cons e hj p => exact ⟨e, List.mem_cons_self⟩

/-- The base of a closed walk of non-identity gain is not isolated. -/
theorem Walk.start_mem_verts [Group Γ] {γ : E → Γ} {u : V} (W : G.Walk u u) (h : W.gain γ ≠ 1) :
    u ∈ G.verts Set.univ := by
  cases W with
  | nil => exact absurd (Walk.gain_nil γ u) h
  | cons e hj p => exact hj.left_mem_verts

/-- If no vertex balances `γ`, then every vertex misses both ends of some edge. -/
@[to_additive /-- If no vertex balances `γ`, then every vertex misses both ends of some edge. -/]
theorem HasNoBalancingVertex.exists_edge_off [Group Γ] {γ : E → Γ}
    (h : G.HasNoBalancingVertex γ) (w : V) : ∃ e, G.tail e ≠ w ∧ G.head e ≠ w := by
  obtain ⟨u, W, hW, hg⟩ := h w
  obtain ⟨e, he⟩ := W.exists_mem_edges_of_gain_ne_one hg
  obtain ⟨h₁, h₂⟩ := W.ends_mem_support he
  exact ⟨e, fun hc => hW (hc ▸ h₁), fun hc => hW (hc ▸ h₂)⟩

/-- With no balancing vertex, each non-isolated `x ≠ w` is joined off `w` to an unbalanced closed
walk. -/
theorem HasNoBalancingVertex.isUnbalancedOff [Group Γ] {γ : E → Γ}
    (h : G.HasNoBalancingVertex γ) (hc : G.ConnectedOff) {w x : V}
    (hx : x ∈ G.verts Set.univ) (hxw : x ≠ w) : G.IsUnbalancedOff γ w x := by
  obtain ⟨u, W, hW, hg⟩ := h w
  have huw : u ≠ w := fun hu => hW (hu ▸ W.start_mem_support)
  obtain ⟨p, hp⟩ := hc w u x (W.start_mem_verts hg) hx huw hxw
  exact ⟨u, W, p, hW, hp, hg⟩

/-- A cycle walk of a length that no balanced cycle has is an unbalanced closed walk. -/
@[to_additive exists_off_of_addCycleWalk
  /-- A cycle walk of a length that no balanced cycle has is an unbalanced closed walk. -/]
theorem exists_off_of_cycleWalk [Group Γ] {γ : E → Γ} {k : ℕ∞}
    (h : ∀ C ∈ G.balancedCycles γ, C.encard ≠ k) {u w : V} {W : G.Walk u u} (hW : W.IsCycleWalk)
    (hk : (W.edges.length : ℕ∞) = k) (hw : w ∉ W.support) :
    ∃ (x : V) (W' : G.Walk x x), w ∉ W'.support ∧ W'.gain γ ≠ 1 :=
  ⟨u, W, hw, fun hg => h _ ((hW.mem_balancedCycles_iff γ).2 hg) (by rw [hW.encard_edgeSet, hk])⟩

/-- Two parallel links off `w`, when no 2-cycle is balanced, are an unbalanced closed walk. -/
@[to_additive exists_off_of_addPair
  /-- Two parallel links off `w`, when no 2-cycle is balanced, are an unbalanced closed walk. -/]
theorem exists_off_of_pair [Group Γ] {γ : E → Γ} (h : ∀ C ∈ G.balancedCycles γ, C.encard ≠ 2)
    {e f : E} {u v w : V} (he : G.Joins e u v) (hf : G.Joins f u v) (hne : e ≠ f) (huv : u ≠ v)
    (hwu : w ≠ u) (hwv : w ≠ v) : ∃ (x : V) (W : G.Walk x x), w ∉ W.support ∧ W.gain γ ≠ 1 :=
  exists_off_of_cycleWalk h (pairWalk_isCycleWalk he hf hne huv)
    (by simp [pairWalk, Walk.edges]) (by simp [pairWalk, Walk.support, hwu, hwv])

/-- A triangle off `w`, when no 3-cycle is balanced, is an unbalanced closed walk. -/
@[to_additive exists_off_of_addTriangle
  /-- A triangle off `w`, when no 3-cycle is balanced, is an unbalanced closed walk. -/]
theorem exists_off_of_triangle [Group Γ] {γ : E → Γ} (h : ∀ C ∈ G.balancedCycles γ, C.encard ≠ 3)
    {a b c : E} {x y z w : V} (ha : G.Joins a x y) (hb : G.Joins b y z) (hc : G.Joins c z x)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
    (hwx : w ≠ x) (hwy : w ≠ y) (hwz : w ≠ z) :
    ∃ (x' : V) (W : G.Walk x' x'), w ∉ W.support ∧ W.gain γ ≠ 1 :=
  exists_off_of_cycleWalk h (triangleWalk_isCycleWalk ha hb hc hab hac hbc hxy hxz hyz)
    (by simp [triangleWalk, Walk.edges]) (by simp [triangleWalk, Walk.support, hwx, hwy, hwz])

end Gain

end OrientedGraph

end Transformer.GainMatroid
