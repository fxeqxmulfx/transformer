module

public import Transformer.GainMatroid.Section2_LinkMinors

/-!
# Induced gain functions realize link minors

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), Proposition 2.4 for link minors, which the source
says "follows immediately from the definitions". A cycle of `Ω / K \ D` lifts to a cycle of `Ω`
through paths of the forest `K` (`LinkMinorMap.exists_isCycle_lift`). Both directions then compare
potentials: a potential on the vertices of the minor is a potential on `G` that is constant on the
components of `K`, and the induced gain is read along it (`LinkMinorMap.switch_eq_one_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

universe uV uE uV' uE'

variable {V : Type uV} {E : Type uE} {V' : Type uV'} {E' : Type uE'} {Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

theorem Walk.edgeSet_cons {u v w : V} (e : E) (h : G.Joins e u v) (p : G.Walk v w) :
    (Walk.cons e h p).edgeSet = insert e p.edgeSet := by
  ext f
  simp [Walk.edgeSet, Walk.edges]

theorem Walk.edgeSet_append {u v w : V} (p : G.Walk u v) (q : G.Walk v w) :
    (p.append q).edgeSet = p.edgeSet ∪ q.edgeSet := by
  ext f
  simp [Walk.edgeSet, Walk.edges_append]

/-- Every vertex of a walk along `K` lies in the component of `K` of its start. -/
theorem Walk.class_eq_of_mem_support {K : Set E} {a b : V} (p : G.Walk a b)
    (hK : ∀ e ∈ p.edges, e ∈ K) {v : V} (hv : v ∈ p.support) :
    Quotient.mk (G.reachableSetoid K) v = Quotient.mk _ a := by
  induction p with
  | nil u =>
    simp only [Walk.support, List.mem_singleton] at hv
    rw [hv]
  | @cons x y z f h p ih =>
    simp only [Walk.support, List.mem_cons] at hv
    rcases hv with rfl | hv
    · rfl
    · have hf : f ∈ K := hK f (by simp [Walk.edges])
      rw [ih (fun e he => hK e (by simp [Walk.edges, he])) hv]
      exact (Quotient.sound ((reachableIn_refl K x).step hf h)).symm

/-- If a balanced cycle has identity gain off the edge `e`, then its gain at `e` is the identity. -/
theorem Walk.IsCycleWalk.eq_one_of_forall_ne [DecidableEq V] [Group Γ] {u : V} {W : G.Walk u u}
    (hW : W.IsCycleWalk) {δ : E → Γ} (hb : W.edgeSet ∈ G.balancedCycles δ) {e : E}
    (he : e ∈ W.edges) (h : ∀ f ∈ W.edges, f ≠ e → δ f = 1) : δ e = 1 := by
  obtain ⟨-, θ, hθ⟩ := OrientedGraph.mem_balancedCycles_iff.1 hb
  have heq : ∀ f ∈ W.edges, f ≠ e → θ (G.tail f) = θ (G.head f) := fun f hf hne => by
    have hsw := hθ f hf
    rwa [switch_apply, h f hf hne, mul_one, inv_mul_eq_one] at hsw
  have hsw := hθ e he
  rw [switch_apply, ← hW.eq_of_forall_ne he θ heq] at hsw
  have hconj := congrArg (fun x => θ (G.tail e) * x * (θ (G.tail e))⁻¹) hsw
  simpa [mul_assoc] using hconj

end OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {Ω' : BiasedGraph V' E'}

/-- An edge of the minor joins, in `G`, vertices of the classes of its ends. -/
theorem LinkMinorMap.exists_ends (m : Ω.LinkMinorMap Ω') {e' : E'} {x z : V'}
    (h : Ω'.graph.Joins e' x z) :
    ∃ s t, Ω.graph.Joins (m.edge e') s t ∧
      Quotient.mk (Ω.graph.reachableSetoid m.contractSet) s = m.vertex x ∧
        Quotient.mk (Ω.graph.reachableSetoid m.contractSet) t = m.vertex z := by
  have hj : (Ω.graph.contract m.contractSet).Joins (m.edge e') (m.vertex x) (m.vertex z) := by
    rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · exact m.joins e'
    · exact (m.joins e').symm
  rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · exact ⟨_, _, Or.inl ⟨rfl, rfl⟩, h₁, h₂⟩
  · exact ⟨_, _, Or.inr ⟨rfl, rfl⟩, h₂, h₁⟩

/-- A path of the minor lifts to a path of `G` between any two vertices of the classes of its ends,
adding only edges of `K`. -/
theorem LinkMinorMap.exists_lift_path (m : Ω.LinkMinorMap Ω') {x y : V'} (p : Ω'.graph.Walk x y)
    (hp : p.IsPath) {a b : V}
    (ha : Quotient.mk (Ω.graph.reachableSetoid m.contractSet) a = m.vertex x)
    (hb : Quotient.mk (Ω.graph.reachableSetoid m.contractSet) b = m.vertex y) :
    ∃ (q : Ω.graph.Walk a b) (K₁ : Set E), q.IsPath ∧ K₁ ⊆ m.contractSet ∧
      q.edgeSet = (fun e' => (m.edge e' : E)) '' p.edgeSet ∪ K₁ ∧
        ∀ v ∈ q.support, ∃ z ∈ p.support,
          Quotient.mk (Ω.graph.reachableSetoid m.contractSet) v = m.vertex z := by
  induction p generalizing a with
  | nil x =>
    obtain ⟨r, hr⟩ : Ω.graph.ReachableIn m.contractSet a b := Quotient.exact (ha.trans hb.symm)
    obtain ⟨q, hq, hsub⟩ := r.exists_isPath
    have hqK : ∀ f ∈ q.edges, f ∈ m.contractSet := fun f hf => hr f (hsub f hf)
    refine ⟨q, q.edgeSet, hq, hqK, ?_, fun v hv => ⟨x, by simp [Walk.support], ?_⟩⟩
    · simp [Walk.edgeSet, Walk.edges]
    · rw [q.class_eq_of_mem_support hqK hv, ha]
  | @cons x z y e' h p ih =>
    obtain ⟨hx, hp'⟩ := List.nodup_cons.1 (show (x :: p.support).Nodup from hp)
    obtain ⟨s, t, hj, hs, ht⟩ := m.exists_ends h
    obtain ⟨r₀, hr₀⟩ : Ω.graph.ReachableIn m.contractSet a s := Quotient.exact (ha.trans hs.symm)
    obtain ⟨r, hr, hsub⟩ := r₀.exists_isPath
    have hrK : ∀ f ∈ r.edges, f ∈ m.contractSet := fun f hf => hr₀ f (hsub f hf)
    have hrcls : ∀ v ∈ r.support,
        Quotient.mk (Ω.graph.reachableSetoid m.contractSet) v = m.vertex x := fun v hv => by
      rw [r.class_eq_of_mem_support hrK hv, ha]
    obtain ⟨q, K₁, hq, hK₁, hedges, hsupp⟩ := ih hp' ht hb
    refine ⟨r.append (.cons (m.edge e') hj q), r.edgeSet ∪ K₁, ?_, Set.union_subset hrK hK₁, ?_, ?_⟩
    · change (r.append (.cons (m.edge e') hj q)).support.Nodup
      rw [Walk.support_append]
      refine List.nodup_append.2 ⟨hr, hq, fun u hu w hw huw => ?_⟩
      subst huw
      obtain ⟨w, hw', hcls⟩ := hsupp u hw
      have hxw : x = w := m.vertex_injective ((hrcls u hu).symm.trans hcls)
      exact hx (by rw [hxw]; exact hw')
    · rw [Walk.edgeSet_append, Walk.edgeSet_cons, Walk.edgeSet_cons, hedges, Set.image_insert_eq]
      ext f
      simp only [Set.mem_union, Set.mem_insert_iff]
      tauto
    · intro v hv
      rw [Walk.support_append] at hv
      rcases List.mem_append.1 hv with hv | hv
      · exact ⟨x, by simp [Walk.support], hrcls v hv⟩
      · obtain ⟨w, hw, hcls⟩ := hsupp v hv
        exact ⟨w, by simp [Walk.support, hw], hcls⟩

/-- A cycle of the minor, together with some edges of `K`, is a cycle of `G`. -/
theorem LinkMinorMap.exists_isCycle_lift (m : Ω.LinkMinorMap Ω') {C' : Set E'}
    (hC' : Ω'.graph.IsCycle C') :
    ∃ K₁ ⊆ m.contractSet, Ω.graph.IsCycle ((fun e' => (m.edge e' : E)) '' C' ∪ K₁) := by
  obtain ⟨u, W, hW, rfl⟩ := hC'
  cases W with
  | nil => exact absurd rfl hW.edges_ne_nil
  | cons e' h p =>
    have hp : p.IsPath := hW.support_tail_nodup
    have he' : e' ∉ p.edges := (List.nodup_cons.1 hW.edges_nodup).1
    obtain ⟨s, t, hj, hs, ht⟩ := m.exists_ends h
    obtain ⟨q, K₁, hq, hK₁, hedges, -⟩ := m.exists_lift_path p hp ht hs
    have hnot : (m.edge e' : E) ∉ q.edges := fun hmem => by
      have hmem' : (m.edge e' : E) ∈ (fun g => (m.edge g : E)) '' p.edgeSet ∪ K₁ := by
        rw [← hedges]
        exact hmem
      rcases hmem' with ⟨g, hg, hge⟩ | hK
      · have hge' : g = e' := m.edge_injective (Subtype.ext hge)
        exact he' (hge' ▸ (hg : g ∈ p.edges))
      · exact (m.edge e').2 (hK₁ hK)
    refine ⟨K₁, hK₁, s, .cons (m.edge e') hj q, ⟨by simp [Walk.edges],
      List.nodup_cons.2 ⟨hnot, hq.edges_nodup⟩, hq⟩, ?_⟩
    rw [Walk.edgeSet_cons, Walk.edgeSet_cons, hedges, Set.image_insert_eq, Set.insert_union]

private theorem switch_aux {X : Type*} [Group Γ] (ζ : X → Γ) {a b t h : X} {i d : Γ}
    (hj : (t = a ∧ h = b) ∨ (t = b ∧ h = a)) (hc : (t = a ∧ i = d) ∨ (h = a ∧ i = d⁻¹)) :
    (ζ a)⁻¹ * i * ζ b = 1 ↔ (ζ t)⁻¹ * d * ζ h = 1 := by
  rcases hc with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · obtain rfl : h = b := by
      rcases hj with ⟨-, h₂⟩ | ⟨h₁, h₂⟩
      · exact h₂
      · exact h₂.trans h₁
    exact Iff.rfl
  · obtain rfl : t = b := by
      rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, -⟩
      · exact h₁.trans h₂
      · exact h₁
    have hinv : (ζ h)⁻¹ * d⁻¹ * ζ t = ((ζ t)⁻¹ * d * ζ h)⁻¹ := by simp [mul_assoc]
    rw [hinv, inv_eq_one]

/-- A potential on the minor, read through the components of `K`, is the identity on the same
edges. -/
theorem LinkMinorMap.switch_eq_one_iff [Group Γ] (m : Ω.LinkMinorMap Ω') {δ : E → Γ} {ι : E' → Γ}
    (hι : ∀ e', ((Ω.graph.contract m.contractSet).tail (m.edge e') = m.vertex (Ω'.graph.tail e') ∧
        ι e' = δ (m.edge e')) ∨
      ((Ω.graph.contract m.contractSet).head (m.edge e') = m.vertex (Ω'.graph.tail e') ∧
        ι e' = (δ (m.edge e'))⁻¹))
    (ζ : Quotient (Ω.graph.reachableSetoid m.contractSet) → Γ) (e' : E') :
    switch Ω'.graph (fun x => ζ (m.vertex x)) ι e' = 1 ↔
      switch Ω.graph (fun v => ζ (Quotient.mk _ v)) δ (m.edge e') = 1 :=
  switch_aux ζ (m.joins e') (hι e')

end BiasedGraph

end Transformer.GainMatroid
