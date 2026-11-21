module

public import Transformer.GainMatroid.Section2_Matroids
public import Transformer.GainMatroid.Section3_Classes

/-!
# Frame circuits on three vertices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2. On a loopless graph with three vertices and no
three parallel links, every tight handcuff and every theta has exactly four edges, and there is no
loose handcuff. This is what the counterexample to Theorem 1 (i) in `Section1_Theorem1` uses to
identify its frame matroid with `U_{3,5}`.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

theorem Walk.length_support {u v : V} (p : G.Walk u v) :
    p.support.length = p.edges.length + 1 := by
  induction p with
  | nil => simp [Walk.support, Walk.edges]
  | cons e h p ih => simp [Walk.support, Walk.edges, ih]

theorem Walk.encard_edgeSet_of_nodup {u v : V} {p : G.Walk u v} (h : p.edges.Nodup) :
    p.edgeSet.encard = p.edges.length := by
  classical
  have hs : p.edgeSet = ↑p.edges.toFinset := by
    ext e
    simp [Walk.edgeSet]
  rw [hs, Set.encard_coe_eq_coe_finsetCard, List.toFinset_card_of_nodup h]

/-- A path of two edges has a vertex other than its ends. -/
theorem Walk.IsPath.exists_interior {x y : V} {p : G.Walk x y} (hp : p.IsPath)
    (h : p.edges.length = 2) : ∃ m ∈ p.support, m ≠ x ∧ m ≠ y := by
  cases p with
  | nil => simp [Walk.edges] at h
  | cons e he q =>
    rename_i v
    cases q with
    | nil => simp [Walk.edges] at h
    | cons f hf r =>
      have hnd := hp
      simp only [Walk.IsPath, Walk.support, List.nodup_cons, List.mem_cons, not_or] at hnd
      exact ⟨v, by simp [Walk.support], fun hvx => hnd.1.1 hvx.symm,
        fun hvy => hnd.2.1 (hvy.symm ▸ Walk.end_mem_support r)⟩

/-- A path of one edge is that edge. -/
theorem Walk.exists_edge_of_length_one {x y : V} (p : G.Walk x y) (h : p.edges.length = 1) :
    ∃ e, p.edges = [e] ∧ G.Joins e x y := by
  cases p with
  | nil => simp [Walk.edges] at h
  | cons e he q =>
    cases q with
    | nil => exact ⟨e, rfl, he⟩
    | cons f hf r => simp [Walk.edges] at h

/-- A cycle of a loopless graph has two distinct vertices. -/
theorem IsCycle.exists_two_verts (hG : G.Loopless) {C : Set E} (hC : G.IsCycle C) :
    ∃ p q, p ≠ q ∧ p ∈ G.verts C ∧ q ∈ G.verts C := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
  have he' : e ∈ W.edgeSet := he
  exact ⟨G.tail e, G.head e, hG e, mem_verts.2 ⟨e, he', Or.inl rfl⟩,
    mem_verts.2 ⟨e, he', Or.inr rfl⟩⟩

theorem Joins.left_mem_verts_of_mem {C : Set E} {e : E} {u v : V} (h : G.Joins e u v)
    (he : e ∈ C) : u ∈ G.verts C := by
  rcases h with ⟨h₁, -⟩ | ⟨-, h₂⟩
  · exact mem_verts.2 ⟨e, he, Or.inl h₁⟩
  · exact mem_verts.2 ⟨e, he, Or.inr h₂⟩

/-- Two parallel links form a cycle. -/
theorem isCycle_pair {e f : E} {u v : V} (he : G.Joins e u v) (hf : G.Joins f u v) (hne : e ≠ f)
    (huv : u ≠ v) : G.IsCycle {e, f} :=
  ⟨u, pairWalk he hf, pairWalk_isCycleWalk he hf hne huv, by
    ext g
    simp [pairWalk, Walk.edgeSet, Walk.edges]⟩

/-- Two parallel links and a path of two edges around them form a theta. -/
theorem isThetaSet_of_parallel {x m y : V} {e f g h : E} (hxm : x ≠ m) (hxy : x ≠ y) (hmy : m ≠ y)
    (he : G.Joins e x y) (hf : G.Joins f x y) (hg : G.Joins g x m) (hh : G.Joins h m y)
    (hef : e ≠ f) (heg : e ≠ g) (heh : e ≠ h) (hfg : f ≠ g) (hfh : f ≠ h) :
    G.IsThetaSet {e, f, g, h} := by
  refine ⟨x, y, .cons e he (.nil y), .cons f hf (.nil y), .cons g hg (.cons h hh (.nil y)),
    ⟨hxy, ?_, ?_, ?_, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩, ?_⟩
  · simp [Walk.IsPath, Walk.support, hxy]
  · simp [Walk.IsPath, Walk.support, hxy]
  · simp [Walk.IsPath, Walk.support, hxm, hxy, hmy]
  · intro a ha hb
    simp only [Walk.edges, List.mem_singleton] at ha hb
    exact hef (ha.symm.trans hb)
  · intro z hz _
    simpa [Walk.support] using hz
  · intro a ha hb
    simp only [Walk.edges, List.mem_cons, List.not_mem_nil, or_false] at ha hb
    rcases hb with rfl | rfl
    · exact heg ha.symm
    · exact heh ha.symm
  · intro z hz _
    simpa [Walk.support] using hz
  · intro a ha hb
    simp only [Walk.edges, List.mem_cons, List.not_mem_nil, or_false] at ha hb
    rcases hb with rfl | rfl
    · exact hfg ha.symm
    · exact hfh ha.symm
  · intro z hz _
    simpa [Walk.support] using hz
  · ext a
    simp only [Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_singleton_iff, Set.mem_union,
      Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false]
    tauto

section Three

variable {E : Type*} {G : OrientedGraph (Fin 3) E}

/-- A cycle of a loopless graph on three vertices has two or three edges. -/
theorem IsCycle.encard_eq_two_or_three (hG : G.Loopless) {C : Set E} (hC : G.IsCycle C) :
    C.encard = 2 ∨ C.encard = 3 := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  rw [hW.encard_edgeSet]
  have hle : W.edges.length ≤ 3 := by
    have h := hW.support_tail_nodup.length_le_card
    simpa [Walk.length_support] using h
  have h2 : 2 ≤ W.edges.length := by
    cases W with
    | nil => exact absurd rfl hW.edges_ne_nil
    | cons e he p =>
      rename_i v
      have huv : u ≠ v := Joins.ne_of_loopless hG he
      cases p with
      | nil => exact absurd rfl huv
      | cons f hf q => simp [Walk.edges]
  have h : W.edges.length = 2 ∨ W.edges.length = 3 := by omega
  rcases h with h | h <;> simp [h]

/-- A cycle of three edges on three vertices passes through all of them. -/
theorem IsCycle.verts_eq_univ_of_three {C : Set E} (hC : G.IsCycle C) (h3 : C.encard = 3) :
    G.verts C = Set.univ := by
  obtain ⟨a, b, c, x, y, z, -, -, -, hxy, hxz, hyz, ha, hb, hc, rfl⟩ := hC.exists_triangle h3
  refine Set.eq_univ_of_forall fun w => ?_
  have hw : w = x ∨ w = y ∨ w = z :=
    (by decide : ∀ x y z w : Fin 3, x ≠ y → x ≠ z → y ≠ z → w = x ∨ w = y ∨ w = z)
      x y z w hxy hxz hyz
  rcases hw with rfl | rfl | rfl
  · exact ha.left_mem_verts_of_mem (by simp)
  · exact hb.left_mem_verts_of_mem (by simp)
  · exact hc.left_mem_verts_of_mem (by simp)

/-- A tight handcuff on three vertices has four edges: both of its cycles are 2-cycles. -/
theorem IsTightHandcuff.encard_eq_four (hG : G.Loopless) {X : Set E} (h : G.IsTightHandcuff X) :
    X.encard = 4 := by
  obtain ⟨C, D, v, hC, hD, hdisj, hv, rfl⟩ := h
  have key : ∀ {A B : Set E}, G.IsCycle A → G.IsCycle B → G.verts A ∩ G.verts B = {v} →
      A.encard = 2 := by
    intro A B hA hB hAB
    rcases hA.encard_eq_two_or_three hG with h2 | h3
    · exact h2
    · exfalso
      obtain ⟨p, q, hpq, hp, hq⟩ := hB.exists_two_verts hG
      have hAu := hA.verts_eq_univ_of_three h3
      have hp' : p ∈ G.verts A ∩ G.verts B := ⟨hAu ▸ Set.mem_univ p, hp⟩
      have hq' : q ∈ G.verts A ∩ G.verts B := ⟨hAu ▸ Set.mem_univ q, hq⟩
      rw [hAB] at hp' hq'
      exact hpq (hp'.trans hq'.symm)
  rw [Set.encard_union_eq hdisj, key hC hD hv, key hD hC (by rwa [Set.inter_comm])]
  norm_num

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem no_four_distinct : ∀ p q r s : Fin 3, p ≠ q → r ≠ s → p ≠ r → p ≠ s → q ≠ r →
    q ≠ s → False := by
  decide

/-- A loopless graph on three vertices has no loose handcuff: two vertex-disjoint cycles would need
four vertices. -/
theorem not_isLooseHandcuff (hG : G.Loopless) (X : Set E) : ¬G.IsLooseHandcuff X := by
  rintro ⟨C, D, x, y, P, hC, hD, hdisj, -⟩
  obtain ⟨p, q, hpq, hp, hq⟩ := hC.exists_two_verts hG
  obtain ⟨r, s, hrs, hr, hs⟩ := hD.exists_two_verts hG
  have hne : ∀ {a b : Fin 3}, a ∈ G.verts C → b ∈ G.verts D → a ≠ b :=
    fun ha hb hab => Set.disjoint_left.1 hdisj ha (hab ▸ hb)
  exact no_four_distinct p q r s hpq hrs (hne hp hr) (hne hp hs) (hne hq hr) (hne hq hs)

end Three

end OrientedGraph

end Transformer.GainMatroid
