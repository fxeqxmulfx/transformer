module

public import Transformer.GainMatroid.Section4_U24

/-!
# Frame and lift circuits on two vertices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2 and §4.1. On two vertices, with at most one
loop at each vertex, every tight handcuff, loose handcuff and theta has exactly three edges. This is
what identifies the matroids of `𝖴₂` and `𝖴₃` with `U_{2,4}` in `Section4_U24Matroid`.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

section Constructors

variable {V E : Type*} {G : OrientedGraph V E}

/-- Three parallel links form a theta. -/
theorem isThetaSet_of_three {x y : V} {e f g : E} (hxy : x ≠ y) (he : G.Joins e x y)
    (hf : G.Joins f x y) (hg : G.Joins g x y) (hef : e ≠ f) (heg : e ≠ g) (hfg : f ≠ g) :
    G.IsThetaSet {e, f, g} := by
  have hpath : ∀ {a} (ha : G.Joins a x y), (Walk.cons a ha (.nil y)).IsPath := fun _ => by
    simp [Walk.IsPath, Walk.support, hxy]
  have hopen : ∀ {a b} (ha : G.Joins a x y) (hb : G.Joins b x y), a ≠ b →
      Walk.Openly (.cons a ha (.nil y)) (.cons b hb (.nil y)) := fun _ _ hab =>
    ⟨fun c hc hc' => hab ((List.mem_singleton.1 hc).symm.trans (List.mem_singleton.1 hc')),
      fun z hz _ => by simpa [Walk.support] using hz⟩
  refine ⟨x, y, _, _, _, ⟨hxy, hpath he, hpath hf, hpath hg, hopen he hf hef, hopen he hg heg,
    hopen hf hg hfg⟩, ?_⟩
  ext a
  simp only [Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_singleton_iff, Set.mem_union,
    Set.mem_ofPred_eq, List.mem_singleton]
  tauto

/-- A loop at `u` meets only the vertex `u`. -/
theorem verts_singleton_of_loop {l : E} {u : V} (h₁ : G.tail l = u) (h₂ : G.head l = u) :
    G.verts {l} = {u} := by
  ext w
  simp only [mem_verts, Set.mem_singleton_iff, exists_eq_left, h₁, h₂, or_self]
  exact eq_comm

/-- A loop at `u` and two parallel links at `u` form a tight handcuff. -/
theorem isTightHandcuff_of_loop_pair {l e f : E} {u v : V} (h₁ : G.tail l = u) (h₂ : G.head l = u)
    (he : G.Joins e u v) (hf : G.Joins f u v) (hef : e ≠ f) (huv : u ≠ v) (hle : l ≠ e)
    (hlf : l ≠ f) : G.IsTightHandcuff {l, e, f} := by
  refine ⟨{l}, {e, f}, u, isCycle_singleton_of_loop (h₁.trans h₂.symm),
    isCycle_pair he hf hef huv, ?_, ?_, Set.singleton_union.symm⟩
  · simp [hle, hlf]
  · rw [verts_singleton_of_loop h₁ h₂]
    exact Set.inter_eq_left.2 (Set.singleton_subset_iff.2 (he.left_mem_verts_of_mem (by simp)))

/-- Loops at two distinct vertices and a link joining them form a loose handcuff. -/
theorem isLooseHandcuff_of_loops {c d p : E} {x y : V} (hc₁ : G.tail c = x) (hc₂ : G.head c = x)
    (hd₁ : G.tail d = y) (hd₂ : G.head d = y) (hp : G.Joins p x y) (hxy : x ≠ y) :
    G.IsLooseHandcuff {c, d, p} := by
  have hC := verts_singleton_of_loop hc₁ hc₂
  have hD := verts_singleton_of_loop hd₁ hd₂
  refine ⟨{c}, {d}, x, y, .cons p hp (.nil y), isCycle_singleton_of_loop (hc₁.trans hc₂.symm),
    isCycle_singleton_of_loop (hd₁.trans hd₂.symm), ?_, ?_, hC ▸ Set.mem_singleton x,
    hD ▸ Set.mem_singleton y, fun z _ hz => by rwa [hC] at hz, fun z _ hz => by rwa [hD] at hz, ?_⟩
  · rw [hC, hD]
    exact Set.disjoint_singleton.2 hxy
  · simp [Walk.IsPath, Walk.support, hxy]
  · ext a
    simp only [Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_singleton_iff, Set.mem_union,
      Set.mem_ofPred_eq, List.mem_singleton]
    tauto

end Constructors

variable {E : Type*} {G : OrientedGraph (Fin 2) E}

/-- A cycle of two edges on two vertices passes through both. -/
theorem IsCycle.verts_eq_univ_of_encard_two {C : Set E} (hC : G.IsCycle C) (h2 : C.encard = 2) :
    G.verts C = Set.univ := by
  obtain ⟨e, f, u, v, -, huv, he, -, rfl⟩ := hC.exists_pair h2
  refine Set.eq_univ_of_forall fun w => ?_
  have hw : w = u ∨ w = v := (by decide : ∀ u v w : Fin 2, u ≠ v → w = u ∨ w = v) u v w huv
  rcases hw with rfl | rfl
  · exact he.left_mem_verts_of_mem (by simp)
  · exact he.symm.left_mem_verts_of_mem (by simp)

/-- A path between the two vertices is one edge. -/
theorem Walk.IsPath.length_eq_one_of_two {x y : Fin 2} {p : G.Walk x y} (hp : p.IsPath)
    (hxy : x ≠ y) : p.edges.length = 1 := by
  have hle := (show p.support.Nodup from hp).length_le_card
  rw [Walk.length_support, Fintype.card_fin] at hle
  have hpos : p.edges.length ≠ 0 := fun h0 =>
    Walk.edges_ne_nil_of_ne hxy p (List.length_eq_zero_iff.1 h0)
  omega

/-- A vertex on a loop is the end of the loop. -/
theorem tail_eq_of_mem_verts_loop {V E : Type*} {G : OrientedGraph V E} {e : E} {v : V}
    (he : G.tail e = G.head e) (hv : v ∈ G.verts {e}) : G.tail e = v := by
  obtain ⟨g, hg, h | h⟩ := mem_verts.1 hv
  · rwa [Set.mem_singleton_iff.1 hg] at h
  · rw [Set.mem_singleton_iff.1 hg] at h
    exact he.trans h

/-- With at most one loop at each vertex, a tight handcuff on two vertices has three edges. -/
theorem IsTightHandcuff.encard_eq_three_of_two
    (hloops : ∀ e f : E, e ≠ f → G.tail e = G.head e → G.tail f = G.head f → G.tail e ≠ G.tail f)
    {X : Set E} (h : G.IsTightHandcuff X) : X.encard = 3 := by
  obtain ⟨C, D, v, hC, hD, hdisj, hv, rfl⟩ := h
  rw [Set.encard_union_eq hdisj]
  rcases hC.encard_eq_one_or_two with hc | hc <;> rcases hD.encard_eq_one_or_two with hd | hd
  · exfalso
    obtain ⟨e, he, rfl⟩ := hC.exists_loop hc
    obtain ⟨f, hf, rfl⟩ := hD.exists_loop hd
    have hef : e ≠ f := fun hc' => Set.disjoint_left.1 hdisj rfl (hc' ▸ rfl)
    have hvm : v ∈ G.verts {e} ∩ G.verts {f} := hv ▸ Set.mem_singleton v
    exact hloops e f hef he hf ((tail_eq_of_mem_verts_loop he hvm.1).trans
      (tail_eq_of_mem_verts_loop hf hvm.2).symm)
  · rw [hc, hd]
    norm_num
  · rw [hc, hd]
    norm_num
  · exfalso
    rw [hC.verts_eq_univ_of_encard_two hc, hD.verts_eq_univ_of_encard_two hd, Set.inter_self] at hv
    have h0 : (0 : Fin 2) = v := by
      have : (0 : Fin 2) ∈ ({v} : Set (Fin 2)) := hv ▸ Set.mem_univ _
      exact this
    have h1 : (1 : Fin 2) = v := by
      have : (1 : Fin 2) ∈ ({v} : Set (Fin 2)) := hv ▸ Set.mem_univ _
      exact this
    exact absurd (h0.trans h1.symm) (by decide)

/-- A loose handcuff on two vertices has three edges: two loops and one link. -/
theorem IsLooseHandcuff.encard_eq_three_of_two {X : Set E} (h : G.IsLooseHandcuff X) :
    X.encard = 3 := by
  obtain ⟨C, D, x, y, P, hC, hD, hdisj, hP, hx, hy, -, -, rfl⟩ := h
  have hxy : x ≠ y := fun hc => Set.disjoint_left.1 hdisj hx (hc ▸ hy)
  have hone : ∀ {A B : Set E} {b : Fin 2}, G.IsCycle A → Disjoint (G.verts A) (G.verts B) →
      b ∈ G.verts B → A.encard = 1 := by
    intro A B b hA hd hb
    rcases hA.encard_eq_one_or_two with h | h
    · exact h
    · rw [hA.verts_eq_univ_of_encard_two h] at hd
      exact absurd hb (Set.disjoint_left.1 hd (Set.mem_univ b))
  obtain ⟨c, hc, rfl⟩ := hC.exists_loop (hone hC hdisj hy)
  obtain ⟨d, hd, rfl⟩ := hD.exists_loop (hone hD hdisj.symm hx)
  obtain ⟨p, hp, hj⟩ := P.exists_edge_of_length_one (hP.length_eq_one_of_two hxy)
  have hlink : G.tail p ≠ G.head p := by
    rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
    · rw [h₁, h₂]
      exact hxy
    · rw [h₁, h₂]
      exact hxy.symm
  have hcd : c ≠ d := fun hcd => by
    subst hcd
    exact Set.disjoint_left.1 hdisj hx hx |>.elim
  have hcp : c ≠ p := fun h' => hlink (h' ▸ hc)
  have hdp : d ≠ p := fun h' => hlink (h' ▸ hd)
  refine Set.encard_eq_three.2 ⟨c, d, p, hcd, hcp, hdp, ?_⟩
  ext a
  simp only [Walk.edgeSet, hp, Set.mem_insert_iff, Set.mem_singleton_iff, Set.mem_union,
    Set.mem_ofPred_eq, List.mem_singleton]
  tauto

/-- A theta on two vertices has three edges. -/
theorem IsThetaSet.encard_eq_three_of_two {X : Set E} (h : G.IsThetaSet X) : X.encard = 3 := by
  obtain ⟨x, y, P, Q, R, hT, rfl⟩ := h
  have hd₁ : Disjoint P.edgeSet Q.edgeSet := Set.disjoint_left.2 fun e he => hT.openPQ.edges e he
  have hd₂ : Disjoint (P.edgeSet ∪ Q.edgeSet) R.edgeSet := by
    rw [Set.disjoint_union_left]
    exact ⟨Set.disjoint_left.2 fun e he => hT.openPR.edges e he,
      Set.disjoint_left.2 fun e he => hT.openQR.edges e he⟩
  rw [Set.encard_union_eq hd₂, Set.encard_union_eq hd₁,
    Walk.encard_edgeSet_of_nodup hT.pathP.edges_nodup,
    Walk.encard_edgeSet_of_nodup hT.pathQ.edges_nodup,
    Walk.encard_edgeSet_of_nodup hT.pathR.edges_nodup,
    hT.pathP.length_eq_one_of_two hT.ne, hT.pathQ.length_eq_one_of_two hT.ne,
    hT.pathR.length_eq_one_of_two hT.ne]
  norm_num

end OrientedGraph

end Transformer.GainMatroid
