module

public import Transformer.GainMatroid.Section2_ContractCycles

/-!
# Walks along maps of vertices and edges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1 and §2.3, take minors of biased graphs up to
isomorphism. `BiasedGraph.LinkMinorMap` embeds a link minor in `G / K` by injective maps of its
vertices and edges under which each edge joins the images of its ends, in either direction. Here
such maps `ν` of the vertices and `ι` of the edges of `H` into any `G` are considered.

Such maps carry walks to walks (`Walk.exists_map_both`), and a walk along the image of `ι` from the
image of a vertex lifts back (`Walk.exists_lift_both`). Hence the cycles (`IsCycle.map_both`,
`IsCycle.of_map_both`), the vertices of sets of edges (`verts_map_both`) and reachability
(`reachableIn_map_both_iff`) correspond.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V W E F : Type*} {H : OrientedGraph V E} {G : OrientedGraph W F} {ν : V → W} {ι : E → F}

/-- An edge of `H` joining `a` and `b` is carried to an edge of `G` joining `ν a` and `ν b`. -/
theorem Joins.map_both (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {e : E} {a b : V}
    (h : H.Joins e a b) : G.Joins (ι e) (ν a) (ν b) := by
  rcases h with ⟨ha, hb⟩ | ⟨ha, hb⟩
  · rw [← ha, ← hb]
    exact hJ e
  · rw [← ha, ← hb]
    exact (hJ e).symm

/-- If `ι e` joins `ν x` and `c`, then `e` joins `x` and a vertex carried to `c`. -/
theorem Joins.exists_lift_both (hν : Function.Injective ν)
    (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {e : E} {x : V} {c : W}
    (h : G.Joins (ι e) (ν x) c) : ∃ y, H.Joins e x y ∧ ν y = c := by
  rcases h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rcases hJ e with ⟨h₃, h₄⟩ | ⟨h₃, h₄⟩
  · exact ⟨H.head e, Or.inl ⟨hν (h₃.symm.trans h₁), rfl⟩, h₄.symm.trans h₂⟩
  · exact ⟨H.tail e, Or.inr ⟨rfl, hν (h₃.symm.trans h₁)⟩, h₄.symm.trans h₂⟩
  · exact ⟨H.tail e, Or.inr ⟨rfl, hν (h₄.symm.trans h₂)⟩, h₃.symm.trans h₁⟩
  · exact ⟨H.head e, Or.inl ⟨hν (h₄.symm.trans h₂), rfl⟩, h₃.symm.trans h₁⟩

/-- The maps carry a walk of `H` to a walk of `G`. -/
theorem Walk.exists_map_both (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {x y : V}
    (p : H.Walk x y) :
    ∃ q : G.Walk (ν x) (ν y), q.edges = p.edges.map ι ∧ q.support = p.support.map ν := by
  induction p with
  | nil x => exact ⟨.nil (ν x), rfl, rfl⟩
  | @cons a b c e he p ih =>
    obtain ⟨q, hq₁, hq₂⟩ := ih
    exact ⟨.cons (ι e) (he.map_both hJ) q, by simp [Walk.edges, hq₁], by simp [Walk.support, hq₂]⟩

/-- A walk of `G` from `ν x` along edges in the range of `ι` lifts to a walk of `H` from `x`. -/
theorem Walk.exists_lift_both (hν : Function.Injective ν)
    (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {a b : W} (q : G.Walk a b)
    (hq : ∀ f ∈ q.edges, f ∈ Set.range ι) {x : V} (hx : ν x = a) :
    ∃ (y : V) (p : H.Walk x y), ν y = b ∧ p.edges.map ι = q.edges ∧
      p.support.map ν = q.support := by
  induction q generalizing x with
  | nil a => exact ⟨x, .nil x, hx, rfl, by simp [Walk.support, hx]⟩
  | @cons s t r f hf q ih =>
    obtain ⟨e, rfl⟩ := hq f List.mem_cons_self
    subst hx
    obtain ⟨y', hj, hy'⟩ := hf.exists_lift_both hν hJ
    obtain ⟨y, p, hy, hp₁, hp₂⟩ := ih (fun f hf => hq f (List.mem_cons_of_mem _ hf)) hy'
    exact ⟨y, .cons e hj p, hy, by simp [Walk.edges, hp₁], by simp [Walk.support, hp₂]⟩

/-- The maps carry a cycle of `H` to a cycle of `G`. -/
theorem IsCycle.map_both (hν : Function.Injective ν) (hι : Function.Injective ι)
    (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {C : Set E} (hC : H.IsCycle C) :
    G.IsCycle (ι '' C) := by
  obtain ⟨a, P, hP, rfl⟩ := hC
  obtain ⟨q, hq₁, hq₂⟩ := P.exists_map_both hJ
  refine ⟨ν a, q, ⟨?_, ?_, ?_⟩, ?_⟩
  · rw [hq₁, Ne, List.map_eq_nil_iff]
    exact hP.edges_ne_nil
  · rw [hq₁]
    exact hP.edges_nodup.map hι
  · rw [hq₂, ← List.map_tail]
    exact hP.support_tail_nodup.map hν
  · ext f
    simp [Walk.edgeSet, hq₁]

/-- A set of edges of `H` whose image is a cycle of `G` is a cycle of `H`. -/
theorem IsCycle.of_map_both (hν : Function.Injective ν) (hι : Function.Injective ι)
    (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {C : Set E}
    (hC : G.IsCycle (ι '' C)) : H.IsCycle C := by
  obtain ⟨a, q, hq, hqC⟩ := hC
  have hrange : ∀ f ∈ q.edges, f ∈ Set.range ι := fun f hf => by
    have hf' : f ∈ ι '' C := by
      rw [← hqC]
      exact hf
    obtain ⟨e, -, rfl⟩ := hf'
    exact ⟨e, rfl⟩
  obtain ⟨x, hx⟩ : ∃ x, ν x = a := by
    cases q with
    | nil => exact absurd rfl hq.edges_ne_nil
    | cons f hf q' =>
      obtain ⟨e, rfl⟩ := hrange f List.mem_cons_self
      rcases hf.tail_or_head with h | h <;> rcases hJ e with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · exact ⟨H.tail e, h₁.symm.trans h⟩
      · exact ⟨H.head e, h₁.symm.trans h⟩
      · exact ⟨H.head e, h₂.symm.trans h⟩
      · exact ⟨H.tail e, h₂.symm.trans h⟩
  obtain ⟨y, p, hy, hp₁, hp₂⟩ := q.exists_lift_both hν hJ hrange hx
  obtain rfl : y = x := hν (hy.trans hx.symm)
  refine ⟨_, p, ⟨?_, ?_, ?_⟩, ?_⟩
  · intro h
    apply hq.edges_ne_nil
    rw [← hp₁, h, List.map_nil]
  · refine List.Nodup.of_map ι ?_
    rw [hp₁]
    exact hq.edges_nodup
  · refine List.Nodup.of_map ν ?_
    rw [List.map_tail, hp₂]
    exact hq.support_tail_nodup
  · ext e
    have he : ι e ∈ q.edgeSet ↔ ι e ∈ ι '' C := by rw [hqC]
    rw [hι.mem_set_image, Walk.edgeSet, Set.mem_ofPred_eq, ← hp₁,
      List.mem_map_of_injective hι] at he
    exact he

/-- The vertices of the image of a set of edges are the images of its vertices. -/
theorem verts_map_both (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {C : Set E} :
    G.verts (ι '' C) = ν '' H.verts C := by
  ext z
  simp only [mem_verts, Set.mem_image]
  constructor
  · rintro ⟨_, ⟨e, he, rfl⟩, hz⟩
    rcases hJ e with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rcases hz with hz | hz
    · exact ⟨H.tail e, ⟨e, he, Or.inl rfl⟩, h₁.symm.trans hz⟩
    · exact ⟨H.head e, ⟨e, he, Or.inr rfl⟩, h₂.symm.trans hz⟩
    · exact ⟨H.head e, ⟨e, he, Or.inr rfl⟩, h₁.symm.trans hz⟩
    · exact ⟨H.tail e, ⟨e, he, Or.inl rfl⟩, h₂.symm.trans hz⟩
  · rintro ⟨x, ⟨e, he, hx⟩, rfl⟩
    refine ⟨ι e, ⟨e, he, rfl⟩, ?_⟩
    rcases hJ e with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rcases hx with rfl | rfl
    · exact Or.inl h₁
    · exact Or.inr h₂
    · exact Or.inr h₂
    · exact Or.inl h₁

/-- Two vertices of `H` are linked along `I` exactly when their images are linked along the image
of `I`. -/
theorem reachableIn_map_both_iff (hν : Function.Injective ν) (hι : Function.Injective ι)
    (hJ : ∀ e, G.Joins (ι e) (ν (H.tail e)) (ν (H.head e))) {I : Set E} {x y : V} :
    H.ReachableIn I x y ↔ G.ReachableIn (ι '' I) (ν x) (ν y) := by
  constructor
  · rintro ⟨p, hp⟩
    obtain ⟨q, hq, -⟩ := p.exists_map_both hJ
    refine ⟨q, fun f hf => ?_⟩
    rw [hq, List.mem_map] at hf
    obtain ⟨e, he, rfl⟩ := hf
    exact ⟨e, hp e he, rfl⟩
  · rintro ⟨q, hq⟩
    obtain ⟨y', p, hy', hp, -⟩ :=
      q.exists_lift_both hν hJ (fun f hf => Set.image_subset_range _ _ (hq f hf)) rfl
    obtain rfl := hν hy'
    refine ⟨p, fun e he => hι.mem_set_image.1 (hq _ ?_)⟩
    rw [← hp]
    exact List.mem_map_of_mem he

example := Joins.map_both (H := pumpkin) (G := pumpkin) (ν := id) (ι := id)
  (fun _ => Or.inl ⟨rfl, rfl⟩) (e := 0) (Or.inl ⟨rfl, rfl⟩)

example := Joins.exists_lift_both (H := pumpkin) (G := pumpkin) (ν := id) (ι := id)
  Function.injective_id (fun _ => Or.inl ⟨rfl, rfl⟩) (e := 0) (x := 0) (Or.inl ⟨rfl, rfl⟩)

example := (pumpkinWalk 0).exists_map_both (G := pumpkin) (ν := id) (ι := id)
  fun _ => Or.inl ⟨rfl, rfl⟩

example := (pumpkinWalk 0).exists_lift_both (H := pumpkin) (ν := id) (ι := id)
  Function.injective_id (fun _ => Or.inl ⟨rfl, rfl⟩) (fun f _ => ⟨f, rfl⟩) (x := 0) rfl

example := pumpkin_pair_isCycle.map_both (G := pumpkin) (ν := id) (ι := id)
  Function.injective_id Function.injective_id fun _ => Or.inl ⟨rfl, rfl⟩

example := IsCycle.of_map_both (H := pumpkin) (G := pumpkin) (ν := id) (ι := id)
  (C := {0, 1}) Function.injective_id Function.injective_id (fun _ => Or.inl ⟨rfl, rfl⟩)
  (by rw [Set.image_id]; exact pumpkin_pair_isCycle)

example := verts_map_both (H := pumpkin) (G := pumpkin) (ν := id) (ι := id)
  (fun _ => Or.inl ⟨rfl, rfl⟩) (C := {0})

example := reachableIn_map_both_iff (H := pumpkin) (G := pumpkin) (ν := id) (ι := id)
  Function.injective_id Function.injective_id (fun _ => Or.inl ⟨rfl, rfl⟩) (I := {0})
  (x := 0) (y := 1)

end OrientedGraph

end Transformer.GainMatroid
