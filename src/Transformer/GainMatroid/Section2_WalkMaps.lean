module

public import Transformer.GainMatroid.Section2_CyclePaths

/-!
# Walks along a vertex map

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, compare the circuits of a biased graph with
the cycles of a graph on more vertices: in the special case, `H` has a new vertex `x`. A map `ι` of
the vertices carries a walk of `G` to a walk of `H` when each of its edges has as ends in `H` the
images of its ends in `G` (`Walk.exists_map_vertex`). With `ι` injective, a walk of `H` from an
image lifts back (`Walk.exists_lift`), so cycles go both ways (`IsCycle.map_vertex`,
`IsCycle.of_map_vertex`). A vertex of a walk other than its start is an end of one of its edges
(`Walk.exists_mem_edges_of_mem_support`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V W E : Type*} {G : OrientedGraph V E} {H : OrientedGraph W E}

/-- A vertex of a walk other than its start is an end of one of its edges. -/
theorem Walk.exists_mem_edges_of_mem_support {x y w : V} (p : G.Walk x y) (hw : w ∈ p.support)
    (hwx : w ≠ x) : ∃ g ∈ p.edges, G.tail g = w ∨ G.head g = w := by
  induction p with
  | nil a => exact absurd (List.mem_singleton.1 hw) hwx
  | @cons a b c g hg p ih =>
    rcases List.mem_cons.1 hw with h | h
    · exact absurd h hwx
    · by_cases hwb : w = b
      · refine ⟨g, List.mem_cons_self, ?_⟩
        rw [hwb]
        exact hg.symm.tail_or_head
      · obtain ⟨g', hg', h'⟩ := ih h hwb
        exact ⟨g', List.mem_cons_of_mem _ hg', h'⟩

/-- A walk of `G` whose edges have as ends in `H` the images of their ends under `ι` is a walk of
`H`. -/
theorem Walk.exists_map_vertex (ι : V → W) {x y : V} (p : G.Walk x y)
    (hp : ∀ g ∈ p.edges, H.tail g = ι (G.tail g) ∧ H.head g = ι (G.head g)) :
    ∃ q : H.Walk (ι x) (ι y), q.edges = p.edges ∧ q.support = p.support.map ι := by
  induction p with
  | nil x => exact ⟨.nil (ι x), rfl, rfl⟩
  | @cons a b c g hg p ih =>
    obtain ⟨q, hq₁, hq₂⟩ := ih fun g' hg' => hp g' (List.mem_cons_of_mem _ hg')
    obtain ⟨h₁, h₂⟩ := hp g List.mem_cons_self
    have hj : H.Joins g (ι a) (ι b) := by
      rcases hg with ⟨ha, hb⟩ | ⟨ha, hb⟩
      · exact Or.inl ⟨h₁.trans (congrArg ι ha), h₂.trans (congrArg ι hb)⟩
      · exact Or.inr ⟨h₁.trans (congrArg ι ha), h₂.trans (congrArg ι hb)⟩
    exact ⟨.cons g hj q, by simp [Walk.edges, hq₁], by simp [Walk.support, hq₂]⟩

/-- With `ι` injective, a walk of `H` from `ι x` whose edges have as ends the images of their ends
in `G` lifts to a walk of `G` from `x`. -/
theorem Walk.exists_lift {ι : V → W} (hι : Function.Injective ι) {a b : W} (q : H.Walk a b)
    (hq : ∀ g ∈ q.edges, H.tail g = ι (G.tail g) ∧ H.head g = ι (G.head g)) {x : V}
    (hx : ι x = a) :
    ∃ (y : V) (p : G.Walk x y), ι y = b ∧ p.edges = q.edges ∧ p.support.map ι = q.support := by
  induction q generalizing x with
  | nil a => exact ⟨x, .nil x, hx, rfl, by simp [Walk.support, hx]⟩
  | @cons s t r g hg q ih =>
    obtain ⟨h₁, h₂⟩ := hq g List.mem_cons_self
    obtain ⟨c, hc, hj⟩ : ∃ c, ι c = t ∧ G.Joins g x c := by
      rcases hg with ⟨ht, hh⟩ | ⟨ht, hh⟩
      · exact ⟨G.head g, h₂.symm.trans hh, Or.inl ⟨hι (h₁.symm.trans (ht.trans hx.symm)), rfl⟩⟩
      · exact ⟨G.tail g, h₁.symm.trans ht, Or.inr ⟨rfl, hι (h₂.symm.trans (hh.trans hx.symm))⟩⟩
    obtain ⟨y, p, hy, hp₁, hp₂⟩ := ih (fun g' hg' => hq g' (List.mem_cons_of_mem _ hg')) hc
    exact ⟨y, .cons g hj p, hy, by simp [Walk.edges, hp₁], by simp [Walk.support, hp₂, hx]⟩

/-- A cycle of `G` whose edges have as ends in `H` the images of their ends under an injective `ι`
is a cycle of `H`. -/
theorem IsCycle.map_vertex {ι : V → W} (hι : Function.Injective ι) {C : Set E} (hC : G.IsCycle C)
    (h : ∀ g ∈ C, H.tail g = ι (G.tail g) ∧ H.head g = ι (G.head g)) : H.IsCycle C := by
  obtain ⟨a, P, hP, rfl⟩ := hC
  obtain ⟨q, hq₁, hq₂⟩ := P.exists_map_vertex ι h
  refine ⟨ι a, q, ⟨?_, ?_, ?_⟩, ?_⟩
  · rw [hq₁]
    exact hP.edges_ne_nil
  · rw [hq₁]
    exact hP.edges_nodup
  · rw [hq₂, ← List.map_tail]
    exact List.Nodup.map hι hP.support_tail_nodup
  · unfold Walk.edgeSet
    rw [hq₁]

/-- A cycle of `H` whose edges have as ends the images under an injective `ι` of their ends in `G`
is a cycle of `G`. -/
theorem IsCycle.of_map_vertex {ι : V → W} (hι : Function.Injective ι) {C : Set E}
    (hC : H.IsCycle C) (h : ∀ g ∈ C, H.tail g = ι (G.tail g) ∧ H.head g = ι (G.head g)) :
    G.IsCycle C := by
  obtain ⟨a, q, hq, rfl⟩ := hC
  obtain ⟨x, hx⟩ : ∃ x, ι x = a := by
    cases q with
    | nil => exact absurd rfl hq.edges_ne_nil
    | cons g hg q' =>
      obtain ⟨h₁, h₂⟩ := h g List.mem_cons_self
      rcases hg with ⟨ht, -⟩ | ⟨-, hh⟩
      · exact ⟨G.tail g, h₁.symm.trans ht⟩
      · exact ⟨G.head g, h₂.symm.trans hh⟩
  obtain ⟨y, p, hy, hp₁, hp₂⟩ := q.exists_lift hι h hx
  obtain rfl : y = x := hι (hy.trans hx.symm)
  refine ⟨_, p, ⟨?_, ?_, ?_⟩, ?_⟩
  · rw [hp₁]
    exact hq.edges_ne_nil
  · rw [hp₁]
    exact hq.edges_nodup
  · refine List.Nodup.of_map ι ?_
    rw [List.map_tail, hp₂]
    exact hq.support_tail_nodup
  · unfold Walk.edgeSet
    rw [hp₁]

example := (pumpkinWalk 0).exists_mem_edges_of_mem_support (w := 1)
  (by simp [pumpkinWalk, Walk.support, pumpkin]) (by decide)

example := (pumpkinWalk 0).exists_map_vertex (H := pumpkin) id fun _ _ => ⟨rfl, rfl⟩

example := (pumpkinWalk 0).exists_lift (G := pumpkin) Function.injective_id
  (fun _ _ => ⟨rfl, rfl⟩) rfl

example := pumpkin_isTheta.isCycle.1.map_vertex (H := pumpkin) Function.injective_id
  fun _ _ => ⟨rfl, rfl⟩

example := pumpkin_isTheta.isCycle.1.of_map_vertex (G := pumpkin) Function.injective_id
  fun _ _ => ⟨rfl, rfl⟩

end OrientedGraph

end Transformer.GainMatroid
