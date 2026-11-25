module

public import Transformer.GainMatroid.Section2_BiasedExamples
public import Transformer.GainMatroid.Section2_Minors

/-!
# Cycles and forests of a deletion

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, delete edges of a biased graph: `G \ D` keeps
the vertices of `G` and the edges outside `D`, with their ends (`OrientedGraph.delete`).

A map `ι` from the edges of `H` to those of `G` keeping the ends carries each walk of `H` to a walk
of `G` (`Walk.exists_map_edge`), and a walk of `G` along edges in the range of `ι` lifts back
(`Walk.exists_lift_edge`). With `ι` injective, cycles go both ways (`IsCycle.map_edge`,
`IsCycle.of_map_edge`). Hence the cycles and the forests of `G \ D` are those of `G` avoiding `D`
(`isCycle_delete_iff`, `isForest_delete_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E E' : Type*} {G : OrientedGraph V E} {H : OrientedGraph V E'} {ι : E' → E}

/-- An edge map `ι` keeping the ends carries a walk of `H` to a walk of `G` through the same
vertices. -/
theorem Walk.exists_map_edge (hι : ∀ f, G.tail (ι f) = H.tail f ∧ G.head (ι f) = H.head f)
    {x y : V} (p : H.Walk x y) :
    ∃ q : G.Walk x y, q.edges = p.edges.map ι ∧ q.support = p.support := by
  induction p with
  | nil x => exact ⟨.nil x, rfl, rfl⟩
  | @cons a b c f hf p ih =>
    obtain ⟨q, hq₁, hq₂⟩ := ih
    obtain ⟨h₁, h₂⟩ := hι f
    have hj : G.Joins (ι f) a b := by
      rcases hf with ⟨ha, hb⟩ | ⟨ha, hb⟩
      · exact Or.inl ⟨h₁.trans ha, h₂.trans hb⟩
      · exact Or.inr ⟨h₁.trans ha, h₂.trans hb⟩
    exact ⟨.cons (ι f) hj q, by simp [Walk.edges, hq₁], by simp [Walk.support, hq₂]⟩

/-- A walk of `G` along edges in the range of an edge map `ι` keeping the ends lifts to a walk of
`H` through the same vertices. -/
theorem Walk.exists_lift_edge (hι : ∀ f, G.tail (ι f) = H.tail f ∧ G.head (ι f) = H.head f)
    {x y : V} (q : G.Walk x y) (hq : ∀ e ∈ q.edges, e ∈ Set.range ι) :
    ∃ p : H.Walk x y, p.edges.map ι = q.edges ∧ p.support = q.support := by
  induction q with
  | nil x => exact ⟨.nil x, rfl, rfl⟩
  | @cons a b c e he q ih =>
    obtain ⟨p, hp₁, hp₂⟩ := ih fun e' he' => hq e' (List.mem_cons_of_mem _ he')
    obtain ⟨f, rfl⟩ := hq e List.mem_cons_self
    obtain ⟨h₁, h₂⟩ := hι f
    have hj : H.Joins f a b := by
      rcases he with ⟨ha, hb⟩ | ⟨ha, hb⟩
      · exact Or.inl ⟨h₁.symm.trans ha, h₂.symm.trans hb⟩
      · exact Or.inr ⟨h₁.symm.trans ha, h₂.symm.trans hb⟩
    exact ⟨.cons f hj p, by simp [Walk.edges, hp₁], by simp [Walk.support, hp₂]⟩

/-- An injective edge map keeping the ends carries a cycle of `H` to a cycle of `G`. -/
theorem IsCycle.map_edge (hinj : Function.Injective ι)
    (hι : ∀ f, G.tail (ι f) = H.tail f ∧ G.head (ι f) = H.head f) {C : Set E'}
    (hC : H.IsCycle C) : G.IsCycle (ι '' C) := by
  obtain ⟨a, P, hP, rfl⟩ := hC
  obtain ⟨q, hq₁, hq₂⟩ := P.exists_map_edge hι
  refine ⟨a, q, ⟨?_, ?_, ?_⟩, ?_⟩
  · rw [hq₁, Ne, List.map_eq_nil_iff]
    exact hP.edges_ne_nil
  · rw [hq₁]
    exact hP.edges_nodup.map hinj
  · rw [hq₂]
    exact hP.support_tail_nodup
  · ext e
    simp [Walk.edgeSet, hq₁]

/-- For an injective edge map keeping the ends, a set of edges of `H` whose image is a cycle of `G`
is a cycle of `H`. -/
theorem IsCycle.of_map_edge (hinj : Function.Injective ι)
    (hι : ∀ f, G.tail (ι f) = H.tail f ∧ G.head (ι f) = H.head f) {C : Set E'}
    (hC : G.IsCycle (ι '' C)) : H.IsCycle C := by
  obtain ⟨a, q, hq, hqC⟩ := hC
  obtain ⟨p, hp₁, hp₂⟩ := q.exists_lift_edge hι fun e he => by
    have he' : e ∈ ι '' C := by
      rw [← hqC]
      exact he
    obtain ⟨f, -, rfl⟩ := he'
    exact ⟨f, rfl⟩
  refine ⟨a, p, ⟨?_, ?_, ?_⟩, ?_⟩
  · intro h
    apply hq.edges_ne_nil
    rw [← hp₁, h, List.map_nil]
  · refine List.Nodup.of_map ι ?_
    rw [hp₁]
    exact hq.edges_nodup
  · rw [hp₂]
    exact hq.support_tail_nodup
  · ext f
    have hf : ι f ∈ q.edgeSet ↔ ι f ∈ ι '' C := by rw [hqC]
    rw [hinj.mem_set_image, Walk.edgeSet, Set.mem_ofPred_eq, ← hp₁,
      List.mem_map_of_injective hinj] at hf
    exact hf

variable {D : Set E}

/-- The cycles of `G \ D` are the cycles of `G` avoiding `D`. -/
theorem isCycle_delete_iff {C : Set ↥Dᶜ} :
    (G.delete D).IsCycle C ↔ G.IsCycle (Subtype.val '' C) :=
  ⟨fun h => h.map_edge Subtype.val_injective fun _ => ⟨rfl, rfl⟩,
    fun h => h.of_map_edge Subtype.val_injective fun _ => ⟨rfl, rfl⟩⟩

/-- The forests of `G \ D` are the forests of `G` avoiding `D`. -/
theorem isForest_delete_iff {I : Set ↥Dᶜ} :
    (G.delete D).IsForest I ↔ G.IsForest (Subtype.val '' I) := by
  constructor
  · intro hI C hC hCI
    refine hI (Subtype.val ⁻¹' C) (isCycle_delete_iff.2 ?_) fun f hf => ?_
    · rwa [Set.image_preimage_eq_of_subset (hCI.trans (Set.image_subset_range _ _))]
    · exact Subtype.val_injective.mem_set_image.1 (hCI hf)
  · intro hI C hC hCI
    exact hI _ (isCycle_delete_iff.1 hC) (Set.image_mono hCI)

/-- The edge `0` of `pumpkin` after deleting the edge `2`. -/
def pumpkinDeleteZero : ↥(({2} : Set (Fin 3))ᶜ) := ⟨0, by simp⟩

/-- The cycle `{0, 1}` of `pumpkin` avoids its edge `2`. -/
theorem pumpkin_pair_subset_range :
    ({0, 1} : Set (Fin 3)) ⊆ Set.range (Subtype.val : ↥(({2} : Set (Fin 3))ᶜ) → Fin 3) := by
  rintro x (rfl | rfl)
  · exact ⟨pumpkinDeleteZero, rfl⟩
  · exact ⟨⟨1, by simp⟩, rfl⟩

theorem pumpkin_delete_isCycle :
    (pumpkin.delete {2}).IsCycle (Subtype.val ⁻¹' ({0, 1} : Set (Fin 3))) := by
  rw [isCycle_delete_iff, Set.image_preimage_eq_of_subset pumpkin_pair_subset_range]
  exact pumpkin_pair_isCycle

example := Walk.exists_map_edge (G := pumpkin) (H := pumpkin.delete {2}) (ι := Subtype.val)
  (fun _ => ⟨rfl, rfl⟩) (.cons pumpkinDeleteZero (Or.inl ⟨rfl, rfl⟩) (.nil 1))

example := (pumpkinWalk 0).exists_lift_edge (H := pumpkin.delete {2}) (ι := Subtype.val)
  (fun _ => ⟨rfl, rfl⟩) fun e he => ⟨⟨e, by simp_all [pumpkinWalk, Walk.edges]⟩, rfl⟩

example := pumpkin_delete_isCycle.map_edge (G := pumpkin) Subtype.val_injective
  fun _ => ⟨rfl, rfl⟩

example := IsCycle.of_map_edge (G := pumpkin) (H := pumpkin.delete {2})
  (C := Subtype.val ⁻¹' ({0, 1} : Set (Fin 3))) Subtype.val_injective (fun _ => ⟨rfl, rfl⟩)
  (by rw [Set.image_preimage_eq_of_subset pumpkin_pair_subset_range]; exact pumpkin_pair_isCycle)

example : ¬(pumpkin.delete {2}).IsForest (Subtype.val ⁻¹' ({0, 1} : Set (Fin 3))) :=
  fun h => isForest_delete_iff.1 h _ pumpkin_pair_isCycle
    (Set.image_preimage_eq_of_subset pumpkin_pair_subset_range).ge

end OrientedGraph

end Transformer.GainMatroid
