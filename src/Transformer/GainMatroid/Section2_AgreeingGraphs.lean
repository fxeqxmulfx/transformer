module

public import Transformer.GainMatroid.Section2_RollUps

/-!
# Graphs that agree on a set of edges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, compare the frame circuits of a biased graph
with those of its roll-ups. A roll-up moves the ends of the rolled-up edges only, so the two graphs
agree on every other edge. Two graphs on the same vertices that give the edges of a set the same
ends (`AgreeOn`) have the same walks along those edges (`Walk.exists_of_agreeOn`, which may also
rename the ends of the walk by equal vertices), and the same cycles, vertices, tight and loose
handcuffs and thetas inside the set. A path that meets a set of edges only at one vertex has none
of its edges (`Walk.IsPath.notMem_of_forall`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G H : OrientedGraph V E}

/-- `G` and `H` give each edge of `X` the same ends. -/
def AgreeOn (G H : OrientedGraph V E) (X : Set E) : Prop :=
  ∀ g ∈ X, G.tail g = H.tail g ∧ G.head g = H.head g

theorem AgreeOn.symm {X : Set E} (h : G.AgreeOn H X) : H.AgreeOn G X :=
  fun g hg => ⟨(h g hg).1.symm, (h g hg).2.symm⟩

theorem AgreeOn.mono {X Y : Set E} (h : G.AgreeOn H X) (hYX : Y ⊆ X) : G.AgreeOn H Y :=
  fun g hg => h g (hYX hg)

theorem AgreeOn.joins {X : Set E} (h : G.AgreeOn H X) {g : E} (hg : g ∈ X) {a b : V}
    (hj : G.Joins g a b) : H.Joins g a b := by
  rw [Joins, ← (h g hg).1, ← (h g hg).2]
  exact hj

/-- `G` and `H` give a set on whose edges they agree the same vertices. -/
theorem AgreeOn.verts_eq {X : Set E} (h : G.AgreeOn H X) : G.verts X = H.verts X := by
  ext v
  rw [mem_verts, mem_verts]
  constructor
  · rintro ⟨g, hg, hv⟩
    exact ⟨g, hg, by rwa [← (h g hg).1, ← (h g hg).2]⟩
  · rintro ⟨g, hg, hv⟩
    exact ⟨g, hg, by rwa [(h g hg).1, (h g hg).2]⟩

/-- A walk of `G` along edges on which `G` and `H` agree is a walk of `H`, between the same ends
or equal ones. -/
theorem Walk.exists_of_agreeOn {x y x' y' : V} (p : G.Walk x y) (hx : x = x') (hy : y = y')
    (h : G.AgreeOn H p.edgeSet) :
    ∃ q : H.Walk x' y', q.edges = p.edges ∧ q.support = p.support := by
  subst hx hy
  induction p with
  | nil x => exact ⟨.nil x, rfl, rfl⟩
  | @cons a b c g hg p ih =>
    obtain ⟨q, hq₁, hq₂⟩ := ih fun g' hg' => h g' (List.mem_cons_of_mem _ hg')
    exact ⟨.cons g (h.joins List.mem_cons_self hg) q, by simp [Walk.edges, hq₁],
      by simp [Walk.support, hq₂]⟩

/-- A cycle of `G` on whose edges `G` and `H` agree is a cycle of `H`. -/
theorem IsCycle.of_agreeOn {C : Set E} (hC : G.IsCycle C) (h : G.AgreeOn H C) : H.IsCycle C := by
  obtain ⟨a, W, hW, rfl⟩ := hC
  obtain ⟨q, hq₁, hq₂⟩ := W.exists_of_agreeOn rfl rfl h
  refine ⟨a, q, ⟨?_, ?_, ?_⟩, ?_⟩
  · rw [hq₁]
    exact hW.edges_ne_nil
  · rw [hq₁]
    exact hW.edges_nodup
  · rw [hq₂]
    exact hW.support_tail_nodup
  · unfold Walk.edgeSet
    rw [hq₁]

theorem IsTightHandcuff.of_agreeOn {X : Set E} (hX : G.IsTightHandcuff X) (h : G.AgreeOn H X) :
    H.IsTightHandcuff X := by
  obtain ⟨C, D, v, hC, hD, hCD, hv, rfl⟩ := hX
  have hC' := h.mono Set.subset_union_left
  have hD' := h.mono Set.subset_union_right
  refine ⟨C, D, v, hC.of_agreeOn hC', hD.of_agreeOn hD', hCD, ?_, rfl⟩
  rwa [← hC'.verts_eq, ← hD'.verts_eq]

theorem IsLooseHandcuff.of_agreeOn {X : Set E} (hX : G.IsLooseHandcuff X) (h : G.AgreeOn H X) :
    H.IsLooseHandcuff X := by
  obtain ⟨C, D, x, y, P, hC, hD, hCD, hP, hx, hy, hPC, hPD, rfl⟩ := hX
  have hC' := h.mono (Set.subset_union_left.trans Set.subset_union_left)
  have hD' := h.mono (Set.subset_union_right.trans Set.subset_union_left)
  obtain ⟨q, hq₁, hq₂⟩ := P.exists_of_agreeOn rfl rfl (h.mono Set.subset_union_right)
  rw [hC'.verts_eq] at hCD hx hPC
  rw [hD'.verts_eq] at hCD hy hPD
  refine ⟨C, D, x, y, q, hC.of_agreeOn hC', hD.of_agreeOn hD', hCD, ?_, hx, hy,
    fun z hz => hPC z (hq₂ ▸ hz), fun z hz => hPD z (hq₂ ▸ hz), ?_⟩
  · unfold Walk.IsPath
    rw [hq₂]
    exact hP
  · unfold Walk.edgeSet
    rw [hq₁]

theorem IsThetaSet.of_agreeOn {X : Set E} (hX : G.IsThetaSet X) (h : G.AgreeOn H X) :
    H.IsThetaSet X := by
  obtain ⟨x, y, P, Q, R, hT, rfl⟩ := hX
  obtain ⟨P', Q', R', hT', hP, hQ, hR⟩ := hT.exists_of_agree (G := H)
    (fun g hg => h.symm g (Or.inl (Or.inl hg))) (fun g hg => h.symm g (Or.inl (Or.inr hg)))
    fun g hg => h.symm g (Or.inr hg)
  exact ⟨x, y, P', Q', R', hT', by rw [hP, hQ, hR]⟩

/-- An end of an edge of a walk lies on the walk. -/
theorem Walk.mem_support_of_mem_verts {x y z : V} {p : G.Walk x y}
    (hz : z ∈ G.verts p.edgeSet) : z ∈ p.support := by
  obtain ⟨g, hg, hgz⟩ := mem_verts.1 hz
  exact mem_support_of_mem_edges hg hgz

/-- A path that meets the vertices of `D` at one vertex at most has no edge of `D`: such an edge
would be a loop. -/
theorem Walk.IsPath.notMem_of_forall {x y w : V} {p : G.Walk x y} (hp : p.IsPath) {D : Set E}
    (h : ∀ z ∈ p.support, z ∈ G.verts D → z = w) {g : E} (hg : g ∈ p.edges) : g ∉ D :=
  fun hgD => hp.isLink_of_mem hg
    ((h _ (p.ends_mem_support hg).1 (mem_verts.2 ⟨g, hgD, Or.inl rfl⟩)).trans
      (h _ (p.ends_mem_support hg).2 (mem_verts.2 ⟨g, hgD, Or.inr rfl⟩)).symm)

example := AgreeOn.symm (G := pumpkin) (H := pumpkin) (X := ∅) fun _ h => h.elim
example := AgreeOn.mono (G := pumpkin) (H := pumpkin) (X := Set.univ) (fun _ _ => ⟨rfl, rfl⟩)
  (Set.empty_subset _)
example := AgreeOn.joins (G := pumpkin) (H := pumpkin) (X := Set.univ) (fun _ _ => ⟨rfl, rfl⟩)
  (Set.mem_univ 0) (a := 0) (b := 1) (Or.inl ⟨rfl, rfl⟩)
example := AgreeOn.verts_eq (G := pumpkin) (H := pumpkin) (X := Set.univ) fun _ _ => ⟨rfl, rfl⟩
example := (pumpkinWalk 0).exists_of_agreeOn (H := pumpkin) rfl rfl fun _ _ => ⟨rfl, rfl⟩
example := pumpkin_isTheta.isCycle.1.of_agreeOn (H := pumpkin) fun _ _ => ⟨rfl, rfl⟩
example := (isTightHandcuff_of_loop_pair (G := U2) (l := 1) (e := 2) (f := 3) (u := 1) (v := 0)
  rfl rfl (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide) (by decide)
  (by decide)).of_agreeOn (H := U2) fun _ _ => ⟨rfl, rfl⟩
example := (isLooseHandcuff_of_loops (G := U2) (c := 0) (d := 1) (p := 3) (x := 0) (y := 1)
  rfl rfl rfl rfl (Or.inr ⟨rfl, rfl⟩) (by decide)).of_agreeOn (H := U2) fun _ _ => ⟨rfl, rfl⟩
example := IsThetaSet.of_agreeOn (G := pumpkin) (H := pumpkin)
  ⟨0, 1, _, _, _, pumpkin_isTheta, rfl⟩ fun _ _ => ⟨rfl, rfl⟩
example := (pumpkinWalk_isPath 0).notMem_of_forall (D := ∅) (w := 0)
  (fun _ _ hz => absurd hz (by simp [verts])) List.mem_cons_self
example := Walk.mem_support_of_mem_verts (p := pumpkinWalk 0)
  (mem_verts.2 ⟨0, List.mem_cons_self, Or.inl rfl⟩)

end OrientedGraph

end Transformer.GainMatroid
