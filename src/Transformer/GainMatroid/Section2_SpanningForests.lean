module

public import Transformer.GainMatroid.Section2_FrameIndep

/-!
# Spanning forests and pseudoforests

Counting lemmas for the frame matroid of a biased graph (arXiv:1609.05574v5, §2.2), whose
independent sets are the sets with no balanced cycle in which no two distinct cycles are linked
(`BiasedGraph.isFrameIndep_iff`).

* Every set of edges `K` has a spanning forest (`OrientedGraph.exists_spanning_forest`), and the
  nullity of `K` is the number of its edges off such a forest
  (`OrientedGraph.nullity_eq_ncard_sdiff`).
* The edges off a spanning forest close cycles. If no two distinct cycles of `K` are linked, these
  cycles lie in distinct components, none of them a single vertex, so `K` has at most as many edges
  as vertices (`OrientedGraph.ncard_le_ncard_verts`).
* Every component of `K` with a cycle contains an edge off the spanning forest, so the vertices
  linked to a cycle of `K` are at most as many as the edges of `K`
  (`OrientedGraph.ncard_reachesCycle_le`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- `v` is linked along `K` to a vertex of a cycle inside `K`. -/
def ReachesCycle (G : OrientedGraph V E) (K : Set E) (v : V) : Prop :=
  ∃ C ⊆ K, G.IsCycle C ∧ ∃ u ∈ G.verts C, G.ReachableIn K u v

theorem ReachesCycle.of_reachableIn {K : Set E} {u v : V} (hu : G.ReachesCycle K u)
    (h : G.ReachableIn K u v) : G.ReachesCycle K v := by
  obtain ⟨C, hCK, hC, w, hw, hwu⟩ := hu
  exact ⟨C, hCK, hC, w, hw, hwu.trans h⟩

theorem IsCycle.reachesCycle {K C : Set E} (hC : G.IsCycle C) (hCK : C ⊆ K) {v : V}
    (hv : v ∈ G.verts C) : G.ReachesCycle K v :=
  ⟨C, hCK, hC, v, hv, reachableIn_refl K v⟩

/-- The ends of an edge of `K` are both linked to a cycle of `K` or neither is. -/
theorem reachesCycle_tail_iff {K : Set E} {e : E} (he : e ∈ K) :
    G.ReachesCycle K (G.tail e) ↔ G.ReachesCycle K (G.head e) :=
  ⟨fun h => h.of_reachableIn (reachableIn_of_mem he),
    fun h => h.of_reachableIn (reachableIn_of_mem he).symm⟩

/-- If a property of vertices holds at both ends of each edge of `K` or at neither, then a walk
along `K` from a vertex with the property uses only edges whose tails have it. -/
theorem ReachableIn.restrict {K : Set E} {P : V → Prop}
    (hP : ∀ f ∈ K, P (G.tail f) ↔ P (G.head f)) {u v : V} (h : G.ReachableIn K u v) (hu : P u) :
    G.ReachableIn {f ∈ K | P (G.tail f)} u v := by
  refine (ReachableIn.rel (r := fun a b => (P a ↔ P b) ∧
    (P a → G.ReachableIn {f ∈ K | P (G.tail f)} a b))
    (fun a => ⟨Iff.rfl, fun _ => reachableIn_refl _ a⟩) ?_ ?_ h).2 hu
  · exact fun a b c hab hbc =>
      ⟨hab.1.trans hbc.1, fun ha => (hab.2 ha).trans (hbc.2 (hab.1.1 ha))⟩
  · intro f hf
    have hmem : P (G.tail f) → f ∈ {f ∈ K | P (G.tail f)} := fun ht => ⟨hf, ht⟩
    exact ⟨⟨hP f hf, fun ht => reachableIn_of_mem (hmem ht)⟩,
      ⟨(hP f hf).symm, fun hh => (reachableIn_of_mem (hmem ((hP f hf).2 hh))).symm⟩⟩

/-- An edge whose ends are linked along `F` closes a cycle with `F`. -/
theorem exists_isCycle_insert {F : Set E} {e : E} (he : e ∉ F)
    (hr : G.ReachableIn F (G.tail e) (G.head e)) : ∃ C ⊆ insert e F, G.IsCycle C ∧ e ∈ C := by
  obtain ⟨p, hp⟩ := hr
  obtain ⟨Q, hQ, hQp⟩ := p.exists_isPath
  exact ⟨insert e Q.edgeSet, Set.insert_subset_insert fun f hf => hp f (hQp f hf),
    isCycle_insert_of_isPath (Or.inr ⟨rfl, rfl⟩) hQ fun h => he (hp e (hQp e h)),
    Set.mem_insert _ _⟩

variable [Finite E]

/-- Every set of edges has a **spanning forest**: a forest inside it along which the ends of each
of its edges are linked. -/
theorem exists_spanning_forest (K : Set E) :
    ∃ F ⊆ K, G.IsForest F ∧ ∀ e ∈ K, G.ReachableIn F (G.tail e) (G.head e) := by
  induction K, K.toFinite using Set.Finite.induction_on with
  | empty => exact ⟨∅, le_rfl, isForest_empty, fun e he => absurd he (Set.notMem_empty e)⟩
  | @insert a s ha hs ih =>
    obtain ⟨F, hFs, hF, hFr⟩ := ih
    by_cases hr : G.ReachableIn F (G.tail a) (G.head a)
    · refine ⟨F, hFs.trans (Set.subset_insert a s), hF, ?_⟩
      rintro e (rfl | he)
      · exact hr
      · exact hFr e he
    · refine ⟨insert a F, Set.insert_subset_insert hFs, hF.insert hr, ?_⟩
      rintro e (rfl | he)
      · exact reachableIn_of_mem (Set.mem_insert _ _)
      · exact (hFr e he).mono (Set.subset_insert a F)

variable [Finite V]

/-- The nullity of `K` is the number of its edges off a spanning forest. -/
theorem nullity_eq_ncard_sdiff {K F : Set E} (hFK : F ⊆ K) (hF : G.IsForest F)
    (hr : ∀ e ∈ K, G.ReachableIn F (G.tail e) (G.head e)) : G.nullity K = (K \ F).ncard := by
  have h₁ : G.compCount F ≤ G.compCount K := compCount_le_of_forall hr
  have h₂ : G.compCount K ≤ G.compCount F :=
    compCount_le_of_forall fun e he => reachableIn_of_mem (hFK he)
  have h₃ := (card_le_ncard_add_compCount (G := G) F).2.2 hF
  have h₄ := nullity_add_card (G := G) K
  have h₅ := Set.ncard_sdiff_add_ncard_of_subset hFK
  omega

/-- **Pseudoforests.** If no two distinct cycles inside `K` are linked along `K`, then `K` has at
most as many edges as vertices. -/
theorem ncard_le_ncard_verts {K : Set E}
    (hK : ∀ C D, G.IsCycle C → G.IsCycle D → C ⊆ K → D ⊆ K →
      ∀ u ∈ G.verts C, ∀ v ∈ G.verts D, G.ReachableIn K u v → C = D) :
    K.ncard ≤ (G.verts K).ncard := by
  obtain ⟨F, hFK, hF, hr⟩ := exists_spanning_forest (G := G) K
  have hcyc : ∀ e ∈ K \ F, ∃ C ⊆ insert e F, G.IsCycle C ∧ e ∈ C :=
    fun e he => exists_isCycle_insert he.2 (hr e he.1)
  choose! Z hZF hZ heZ using hcyc
  have hZK : ∀ e ∈ K \ F, Z e ⊆ K := fun e he => (hZF e he).trans (Set.insert_subset he.1 hFK)
  let ι : ↥(G.verts K)ᶜ ⊕ ↥(K \ F) → Quotient (G.reachableSetoid K) :=
    Sum.elim (fun v => Quotient.mk _ v.1) (fun e => Quotient.mk _ (G.tail e.1))
  have hι : Function.Injective ι := by
    rintro (⟨v, hv⟩ | ⟨e, he⟩) (⟨w, hw⟩ | ⟨f, hf⟩) h
    · have h' : G.ReachableIn K v w := Quotient.exact h
      obtain rfl := h'.eq_of_notMem_verts hv
      rfl
    · have h' : G.ReachableIn K v (G.tail f) := Quotient.exact h
      exact absurd (h'.eq_of_notMem_verts hv ▸ mem_verts.2 ⟨f, hf.1, Or.inl rfl⟩) hv
    · have h' : G.ReachableIn K (G.tail e) w := Quotient.exact h
      exact absurd (h'.symm.eq_of_notMem_verts hw ▸ mem_verts.2 ⟨e, he.1, Or.inl rfl⟩) hw
    · have h' : G.ReachableIn K (G.tail e) (G.tail f) := Quotient.exact h
      have hEq := hK (Z e) (Z f) (hZ e he) (hZ f hf) (hZK e he) (hZK f hf) _
        (mem_verts.2 ⟨e, heZ e he, Or.inl rfl⟩) _ (mem_verts.2 ⟨f, heZ f hf, Or.inl rfl⟩) h'
      rcases hZF f hf (hEq ▸ heZ e he) with rfl | heF
      · rfl
      · exact absurd heF he.2
  have hle : (G.verts K)ᶜ.ncard + (K \ F).ncard ≤ G.compCount K := by
    have := Nat.card_le_card_of_injective ι hι
    rwa [Nat.card_sum, Nat.card_coe_set_eq, Nat.card_coe_set_eq] at this
  have h₁ := nullity_eq_ncard_sdiff hFK hF hr
  have h₂ := nullity_add_card (G := G) K
  have h₃ := Set.ncard_add_ncard_compl (G.verts K)
  omega

/-- At most `|K|` vertices are linked along `K` to a cycle inside `K`. -/
theorem ncard_reachesCycle_le (K : Set E) : {v | G.ReachesCycle K v}.ncard ≤ K.ncard := by
  obtain ⟨F, hFK, hF, hr⟩ := exists_spanning_forest (G := G) K
  let σ : ↥{v | ¬G.ReachesCycle K v} ⊕ ↥(K \ F) → Quotient (G.reachableSetoid K) :=
    Sum.elim (fun v => Quotient.mk _ v.1) (fun e => Quotient.mk _ (G.tail e.1))
  have hσ : Function.Surjective σ := by
    intro q
    obtain ⟨v, rfl⟩ := Quotient.mk_surjective q
    by_cases hv : G.ReachesCycle K v
    · obtain ⟨C, hCK, hC, u, hu, huv⟩ := hv
      obtain ⟨g, hgC, hgF⟩ : ∃ g ∈ C, g ∉ F := by
        by_contra! h
        exact hF C hC h
      exact ⟨Sum.inr ⟨g, hCK hgC, hgF⟩, Quotient.sound
        (((hC.reachableIn (mem_verts.2 ⟨g, hgC, Or.inl rfl⟩) hu).mono hCK).trans huv)⟩
    · exact ⟨Sum.inl ⟨v, hv⟩, rfl⟩
  have hle : G.compCount K ≤ {v | ¬G.ReachesCycle K v}.ncard + (K \ F).ncard := by
    have := Nat.card_le_card_of_surjective σ hσ
    rwa [Nat.card_sum, Nat.card_coe_set_eq, Nat.card_coe_set_eq] at this
  have h₁ := nullity_eq_ncard_sdiff hFK hF hr
  have h₂ := nullity_add_card (G := G) K
  have h₃ := Set.ncard_add_ncard_compl {v | G.ReachesCycle K v}
  have h₄ : {v | G.ReachesCycle K v}ᶜ = {v | ¬G.ReachesCycle K v} := rfl
  rw [h₄] at h₃
  omega

example := (pumpkin_pair_isCycle.reachesCycle (K := {0, 1}) le_rfl
  (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inl rfl⟩)).of_reachableIn
  (reachableIn_of_mem (e := (1 : Fin 3)) (Set.mem_insert_of_mem 0 (Set.mem_singleton 1)))

example := pumpkin_pair_isCycle.reachesCycle (K := {0, 1}) le_rfl
  (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inl rfl⟩)

example := reachesCycle_tail_iff (G := pumpkin) (K := {0, 1}) (Set.mem_insert 0 {1})

example := (reachableIn_of_mem (G := pathThree) (Set.mem_singleton (0 : Fin 2))).restrict
  (P := fun v => v ≠ 2) (fun f hf => by obtain rfl := hf; decide) (by decide)

example := exists_isCycle_insert (G := pumpkin) (F := {0}) (e := 1) (by simp)
  (reachableIn_of_mem (e := (0 : Fin 3)) (Set.mem_singleton 0))

example := nullity_eq_ncard_sdiff (G := pumpkin) (K := {0, 1}) (F := {0})
  (Set.singleton_subset_iff.2 (Set.mem_insert 0 {1})) pumpkin_isForest
  fun _ _ => reachableIn_of_mem (e := (0 : Fin 3)) (Set.mem_singleton 0)

example := ncard_le_ncard_verts (G := pumpkin) (K := {0, 1}) fun _ _ hC hD hCK hDK _ _ _ _ _ =>
  (pumpkin_pair_isCycle.eq_of_subset hC hCK).trans (pumpkin_pair_isCycle.eq_of_subset hD hDK).symm

end OrientedGraph

end Transformer.GainMatroid
