module

public import Transformer.GainMatroid.Section2_CycleEars
public import Transformer.GainMatroid.Section2_RollUpClasses
public import Transformer.GainMatroid.Section2_RollUpCycles

/-!
# Sets that roll up without changing the frame circuits

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "For each unbalancing class `U` of `Σ_Ĝ(u)` there is a biased graph `(G_U, B_U)` for which
> `F(G_U, B_U) = F(G, B) = F(Ĝ, B̂)` […] (this fact is straightforward to check by comparing
> circuits […])."

Comparing circuits uses four properties of `U`, collected in `IsRollUpClass`: `u` is a balancing
vertex, the edges of `U` are links at `u`, a cycle through two edges of `U` is balanced, and a
balanced cycle through an edge of `U` has its other link at `u` in `U`. Each unbalancing class of
an unrolling has them (`IsUnrolling.isRollUpClass`), and so has the set `J` of joints off `u`
(`IsUnrolling.isRollUpClass_offJoints`), also when it is not a class; in a balanced biased graph so
has `δ(u)` (`IsBalanced.isRollUpClass`).

A balanced cycle of `Ω` through an edge of `U` is two edges of `U` and a path off `u`, a path of the
roll-up (`IsRollUpClass.exists_of_mem_balanced`); an unbalanced one is one edge of `U` and a path
back to `u` with no other (`IsRollUpClass.exists_of_notMem_balanced`). So a set with at most one
edge of `U` and no balanced cycle of the roll-up has none of `Ω`
(`IsRollUpClass.notMem_balanced_of_rollUp`); the converse holds for every `U`
(`notMem_balanced_rollUp`, in `Section2_RollUpCycles`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V} {U : Set E}

/-- `U` rolls up from `u` without changing the frame circuits (§2.5). -/
structure IsRollUpClass (Ω : BiasedGraph V E) (u : V) (U : Set E) : Prop where
  /-- `u` is a balancing vertex. -/
  vertex : Ω.IsBalancingVertex u
  /-- The edges of `U` are links at `u`. -/
  link : U ⊆ Ω.linkStar u
  /-- A cycle through two edges of `U` is balanced. -/
  pair : ∀ C, Ω.graph.IsCycle C → ∀ e ∈ C, ∀ f ∈ C, e ≠ f → e ∈ U → f ∈ U → C ∈ Ω.balanced
  /-- A balanced cycle through an edge of `U` has its links at `u` in `U`. -/
  single : ∀ C ∈ Ω.balanced, ∀ e ∈ C, e ∈ U → ∀ f ∈ C, f ∈ Ω.linkStar u → f ∈ U

/-- Each unbalancing class of the unrolling rolls up. -/
theorem IsUnrolling.isRollUpClass (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsLinkBalancingVertex u)
    (hU : U ∈ Ω'.unbalancingClasses u) : Ω'.IsRollUpClass u U := by
  obtain ⟨k, hk, rfl⟩ := hU
  have hS : ∀ {x}, x ∈ Ω'.sigmaStar u → x ∈ Ω.sigmaStar u := fun hx => by
    rw [← h.sigmaStar_eq]
    exact hx
  refine ⟨h.isBalancingVertex, h.subset_linkStar ⟨k, hk, rfl⟩,
    fun C hC e heC f hfC hne he hf => ?_, fun C hC e heC he f hfC hf => ?_⟩
  · exact (h.mem_balanced_iff hΩ (hS he.1) (hS hf.1) hne hC heC hfC).2
      (h.unbalancingRel_of_rel (hS he.1) (hS hf.1)
        (h.unbalancingRel_trans hΩ (hS he.1) (hS hk) (hS hf.1) he.2.symm hf.2))
  · exact ⟨Or.inl hf, h.unbalancingRel_trans hΩ (hS hk) (hS he.1) (hS (Or.inl hf)) he.2
      (Or.inr (Or.inl ⟨C, hC, heC, hfC⟩))⟩

/-- The joints `J` off `u` roll up in the unrolling, also when they lie in several components of
`G − u` and do not form one class (`unbalancingClass_printed_fails`). -/
theorem IsUnrolling.isRollUpClass_offJoints (h : Ω.IsUnrolling u Ω')
    (hΩ : Ω.IsLinkBalancingVertex u) : Ω'.IsRollUpClass u (Ω.offJoints u) := by
  have hL : ∀ {x}, x ∈ Ω'.linkStar u → x ∈ Ω.sigmaStar u := fun hx => by
    rw [h.linkStar_eq] at hx
    exact hx
  refine ⟨h.isBalancingVertex, fun e he => ?_, fun C hC e heC f hfC hne he hf => ?_,
    fun C hC e heC he f hfC hf => ?_⟩
  · rw [h.linkStar_eq]
    exact Or.inr he
  · exact (h.mem_balanced_iff hΩ (Or.inr he) (Or.inr hf) hne hC heC hfC).2
      (Or.inr (Or.inr ⟨he, hf⟩))
  · by_cases hef : e = f
    · exact hef ▸ he
    rcases (h.mem_balanced_iff hΩ (Or.inr he) (hL hf) hef (Ω'.isCycle_of_mem C hC) heC hfC).1 hC
      with hr | ⟨D, hD, heD, -⟩ | ⟨-, hfJ⟩
    · exact absurd hr hef
    · rw [(Ω.isCycle_of_mem D hD).eq_singleton_of_loop heD he.1.1] at hD
      exact absurd hD he.1.2
    · exact hfJ

/-- In a balanced biased graph the links at any vertex roll up. -/
theorem IsBalanced.isRollUpClass (hΩ : Ω.IsBalanced) (u : V) : Ω.IsRollUpClass u (Ω.linkStar u) :=
  ⟨fun C hC => (hC.2 (hΩ C hC.1)).elim, subset_rfl, fun C hC _ _ _ _ _ _ _ => hΩ C hC,
    fun _ _ _ _ _ _ _ hf => hf⟩

namespace IsRollUpClass

/-- A balanced cycle of `Ω` through an edge `e` of `U` is `e`, a second edge `f` of `U` and a path
off `u` between their other ends, which is a path of the roll-up. -/
theorem exists_of_mem_balanced (hU : Ω.IsRollUpClass u U) {C : Set E} (hCB : C ∈ Ω.balanced)
    {e : E} (heC : e ∈ C) (he : e ∈ U) :
    ∃ f ∈ C, f ∈ U ∧ f ≠ e ∧ ∃ P : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e)
      (Ω.graph.otherEnd u f), P.IsPath ∧ u ∉ P.support ∧ C = insert e (insert f P.edgeSet) := by
  have hC := Ω.isCycle_of_mem C hCB
  obtain ⟨hl, heu⟩ := hU.link he
  obtain ⟨f, hfC, hfe, hfu, -⟩ := hC.exists_partner heC hl heu
  have hfl : Ω.graph.IsLink f := fun hloop => by
    rw [hC.eq_singleton_of_loop hfC hloop] at heC
    exact hfe (Set.mem_singleton_iff.1 heC).symm
  obtain ⟨P, hP, huP, hCP⟩ := hC.exists_isPath_of_mem hfe.symm heC hfC heu hfu
  obtain ⟨q, hq₁, hq₂⟩ := P.exists_of_agreeOn (H := (Ω.rollUp u U).graph) rfl rfl
    (agreeOn_rollUp (Set.disjoint_left.2 fun g hg hgU =>
      huP (Walk.mem_support_of_mem_edges (p := P) hg (hU.link hgU).2))).symm
  have hqP : q.edgeSet = P.edgeSet := by
    unfold Walk.edgeSet
    rw [hq₁]
  refine ⟨f, hfC, hU.single C hCB e heC he f hfC ⟨hfl, hfu⟩, hfe, q, ?_, ?_, ?_⟩
  · unfold Walk.IsPath
    rw [hq₂]
    exact hP
  · rw [hq₂]
    exact huP
  · rw [hqP]
    exact hCP

/-- An unbalanced cycle of `Ω` through an edge `e` of `U` is `e` and a path of the roll-up from the
other end of `e` back to `u` along vertices of the cycle: a second edge of `U` would balance it. -/
theorem exists_of_notMem_balanced (hU : Ω.IsRollUpClass u U) {C : Set E} (hC : Ω.graph.IsCycle C)
    (hCB : C ∉ Ω.balanced) {e : E} (heC : e ∈ C) (he : e ∈ U) :
    ∃ R : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e) u, R.IsPath ∧ e ∉ R.edges ∧
      C = insert e R.edgeSet ∧ ∀ z ∈ R.support, z ∈ Ω.graph.verts C := by
  obtain ⟨-, heu⟩ := hU.link he
  obtain ⟨R, hR, heR, hCR⟩ := hC.exists_isPath_insert_of_link heC heu
  have hRU : Disjoint R.edgeSet U := Set.disjoint_left.2 fun g hg hgU => by
    refine hCB (hU.pair C hC e heC g ?_ (fun hge => heR ?_) he hgU)
    · rw [hCR]
      exact Or.inr hg
    · rw [← hge] at hg
      exact hg
  obtain ⟨q, hq₁, hq₂⟩ := R.exists_of_agreeOn (H := (Ω.rollUp u U).graph) rfl rfl
    (agreeOn_rollUp hRU).symm
  have hqR : q.edgeSet = R.edgeSet := by
    unfold Walk.edgeSet
    rw [hq₁]
  refine ⟨q, ?_, ?_, ?_, fun z hz => ?_⟩
  · unfold Walk.IsPath
    rw [hq₂]
    exact hR
  · rw [hq₁]
    exact heR
  · rw [hqR]
    exact hCR
  · rw [hq₂] at hz
    by_cases hze : z = Ω.graph.otherEnd u e
    · rw [hze]
      exact mem_verts.2 ⟨e, heC, (joins_otherEnd heu).tail_or_head⟩
    · obtain ⟨g, hg, hgz⟩ := R.exists_mem_edges_of_mem_support hz hze
      refine mem_verts.2 ⟨g, ?_, hgz⟩
      rw [hCR]
      exact Or.inr hg

/-- A set with at most one edge of `U` and no balanced cycle of the roll-up has no balanced cycle
of `Ω`: a balanced cycle of `Ω` through an edge of `U` has two. -/
theorem notMem_balanced_of_rollUp (hU : Ω.IsRollUpClass u U) {X : Set E}
    (hX : ∀ C ⊆ X, (Ω.rollUp u U).graph.IsCycle C → C ∉ (Ω.rollUp u U).balanced)
    (hXU : ∀ e ∈ X, ∀ f ∈ X, e ∈ U → f ∈ U → e = f) :
    ∀ C ⊆ X, Ω.graph.IsCycle C → C ∉ Ω.balanced := by
  intro C hCX hC hCB
  by_cases hd : Disjoint C U
  · exact hX C hCX (hC.of_agreeOn (agreeOn_rollUp hd).symm) ⟨hCB, hd⟩
  obtain ⟨e, heC, he⟩ := Set.not_disjoint_iff.1 hd
  obtain ⟨f, hfC, hf, hfe, -⟩ := hU.exists_of_mem_balanced hCB heC he
  exact hfe (hXU f (hCX hfC) e (hCX heC) hf he)

end IsRollUpClass

end BiasedGraph

open BiasedGraph

/-- The edge `0` of `twoJointsHat` is in its class. -/
theorem twoJointsHat_zero_mem_class :
    (0 : Fin 6) ∈ twoJointsHat.contrabalanced.unbalancingClass 0 0 :=
  ⟨twoJointsHat_zero_mem, Or.inl rfl⟩

example := twoJoints_isUnrolling.isRollUpClass twoJoints_isRollable.balancing twoJointsHat_class_mem
example := twoJoints_isUnrolling.isRollUpClass_offJoints twoJoints_isRollable.balancing
example := (gainBiased_one_isBalanced (Γ := ℚˣ) pumpkin).isRollUpClass 0
example := ((gainBiased_one_isBalanced (Γ := ℚˣ) pumpkin).isRollUpClass 0).exists_of_mem_balanced
  (gainBiased_one_isBalanced pumpkin _ pumpkin_isTheta.isCycle.1) (Or.inl List.mem_cons_self)
  (pumpkin_mem_linkStar 0)
example := (twoJoints_isUnrolling.isRollUpClass twoJoints_isRollable.balancing
  twoJointsHat_class_mem).exists_of_notMem_balanced (C := {0, 1})
  (isCycle_pair (G := twoJointsHat) (e := 0) (f := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (by decide) (by decide)) (Set.notMem_empty _) (Or.inl rfl) twoJointsHat_zero_mem_class
example := (twoJoints_isUnrolling.isRollUpClass twoJoints_isRollable.balancing
  twoJointsHat_class_mem).notMem_balanced_of_rollUp (X := {0}) (fun _ _ _ hb => hb.1)
  fun _ he _ hf _ _ => he.trans hf.symm

end Transformer.GainMatroid
