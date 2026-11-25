module

public import Transformer.GainMatroid.Section2_ContractEmbedding
public import Transformer.GainMatroid.Section2_ExchangeLabels

/-!
# The two sides of an exchange pair, carried into one graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. In an exchange pair (`IsExchange`) on
`X = {a, b, c}`, `X` is a triangle `a = yz`, `b = xz`, `c = xy` of `T` and a star `a = vx`,
`b = vy`, `c = vz` of `S`, and the other edges keep their ends. Contracting `c` in `S` merges `v`
with `z`, so `a` comes to join `x` and `z`, as `b` does in `T`, and `b` comes to join `y` and `z`,
as `a` does in `T`: `S / c` is `T \ c` with `a` and `b` exchanged.

So `T` on its vertices not sent to `v`, with the edges of a set `D` (`IsExchange.offGraph`), is
carried into `T` itself (`IsExchange.tEmbedding`), and for `D` off `X` into `S` itself
(`IsExchange.sEmbedding₀`); for `D` off `c` it is carried into `S / c` in
`Section2_ExchangeContraction`. Its cycles are the cycles of `T` inside `D`
(`IsExchange.isCycle_val`, `IsExchange.isCycle_offGraph`). In `S`, the edge `c` is a forest
(`IsExchange.isForest_c`) whose ends `v` and `ι z` merge in `S / c` (`IsExchange.class_v_eq`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V W E : Type*}

/-- Along no edges a vertex reaches only itself. -/
theorem ReachableIn.eq_of_empty {G : OrientedGraph V E} {u w : V} (h : G.ReachableIn ∅ u w) :
    u = w :=
  ReachableIn.rel (r := Eq) (fun _ => rfl) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
    (fun _ he => absurd he (Set.notMem_empty _)) h

/-- Along one edge, two distinct vertices reached from each other are its ends. -/
theorem ReachableIn.mem_verts_of_ne {G : OrientedGraph V E} {e : E} {u w : V}
    (h : G.ReachableIn {e} u w) (hne : u ≠ w) : u ∈ G.verts {e} ∧ w ∈ G.verts {e} :=
  ⟨Classical.byContradiction fun hu => hne (h.eq_of_notMem_verts hu).symm,
    Classical.byContradiction fun hw => hne (h.symm.eq_of_notMem_verts hw)⟩

/-- An edge outside `K` joins in `G / K` the classes of its ends. -/
theorem Joins.contract_mk {G : OrientedGraph V E} {K : Set E} {f : E} {p q : V}
    (h : G.Joins f p q) (e : ↥Kᶜ) (he : e.val = f) :
    (G.contract K).Joins e (Quotient.mk _ p) (Quotient.mk _ q) := by
  subst he
  rcases h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · exact Or.inl ⟨congrArg (Quotient.mk _) h₁, congrArg (Quotient.mk _) h₂⟩
  · exact Or.inr ⟨congrArg (Quotient.mk _) h₁, congrArg (Quotient.mk _) h₂⟩

/-- An exchange of two labels outside `c` is not `c`. -/
theorem swap_ne [DecidableEq E] {a b c e : E} (hac : a ≠ c) (hbc : b ≠ c) (he : e ≠ c) :
    Equiv.swap a b e ≠ c := by
  intro hc
  apply he
  rw [← Equiv.swap_apply_self a b e, hc, Equiv.swap_apply_of_ne_of_ne hac.symm hbc.symm]

namespace IsExchange

variable {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E}
  {x y z : V}

/-- `T` on its vertices not sent to `v`, with the edges of `D`. -/
def offGraph (h : IsExchange T S ι v a b c x y z) (D : Set E) :
    OrientedGraph {u // ι u ≠ v} ↥D :=
  ⟨fun e => ⟨T.tail e, h.tail_ne e⟩, fun e => ⟨T.head e, h.head_ne e⟩⟩

/-- A cycle of `offGraph` is a cycle of `T`. -/
theorem isCycle_val (h : IsExchange T S ι v a b c x y z) {D : Set E} {C : Set ↥D}
    (hC : (h.offGraph D).IsCycle C) : T.IsCycle (Subtype.val '' C) :=
  hC.map_both Subtype.val_injective Subtype.val_injective fun _ => Or.inl ⟨rfl, rfl⟩

/-- A cycle of `T` inside `D` is a cycle of `offGraph`. -/
theorem isCycle_offGraph (h : IsExchange T S ι v a b c x y z) {D C : Set E} (hCD : C ⊆ D)
    (hC : T.IsCycle C) : (h.offGraph D).IsCycle (Subtype.val ⁻¹' C) :=
  IsCycle.of_map_both Subtype.val_injective Subtype.val_injective (fun _ => Or.inl ⟨rfl, rfl⟩)
    (by rwa [Subtype.image_preimage_coe, Set.inter_eq_right.2 hCD])

/-- `ι z` is not `v`. -/
theorem iota_z_ne (h : IsExchange T S ι v a b c x y z) : ι z ≠ v :=
  h.ne_of_joins h.triangle.joins_a.symm

/-- The ends of `c` in `S` other than `v` are `ι z`. -/
theorem eq_of_mem_verts_c (h : IsExchange T S ι v a b c x y z) {p : W} (hp : p ∈ S.verts {c})
    (hpv : p ≠ v) : p = ι z := by
  simp only [verts, Set.image_singleton, Set.mem_union, Set.mem_singleton_iff] at hp
  rcases h.joins_c with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · rcases hp with hp | hp
    · exact absurd (hp.trans h₁) hpv
    · exact hp.trans h₂
  · rcases hp with hp | hp
    · exact hp.trans h₁
    · exact absurd (hp.trans h₂) hpv

/-- The edge `c` alone is a forest of `S`. -/
theorem isForest_c (h : IsExchange T S ι v a b c x y z) : S.IsForest {c} := by
  have hne : S.tail c ≠ S.head c := by
    rcases h.joins_c with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
    · rw [h₁, h₂]
      exact h.iota_z_ne.symm
    · rw [h₁, h₂]
      exact h.iota_z_ne
  have := (isForest_empty (G := S)).insert (e := c) fun hr => hne hr.eq_of_empty
  rwa [insert_empty_eq] at this

/-- In `S / c`, the classes of `v` and `ι z` agree. -/
theorem class_v_eq (h : IsExchange T S ι v a b c x y z) :
    Quotient.mk (S.reachableSetoid {c}) v = Quotient.mk _ (ι z) := by
  have := contract_class_eq (G := S) (Set.mem_singleton c)
  rcases h.joins_c with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · rwa [h₁, h₂] at this
  · rw [h₁, h₂] at this
    exact this.symm

end IsExchange

end OrientedGraph

namespace OrientedGraph.IsExchange

open BiasedGraph

variable {V W E : Type*} {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W}
  {a b c : E} {x y z : V}

/-- `offGraph` carried into `T` itself, contracting nothing. -/
def tEmbedding {Ω : BiasedGraph V E} (h : IsExchange Ω.graph S ι v a b c x y z) (D : Set E) :
    Ω.ContractEmbedding (h.offGraph D) where
  contractSet := ∅
  vertex u := Quotient.mk _ u.val
  edge e := ⟨e.val, Set.notMem_empty _⟩
  isForest := isForest_empty
  vertex_injective _ _ huw := Subtype.ext (ReachableIn.eq_of_empty (Quotient.exact huw))
  edge_injective _ _ hef := by
    have := congrArg Subtype.val hef
    exact Subtype.ext this
  joins _ := Or.inl ⟨rfl, rfl⟩

theorem tEmbedding_image {Ω : BiasedGraph V E} (h : IsExchange Ω.graph S ι v a b c x y z)
    (D : Set E) (C : Set ↥D) : Subtype.val '' ((h.tEmbedding D).edge '' C) = Subtype.val '' C :=
  Set.image_image _ _ _

/-- Through `tEmbedding`, a set is balanced when its edges are balanced in `Ω`. -/
theorem tEmbedding_balanced_iff {Ω : BiasedGraph V E} (h : IsExchange Ω.graph S ι v a b c x y z)
    (D : Set E) (C : Set ↥D) : (h.tEmbedding D).Balanced C ↔ Subtype.val '' C ∈ Ω.balanced := by
  rw [ContractEmbedding.Balanced, h.tEmbedding_image]
  refine ⟨fun ⟨K₀, hK₀, hB⟩ => ?_, fun hB => ⟨∅, Set.empty_subset _, by rwa [Set.union_empty]⟩⟩
  rwa [Set.subset_empty_iff.1 hK₀, Set.union_empty] at hB

theorem tEmbedding_carry {Ω : BiasedGraph V E} (h : IsExchange Ω.graph S ι v a b c x y z)
    (D : Set E) (I : Set ↥D) : (h.tEmbedding D).carry I = Subtype.val '' I := by
  rw [ContractEmbedding.carry, h.tEmbedding_image]
  exact Set.union_empty _

/-- `offGraph` off `X` carried into `S` itself, contracting nothing. -/
def sEmbedding₀ {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z) :
    Ω.ContractEmbedding (h.offGraph ({a, b, c} : Set E)ᶜ) where
  contractSet := ∅
  vertex u := Quotient.mk _ (ι u.val)
  edge e := ⟨e.val, Set.notMem_empty _⟩
  isForest := isForest_empty
  vertex_injective _ _ huw := Subtype.ext (h.inj (ReachableIn.eq_of_empty (Quotient.exact huw)))
  edge_injective _ _ hef := by
    have := congrArg Subtype.val hef
    exact Subtype.ext this
  joins e := Or.inl ⟨congrArg (Quotient.mk _) (h.agree e.val e.prop).1,
    congrArg (Quotient.mk _) (h.agree e.val e.prop).2⟩

theorem sEmbedding₀_image {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z)
    (C : Set ↥({a, b, c} : Set E)ᶜ) : Subtype.val '' (h.sEmbedding₀.edge '' C) = Subtype.val '' C :=
  Set.image_image _ _ _

/-- Through `sEmbedding₀`, a set is balanced when its edges are balanced in `Ω`. -/
theorem sEmbedding₀_balanced_iff {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z)
    (C : Set ↥({a, b, c} : Set E)ᶜ) : h.sEmbedding₀.Balanced C ↔ Subtype.val '' C ∈ Ω.balanced := by
  rw [ContractEmbedding.Balanced, h.sEmbedding₀_image]
  refine ⟨fun ⟨K₀, hK₀, hB⟩ => ?_, fun hB => ⟨∅, Set.empty_subset _, by rwa [Set.union_empty]⟩⟩
  rwa [Set.subset_empty_iff.1 hK₀, Set.union_empty] at hB

theorem sEmbedding₀_carry {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z)
    (I : Set ↥({a, b, c} : Set E)ᶜ) : h.sEmbedding₀.carry I = Subtype.val '' I := by
  rw [ContractEmbedding.carry, h.sEmbedding₀_image]
  exact Set.union_empty _

end OrientedGraph.IsExchange

end Transformer.GainMatroid
