module

public import Transformer.GainMatroid.Section2_DeletionCycles
public import Transformer.GainMatroid.Section2_SpanningForests

/-!
# Cycles of a contraction by a forest

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, contract a forest `K` of `G`: the vertices of
`G / K` are the components of `K`, and its edges are those outside `K` (`OrientedGraph.contract`).

Two vertices of `G` lie in components linked along `S` in `G / K` exactly when they are linked along
`S ∪ K` in `G` (`OrientedGraph.reachableIn_contract_iff`), so `S` has as many components in `G / K`
as `S ∪ K` in `G` (`OrientedGraph.compCount_contract`), and the same nullity when `K` is a forest
(`OrientedGraph.nullity_contract`). Hence every cycle of `G / K` lifts to a cycle of `G` through
some edges of `K` (`OrientedGraph.IsCycle.exists_lift_contract`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E} {K : Set E}

/-- A walk of `G` along `S ∪ K` gives a walk of `G / K` along `S` between the components of its
ends. -/
theorem ReachableIn.contract {S : Set ↥Kᶜ} {u w : V}
    (h : G.ReachableIn (Subtype.val '' S ∪ K) u w) :
    (G.contract K).ReachableIn S (Quotient.mk _ u) (Quotient.mk _ w) := by
  refine ReachableIn.rel
    (r := fun u w => (G.contract K).ReachableIn S (Quotient.mk _ u) (Quotient.mk _ w))
    (fun _ => reachableIn_refl _ _) (fun _ _ _ h₁ h₂ => h₁.trans h₂) (fun f hf => ?_) h
  rcases hf with ⟨g, hg, rfl⟩ | hfK
  · exact ⟨reachableIn_of_mem hg, (reachableIn_of_mem hg).symm⟩
  · rw [contract_class_eq hfK]
    exact ⟨reachableIn_refl _ _, reachableIn_refl _ _⟩

/-- A walk of `G / K` along `S` between the components of `u` and `w` gives a walk of `G` along
`S ∪ K` from `u` to `w`. -/
theorem ReachableIn.of_contract {S : Set ↥Kᶜ} {u w : V}
    (h : (G.contract K).ReachableIn S (Quotient.mk _ u) (Quotient.mk _ w)) :
    G.ReachableIn (Subtype.val '' S ∪ K) u w := by
  have hK : ∀ a b, G.ReachableIn K a b → G.ReachableIn (Subtype.val '' S ∪ K) a b :=
    fun _ _ h => h.mono Set.subset_union_right
  refine ReachableIn.rel (r := fun a b => ∀ u w, Quotient.mk (G.reachableSetoid K) u = a →
    Quotient.mk (G.reachableSetoid K) w = b → G.ReachableIn (Subtype.val '' S ∪ K) u w)
    ?_ ?_ ?_ h u w rfl rfl
  · rintro a u w rfl huw
    exact hK _ _ (Quotient.exact huw).symm
  · intro a b c h₁ h₂ u w hu hw
    obtain ⟨m, rfl⟩ := Quotient.mk_surjective b
    exact (h₁ u m hu rfl).trans (h₂ m w rfl hw)
  · intro f hf
    have hfS : (f : E) ∈ Subtype.val '' S ∪ K := Or.inl ⟨f, hf, rfl⟩
    refine ⟨fun u w hu hw => ?_, fun u w hu hw => ?_⟩
    · exact ((hK _ _ (Quotient.exact hu)).trans (reachableIn_of_mem hfS)).trans
        (hK _ _ (Quotient.exact hw).symm)
    · exact ((hK _ _ (Quotient.exact hu)).trans (reachableIn_of_mem hfS).symm).trans
        (hK _ _ (Quotient.exact hw).symm)

/-- Two components of `K` are linked along `S` in `G / K` exactly when their vertices are linked
along `S ∪ K` in `G`. -/
theorem reachableIn_contract_iff {S : Set ↥Kᶜ} {u w : V} :
    (G.contract K).ReachableIn S (Quotient.mk _ u) (Quotient.mk _ w) ↔
      G.ReachableIn (Subtype.val '' S ∪ K) u w :=
  ⟨ReachableIn.of_contract, ReachableIn.contract⟩

/-- The vertices of `G / K` met by `S` are the components of the vertices of `G` met by `S`. -/
theorem verts_contract {S : Set ↥Kᶜ} :
    (G.contract K).verts S = Quotient.mk _ '' G.verts (Subtype.val '' S) := by
  simp only [verts, Set.image_union, Set.image_image]
  rfl

/-- `S` has as many components in `G / K` as `S ∪ K` has in `G`. -/
theorem compCount_contract {S : Set ↥Kᶜ} :
    (G.contract K).compCount S = G.compCount (Subtype.val '' S ∪ K) := by
  let f : Quotient (G.reachableSetoid (Subtype.val '' S ∪ K)) →
      Quotient ((G.contract K).reachableSetoid S) :=
    Quotient.map' (Quotient.mk _) fun _ _ h => ReachableIn.contract h
  have hf (u : V) : f (Quotient.mk _ u) = Quotient.mk _ (Quotient.mk _ u) := rfl
  have hbij : Function.Bijective f := by
    refine ⟨fun c d hcd => ?_, fun c => ?_⟩
    · obtain ⟨u, rfl⟩ := Quotient.mk_surjective c
      obtain ⟨w, rfl⟩ := Quotient.mk_surjective d
      rw [hf, hf] at hcd
      exact Quotient.sound (ReachableIn.of_contract (Quotient.exact hcd))
    · obtain ⟨q, rfl⟩ := Quotient.mk_surjective c
      obtain ⟨u, rfl⟩ := Quotient.mk_surjective q
      exact ⟨Quotient.mk _ u, rfl⟩
  exact (Nat.card_congr (Equiv.ofBijective f hbij)).symm

variable [Finite V] [Finite E]

/-- Contracting a forest `K` keeps the nullity: `S` has in `G / K` the nullity of `S ∪ K` in
`G`. -/
theorem nullity_contract (hK : G.IsForest K) {S : Set ↥Kᶜ} :
    (G.contract K).nullity S = G.nullity (Subtype.val '' S ∪ K) := by
  have h₁ := nullity_add_card (G := G.contract K) S
  have h₂ := nullity_add_card (G := G) (Subtype.val '' S ∪ K)
  have h₃ := nullity_add_card (G := G) K
  rw [nullity_eq_zero_iff.2 hK] at h₃
  have hcard : Nat.card (Quotient (G.reachableSetoid K)) = G.compCount K := rfl
  have hdisj : Disjoint (Subtype.val '' S) K :=
    Set.disjoint_left.2 fun _ ⟨f, _, hf⟩ hK => f.2 (hf ▸ hK)
  rw [Set.ncard_union_eq hdisj, Set.ncard_image_of_injective _ Subtype.val_injective] at h₂
  rw [compCount_contract, hcard] at h₁
  omega

/-- A set of edges with positive nullity contains a cycle. -/
theorem exists_isCycle_of_nullity_pos {S : Set E} (h : 0 < G.nullity S) :
    ∃ C ⊆ S, G.IsCycle C := by
  by_contra! hS
  exact h.ne' (nullity_eq_zero_iff.2 fun C hC hCS => hS C hCS hC)

/-- **Lifting a cycle of `G / K`.** A cycle `C` of `G / K` and some edges of the forest `K` form a
cycle of `G`. -/
theorem IsCycle.exists_lift_contract (hK : G.IsForest K) {C : Set ↥Kᶜ}
    (hC : (G.contract K).IsCycle C) :
    ∃ L, G.IsCycle L ∧ Subtype.val '' C ⊆ L ∧ L ⊆ Subtype.val '' C ∪ K := by
  have h₁ : G.nullity (Subtype.val '' C ∪ K) = 1 := by
    rw [← nullity_contract hK, hC.nullity_eq_one]
  obtain ⟨L, hLsub, hL⟩ := exists_isCycle_of_nullity_pos (h₁ ▸ zero_lt_one)
  refine ⟨L, hL, ?_, hLsub⟩
  have hXC : Subtype.val ⁻¹' L ⊆ C := fun f hf => by
    rcases hLsub hf with ⟨g, hg, hgf⟩ | hfK
    · rwa [← Subtype.val_injective hgf]
    · exact absurd hfK f.2
  have hLX : L ⊆ Subtype.val '' (Subtype.val ⁻¹' L) ∪ K := fun f hf => by
    by_cases hfK : f ∈ K
    · exact Or.inr hfK
    · exact Or.inl ⟨⟨f, hfK⟩, hf, rfl⟩
  have hpos : 0 < (G.contract K).nullity (Subtype.val ⁻¹' L) := by
    rw [nullity_contract hK]
    exact zero_lt_one.trans_le (hL.nullity_eq_one.symm.le.trans (nullity_mono hLX))
  obtain ⟨C', hC'X, hC'⟩ := exists_isCycle_of_nullity_pos hpos
  rw [hC.eq_of_subset hC' (hC'X.trans hXC)] at hC'X
  exact Set.image_subset_iff.2 hC'X

/-- The edge `1` of `pumpkin` after contracting its edge `0`. -/
def pumpkinContractOne : ↥(({0} : Set (Fin 3))ᶜ) := ⟨1, by simp⟩

/-- Contracting the edge `0` of `pumpkin` makes its edge `1` a loop. -/
theorem pumpkin_contract_isCycle : (pumpkin.contract {0}).IsCycle {pumpkinContractOne} :=
  isCycle_singleton_of_loop (contract_class_eq (G := pumpkin) (F := {0}) (f := 0) rfl)

example := ReachableIn.contract (G := pumpkin) (K := {0}) (S := {pumpkinContractOne})
  (reachableIn_of_mem (Or.inl ⟨pumpkinContractOne, rfl, rfl⟩))

example := ReachableIn.of_contract (G := pumpkin) (K := {0}) (S := ∅) (u := 0) (w := 1)
  (by
    have h : Quotient.mk (pumpkin.reachableSetoid {0}) 0 = Quotient.mk _ 1 :=
      contract_class_eq (G := pumpkin) (F := {0}) (f := 0) rfl
    rw [h]
    exact reachableIn_refl _ _)

example := exists_isCycle_of_nullity_pos (G := pumpkin) (S := {0, 1}) (by
  rw [pumpkin_pair_isCycle.nullity_eq_one]
  exact zero_lt_one)

example := pumpkin_contract_isCycle.exists_lift_contract pumpkin_isForest

example := nullity_contract (G := pumpkin) pumpkin_isForest (S := {pumpkinContractOne})

end OrientedGraph

end Transformer.GainMatroid
