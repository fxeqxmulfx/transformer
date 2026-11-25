module

public import Transformer.GainMatroid.Section2_LinkMinorIndep
public import Transformer.GainMatroid.Section2_UnbalancingClasses

/-!
# Cycles of a gain graph with gain one on a forest

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3: a gain function `γ` realizes `(G, B)` when the
balanced cycles are those of gain one, and Proposition 2.1 switches `γ` to one on a forest. A cycle
whose other edges have gain one is then balanced exactly when its last edge has gain one
(`OrientedGraph.eq_one_of_mem_balancedCycles`, `OrientedGraph.mem_balancedCycles_of_forall_eq_one`).

Contracting the forest makes loops of the edges whose ends it links: along loops a vertex reaches
only itself (`OrientedGraph.ReachableIn.eq_of_forall_loop`), and the only vertex of a loop is its
tail (`OrientedGraph.eq_tail_of_mem_verts_loop`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E Γ : Type*} {G : OrientedGraph V E}

/-- Along loops, a vertex reaches only itself. -/
theorem ReachableIn.eq_of_forall_loop {S : Set E} (hS : ∀ e ∈ S, G.tail e = G.head e) {u w : V}
    (h : G.ReachableIn S u w) : u = w :=
  ReachableIn.rel (r := Eq) (fun _ => rfl) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
    (fun e he => ⟨hS e he, (hS e he).symm⟩) h

/-- The only vertex of a loop is its tail. -/
theorem eq_tail_of_mem_verts_loop {e : E} (hl : G.tail e = G.head e) {u : V}
    (hu : u ∈ G.verts {e}) : u = G.tail e := by
  obtain ⟨g, hg, hgu⟩ := mem_verts.1 hu
  rw [Set.mem_singleton_iff.1 hg] at hgu
  exact hgu.elim Eq.symm fun h => h.symm.trans hl.symm

/-- The edges of `I` outside `T`, as edges of `G / T`, together with `T ⊆ I` make up `I`. -/
theorem image_preimage_val_union_eq {I T : Set E} (hTI : T ⊆ I) :
    Subtype.val '' (Subtype.val ⁻¹' I : Set ↥Tᶜ) ∪ T = I :=
  Set.Subset.antisymm (Set.union_subset (Set.image_preimage_subset _ _) hTI)
    (subset_image_preimage_val_union I)

variable [DecidableEq V]

/-- A balanced cycle whose edges other than `e` have gain one has gain one on `e`. -/
@[to_additive eq_zero_of_mem_addBalancedCycles
  /-- A balanced cycle whose edges other than `e` have gain zero has gain zero on `e`. -/]
theorem eq_one_of_mem_balancedCycles [Group Γ] {γ : E → Γ} {C : Set E}
    (hC : C ∈ G.balancedCycles γ) {e : E} (he : e ∈ C) (h : ∀ f ∈ C, f ≠ e → γ f = 1) :
    γ e = 1 := by
  obtain ⟨hcyc, η, hη⟩ := mem_balancedCycles_iff.1 hC
  have hr : G.ReachableIn (C \ {e}) (G.head e) (G.tail e) :=
    hcyc.reachableIn_of_mem_insert he
      (by rw [Set.insert_sdiff_singleton]; exact Set.subset_insert e C)
      (mem_verts.2 ⟨e, he, Or.inr rfl⟩)
  have heq : η (G.head e) = η (G.tail e) := by
    refine ReachableIn.rel (r := fun a b => η a = η b) (fun _ => rfl)
      (fun _ _ _ h₁ h₂ => h₁.trans h₂) (fun f hf => ?_) hr
    have hf' := hη f hf.1
    rw [switch_apply, h f hf.1 hf.2, mul_one, inv_mul_eq_one] at hf'
    exact ⟨hf', hf'.symm⟩
  have he' := hη e he
  rw [switch_apply, heq, mul_eq_one_iff_eq_inv] at he'
  exact mul_eq_left.1 he'

/-- A cycle whose edges all have gain one is balanced. -/
@[to_additive mem_addBalancedCycles_of_forall_eq_zero
  /-- A cycle whose edges all have gain zero is balanced. -/]
theorem mem_balancedCycles_of_forall_eq_one [Group Γ] {γ : E → Γ} {C : Set E} (hC : G.IsCycle C)
    (h : ∀ f ∈ C, γ f = 1) : C ∈ G.balancedCycles γ :=
  mem_balancedCycles_iff.2 ⟨hC, 1, fun f hf => by rw [switch_one]; exact h f hf⟩

end OrientedGraph

open OrientedGraph

example := ReachableIn.eq_of_forall_loop (G := twoLoops) (S := Set.univ) (fun _ _ => rfl)
  (reachableIn_of_mem (Set.mem_univ true))

example := eq_tail_of_mem_verts_loop (G := twoLoops) (e := true) rfl
  (mem_verts.2 ⟨true, rfl, Or.inl rfl⟩)

example := image_preimage_val_union_eq (I := ({0, 1} : Set (Fin 3))) (T := {0}) (by simp)

example := eq_one_of_mem_balancedCycles
  (mem_balancedCycles_of_forall_eq_one (γ := (1 : Fin 3 → ℚˣ)) pumpkin_pair_isCycle fun _ _ => rfl)
  (e := 0) (Or.inl rfl) fun _ _ _ => rfl

example := eq_zero_of_mem_addBalancedCycles
  (mem_addBalancedCycles_of_forall_eq_zero (γ := (0 : Fin 3 → ℚ)) pumpkin_pair_isCycle
    fun _ _ => rfl) (e := 0) (Or.inl rfl) fun _ _ _ => rfl

end Transformer.GainMatroid
