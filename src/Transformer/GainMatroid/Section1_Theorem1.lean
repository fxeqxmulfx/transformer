module

public import Transformer.GainMatroid.Section1_Fan

/-!
# Theorem 1 (i) fails on an almost-balanced biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1. Theorem 1 (i) reads: for a 3-connected matroid
`M` of rank greater than two, a loopless biased graph `(G, B)` representing `M` and a field `F`,
"the canonical frame matrices given by two `Fˣ`-gain functions `φ` and `ψ` realizing `(G, B)` are
projectively equivalent if and only if `φ` and `ψ` are switching equivalent". The "only if" fails.

Take `M = U_{3,5}`, the contrabalanced `fanGraph`, and over `𝔽₇` the gains `φ = (1, 2, 3, 4, 5)`
and `ψ = (1, 1, 2, 3, 4)`. Adding the rows of `x` and `y` to the row of `u` turns the frame matrix
of `φ` into that of `ψ`, and no switching does. The graph is almost balanced, with balancing vertex
`u`. Theorem 5.1 (i) gives the claim for properly unbalanced graphs. For almost-balanced ones,
Theorem 5.4 compares the derived additive gains instead, and here these are switching-and-scaling
equivalent (`fanDerived_switchScaleEquiv`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

local instance : Fact (Nat.Prime 7) := ⟨by decide⟩

/-- The unit `2` of `𝔽₇`. -/
def fanTwo : (ZMod 7)ˣ := ⟨2, 4, by decide, by decide⟩

/-- The unit `3` of `𝔽₇`. -/
def fanThree : (ZMod 7)ˣ := ⟨3, 5, by decide, by decide⟩

/-- The gain function `φ = (1, 2, 3, 4, 5)` on `fanGraph`. -/
def fanGain₁ : Fin 5 → (ZMod 7)ˣ :=
  ![1, fanTwo, fanThree, ⟨4, 2, by decide, by decide⟩, ⟨5, 3, by decide, by decide⟩]

/-- The gain function `ψ = (1, 1, 2, 3, 4)` on `fanGraph`. -/
def fanGain₂ : Fin 5 → (ZMod 7)ˣ :=
  ![1, 1, fanTwo, fanThree, ⟨4, 2, by decide, by decide⟩]

/-- The additive gain function derived from `fanGain₁` (§5.1): `0` on the link off `u`. -/
def fanDerived₁ : Fin 5 → ZMod 7 := ![0, 2, 3, 4, 5]

/-- The additive gain function derived from `fanGain₂` (§5.1): `0` on the link off `u`. -/
def fanDerived₂ : Fin 5 → ZMod 7 := ![0, 1, 2, 3, 4]

/-- A gain function on `fanGraph` with no balanced cycle of two or three edges realizes the
contrabalanced biased graph, because every cycle has two or three edges. -/
theorem fan_realizes_of_small {γ : Fin 5 → (ZMod 7)ˣ}
    (h2 : ∀ C ∈ fanGraph.balancedCycles γ, C.encard ≠ 2)
    (h3 : ∀ C ∈ fanGraph.balancedCycles γ, C.encard ≠ 3) :
    fanGraph.contrabalanced.Realizes γ := by
  refine Set.eq_empty_of_forall_notMem fun C hC => ?_
  have hC' : C ∈ fanGraph.balancedCycles γ := hC
  rcases (mem_balancedCycles_iff.1 hC').1.encard_eq_two_or_three fanGraph_loopless with h | h
  · exact h2 C hC' h
  · exact h3 C hC' h

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
theorem fanGain₁_realizes : fanGraph.contrabalanced.Realizes fanGain₁ :=
  fan_realizes_of_small
    (encard_ne_two_of_parallel (by simp only [Joins, dart, fanGraph, fanGain₁]; decide +kernel))
    (encard_ne_three_of_triangle (by simp only [Joins, dart, fanGraph, fanGain₁]; decide +kernel))

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
theorem fanGain₂_realizes : fanGraph.contrabalanced.Realizes fanGain₂ :=
  fan_realizes_of_small
    (encard_ne_two_of_parallel (by simp only [Joins, dart, fanGraph, fanGain₂]; decide +kernel))
    (encard_ne_three_of_triangle (by simp only [Joins, dart, fanGraph, fanGain₂]; decide +kernel))

/-- The row operation adding the rows of `x` and `y` to the row of `u`. -/
def fanRowOp : Matrix (Fin 3) (Fin 3) (ZMod 7) := !![1, 0, 0; 0, 1, 0; 1, 1, 1]

theorem isUnit_fanRowOp : IsUnit fanRowOp :=
  IsUnit.of_mul_eq_one !![1, 0, 0; 0, 1, 0; -1, -1, 1] (by
    ext i j
    fin_cases i <;> fin_cases j <;> decide +kernel)

theorem fanRowOp_mul_frameMatrix :
    fanRowOp * frameMatrix fanGraph fanGain₁ = frameMatrix fanGraph fanGain₂ := by
  ext r e
  rw [mul_frameMatrix_apply]
  fin_cases r <;> fin_cases e <;> decide +kernel

/-- The two frame matrices are projectively equivalent. -/
theorem fan_projEquiv :
    ProjEquiv (frameMatrix fanGraph fanGain₁) (frameMatrix fanGraph fanGain₂) := by
  rw [projEquiv_iff_exists_mul_mul_diagonal]
  refine ⟨fanRowOp, 1, isUnit_fanRowOp, ?_⟩
  simpa using fanRowOp_mul_frameMatrix

/-- The two gain functions are not switching equivalent: a switching would multiply the gains `2`
and `3` of the links `x → u` by one scalar, sending them to `1` and `2`, and `3 ≠ 2 · 2`. -/
theorem fan_not_switchEquiv : ¬SwitchEquiv fanGraph fanGain₁ fanGain₂ := by
  rintro ⟨η, h⟩
  have h₁ : (η 0)⁻¹ * fanTwo * η 2 = 1 := congrFun h 1
  have h₂ : (η 0)⁻¹ * fanThree * η 2 = fanTwo := congrFun h 2
  have key : fanThree = fanTwo * fanTwo := by
    calc fanThree = fanThree * ((η 0)⁻¹ * fanTwo * η 2) := by rw [h₁, mul_one]
      _ = (η 0)⁻¹ * fanThree * η 2 * fanTwo := by ac_rfl
      _ = fanTwo * fanTwo := by rw [h₂]
  exact absurd key (by decide)

/-- The derived additive gains differ by a switching, which is the criterion of Theorem 5.4. -/
theorem fanDerived_switchScaleEquiv : SwitchScaleEquiv fanGraph fanDerived₁ fanDerived₂ := by
  refine ⟨![1, 1, 0], 1, ?_⟩
  funext e
  revert e
  decide +kernel

/-- `fanGraph` with no balanced cycle is almost balanced: vertex `u = 2` lies on every cycle of at
least two edges, since only the link `0` misses it. -/
theorem fan_isAlmostBalanced : fanGraph.contrabalanced.IsAlmostBalanced := by
  have hhead : ∀ e : Fin 5, e ≠ 0 → fanGraph.head e = 2 := by
    simp only [fanGraph]
    decide
  refine ⟨fun hB => Set.notMem_empty _ (hB {1, 2} (isCycle_pair (u := 0) (v := 2)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide))), 2, fun C _ h2 => ?_⟩
  by_contra hv
  have hsub : C ⊆ {0} := fun e he => by
    by_contra hne
    exact hv (mem_verts.2 ⟨e, he, Or.inr (hhead e hne)⟩)
  have hle := Set.encard_le_encard hsub
  rw [Set.encard_singleton] at hle
  exact absurd (h2.trans hle) (by norm_num)

/-- **Theorem 1 (i) fails as printed** (§1). The source claims that for a 3-connected matroid `M` of
rank greater than two and a loopless biased graph representing it, two `Fˣ`-gain functions realizing
the graph have projectively equivalent canonical frame matrices only if they are switching
equivalent. Here `M = U_{3,5}`, the biased graph is the almost-balanced contrabalanced `fanGraph`,
`F = 𝔽₇`, and `fanGain₁`, `fanGain₂` have projectively equivalent frame matrices without being
switching equivalent. -/
theorem theorem1_i_fails :
    ∃ (M : Matroid (Fin 5)) (Ω : BiasedGraph (Fin 3) (Fin 5)) (φ ψ : Fin 5 → (ZMod 7)ˣ),
      IsNConnected M 3 ∧ 2 < M.eRank ∧ Ω.graph.Loopless ∧ Ω.IsFrameMatroid M ∧
        Ω.Realizes φ ∧ Ω.Realizes ψ ∧
        ProjEquiv (frameMatrix Ω.graph φ) (frameMatrix Ω.graph ψ) ∧
        ¬SwitchEquiv Ω.graph φ ψ :=
  ⟨uniformMatroid (Fin 5) 3, fanGraph.contrabalanced, fanGain₁, fanGain₂,
    uniformMatroid_three_five_isNConnected,
    by rw [uniformMatroid_three_five_eRank]; exact_mod_cast (by omega : (2 : ℕ) < 3),
    fanGraph_loopless, fan_isFrameMatroid, fanGain₁_realizes, fanGain₂_realizes, fan_projEquiv,
    fan_not_switchEquiv⟩

/-- The triangle `x → y → u → x` of `fanGraph`. -/
private def fanTriangle : fanGraph.Walk 0 0 :=
  triangleWalk (a := 0) (b := 3) (c := 1) (y := 1) (z := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (Or.inr ⟨rfl, rfl⟩)

private theorem fanTriangle_isCycleWalk : fanTriangle.IsCycleWalk :=
  triangleWalk_isCycleWalk _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide)

private theorem fanTriangle_isCycle : fanGraph.IsCycle fanTriangle.edgeSet :=
  ⟨0, fanTriangle, fanTriangle_isCycleWalk, rfl⟩

/-- The path `x → y → u` of `fanGraph`. -/
private def fanPath : fanGraph.Walk 0 2 :=
  .cons 0 (show fanGraph.Joins 0 0 1 from Or.inl ⟨rfl, rfl⟩)
    (.cons 3 (show fanGraph.Joins 3 1 2 from Or.inl ⟨rfl, rfl⟩) (.nil 2))

private theorem fanPath_isPath : fanPath.IsPath := by
  simp [Walk.IsPath, fanPath, Walk.support]

example : fanTriangle.edgeSet.encard = 2 ∨ fanTriangle.edgeSet.encard = 3 :=
  fanTriangle_isCycle.encard_eq_two_or_three fanGraph_loopless

example : fanGraph.verts fanTriangle.edgeSet = Set.univ :=
  fanTriangle_isCycle.verts_eq_univ_of_three (by
    rw [fanTriangle_isCycleWalk.encard_edgeSet]
    simp [fanTriangle, triangleWalk, Walk.edges])

example : ∃ p q, p ≠ q ∧ p ∈ fanGraph.verts fanTriangle.edgeSet ∧
    q ∈ fanGraph.verts fanTriangle.edgeSet :=
  fanTriangle_isCycle.exists_two_verts fanGraph_loopless

example : (0 : Fin 3) ∈ fanGraph.verts ({0} : Set (Fin 5)) :=
  Joins.left_mem_verts_of_mem (e := 0) (v := 1) (Or.inl ⟨rfl, rfl⟩) (Set.mem_singleton _)

example : ∃ m ∈ fanPath.support, m ≠ 0 ∧ m ≠ 2 :=
  fanPath_isPath.exists_interior (by simp [fanPath, Walk.edges])

example : fanPath.edgeSet.encard = 2 := by
  rw [Walk.encard_edgeSet_of_nodup fanPath_isPath.edges_nodup]
  simp [fanPath, Walk.edges]

example : ∃ e, (Walk.cons 0 (show fanGraph.Joins 0 0 1 from Or.inl ⟨rfl, rfl⟩)
    (.nil 1)).edges = [e] ∧ fanGraph.Joins e 0 1 :=
  Walk.exists_edge_of_length_one _ (by simp [Walk.edges])

end Transformer.GainMatroid
