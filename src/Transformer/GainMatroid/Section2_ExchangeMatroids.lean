module

public import Transformer.GainMatroid.Section2_ExchangeGraphs

/-!
# The matroids of the small graphs of the exchanges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The frame and lift matroids of the balanced
triangle, of its star, of the balanced `clawGraph` and of its Y-Δ exchange `clawNabla`, each built
from its independent sets. `clawMatroid_triad` shows that the star of `clawGraph` is a triad, and
`claw_isNablaExchange` evaluates `∇_Y B`: all six cycles of `clawNabla` are balanced.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

open scoped symmDiff

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {M : Matroid E}

/-- A matroid whose circuits are the cycles of a balanced biased graph is its frame matroid. -/
theorem IsBalanced.isFrameMatroid (hΩ : Ω.IsBalanced) (hE : M.E = Set.univ)
    (h : ∀ X, M.IsCircuit X ↔ Ω.graph.IsCycle X) : Ω.IsFrameMatroid M :=
  ⟨hE, fun X => (h X).trans hΩ.isFrameCircuit_iff.symm⟩

/-- A matroid whose circuits are the cycles of a balanced biased graph is its lift matroid. -/
theorem IsBalanced.isLiftMatroid (hΩ : Ω.IsBalanced) (hE : M.E = Set.univ)
    (h : ∀ X, M.IsCircuit X ↔ Ω.graph.IsCycle X) : Ω.IsLiftMatroid M :=
  ⟨hE, fun X => (h X).trans hΩ.isLiftCircuit_iff.symm⟩

end BiasedGraph

/-- A cycle of `G` is balanced for the identity gain function. -/
theorem mem_balancedCycles_one {V E Γ : Type*} [DecidableEq V] [Group Γ] {G : OrientedGraph V E}
    {C : Set E} : C ∈ G.balancedCycles (1 : E → Γ) ↔ G.IsCycle C :=
  ⟨(G.gainBiased (1 : E → Γ)).isCycle_of_mem C, gainBiased_one_isBalanced G C⟩

/-- `triangle` with its cycle balanced. -/
def triangleBiased : BiasedGraph (Fin 3) (Fin 3) :=
  triangle.singleCycleBiased {1, 2, 0} triangle_isLabelledTriangle.isCycle

theorem triangleBiased_isBalanced : triangleBiased.IsBalanced := by
  intro C hC
  obtain ⟨C, rfl⟩ := (Set.toFinite C).exists_finset_coe
  obtain rfl := (triangle_isCycle_iff C).1 hC
  exact Set.mem_singleton_iff.2 (by simp)

private theorem ncard_triangle : ({1, 2, 0} : Set (Fin 3)).ncard = 3 := by
  have h : ({1, 2, 0} : Set (Fin 3)) = (({1, 2, 0} : Finset (Fin 3)) : Set (Fin 3)) := by simp
  rw [h, Set.ncard_coe_finset]
  rfl

/-- The exchange of the balanced triangle: its only balanced cycle leaves no balanced cycle. -/
theorem deltaBalanced_triangle :
    deltaBalanced ({{1, 2, 0}} : Set (Set (Fin 3))) {1, 2, 0} = ∅ := by
  refine Set.eq_empty_of_forall_notMem ?_
  rintro D (⟨hD, h⟩ | ⟨C, hC, h, -⟩)
  · obtain rfl : D = {1, 2, 0} := hD
    rw [Set.inter_self, ncard_triangle] at h
    omega
  · obtain rfl : C = {1, 2, 0} := hC
    rw [Set.inter_self, ncard_triangle] at h
    omega

/-- The star with no balanced cycle is `Δ_X` of the balanced triangle. -/
theorem triangle_isDeltaExchange : triangleBiased.IsDeltaExchange 1 2 0 0 1 2
    (triangle.deltaGraph 1 2 0 0 1 2).contrabalanced :=
  ⟨triangle_isLabelledTriangle, rfl, rfl, deltaBalanced_triangle.symm⟩

/-- `U_{3,3}`, the frame and lift matroid of the star. -/
noncomputable def freeThree : Matroid (Fin 3) :=
  matroidOfFinset (fun I => I.card ≤ 3) (by decide) (by decide +kernel) (by decide +kernel)

set_option maxRecDepth 100000 in
theorem twoOfThree_isCircuit_iff (X : Set (Fin 3)) :
    twoOfThree.IsCircuit X ↔ triangle.IsCycle X := by
  obtain ⟨X, rfl⟩ := (Set.toFinite X).exists_finset_coe
  refine (matroidOfFinset_isCircuit_iff X).trans (Iff.trans ?_ (triangle_isCycle_iff X).symm)
  revert X
  decide +kernel

theorem freeThree_isCircuit_iff (X : Set (Fin 3)) :
    freeThree.IsCircuit X ↔ triangleStar.IsCycle X := by
  obtain ⟨X, rfl⟩ := (Set.toFinite X).exists_finset_coe
  refine (matroidOfFinset_isCircuit_iff X).trans
    ⟨fun h => (h.1 ?_).elim, fun h => (triangleStar_not_isCycle _ h).elim⟩
  simpa using Finset.card_le_univ X

/-- The star of the exchange is balanced, having no cycle. -/
theorem triangleStar_isBalanced :
    (triangle.deltaGraph 1 2 0 0 1 2).contrabalanced.IsBalanced := fun C hC =>
  (triangleStar_not_isCycle C (triangle_deltaGraph ▸ hC)).elim

theorem freeThree_isCircuit_iff' (X : Set (Fin 3)) :
    freeThree.IsCircuit X ↔ (triangle.deltaGraph 1 2 0 0 1 2).contrabalanced.graph.IsCycle X := by
  rw [freeThree_isCircuit_iff]
  exact iff_of_eq (by rw [← triangle_deltaGraph]; rfl)

/-- The independent sets of the cycle matroid of `clawGraph`. -/
def clawIndep (I : Finset (Fin 5)) : Prop := I.card ≤ 3 ∧ I ≠ {0, 1, 3} ∧ I ≠ {0, 2, 4}

instance : DecidablePred clawIndep := fun I => by
  unfold clawIndep
  infer_instance

/-- The independent sets of the cycle matroid of `clawNabla`. -/
def nablaIndep (I : Finset (Fin 5)) : Prop :=
  I.card ≤ 2 ∧ ¬(2 ∈ I ∧ 3 ∈ I) ∧ ¬(1 ∈ I ∧ 4 ∈ I)

instance : DecidablePred nablaIndep := fun I => by
  unfold nablaIndep
  infer_instance

set_option maxRecDepth 100000 in
/-- The cycle matroid of `clawGraph`. -/
noncomputable def clawMatroid : Matroid (Fin 5) :=
  matroidOfFinset clawIndep (by decide) (by decide +kernel) (by decide +kernel)

set_option maxRecDepth 100000 in
/-- The cycle matroid of `clawNabla`. -/
noncomputable def nablaMatroid : Matroid (Fin 5) :=
  matroidOfFinset nablaIndep (by decide) (by decide +kernel) (by decide +kernel)

set_option maxRecDepth 100000 in
theorem clawMatroid_isCircuit_iff (X : Set (Fin 5)) :
    clawMatroid.IsCircuit X ↔ clawGraph.IsCycle X := by
  obtain ⟨X, rfl⟩ := (Set.toFinite X).exists_finset_coe
  refine (matroidOfFinset_isCircuit_iff X).trans (Iff.trans ?_ (clawGraph_isCycle_iff X).symm)
  revert X
  decide +kernel

set_option maxRecDepth 100000 in
theorem nablaMatroid_isCircuit_iff (X : Set (Fin 5)) :
    nablaMatroid.IsCircuit X ↔ clawNabla.IsCycle X := by
  obtain ⟨X, rfl⟩ := (Set.toFinite X).exists_finset_coe
  refine (matroidOfFinset_isCircuit_iff X).trans (Iff.trans ?_ (clawNabla_isCycle_iff X).symm)
  revert X
  decide +kernel

set_option maxRecDepth 100000 in
/-- The star of `clawGraph` is a triad of its cycle matroid. -/
theorem clawMatroid_triad : clawMatroid✶.IsCircuit {0, 1, 2} := by
  have h : ({0, 1, 2} : Set (Fin 5)) = (({0, 1, 2} : Finset (Fin 5)) : Set (Fin 5)) := by simp
  rw [h]
  exact (matroidOfFinset_dual_isCircuit_iff _).2 (by decide +kernel)

end Transformer.GainMatroid
