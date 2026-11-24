module

public import Transformer.GainMatroid.Section2_LoopUnrolling
public import Transformer.GainMatroid.Section2_RollUpCircuits

/-!
# The frame circuits of a biased graph balanced off its joints

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. The source's special case is a biased graph
balanced after removing its joints:

> "Let `x` be a new isolated vertex added to `V(G)`. Then `J` may be unrolled to any vertex of `G`,
> including `x`. In the case that `J` is unrolled to `x`, we obtain a balanced biased graph
> `(H, C(H))`, so `F(G, B)` is equal to the cycle matroid `M(H)` of `H`. The reverse operation may
> be applied to any graph. Given a graph `H` and a vertex `x ∈ V(H)`, let `(G, B)` be the rollup of
> the set of edges incident to `x` […]. Then `M(H) = F(G, B)`."

Matroid equalities are stated on circuits, which determine a matroid (`Matroid.ext_isCircuit`).
`H` is `unrollToNew`, on the vertices `Option V` with `none` the new vertex. Balanced after removing
the joints means that every unbalanced cycle meets `J`, and a cycle through a loop is that loop, so
the hypothesis is that every unbalanced cycle has one edge. Then every vertex of `G` is a balancing
vertex after deleting the joints (`IsLinkBalancingVertex`), so the unrolling to it exists and keeps
the frame circuits (`exists_isUnrolling_of_balanced_off`, by `Section2_RollUpCircuits`). Both
comparisons with `M(H)` follow from `IsLoopUnrolling.isFrameCircuit_iff`: when `H` unrolls the
loops `U` of `Ω` to a vertex, the frame circuits of `Ω` are the cycles of `H`. An unbalanced cycle
of `Ω` is one loop of `U`, so a frame circuit with no balanced cycle is a tight or loose handcuff of
two such loops, which `H` closes into a cycle through the vertex, and no theta has a cycle of one
edge. Conversely, a cycle of `H` through the vertex is two links of `U` and a path, which lift to
two joints and a path of `Ω` (`isFrameCircuit_of_joints`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V W E : Type*}

open OrientedGraph

namespace BiasedGraph

namespace IsLoopUnrolling

variable {Ω : BiasedGraph V E} {H : OrientedGraph W E} {ι : V → W} {z : W} {U : Set E}

/-- §2.5: when `H` unrolls the loops `U` of `Ω` to `z`, the frame circuits of `Ω` are the cycles
of `H`. -/
theorem isFrameCircuit_iff (h : IsLoopUnrolling Ω H ι z U) (X : Set E) :
    Ω.IsFrameCircuit X ↔ H.IsCycle X := by
  constructor
  · rintro (hb | ⟨hT | hL | hθ, hno⟩)
    · have hC := Ω.isCycle_of_mem X hb
      have hd := (h.balanced X hC).1 hb
      exact hC.map_vertex h.inj fun g hg => h.keep g (Set.disjoint_left.1 hd hg)
    · obtain ⟨C, D, v, hC, hD, hCD, hv, rfl⟩ := hT
      obtain ⟨e, he, rfl⟩ := h.eq_singleton_of_notMem hC (hno _ Set.subset_union_left hC)
      obtain ⟨f, hf, rfl⟩ := h.eq_singleton_of_notMem hD (hno _ Set.subset_union_right hD)
      have hvv : v ∈ Ω.graph.verts {e} ∩ Ω.graph.verts {f} := by
        rw [hv]
        exact Set.mem_singleton v
      have hfe : H.Joins f (ι (Ω.graph.tail e)) z := by
        rw [h.tail_eq_of_mem_verts he hvv.1, ← h.tail_eq_of_mem_verts hf hvv.2]
        exact h.joins f hf
      exact isCycle_pair (h.joins e he) hfe (Set.disjoint_singleton.1 hCD) (h.ne e he)
    · obtain ⟨C, D, x, y, P, hC, hD, hCD, hP, hx, hy, -, -, rfl⟩ := hL
      obtain ⟨e, he, rfl⟩ := h.eq_singleton_of_notMem hC
        (hno _ (Set.subset_union_left.trans Set.subset_union_left) hC)
      obtain ⟨f, hf, rfl⟩ := h.eq_singleton_of_notMem hD
        (hno _ (Set.subset_union_right.trans Set.subset_union_left) hD)
      have hne : e ≠ f := by
        rintro rfl
        exact Set.disjoint_left.1 hCD hx hx
      obtain rfl := h.tail_eq_of_mem_verts he hx
      obtain rfl := h.tail_eq_of_mem_verts hf hy
      obtain ⟨q, hq₁, hq₂⟩ := P.exists_map_vertex ι fun g hg =>
        h.keep_of_isLink (hP.isLink_of_mem hg)
      have hq : q.IsPath := by
        unfold Walk.IsPath
        rw [hq₂]
        exact List.Nodup.map h.inj hP
      have hzq : z ∉ q.support := by
        intro hz
        obtain ⟨g, hg, hgz⟩ := q.exists_mem_edges_of_mem_support hz (h.ne e he).symm
        rw [hq₁] at hg
        have hgl := hP.isLink_of_mem hg
        obtain ⟨h₁, h₂⟩ := h.keep_of_isLink hgl
        exact hgl (h.loop g (h.star g (fun hH => hgl (h.inj (h₁.symm.trans (hH.trans h₂)))) hgz))
      have hqP : q.edgeSet = P.edgeSet := by
        unfold Walk.edgeSet
        rw [hq₁]
      rw [← hqP]
      exact isCycle_pair_union (h.joins e he) (h.joins f hf) hne hq hzq
    · obtain ⟨x, y, P, Q, R, hT, rfl⟩ := hθ
      have hPQ := hT.isCycle.1
      obtain ⟨e, -, he⟩ := h.eq_singleton_of_notMem hPQ (hno _ Set.subset_union_left hPQ)
      obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hT.ne P)
      obtain ⟨q, hq⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hT.ne Q)
      have hp' : p ∈ ({e} : Set E) := by
        rw [← he]
        exact Or.inl hp
      have hq' : q ∈ ({e} : Set E) := by
        rw [← he]
        exact Or.inr hq
      refine (hT.openPQ.edges p hp ?_).elim
      rw [(Set.mem_singleton_iff.1 hp').trans (Set.mem_singleton_iff.1 hq').symm]
      exact hq
  · intro hX
    by_cases hd : Disjoint X U
    · exact Or.inl ((h.balanced X (hX.of_map_vertex h.inj fun g hg =>
        h.keep g (Set.disjoint_left.1 hd hg))).2 hd)
    obtain ⟨e, heX, he⟩ := Set.not_disjoint_iff.1 hd
    obtain ⟨heL, heu⟩ := h.isLink he
    obtain ⟨f, hfX, hfe, hfu, -⟩ := hX.exists_partner heX heL heu
    have hfL : H.IsLink f := fun hl => by
      have hef : e ∈ ({f} : Set E) := by
        rw [← hX.eq_singleton_of_loop hfX hl]
        exact heX
      exact hfe (Set.mem_singleton_iff.1 hef).symm
    have hf := h.star f hfL hfu
    obtain ⟨P, hP, hzP, rfl⟩ := hX.exists_isPath_of_mem hfe.symm heX hfX heu hfu
    obtain ⟨y, p, hy, hp₁, hp₂⟩ := P.exists_lift h.inj
      (fun g hg => h.keep g fun hgU => hzP (Walk.mem_support_of_mem_edges hg (h.isLink hgU).2))
      (h.joins e he).otherEnd_eq.symm
    obtain rfl : y = Ω.graph.tail f := h.inj (hy.trans (h.joins f hf).otherEnd_eq)
    have hp : p.IsPath := by
      refine List.Nodup.of_map ι ?_
      rw [hp₂]
      exact hP
    have hpP : p.edgeSet = P.edgeSet := by
      unfold Walk.edgeSet
      rw [hp₁]
    have := isFrameCircuit_of_joints (h.isJoint he) (h.isJoint hf) hfe.symm hp
    rwa [hpP, Set.insert_union, Set.singleton_union] at this

end IsLoopUnrolling

variable {Ω : BiasedGraph V E}

/-- §2.5: when `(G, B)` is balanced after removing its joints, "`J` may be unrolled to any vertex
of `G`", and the frame matroid does not change. No unbalanced cycle has two edges, so every vertex
is a balancing vertex after deleting the joints. -/
theorem exists_isUnrolling_of_balanced_off (hbal : ∀ C, Ω.IsUnbalancedCycle C → ∃ e, C = {e})
    (w : V) : ∃ Ω', Ω.IsUnrolling w Ω' ∧ ∀ X, Ω'.IsFrameCircuit X ↔ Ω.IsFrameCircuit X := by
  have hw : Ω.IsLinkBalancingVertex w := fun C hC h2 => by
    obtain ⟨e, rfl⟩ := hbal C hC
    rw [Set.encard_singleton] at h2
    exact absurd h2 (by norm_num)
  obtain ⟨Ω', h⟩ := hw.exists_isUnrolling
  exact ⟨Ω', h, h.isFrameCircuit_iff hw⟩

/-- §2.5: "`F(G, B)` is equal to the cycle matroid `M(H)` of `H`". -/
theorem isFrameCircuit_iff_unrollToNew (hbal : ∀ C, Ω.IsUnbalancedCycle C → ∃ e, C = {e})
    (X : Set E) : Ω.IsFrameCircuit X ↔ Ω.unrollToNew.IsCycle X :=
  (isLoopUnrolling_unrollToNew hbal).isFrameCircuit_iff X

/-- §2.5: rolling up the links at a vertex `x` of a graph `H`, "`M(H) = F(G, B)`". The source rolls
up "the set of edges incident to `x`" as "a single unbalancing class", a part of `Σ(x)`, which in a
graph is its links at `x` (`linkStar`). A loop at `x` has no other end and stays. -/
theorem IsBalanced.isFrameCircuit_rollUp (hΩ : Ω.IsBalanced) (x : V) (X : Set E) :
    (Ω.rollUp x (Ω.linkStar x)).IsFrameCircuit X ↔ Ω.graph.IsCycle X :=
  (hΩ.isLoopUnrolling_rollUp x).isFrameCircuit_iff X

end BiasedGraph

open BiasedGraph

example := triangle_isLoopUnrolling.isFrameCircuit_iff ∅

example := exists_isUnrolling_of_balanced_off twoLoops_unbalanced_singleton ()

example := isFrameCircuit_iff_unrollToNew twoLoops_unbalanced_singleton ∅

example := IsBalanced.isFrameCircuit_rollUp (gainBiased_one_isBalanced (Γ := ℚˣ) triangle) 0 ∅

end Transformer.GainMatroid
