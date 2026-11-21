module

public import Transformer.GainMatroid.Section2_Unrolling

/-!
# Two claims of §2.5 on the unrolling fail as printed

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5.

* `printedUnrolling_fails` refutes "`F(Ĝ, B̂) = F(G, B)`" for `B̂` as printed. On two digons at `u`
  with a joint at `u` and a joint at another vertex, every class is a single edge, so the printed
  `B̂` is the joint at `u` alone, and that joint is a circuit of `F(Ĝ, B̂)` but not of `F(G, B)`.
  The source sets `(Ĝ, B̂) = (G, B)` when `J` is empty, so the counterexample has `J` nonempty.
  `centreJoint_isUnrolling` shows that `unrollBalanced` repairs this graph.
* `unbalancingClass_printed_fails`, in `Section2_UnrollingClasses`, refutes "the unbalancing classes
  of `Σ_G(u)` and of `Σ_Ĝ(u)` are the same" and "`J` is an unbalancing class of `Σ_Ĝ(u)`". With two
  joints in different components of `G − u`, no cycle of `Ĝ` contains both.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {u : V}

/-- A frame circuit of one edge is a balanced loop: handcuffs and thetas have at least two edges. -/
theorem isFrameCircuit_singleton_iff {Ω : BiasedGraph V E} {e : E} :
    Ω.IsFrameCircuit {e} ↔ {e} ∈ Ω.balanced := by
  refine ⟨fun h => h.resolve_right ?_, Or.inl⟩
  rintro ⟨h | h | h, -⟩
  · obtain ⟨C, D, -, hC, hD, hdis, -, hX⟩ := h
    obtain ⟨c, hc⟩ := hC.nonempty
    obtain ⟨d, hd⟩ := hD.nonempty
    have hce : c = e := (Set.ext_iff.1 hX c).2 (Or.inl hc)
    have hde : d = e := (Set.ext_iff.1 hX d).2 (Or.inr hd)
    exact Set.disjoint_left.1 hdis (hce ▸ hc) (hde ▸ hd)
  · obtain ⟨C, D, x, y, P, hC, hD, hdis, -, -, -, -, -, hX⟩ := h
    obtain ⟨c, hc⟩ := hC.nonempty
    obtain ⟨d, hd⟩ := hD.nonempty
    have hce : c = e := (Set.ext_iff.1 hX c).2 (Or.inl (Or.inl hc))
    have hde : d = e := (Set.ext_iff.1 hX d).2 (Or.inl (Or.inr hd))
    exact Set.disjoint_left.1 hdis (mem_verts.2 ⟨e, hce ▸ hc, Or.inl rfl⟩)
      (mem_verts.2 ⟨e, hde ▸ hd, Or.inl rfl⟩)
  · obtain ⟨x, y, P, Q, R, hT, hX⟩ := h
    obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hT.ne P)
    obtain ⟨q, hq⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hT.ne Q)
    have hpe : p = e := (Set.ext_iff.1 hX p).2 (Or.inl (Or.inl hp))
    have hqe : q = e := (Set.ext_iff.1 hX q).2 (Or.inl (Or.inr hq))
    exact hT.openPQ.edges e (hpe ▸ hp) (hqe ▸ hq)

/-- In a contrabalanced graph, the class of an edge of `Σ(u)` outside `J` is that edge alone. -/
theorem contrabalanced_unbalancingClass {G : OrientedGraph V E} {e : E}
    (hs : e ∈ G.contrabalanced.sigmaStar u) (hJ : e ∉ G.contrabalanced.offJoints u) :
    G.contrabalanced.unbalancingClass u e = {e} := by
  ext f
  constructor
  · rintro ⟨-, h | ⟨C, hC, -⟩ | ⟨he, -⟩⟩
    · exact h.symm
    · exact (Set.notMem_empty C hC).elim
    · exact (hJ he).elim
  · intro hf
    obtain rfl : f = e := hf
    exact ⟨hs, Or.inl rfl⟩

/-- A member of the printed `B̂` misses every edge of `Σ(u)` whose class is that edge alone. -/
theorem notMem_of_printed {Ω : BiasedGraph V E} {C : Set E} (hC : C ∈ Ω.printedUnrollBalanced u)
    {c : E} (hs : c ∈ Ω.sigmaStar u) (hcl : Ω.unbalancingClass u c = {c}) : c ∉ C := by
  intro hc
  rcases hC.2 c hs with h | h <;>
    rw [hcl, Set.inter_singleton_of_mem hc, Set.encard_singleton] at h
  · exact one_ne_zero h
  · exact absurd h (by norm_num)

end BiasedGraph

open BiasedGraph

/-- Two digons at `0`, a joint at `0` and a joint at `1`. -/
def centreJoint : OrientedGraph (Fin 3) (Fin 6) := ⟨![0, 0, 0, 0, 0, 1], ![1, 1, 2, 2, 0, 1]⟩

/-- `Ĝ` for `centreJoint`: the joint at `1` becomes a link to `0`. -/
def centreJointHat : OrientedGraph (Fin 3) (Fin 6) := ⟨![0, 0, 0, 0, 0, 1], ![1, 1, 2, 2, 0, 0]⟩

theorem centreJoint_isRollable : centreJoint.contrabalanced.IsRollable 0 :=
  isRollable_of_digons (a := 0) (a' := 1) (b := 2) (b' := 3) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide)

theorem centreJoint_mem_offJoints (e : Fin 6) :
    e ∈ centreJoint.contrabalanced.offJoints 0 ↔ e = 5 := by
  have h : ∀ e : Fin 6, centreJoint.tail e = centreJoint.head e → centreJoint.tail e ≠ 0 →
      e = 5 := by decide
  refine ⟨fun he => h e he.1.1 he.2, ?_⟩
  rintro rfl
  exact ⟨⟨rfl, Set.notMem_empty _⟩, by decide⟩

theorem centreJoint_unrollGraph : centreJoint.contrabalanced.unrollGraph 0 = centreJointHat := by
  have key : ∀ e : Fin 6, e ≠ 5 → centreJoint.head e = centreJointHat.head e := by decide
  refine eq_of_tail_head rfl (funext fun e => ?_)
  by_cases h : e ∈ centreJoint.contrabalanced.offJoints 0
  · refine (ite_eq_left h).trans ?_
    obtain rfl := (centreJoint_mem_offJoints e).1 h
    rfl
  · exact (ite_eq_right h).trans (key e fun he => h ((centreJoint_mem_offJoints e).2 he))

theorem centreJoint_mem_sigmaStar {c : Fin 6} (hc : c ≠ 4) :
    c ∈ centreJoint.contrabalanced.sigmaStar 0 := by
  have h : ∀ c : Fin 6, c ≠ 4 → c ≠ 5 → centreJoint.IsLink c ∧ centreJoint.tail c = 0 := by
    decide
  by_cases h5 : c = 5
  · exact Or.inr ((centreJoint_mem_offJoints c).2 h5)
  · exact Or.inl ⟨(h c hc h5).1, Or.inl (h c hc h5).2⟩

/-- Every unbalancing class of `centreJoint` at `0` is a single edge. -/
theorem centreJoint_class {c : Fin 6} (hs : c ∈ centreJoint.contrabalanced.sigmaStar 0) :
    centreJoint.contrabalanced.unbalancingClass 0 c = {c} := by
  by_cases hJ : c ∈ centreJoint.contrabalanced.offJoints 0
  · rw [unbalancingClass_of_mem_offJoints hJ]
    obtain rfl := (centreJoint_mem_offJoints c).1 hJ
    exact Set.ext centreJoint_mem_offJoints
  · exact contrabalanced_unbalancingClass hs hJ

/-- The printed `B̂` of `centreJoint` is the joint at `0` alone. -/
theorem centreJoint_printed : centreJoint.contrabalanced.printedUnrollBalanced 0 = {{4}} := by
  have hnot : (4 : Fin 6) ∉ centreJoint.contrabalanced.sigmaStar 0 := by
    rintro (⟨hl, -⟩ | hj)
    · exact hl rfl
    · exact absurd ((centreJoint_mem_offJoints 4).1 hj) (by decide)
  ext C
  constructor
  · intro hC
    obtain ⟨c, hc⟩ := hC.1.nonempty
    have h4 : ∀ d ∈ C, d = 4 := fun d hd => by
      by_contra hne
      have hs := centreJoint_mem_sigmaStar hne
      exact notMem_of_printed hC hs (centreJoint_class hs) hd
    exact Set.eq_singleton_iff_unique_mem.2 ⟨h4 c hc ▸ hc, h4⟩
  · rintro rfl
    refine ⟨?_, fun e _ => Or.inl (Set.encard_eq_zero.2 (Set.eq_empty_of_forall_notMem
      fun f hf => ?_))⟩
    · rw [centreJoint_unrollGraph]
      exact isCycle_singleton_of_loop rfl
    · obtain rfl : f = 4 := hf.1
      exact hnot hf.2.1

/-- With `B̂` corrected, the joint at `0` of `centreJoint` stays unbalanced in `Ĝ`. -/
theorem centreJoint_isUnrolling :
    centreJoint.contrabalanced.IsUnrolling 0 centreJointHat.contrabalanced where
  graph_eq := centreJoint_unrollGraph.symm
  balanced_eq := by
    refine (Set.eq_empty_of_forall_notMem fun C hC => ?_).symm
    have hC4 : C ∈ ({{4}} : Set (Set (Fin 6))) := by
      rw [← centreJoint_printed]
      exact hC.1
    exact hC.2 4 hC4 ⟨rfl, Set.notMem_empty _⟩

/-- §2.5 claims "`F(Ĝ, B̂) = F(G, B)`" for "the set of those cycles having intersection of size 0
or 2 with each unbalancing class". When `J` is nonempty, that set can contain a joint at `u`, which
is then a circuit of `F(Ĝ, B̂)` and not of `F(G, B)`. -/
theorem printedUnrolling_fails :
    ∃ (Ω Ω' : BiasedGraph (Fin 3) (Fin 6)) (u : Fin 3) (X : Set (Fin 6)), Ω.IsRollable u ∧
      (Ω.offJoints u).Nonempty ∧ Ω'.graph = Ω.unrollGraph u ∧
        Ω'.balanced = Ω.printedUnrollBalanced u ∧ Ω'.IsFrameCircuit X ∧ ¬Ω.IsFrameCircuit X :=
  ⟨centreJoint.contrabalanced, centreJointHat.singleCycleBiased {4} (isCycle_singleton_of_loop rfl),
    0, {4}, centreJoint_isRollable, ⟨5, (centreJoint_mem_offJoints 5).2 rfl⟩,
    centreJoint_unrollGraph.symm, centreJoint_printed.symm, Or.inl rfl,
    fun h => Set.notMem_empty _ (isFrameCircuit_singleton_iff.1 h)⟩

/-- Two digons at `0` and a joint at `0`: a rollable graph with `J` empty. -/
def centreLoop : OrientedGraph (Fin 3) (Fin 5) := ⟨![0, 0, 0, 0, 0], ![1, 1, 2, 2, 0]⟩

theorem centreLoop_isRollable : centreLoop.contrabalanced.IsRollable 0 :=
  isRollable_of_digons (a := 0) (a' := 1) (b := 2) (b' := 3) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide)

theorem centreLoop_offJoints : centreLoop.contrabalanced.offJoints 0 = ∅ := by
  have h0 : ∀ e : Fin 5, centreLoop.tail e = 0 := by decide
  exact Set.eq_empty_of_forall_notMem fun e he => he.2 (h0 e)

example := (isFrameCircuit_singleton_iff (Ω := centreJointHat.singleCycleBiased {4}
  (isCycle_singleton_of_loop rfl))).2 (Set.mem_singleton _)

example := centreJoint_class (centreJoint_mem_sigmaStar (c := 5) (by decide))

example := contrabalanced_unbalancingClass (centreJoint_mem_sigmaStar (c := 0) (by decide))
  fun h => absurd ((centreJoint_mem_offJoints 0).1 h) (by decide)

end Transformer.GainMatroid
