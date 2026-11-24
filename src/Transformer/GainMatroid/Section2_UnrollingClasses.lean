module

public import Transformer.GainMatroid.Section2_UnrollingFailures

/-!
# The unbalancing classes of the unrolling

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "Observe that `u` is a balancing vertex of `(Ĝ, B̂)`, that `Σ_Ĝ(u) = Σ_G(u)`, and that
> `Σ_Ĝ(u)/∼ = Σ_G(u)/∼`; that is, the unbalancing classes of `Σ_G(u)` and of `Σ_Ĝ(u)` are the
> same."

The first two claims are `IsUnrolling.isBalancingVertex` and `IsUnrolling.sigmaStar_eq`. The third,
and the source's later "`J` is an unbalancing class of `Σ_Ĝ(u)`", fail when two joints of `J` lie in
different components of `G − u` (`unbalancingClass_printed_fails`, on `twoJoints`). The corrected
`IsUnrolling.unbalancingClass_eq` assumes `JointsConnected`.

This module also states what the source leaves to a check by comparing circuits: `(Ĝ, B̂)` is a
biased graph and `F(Ĝ, B̂) = F(G, B)`. Both use the corrected `B̂` of `unrollBalanced`, which is `B`
when `J` is empty (`Section2_UnrollingEmptyJ`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

open BiasedGraph

/-- Two digons at `0` and a joint at each other vertex, so that `G − 0` has two components. -/
def twoJoints : OrientedGraph (Fin 3) (Fin 6) := ⟨![0, 0, 0, 0, 1, 2], ![1, 1, 2, 2, 1, 2]⟩

/-- `Ĝ` for `twoJoints`: each joint becomes a link to `0`. -/
def twoJointsHat : OrientedGraph (Fin 3) (Fin 6) := ⟨![0, 0, 0, 0, 1, 2], ![1, 1, 2, 2, 0, 0]⟩

theorem twoJoints_isRollable : twoJoints.contrabalanced.IsRollable 0 :=
  isRollable_of_digons (a := 0) (a' := 1) (b := 2) (b' := 3) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide)

theorem twoJoints_mem_offJoints (e : Fin 6) :
    e ∈ twoJoints.contrabalanced.offJoints 0 ↔ e = 4 ∨ e = 5 := by
  have h : ∀ e : Fin 6, twoJoints.tail e = twoJoints.head e → twoJoints.tail e ≠ 0 →
      e = 4 ∨ e = 5 := by decide
  constructor
  · exact fun he => h e he.1.1 he.2
  · rintro (rfl | rfl) <;> exact ⟨⟨rfl, Set.notMem_empty _⟩, by decide⟩

theorem twoJoints_unrollGraph : twoJoints.contrabalanced.unrollGraph 0 = twoJointsHat := by
  have key : ∀ e : Fin 6, ¬(e = 4 ∨ e = 5) → twoJoints.head e = twoJointsHat.head e := by decide
  refine eq_of_tail_head rfl (funext fun e => ?_)
  by_cases h : e ∈ twoJoints.contrabalanced.offJoints 0
  · refine (ite_eq_left h).trans ?_
    rcases (twoJoints_mem_offJoints e).1 h with rfl | rfl <;> rfl
  · exact (ite_eq_right h).trans (key e fun hor => h ((twoJoints_mem_offJoints e).2 hor))

theorem twoJointsHat_loopless : twoJointsHat.Loopless := by decide

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem twoJointsHat_degrees : ∀ C : Finset (Fin 6), C.Nonempty →
    (∀ c ∈ C, c = 4 ∨ c = 5) →
    ¬∀ w : Fin 3, (C.filter fun e => twoJointsHat.tail e = w ∨ twoJointsHat.head e = w).card = 0 ∨
      (C.filter fun e => twoJointsHat.tail e = w ∨ twoJointsHat.head e = w).card = 2 := by
  decide +kernel

/-- The joints of `twoJoints` form a path of `Ĝ`, so no cycle of `Ĝ` lies among them. -/
theorem twoJointsHat_not_isCycle {C : Set (Fin 6)} (hC : twoJointsHat.IsCycle C) :
    ¬∀ c ∈ C, c = 4 ∨ c = 5 := by
  obtain ⟨C, rfl⟩ := (Set.toFinite C).exists_finset_coe
  intro h
  exact twoJointsHat_degrees C (by simpa using hC.nonempty) (fun c hc => h c hc)
    (IsCycle.card_incident twoJointsHat_loopless hC)

theorem twoJoints_isUnrolling :
    twoJoints.contrabalanced.IsUnrolling 0 twoJointsHat.contrabalanced where
  graph_eq := twoJoints_unrollGraph.symm
  balanced_eq := by
    refine (Set.eq_empty_of_forall_notMem fun C hC => ?_).symm
    have hcyc : twoJointsHat.IsCycle C := twoJoints_unrollGraph ▸ hC.1.1
    refine twoJointsHat_not_isCycle hcyc fun c hc => ?_
    by_contra hne
    have h : ∀ c : Fin 6, ¬(c = 4 ∨ c = 5) → twoJoints.IsLink c ∧ twoJoints.tail c = 0 := by
      decide
    exact notMem_of_printed hC.1 (Or.inl ⟨(h c hne).1, Or.inl (h c hne).2⟩)
      (contrabalanced_unbalancingClass (Or.inl ⟨(h c hne).1, Or.inl (h c hne).2⟩)
        fun hj => hne ((twoJoints_mem_offJoints c).1 hj)) hc

/-- §2.5 claims that "the unbalancing classes of `Σ_G(u)` and of `Σ_Ĝ(u)` are the same" and that
"`J` is an unbalancing class of `Σ_Ĝ(u)`". With one joint in each component of `G − u`, no
unbalancing class of `Σ_Ĝ(u)` is `J`, while `J` is a class of `Σ_G(u)`
(`unbalancingClass_of_mem_offJoints`). -/
theorem unbalancingClass_printed_fails :
    ∃ (Ω Ω' : BiasedGraph (Fin 3) (Fin 6)) (u : Fin 3), Ω.IsRollable u ∧ Ω.IsUnrolling u Ω' ∧
      (Ω.offJoints u).Nonempty ∧ ∀ e, Ω'.unbalancingClass u e ≠ Ω.offJoints u := by
  refine ⟨_, _, 0, twoJoints_isRollable, twoJoints_isUnrolling,
    ⟨4, (twoJoints_mem_offJoints 4).2 (Or.inl rfl)⟩, fun e h => ?_⟩
  have hrel : ∀ f, f ∈ twoJoints.contrabalanced.offJoints 0 → e = f := fun f hf => by
    have hf' : f ∈ twoJointsHat.contrabalanced.unbalancingClass 0 e := by
      rw [h]
      exact hf
    rcases hf'.2 with hr | ⟨C, hC, -⟩ | ⟨he, -⟩
    · exact hr
    · exact (Set.notMem_empty C hC).elim
    · rw [twoJoints_isUnrolling.offJoints_eq_empty] at he
      exact (Set.notMem_empty e he).elim
  exact absurd ((hrel 4 ((twoJoints_mem_offJoints 4).2 (Or.inl rfl))).symm.trans
    (hrel 5 ((twoJoints_mem_offJoints 5).2 (Or.inr rfl)))) (by decide)

namespace BiasedGraph

variable {Ω Ω' : BiasedGraph V E} {u : V}

/-- §2.5: `(Ĝ, B̂)` is a biased graph, for the corrected `B̂` of `unrollBalanced`. -/
theorem IsRollable.exists_isUnrolling (h : Ω.IsRollable u) : ∃ Ω', Ω.IsUnrolling u Ω' := by
  sorry

/-- §2.5: "`F(Ĝ, B̂) = F(G, B)`", compared by circuits as in the source, for the corrected `B̂` of
`unrollBalanced`. -/
theorem IsUnrolling.isFrameCircuit_iff (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u) (X : Set E) :
    Ω'.IsFrameCircuit X ↔ Ω.IsFrameCircuit X := by
  sorry

/-- §2.5: "the unbalancing classes of `Σ_G(u)` and of `Σ_Ĝ(u)` are the same", for joints of `J` in
one component of `G − u`. -/
theorem IsUnrolling.unbalancingClass_eq (h : Ω.IsUnrolling u Ω') (hΩ : Ω.IsRollable u)
    (hJ : Ω.JointsConnected u) {e : E} (he : e ∈ Ω.sigmaStar u) :
    Ω'.unbalancingClass u e = Ω.unbalancingClass u e := by
  sorry

end BiasedGraph

/-- The single joint of `digonJoint` is connected to itself off `0`. -/
theorem digonJoint_jointsConnected : digonJoint.contrabalanced.JointsConnected 0 := by
  intro e he f hf
  have h4 : ∀ c : Fin 5, digonJoint.tail c = digonJoint.head c → c = 4 := by decide
  obtain rfl := h4 e he.1.1
  obtain rfl := h4 f hf.1.1
  exact ⟨.nil _, fun h => he.2 (List.mem_singleton.1 h).symm⟩

example := jointsConnected_of_connectedOff (Ω := twoC3.contrabalanced) (u := 0)
  twoC3_isKConnected.connectedOff

example := unbalancingClass_of_mem_offJoints ((twoJoints_mem_offJoints 4).2 (Or.inl rfl))

example := twoJoints_isUnrolling.tail_eq 0

example := twoJoints_isUnrolling.head_of_mem ((twoJoints_mem_offJoints 4).2 (Or.inl rfl))

example := twoJoints_isUnrolling.head_of_notMem (e := 0)
  fun h => absurd ((twoJoints_mem_offJoints 0).1 h) (by decide)

example := twoJoints_isUnrolling.end_of_mem_sigmaStar (e := 4)
  (Or.inr ((twoJoints_mem_offJoints 4).2 (Or.inl rfl)))

example := twoJoints_isUnrolling.isBalancingVertex

example := twoJoints_isUnrolling.offJoints_eq_empty

example := twoJoints_isUnrolling.linkStar_eq

example := twoJoints_isUnrolling.sigmaStar_eq

example (M : Matroid (Fin 6)) := twoJoints_isUnrolling.isFrameMatroid_iff_isLiftMatroid (M := M)

example := twoJoints_isRollable.exists_isUnrolling

example := twoJoints_isUnrolling.isFrameCircuit_iff twoJoints_isRollable {4}

example := centreJoint_isUnrolling.isFrameCircuit_iff centreJoint_isRollable {4}

example := digonJoint_isRollable.exists_isUnrolling.imp fun _ h =>
  h.unbalancingClass_eq digonJoint_isRollable digonJoint_jointsConnected (Or.inl digonJoint_link₀)

end Transformer.GainMatroid
